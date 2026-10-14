"""Validate rank evidence and produce performance/operator reports (fail closed)."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from attention_cpu.acceptance.common import atomic_json, digest


def percentile(values,q):
    values=sorted(values)
    pos=(len(values)-1)*q
    a=int(pos)
    return values[a]+(values[min(a+1,len(values)-1)]-values[a])*(pos-a)


def summarize(root):
    root=Path(root).resolve()
    errors=[]
    def require(condition,message):
        if not condition:
            errors.append(message)
    manifest=json.loads((root/'manifest.json').read_text())
    world=manifest['world_size']
    rows=[]
    for rank in range(world):
        path=root/'rank_reports'/f'rank_{rank}.json'
        if not path.exists():
            errors.append(f'missing rank {rank}')
            continue
        row=json.loads(path.read_text())
        require(row.get('rank')==rank,f'rank identity mismatch: {path}')
        require(row.get('manifest_sha256')==digest(root/'manifest.json'),f'manifest mismatch rank {rank}')
        expected='LOCAL_ONLY' if manifest['local_smoke'] else 'PROBE_PASS' if manifest['probe'] else 'PASS'
        require(row.get('status')==expected,f'rank {rank} status {row.get("status")}: {row.get("error", "")}')
        require(row.get('identity')==manifest['peers'][rank],f'peer identity mismatch rank {rank}')
        rows.append(row)
    require(len(list((root/'rank_reports').glob('rank_*.json')))==world,'unexpected or missing rank report files')
    suite=manifest['suite']
    expected_cases=[f'tp{tp}_dp{dp}' for tp,dp in manifest['topologies']]
    expected_numeric=([f'{case}_{dtype}_{drop}' for case in expected_cases
                       for dtype in ['float64','float32'] for drop in [0.0,0.2]]
                      if suite=='correctness-performance' else expected_cases)
    reference_hashes={digest(p) for p in (root/'references').rglob('manifest.json')} if (root/'references').exists() else set()
    for ref in (root/'references').rglob('manifest.json'):
        for name,expected_hash in json.loads(ref.read_text()).get('hashes',{}).items():
            path=ref.parent/name
            require(path.is_file() and digest(path)==expected_hash,f'corrupt reference: {path}')
    for row in rows:
        rank=row['rank']
        checks=row.get('numerical',[])
        if not manifest['probe']:
            require([c['case'] for c in checks]==expected_numeric,f'numerical coverage rank {rank}')
            for c in checks:
                require(c.get('passed') and c.get('all_ranks_passed'),f'numerical FAIL rank {rank} {c["case"]}')
                require(c.get('reference_sha256') in reference_hashes,f'missing reference evidence rank {rank}')
                names=c.get('checks',{})
                required={'loss','dInput','output.micro_0','gradient.qkv_proj.weight','gradient.out_proj.weight',
                          'updated.qkv_proj.weight','updated.out_proj.weight'}
                require(required<=set(names),f'incomplete tensor comparisons rank {rank}')
                require(all(v.get('passed') for v in names.values()),f'tensor mismatch rank {rank}')
        require([c['case'] for c in row.get('benchmarks',[])]==expected_cases,f'benchmark coverage rank {rank}')
    performance={}
    operator_sections=[]
    for case in expected_cases:
        samples=[b for row in rows for b in row.get('benchmarks',[]) if b['case']==case]
        require(len(samples)==world,f'incomplete timing ranks {case}')
        steps=3 if manifest['probe'] else manifest['steps']
        usable=all(len(b['times'])==steps and b.get('finite') and b.get('all_ranks_finite')
                   and math.isfinite(b['duration_seconds']) and b['duration_seconds']>0
                   and all(all(math.isfinite(v) and v>=0 for v in t.values()) for t in b['times']) for b in samples)
        require(usable,f'invalid timing or nonfinite computation {case}')
        if not samples or not usable:
            continue
        slow=max(samples,key=lambda b:b['duration_seconds'])
        maximum=slow['duration_seconds']
        stages={key:dict(p50_ms=1000*percentile([max(b['times'][i][key] for b in samples) for i in range(steps)],.5),
                         p95_ms=1000*percentile([max(b['times'][i][key] for b in samples) for i in range(steps)],.95))
                for key in ['forward','loss','backward','optimizer','step']}
        nodes=defaultdict(lambda:dict(cpu_seconds=0.,peak_rss_sum_bytes=0,worker_threads_sum=0,context_switches=0))
        comm=defaultdict(lambda:dict(calls=0,logical_bytes=0,completion_seconds_sum=0.))
        for b in samples:
            peer=manifest['peers'][b['rank']]
            node=nodes[peer['hostname']]
            node['cpu_seconds']+=b['after']['cpu_seconds']-b['before']['cpu_seconds']
            node['peak_rss_sum_bytes']+=b['after']['peak_rss_bytes']
            node['worker_threads_sum']+=b['after']['os_threads']
            node['context_switches']+=sum(b['after'][k]-b['before'][k] for k in ['voluntary_switches','involuntary_switches'])
            for event in b['communication']:
                record=comm[event['domain']]
                record['calls']+=1
                record['logical_bytes']+=event['logical_bytes']
                record['completion_seconds_sum']+=event['seconds']
        for hostname,node in nodes.items():
            cpus={cpu for p in manifest['peers'] if p['hostname']==hostname for cpu in p['affinity']}
            node['allocated_cpu_count']=len(cpus)
            node['process_cpu_utilization_percent']=100*node['cpu_seconds']/maximum/len(cpus)
        performance[case]=dict(stages=stages,window_max_seconds=maximum,slowest_rank=slow['rank'],
                              effective_tokens_per_second=samples[0]['global_batch']*samples[0]['seq']*steps/maximum,
                              samples_per_second=samples[0]['global_batch']*steps/maximum,
                              nodes=dict(nodes),communication=dict(comm),
                              per_rank_window_seconds={str(b['rank']):b['duration_seconds'] for b in samples})
        if suite=='max-parallel' and not manifest['probe']:
            selected={p['rank'] for p in manifest['peers'] if p['local_rank']==0}|{slow['rank']}
            profiled={b['rank'] for b in samples if b.get('profile')}
            require(selected<=profiled,'missing representative CPU profiles')
            operator_sections.append(f'## {case}\n\n最大实测 worker 数：{world}；节点数：{len(nodes)}；'
                                     f'最慢 rank：{slow["rank"]}；吞吐：{performance[case]["effective_tokens_per_second"]:.2f} tokens/s。\n')
            tp=dict(zip(expected_cases,[t[0] for t in manifest['topologies']]))[case]
            operator_sections.append(f'每 rank head 数：{manifest["config"]["num_heads"]//tp}；'
                f'实际每 rank batch 范围：{min(b["local_batch"] for b in samples)}～{max(b["local_batch"] for b in samples)}。'
                f'窗口最慢/最快 rank 比：{maximum/min(b["duration_seconds"] for b in samples):.3f}。\n')
            operator_sections.append('| 阶段（各步最慢 rank） | p50 ms | p95 ms |\n|---|---:|---:|')
            for stage,values in stages.items():
                operator_sections.append(f'| {stage} | {values["p50_ms"]:.3f} | {values["p95_ms"]:.3f} |')
            operator_sections.append('\n| 通信域 | 调用总数 | 逻辑字节总数 | 完成耗时总和 s（可重叠） |\n|---|---:|---:|---:|')
            for domain,values in comm.items():
                operator_sections.append(f'| {domain} | {values["calls"]} | {values["logical_bytes"]} | {values["completion_seconds_sum"]:.3f} |')
            operator_sections.append('\n| 节点 | worker CPU 利用率 % | RSS 峰值求和 GiB | 线程总数 | 上下文切换 |\n|---|---:|---:|---:|---:|')
            for host,values in nodes.items():
                operator_sections.append(f'| {host} | {values["process_cpu_utilization_percent"]:.2f} | {values["peak_rss_sum_bytes"]/2**30:.3f} | {values["worker_threads_sum"]} | {values["context_switches"]} |')
            for b in samples:
                profile=b.get('profile')
                if not profile:
                    continue
                path=Path(profile['trace'])
                require(path.exists() and digest(path)==profile['sha256'],f'missing/corrupt trace rank {b["rank"]}')
                ops=profile['operators']
                require(any(o['name']=='attention::qk_matmul' for o in ops),f'missing semantic profiling rank {b["rank"]}')
                aten=[o for o in ops if o['name'].startswith('aten::')]
                denominator=sum(o['self_cpu_ms'] for o in aten)
                operator_sections.append(f'### rank {b["rank"]}\n\n| 算子 | shapes | 次数 | self ms | total ms | aten self 占比 | self 内存字节 |\n|---|---|---:|---:|---:|---:|---:|')
                for o in aten[:10]:
                    fraction=100*o['self_cpu_ms']/denominator if denominator else 0
                    operator_sections.append(f'| {o["name"]} | {o["shapes"]} | {o["count"]} | {o["self_cpu_ms"]:.3f} | {o["total_cpu_ms"]:.3f} | {fraction:.1f}% | {o["self_cpu_memory_bytes"]} |')
                e=manifest['config']['embed_dim']; h=manifest['config']['num_heads']; seq=b['seq']; batch=b['local_batch']
                flops={'qkv_projection':6*batch*seq*e*(e/tp), 'output_projection':2*batch*seq*(e/tp)*e,
                       'qk_matmul':2*batch*(h/tp)*seq*seq*(e/h),'pv_matmul':2*batch*(h/tp)*seq*seq*(e/h)}
                operator_sections.append('\n| 前向范围 | 主要矩阵 FLOPs/调用 | 范围均时 ms | 有效 GFLOP/s |\n|---|---:|---:|---:|')
                for name,work in flops.items():
                    matches=[o for o in ops if o['name']=='attention::'+name]
                    if matches:
                        elapsed=sum(o['total_cpu_ms'] for o in matches)/sum(o['count'] for o in matches)
                        operator_sections.append(f'| {name} | {work:.0f} | {elapsed:.4f} | {work/elapsed/1e6 if elapsed else 0:.3f} |')
                operator_sections.append('\n有效 GFLOP/s 的范围可能包含复制和 output projection 的 TP 同步，不代表纯 GEMM 峰值利用率。')
                operator_sections.append('\n| 语义范围（含子事件） | 累计 ms | 调用数 |\n|---|---:|---:|')
                for op in ops:
                    if op['name'].startswith('attention::'):
                        operator_sections.append(f'| {op["name"]} | {op["total_cpu_ms"]:.3f} | {op["count"]} |')
                if aten:
                    dominant=aten[0]['name']
                    advice=('优先检查张量布局和 copy/contiguous 的必要性。' if any(k in dominant for k in ['copy','clone','contiguous'])
                            else '结合 GEMM shape 与每 rank head 数，优先比较降低 TP/进程密度后的矩阵粒度。' if any(k in dominant for k in ['mm','matmul'])
                            else '优先检查该算子的 shape、调用频率与临时分配，再决定是否减少调用或合并操作。')
                    operator_sections.append('\n后续调优方向：'+advice)
                    operator_sections.append(f'\n主要 self CPU 算子为 `{aten[0]["name"]}`；按 shape 区分投影、QK/PV 和反向，详见 trace `{path}`。\n')
            operator_sections.append('\n通信完成耗时可与计算重叠，累计值不除以整步时间当作通信占比；'
                                     'Top 表占比是采样 rank 的 aten self CPU 时间占比，不是全局墙钟占比。'
                                     '节点峰值 RSS 求和是各进程历史峰值之和，不代表同一时刻峰值。'
                                     'CPU 利用率基于 worker 进程 CPU 时间，不含其他作业；上下文切换不能单独证明 NUMA 或带宽瓶颈。\n')
    if manifest.get('reference_performance_required'):
        path=root/'reference_performance.json'
        require(path.exists(),'missing single-process performance baseline')
        if path.exists():
            baseline=json.loads(path.read_text())
            require(baseline.get('finite') and len(baseline['times'])==manifest['steps'],'invalid reference performance')
            performance['single_process_reference']=baseline
    state=('FAIL' if errors else 'LOCAL_ONLY' if manifest['local_smoke'] else 'PROBE_PASS' if manifest['probe'] else 'PASS')
    result=dict(suite=suite,status=state,world_size=world,errors=errors,performance=performance,
                actual_multi_node_passed=state=='PASS' and manifest['expected_nodes']>1,
                scope='maximum observed, not proven physical maximum' if suite=='max-parallel' else 'two four-rank topologies')
    atomic_json(root/'acceptance.json',result)
    atomic_json(root/'performance.json',performance)
    atomic_json(root/'numerical_report.json', {str(r['rank']):r.get('numerical',[]) for r in rows})
    if operator_sections:
        (root/'operator_analysis.md').write_text('# 最大并行度算子执行效果分析\n\n'
            +f'验收状态：**{state}**。LOCAL_ONLY 仅为单进程工具链自检，不能代表集群最大容量。\n\n'
            +'\n'.join(operator_sections),encoding='utf-8')
    text=f'# 验收报告\n\n类别：{suite}；状态：**{state}**；worker 数：{world}。\n\n'
    text+='\n'.join('- '+error for error in errors)
    for case,perf in performance.items():
        if 'stages' in perf:
            text+=f'\n\n{case}：{perf["effective_tokens_per_second"]:.2f} tokens/s，step p50={perf["stages"]["step"]["p50_ms"]:.3f} ms，p95={perf["stages"]["step"]["p95_ms"]:.3f} ms。'
    text+='\n\nLOCAL_ONLY 仅证明本机单进程工具链可运行；PROBE_PASS 仅证明短容量探测完成。详细算子数据见 operator_analysis.md（若已采集）。\n'
    (root/'acceptance.md').write_text(text,encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+')
    args=p.parse_args()
    failed=False
    for directory in args.directories:
        try:
            report=summarize(directory)
            print(directory,report['status'])
            failed|=report['status']=='FAIL'
        except (OSError,ValueError,KeyError,TypeError) as exc:
            print(directory,'FAIL',str(exc))
            failed=True
    raise SystemExit(1 if failed else 0)


if __name__=='__main__':
    main()
