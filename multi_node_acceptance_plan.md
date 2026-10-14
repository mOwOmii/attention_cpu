# CPU Attention 两类验收方案

更新：2026-10-14。按最新要求，将原分级矩阵收敛为两类测试：**①性能与正确性验证；②最大并行度测试及算子执行效果分析。** 环境检查合并到作业启动，不再逐级进行 1/2/4/8/16 节点验收；故障注入、恢复、两小时长稳和全功能组合不作为本轮必测项。

本文为设计，尚未执行集群测试。现有 TP/DDP 训练与保存分析代码可复用；正式数值对照、性能计时、profiler、启动与汇总工具现已实现，运行入口见 [README.md](README.md#多机验收)。实际作业应在允许通信的集群环境启动。

## 1. 两类测试的最小安排

| 类别 | 使用资源 | 工作内容 | 最终产物 |
| --- | --- | --- | --- |
| ① 性能与正确性 | 单进程参考＋2 节点 × 2 进程，TP=4、DP=1；同一分布式作业另建 TP=2、DP=2 | 对照输出、梯度和更新；测量代表形状的前向/反向/整步性能 | 正确性误差表、性能基线表 |
| ② 最大并行度 | 直接使用全部 16 节点，每节点尽可能多的有效 worker | 短容量探测后，在最高可运行规模完成正式计时与算子剖析 | 最大实测并行度、算子/通信/资源分析报告 |

第一类保留两个四进程拓扑，是为了分别覆盖跨机 TP，以及机内 TP＋跨机 DP；不再展开多个节点规模的组合。两类均只用一个固定 seed=7，不要求三次独立重复；性能每次采集足够多步并报告波动。必要的无 profiler 计时窗口和有 profiler 分析窗口属于同一类测试的两个窗口。

启动前一次性检查 hosts、代码/环境哈希、CPU/cpuset、内存、通信网卡、共享路径及 rank 数量。主机名范围 `cpui01-cpui016` 按实际清单填写，不推断补零规则。每台节点启动一个 torchrun agent。作业初始化时做一次 rank/hostname all_gather 和整数 all_reduce 校验即可，不单独组织全节点配对测试。

## 2. 第一类：性能与正确性验证

### 2.1 正确性

使用已有 `configs/acceptance_fp64.json` 和 `configs/acceptance_fp32.json`：E=128、H=32、head_dim=4、causal、RoPE。固定全局 B=9、S=17、梯度累积 A=2，保留不均匀 DP 样本和有效长度。分别验证 dropout=0 与显式固定 mask 的 dropout=0.2；后者生成独立配置并保存哈希。

单进程与并行实现读取相同输入、初始权重、目标、mask 和上游梯度。单进程模块的数学依据沿用已完成的独立 NumPy/PyTorch 基准，不重跑整个功能组合矩阵。每个拓扑做一次前反向和一次 SGD 更新，比较：

- 最终输出和全局 loss；Q/K/V、概率、context 在出错时用于定位。
- dInput、QKV/输出投影的参数梯度，包括 bias。
- 一次 optimizer 更新后的参数。

FP64 门槛为 `atol=1e-10, rtol=1e-8`，FP32 为 `atol=1e-5, rtol=1e-4`，采用逐元素 `|actual-reference| <= atol + rtol*|reference|`。记录最大绝对误差、相对 L2 和最差位置。输出、梯度、参数和 loss 应有限，mask 产生的合法负无穷 logits 单独处理。

归约口径沿用现有训练实现：局部 loss 为局部有效元素误差之和乘 `DP/全局有效元素数`，DDP 最后平均。比较最终 post_dp 参数梯度；dInput 取 TP SUM 后的结果，除以 DP 再按全局样本索引拼接。TP 分片按索引还原，复制参数只取一份并检查副本一致。不把不同 DP 样本的中间结果直接拼成一个 TP 组。

**通过条件：** 两种拓扑、两种精度、两种 dropout 条件的上述检查全部通过，所有 4 个 rank 报告齐全。普通 loss 下降或离线 TP 重放不能替代真实 DDP 梯度对照。

### 2.2 性能

使用 `configs/acceptance_performance.json`：FP32、E=512、H=32、head_dim=16、dropout=0，固定全局 B=8、S=128、A=1；同一输入形状运行单进程以及上述两个四进程拓扑。

资源安排：单进程参考使用 44 个物理核；每节点两个 worker，各绑定一个 NUMA 节点的 44 个物理核（按实际 cpuset 修正），设置 OMP/MKL 与 `--threads` 一致。单进程与分布式使用的核心总数不同，因此报告相对耗时/吞吐和实际资源数，不将比值称作线性扩展效率。

预热 10 步、正式计时 50 步，关闭 capture 与 profiler。报告前向、反向、optimizer、整步 p50/p95、有效 tokens/s、各 rank 峰值 RSS。采用单调时钟；作业同步测量窗口起点，窗口结束后汇总各 rank 时间，报告每步最慢 rank 耗时及整个窗口最大耗时。吞吐分子只计一次全局有效 token，不乘 TP，也不重复计 DP 副本。

初始化、数据准备、保存和最终权重汇总单独记时，不混入稳态算子性能。没有预先约定的加速 SLA，本项输出可复现性能数据和瓶颈说明；若并行变慢，保留结果并解释，不能隐藏。

## 3. 第二类：最大并行度测试

### 3.1 最大并行度的定义与目标

本轮最大并行度指：**固定 Attention 形状、使用全部 16 节点，在不超配计算物理核的条件下，能够完成规定计算窗口的最大实测 worker 数。** 同时报告最大配置是否达到资源上界；它不等于最高吞吐，也不等于最大 TP。

按已记录的每节点 96 个物理核：初始目标是 **16 × 96 = 1536 个计算 worker**，每 worker 1 个计算线程。该数字是待验证上界，不是已支持能力；Gloo 后台线程、Python 进程内存、进程组连接数、文件描述符和实际 cpuset 可能使可运行值更低。torchrun agent 不计入 W，超线程不额外计为物理核。

统一使用 TP=32、DP=W/32；每节点 worker 数 P 取偶数，使 W=16P 可被 32 整除。例如：

| 每节点 P | 全局 W | TP | DP | 性质 |
| ---: | ---: | ---: | ---: | --- |
| 96 | 1536 | 32 | 48 | 首选目标，资源允许时直接尝试 |
| 48 | 768 | 32 | 24 | 目标无法运行时的容量回退示例 |
| 24 | 384 | 32 | 12 | 再次回退示例 |

表格不是三轮必测矩阵。最终只对选中的最高实测规模做完整计时和剖析。现有 H=32 使 TP 上限为 32，其余并行度由 DP 扩展。P=96 时每节点有三个机内 TP 组，DP 横跨节点；跨机 TP 的正确性由第一类 TP=4 测试承担。

### 3.2 减少执行次数的容量选择

1. 从实际 cpuset、进程/文件描述符限制和单 worker 内存估计得到 P 的候选上限，向下取偶数，最多 96。启动器使用互不重叠的 CPU 绑定，每 worker `OMP_NUM_THREADS=MKL_NUM_THREADS=--threads=1`。
2. 直接在 16 节点以候选 P 启动，完成组初始化及 3 步前反向。这是第二类作业内的短容量探测，不重复第一类全部数值用例。
3. 若成功，保留探测记录并启动同规模正式作业；若因资源容量失败，清理本作业后降低 P 重试。优先减半回退，必要时在成功/失败值之间按偶数二分，只跑短探测。网络错误或数值错误应定位修复，不能伪装成容量上限。
4. 少轮模式下允许止于最高已成功 P，报告“最大实测并行度 W”，同时列出更高失败/未测值；只有所有更高候选已排除或达到资源上界时，才称“本约束下最大并行度”。不为了凑最大数启动空数据 worker。

预计内存必须包括 Python/PyTorch 常驻开销、Gloo 组/线程、完整权重初始化、每 rank 的全局数据复制和激活。当前实现并非只持有本地数据；按每节点总内存估算，而非仅看单张 Attention 张量。进程组数也随 W 增大，应单列初始化耗时与连接失败。

### 3.3 固定负载与通过条件

主负载保持 FP32、E=512、H=32、S=128、causal、RoPE、dropout=0、A=1、固定全局 B=96，使用 `acceptance_performance.json`。在目标 TP32/DP48 下，每 DP 副本处理 2 个样本，每 worker 至少处理一个 head 和非空数据。较低 P 仍使用 B=96；这样容量探测不通过缩小计算任务换取更多进程。

每 rank 1 个计算线程；预热 10 步、正式无 profiler 测量 50 步，随后采集短 profiler 窗口。第一个正式步检查输出、梯度和更新均有限，并在所有 rank 按分片比较一次全局参考结果（FP32 门槛沿用第一类）；校验在计时窗口外执行，参考可在单独进程预生成。数值比较尽量按分片读取/流式比较，避免 rank 0 一次聚合全部中间激活。

**通过条件：** 全部 16 节点实际参与，W 个不同 rank 均完成计算、数值检查和计时，无空 worker、无缺失结果、无挂起；有完整资源清单与算子分析报告。报告 W、TP、DP、每节点 P、线程数、最高失败或未测配置及限制原因。吞吐下降不否定容量通过，但必须纳入执行效果结论。

## 4. 最大规模的算子执行效果分析

### 4.1 采集方法

正式性能来自无 profiler 的 50 步窗口。随后使用 `torch.profiler` CPU 活动，建议 wait=1、warmup=1、active=3、repeat=1；按 input shape 聚合，开启 `record_shapes`，代表 rank 开启 `profile_memory`，默认关闭 `with_stack` 以控制开销。profiler trace 单独保存，不将其步时混入正式吞吐。

全 rank 仅记录轻量的阶段计时和 RSS。详细 profiler 选择每节点 local_rank=0，以及无 profiler 窗口中最慢的 rank（去重）；若不同本地 TP 组表现不同，再补该组代表，不对 1536 个进程全部导出大 trace。保留 rank/group/hostname 对应关系。CPU 内存 profiler 反映张量分配事件，不替代进程 RSS；线程池和 Gloo 等开销用系统指标补充。

现有模块需加 `record_function` 范围标记 QKV projection、RoPE、QK matmul、mask/scale、softmax、PV matmul、output projection、TP collective、loss、backward 和 optimizer。查看 aten 事件与形状，将同名算子的前向/反向调用区分。DDP 通信通过 profiler 可见事件或保持原有归约语义的测量包装定位，异步调用要记录完成时间，不能只计 enqueue 耗时。

### 4.2 必须回答的问题

| 分析对象 | 采集指标 | 应形成的结论 |
| --- | --- | --- |
| QKV/输出投影 | mm/addmm 等事件形状、调用次数、self/total CPU time | head 分片后 GEMM 是否过小，投影是否主要瓶颈 |
| QKᵀ 与 PV | bmm/matmul 的形状、单次及累计耗时 | TP32 每 rank 仅一个 head 的计算粒度及成本 |
| mask、scale、softmax、RoPE | elementwise/softmax 时间、临时分配、copy/contiguous 时间 | 非矩阵计算和数据搬运是否超过矩阵乘法 |
| backward/optimizer | 反向算子和更新耗时、前反向比 | 梯度计算、梯度归约和参数更新各占多少 |
| TP/DP 通信 | 调用数、逻辑 tensor bytes、提交/完成时间、最慢 rank | 小消息开销、等待与计算通信重叠；不得将逻辑字节称作实际网络线速 |
| CPU 与内存 | 每节点 CPU 使用、worker/线程数、RSS、可用内存、上下文切换 | 进程过密、后台线程竞争、NUMA 远端访问的证据 |
| rank 不均衡 | 全 rank 步时 p50/p95/max、最慢 host/rank、阶段差异 | 尾部来自计算、通信还是调度等待 |

报告 Top 10 算子，包含 name、input shapes、count、self CPU ms、total CPU ms、占比和内存事件。self 时间用于避免父子嵌套重复计数；跨线程/异步事件可能重叠，所有事件 CPU 时间之和不等于端到端墙钟时间。通信占比用关键路径或明确的非重叠等待口径给出；无法可靠拆分时报告范围耗时及重叠限制，不杜撰百分比。

估算单 rank 前向主要矩阵 FLOPs：QKV 为 `6*b*S*E*(E/TP)`，输出投影为 `2*b*S*(E/TP)*E`，QK/PV 合计 `4*b*(H/TP)*S*S*d`，其中 b 为该 DP 副本的实际 batch，d=E/H。这里只是主要矩阵前向 FLOPs，不包含 RoPE、softmax、反向、优化器或通信。用对应算子的墙钟耗时估算 GFLOP/s，注明口径；无硬件计数器时不能据此断言已饱和内存带宽或 CPU 峰值。

结论至少回答：最大实测规模是多少；主导算子是什么；计算还是通信/调度占主导；增加进程是否使矩阵过小；哪些 rank 拖慢整体；下一步最值得调整的是进程密度、线程/NUMA、TP/DP 还是张量复制。第一类形状/资源与第二类不同，二者吞吐不直接换算为扩展效率。

## 5. 实施范围与交付文件

保留现有三个 `configs/acceptance_*.json`。已提供统一验收入口，通过 `--suite correctness-performance` 和 `--suite max-parallel` 区分两类：

- `tools/acceptance_worker.py`：已集成启动预检、固定基准、并行数值比较、阶段计时和选择性 profiler。
- `scripts/run_acceptance_node.sh`：按节点 manifest 设置 torchrun、线程、worker 核绑定；第二类短容量探测由外层控制。
- `tools/aggregate_acceptance.py`：校验全部 rank，汇总误差、吞吐、容量与算子表，输出 JSON＋Markdown。

复用已有训练和 TP/DDP 逻辑，不另建数学实现。现有 `tools.verify_parallel` 是本机 spawn 测试；普通训练 CLI 不承担 profiler 和正式逐张量分布式对照；使用新增 `tools.acceptance_worker` 执行。当前本机脚本默认 lo，跨机启动必须显式使用实际通信网卡和共同 rendezvous 地址。

结果目录仅分两类：

```text
runs/cluster_acceptance/<run_id>/
  manifest.json
  correctness_performance/
    numerical_report.json
    performance.json
  max_parallel/
    capacity_attempts.json
    numerical_report.json
    performance.json
    operator_analysis.md
    traces/host_<host>_rank_<rank>.json
  logs/
  rank_reports/
  acceptance.md
```

最终报告只有两项验收结论，均区分 PASS、FAIL、BLOCKED、NOT_RUN。第一类交付数值正确性与性能基线；第二类交付最大实测规模和有 trace/资源数据支持的算子执行效果分析。性能与容量不能用未执行的估算值代替实测。

技术依据：[PyTorch 2.14 torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)、[分布式通信](https://docs.pytorch.org/docs/2.14/distributed.html)、[CPU profiler](https://docs.pytorch.org/docs/2.14/profiler.html)。

## 6. 已落地的实现与验证状态

- `attention_cpu/acceptance/runner.py`：rank 0 分块生成参考并保存共享基准，校验 SHA256；真实 TP/DDP 数值对照、SGD 更新、性能窗口与 profiler 窗口。
- `attention_cpu/acceptance/common.py`：通信完成时间/逻辑字节记录、进程资源采样、环境标识和原子报告。
- `attention_cpu/profiling.py`：按需启用语义范围标记，普通训练关闭采集。
- `tools/launch_acceptance.py`：SSH 命令预览、显式执行、逐节点调度器入口、短容量探测与明确资源错误回退。
- `tools/bind_acceptance_worker.py` 与 `scripts/bind_cpu_worker.sh`：在导入 PyTorch 前按实际 cpuset 绑定不重叠物理核，采用 first-touch 内存分配。
- `tools/aggregate_acceptance.py`：检查 rank 完整性、数值项、参考/trace 校验和，输出性能与算子分析。
- `configs/acceptance_cluster.example.json`：待填写真实 hostname、网卡和 master IP；不猜测节点补零格式。

本地工具测试和本机真实 Gloo 测试已通过；两机与 16 节点验收仍需在目标集群执行。中间报告和 CPU trace 不随源码保存。

实施时采用一次独立短探测加一次正式作业；自动回退按偶数减半，不自动二分穷举所有规模。严格 NUMA membind 和硬件带宽计数器未实现；可在调度器已分配节点内使用逐节点入口。正式性能数据采用 SGD 更新，与数值测试相同，报告 optimizer 名称。参考生成、文件 I/O 与 profiler 开销均不纳入正式吞吐。

## 7. jsub 提交路径

新增登录节点入口 `scripts/submit_acceptance_jsub.sh`，通过 `tools/jsub_acceptance.py` 提交两项依赖作业，申请 192/1536 CPU slots。节点内启动采用调度器 `jexec`，不依赖普通 SSH。执行前校验 JH_HOSTS 与 inventory 的 scheduler_host 一致；controller 心跳失效后远端 guard 停止计算。运行前须核对 README 的“多机验收”说明与现场调度资源；目前尚无真实集群作业号。
