# comput6、comput7 两节点 TP/DP 验证方案

更新：2026-10-14。当前可用计算节点为 `comput6` 和 `comput7`，每台有 56 个可用 CPU，使用共享的 `/data1/code_home/wangtd/attention_cpu` 目录和 `cpu_tp` 环境。两节点均通过 `enp25s0f0`（10.10.10.6/7）通信。节点配置见 [configs/acceptance_comput6_comput7.json](configs/acceptance_comput6_comput7.json)。

本轮在两个节点上逐级增加进程数，验证真实 Gloo 通信中的 TP 和 DP Attention 前反向计算。性能计时用于观察各级耗时，不设置加速比门槛；**通过条件只取决于数值正确性、结果完整性和有限值检查**。不运行原方案的 16 节点容量探测、1536 worker 测试或大规模 profiler。

## 进程梯度与拓扑

每台启动一个 `torchrun` agent，各 worker 使用一个不重叠的物理核和一个计算线程。逐级运行：

| 每节点进程 P | 总进程 W | 跨节点 TP | 跨节点 DP |
| ---: | ---: | --- | --- |
| 1 | 2 | TP=2，DP=1 | TP=1，DP=2 |
| 2 | 4 | TP=4，DP=1 | TP=2，DP=2 |
| 4 | 8 | TP=8，DP=1 | TP=4，DP=2 |
| 8 | 16 | TP=16，DP=1 | TP=8，DP=2 |

全局 rank 按节点连续分配：先 `comput6`，再 `comput7`。第一种拓扑的 TP 组跨两节点；第二种拓扑的 DP 组跨两节点。各级使用相同模型、输入形状和 seed，不同时运行多个梯度。每级输出独立目录和完整 rank 报告。

## 数值通过条件

每种拓扑验证 FP64、FP32，以及 dropout=0 和固定 mask 的 dropout=0.2，共 8 组。数值用例采用 E=128、H=32、全局 B=9、S=17、梯度累积 2 次，包含不均匀 DP 样本和变长有效 token。

单进程参考使用相同输入、初始权重、mask 与目标。逐 rank 比较输出、全局 loss、TP 求和后的 dInput、QKV/输出投影参数梯度和一次 SGD 更新后的参数；复制参数及显式分片按原实现校验。FP64 容差 `atol=1e-10, rtol=1e-8`，FP32 容差 `atol=1e-5, rtol=1e-4`。全部检查、Gloo collective、两节点身份、源码/环境一致性及 rank 报告齐全才记为 PASS。任一数值失败即为 FAIL，不能由 loss 下降或性能数据代替。

## 性能记录

同一级在 E=512、H=32、FP32、全局 B=8、S=128、dropout=0 的固定负载上预热 2 步、计时 10 步。记录前向、反向、optimizer 和整步 p50/p95、有效 tokens/s、最慢 rank、CPU/RSS 与 TP/DP 通信完成时间；各级另记录单进程时间参考。计时关闭 profiler、capture 和 checkpoint。由于并行级数改变了总计算核心数，数据只用于观察趋势，不宣称线性扩展效率；节点上其他作业也可能影响绝对时间。

## 启动与报告

在 `admin1` 的项目根目录运行，先将两个节点的 SSH 主机密钥记录到可信的 known_hosts 文件。当前终端的系统 SSH 配置权限异常，因此启动时显式传入 `--ssh-config /dev/null`。每一级使用独立端口和新输出目录，例如：

```bash
conda activate cpu_tp
python -m tools.launch_acceptance \
  --inventory configs/acceptance_comput6_comput7.json \
  --suite correctness-performance --workers-per-node 2 --threads 1 \
  --warmup 2 --steps 10 --port 29602 \
  --ssh-config /dev/null --ssh-known-hosts /tmp/attention_cpu_known_hosts \
  --output runs/two_node_p2 --execute
```

把 `--workers-per-node` 和输出目录分别改为 1、2、4、8，端口分别用 29601～29604。启动器在每个节点检查物理核、共享文件系统、通信网卡、源码哈希、PyTorch 版本和 rank 数，实际结果由 `tools.aggregate_acceptance` 汇总。单级报告含 `formal/acceptance.json`、`formal/numerical_report.json`、`formal/performance.json`、每 rank 报告及两个节点的日志；四级汇总需列出每级状态和性能指标。结果写入被 Git 忽略的 `runs/`，不提交测试产物。

若节点连接、Gloo 初始化或资源检查失败，记录为 BLOCKED 或 FAIL 并保留日志；只对明确的环境问题重试，不把未运行的梯度记为通过。
