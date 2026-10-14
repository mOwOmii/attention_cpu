# CPU Attention 实施方案

本文根据当前目录的 `attention_requirement.txt` 及后续需求调整制定，最近更新日期为 2026-10-14。最新范围调整为：不要求完全对齐昇腾场景，暂不考虑昇腾融合算子；原需求文件中的严格昇腾对齐要求以此调整为准。

推荐路线：使用现有 `pytorch_py311` 环境，通过标准 PyTorch CPU 运算实现完整 Attention，以独立数学基准验证前反向正确性，通过 Gloo 实现 TP 和 DP，并提供结果保存、重放和误差分析工具。MindSpeed-LLM 的通用 Attention 和并行结构可作为设计参考，后续按需增加接口或配置兼容。

本文是实施方案及阶段状态记录。阶段①至③已落地；阶段④ TP/DDP 代码、训练和分片恢复已交付，但实际 Gloo/DDP 多进程测试受 socket 权限限制，尚未验收。阶段⑤已形成面向 16 个 CPU 节点的[两类验收方案](multi_node_acceptance_plan.md)，验收工具已交付，真实集群验收待执行；阶段⑥性能优化仍未完成，运行方式见 `README.md`。

当前累计 44 项测试（基础计算 19 项、保存分析 12 项、并行 8 项、验收工具 5 项），42 项通过，2 项真实 Gloo 测试因环境跳过。单进程完整记录的 34 个前反向张量零容差重放仍作为阶段③证据；新增并行证据严格区分线程模拟的代数验证与实际网络通信验收。

## 1. 需求与交付范围

| 当前需求 | 实施内容 | 验收方式 |
| --- | --- | --- |
| Attention 必须为 CPU 版本 | QKV 投影、位置变换、Attention 核心、输出投影和反向均在 CPU 执行 | 检查张量设备，运行前反向和训练测试 |
| 标准 Attention 计算正确，参考 MindSpeed-LLM 通用结构 | 明确定义标准计算语义，使用普通 CPU 运算实现，暂不适配昇腾融合算子 | 与独立数学基准及 PyTorch 标准参考比较输出与梯度 |
| 支持跨机多 CPU 训练，首先支持 TP、DP | Gloo 进程组、TP 并行层、DP 梯度同步、多机启动脚本 | 单进程与并行等价性测试、实际两机训练测试 |
| 支持前反向结果保存与分析 | 中间张量和梯度导出、统计、误差比较、分片重建、重放 | 保存后可重现结果并定位偏差 |

首版基线为标准因果多头自注意力（MHA），同时支持非因果模式与显式 mask。无需先指定 MindSpeed-LLM 版本或取得 NPU 基准即可推进实现和验收；外部框架兼容作为后续可选扩展。

“完整 Attention”包含可训练的输入输出投影、位置处理、Attention 计算、dropout 和全部对应梯度，并提供 loss、optimizer step 和 checkpoint 的训练示例。FFN、模型级归一化和完整 LLM 训练框架不属于本方案首版范围。

## 2. 已验证的机器与环境

| 项目 | 实测情况 |
| --- | --- |
| 工作目录 | `/share/home/renhj/attention_cpu` |
| conda 环境 | `pytorch_py311` |
| Python | `3.11.17` |
| PyTorch | `2.14.0+cpu`，`torch.version.cuda=None` |
| CPU | Intel Xeon Platinum 8558P，双路，共 96 核 |
| NUMA | 两个节点，每节点 48 核 |
| 内存 | 约 503 GiB |
| 数学库 | MKL、oneDNN、OpenMP，PyTorch 报告 AVX512 能力 |
| 分布式后端 | Gloo 已编译；MPI、NCCL 未编译 |
| 现有依赖 | PyTorch、NumPy 可用；pytest、PyYAML、MindSpeed-LLM、Megatron、torch_npu 未安装 |
| CPU 计算探测 | SDPA 在形状 `[2,4,16,32]` 上完成前反向，输入梯度均为有限值 |

优先使用现有 PyTorch、NumPy 和 Python 标准库，测试采用 `unittest`。矩阵乘法调用现有数学库，首版不要求新增 C++ 算子或更换 conda 环境。

### 2.1 已记录的执行环境通信限制

此前两进程 Gloo 探测在初始化时失败，报错为无法解析本地回环地址。进一步直接创建 socket 返回：

```text
PermissionError: [Errno 1] Operation not permitted
```

阶段④已重新检查，直接创建 socket 仍返回 `PermissionError: [Errno 1] Operation not permitted`。Gloo 可导入、已编译，但当前会话无法实际启动通信。该限制不能用于判断宿主机本身的通信能力；需要在具备 socket 权限的运行环境执行严格验收。

可以在当前环境完成实现、单进程测试以及不依赖 socket 的计算验证。真实多进程 collective、多机训练验收需在允许 socket 的运行环境中执行，并分别记录“通过”“失败”“因环境未执行”，不能将模拟测试作为实际通信验收。

## 3. 固定标准计算规格与正确性基准

首版以标准缩放点积 Attention 为计算规格，明确定义张量布局、权重布局、缩放、mask、位置处理、dropout 和精度。实现使用普通 PyTorch CPU 运算，不依赖 MindSpeed-LLM、torch_npu 或 CANN，不实现昇腾融合算子的接口、返回值或硬件特定行为。

已固定 `configs/attention_spec.json`，并通过 `tools/generate_reference.py` 生成 `runs/reference/reference_manifest.json`，包含：

- 本项目版本、Python/PyTorch/NumPy 版本和 CPU 运行信息。
- 标准参考实现的来源、版本和运行方式。
- Attention 配置、QKV 权重布局和内部/外部张量布局。
- dtype、head 数、KV head 数、head dimension、缩放系数。
- mask、RoPE、dropout 及各阶段计算精度。
- TP、DP 配置，以及基准数据生成方式。

基准采用两个层次：小尺寸使用独立 NumPy FP64 公式实现前向和核心反向；完整模块使用独立的 PyTorch 标准参考，并按一致布局映射投影权重。PyTorch SDPA 和 `nn.MultiheadAttention` 可分别用于适用配置下的交叉验证，不能将被测模块调用自身视为独立基准。RoPE 等附加功能单独验证后，再验证组合模块。

### 3.1 正确性与并行等价性的定义

正确性包含符合约定的计算语义，以及与独立参考的前反向数值在固定误差门槛内一致。并行等价性要求相同输入、权重和全局 batch 下，TP/DP 结果与单进程结果一致。不同矩阵乘法与归约顺序可产生浮点误差，验收采用误差门槛，不要求逐 bit 相同。

验收覆盖 Attention 输出、完整模块输出、输入梯度、QKV 梯度、投影参数梯度，以及一次 optimizer 更新后的参数。

### 3.2 基准数据协议

被测实现与参考实现使用同一份张量；保存的基准包也用于跨进程重放，不以“相同 seed 下独立生成”替代相同输入。基准包至少包含：

```text
metadata.json
inputs.pt       # X 或 Q/K/V、mask、位置参数
weights.pt      # 投影权重和偏置，标注布局
upstream.pt     # dOutput
expected.pt     # Output、dQ/dK/dV、dX、参数梯度
```

先在 dropout 为零时验证确定性计算，再使用固定并共享的 dropout mask 比较训练路径。不同实现或不同 TP 切分即使设置相同 seed，也可能因随机数消费顺序不同而产生不同 mask，因此提供显式 mask 注入用于数值验证和重放，同时验证正常随机 dropout 路径的可复现性。

已提供 `tools/generate_reference.py` 在 CPU 环境生成数学基准和标准参考结果，以及 `tools/verify_reference.py` 验证输出与梯度。基准记录参考源码及各数据文件 SHA256，验证前检查文件未被修改。首版验收不依赖昇腾设备或 NPU 导出数据。

### 3.3 后续可选的框架兼容

如后续需要接入 MindSpeed-LLM，再固定版本并添加输入输出布局、权重格式、mask 和普通 Attention 配置的适配。昇腾数值比较及融合算子行为属于另行明确的扩展范围，不是当前实现、验收或性能优化的前置条件。

## 4. CPU Attention 实现

### 4.1 分层与接口

| 模块 | 职责 |
| --- | --- |
| `CPUAttentionCore` | QK 转置乘法、缩放、mask、softmax、attention dropout、与 V 相乘 |
| `CPUSelfAttention` | QKV 投影、head 重排、可配置位置处理、Attention 核心、输出投影 |
| 当前单进程训练入口 | 前向、loss、反向、optimizer step、checkpoint、保存与分析接入 |
| 已交付分布式训练入口，待实际通信验收 | TP/DP 进程组、并行层、DDP 梯度同步、累积、变长样本归一化及分片 checkpoint |

核心内部统一使用 `[B,H,S,D]` 布局，完整模块对外使用 `[B,S,E]`，其中 `E=H*D`。可提供 `[S,B,E]` 布局转换，所有转换需保留梯度。首版投影权重使用 PyTorch 线性层的 `[out_features,in_features]` 布局，合并 QKV 按完整 Q、K、V 三段排列，记录 head 索引与分片映射。

### 4.2 计算语义

标准分支为：

```text
Q, K, V = QKVProjection(X)
Q, K = PositionTransform(Q, K)
L = scale * (Q @ K.transpose(-2, -1)) + positional_bias
P = softmax(mask(L), dim=-1)
C = dropout(P) @ V
Y = OutputProjection(merge_heads(C))
```

`scale` 默认取 `1/sqrt(head_dim)`，允许显式配置。标准规格固定为先缩放 `QKᵀ` 再加位置偏置，不实现昇腾 PSE 类型或融合算子的专用分支。位置处理支持关闭或标准 RoPE；未启用位置偏置时其值为零。

本项目 bool mask 统一约定 `True` 表示屏蔽，支持广播至 `[B,H,Sq,Skv]`；浮点 mask 作为 logits 的加性偏置。因果 mask 与外部 mask 显式合并，padding mask 标注其屏蔽 key 的含义。完全被屏蔽的 query 行定义为概率及核心输出为零，对应核心梯度为零，通过安全 softmax 处理避免 NaN；完整模块若含输出偏置，该行最终输出仍可能为偏置。矩形序列的 causal mask 固定采用左上角对齐，并独立测试。PyTorch SDPA 的 bool `True` 表示参与计算，适配时需要转换；使用标准参考时应选取支持这些约定的分支。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 4.3 可观测路径与性能路径

- 可观测路径显式执行矩阵乘法、mask、softmax、dropout，便于保留中间张量并分析数值偏差。
- CPU SDPA 为后续可选的性能路径，仅在语义兼容且验证通过后启用；完整中间结果导出时切换回可观测路径。首版功能与验收不依赖任何 Attention 融合算子，暂不开发昇腾融合算子适配。

精度配置分离存储 dtype、计算 dtype 和 softmax dtype。FP32 是首版必需的训练基线，FP64 用于数学基准和小规模梯度检查；BF16/FP16 作为后续 CPU 精度与性能扩展，明确累加、softmax 和转换策略后单独验收，不作为首版交付前置条件。

### 4.4 前反向与功能范围

已实现 PyTorch autograd 反向，可通过 `return_intermediates=True` 取得中间结果并保留对应梯度。独立数学基准和 FP64 `gradcheck` 已验证核心、完整模块和 RoPE。通用保存分析工具已接入此接口；TP 通信反向已实现，真实 Gloo 通信下的验证仍待完成。

首版基线包括标准 MHA、因果/非因果模式、padding/显式 mask、可配置投影偏置、dropout 和可关闭的标准 RoPE。随后扩展 GQA/MQA；MLA、packed sequence、稀疏 Attention、滑动窗口和特殊 RoPE scaling 留作按需扩展，不影响标准 MHA 首版验收。

## 5. TP 与 DP 并行设计（代码已交付，待实际通信验收）

一个进程对应一个通信 rank，每个 rank 使用若干 CPU 核进行本地计算。采用：

```text
world_size = tp_size * dp_size
global_rank = dp_rank * tp_size + tp_rank
```

例如 `TP=2, DP=2`：

| 分组 | 成员 | 用途 |
| --- | --- | --- |
| TP 组 | `[0,1]`、`[2,3]` | 共同计算同一批样本 |
| DP 组 | `[0,2]`、`[1,3]` | 同步对应参数分片的梯度 |

所有 rank 按一致顺序创建进程组。TP 组内输入数据、mask 和位置参数一致；不同 DP 副本处理不同样本。

### 5.1 TP 权重切分和通信

采用列并行 QKV 投影和行并行输出投影，对应 Megatron 常用并行结构。[Megatron 并行线性层源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py)

| 位置 | 前向 | 反向 |
| --- | --- | --- |
| QKV 列并行投影 | 各 rank 计算本地 head | 对局部输入梯度做 TP `all_reduce(sum)` |
| Attention 核心 | 各 rank 计算本地 head | 本地计算 QKV 梯度 |
| 输出行并行投影 | 局部输出做 TP `all_reduce(sum)` | 复制的上游梯度参与本地反向，不重复求和 |
| 参数梯度同步 | 无 | 在对应 DP 组内同步 |

输出投影的共享 bias 在 TP 求和后加入一次。QKV 按本项目规定的 Q/K/V 三段布局和 head/group 索引切分，每个 rank 分别取得对应的 Q、K、V 行，不能直接对合并投影矩阵做一次连续均分。基础 MHA 要求 head 数可被 TP 整除，不合法配置在启动时明确报错。

已使用 `torch.autograd.Function` 实现前向复制、反向 TP SUM，以及前向 TP SUM、反向恒等的映射。输出投影 bias 在求和后只加一次。所有 TP peers 在训练时计算 dInput，避免仅部分 rank 开启捕获时出现 collective 次数不同。

完整训练路径无需每步收集全部 head。`all_gather` 用于验证、分片结果重建和权重导出。共享参数需明确所属同步域，避免同一梯度被重复归约。

GQA 先实现 KV head 可被 TP 整除的分支。MQA 或 KV head 少于 TP 时增加 KV 复制与复制域梯度求和，并独立验证，不能直接套用普通 head 切分。

### 5.2 DP 训练语义

使用 CPU `DistributedDataParallel`，显式配置：

```python
DistributedDataParallel(model, device_ids=None, process_group=dp_group)
```

DDP 仅覆盖对应 DP 组。loss 归一化、梯度平均与梯度累积共同决定训练等价性；对于有效 token 数不一致的 batch，应按全局有效 token 数定义权重，避免简单平均各副本的局部均值。[PyTorch DDP 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)

训练时明确优化器持有本地参数分片；同一 DP 组中的对应分片在同步梯度后执行一致更新。

### 5.3 已实现的训练与恢复

- `attention_cpu/parallel/groups.py`：固定 rank 映射，按统一顺序建立 Gloo TP/DP 组；检查 world size、backend 和拓扑一致性。
- `mappings.py`、`linear.py`、`attention.py`：可微 TP 通信、QKV 三段 head 索引切分、输出列分片及 MHA。当前仅支持 head 数可被 TP 整除的标准 MHA。
- `train.py`：DDP 仅绑定 DP 组，通过 `no_sync()` 累积 microbatch，最后一次反向执行平均。CLI `--batch-size` 表示跨 DP 与所有 microbatch 的全局 batch，可不均分，但每个 DP/microbatch 至少需一个样本。
- loss：各 microbatch 的局部有效元素误差和乘 `dp_size/global_valid_element_count`，与 DDP 平均配合得到全局有效元素均值。支持 `--variable-lengths` 合成样本验证不同有效 token 数。
- dropout：训练根据 seed、optimizer step 和全局样本/head 索引生成确定 mask，再切分到 DP/TP；直接模块 API 不传 mask 时，TP peers 需保持相同 RNG 状态。
- `checkpoint.py`：schema-v2 按 rank 保存参数、optimizer、RNG 和配置；所有 rank 写完后提交完整 manifest 和 SHA256。仅支持相同 TP/DP、batch、累积、优化器与线程配置恢复。schema-v1 单进程 checkpoint 保留原入口，不自动转为并行格式。

当前属于正确性实现：每个 rank 会构造完整初始化权重、合成 batch 和全局 dropout mask；导出完整权重也会聚合 TP 分片。初始化峰值内存、数据加载扩展、通信重叠与大规模 checkpoint 性能留到阶段⑥。多机 checkpoint 目前要求各 rank 访问同一路径的共享文件系统。

### 5.4 阶段④验证状态

[并行验证报告](runs/parallel_validation/report.json) 记录并行测试 8 项中 7 项通过、1 项真实 Gloo/DDP 测试因 socket 权限跳过，`stage4_fully_validated=false`。

已验证：`(TP,DP)=(2,1),(1,2),(2,2)` 的输出、dInput、参数梯度、一次 SGD 更新、不等有效 token 数及梯度累积；另覆盖 FP32/SBE/无 bias、可微 head bias、仅捕获一个 TP rank、分片重建和局部重放。这些多 rank 算法测试使用测试代码中的进程内线程 SUM，不是 Gloo 或生产通信后端。

实际 Gloo 测试入口已提供，具备 socket 权限时会启动真实进程并使用 DDP，验证三个拓扑、参数更新、累积与 checkpoint 续训。严格验收命令为：

```bash
python -m tools.verify_parallel --require-gloo --output runs/parallel_validation_new/report.json
```

退出码 0 表示全部通过，1 表示测试失败，2 表示因环境跳过实际 Gloo 验收。当前严格执行结果为 2，不能将阶段④标记为验收完成。

还已实际运行无通信的单 rank 并行训练路径：全局 batch 5、累积 2 次、变长样本、dropout/RoPE，完成四步并从 schema-v2 恢复到六步。证据为 [训练报告](runs/parallel_single_rank/training_report.json)、[恢复报告](runs/parallel_single_rank_resumed/training_report.json) 及 [局部重放报告](runs/parallel_single_rank/replay_report.json)。这验证了新训练入口和恢复流程，不代表多 rank 已通过。

## 6. 多机启动与 CPU 配置

采用 Gloo 和 `torchrun`，配置节点数、节点编号、主节点地址、端口及通信网卡。[PyTorch 分布式文档](https://docs.pytorch.org/docs/2.14/distributed.html)

以下为已交付训练入口的两机启动配置，每机两个进程，总计 `TP=2, DP=2`。代码已支持这些参数，但当前环境不能运行 socket，因此此配置尚未经过真实两机验收；需在允许 Gloo 互通、代码/环境一致且输出目录共享的机器上执行。

主节点：

```bash
conda activate pytorch_py311
export OMP_NUM_THREADS=44
export MKL_NUM_THREADS=44
export GLOO_SOCKET_IFNAME=<实际通信网卡>

python -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=2 \
  --node-rank=0 \
  --master-addr=<主节点IP> \
  --master-port=29500 \
  -m attention_cpu.train --tp-size=2 --dp-size=2 --batch-size=4 \
  --threads=44 --output=<共享输出目录>
```

第二台使用相同启动参数，将 `--node-rank` 改为 `1`；每台机器均需根据实际环境设置网卡和线程数。两机配置、代码及基准权重需一致。除 rendezvous 端口外，必须允许 Gloo 实际数据通信。

上述 rank 映射使 TP 位于机内、DP 跨机。跨机 TP 需另外测试，例如相同四进程启动配置使用 `TP=4, DP=1`。

当前机器优先测试每个 NUMA 节点一个进程，根据 `local_rank` 绑定互不重叠的 CPU 核。每进程以约 40～46 个计算线程、inter-op 线程为 1 起步，再比较四进程配置。实际线程数按性能测试确定，避免每个进程沿用默认 96 线程。

增加启动前检查：socket 可用性、节点连通性、进程数与并行配置合法性、head 切分合法性、环境版本一致性及输出目录可写性。通信超时应输出 rank、组成员和失败阶段。

本机四进程启动脚本已提供：`bash scripts/train_parallel_local.sh --steps 4 --output runs/parallel_example`。也可手动使用 `torch.distributed.run --standalone --nproc-per-node=4`。默认 TP=2、DP=2，`--distributed-timeout` 可配置 Gloo 超时。多机节点连通性、NUMA 性能及故障场景仍属于阶段⑤/⑥验收。

## 7. 前反向保存、重放和分析

### 7.1 数据目录与元信息

```text
<capture_root>/<run_id>/step_<step>/layer_<layer>/rank_<rank>/
    metadata.json
    forward.pt
    backward.pt
    summary.json
    replay.pt       # 仅 full 模式，保存输入、权重、随机状态和上游梯度
```

`capture_root` 可显式配置；训练 CLI 默认使用 `<output>/captures`。单进程训练的 layer、rank 均为 0；并行路径使用真实全局 rank，并在 `step_<step>/` 下增加 `micro_<micro>/` 目录。step 是从 1 开始的 optimizer 全局步数，恢复后继续递增。

已实现保存输入、QKV、位置处理结果、logits、概率、dropout mask、Attention 输出及梯度。`backward.pt` 通过 hooks 保存 DP 平均前的本次调用贡献；并行路径的 `post_tp.pt` 保存 TP 求和后 dInput，最后一个同步 microbatch 的 `post_dp.pt` 保存 DP 平均后的完整累积参数梯度。元数据记录各阶段语义、TP/DP rank、microbatch 和显式分片索引。

元信息记录版本、配置、局部 shape/dtype 统计、全局分片映射、布局、rank、线程数、训练模式和保存时间。完整记录另存输入和参数的 requires-grad 配置、前向时权重、前向前 CPU RNG 状态与上游梯度。保存使用 `detach().clone()` 独立快照；退出上下文后移除 hooks 并释放工具持有的计算图。临时目录写完后发布完整记录，加载时验证完成标记和 SHA256。

### 7.2 保存级别

已支持 `none`、`summary`、`selected`、`full`，并按 step、layer、rank 和张量名称 glob 筛选。默认训练关闭保存。`summary` 仅写统计，`selected` 仅写选中张量，`full` 同时保存独立重放数据。仅 `full` 支持完整重放；摘要没有原始张量，不能执行数值比较。

| 模式 | 实际文件 | 用途 |
| --- | --- | --- |
| `none` | 不创建记录目录 | 正常训练 |
| `summary` | `metadata.json`、`summary.json` | 检查统计、NaN/Inf 与梯度范数 |
| `selected` | 元数据、统计、选中张量的 `forward.pt` / `backward.pt` | 比较选定计算阶段，不支持完整重放 |
| `full` | 元数据、统计、全部前反向张量、`replay.pt` | 独立重放并逐项比较 |

对于 `B=2, H=32, S=4096, FP32`，单个完整 Attention 分数矩阵占 4 GiB；概率、梯度和序列化副本会继续增加开销。因此全量导出用于受控调试，不能默认为所有训练 step 开启。

目前摘要和筛选模式减少磁盘占用，但仍使用显式可观测计算及中间梯度保留，不能视为峰值内存优化。

### 7.3 分析能力

- 单份检查：shape、dtype、均值、标准差、极值、范数、NaN/Inf。
- 双份比较：最大绝对误差、相对 L2 误差、RMSE、超阈值比例。
- 分片重建：按照显式轴索引恢复 head、QKV 权重行、投影矩阵列及对应梯度；校验完整副本一致性。
- 偏差定位：按计算顺序报告最早不一致的张量、索引和两侧数值。
- 重放：读取输入、权重、mask、随机状态或固定 dropout mask、上游梯度，重现前反向。

上述能力已通过 `attention_cpu.capture`、`attention_cpu.analyze`、`attention_cpu.replay` 和对应 `tools` 命令实现。比较按计算顺序报告首个偏差及最差索引，缺失项或 NaN 不会被判为一致；相同符号的无穷允许相等。离线重建支持显式轴索引及副本校验，已用模拟分片验证 head、QKV 行和矩阵列；部分和与实际 TP/DP 通信不属于当前工具能力。完整记录重放恢复本次调用的前向与反向，不替代 optimizer checkpoint 恢复。

原单进程 schema-v1 checkpoint 继续保存完整参数、optimizer、训练与数据重建配置、loss 和 RNG。并行 schema-v2 已按 rank 保存分片及 optimizer 状态，并提供同拓扑恢复。真实数据采样器、不同拓扑重分片与 distributed/ZeRO optimizer 不在当前实现范围。

### 7.4 已交付接口与执行顺序

| 文件或入口 | 当前职责 |
| --- | --- |
| `attention_cpu/capture.py` | `CaptureConfig` 与 `CaptureWriter`；模式、范围和名称筛选；独立快照与事务式写入 |
| `attention_cpu/analyze.py` | 单份统计、逐项比较、最早偏差定位、轴索引/副本重建 |
| `attention_cpu/replay.py` | 加载完整记录，恢复输入/权重/模式/随机状态并比较前反向 |
| `attention_cpu/train.py` | 训练 CLI 的 `--capture-*` 参数，反向后、optimizer 更新前保存 |
| `tools/analyze_capture.py` | 单份记录分析与 JSON 报告 |
| `tools/compare_results.py` | 两份原始张量记录比较；不一致时返回非零状态 |
| `tools/replay_capture.py` | 独立重放；保存 `replay_report.json`，可选保留重算记录 |
| `tools/reconstruct_results.py` | 离线重建显式分片，输出全局张量 `.pt` 与统计 `.json` |

Python 中的调用顺序为：创建 writer → 进入 `writer.record(...)` 上下文 → 调用一次 `record.forward(...)` → 在上下文内 backward → 退出并保存 → optimizer 更新。loss 应通过捕获的输出向模块传播梯度。工具支持核心和完整模块，正常退出后清理 hooks；异常退出不发布完成记录，同一标识的已有记录不会被覆盖。

重放重新创建模块，恢复输入别名、requires-grad 配置、冻结参数、训练/eval 模式、CPU 线程数与前向前 RNG，并使用保存的上游梯度反传。结束后恢复调用方 RNG 与线程数。只有前向的完整记录也可重放；该操作复现一次模块调用，不执行 optimizer 更新。

### 7.5 保存、分析与重放的复现命令

以下命令在项目根目录、`pytorch_py311` 环境中执行。保存示例使用新的输出目录，避免与已有记录冲突。

```bash
conda activate pytorch_py311

python -m attention_cpu.train \
  --config configs/capture_demo.json \
  --steps 4 --seq-len 9 --threads 1 \
  --output runs/capture_example \
  --capture-mode full --capture-steps 1,3 --capture-run-id demo

python -m tools.analyze_capture \
  runs/capture_example/captures/demo/step_1/layer_0/rank_0 \
  --output runs/capture_example/analysis_report.json

python -m tools.replay_capture \
  runs/capture_example/captures/demo/step_1/layer_0/rank_0 \
  --output runs/capture_example/replayed --atol 0 --rtol 0

python -m tools.compare_results \
  runs/capture_example/replayed/replay/step_1/layer_0/rank_0 \
  runs/capture_example/captures/demo/step_1/layer_0/rank_0 \
  --output runs/capture_example/comparison_report.json --atol 0 --rtol 0
```

步骤筛选支持 `1,3-5` 这样的逗号列表及闭区间。`selected` 模式可使用 `--capture-tensors 'forward.output,backward.param.*'`；未匹配到张量时明确报错。当前训练仅 layer/rank 0，其他标签的筛选不会产生训练记录。

离线分片重建命令为 `python -m tools.reconstruct_results <记录目录1> <记录目录2> --output runs/reconstructed.pt`。各记录应保存相同张量名称，并具有一致的 step、layer、梯度阶段、模块类型及训练模式；元数据中须提供正确的 `global_shape`、`axis` 和 `indices`，或将完整副本标为 `replicated`。工具不推断缺失分片，不执行 collective 或部分和归约。

### 7.6 实际报告与验收范围

| 检查 | 已有结果与证据 |
| --- | --- |
| 捕获训练 | [训练报告](runs/capture_demo/training_report.json)：含 dropout/RoPE 的四步训练，仅保存第 1、3 步 |
| 单份统计 | [分析报告](runs/capture_demo/analysis_report.json)：34 个前反向张量，未发现 NaN |
| 独立重放 | [重放报告](runs/capture_demo/captures/demo/step_1/layer_0/rank_0/replay_report.json)：34 项全部通过，`atol=0, rtol=0` |
| 原记录与重放比较 | [比较报告](runs/capture_demo/comparison_report.json)：34 项全部通过，最大绝对误差均为 0，无首个偏差 |
| 完整副本重建 | [重建统计](runs/capture_demo/reconstructed.json)：CLI 已运行并生成 34 个张量；索引分片使用单元测试验证 |
| 保存行为 | `tests/test_capture.py` 验证保存开关不改变训练 loss、参数及 RNG；同时覆盖筛选、文件损坏、NaN/Inf、缺失项、冻结参数和梯度累积 |

比较仅覆盖两侧实际保存的张量，选定项通过不代表未保存项也通过。相同符号的无穷允许相等，NaN 判为不一致；离散类型精确比较。不同梯度通信阶段不能直接混比。零容差重放是当前相同运行环境下的实测结果，跨版本或跨机器时按固定精度门槛验收。

这些报告与张量位于 `.gitignore` 中的 `runs/` 下，可使用上述命令在新目录重新生成。TP/DP 代码现已交付，真实 Gloo/DDP 多进程和跨机训练仍未验收。

### 7.7 并行记录的离线重放

单个 TP 分片缺少其他 head 的权重，不能使用单进程 `tools.replay_capture` 独立恢复完整输出。应提供同一 DP 副本、相同 step/micro/layer 的全部 TP peers，使用 `tools.replay_parallel_capture` 重建完整权重并运行标准 CPU 模块，再根据显式分片索引逐项比较本次前反向及 TP dInput。

该工具明确排除 `post_dp.*`，因为局部调用和上游梯度不足以重放 DDP 的累积平均。这些张量的验收必须由真实多进程测试或全体 DP 样本基准完成。缺失 TP peers、混入不同 DP 样本或不一致的复制输入会报错。

`runs/parallel_validation/report.json` 中的 `offline_equivalence`、`offline_replay` 和 `offline_captures/` 提供带 `in_process_test_simulation` 标记的模拟证据；它们不会改变报告中 `actual_gloo_status=blocked_by_environment` 的状态。

## 8. 验证与验收标准

| 维度 | 必须检查的内容 |
| --- | --- |
| 单进程计算 | 独立数学基准、FP64 gradcheck、输出、输入梯度、参数梯度 |
| mask 与配置 | 因果/padding/显式 mask、dropout 开关、布局、位置处理、非连续输入及不合法参数 |
| 并行等价性 | `(TP,DP)=(1,1),(2,1),(1,2),(2,2)` 与相同全局 batch 的单进程结果比较 |
| 更新与恢复 | 一次参数更新、连续多步 loss、梯度累积、checkpoint 恢复后继续训练 |
| 多机通信 | 实际跨机 DP、实际跨机 TP、collective 正确性、超时及故障诊断 |
| 标准参考交叉验证 | 相同输入、权重、mask 和上游梯度下，与独立 NumPy/PyTorch 参考比较输出及梯度 |
| 保存与重放 | 保存开关等价性、范围/名称筛选、完整调用重放、冻结参数与局部梯度捕获 |
| 分析与重建 | 非有限值处理、缺失项与偏差定位、显式分片覆盖/重叠检查和副本一致性 |

测试覆盖多个 batch、序列长度、head 数和 head dimension，包含小尺寸、非整齐序列长度、完整屏蔽行和较大代表性用例。dropout 为零时先建立并行等价性，再使用可重放且按全局 head 索引切分的固定 mask 验证 dropout 路径。

FP32 CPU 等价性可以先以 `atol=1e-5, rtol=1e-4` 为候选门槛；FP64 梯度检查按小规模数值验证要求配置。后续 BF16/FP16 的门槛按其明确的 CPU 精度策略另行制定。所有正式门槛在验收前固定，不能通过不断放宽阈值掩盖实现问题。

测试报告明确区分已通过、失败和因环境缺失未执行的项目。首版完成标准是 CPU 数学正确性、完整模块前反向、保存分析、实际 TP/DP 训练及多机测试通过；不要求 NPU 基准或昇腾场景对齐。真实多机测试未执行时，跨机训练要求仍保持未验收状态。

## 9. 当前项目结构与后续新增文件

已实现文件：

```text
attention_cpu/
    __init__.py
    config.py
    core.py
    attention.py
    positional.py
    reference.py
    capture.py
    analyze.py
    replay.py
    checkpoint.py
    train.py
    parallel/
        __init__.py
        groups.py
        mappings.py
        linear.py
        attention.py
        train.py
        checkpoint.py
        replay.py
tools/
    generate_reference.py
    verify_reference.py
    compare_results.py
    analyze_capture.py
    replay_capture.py
    reconstruct_results.py
    replay_parallel_capture.py
    verify_parallel.py
tests/
    test_attention.py
    test_capture.py
    test_parallel.py
configs/
    attention_spec.json
    single_process.json
    capture_demo.json
scripts/
    train_parallel_local.sh
README.md
attention_requirement.txt
attention_implementation_plan.md
```

并行目录和本机启动脚本现已交付。跨机启动 `tools/launch_acceptance.py`、两类验收 `tools/acceptance_worker.py`、汇总 `tools/aggregate_acceptance.py` 及 CPU 绑定脚本已交付；性能计时集成在验收入口，不再单独依赖 `tools/benchmark.py`。

配置优先使用 JSON 和命令行参数，避免首版依赖未安装的 YAML 包。源码与结果数据分目录管理，基准数据包含版本清单。

## 10. 实施阶段

| 阶段 | 交付物 | 完成标准 | 当前状态 |
| --- | --- | --- | --- |
| ① 固定规格与基准 | 标准 Attention 配置、版本清单、CPU 基准生成和验证工具 | 计算语义明确，独立参考可重现 | 已完成，六组基准已生成并验证 |
| ② 单进程实现 | Attention 核心、完整模块、训练示例、单进程 checkpoint | 前反向、独立基准和梯度检查通过 | 已完成，19 项测试及训练/恢复示例通过 |
| ③ 保存与分析 | dump、筛选、统计、重放、比较和离线分片重建工具 | 能重现结果并定位中间张量及梯度偏差 | 已完成；新增 12 项测试，累计 31 项通过；实际 34 个张量零容差重放一致 |
| ④ TP＋DP | Gloo 组、可微 TP、DDP、累积、保存与分片恢复 | 实际并行结果与单进程训练等价 | 代码已交付；并行 8 项测试中 7 通过、真实 Gloo 1 项跳过；未验收完成 |
| ⑤ 多机验收 | 两类验收方案、数值/性能报告、最大并行度算子分析 | 性能与正确性通过；全部 16 节点达到最大实测并行度并完成算子分析 | 两类验收入口、CPU 绑定、profiler 和汇总已交付；本地自检通过，真实集群实测待执行 |
| ⑥ 性能优化 | 线程及 NUMA 调优、性能报告；可选 CPU SDPA 路径 | 优化后重新通过相关数值验收 | 待实现 |

本机实际结果：FP32 输出相对 NumPy FP64 基准的最大绝对误差约 `1.15e-7`；默认配置 30 步训练 loss 从 `0.232556` 降至 `0.004211`，checkpoint 已成功恢复至 35 步。包含 dropout 和 RoPE 的独立测试确认分段恢复与连续训练结果一致。详细运行入口与报告路径见 `README.md`。

阶段③实际示例位于 `runs/capture_demo/`：包含 dropout/RoPE 的四步训练保存第 1、3 步；分析、独立重放、比较及副本重建均已运行。测试同时验证保存开关不改变训练的 loss、参数与 RNG 状态。真实多机和 TP/DP 状态不因离线分片测试而改变。

阶段④剩余验收工作：

1. 在具备 socket 权限的环境运行严格命令，取得实际 Gloo/DDP 三个拓扑的通过报告。
2. 确认真实通信下的输出、输入/参数梯度、更新、累积及分片恢复与参考一致，并检查 `post_dp` 记录。
3. 阶段⑤按最新要求仅进行性能与正确性验证，以及 16 节点最大并行度测试并分析算子执行效果；不再要求逐级节点扩展、故障注入或长稳矩阵。

标准规格、CPU 基础模块和参考工具均可在现有环境推进，无需等待昇腾环境或 MindSpeed-LLM 配置。通信环境受限时继续完成独立工作，但保留实际通信验收为未完成状态。外部框架兼容与低精度扩展不阻塞首版交付。

性能报告记录前向、反向、optimizer、通信和导出的耗时，以及内存、吞吐和并行效率。性能路径与导出路径分开测量，不预先承诺未经实测的加速比。

## 11. 多机验收需要的环境与后续可选信息

- 允许 socket 的 CPU 执行环境，以及至少两台可互通机器，用于真实跨机训练验收。
- 各节点地址、通信网卡、可用核心数、内存与启动方式。
- 如后续需要框架兼容，再补充 MindSpeed-LLM 等框架的版本、配置、接口和权重格式；这些信息不是当前阶段的前置条件。

当前阶段不需要昇腾设备、torch_npu、CANN 或融合算子资料。多机条件齐备前，可完成 CPU 实现和本地数值验证，但不能将尚未执行的真实多机通信标记为已通过。

### 11.1 两类多机验收（最新范围）

详细设计见 [multi_node_acceptance_plan.md](multi_node_acceptance_plan.md)。本轮只做两类：① 单进程参考与两机四进程的性能/正确性验证，覆盖 TP4/DP1 和 TP2/DP2；② 直接使用全部 16 节点，探测最大有效 worker 数并做算子执行效果分析。按每节点 96 个物理核，目标上界为 1536 worker（TP32/DP48、每 worker 一个计算线程），实际值由 cpuset、内存、Gloo 线程/连接及进程限制决定。

第二类只在最高实测规模做正式计时和短 profiler 采集，分析矩阵乘法、softmax、RoPE、数据搬运、反向、TP/DP 通信及 CPU/内存表现。取消原 L0～L7 分级矩阵；故障恢复和两小时长稳不作为本轮门槛。三份 `configs/acceptance_*.json` 继续使用。本节与新验收文档优先于前文一般性测试建议；当前已交付实现与本地端到端自检，报告位于 `runs/acceptance_local_v1/`；状态为 LOCAL_ONLY，尚无真实集群通过证据。
