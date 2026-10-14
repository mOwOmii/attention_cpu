# CPU Attention

本项目已完成 `attention_implementation_plan.md` 的阶段①固定规格与基准、阶段②单进程实现，以及阶段③保存与分析。使用 `pytorch_py311` 中已有的 PyTorch CPU、NumPy 和 Python 标准库，无需额外安装依赖。

阶段④的 Gloo TP/DDP、梯度累积、分片 checkpoint 和并行记录已实现。当前会话仍禁止 socket，实际 Gloo/DDP 多进程验收尚未通过；已验证的并行等价性使用明确标注的进程内线程模拟 SUM，不能替代网络通信验收。

当前实现为标准 MHA，包含输入输出投影、因果/非因果模式、bool/浮点 mask、padding mask、attention dropout、可选标准 RoPE、前反向和单进程训练恢复。所有计算使用显式 CPU 运算，未使用 Attention 融合算子。

## 快速运行

以下命令在项目根目录执行：

```bash
conda activate pytorch_py311
python -m unittest discover -s tests -v
python -m attention_cpu.train --config configs/single_process.json --steps 30 --output runs/example
python -m attention_cpu.train --resume runs/example/checkpoint.pt --steps 35 --output runs/example_resumed
```

也可以使用 `conda run --no-capture-output -n pytorch_py311 python ...`，无需先激活环境。

`--steps` 是包括已完成步骤在内的总目标步数。恢复时自动读取原模型、数据、优化器和线程配置；显式传入的配置若与 checkpoint 不一致会报错。训练示例使用固定合成 batch，学习序列的因果累计均值或非因果均值，用于验证训练流程，不代表真实数据训练效果。

## 固定基准

计算规格保存在 [configs/attention_spec.json](configs/attention_spec.json)，默认完整模块配置为 [configs/single_process.json](configs/single_process.json)。

```bash
python -m tools.generate_reference --output runs/reference_new --threads 1
python -m tools.verify_reference --reference runs/reference_new --threads 1
```

生成器要求目标目录不存在或为空，避免覆盖已有基准。基准目录中包含 `reference_manifest.json` 及六组案例，每组包含：

```text
metadata.json
inputs.pt
weights.pt
upstream.pt
expected.pt
```

NumPy FP64 参考独立实现核心及完整模块的前向和解析反向，包括投影参数梯度与 RoPE 的逆旋转梯度。适用案例额外保存 `torch.nn.MultiheadAttention` 的结果及梯度。生成器不运行被测 Attention 模块；生成与验证为两个独立命令。

manifest 记录环境版本、CPU、seed、参考源码及基准文件的 SHA256。验证工具先校验文件，再按固定误差门槛检查输出、输入梯度、QKV 梯度和投影参数梯度；成功时写出 `verification_report.json`，失败时返回非零状态且清除旧的成功报告。SHA256 用于发现基准内容变更，不作为来源认证。

默认容差：FP32 为 `atol=1e-5, rtol=1e-4`；FP64 为 `atol=1e-10, rtol=1e-8`。NumPy 参考始终采用 FP64；非默认 softmax 精度策略应单独制定参考验证，当前固定案例采用默认精度。

已生成的实际基准为 [runs/reference/reference_manifest.json](runs/reference/reference_manifest.json)，验证结果为 [runs/reference/verification_report.json](runs/reference/verification_report.json)。`runs/` 保存可再生的数据，已加入 `.gitignore`。

## Python 接口

```python
import torch
from attention_cpu import AttentionConfig, CPUSelfAttention

torch.set_num_threads(2)
config = AttentionConfig(embed_dim=64, num_heads=4, causal=True, rope=True)
model = CPUSelfAttention(config)
x = torch.randn(2, 16, 64, requires_grad=True)
y = model(x)  # [B,S,E]
y.square().mean().backward()
assert x.grad is not None
```

完整模块签名：

```python
model(
    x,
    attn_mask=None,
    key_padding_mask=None,
    positional_bias=None,
    position_ids=None,
    dropout_mask=None,
    return_intermediates=False,
)
```

核心接口：

```python
from attention_cpu import CPUAttentionCore

core = CPUAttentionCore(causal=False, dropout=0.0, scale=None)
q = torch.randn(2, 4, 7, 16, requires_grad=True)
k = torch.randn(2, 4, 9, 16, requires_grad=True)
v = torch.randn(2, 4, 9, 12, requires_grad=True)
context = core(q, k, v)  # [2,4,7,12]
context.sum().backward()
```

核心支持 query 与 key/value 的不同序列长度及不同 value dimension；完整模块是 self-attention，输入输出维度相同。当前只支持 FP32、FP64，所有张量和参数必须位于 CPU。FP64 模块应通过 `AttentionConfig(dtype="float64")` 创建；输入 dtype 必须与配置一致，不能仅转换参数而保留旧配置。

### mask、dropout 和 RoPE 约定

- bool `attn_mask` 的 `True` 表示屏蔽；形状必须按 PyTorch 广播规则广播到 `[B,H,Sq,Skv]`。常用形状为 `[Sq,Skv]`、`[B,1,Sq,Skv]`、`[1,H,Sq,Skv]`。每 batch 的 mask 使用显式 head 维，避免将 `[B,Sq,Skv]` 误当成 batch 广播。
- 浮点 mask 使用输入 dtype，作为 logits 加性偏置；允许 `-inf`，拒绝 NaN 和 `+inf`。位置偏置使用输入 dtype，要求有限值，在 `QKᵀ` 缩放后加入。
- `key_padding_mask` 为 bool `[B,Skv]`，`True` 屏蔽 key。它不自动屏蔽 query 或从 loss 中排除 padding；需要时由调用方另行提供 query mask 和有效 token loss。
- 因果规则为 `key_index > query_index` 时屏蔽，包括矩形序列的左上角对齐。它是本阶段固定规格，不自动实现 KV cache 的右下角对齐。
- 全屏蔽行的概率、核心输出及核心梯度为零；完整模块的输出投影 bias 仍可能保留。
- dropout 只作用于 attention 概率，在训练时乘 `keep_mask/(1-p)`，`eval()` 时关闭。显式 `dropout_mask` 为 bool、`True=保留`，只能在训练且 `p>0` 时注入。
- RoPE 使用相邻偶数/奇数维配对，支持偶数 `rotary_dim <= head_dim` 的部分旋转。`position_ids` 为非负整数 `[S]` 或 `[B,S]`，默认 `0..S-1`；未启用 RoPE 时传 position IDs 会报错。该配对方式已固定，不等同于所有外部框架的半维配对方式。

### 中间结果与梯度

```python
output, details = model(x, return_intermediates=True)
output.sum().backward()
q_grad = details["query"].grad
probabilities = details["probabilities"]
```

返回的字典包括 QKV 投影、RoPE 前后的 Q/K、scores、logits、概率、dropout mask、context、最终输出等；对需要梯度的张量自动保留梯度。默认不返回或保留这些中间梯度，降低训练开销。中间结果仍与当前计算图相关，需要持久化时由调用方显式 `detach().clone()`。

该接口已接入下面的保存、分析和独立重放工具。

## 前反向保存与分析

### 在训练中保存

下面示例使用包含 dropout 和 RoPE 的配置，在第 1、3 步保存完整记录。目标目录应使用新的路径；同一 run、step、layer、rank 的已有记录不会被覆盖。

```bash
python -m attention_cpu.train \
  --config configs/capture_demo.json \
  --steps 4 --seq-len 9 --threads 1 \
  --output runs/capture_example \
  --capture-mode full --capture-steps 1,3 --capture-run-id demo
```

输出目录：

```text
runs/capture_example/captures/demo/step_1/layer_0/rank_0/
    metadata.json
    summary.json
    forward.pt
    backward.pt
    replay.pt
```

保存级别：

| 模式 | 保存内容 | 独立前反向重放 |
| --- | --- | --- |
| `none` | 不保存，默认值 | 不支持 |
| `summary` | 元数据及逐张量统计，不写 `.pt` 张量文件 | 不支持 |
| `selected` | 元数据、选中张量的统计及前反向张量 | 不支持，未保存完整输入和权重 |
| `full` | 全部前反向张量、统计、权重、输入、随机状态和上游梯度 | 支持 |

使用 `--capture-steps 1,3-5` 指定步骤，默认捕获所有步骤；`--capture-layers`、`--capture-ranks` 使用相同语法。当前单层单进程训练的 layer 和 rank 均为 0，只有对应筛选条件匹配时才写入记录。`*` 表示不限制。

按名称筛选的示例：

```bash
python -m attention_cpu.train --steps 3 --output runs/capture_selected \
  --capture-mode selected --capture-steps 1 \
  --capture-tensors 'forward.output,forward.probabilities,backward.param.*'
```

名称采用 `forward.<张量名>`、`backward.<张量名>`。例如 `forward.query`、`forward.arg.attn_mask`、`backward.input`、`backward.param.qkv_proj.weight`。筛选支持 glob；没有匹配项时明确报错。

`backward.pt` 保存本次调用、DP 平均前的梯度贡献。临时 autograd hooks 避免将先前累积的 `.grad` 混入记录。TP 记录另有 `post_tp.pt` 保存 TP 求和后的 dInput；最后一个同步 microbatch 的 `post_dp.pt` 保存 DP 平均后、包含全部 microbatch 累积的参数梯度。各阶段语义写入元数据，不能混比。保存发生在 optimizer 更新之前；数据为独立快照，完成后移除 hooks 并释放记录对象持有的计算图。

`summary`、`selected` 可减少磁盘占用，但当前仍通过显式可观测路径保留中间梯度，不能保证降低计算阶段的峰值内存。长序列建议只捕获少量 step。并行记录包含真实 rank 和权重/激活分片索引，支持 `post_tp.*`、`post_dp.*` 名称筛选。

### 统计、比较与独立重放

```bash
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

统计包括 shape、dtype、字节数、有限元素均值/标准差/极值/L2 范数，以及零、NaN、正负无穷的计数。mask 产生的 `-inf` 单独记录；它不会被误计为 NaN。没有有限元素时统计值为 `null`，报告使用严格 JSON。

比较会检查两侧保存的全部张量名称，报告缺失项、shape 差异、最大绝对误差、相对 L2 误差、RMSE、超阈值比例、最差索引及两侧数值，并按约定的计算顺序指出首个不一致项。相同符号的无穷可以相等，NaN 即使在两侧同位置出现也判为不一致。bool/整数采用精确比较。浮点误差以第二个记录为参考；默认 `atol=1e-5, rtol=1e-4`，可显式调整。比较失败时命令返回非零状态。

比较范围为实际保存的张量；`selected` 记录相互比较通过不代表未保存的中间结果也通过。`summary` 没有原始数据，工具会拒绝数值比较。不同通信阶段的梯度也不能直接混比。

完整记录的重放不依赖原训练任务：重新创建模块，载入前向时权重、输入及 requires-grad 配置，恢复训练/eval 模式、CPU 线程数与前向前 RNG，然后使用保存的上游梯度执行反向。实际 dropout mask 也包含在记录中供核对。它使用独立的 RNG 作用域，结束后恢复调用方 RNG 和线程数；默认在临时目录重算，指定 `--output` 可保留完整重放记录。

重放结果写入源记录目录的 `replay_report.json`，包含输出、中间结果、输入梯度和参数梯度的逐项比较。只有 forward 的记录也可重放。`--atol 0 --rtol 0` 用于本机相同环境下的精确重放验证；跨版本/机器可按计算精度采用固定容差，不承诺跨环境逐 bit 一致。

记录通过临时目录完成写入后再发布，加载时检查 schema、完成标记和文件 SHA256。损坏或不完整的记录会被拒绝。训练恢复时 step 继续使用全局步数，因此可以另选恢复后的步骤保存。

### 在 Python 中捕获一次调用

```python
from attention_cpu.capture import CaptureConfig, CaptureWriter

writer = CaptureWriter(CaptureConfig(root="runs", run_id="python_capture", mode="full"))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
with writer.record(model, step=1, layer=0, rank=0) as record:
    output = record.forward(x)
    output.square().mean().backward()
optimizer.step()
print(record.path)
```

支持 `CPUSelfAttention` 和 `CPUAttentionCore`，每个上下文只执行一次 forward，反向应在上下文内、optimizer 更新之前完成。loss 应通过捕获的输出向该模块传播梯度。输入未开启 requires-grad 时，工具为模块输入建立局部可微叶子以保存 dInput，不改变调用方输入的 requires-grad 标志。

### 显式分片重建

已提供离线工具，按元数据恢复保存的分片。该工具不执行通信，离线重建通过也不能证明 Gloo/DDP 运行正确。

```bash
python -m tools.reconstruct_results <rank_0记录目录> <rank_1记录目录> \
  --output runs/reconstructed.pt
```

`metadata.json` 的 `partitions` 为每个完整张量名指定：

- `replicated`：`global_shape` 与本地 shape 一致，所有副本必须逐元素相同。
- `indexed`：指定 `global_shape`、`axis` 和本地元素对应的全局 `indices`，索引须无重叠、无缺口。

例如完整 QKV 权重 shape 为 `[12,4]`、每段宽度为 4，两个 rank 按 head 切分时，行索引分别为 `[0,1,4,5,8,9]` 和 `[2,3,6,7,10,11]`，不能直接将合并矩阵上下均分。捕获时可通过 `writer.record(..., partitions={"forward.query": {...}})` 传入已知分片映射；未指定的张量默认为完整副本。

重建同时支持前向张量、输入梯度和参数梯度的轴索引排列。单元测试覆盖 head、QKV 权重行、输出投影列和副本校验；部分和、GQA KV 复制归约及改变并行规模的 optimizer 状态重建不在此离线工具范围内。工具拒绝覆盖已有重建结果。

本机已运行示例为 [runs/capture_demo/training_report.json](runs/capture_demo/training_report.json)，分析报告为 [runs/capture_demo/analysis_report.json](runs/capture_demo/analysis_report.json)，重放和比较报告为 [runs/capture_demo/captures/demo/step_1/layer_0/rank_0/replay_report.json](runs/capture_demo/captures/demo/step_1/layer_0/rank_0/replay_report.json) 与 [runs/capture_demo/comparison_report.json](runs/capture_demo/comparison_report.json)。

## 单进程 checkpoint

训练入口使用 AdamW，并保存：

- 模型配置和参数。
- 优化器状态与已完成 step。
- 固定 batch 的重建配置。
- CPU PyTorch 随机状态和 loss 历史。

训练使用独立随机生成器重建数据，并在恢复参数和优化器后恢复 dropout 随机状态。测试验证了包含 dropout、RoPE 的训练中，连续运行与分段恢复的参数、优化器状态、随机状态和 loss 历史完全一致。

## 阶段④：TP/DDP 训练

### 实现与运行条件

`attention_cpu/parallel/` 包含进程组、可微通信映射、并行线性层、Attention、训练、分片 checkpoint 和离线并行重放。全局 rank 为 `dp_rank * tp_size + tp_rank`，TP 组内输入和位置/mask 配置须一致，DDP 仅绑定到对应 DP 组。

QKV 按每段内的 head 索引切分；输出投影切分输入列，局部输出求和后加入一次共享 bias。复制输入在反向进行 TP SUM，输出求和的反向为恒等。MHA 要求 head 数可被 TP 整除；GQA/MQA 仍未实现。

以下命令要求允许 socket 的运行环境。四进程运行 `TP=2, DP=2`，在可用环境中可执行：

```bash
conda activate pytorch_py311
export GLOO_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m attention_cpu.train --config configs/capture_demo.json \
  --tp-size 2 --dp-size 2 --batch-size 5 --grad-accum-steps 2 \
  --variable-lengths --seq-len 9 --steps 4 --threads 1 \
  --output runs/tp2_dp2 --capture-mode full --capture-steps 1
```

也可运行 `bash scripts/train_parallel_local.sh --config configs/capture_demo.json --steps 4 --output runs/parallel_example`。`lo` 仅适合本机；跨机需设置实际互通网卡。各进程必须看到同一份代码与配置；当前多机输出/checkpoint 需要所有 rank 可访问的同一路径共享文件系统。

`--batch-size` 在并行训练中表示跨 DP 和全部 microbatch 的全局有效 batch，必须至少为 `dp_size * grad_accum_steps`，无需整除。`--variable-lengths` 开启不同有效 token 数的合成样本。loss 使用局部平方误差和乘 `dp_size/global_valid_element_count`，配合 DDP 平均和 `no_sync()` 累积，得到全局有效元素均值。

训练的 dropout mask 根据 seed、optimizer step 和全局样本/head 索引确定，再切到 DP 样本与 TP head，保持比较及恢复时的随机语义。直接使用 `TensorParallelSelfAttention` 而未传 mask 时，TP peers 需要相同 CPU RNG 状态。训练数据、完整 dropout mask 和初始化完整权重目前会在每个 rank 上构造；这是正确性实现，初始化峰值内存与通信重叠优化留在性能阶段。

### 保存、重建与并行重放

并行训练记录按 optimizer step 和 microbatch 区分：

```text
<output>/captures/<run_id>/step_1/micro_0/layer_0/rank_0/
<output>/captures/<run_id>/step_1/micro_0/layer_0/rank_1/
```

完整并行记录保存局部权重与显式全局索引。单个 TP 分片不能用 `tools.replay_capture` 重算完整模块，需要提供同一 DP 副本、相同步骤/microbatch 的全部 TP peers：

```bash
python -m tools.replay_parallel_capture \
  runs/tp2_dp2/captures/training/step_1/micro_0/layer_0/rank_0 \
  runs/tp2_dp2/captures/training/step_1/micro_0/layer_0/rank_1 \
  --output runs/tp2_dp2/local_replay_report.json
```

该工具离线重建完整权重，运行单个完整 CPU Attention，再切回原分片逐项验证前向、本次反向和 TP dInput 求和。它不重放 DDP collective，明确列出未验证的 `post_dp.*`；不能将它的通过结果当作 DP 同步验收。跨 DP 的样本不同，不应混成同一 TP 重建组。

### 分片 checkpoint 与恢复

并行路径使用 schema-v2 checkpoint：`checkpoints/step_N/rank_R.pt` 保存参数、optimizer、RNG、配置及 step，各 rank 完成后由 rank 0 提交带全部 SHA256 的 `manifest.json`。加载时校验完整性与 topology，只支持相同 TP/DP、batch、累积、优化器和线程配置。训练结束还输出重建的 `full_model.pt`，以及各 rank 的训练报告。

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m attention_cpu.train --resume runs/tp2_dp2/checkpoints/step_4 \
  --steps 6 --output runs/tp2_dp2_resumed
```

TP/DP 与训练配置可从 manifest 自动读取，但启动进程数仍须匹配。schema-v1 单进程 checkpoint 保持原格式和原入口；不支持直接转换成并行 checkpoint 或改变并行规模。

### 当前验证状态

```bash
python -m tools.verify_parallel --require-gloo --output runs/parallel_validation_new/report.json
```

该入口覆盖 `(TP,DP)=(2,1),(1,2),(2,2)` 的等价性；在 socket 可用时还会启动真实 Gloo + DDP，并检查梯度累积与分片恢复。`--require-gloo` 在通信测试因环境跳过时返回 **2**，测试失败返回 **1**，全部验收通过才返回 **0**。普通 `unittest` 会明确显示跳过项。

可使用 `--offline-capture-root <新目录>` 另存明确标注为模拟的 TP/DP 记录和离线重放证据。当前 [并行验证报告](runs/parallel_validation/report.json) 记录 8 项并行测试中 7 项通过、1 项真实 Gloo 测试被 socket 权限阻止，`stage4_fully_validated=false`。模拟 SUM 仅存在于测试代码，不作为生产通信后端。

不依赖 socket 的单 rank 并行训练路径也已实际运行：全局 batch 5、两次梯度累积、变长样本和 dropout/RoPE，四步后恢复到六步。见 [训练报告](runs/parallel_single_rank/training_report.json)、[恢复报告](runs/parallel_single_rank_resumed/training_report.json) 和 [局部重放报告](runs/parallel_single_rank/replay_report.json)。这些结果不表示多进程或跨机已验收。

## 本机验证结果

环境：Python `3.11.17`、PyTorch `2.14.0+cpu`、NumPy `2.4.6`，CPU Intel Xeon Platinum 8558P。

| 检查 | 实际结果 |
| --- | --- |
| `unittest` | 39 项：38 项通过，1 项实际 Gloo 测试因 socket 权限跳过 |
| 已保存的独立基准 | 六组均通过，覆盖核心与完整模块 |
| FP32 输出 vs NumPy FP64 | 六组案例中 FP32 输出最大绝对误差约 `1.15e-7` |
| FP64 输出 vs NumPy FP64 | 六组案例中 FP64 输出最大绝对误差约 `2.22e-16` |
| 梯度检查 | 核心、带 RoPE 完整模块输入及全部参数、RoPE 均通过 FP64 gradcheck |
| 标准参考 | 完整模块与 `nn.MultiheadAttention` 输出/梯度及一次 SGD 更新通过交叉验证；核心与 SDPA 比较通过 |
| 默认 30 步训练 | loss 从 `0.232556` 降至 `0.004211`，更新后 eval loss 为 `0.003974` |
| 命令行恢复 | 从 30 步 checkpoint 恢复至 35 步，最后训练 loss 为 `0.002958` |
| dropout 恢复测试 | 分段续训与连续训练逐张量一致 |
| 保存与重放 | 完整记录可独立重放；dropout、RoPE、前反向、冻结参数与已有梯度累积均已验证 |
| 保存开关等价性 | 相同训练的 loss、参数和 RNG 状态逐项一致 |
| 筛选/统计/比较 | 模式与范围筛选、文件损坏检测、NaN/Inf、缺失张量和最早偏差定位测试通过 |
| 离线分片重建 | head、QKV 行、矩阵列和副本检查测试通过；实际 TP 通信待验收 |
| TP/DP 算法 | 输出、dInput、参数梯度和 SGD 更新与单进程参考一致；不等有效 token 数和梯度累积通过离线验证 |
| 并行 checkpoint | 单 rank schema-v2 参数/优化器精确恢复通过；实际多 rank 恢复入口已提供、待通信环境验收 |

实际训练报告为 [runs/single_process/training_report.json](runs/single_process/training_report.json) 和 [runs/resumed/training_report.json](runs/resumed/training_report.json)。

TP/DP 代码已交付，实际 Gloo/DDP 多进程和跨机运行仍待具备 socket 权限的环境验收。GQA/MQA、BF16/FP16 与融合性能路径继续留作后续阶段。

## 两类多机验收（新增入口）

完整设计见 [multi_node_acceptance_plan.md](multi_node_acceptance_plan.md)。已提供真实 Gloo 验收 worker、固定基准对照、CPU 绑定、SSH/逐节点启动、容量短探测、CPU profiler 和严格报告汇总。结果中的 `LOCAL_ONLY`、`PROBE_PASS` 都不表示多机正式验收通过。

### 配置节点与启动

已根据实际节点清单生成 `configs/acceptance_cluster.json`：`cpui01-2`～`cpui16-2` 对应 `10.10.11.22`～`10.10.11.37`，master 为 `10.10.11.22`。SSH 使用 `ip` 字段，`interface: "auto"` 在各节点通过 `ip -j -4 addr show` 匹配本节点 IP；匹配不到或有歧义则停止。可手动填写具体网卡名覆盖自动识别。已有配置无需再复制模板。

复制 `configs/acceptance_cluster.example.json` 为 `configs/acceptance_cluster.json`，填写实际 16 台 hostname、各自通信网卡、master 通信 IP、所有节点一致的 conda Python 绝对路径和共享代码目录。节点名称补零规则请使用实际主机清单。结果目录也必须是所有节点可见的相同绝对路径。每次执行使用新目录。

激活 `pytorch_py311` 后，在共享项目目录预览启动命令：

```bash
python -m tools.launch_acceptance \
  --inventory configs/acceptance_cluster.json \
  --suite correctness-performance \
  --output /share/home/renhj/attention_cpu/runs/cluster_correctness_001
```

默认只打印 SSH 命令，不执行。加入 `--execute` 后会在清单前两台机器各启动两个 worker，自动完成 TP4/DP1、TP2/DP2 的数值验证及性能测量，并汇总至输出目录下 `formal/acceptance.md`。默认每 worker 绑定 44 个物理核；可用 cpuset 不足时以 `--threads` 调整。

最大并行度测试直接使用清单全部 16 台机器，每 worker 一个计算线程：

```bash
python -m tools.launch_acceptance \
  --inventory configs/acceptance_cluster.json \
  --suite max-parallel --workers-per-node 96 --threads 1 \
  --capacity-search --port 29501 \
  --output /share/home/renhj/attention_cpu/runs/cluster_max_001 \
  --execute
```

`--capacity-search` 先做 3 步短探测，仅在日志明确显示物理核不足、内存分配失败、线程资源不足或文件描述符不足时按偶数减半回退。网络、数值、超时或 SSH 断连不会被当作容量问题自动回退。成功后另起正式作业做数值检查、10 步预热、50 步计时及 3 步 active profiler。省略该选项即直接进行指定规模的完整测试；可用 `--workers-per-node 48` 等显式选择较低进程数。

自动探测输出 `capacity_attempts.json` 与 `launch_summary.json`，只声明最大实测规模，不声称证明所有中间进程数都无法运行。节点级超时会清理本作业进程组；SSH 断连时远端 timeout 提供生命周期上限，控制端停止继续探测。SSH 模式要求免交互登录及远端 `timeout` 命令。

采用调度器时，每台分配节点只运行一个 agent，并各自传入不同 `--node-rank`：

```bash
bash scripts/run_acceptance_node.sh \
  --inventory configs/acceptance_cluster.json --suite max-parallel \
  --workers-per-node 96 --threads 1 --node-rank 0 --port 29501 \
  --output /share/home/renhj/attention_cpu/runs/scheduled_max_001
```

其他节点分别改为 `--node-rank 1` 至 `15`，全部完成后运行：

```bash
python -m tools.aggregate_acceptance runs/scheduled_max_001
```

绑定器根据当前 cpuset 与 sysfs 物理 core/package 拓扑选择互不重叠的核心，进程启动后采用 first-touch 分配；不宣称强制 NUMA membind。调度器应给每台 agent 分配该节点所有计划使用的核，由绑定器切分给 worker。

### 报告与算子分析

- `acceptance.md/json`：状态、完整 rank 检查、错误原因和性能摘要。
- `numerical_report.json`：每 rank 的输出、loss、dInput、参数梯度和 SGD 更新误差；参考文件有 SHA256 校验。
- `performance.json`：前向/反向/optimizer/整步 p50/p95、吞吐、最慢 rank、进程 CPU/RSS/线程/上下文切换、TP/DP 通信完成时间及逻辑字节。
- `operator_analysis.md`：采样 rank 的 Top 10 算子、shape、调用数、self/total CPU 时间、内存事件、主要矩阵 FLOPs 与有效 GFLOP/s。
- `tp*/traces/rank_*.json`：CPU trace。每节点 local rank 0 和最慢 rank 采集，避免所有 worker 同时导出大 trace。

profiler 窗口与正式计时分开，通信完成耗时可能与计算重叠，不直接相加求墙钟占比。当前资源指标来自 worker 进程与 `/proc`，未采集硬件带宽计数器；分析不能据此声称达到 CPU/内存峰值。

当前会话可运行的端到端自检：

```bash
python -m tools.acceptance_worker --suite correctness-performance \
  --local-smoke --steps 3 --warmup 1 --output runs/local_correctness_new
python -m tools.acceptance_worker --suite max-parallel \
  --local-smoke --steps 3 --warmup 1 --output runs/local_max_new
python -m tools.aggregate_acceptance runs/local_correctness_new runs/local_max_new
python -m unittest tests.test_acceptance -v
```

本次本地示例见 [正确性与性能报告](runs/acceptance_local_v1/correctness/acceptance.md) 和 [最大并行度工具链的单进程算子报告](runs/acceptance_local_v1/max/operator_analysis.md)，均标为 `LOCAL_ONLY`。尚无真实 16 节点或 1536 worker 通过证据。

### 远程执行记录与便捷入口

2026-10-14 已尝试真实 SSH 启动；先遇系统 SSH 配置权限错误，指定 `--ssh-config /dev/null` 后，当前会话仍因 `socket: Operation not permitted` 无法连接节点。远端 worker 未启动，详见 [执行分析](runs/remote_acceptance_20261014_155058/execution_analysis.md)。

在允许 SSH/socket 的服务器普通终端执行 `bash scripts/run_remote_acceptance.sh`，即可顺序运行两类验收并自动汇总。脚本使用新的时间戳目录；可用第一个参数指定新的结果根目录。

### SSH 主机密钥确认

若节点日志报 `Host key verification failed`，需要在实际运行控制脚本的机器（如 mgt02）、同一用户下先交互式连接节点。验收脚本使用 IP、`-F /dev/null` 和 `BatchMode=yes`，无法弹出首次连接确认。可先执行：

```bash
for last in $(seq 22 37); do
  ssh -F /dev/null "10.10.11.$last" true || break
done
```

首次连接时核对主机指纹并确认；若提示密钥已更改，应先核实原因，不关闭主机密钥检查。若要求输入密码，需配置免密登录后再运行非交互验收。全部连接成功后重新执行 `bash scripts/run_remote_acceptance.sh`。启动器现在会直接打印失败节点及日志尾部。

## 通过景行 jsub 提交（登录节点）

已按《景行资源管理与调度软件 V6.4 用户手册》实现 `jsub` 路径：资源约束见印刷页 94，提交/依赖参数见第 4.1 节（145 页起），原生远程执行 `jexec` 见第 4.15 节（170 页）。现场安装版本的 `jsub -h`、`jexec -h` 也已核对；`JH_HOSTS` 的 host/slots 成对格式来自现场 `/share/jhjob_script/getJobHosts.sh`。

在 **mgt02 登录节点，只执行一次**：

```bash
cd /share/home/renhj/attention_cpu
bash scripts/submit_acceptance_jsub.sh
```

默认提交到 `normal` 队列，可加 `--queue 队列名`。只生成脚本而不提交时加 `--dry-run`。每次使用新结果目录；也可用 `--output /绝对路径/新目录` 指定。

| 作业 | 节点 | 申请 CPU slots | 实际 Attention worker |
|---|---|---:|---:|
| correctness-performance | cpui01、cpui02 | 192（每节点 96） | 4（每节点 2，每 worker 44 线程） |
| max-parallel | cpui01～cpui16 | 1536（每节点 96） | 目标 1536（每 worker 1 线程），容量不足时短探测回退 |

提交器自动将第二项设置为 `-w 'done(第一项作业号)'`，第一项成功结束后第二项才可调度。CPU slots 是资源预留量，不能把第一项申请成 4 slots 后启动 176 个计算线程。两项都用 `span[ptile=96]`，单项运行上限为 120 分钟。

`configs/acceptance_cluster.json` 新增 `scheduler_host`（cpui01～cpui16），用于调度资源选择；`host` 保留 cpui01-2 等通信别名，`ip` 保留 10.10.11.22～37。若登录节点 `jhosts` 显示的调度节点名不同，应据此修改 `scheduler_host`。已观察到当前节点实际 hostname 为 cpui01；全清单尚不能在受限会话内向调度服务核实。

作业启动后检查 `JH_JOBID` 和实际 `JH_HOSTS`，只允许在申请到的节点运行，每节点分配不足 96 slots 时停止。使用调度器 `jexec` 启动每节点的 torchrun agent，**不调用普通 SSH，也不依赖 MPI**。该远程执行方式需要当前用户具有 `jexec` 权限；若调度服务拒绝，错误会写入 node 日志，不回退为未分配资源上的 SSH 运行。

controller 在共享目录维护心跳；远端 guard 在 controller 退出或心跳超过 90 秒未更新时停止该作业的 agent。该机制补充 `jexec` 派生进程的生命周期控制。正常任务的节点级超时也会清理本作业进程组。

输出根目录保存两份 `.jsub` 脚本、`submission.json`（真实作业号/依赖）、`.submit.log`、调度 stdout/stderr 和 allocation 清单。正式验收结果位于各类别的 `formal/` 子目录。常用查询：

```bash
jjobs
jjobs -l 作业号
jctrl peek 作业号
```

只有 `submission.json` 记录真实 job ID 才表示提交已确认。提交超时或无法解析确认信息时先检查 `jjobs`，不要反复提交重复作业。

当前已生成可审阅脚本：[两机任务](runs/jsub_prepared_20261014/correctness-performance.jsub)、[16 节点任务](runs/jsub_prepared_20261014/max-parallel.jsub)。从当前会话连接 mgt02 提交的尝试被 socket 权限阻止，尚无真实 jsub 作业号，记录见 [提交尝试](runs/jsub_prepared_20261014/submission_attempt.md)。

## 仅验证 cpui01 与 gpu03 的 CPU

专用配置 `configs/acceptance_cpu_gpu.json`：cpui01-2（10.10.11.22）＋gpu03-2（10.10.11.18）。每台 2 worker、每 worker 1 个 CPU 线程，覆盖 TP4/DP1 和 TP2/DP2，仅执行第一类验收，不使用 GPU，也不启动 16 节点测试。

在登录节点执行：

```bash
cd /share/home/renhj/attention_cpu
bash scripts/submit_acceptance_jsub.sh \
  --inventory configs/acceptance_cpu_gpu.json \
  --suite correctness-performance --slots-per-node 2 --threads 1
```

只提交一个作业，共申请 4 CPU slots，每节点 2 slots。队列默认为 normal；若 gpu03 仅属于其他队列，使用 `--queue` 指定能同时调度两台节点的队列。调度节点名 cpui01/gpu03 应与登录节点 `jhosts` 输出一致。

本轮属于异构两机、低线程验证，不能将结果当作同构 16 节点扩展效率。当前会话已尝试连接两台节点，但均因本地 socket 权限失败，尚无实际通过结果，见 `runs/cpu_gpu_validation_attempt_20261014/analysis.md`。
