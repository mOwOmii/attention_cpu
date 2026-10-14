# CPU Attention

本项目实现标准多头自注意力的 CPU 前向和反向计算，支持 FP32/FP64、因果与非因果模式、Attention mask、padding mask、dropout 和 RoPE。提供单进程训练与 checkpoint、前反向记录和重放，以及基于 Gloo 的张量并行（TP）和数据并行（DP）。

实施方案见 [attention_implementation_plan.md](attention_implementation_plan.md)；多机验收设计见 [multi_node_acceptance_plan.md](multi_node_acceptance_plan.md)。

## 环境与自检

需要 Python 3.11、PyTorch 和 NumPy。本机 `cpu_tp` Conda 环境已包含这些依赖。以下命令均在项目根目录执行：

```bash
conda activate cpu_tp
python -m unittest discover -s tests -v
python -m tools.verify_parallel --require-gloo --output runs/parallel_validation/report.json
```

第二条验证真实 Gloo 通信，需允许本机进程创建 socket；如果环境禁止 socket，工具以退出码 2 报告该项未验证。`runs/` 中的结果由命令生成，不随源码保存。

## 单进程训练

```bash
python -m attention_cpu.train \
  --config configs/single_process.json --steps 30 --output runs/example
python -m attention_cpu.train \
  --resume runs/example/checkpoint.pt --steps 35 --output runs/example_resumed
```

`--steps` 是包含已完成步骤的总目标步数。恢复训练时沿用 checkpoint 中的模型、数据、优化器和线程配置。示例使用固定合成 batch 验证训练流程。

Python 接口示例：

```python
import torch
from attention_cpu import AttentionConfig, CPUSelfAttention

config = AttentionConfig(embed_dim=64, num_heads=4, causal=True, rope=True)
model = CPUSelfAttention(config)
x = torch.randn(2, 16, 64, requires_grad=True)
y = model(x)
y.square().mean().backward()
```

模块输入布局由配置的 `input_layout` 指定。所有张量和参数必须位于 CPU，且输入 dtype 必须与配置一致。bool `attn_mask` 与 `key_padding_mask` 的 `True` 表示屏蔽；浮点 `attn_mask` 会加到 Attention logits。完整模块还支持 `positional_bias`、`position_ids`、`dropout_mask` 和 `return_intermediates` 参数。

## 参考基准

```bash
python -m tools.generate_reference --output runs/reference --threads 1
python -m tools.verify_reference --reference runs/reference --threads 1
```

生成器用 NumPy FP64 独立计算前向与解析梯度，验证器对照 CPU Attention 的输出和梯度。生成器要求输出目录不存在或为空。

## 前反向记录与重放

```bash
python -m attention_cpu.train \
  --config configs/capture_demo.json --steps 4 --threads 1 \
  --output runs/capture_demo --capture-mode full --capture-steps 1,3
python -m tools.analyze_capture \
  runs/capture_demo/captures/training/step_1/layer_0/rank_0
python -m tools.replay_capture \
  runs/capture_demo/captures/training/step_1/layer_0/rank_0
```

`--capture-mode` 可设为 `none`（默认）、`summary`、`selected` 或 `full`；只有 `full` 保存独立重放所需的完整输入、权重、随机状态和梯度。`tools.compare_results` 可比较两份记录，`tools.reconstruct_results` 可按显式分片索引重建张量。并行记录需用 `tools.replay_parallel_capture` 提供同一 DP 副本的全部 TP 分片。

## 本机 TP/DP 训练

```bash
bash scripts/train_parallel_local.sh \
  --config configs/capture_demo.json --steps 4 --output runs/parallel_example
```

脚本用四个进程运行 TP=2、DP=2，默认通过本机 `lo` 网卡通信。跨机运行时需配置实际互通网卡、torchrun 的主节点地址与端口，并让所有 rank 访问同一份代码及输出目录。并行 checkpoint 只支持以相同 TP/DP、batch、梯度累积和线程配置恢复。

## 多机验收

正式多机验收的资源、拓扑和报告要求以 [multi_node_acceptance_plan.md](multi_node_acceptance_plan.md) 为准。`configs/acceptance_cluster.example.json` 提供节点清单模板；`tools.launch_acceptance` 提供远程启动入口，`scripts/submit_acceptance_jsub.sh` 提供景行调度器提交入口。使用这些入口前须核对节点清单、共享路径、Python 路径和调度资源。

源码、配置与测试保留在仓库中；训练 checkpoint、捕获张量、trace 和验收报告写入已忽略的 `runs/` 目录。
