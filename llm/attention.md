---
description: 从缩放点积注意力理解 Transformer 的信息聚合方式。
date: 2026-09-22
---

# Transformer 注意力机制

Attention 允许模型根据相关性动态聚合上下文信息，是 Transformer 的核心组件。

## Scaled Dot-Product Attention

给定查询、键和值矩阵，注意力计算为：

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

缩放因子 $\sqrt{d_k}$ 可以避免维度较大时点积值过大，减轻 softmax 梯度过小的问题。

## PyTorch 示例

```python
import torch
from torch import nn

x = torch.randn(8, 32, 64)
attention = nn.MultiheadAttention(64, 4, batch_first=True)
output, weights = attention(x, x, x)
```

