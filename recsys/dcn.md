# 深度交叉网络（Deep & Cross Network，DCN）

> DCN（Deep & Cross Network）是 Google 于 2017 年提出的推荐模型，发表于 AdKDD 2017，论文题为 Deep & Cross Network for Ad Click Predictions。[1]

DCN 的核心思想是通过 Cross Network 显式学习有界阶数的特征交叉，同时利用 Deep Network 学习复杂的非线性特征关系，将两者结合用于点击率（CTR）预测。

## 1. 背景（Background）

特征交叉是 CTR 预测中的重要技术。传统线性模型通常依赖人工构造交叉特征，例如用户性别与商品类别的组合，而深度神经网络（DNN）虽然能够自动学习非线性关系，但其特征交互是隐式建模的，不一定能高效学习特定的交叉关系。

DCN 针对这一问题，引入 Cross Network，通过特殊的交叉层结构自动学习显式的高阶特征组合，并与 DNN 并行建模，在减少人工特征工程的同时，提高模型对复杂特征关系的表达能力。

## 2. 模型架构（Model Architecture）

### 2.1 整体网络结构

DCN 的整体模型架构如下图所示。

[deep & cross network for ad click predictions - daiwk-github博客](https://images.openai.com/static-rsc-4/HPpBNJ3HESOkba0VvPtwmzPE2qtvbBBegCaxIj6zU4u6_GqSKvhx2y5Sz8ybAeSDeiogyxxsCH_FkCFodBGAXyOBkFRyL--po6XEce2cLNwvx1-4N8IzthHUgANGZnEuVmjKFaj_2juq45AYayoEFKBEgGDOyVfmQh2eHP-rwOE?purpose=inline)

[daiwk.github.io](https://daiwk.github.io/posts/dl-deep-cross-network.html)

图 1：DCN 模型整体架构。参考 Wang 等人提出的 Deep & Cross Network [1]。

DCN 主要由四个部分组成：

- Embedding & Stacking Layer：将稀疏类别特征转换为低维 Embedding，并与数值特征拼接，形成统一输入。
- Cross Network：通过交叉层显式建模特征之间的有界阶数交互，是 DCN 的核心创新。
- Deep Network：通过多层全连接神经网络学习复杂的非线性特征关系。
- Combination Layer：拼接 Cross Network 和 Deep Network 的输出，生成最终预测结果。

原始 DCN 采用并行结构，Cross Network 和 Deep Network 接收相同的输入，分别学习不同类型的特征关系，最后将两部分输出融合进行预测。

### 2.2 输入层（Embedding & Stacking Layer）

推荐系统的输入通常包含两类特征：

- 稀疏类别特征：例如用户 ID、商品 ID、商品类别等。
- 稠密数值特征：例如用户年龄、商品价格、历史点击次数等。

对于稀疏类别特征，首先通过 Embedding 层映射为低维稠密向量：

\\[ e_i=\operatorname{Embedding}(x_i) \\]

然后将各个 Embedding 向量与归一化后的数值特征进行拼接：

\\[ x_0=[e_1;e_2;\cdots;e_k;x\_{\mathrm{dense}}] \\]

其中：

- \\(e_i\\)：第 \\(i\\) 个类别特征的 Embedding 向量。
- \\(x\_{\mathrm{dense}}\\)：归一化后的数值特征。
- \\(x_0\in\mathbb{R}^{d}\\)：拼接后得到的统一输入向量。

这一输入向量将同时送入 Cross Network 和 Deep Network。

### 2.3 交叉网络（Cross Network）

Cross Network 是 DCN 的核心模块，其主要作用是通过显式的特征交叉构建高阶组合特征。

#### （1）Cross Layer 的计算公式

对于第 \\(l\\) 层 Cross Layer，原始 DCN 定义：

\\[ \boxed{x\_{l+1}=x_0(x_l^Tw_l)+b_l+x_l} \\]

其中：

- \\(x_0\in\mathbb{R}^{d}\\)：原始输入向量。
- \\(x_l\in\mathbb{R}^{d}\\)：第 \\(l\\) 层的输入。
- \\(w_l\in\mathbb{R}^{d}\\)：可学习的权重向量。
- \\(b_l\in\mathbb{R}^{d}\\)：偏置向量。
- \\(x\_{l+1}\in\mathbb{R}^{d}\\)：当前 Cross Layer 的输出。

该公式可以拆分为三个部分：

\\[ \underbrace{x_0(x_l^Tw_l)}\_{\text{显式特征交叉}} + \underbrace{b_l}\_{\text{偏置项}} + \underbrace{x_l}\_{\text{残差连接}} \\]

首先计算 \\(x_l^Tw_l\\)，得到一个标量；然后将该标量乘以原始输入 \\(x_0\\)，生成交叉特征表示；最后加入偏置项和上一层输入，得到新的输出。

[DCN — easy_rec 0.8.8 documentation](https://images.openai.com/static-rsc-4/X5o686UYglcfjq-QhGQGlJ-8sGUHMfOmeJOIhMCAjqmfFb-mUVR0M1oul4XyNlc7TxiCTJqqGBEWORc1CI-7cepEPnZGcf2DFcYLZW33QbIq1XaMQjJ33QBFKdR8F20RISUnYG9-pwNNOhQ83BGyI1vYLnzOn__V1_wez55W-Mo?purpose=inline)

[easyrec.readthedocs.io](https://easyrec.readthedocs.io/en/latest/models/dcn.html)

图 2：Cross Layer 计算结构。原始输入与当前层表示进行交叉，并通过残差连接保留已有特征。

#### （2）高阶特征交叉

Cross Network 的关键特性是：随着网络层数增加，模型能够构建更高阶的特征交互。

为了直观理解，暂时忽略偏置项，假设输入向量为：

\\[ x_0= \begin{bmatrix} a\\\b \end{bmatrix} \\]

第一层 Cross Layer：

\\[ x_1=x_0(x_0^Tw_0)+x_0 \\]

设：

\\[ w_0= \begin{bmatrix} w_1\\\w_2 \end{bmatrix} \\]

则：

\\[ x_0^Tw_0=aw_1+bw_2 \\]

代入可得：

\\[ x_1= \begin{bmatrix} a^2w_1+abw_2+a\\\ abw_1+b^2w_2+b \end{bmatrix} \\]

可以看到，经过一层 Cross Layer，模型便能够生成：

\\[ a^2,\quad ab,\quad b^2 \\]

这些二阶交叉特征。

继续堆叠 Cross Layer，还可以进一步构建：

\\[ a^3,\quad a^2b,\quad ab^2,\quad b^3 \\]

等更高阶交互项。

对于包含 \\(L\\) 层的 Cross Network，输出特征关于原始输入的多项式次数最高可达到：

\\[ \boxed{L+1} \\]

因此，通过控制 Cross Layer 的层数，可以控制模型显式特征交叉的最高阶数。

需要注意，Cross Network 并不是为所有交叉项分别学习独立参数，而是采用参数共享的方式表示这些交互，因此表达能力会受到其结构约束。

#### （3）参数量与计算复杂度

Cross Layer 的参数只有两个长度为 \\(d\\) 的向量：

\\[ w_l,b_l\in\mathbb{R}^{d} \\]

因此，单层参数量为：

\\[ 2d \\]

假设包含 \\(L\\) 层 Cross Layer，则总参数量为：

\\[ \boxed{2dL} \\]

单层 Cross Layer 主要包含一次向量内积、一次向量与标量乘法及向量加法，因此每个样本的计算复杂度为：

\\[ O(d) \\]

整个 Cross Network 的计算复杂度为：

\\[ O(Ld) \\]

与显式枚举全部高阶组合特征相比，这种结构能够以较低的参数与计算开销学习有界阶数的特征交互。

### 2.4 深度网络（Deep Network）

Cross Network 主要负责显式特征交叉，但由于其交叉结构受到一定约束，单独使用时难以充分表达复杂的非线性关系。

因此，DCN 在 Cross Network 之外引入并行的 Deep Network，用于学习更加灵活的非线性特征表示。

Deep Network 通常由多层全连接神经网络构成：

\\[ h\_{l+1}=\sigma(W_lh_l+b_l) \\]

其中：

- \\(h_l\\)：第 \\(l\\) 层隐藏表示。
- \\(W_l\\)：权重矩阵。
- \\(b_l\\)：偏置向量。
- \\(\sigma\\)：非线性激活函数，原论文使用 ReLU。

其初始输入为：

\\[ h_0=x_0 \\]

经过多层非线性变换，得到 Deep Network 的最终输出：

\\[ h\_{L_d}=\operatorname{DNN}(x_0) \\]

Cross Network 与 Deep Network 的区别如下：

| 对比维度   | Cross Network | Deep Network |
| ------ | ------------- | ------------ |
| 主要作用   | 显式特征交叉        | 隐式非线性特征建模    |
| 核心计算   | 向量内积与交叉       | 全连接层与激活函数    |
| 特征交叉阶数 | 最高阶数受层数控制     | 无直接对应的阶数约束   |
| 参数量    | 随输入维度线性增长     | 取决于网络宽度和深度   |

两者并行使用，可以结合显式交叉和隐式非线性建模的优势。

### 2.5 融合层（Combination Layer）

经过 Cross Network 和 Deep Network 后，分别得到：

\\[ x\_{L_c}\in\mathbb{R}^{d} \\]

\\[ h\_{L_d}\in\mathbb{R}^{m} \\]

其中 \\(L_c\\) 和 \\(L_d\\) 分别表示两个网络的层数。

将两个网络的输出进行拼接：

\\[ z=[x\_{L_c};h\_{L_d}] \\]

然后通过线性层和 Sigmoid 函数输出预测点击率：

\\[ \boxed{\hat y=\sigma(w^Tz+b)} \\]

其中：

\\[ \sigma(t)=\frac{1}{1+e^{-t}} \\]

最终预测结果：

\\[ \hat y\in(0,1) \\]

可以解释为模型预测的点击概率。

### 2.6 损失函数（Loss Function）

DCN 主要面向 CTR 预测任务，因此通常采用二元交叉熵（Binary Cross-Entropy，BCE）作为损失函数：

\\[ \mathcal{L}\_{BCE} = -\frac{1}{N}\sum\_{i=1}^{N} \left[ y_i\log\hat y_i+ (1-y_i)\log(1-\hat y_i) \right] \\]

其中：

- \\(N\\)：训练样本数量。
- \\(y_i\in\\{0,1\\}\\)：真实点击标签。
- \\(\hat y_i\\)：预测点击概率。

此外，原论文还引入 L2 正则化约束模型参数：

\\[ \mathcal{L} = \mathcal{L}\_{BCE} + \lambda\sum\_{\theta\in\Theta\_{\mathrm{reg}}}\\|\theta\\|\_2^2 \\]

其中，\\(\Theta\_{\mathrm{reg}}\\) 表示参与正则化的模型参数集合，\\(\lambda\\) 为正则化系数。

Cross Network 和 Deep Network 通过同一个预测目标进行联合训练，使显式特征交叉与隐式特征学习能够相互补充。

## 3. 模型实现（Implementation）

### 3.1 核心代码实现

下面使用 PyTorch 实现原始 DCN 的核心结构，包含 Cross Network、Deep Network 和 Combination Layer。

为了突出模型原理，假设输入特征已经完成 Embedding 编码和拼接，直接使用稠密向量 \\(x_0\\) 作为模型输入。

（1）Cross Network 实现

```
import torchimport torch.nn as nnclass CrossNetwork(nn.Module):    def __init__(self, input_dim, num_layers=3):        super().__init__()        self.weights = nn.ParameterList([            nn.Parameter(torch.empty(input_dim))            for _ in range(num_layers)        ])        self.biases = nn.ParameterList([            nn.Parameter(torch.zeros(input_dim))            for _ in range(num_layers)        ])        for weight in self.weights:            nn.init.normal_(weight, std=0.01)    def forward(self, x):        x0 = x        xl = x        for weight, bias in zip(            self.weights, self.biases        ):            # [B, D] @ [D] -> [B, 1]            interaction = torch.sum(                xl * weight, dim=1, keepdim=True            )            # x_{l+1} = x0 * (xl^T w) + b + xl            xl = x0 * interaction + bias + xl        return xl
```

其中最重要的计算为：

```
xl = x0 * interaction + bias + xl
```

对应原始 DCN 的 Cross Layer 公式：

\\[ x\_{l+1}=x_0(x_l^Tw_l)+b_l+x_l \\]

在整个计算过程中，Cross Network 的输入和输出维度保持不变，均为 `[B, D]`。

（2）DCN 模型定义

```
class DCN(nn.Module):    def __init__(        self,        input_dim,        num_cross_layers=3,        deep_dims=(128, 64)    ):        super().__init__()        # Cross Network        self.cross_network = CrossNetwork(            input_dim,            num_cross_layers        )        # Deep Network        layers = []        prev_dim = input_dim        for hidden_dim in deep_dims:            layers.extend([                nn.Linear(prev_dim, hidden_dim),                nn.ReLU()            ])            prev_dim = hidden_dim        self.deep_network = nn.Sequential(*layers)        # Combination Layer        self.output_layer = nn.Linear(            input_dim + prev_dim, 1        )    def forward(self, x):        cross_output = self.cross_network(x)        deep_output = self.deep_network(x)        combined = torch.cat(            [cross_output, deep_output],            dim=-1        )        logits = self.output_layer(combined)        return logits.squeeze(-1)
```

模型采用并行结构，输入特征分别经过 Cross Network 和 Deep Network，之后将两部分表示进行拼接，最终输出 CTR 预测的 Logits。

这里没有在 `forward` 中直接执行 Sigmoid，主要是为了配合 PyTorch 的 `BCEWithLogitsLoss`，提高损失计算的数值稳定性。

（3）模型前向计算

```
model = DCN(    input_dim=64,    num_cross_layers=3,    deep_dims=(128, 64))x = torch.randn(8, 64)logits = model(x)predictions = torch.sigmoid(logits)print("Logits:", logits.shape)print("Predictions:", predictions.shape)
```

输出张量形状为：

```
Logits: torch.Size([8])
Predictions: torch.Size([8])
```

其中：

- Batch Size 为 8。
- 输入维度为 64。
- Cross Network 包含 3 层。
- Deep Network 包含两个隐藏层，维度分别为 128 和 64。

### 3.2 模型训练

DCN 可以使用标准二分类损失进行端到端训练。

```
criterion = nn.BCEWithLogitsLoss()optimizer = torch.optim.Adam(    model.parameters(),    lr=0.001)# 模拟点击标签labels = torch.randint(    0, 2, (8,)).float()# 前向传播logits = model(x)# 二元交叉熵损失loss = criterion(logits, labels)# 反向传播optimizer.zero_grad()loss.backward()optimizer.step()print("Loss:", loss.item())
```

这里仅展示核心训练流程，实际应用中还需要配合类别特征 Embedding、数值特征归一化、训练集与验证集划分，以及 AUC、LogLoss 等指标评估模型效果。

## 4. 总结（Conclusion）

DCN 通过引入 Cross Network，实现了高效的显式高阶特征交叉，并与 Deep Network 并行学习复杂的非线性关系。相比依赖人工设计交叉特征的方法，DCN 能够自动构建有界阶数的特征组合，同时保持较低的额外参数开销。

DCN 的核心优势在于 Cross Layer 的特殊结构：通过原始输入与当前层表示之间的交叉计算，在不显式枚举全部组合特征的情况下实现高阶交互建模。然而，原始 Cross Network 采用向量参数生成交叉项，表达能力受到一定限制。后续提出的 DCN V2 将 Cross Layer 扩展为矩阵形式，并引入低秩分解与专家混合结构，以进一步增强特征交叉的建模能力。[2]

## 5. 参考文献（References）

[1] Wang R, Fu B, Fu G, Wang M. Deep & Cross Network for Ad Click Predictions. AdKDD, 2017. https\://arxiv.org/abs/1708.05123

[2] Wang R, Shivanna R, Cheng D, et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. 2020. https\://arxiv.org/abs/2008.13535

[3] Cheng H T, Koc L, Harmsen J, et al. Wide & Deep Learning for Recommender Systems. DLRS, 2016. https\://arxiv.org/abs/1606.07792

[4] Google. Ranking with Deep & Cross Networks. Keras Documentation. https\://keras.io/keras_rs/examples/dcn/

[5] PyTorch. PyTorch Documentation. https\://docs.pytorch.org/docs/stable/index.html
