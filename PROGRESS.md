# 学习进度

> **下次上课从这里继续。** 最后更新：2026-08-04

## 📍 当前位置

- **已完成**：第 1-7 课（基础）+ 第 9 课（KQV）+ 第 10 课（Embedding + 位置编码）
- **进行中**：第 9-10 课代码已完成并验证跑通
- **下次上课**：第 11 课——把散装代码包成 `nn.Module` 类，然后进 Transformer Block（MLP/LN/残差）

## 📊 进度仪表盘

```
nanoGPT 实战主线：2 / 7 里程碑（M1 KQV ✅ + M2 Embedding ✅）
[██░░░░░] 28%

整体课程：第 10 / 15 课完成，下次进第 11 课
```

## ✅ 已完成课程

### 基础课（模块一 + 模块二）
1. ✅ 线性回归——模型 = 找规律
2. ✅ 梯度下降——蒙着眼下山
3. ✅ 逻辑回归——把数字压成概率
4. ✅ 神经网络——当直线不够用
5. ✅ 激活函数——Sigmoid压扁信息，ReLU不压，GELU留一点
6. ✅ 多层网络——每层做一次综合，层数越深理解越复杂
7. ✅ 反向传播——客户投诉追责，从后往前算每层该改多少

（第 8 课手写数字识别实战进行中，PyTorch 基础已掌握：nn.Module、训练循环、optimizer，跑通 97.4%）

### nanoGPT 实战主线
9. ✅ **KQV / Self-Attention**（2026-08-02，2026-08-04 补 mask）
   - 代码：`lesson09_attention.py`（学生从 0 手写）
   - 核心概念：Query/Key/Value 三顶帽子、切头、scaled attention、causal mask、softmax、拼头、W_o 输出投影
   - 关键产出：能跑通完整 self-attention 前向，与官方 model.py 逻辑一致
   - 2026-08-04 补：causal mask（下三角挡未来），从 BERT 式升级成 GPT 式

10. ✅ **Embedding + 位置编码**（2026-08-04）
    - 代码：`lesson09_attention.py`（学生把 embedding 接进同一文件，未单独建 lesson10）
    - 核心概念：token embedding（字符查表）、position embedding（位置查表）、相加融合
    - 关键点：第 9 课把 embedding 当黑盒，这里拆开；彻底参数化（零硬编码维度，抽出 tokens_size/heads_num/batches_num/tokens_num）
    - 关键产出：输入从随机数变成真实 embedding，跑通 [1,5,768] 输出

## 🔜 下次上课内容

### 第 11 课进行中：Transformer Block 三零件
已学 2/3 零件（代码已加进 lesson09_attention.py）：
- ✅ **残差连接**：`output = output + x_base`（输入直接加输出，给梯度抄近路）
  - 踩坑：残差加的是**原始输入**，不是 LayerNorm 后的值。学生用 `x_base = x` 保存原始值（可行）；官方一行写法 `x = x + attn(ln(x))` 更不易错
- ✅ **LayerNorm**：`ln1 = nn.LayerNorm(768)`，放在 attention **之前**（Pre-LN），对 768 维 token 向量做
  - 踩坑：维度必须对（768 对 768，不能对 score 的 5）；不能覆盖原始 x，要用新变量 `x_norm` 或官方写法
- 🔜 **MLP**：第三个零件，还没讲。讲完 Block 就齐了

### 下次上课流程
1. 讲 MLP（多层感知机：768→3072→768，先扩张再压缩 + GELU 激活）
2. 把 MLP 加进代码（attention 之后，带自己的残差 `x = x + mlp(ln2(x))`）
3.（可选）把散装代码包成 `nn.Module` 类，为堆叠多层做准备
4. 然后进第 12 课：堆叠多层 Block = 完整 GPT

### Block 完整结构（目标）
```
x ──→ ln1 ──→ attention ──→ + ──→ x1
 │                           │
 └────────残差───────────────┘
     x1 ──→ ln2 ──→ MLP ──→ + ──→ x2
      │                     │
      └────────残差─────────┘
```

## 🧠 待回收思考题

### 第 9 课思考题 9-1（未答）
> 如果把你写的 attention 用在"翻译"任务上（英译中），加 causal mask 合适吗？为什么？
> 提示：翻译时 Encoder 要看整句英文才能翻译，但 Decoder 生成中文时只能一个字一个字往后吐。哪些该加 mask，哪些不该加？

不用写代码，用脑想。

## 📝 第 9 课详细回顾（供复习）

### 学到的核心
1. **Q/K/V 同源**（self 的含义）：同一个输入 x，经三个权重矩阵 Wq/Wk/Wv 变出来——"同一个人戴三顶帽子"
2. **切头**：768 维劈成 12 个 64 维头，各自独立算 attention。`reshape + transpose`
3. **算分**：`q @ kᵀ` → `[1,12,5,5]` 谁关注谁
4. **Scaled attention**：`score / √64` 防止 softmax 饱和（维度高→分数大→一个独大→梯度消失）
5. **Softmax**：不只变比例，还用 `e^x` 放大差距、处理负数
6. **混合 v**：`score @ v` → 融合上下文的新值
7. **拼头**：`transpose + reshape` 把 12 个头合回 `[1,5,768]`
8. **W_o 输出投影**：12 个头结论做总合成

### 学生代码 vs 官方对照
| 步骤 | 学生代码 | 官方 model.py |
|------|---------|--------------|
| 切头 | `reshape + transpose` | `view + transpose` |
| 算分 | `q @ k`（k 已转置） | `q @ k.transpose(-2,-1)` |
| 缩放 | `score / (768/12)**0.5` | `* (1.0/math.sqrt(k.size(-1)))` |
| 归一化 | `torch.softmax(dim=-1)` | `F.softmax(dim=-1)` |
| 混合 | `score @ v` | `att @ v` |
| 拼头 | `transpose + reshape` | `transpose + contiguous + view` |
| 输出投影 | `w_o(output)` | `self.c_proj(y)` |

逻辑完全一致 ✅

### 踩坑预警（已踩/已提醒）
- `import pytorch` ❌ → 应为 `import torch`（包名）
- 张量维度：`[batch, seq, dim]` 三维，`nn.Linear` 只看最后一维
- 768 必须能被头数整除（768÷12=64）
- `transpose` 后内存不连续，`view` 会报错，加 `.contiguous()`
- causal mask 必须在 softmax **之前**加（否则比例不对）
- `tokens_size` 和 `tokens_num` 命名太像易混（官方用 `n_embd`/`T` 区分）

## 📝 第 10 课详细回顾（供复习）

### 学到的核心
1. **Token embedding**：每个字符 → 768 维向量，靠 `nn.Embedding(65, 768)` 查表。表里 5 万个数随机初始化、训练学
2. **Position embedding**：每个位置 → 768 维向量，`nn.Embedding(1024, 768)`。解决 Transformer 本身没有位置感、分不清"猫追狗/狗追猫"的问题
3. **相加而非拼接**：内容向量 + 位置向量 直接相加，不增维度（拼接会 768→1536，后面全得改）
4. **参数化**：把所有硬编码维度抽成变量（tokens_size/heads_num/batches_num/tokens_num），换超参只改开头几行

### 为什么需要位置编码（关键理解）
attention 计算 `score[i][j] = q[i]·k[j]` 只用内容、不用位置 → 同样字符不同顺序算出 score 一样 → 必须额外注入位置信息

## 🗺️ 完整里程碑路线图（与 Haiku 讨论定稿）

核心设计：理论课与 nanoGPT 实战交织，**增量构建**——每课从 0 写 20-30 行，下课必须能跑出结果。乱码/挫败期压成同课 30 秒演示道具，不让学生抱乱码过夜。

- **M0 启动**（插班，第 8 课后）：clone + 数据准备 → ✅ 完成（2026-08-02，shakespeare_char 数据已生成）
- **M1**（第 9 课）KQV：→ ✅ 完成
- **M2**（第 10 课）Embedding + 位置编码 + 数据加载
- 第 11 课 RNN 痛点：恢复！讲"传话游戏传丢了"动机，为 Attention 铺垫（虽已讲过 attention，但 RNN 动机仍有价值）
- **M3**（第 12 课）Attention 理论深化 + 完整版（含 mask）
- **M4**（第 13 课上）手写完整 Block
- **M5**（第 13 课下）完整 GPT 组装，逐行对照官方 model.py
- **M6**（第 14 课）训练 + 生成：莎士比亚重现
- **M7**（毕业项目）换中文语料

## 📌 备忘

- 环境：见 `ENV.md`
- 教学规范：见 `TEACHING.md`
- 学生偏好：互动式、少公式、多类比；代码要自己写，AI 不代写
- CPU 训练慢，每课控制在 5 分钟内，用小语料
