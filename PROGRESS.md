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

### 第 11 课：Transformer Block —— 代码已大幅跃进
学生在别的 AI 学了大量内容，代码从"散装 attention"直接跳到**完整可训练 GPT**（lesson09_attention.py）。已掌握的知识点（学生在代码里写了详细注释，理解到位）：
- ✅ Block 类：attention + MLP + 2×LayerNorm + 2×残差
- ✅ MLP：`mlp_fc(hidden→4×hidden) → GELU → mlp_proj(4×hidden→hidden)`，先扩张4倍再压缩，GELU"去线性化"
- ✅ 6 层 Block 堆叠
- ✅ lm_head（hidden→vocab）+ cross_entropy loss + Adam optimizer + 训练循环
- ✅ 字符级 tokenizer（stoi/itos）
- ✅ x 形状 `[batch, seq, hidden]` 由 idxs 决定（前两维照搬 idxs，第三维来自 wte 定义）
- ✅ `@` = 矩阵乘法 = 一堆点乘（相乘后求和，消掉中间维）
- ✅ scaled attention：除以 √head_dim（不是 √hidden_size），因为 64 项相加数值撑大 √64 倍

### ⚠️ 学生标记"还不懂、先记下来"的点
**LayerNorm 的位置——什么时候要归一化？**
- 老师讲了口诀：**"要变换（attn/mlp/lm_head）→ 先归一化；只搬运（残差/相加）→ 不归一化"**
- 代码里有 3 处 LN：ln1(attn前)、ln2(mlp前)、ln_f(最后输出前)
- Pre-LN 结构（变换之前归一化），现代大模型都用这个
- **学生表示还没完全消化，下次需要再讲/举例巩固**。可以画图对比"加 LN vs 不加 LN"的训练曲线差异

### 下次上课流程
1. 先回顾 LayerNorm 位置这个待消化点（用实验或图示巩固）
2. 加**生成功能**（sample）——代码现在只训练没生成，让模型真正"吐字"，最有成就感的 wow 时刻
3. 接完整莎士比亚数据（但要训练需考虑 GPU，见下方 GPU 踩坑）

### Block 完整结构（学生已实现）
```
x ──→ ln1 ──→ attention ──→ + ──→ x1
 │                           │
 └────────残差───────────────┘
     x1 ──→ ln2 ──→ MLP ──→ + ──→ x2
      │                     │
      └────────残差─────────┘
```

## 🖥️ GPU 配置踩坑（2026-08-04）

机器有 **RTX 5080（16GB，Blackwell sm_120）**，但当前环境用不上 GPU：
- `torch.cuda.is_available()` = False（装的是 CPU 版 torch 2.13.0+cpu）
- **根因**：Python 3.14 + RTX 5080 都是最新，常规源（pytorch.org cu128、清华源）**没有 py3.14 的 GPU 版 torch 包**
- 尝试过：pytorch.org cu128（国内下载极慢卡住）、清华源 cu128（无 py3.14 包）

### 后续要上 GPU 的方案（等需要训练大数据时再做）
1. **PyTorch nightly**（支持新架构/新 Python 更早）：`pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
2. **降级 Python 到 3.12/3.13**（最稳妥，新建 venv）
3. **云 GPU**（Colab/Kaggle 免费 T4）

### 当前策略
- **先用 CPU 继续学知识点**（attention/embedding/block 等概念，CPU 跑小数据完全够）
- 训练数据用一句话（60字符）验证代码能跑通即可，loss 秒降是过拟合（数据太小，正常）
- 等要上完整莎士比亚 100万字符训练时，再花时间配 GPU

## 📝 第 11 课重大进展（2026-08-04）

学生在别的 AI 学了大量内容，代码从"散装 attention"跃进到**完整可训练 GPT**：
- ✅ Block 类（含 attention + MLP + 2×LayerNorm + 2×残差）
- ✅ MLP：`mlp_fc(768→3072) → GELU → mlp_proj(3072→768)`（先扩张4倍再压缩回来）
- ✅ 6 层 Block 堆叠
- ✅ lm_head（hidden→vocab）+ cross_entropy loss
- ✅ Adam optimizer + 训练循环
- ✅ 字符级 tokenizer（stoi/itos）

### 修复的 bug
1. `torch.tril(..., -1)` → 去掉 `-1`（对角线必须保留，token 要能看自己）
2. 训练循环缩进：`ln_f/lm_head/loss/optimizer` 退出 `for b in blocks`（原来每个 block 都重复算一次）
3. **reshape 顺序**：`q.reshape(bs, head_num, tokens_num, -1)` ❌ → `q.reshape(bs, tokens_num, head_num, -1)` ✅
   - head 维度必须在 token 维度**之后**，否则 1024 维切错位，数据全乱（报错还很误导）

### 验证结果
- 代码能跑通，loss 从 3.52 → 0.0001（100步）
- ⚠️ loss 降太快 = 过拟合（数据就一句话60字符，模型死背），不是 bug
- 换完整莎士比亚数据 loss 会正常降到 1.x

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
