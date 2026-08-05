# 学习进度

> **下次上课从这里继续。** 最后更新：2026-08-04

## 📍 当前位置

- **已完成**：第 1-7 课（基础）+ 第 9 课（KQV）+ 第 10 课（Embedding + 位置编码）+ 第 11 课（Block + 完整 GPT 类）
- **进行中**：第 11 课代码已跃进到完整可训练 GPT，**接完整莎士比亚数据，GPU 跑通真实训练，loss 降到 ~2.5**
- **下次上课**：加生成功能（sample）——让模型真正吐字，最有成就感的 wow 时刻

## 📊 进度仪表盘

```
nanoGPT 实战主线：4 / 7 里程碑（M1 KQV ✅ + M2 Embedding ✅ + M3 Block ✅ + M4 完整GPT ✅）
[████░░░] 57%

整体课程：第 11 / 15 课完成，下次加 sample 生成
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

### 第 11 课已完成：散装代码 → 完整 GPT 类 + 真实训练 ✅（2026-08-05）
学生在别的 AI 学了大步，把散装代码包成 `class gpt(nn.Module)`，接完整莎士比亚数据，GPU 跑通真实训练：
- ✅ `class gpt(nn.Module)`：wte/wpe/ModuleList(blocks)/ln_f/lm_head 全包进去
- ✅ `forward(idx, targets=None)` 双返回：训练返回 `(logits, loss)`，生成只返回 `logits`
- ✅ `torch.nn.ModuleList([...])` 装 blocks（普通 list 不注册参数，必须 ModuleList）
- ✅ 接完整莎士比亚数据（100万字符），`randint` 随机抽 batch 训练
- ✅ GPT-2 small 配置：12 层 × 1024 hidden × 16 head（~120M 参数）
- ✅ GPU 跑通，100000 步训练，loss 从 4.5 → 2.5（真实训练，非过拟合）

### 修复的 bug（学生自己改的）
1. `super.__init__()` → `super().__init__()`（少了括号，必崩）
2. `ln_f` 漏 `.to(device)` → 补上
3. `open("F:\...\input.txt")` → `open(r"F:\...\input.txt")`（Windows 路径反斜杠转义，加 r 前缀）

### LayerNorm 位置——学生已通过自己写代码"落地"理解 ✅
上次标记"还不懂"的点，这次写 gpt 类时其实已经写对了：
- 3 处 LN 全在"变换之前"：ln1(attn前)、ln2(mlp前)、ln_f(lm_head前)
- 2 处残差相加没归一化
- 口诀"要变换→先归一化；只搬运→不归一化"已在代码中体现，下次可简单确认是否真懂

### 下次上课流程
1. 加**生成功能**（sample）——核心 3 步：取 logits 最后一行 → `multinomial` 采样 → 接回输入循环
2. 让模型从 `"\n"` 开头续写几百字，看莎士比亚风格输出（wow 时刻）
3.（可选）加 temperature / top-k 采样控制生成质量

### Block 完整结构（学生已实现）
```
x ──→ ln1 ──→ attention ──→ + ──→ x1
 │                           │
 └────────残差───────────────┘
     x1 ──→ ln2 ──→ MLP ──→ + ──→ x2
      │                     │
      └────────残差─────────┘
```

## 🖥️ GPU 配置（✅ 2026-08-05 已搞定）

机器：**RTX 5080（16GB，Blackwell sm_120）**。

**当前状态：GPU 可用** ✅
- torch 2.11.0+cu128（cu128 支持 Blackwell sm_120）
- `torch.cuda.is_available()` = True
- `torch.cuda.get_device_name(0)` = NVIDIA GeForce RTX 5080
- 实际 GPU 矩阵乘法跑通

**踩坑历史**（留档备忘）：
- 原以为 py3.14 + Blackwell 没有合适的 GPU 包，试过 pytorch.org cu128（国内下载极慢卡住）、清华源 cu128（无 py3.14 包）都失败
- 最终通过后台任务装上了 torch 2.11.0+cu128 cp314 包

### 当前策略
- GPU 已就绪，**lesson09_attention.py 已加 `.to(device)` 全套适配**（device='cuda'，模块/张量/mask 都搬 GPU）
- 关键坑：block 里 `mask = torch.ones(..., device=x.device)` 必须带 device 参数，否则 score 在 cuda、mask 在 cpu 报设备不匹配
- 训练数据已换成完整莎士比亚（100万字符），GPU 真正发力，100000 步训练 loss 从 4.5 → 2.5

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

## 📝 第 11 课重大进展（2026-08-05）：完整 GPT 类 + 真实训练

学生继续在别的 AI 学，把散装代码包成 `class gpt(nn.Module)`，接完整莎士比亚数据，GPU 跑通真实训练：
- ✅ `class gpt(nn.Module)`：wte/wpe/ModuleList(blocks)/ln_f/lm_head 全包进去
- ✅ `forward(idx, targets=None)` 双返回（训练返 `(logits,loss)`，生成只返 `logits`）
- ✅ `torch.nn.ModuleList([...])` 装 blocks（关键：普通 list 不注册参数）
- ✅ 接完整莎士比亚 100万字符，`randint` 随机抽 64 字符 batch
- ✅ GPT-2 small 配置：12 层 × 1024 hidden × 16 head（~120M 参数）
- ✅ GPU 跑通，100000 步训练，loss 从 4.5 → 2.5

### 修复的 bug（学生自己改的）
1. `super.__init__()` → `super().__init__()`（少括号必崩）
2. `ln_f` 漏 `.to(device)` → 补上
3. `open("F:\...")` → `open(r"F:\...")`（Windows 路径反斜杠转义坑，加 r 前缀）

### GPU 适配（老师帮加的）
- `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- 所有模块 `.to(device)`，输入张量 `.to(device)`
- **关键坑**：block 里 `mask = torch.ones(..., device=x.device)` 必须带 device，否则 GPU 上 score(cuda) 与 mask(cpu) 设备不匹配报错

### 真实训练 loss 解读
- loss 从 4.5（ln(65)≈4.17，乱猜）→ 2.5（学到单词边界 + 常见词）
- loss 在 2.3~2.8 之间波动是正常的（每步随机 batch 难易不同）
- 对照官方 nanoGPT：2.5≈单词级，1.5≈语法级，1.0≈GPT-2 small 可读莎士比亚
- **还没加生成功能**，下一步加 sample 让模型吐字

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
