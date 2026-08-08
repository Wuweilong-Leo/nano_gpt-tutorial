# 学习进度

> **下次上课从这里继续。** 最后更新：2026-08-09

## 📍 当前位置

- **已完成**：第 1-7 课（基础）+ 第 9-11 课（KQV/Embedding/Block/完整 GPT/训练生成）= 里程碑 M0-M5
- **进行中**：M6 生成质量+评估（KV cache ✅ + temperature 对照实验 ✅ + top-k 手写 ✅，剩 top-p/perplexity/beam search）
- **下次上课**：**M6 剩余**——top-p 实现 + perplexity 困惑度 + beam search + 训练工程细节（AdamW/warmup/CE vs MSE/LN vs BN）+ 欠着的一个思考题（索引为什么必须跟着走）

## 📊 进度仪表盘

```
端到端主线：5 / 15 里程碑完成（M0-M5 ✅）
[███░░░░░░░░░] 33%
三条目标线：A 通晓大模型 / B 通晓 agent / C 通晓 AI infra（2026-08-08 新增）

下次：M6 剩余（top-p/perplexity/beam search/训练工程细节 + infra：PagedAttention/gradient checkpointing/profiling）— KV cache ✅ + temperature ✅ + top-k ✅ 已搞定
最终毕业：M15 完整 agent + 框架对照
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
1. 看 sample 生成质量（loss 2.5 大概是"单词级"，能看出零星真词但不成句）
2. 加 **temperature / top-k** 控制采样——高温多样但乱、低温保守但稳
3.（可选）**KV cache** 优化推理——现在每步重算全长，缓存 K/V 后只算新 token，大幅加速
4. loss 继续降（多训练或调超参），降到 1.x 才会出现可读莎士比亚

### 📝 第 11 课补完：命名规范化 + sample 生成（2026-08-07）
**命名规范（PEP 8，老师帮改）**：
- `class block` → `Block`、`class gpt` → `GPT`（类名大驼峰，硬规范）
- `head_num` → `n_head`（英文 `n_东西` 惯例）
- `batches_num/tokens_num` → `B/T`（贴近官方 nanoGPT 短名）
- 删 GPU 诊断打印，只留 `using device` 一行

**学生新加的高级特性**（在别的 AI 学的）：
- ✅ `Dropout(0.1)` 两处（attention 后 + MLP 后，防过拟合）
- ✅ **权重共享** `lm_head.weight = wte.weight`（weight tying，输入输出共用表，减参数+效果更好）
- ✅ train/val 9:1 划分 + 每步算 val_loss
- ✅ `CosineAnnealingLR` 学习率退火 + `clip_grad_norm_(1.0)` 梯度裁剪
- ✅ **sample 生成循环**：从 `"F"` 开头，取最后 logits → softmax → `multinomial` 抽签 → cat 接回 → 循环 100 步

**修复的 bug（老师指出 + 学生改）**：
1. `loss.items()` → `loss.item()`（多了 s 会崩，tensor 没 items 方法）
2. `train_loss` 后漏冒号
3. 推理缺 `model.eval() + torch.no_grad()`（eval 关 dropout、no_grad 不建计算图，否则 100 步累积 OOM）
4. `block_size=64` 但生成 100 字符 → wpe 越界，改成 128

**关键概念教学——为什么推理要 `torch.no_grad()`**：
- 类比：训练=备课要记笔记（建计算图供 backward），推理=讲课不记笔记
- `model.eval()` 管**层行为**（dropout 关闭），`torch.no_grad()` 管**计算图**（不存激活）——两件独立的事，推理都要
- 学生反应"16GB 显存还好" → 老师提醒：现在数据小扛得住是运气，batch 加大/生成长文/换大模型都会爆；且 no_grad 能快 20-50%（推理行业标准写法，nanoGPT sample.py 也有）

### 📝 KV cache 提前自学完成（2026-08-08，M6 子项）
学生**提前自学**实现了 KV cache 推理加速（本来是 M6 的内容，还没上课就动手了）。在 `lesson09_attention.py` 上改造：
- ✅ `Block.forward(x, kv_cache=None)`：接 cache 参数，cat 拼历史 K/V，返回 `(output, new_kv_cache)`
- ✅ `GPT.forward(idx, targets=None, kv_cache=None, pos_val=None)`：用 list 收集每层 cache，推理返回 `(logits, kv_caches)`
- ✅ 推理循环改成每次只喂新 token（第一步喂"F"，之后喂 `next_char`），cache 一步步传下去
- ✅ `pos_val` 参数解决位置编码偏移（推理第 step 步用位置 step，不再每个字都套位置 0）
- ✅ mask 用 `if kv_cache is None` 包住——推理第二步只喂 1 个新 token 没未来可挡，不需要 mask

**踩的坑（4 个，学生自己改的 + 老师指点的）**：
1. cache 没传下去：`logits, _ = model(...)` 把第二个返回值丢了 → 改成 `logits, new_kv_cache = model(...)` 再赋回 `kv_cache`
2. 元组没拆：`torch.cat([kv_cache, k])` 里 kv_cache 是 `(k_old, v_old)` 元组 → 拆成 `kv_cache[0]`/`kv_cache[1]`
3. k 存进 cache 前转置过：`new_kv_cache=(k,v)` 写在 `k.transpose(-1,-2)` 之后，导致 k/v 形状不一致下次拼崩 → 挪到 transpose **之前**，且要**无条件**执行（不能只在 `if kv_cache is not None` 里，否则训练时 `UnboundLocalError`）
4. pos 设备不匹配：`torch.tensor([[pos_val]])` 默认 CPU，wpe 在 GPU → 加 `.to(DEVICE)`（老坑再现：新张量喂 GPU 模型都得搬设备）

**关键理解——KV cache 为啥能加速**：
- 不用 cache：生成第 N 个字时，前 N-1 个字的 K/V 全部重算一遍（O(N²) 累加）
- 用 cache：前 N-1 个字的 K/V 存着，每步只算新 token 的 K/V 拼上去 → 推理从 O(N²) 降到 O(N)
- **cache 只加速、不改结果**：同样的随机种子，用不用 cache 生成结果应一致（验证 cache 实现对不对的判据）

**遗留定时炸弹**（现在不炸）：`wpe=Embedding(block_size=128)`，生成超过 128 字时 `pos_val=128` 越界崩。现在跑 100 步没事，以后想生成长文再改（调大 block_size 或加截断逻辑）。

### 📝 M6 第 1 课：temperature + top-k（2026-08-09）
**temperature 三温度对照实验**（0.5 / 1.0 / 1.5，各生成 100 字符）：
- 改造：`temperatures = [0.5, 1.0, 1.5]` + `for i in range(3)` 一次运行三连测，`logits = logits / temperatures[i]`（softmax **之前**除——位置正确）
- ✅ **控制变量教训（第一次实验作废）**：最初每次运行脚本都重新训练，三个温度用的是三个不同大脑，对比无效 → 改成一次运行内三连测，同一模型共用
- ✅ **又一个 bug 被学生自己钓出来**：`model_str` 初始化在 `for i` 外面 → 第二轮接着第一轮生成的 101 字符继续写 → 挪进循环内，每轮从 `F` 白纸开始
- **训练步数 100 → 5000**：val_loss 9.8 → 1.5 以下，模型从"不会说话"到"会排剧本格式"（没有好 brain，采样实验全是噪声——巧妇难为无米之炊）

**三组输出体检**（对照实验成功关键）：
| 温度 | 性格 | 诊断 |
|---|---|---|
| 0.5 | `the/and/e` 复读机 | 低温=平庸但保险，模型坚守"最常见"而非"最妙" |
| 1.0 | 角色名+冒号+对白，标准莎翁剧本格式 | 默认推荐 0.7~1.0 |
| 1.5 | 大写乱码 `FOLKIULG` 满天飞 | 高温拉平差距 → 稀有字符（大写）出场率被抬升 |

**高温机理（学生自己想的，方向对）**：温差减小 → 分布变平 → 低概率词概率升高。老师补全：softmax 是 e 的指数不是平方，除温度发生在它前面；算例：logits=[3,1]，B 概率 T=1.0 时 12% → T=2.0 时 27%（冷门翻倍，不改排名只改冷门逆袭率）。

**循环陷阱**：自回归生成 = 自己喂自己，一旦概率最高的 token 是无意义的（如 E），一步错步步错锁死循环——真实产品要 top-k/top-p 配合。

**top-k 手写完成**（学生自己写的五步）：
```python
v, idx = torch.topk(probs, TOP_K)          # 解包 values+indices
probs = torch.zeros_like(probs)            # 全 0 格（不能复用旧张量，否则漏网）
probs.scatter_(1, idx, v)                  # 邮差送信：按 idx 地址投 v 进 65 格
probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新归一化（幸存者之和≠1）
next_char = torch.multinomial(probs, 1)    # 全尺寸抽样 → 直接拿真 token 索引
```
- `scatter_` 邮差送信类比：idx=地址，v=信，dim=1 往行内按列投
- `dim=-1` = 最后一个维度 = 概率维，张量变 3D/4D 也不用改
- 学生问过：dim=-1 是为啥 / 想让老师代写（**被拒**——手写原则不能破，给了骨架+两个空+debug 仪式）

**待办**：
- [ ] TOP_P = 0.9 还没实现（top-k 先看效果）
- [ ] 思考题欠着：为什么 `multinomial(probs2)` 给真 token 而 `multinomial(v)` 给的是"第几名"（索引必须跟着走）
- [ ] 观察点核对：三组开头一样吗 / 谁通顺 / 谁重复 / 谁乱码（已部分完成）

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

## 🗺️ 完整里程碑路线图（2026-08-07 与 Haiku 二次讨论定稿，2026-08-08 补 AI infra 融合线）

**三条目标线（缺一不可）**：
- **目标 A：通晓大模型** —— M6-M11 全部学完（采样/BPE/SFT/LoRA/RLHF/架构进阶/量化），一个都不能少。即使 LoRA/RLHF/RoPE 这些"对搭 agent 不是硬性必需"，但对"通晓大模型"是必修。
- **目标 B：通晓 agent** —— M12-M15 全部学完（四零件/工具/记忆/ReAct/框架对照）。
- **目标 C：通晓 AI infra** —— 理论全覆盖面试考点（训练/推理/部署），+ 亲手部署过一次推理服务。**不等于"能跑多节点训练"**——分布式训练在 16G 单卡上永远纯理论+图解；infra 真实战就三样：gradient checkpointing、profiling、推理部署（量化→serve→OpenAI兼容→streaming）。
- **毕业项目是 agent（M15），但"通晓"的定义是 A+B+C 全懂，不是"能搭出 agent 就毕业"。**

### 🔧 AI infra 融合策略（2026-08-08 与 Haiku 三次讨论定稿）

**策略 C：主线穿插 + 部署仪式**（不单独开 M_infra，避免枯燥名词课+拖慢进 agent）。
- infra 理论**散进 M6（+0.5课）和 M11（+0.5课）**，作为"已学知识的另一副眼镜"。比如 M6 讲 KV cache（学生已实现朴素版）时顺势戴 infra 眼镜看 10 分钟 PagedAttention。
- infra 实战**集中到"部署仪式"**（过渡仪式 半课→1课，净+0.5）：把 Qwen3-0.6B 部署成 OpenAI 兼容 API，全链路跑一遍。
- **infra 和 agent 闭环**：部署仪式产出的本地 API 直接给 M12-M15 agent 当后端——infra 不是和 agent 抢时间，是 agent 的上游。
- **总课数 +1.5**（M6+0.5 / M11+0.5 / 部署仪式+0.5），可接受。时间紧砍 speculative decoding 和 chunked prefill（面试偏冷）。
- **infra 真实战清单（16G 能跑）**：gradient checkpointing（加到 nanoGPT 对比显存）、profiling（torch profiler）、推理部署（GGUF量化→serve→FastAPI→streaming）。其余（DDP/ZeRO/TP/NCCL/集群/通信）纯理论+图解。
- **环境风险**：RTX 5080 Blackwell sm_120 很新，vLLM 在 Windows 原生支持差。部署仪式保底用 Ollama（封装 llama.cpp，Windows 友好，自带 OpenAI 兼容 API），到时再定。

### 🔧 infra 项目配套全景（2026-08-08 与 Haiku 四次审视定稿）

**两个项目的 infra 角色**：
- **项目A（nanoGPT）= 训练 infra 实验田** + 分布式 toy 实验场（学生能改训练代码，每行都懂）
- **项目B（Qwen3-0.6B）= 推理 infra 实验田** + agent 的 hello-world 闭环 demo（现成模型，重点在部署）

**配套全景表**（Haiku 审视后修订，原方案 7/10 → 9/10）：

| infra 知识点 | 配哪个项目 | 怎么配 | 实战度 |
|---|---|---|---|
| gradient checkpointing | A（nanoGPT） | Block 上加 `torch.utils.checkpoint`，对比开/关显存 | ✅实战 |
| **mixed precision AMP** | A | fp32 / fp16+GradScaler / bf16 三组对比显存+loss 稳定性 | ✅**新增必加** |
| **torch.compile** | A | `torch.compile(model)` 对比 step 时间，看 graph break | ✅**新增必加** |
| profiling | A | torch profiler 跑训练，看算子耗时+显存追踪 | ✅实战 |
| **KV cache 内存可视化** | A | 不同 seq_len 下量 KV cache 显存画曲线，给 PagedAttention 埋伏笔 | ✅**新增轻量** |
| **Ring AllReduce toy** | A 旁支 | numpy 单进程模拟 N 节点环形通信，~100 行，验证 O(2n) 通信量 | ✅**toy（原纯理论升级）** |
| **ZeRO 三阶段手算** | A 旁支 | 1B 模型 ZeRO-1/2/3 分片纸笔题（每卡存什么算清楚） | ✅**手算（原纯理论升级）** |
| **张量并行 TP-by-hand** | A 旁支 | nanoGPT MLP 线性层手切列/行并行，验证数值不变，讲 Megatron AllReduce 点 | ✅**toy（原纯理论升级）** |
| **DDP 单进程真代码** | A 旁支 | `world_size=1` 跑真 DDP 样板（init_process_group/DistributedSampler/DDP包装），会写不跑收益 | ✅**真代码（原纯理论升级）** |
| 推理部署全家桶 | B（Qwen3） | GGUF量化→serve→FastAPI→OpenAI兼容→streaming，**多量化级别** | ✅实战 |
| PagedAttention | B | vLLM/Ollama 跑 + **必读 SOSP'23 论文核心图/伪代码**（block table+分页KV） | 🟡看+读 |
| continuous batching | B | 跑混合长度 prompt benchmark，对比 nanoGPT 静态 batch | 🟡看+测 |
| prefix caching | B | vLLM flag 开关测 TTFT 对比 | 🟡看+测 |
| speculative decoding | B | vLLM/Ollama flag + 读 draft/verify 论文摘要 | 🟡看+读 |
| GPTQ / AWQ | B 理论 | Windows 折腾用 GGUF 替代；补读 GPTQ Hessian + AWQ 激活感知权重思路 | ❌理论+阅读 |
| k8s/Slurm/RDMA/InfiniBand | 无 | 集群级扫盲，知道名词 | ❌扫盲 |
| **agent 后端策略** | B→M12 | q4 做量化实验；**q8/fp16 做 agent hello-world 后端**；M12-15 复杂 agent 主后端用智谱 GLM 云 | 🛡️**防陷阱** |
| **HF Spaces 部署（可选）** | B stretch | 推到 HF Spaces 拿公开 endpoint，简历加分 | ✅可选 stretch |

**关键修订（Haiku 指出的三处加码）**：
1. **项目A 漏了 AMP + torch.compile**——两个必加训练 infra。AMP 面试高频考"fp16 为啥要 loss scaling 而 bf16 不要"；compile 对应静态图编译概念，原方案完全漏了。
2. **纯理论那批放弃太早**——Ring AllReduce/ZeRO/TP/DDP 都能 toy 化（numpy/手算/手切/单进程代码），从"我知道"升级到"我手写过"，面试拉开档次。术语澄清：activation checkpointing = gradient checkpointing（同一东西两名，文档写清楚防面试露怯）。
3. **agent 闭环有质量陷阱**——0.6B int4 太弱 function calling 会翻车（漏字段/tool名错），学生分不清是 agent bug 还是模型弱。救法：**双精度部署 + 云主本地辅**——q4 做量化实验，q8/fp16 做 agent hello-world 后端，M12-15 复杂 agent 主后端用智谱 GLM 云，本地 API 只在 M12 第一节课做一次闭环 demo 不污染主课程。

**不开第三项目**（Haiku 同意老师判断）：DDP 租云不必要（toy 实现已够面试）；HF Spaces 是项目B 的可选 stretch 不算新项目。

**核心主线一句话**：你不是在学一堆散件，你是在造一个 agent——只是先把每个零件搞懂。tokenizer=agent 听懂人话的前提，KV cache=agent 响应快的前提，SFT=agent 听指令的前提，采样=agent 多样性的前提。但注意：这条主线是**学习动机**，不是"砍掉用不到的知识"——通晓大模型的所有知识点都要学，主线只是帮你理解每个知识点最终怎么在 agent 里发挥作用。

**设计原则**：理论课与实战交织，**增量构建**——每课从 0 写 20-30 行，下课必须能跑出结果。架构进阶不搞名词轰炸（只 RoPE 深讲，其余压成"行业地图"扫盲）。agent 阶段**先手写禁框架**，框架对照放最后。

### 已完成（M0-M5）
- **M0** 启动：clone + 数据准备 → ✅（2026-08-02）
- **M1** KQV / Self-Attention → ✅（2026-08-02，08-04 补 mask）
- **M2** Embedding + 位置编码 → ✅（2026-08-04）
- **M3** Block（attention+MLP+2LN+2残差）→ ✅（2026-08-04，学生从别的 AI 学完写代码）
- **M4** 完整 GPT 类（`class GPT`，ModuleList，forward 双返回）→ ✅（2026-08-05）
- **M5** 训练+生成 → ✅（2026-08-07，10万步 GPU 训练 loss 613→2.3，sample 生成跑通）

### 待完成（M6-M15）

**大模型补全段（M6-M11）**：

| 里程碑 | 内容 | 课数 | 类型 |
|--------|------|------|------|
| **M6** 生成质量+评估+训练工程 | temperature/top-k/top-p 采样 + perplexity 困惑度 + KV cache 推理加速（已提前自学✅） + **beam search 对比** + **Adam vs AdamW** + **warmup+cosine 调度** + **cross-entropy 为啥用 CE 不用 MSE** + **梯度裁剪回顾** + 🔧infra：**PagedAttention**（vLLM核心，必读SOSP'23论文）+ **continuous batching**（对比nanoGPT静态batch）+ **gradient checkpointing（实战！加到nanoGPT对比显存）** + **mixed precision AMP（实战！fp32/fp16+scaler/bf16三组对比）** + **torch.compile（实战！对比step时间）** + **profiling（实战！torch profiler）** + **KV cache 内存可视化（轻量）** + **speculative decoding** | 3 课 | 理论+改代码 |
| **M7** BPE+中文语料 | BPE 子词分词原理 → 训练 BPE → 用新 tokenizer 重训 GPT（词表 65→BPE） + **BBPE/tokenizer 工程面试题** | 2 课 | 实战代码 |
| **M8** SFT 微调 | 指令数据格式 + SFT 训练循环（主线用 Qwen3-0.6B，详见"项目B技术选型"小节） + **预训练 vs 微调边界** | 1 课 | 实战代码 |
| **M9** LoRA | LoRA/PEFT 原理 + Qwen3-0.6B LoRA 微调（transformers+peft+trl 从0写） + **QLoRA/全参微调对比** + 支线：给自己的 GPT 加小适配头 | 1 课 | 实战代码 |
| **M10** 对齐理论 | reward model → RLHF 三阶段 → DPO + **PPO 概念**（纯讲+图解，不写代码，全课最难） | 1 课 | 纯理论 |
| **M11** 行业地图+训练系统 | RoPE 深讲 + GQA/Flash/MoE/长上下文扫盲 + 量化 int8/int4 体验（bitsandbytes 0.50.0 NF4，诚实标注小模型量化学机制非提速） + **scaling law/Chinchilla** + **涌现能力** + **分布式训练（DDP/DP/ZeRO三阶段加深/张量并行TP/NCCL Ring AllReduce）** + **混合精度 amp** + **Attention O(n²) 复杂度** + **评测 benchmark（MMLU/HumanEval）** + 🔧infra toy 实现：**Ring AllReduce**（numpy单进程模拟）+ **ZeRO 三阶段手算**（1B模型分片纸笔题）+ **TP-by-hand**（nanoGPT MLP手切列/行并行）+ **DDP 单进程真代码**（world_size=1样板）+ **GPTQ/AWQ**（对比朴素量化，读算法思路）+ **prefix caching / chunked prefill** | 2.5 课 | 理论+小实验+toy |
| **🚀 部署仪式（原过渡仪式升级）** | 把 **Qwen3-0.6B 部署成 OpenAI 兼容 API**，**双精度部署**：q4 做量化实验（显存1.2G→几百M对比）+ q8/fp16 做 agent 后端 → vLLM/Ollama serve（看 PagedAttention 实跑）→ FastAPI 包一层 `/v1/chat/completions` → 流式输出 streaming（SSE打字机）→ **三方对比：手写GPT / 本地API / 智谱云API**。**agent 后端策略**：q4 量化实验，q8/fp16 做 M12 hello-world 闭环 demo，M12-15 复杂 agent 主后端用智谱 GLM 云（防 0.6B int4 function calling 翻车）。可选 stretch：部署到 HF Spaces 拿公开 endpoint | 1 课 | 实战部署 |

**agent 段（M12-M15，拆成独立里程碑，每个核心概念一个 M）**：

| 里程碑 | agent 知识点 | 课数 | 类型 |
|--------|-------------|------|------|
| **M12** agent 全貌 + 第一个循环 | agent 四零件（大脑/工具/记忆/规划）+ 手写最小 agent 循环（智谱 GLM-4.7-Flash + openai SDK + 1 个工具 + 循环），先用 API 当大脑跑通"问→调工具→答" | 1 课 | 理论+实战 |
| **M13** 工具调用 + 记忆 | prompt 工程（让 LLM 输出结构化指令）+ 工具调用解析（decode 后 parse）+ 短时记忆（对话历史塞 context window）+ 多轮对话 agent + **RAG**（bge-base-zh-v1.5 + faiss-cpu + pypdf） | 1-2 课 | 实战代码 |
| **M14** ReAct + 多工具路由 | ReAct 模式（Thought→Action→Observation 循环）+ 多工具选择（路由）+ 循环终止条件 + 错误恢复 | 1-2 课 | 实战代码 |
| **M15（毕业）** 完整 agent + 框架对照 | 整合成完整 agent（查词典+多轮对话+多工具）+ 对照轻量框架（看 langchain/官方 SDK 怎么封装同样的事，理解"框架=帮你省代码"） | 2 课 | 实战+对照 |

### 关键设计决策

1. **架构进阶压成一课**：RoPE/Flash/MoE/GQA 不全讲——只 RoPE 深讲（有学生现在的绝对位置编码做锚点），其余"知道存在+为什么这么做"即可，避免名词轰炸。
2. **BPE + 中文语料合并**：原"换中文语料"被吸收进 M7——BPE 训练用中文语料当素材，词表从 65 升级到 BPE，重训 GPT。
3. **SFT/LoRA 用现成小模型**：学生 GPT 太小（120M、字符级）SFT 看不到效果。M8/M9 主线用 **Qwen3-0.6B**（16G GPU LoRA 峰值才 ~5G 余量极大，详见"项目B技术选型"小节）。**支线**：用 LoRA 思想给自己的 GPT 加适配头，让"我的模型"主线不断。
4. **过渡仪式承上启下**：进 agent 前用自己 GPT vs API 对比生成，教学话术——"你的模型让你懂了 LLM 大脑怎么工作，这是纯调包的人没有的底气。现在做 agent 要能听指令的大脑，换用更强的真大脑。但你懂底层，你是'懂发动机的司机'。"
5. **agent 拆成 4 个独立 M**：不把 agent 糊成一个里程碑 2-3 课塞完——M12 全貌+最小循环、M13 工具+记忆、M14 ReAct+多工具、M15 整合+框架对照。每个核心概念单独成课，跟 M1-M5 一个节奏。
6. **agent 先手写禁框架**：网上搜 agent 会被 langchain 淹没。M12-M14 全手写，禁框架。框架对照只放 M15 最后一课，理解"框架=帮你省代码"。
7. **API 选国内能直连的**：**智谱 GLM-4.7-Flash**（真·免费、function calling 完整、OpenAI 兼容、手机号注册），用 openai SDK 一套代码改 3 个字符串就能切 Qwen/Kimi/DeepSeek。为啥不首选 deepseek：function calling 只在 chat 档有、reasoner 档不支持，且偶发非法 JSON 要兜底。

### agent 知识点 ↔ 大模型知识对应（贯穿 M12-M15）

| agent 知识点 | 哪个 M 学 | 对应已学的大模型知识 | 对接说明 |
|--------------|----------|---------------------|----------|
| 大脑 LLM | M12 | GPT 前向、采样、temperature（M6 学） | 直接复用，换 API |
| 工具调用 | M13 | tokenizer encode/decode（M7 学 BPE） | 让 LLM 输出结构化指令，得懂 token 边界 |
| 记忆 | M13 | context window、位置编码（已学 wpe） | 记忆=塞进 context window，wpe 限制=记忆容量 |
| 规划 / ReAct | M14 | next-token 自回归（已学） | 链式生成=最朴素规划，ReAct 是升级版 |
| 多工具路由 | M14 | softmax 概率选择（已学） | 多工具选择=概率/规则路由 |
| 框架对照 | M15 | 全部已学 | 看框架如何封装你手写的东西 |

### 🎯 面试考点覆盖表（2026-08-07 新增·对齐企业面试）

学生明确要求：知识点要覆盖企业面试。故新增此表，标注每个考点"在哪个 M 学"和"面试重要度"（⭐⭐⭐ 高频必问 / ⭐⭐ 偶问 / ⭐ 知道就行）。

**A. 已学但 M6-M11 要补讲透的细节（面试最爱抠这些"用过但没深想"的点）**：

| 考点 | 现状 | 在哪个 M 补讲 | 重要度 |
|------|------|--------------|--------|
| LayerNorm vs BatchNorm，为啥 Transformer 用 LN 不用 BN | 用了 LN 没对比 | M6 | ⭐⭐⭐ |
| Pre-LN vs Post-LN（原始论文 Post-LN） | 写了 Pre-LN 没对比 | M6 | ⭐⭐⭐ |
| 残差连接为啥能训深网络/防梯度消失 | 用了没讲原理 | M6 | ⭐⭐⭐ |
| MLP 为啥扩张 4×（不是 2×/8×） | 用了没讲 | M6 | ⭐⭐ |
| GELU vs ReLU vs **SwiGLU**（现代模型标配） | 只用了 GELU | M6 | ⭐⭐⭐ |
| cross-entropy 为啥用 CE 不用 MSE | 用了没讲 | M6 | ⭐⭐⭐ |
| Adam vs AdamW（大模型/LoRA 标配 AdamW） | 只用了 Adam | M6 | ⭐⭐⭐ |
| 权重共享 weight tying 的原理 | 已用 | M6 复习 | ⭐⭐ |

**B. 新补进路线图的面试考点（之前漏了）**：

| 考点 | 在哪个 M 学 | 重要度 | 类型 |
|------|------------|--------|------|
| beam search 解码 | M6 | ⭐⭐ | 理论+对比采样 |
| warmup + cosine 学习率调度 | M6 | ⭐⭐⭐ | 理论+改代码 |
| scaling law / Chinchilla 定律 | M11 | ⭐⭐⭐ | 纯理论 |
| 涌现能力 emergent abilities | M11 | ⭐⭐ | 纯理论 |
| 分布式训练 DDP/DP/ZeRO | M11 | ⭐⭐⭐ | 纯理论 |
| 混合精度训练 amp | M11 | ⭐⭐ | 理论+小实验 |
| Attention O(n²) 复杂度 | M11 | ⭐⭐⭐ | 理论 |
| 评测 benchmark（MMLU/HumanEval/Perplexity） | M11 | ⭐⭐ | 理论 |
| QLoRA / 全参微调对比 | M9 | ⭐⭐ | 理论 |
| PPO 概念（RLHF 用） | M10 | ⭐⭐ | 纯理论 |
| BBPE / tokenizer 工程题 | M7 | ⭐⭐ | 理论 |

**D. AI infra 段新增的面试考点（2026-08-08 补，目标C通晓infra）**：

| 考点 | 在哪个 M 学 | 重要度 | 类型 |
|------|------------|--------|------|
| **PagedAttention**（vLLM 核心，KV cache 分页） | M6 | ⭐⭐⭐ | 理论+部署仪式看实跑 |
| **continuous batching**（连续批处理） | M6 | ⭐⭐⭐ | 理论 |
| **gradient checkpointing**（省显存换算力，=activation checkpointing 同义） | M6 | ⭐⭐⭐ | **实战（加到nanoGPT）** |
| **mixed precision AMP**（fp16 loss scaling / bf16 不用） | M6 | ⭐⭐⭐ | **实战（三组对比）** |
| **torch.compile**（静态图编译） | M6 | ⭐⭐ | **实战** |
| **speculative decoding**（小模型猜大模型验） | M6 | ⭐⭐ | 纯理论+读论文 |
| **profiling / torch profiler** | M6 | ⭐⭐ | **实战** |
| **ZeRO 三阶段**（1优化器/2+梯度/3+参数） | M11 | ⭐⭐⭐ | **手算（1B模型分片题）** |
| **张量并行 TP**（对比 DDP） | M11 | ⭐⭐ | **toy（nanoGPT手切列/行并行）** |
| **NCCL / Ring AllReduce** | M11 | ⭐⭐ | **toy（numpy单进程模拟）** |
| **DDP 单进程真代码** | M11 | ⭐⭐ | **真代码（world_size=1样板）** |
| **GPTQ / AWQ**（后训练量化算法） | M11 | ⭐⭐ | 纯理论 |
| **prefix caching / chunked prefill** | M11 | ⭐⭐ | 纯理论 |
| **推理部署：GGUF/Ollama/vLLM** | 部署仪式 | ⭐⭐⭐ | **实战** |
| **FastAPI serving + OpenAI 兼容接口** | 部署仪式 | ⭐⭐⭐ | **实战** |
| **流式输出 streaming（SSE）** | 部署仪式 | ⭐⭐⭐ | **实战** |
| TensorRT-LLM / 蒸馏剪枝稀疏化 | M11 扫盲 | ⭐ | 行业扫盲 |

**infra 通晓边界说明**（2026-08-08 Haiku 审视后修订）：理论全覆盖上表；实战分三档——(1) 真实战：gradient checkpointing / **AMP** / **torch.compile** / profiling / 推理部署全家桶；(2) toy 实现：Ring AllReduce（numpy）/ ZeRO（手算）/ TP（手切线性层）/ DDP（单进程样板）——从"我知道"升级到"我手写过"；(3) 纯理论+阅读：GPTQ/AWQ 算法思路、speculative decoding 论文、PagedAttention SOSP'23 论文。集群编排（k8s/Slurm/RDMA/InfiniBand）行业扫盲知道名词即可。**不开第三项目、不租云**——toy 实现已够面试。

**C. agent 段新增的面试考点**：

| 考点 | 在哪个 M 学 | 重要度 |
|------|------------|--------|
| **RAG 检索增强生成** | M13（记忆环节自然引入） | ⭐⭐⭐ |
| **CoT 思维链** | M14（ReAct 前置） | ⭐⭐⭐ |
| function calling 机制 | M12-M14 | ⭐⭐⭐ |
| ReAct 模式 | M14 | ⭐⭐⭐ |
| 多轮对话/上下文管理 | M13 | ⭐⭐ |
| prompt engineering | M13 | ⭐⭐⭐ |

**面试覆盖说明**：以上四表（A大模型细节/B路线图考点/C agent/D infra）覆盖大模型算法岗/应用岗/infra 岗面试 90%+ 高频考点。RAG 和 CoT 原来不在 agent 路线里，因为面试高频，特意塞进 M13/M14。infra 段（D 表）2026-08-08 补，覆盖推理引擎/部署/分布式训练高频考点。剩余 10%（如具体框架 API、业务场景题）靠毕业项目 M15 + 实战积累。

## 🔧 项目B技术选型（2026-08-07 定稿·M8-M15 全程用这套）

学生认可的双项目并行方案中"项目B（现成小模型）"的具体选型。**现在不下载不装**，到 M8 开始前再动手。

| 用途 | 选型 | 理由 |
|---|---|---|
| 本地基座（M8 SFT / M9 LoRA / M11 量化） | **Qwen3-0.6B**（bf16） | 16G 上 LoRA 峰值 ~5G 余量极大；魔搭下载国内稳；36T token 数据同尺寸强 |
| 训练框架 | transformers + peft + trl 从0写 | 透明每步看得见，不用 LLaMA-Factory 黑盒（黑盒排查反而难） |
| agent API（M12-M15） | **智谱 GLM-4.7-Flash** + openai SDK | 真·免费、function calling 完整、OpenAI 兼容、手机号注册；改 3 个字符串切 Qwen/Kimi/DeepSeek |
| RAG embedding（M13） | bge-base-zh-v1.5 | 110MB，CPU 都能跑，教程多 |
| 向量库 / 文档加载（M13） | faiss-cpu / pypdf | Py3.14 有 win wheel；别用 unstructured（系统依赖一堆还装不上 3.14） |

**为啥不是其他选项**：Qwen3（2025-04 发，36T token）强于 Qwen2.5；TinyLlama 中文差不教学。DeepSeek 的 function calling 只在 `deepseek-chat` 有、`deepseek-reasoner` 不支持，且偶发非法 JSON 要兜底；智谱全系列支持 + 免费常驻。

**M8 开始前一次性补依赖**（走清华源）：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft accelerate trl modelscope bitsandbytes openai sentence-transformers faiss-cpu pypdf
```

**下载模型**（魔搭，国内 CDN）：
```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen3-0.6B', cache_dir='F:/study/big_model/models')
```

**三个必须知道的坑**：
1. **量化在小模型上是"学机制"不是"提速"**：bitsandbytes 0.50.0 官方支持 Win+sm_120+Py3.14（能装能跑，v0.45.3 起官方编译了 sm_120 kernel），但 RTX 5090 实测 1.5B 模型 NF4 比 fp16 **慢 42%、能耗涨 29%**，收益要到 7B+ 才转正。M11 量化课定位诚实——学机制为大模型打基础，真正价值是"把 8B 塞进 16G"。
2. **Py3.14 + peft/accelerate/modelscope 大概率能装但无官方背书**：transformers/trl/bitsandbytes 显式列 3.14，但 peft/accelerate/modelscope 的 classifiers 没更新到 3.14（`Requires-Python >=3.10` 无上界，纯 Python 应该能跑）。兜底：装不上降 Python 3.13.7 重建 venv（torch 2.11+cu128 在 3.13 也有官方 wheel，迁移成本低）。
3. **智谱免费政策可能调整**：以控制台实时信息为准，取消就切 Qwen3.7-flash（同样 OpenAI 兼容，改两行代码）。

### 风险与坑（提前知道）

- **学生 GPT loss 能到哪**：字符级+12层继续训到 1.5-1.8（零星成句莎士比亚），但永远到不了"能听指令"——这正好是 M6-M8 的教学动机："光预训练不够，所以要有微调"。
- **RLHF 最难**：M10 分三层讲（reward model→RL 优化→合起来），卡住允许"先记结论，以后回来"。
- **BPE 训练抽象**：先拿 10 个英文单词手算 BPE 合并步骤（白板演示），再上代码。
- **KV cache 维度对齐**：先画图（cache 形状 `[B, n_head, T, head_dim]`），再改 forward。
- **LoRA 显存**：Qwen3-0.6B bf16 约 1.2G，LoRA 训练（r=16, seq 1024）峰值 ~4-5G，16G 极宽松；1.7B 峰值 ~8-10G（开 gradient_checkpointing）也够。加载用 `dtype=bfloat16`（Blackwell 原生支持），别 float32。
- **infra 容易越挖越深**：尤其分布式训练水深，给理论设硬上限——M6/M11 里的 infra 内容严格按时长控制（每个点 5-10 分钟图解），不单独开 infra 深挖课。把 infra 兴趣引流到部署仪式（有出口有产物），别发散到"我也想搞分布式训练"（无底洞）。
- **vLLM Windows 风险**：RTX 5080 Blackwell sm_120 很新，vLLM 在 Windows 原生支持差。部署仪式保底用 Ollama（封装 llama.cpp，Windows 友好，自带 OpenAI 兼容 API）；首选 WSL2 跑 vLLM；最保底 llama.cpp+FastAPI 手写。到部署仪式时再定，现在不动。
- **分布式训练只能纯理论**：DDP/ZeRO/TP/NCCL 在 16G 单卡永远没法真跑，别试图用 `torchrun --nproc_per_node=2` 单卡模拟两卡（学生困惑+跑不出通信开销）。老实画图理解。

## 📌 备忘

- 环境：见 `ENV.md`
- 教学规范：见 `TEACHING.md`
- 学生偏好：互动式、少公式、多类比；代码要自己写，AI 不代写
- CPU 训练慢，每课控制在 5 分钟内，用小语料
