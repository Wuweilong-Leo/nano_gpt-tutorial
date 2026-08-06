# 学习进度

> **下次上课从这里继续。** 最后更新：2026-08-04

## 📍 当前位置

- **已完成**：第 1-7 课（基础）+ 第 9-11 课（KQV/Embedding/Block/完整 GPT/训练生成）= 里程碑 M0-M5
- **进行中**：完整 GPT 训练 + 推理生成全流程跑通（10万步 loss 613→2.3）
- **下次上课**：**M6**——temperature/top-k/top-p 采样 + perplexity 困惑度 + KV cache 推理加速

## 📊 进度仪表盘

```
端到端主线：5 / 12 里程碑完成（M0-M5 ✅）
[████░░░░░░░] 42%

下次：M6 生成质量+评估（temperature/采样/perplexity/KV cache）
最终毕业：M12 手写 agent
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

## 🗺️ 完整里程碑路线图（2026-08-07 与 Haiku 二次讨论定稿·端到端版）

**核心主线一句话**：你不是在学一堆散件，你是在造一个 agent——只是先把每个零件搞懂。tokenizer=agent 听懂人话的前提，KV cache=agent 响应快的前提，SFT=agent 听指令的前提，采样=agent 多样性的前提。

**设计原则**：理论课与实战交织，**增量构建**——每课从 0 写 20-30 行，下课必须能跑出结果。架构进阶不搞名词轰炸（只 RoPE 深讲，其余压成"行业地图"扫盲）。agent 阶段**先手写禁框架**，框架对照放最后。

### 已完成（M0-M5）
- **M0** 启动：clone + 数据准备 → ✅（2026-08-02）
- **M1** KQV / Self-Attention → ✅（2026-08-02，08-04 补 mask）
- **M2** Embedding + 位置编码 → ✅（2026-08-04）
- **M3** Block（attention+MLP+2LN+2残差）→ ✅（2026-08-04，学生从别的 AI 学完写代码）
- **M4** 完整 GPT 类（`class GPT`，ModuleList，forward 双返回）→ ✅（2026-08-05）
- **M5** 训练+生成 → ✅（2026-08-07，10万步 GPU 训练 loss 613→2.3，sample 生成跑通）

### 待完成（M6-M12）

| 里程碑 | 内容 | 课数 | 类型 |
|--------|------|------|------|
| **M6** 生成质量+评估 | temperature/top-k/top-p 采样 + perplexity 困惑度 + KV cache 推理加速 | 2 课 | 理论+改代码 |
| **M7** BPE+中文语料 | BPE 子词分词原理 → 训练 BPE → 用新 tokenizer 重训 GPT（词表 65→BPE） | 2 课 | 实战代码 |
| **M8** SFT 微调 | 指令数据格式 + SFT 训练循环（主线用现成小模型 qwen-0.5B/1.8B） | 1 课 | 实战代码 |
| **M9** LoRA | LoRA/PEFT 原理 + 用现成小模型微调（支线：给自己的 GPT 加小适配头） | 1 课 | 实战代码 |
| **M10** 对齐理论 | reward model → RLHF 三阶段 → DPO（纯讲+图解，不写代码，全课最难） | 1 课 | 纯理论 |
| **M11** 行业地图 | RoPE 深讲（对比绝对位置编码）+ GQA/Flash/MoE/长上下文扫盲 + 量化 int8 体验 | 1 课 | 理论+小实验 |
| **过渡仪式** | 自己 GPT vs API 同 prompt 生成对比 → 亲眼看差距（桥梁，半课） | 半课 | 体验 |
| **M12（毕业）** 第一个 agent | 手写单工具 agent → +memory → ReAct（禁框架），最后对照轻量框架 | 2-3 课 | 实战代码 |

### 关键设计决策

1. **架构进阶压成一课**：RoPE/Flash/MoE/GQA 不全讲——只 RoPE 深讲（有学生现在的绝对位置编码做锚点），其余"知道存在+为什么这么做"即可，避免名词轰炸。
2. **BPE + 中文语料合并**：原"换中文语料"被吸收进 M7——BPE 训练用中文语料当素材，词表从 65 升级到 BPE，重训 GPT。
3. **SFT/LoRA 用现成小模型**：学生 GPT 太小（120M、字符级）SFT 看不到效果。M8/M9 主线用 qwen-0.5B/1.8B（16G GPU LoRA 无压力）。**支线**：用 LoRA 思想给自己的 GPT 加适配头，让"我的模型"主线不断。
4. **过渡仪式承上启下**：进 agent 前用自己 GPT vs API 对比生成，教学话术——"你的模型让你懂了 LLM 大脑怎么工作，这是纯调包的人没有的底气。现在做 agent 要能听指令的大脑，换用更强的真大脑。但你懂底层，你是'懂发动机的司机'。"
5. **agent 先手写禁框架**：网上搜 agent 会被 langchain 淹没。第一条规矩：先手写，禁框架。框架对照放最后一课，理解"框架=帮你省代码"。
6. **API 选国内能直连的**：deepseek/qwen（有 function calling），避免卡在网络分心。

### agent 零件 ↔ 大模型知识对应（毕业项目用）

| agent 零件 | 学生已学的大模型知识 | 对接说明 |
|------------|---------------------|----------|
| 大脑 LLM | GPT 前向、采样、temperature | 直接复用，换 API |
| 工具调用 | tokenizer encode/decode | 要让 LLM 输出结构化指令，得懂 token 边界 |
| 记忆 | context window、位置编码 | 记忆=塞进 context window，wpe 限制=记忆容量 |
| 规划 | next-token 自回归 | 链式生成=最朴素规划，ReAct 是升级版 |

### 风险与坑（提前知道）

- **学生 GPT loss 能到哪**：字符级+12层继续训到 1.5-1.8（零星成句莎士比亚），但永远到不了"能听指令"——这正好是 M6-M8 的教学动机："光预训练不够，所以要有微调"。
- **RLHF 最难**：M10 分三层讲（reward model→RL 优化→合起来），卡住允许"先记结论，以后回来"。
- **BPE 训练抽象**：先拿 10 个英文单词手算 BPE 合并步骤（白板演示），再上代码。
- **KV cache 维度对齐**：先画图（cache 形状 `[B, n_head, T, head_dim]`），再改 forward。
- **LoRA 显存**：qwen-1.8B 全精度约 3.6G，LoRA 训练峰值 8-10G，16G 够。加载时用 `dtype=float16/bfloat16`，别 float32。

## 📌 备忘

- 环境：见 `ENV.md`
- 教学规范：见 `TEACHING.md`
- 学生偏好：互动式、少公式、多类比；代码要自己写，AI 不代写
- CPU 训练慢，每课控制在 5 分钟内，用小语料
