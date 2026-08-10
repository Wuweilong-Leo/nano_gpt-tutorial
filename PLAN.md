# 📚 大模型课程 · 教学计划与进度(可移植版)

> **本文件是"现在教到哪"的唯一入口。** 任何 AI / 老师接手课程,先读本文件,30 秒即可继续教。
> 历史细节、踩坑原委、回顾材料 → 看 `PROGRESS.md`(本文件的详细档案版)。

**最后更新:2026-08-10**

---

## 0️⃣ 30 秒必读(接手就做)

- **学生**:新手,已手写 nanoGPT(`lesson09_attention.py`,120M/6层/12head,GPU 训练过,loss 4.5→2.5),已学 BPE(M7),正在学 SFT(M8)。
- **当前进行中**:M8 SFT(用 Qwen3-0.6B + Alpaca-zh 做指令微调)。
- **学生当前卡在**:M8 Lesson2 第 3 步——写"切割线定位函数"(loss masking 用),概念已讲完,代码还没跑通。
- **教学规则(学生明确要求的,违反会被纠正)**:
  1. 代码**学生自己写**,老师只给骨架/hint/解释,绝不代写完整可运行代码
  2. 语法/API 细节直接给答案,不猜谜("这种语法问题可以直接告诉我")
  3. 不深入公式层,只讲 usage + intuition("太细节了""这个也太细节了吧")
  4. 不要求无意义验证("这些不用验证了,感觉验证了也没啥用")
  5. 一次一小步,不要一次抛太多("你写的太多了,一步一步来")
  6. 风格:互动式、多类比、少公式

---

## 1️⃣ 下一步行动(别家 AI 接手立刻照做)

### 当前任务:M8 SFT Lesson2 · step3 写"切割线定位函数"

**背景**:`lesson13_sft.py` 已完成:加载 Alpaca 数据(48818 条)+ `apply_chat_template` 格式化 + tokenize(一条数据 = 143 个 token)。

**学生已理解的概念**(不用再讲):
- `messages` → `apply_chat_template` → `<|im_start|>user...<|im_end|>` 格式
- loss masking = "只改答案不改题目":非答案部分标 `-100`(PyTorch CE 跳过),答案 + `<|im_end|>` 保留真 id
- 需要定位"答案起点"做切割:前 17 个 token 是问题+assistant头+思考开关,`114566`("以下是")开始是答案

**让学生做的事**(代码由学生写,hint 已给):
- 找到 `assistant_head = torch.tensor([151644, 77091, 198])`(`<|im_start|>assistant\n`)在 input_ids 里的位置
- 它结束的位置 = 切割线;之前全设 `-100`,之后保留原 id
- 输出 `labels` 前 20 个:`[-100, -100, ..., 114566, ...]`
- 学生贴输出 → 老师检查 → 进入 step4 训练循环

**step4 预告(下下步)**:SFT 训练循环——batch 多条数据、只取答案部分算 loss、`loss.backward()`、AdamW step。

### 待补事项(M6 欠账,不紧急但面试会考)

**① 训练工程实战批(AMP 是重点,建议跟 M8 step4 训练循环一起补)**:
- **AMP 三组对比**(fp32 / fp16+GradScaler / bf16):面试高频"fp16 为啥要 loss scaling、bf16 不要";在 nanoGPT 上加,对比显存+loss 稳定性
- gradient checkpointing(加 nanoGPT 对比显存)、torch.compile(对比 step 时间)、profiling(torch profiler)

**② 训练工程概念批**:Adam vs AdamW 深挖、warmup+cosine、CE 为啥不用 MSE、梯度裁剪

**③ 面试细节批**:LN vs BN、Pre/Post-LN、残差原理、MLP 为啥 4×、GELU vs SwiGLU

---

## 2️⃣ 课程总览(M0-M15)

| 课 | 主题 | 状态 | 备注 |
|---|---|---|---|
| M0-M5 | 基础+手写 GPT | ✅ 完成 | 2026-08-02~08-05 |
| M6 | 采样+评估+训练工程 | ✅ 完成(+0.5 infra 眼镜已补) | 2026-08-09 收官 |
| M7 | BPE+中文 token 对比 | ✅ 完成 | 2026-08-09 收官 |
| **M8** | **SFT 指令微调** | 🔄 **进行中** | Qwen3-0.6B + Alpaca-zh |
| M9 | LoRA | ⬜ 待学 | |
| M10 | 推理加速(KV cache/连续batching/PagedAttention) | ⬜ 待学 | 素材:msmodeling |
| M11 | 量化 + 架构进阶(RoPE 深讲,MoE/MLA/MTP 扫盲) | ⬜ 待学 | 素材:msmodeling |
| M12 | 分布式(理论+手算为主) | ⬜ 待学 | 素材:msmodeling |
| M13-M15 | agent(四零件/ReAct/RAG/框架) | ⬜ 待学 | 毕业项目 |

总进度:**5/15 完成(33%)**,三条目标线 A大模型/B agent/C infra 同步推进。

---

## 3️⃣ M8 SFT 详细计划(当前焦点)

目标:让 Qwen3-0.6B 学会听指令(解决它"死循环重复、不答天气"的问题)。

### 5 步走

| 步 | 内容 | 状态 | 关键产出 |
|---|---|---|---|
| 1 | 加载 Alpaca 数据(离线) | ✅ | `ex["instruction"/"input"/"output"]`,48818 条 |
| 2 | 格式化 chat template | ✅ | `<|im_start|>` 文本,143 token |
| **3** | **tokenize + loss masking** | 🔄 | 切割线定位函数(学生正写) |
| 4 | SFT 训练循环 | ⬜ | forward → loss(跳-100) → backward → AdamW |
| 5 | 对比验证 | ⬜ | 训前"死循环" vs 训后"正常回答+收尾" |

### 已讲关键概念(M8 系列,学生已懂)
- `apply_chat_template` = 翻译官:messages(社区通用) → Qwen3 自家 `<|im_start|>` 标记;规则焊在 tokenizer 里
- Qwen3 是"先想后答":token 里有 `...`(151667/151668)思考开关;Alpaca 无思考内容,该位置为空
- loss masking = 老师只改答案:问题/角色标记/思考开关标 `-100`,答案正文+`<|im_end|>` 留真 id
- 微调本质:权重可继续改;Qwen3-0.6B 对齐薄弱,用 48818 条加固"听指令→答"反应
- loss 形状链:`logits(batch,seq,vocab)` → 比真词 log → `loss_per_token(batch,seq)`(vocab 维"挑真词后塌没")→ 跳-100 取平均 → 标量

### 关键 token id 速查(Qwen3)
```
151644 = <|im_start|>    151645 = <|im_end|>
872    = user            77091  = assistant
151667 = 思考开          151668 = 思考关(答案正式起点前的标记)
198    = \n
assistant 头三连: [151644, 77091, 198]
```

---

## 4️⃣ 环境与资产(接手先认路)

| 项 | 值 |
|---|---|
| Python venv | `F:\study\big_model\nanoGPT_venv\Scripts\python.exe`(torch 2.11+cu128 / transformers 5.14 / datasets 5.0 / peft / accelerate / bitsandbytes) |
| 代码目录(git) | `F:\study\big_model\my_gpt\`(remote: github.com/Wuweilong-Leo/nano_gpt-tutorial) |
| 主线代码 | `lesson09_attention.py`(学生自己的 GPT,含采样/评估) |
| SFT 代码 | `lesson13_sft.py`(进行中) |
| BPE 代码 | `lesson12_bpe.py`(tiktoken 小实验) |
| Qwen3 tokenizer | `F:\study\big_model\models\Qwen3-0.6B\`(~/14MB) |
| Qwen3 权重缓存 | `F:\study\big_model\models\_hf_cache\`(596M 参数,fp16 1.5GB) |
| Alpaca 数据缓存 | `F:\study\big_model\data\_hf_cache\shibing624___alpaca-zh\`(48818 条) |
| msmodeling(素材库) | `F:\study\big_model\msmodeling\`(昇腾推理性能仿真器,独立 git 仓库) |
| GPU | RTX 5080(Blackwell sm_120,16G 显存可跑 0.6B 全量 SFT) |

**⚠️ 离线执行纪律(国内环境,脚本必须带)**:`HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`,且**必须在 `import transformers/datasets` 之前**设置(库在 import 时读死开关)。

---

## 5️⃣ 踩坑速查(教学高频)

1. **离线 env 顺序**:先设 env 再 import,否则照样联网超时(已踩两次)
2. `torch_dtype` 在新 transformers 已弃用 → 用 `dtype`(功能相同)
3. `tokenizer(text)` 别调两遍取 `.input_ids`,存一次复用
4. 采样类坑(lesson09 时代):indices 必须整数;新建张量要 `.to(DEVICE)`;scatter_ 还原座位
5. Windows 控制台 GBK 乱码 → 用 UTF-8 运行或写文件看
6. HF 权重下载:直连被墙,xet 镜像 401 → 用 `from_pretrained(cache_dir=...)` 老接口
7. vLLM 在 Windows/Blackwell 支持差 → 部署仪式保底用 Ollama
8. Windows 路径反斜杠 → `open(r"...")` 加 r 前缀

---

## 6️⃣ 素材库:msmodeling(M10-M12 用)

2 个 agent 评审定稿(详见 `PROGRESS.md`「🔧 msmodeling 素材库」节)。核心结论:

- **不融 M6/M8**(话题错位);M10-M12 当"顶级教具"
- **每次只啃 2-3 个函数** + 5 分钟动手验证 + 卡住退回类比
- M10:kv_cache_manager.py(BLOCK_SIZE=128 分页思想,已预热)+ engine.py 连续 batching + performance_model(roofline)
- M11:quantize_utils.py(W8A16/W8A8/W4A8/FP8/MXFP4)+ QuantLinearBase(模拟量化);架构进阶只 RoPE 深讲,MoE/MLA/MTP 扫盲(素材 mla.py/moe_layer.py/mtp.py)
- M12:pipeline_parallel.py 切分逻辑(parallel_group.py 不碰)
- ⚠️ 已否决:sampler.py(实为贪心 argmax,无 top-k/top-p)、qwen3_*.py(只是注册桩,算法在 HF)

---

## 7️⃣ 已讲课程内容速查(怕忘)

- **M6**(完成):temperature/top-k/top-p 采样(手写 pipeline + 封装)、perplexity、beam search 概念、训练工程速查(warmup/AdamW/AMP 概念遗留 M8)
- **M6 补课 infra 眼镜**(2026-08-10 补):朴素 KV cache → 碎片问题 → PagedAttention 分页(block table ~ 操作系统虚拟内存)→ 对照 msmodeling kv_cache_manager(BLOCK_SIZE=128, allocate_slots/free, req_blocks=页表)
- **M7**(完成):BPE 原理(byte-level、频次合并)、tiktoken 实操、"你好" vs 英文 token 数、Qwen3 tokenizer 中文友好(你好=1 token)
- **M8 Lesson1**(完成):加载 Qwen3-0.6B 权重、验证生成("死循环重复"症状= SFT 动机)、fp16/device_map="auto"