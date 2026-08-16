# 📚 大模型课程 · 教学计划与进度(可移植版)

> **本文件是"现在教到哪"的唯一入口。** 任何 AI / 老师接手课程,先读本文件,30 秒即可继续教。
> 历史细节、踩坑原委、回顾材料 → 看 `PROGRESS.md`(本文件的详细档案版)。

**最后更新:2026-08-13**

---

## 0️⃣ 30 秒必读(接手就做)

- **学生**:新手,已手写 nanoGPT(`lesson09_attention.py`,120M/6层/12head,GPU 训练过,loss 4.5→2.5),已学 BPE(M7),正在学 SFT(M8,当前搁置)+ M8.6 迷你编译器支线 + GPU 硬件补课(进行中)。
- **环境(两台机器并存,别搞混)**:
  - **① F 盘台式(有独显,主力)**:`F:\study\big_model\nanoGPT_venv`(Python 3.14.3 + torch 2.11.0+cu128 + transformers 5.14 + cuda 可用)。**RTX 5080(Blackwell sm_120,16GB)**→ M8 SFT 真训练能跑。跑脚本:`F:/study/big_model/nanoGPT_venv/Scripts/python.exe xxx.py`。代码仓库 `F:/study/big_model/my_gpt/` 在这台。
  - **② D 盘另一台(无独显,学原理用)**:`D:\ai\nanoGPT_venv`(Python 3.11.9 + torch 2.13.0+cpu + transformers 5.15 + tiktoken/numpy/tqdm/requests)。**无 NVIDIA 独显**(仅 Intel 核显)→ M0-M7/M8.5/M8.6 能跑,M8 SFT 真训练跑不动。跑脚本:`PYTHONUTF8=1 /d/ai/nanoGPT_venv/Scripts/python.exe xxx.py`。自检:`verify_env.py`。
  - ⚠️ 之前一棒 AI 把 D 盘信息当"误记"用 F 盘覆盖了——**错的,两台都是真实环境,并存**。接手前先确认学生在哪台机器(看当前 venv 路径或问学生)。
  - 🚨 **关键隔离限制(2026-08-16 学生明确)**:**两台是不同电脑,本机(F 盘)的本地资源,另一台(D 盘)的 AI 一概取不到**。给对方 AI 的指令**只写"获取命令 + repo 相对路径",绝不假设对方有本机文件**。
  - 📦 **对方 AI 获取资源的远程命令清单**(照着敲就能拿到,不用问学生):
    ```bash
    # 1. 主代码仓库(PLAN/PROGRESS/mini_compiler/lesson09/lesson13 等全在这)
    git clone https://github.com/Wuweilong-Leo/nano_gpt-tutorial.git
    # 2. AIInfraGuide(infra 主线素材,261 篇文档)
    git clone https://github.com/caomaolufei/AIInfraGuide.git
    # 3. tinygrad(M8.6 编译器参考书,线上读每文件 2-3 函数)
    git clone https://github.com/tinygrad/tinygrad.git
    # 4. msmodeling(华为昇腾推理仿真器,M8.5⑥ runtime 调度教具)
    git clone https://gitcode.com/Ascend/msmodeling.git
    ```
    - **Qwen3-0.6B 模型权重**:走 HuggingFace,`from_pretrained("Qwen/Qwen3-0.6B", cache_dir=...)`,国内配 `HF_ENDPOINT=https://hf-mirror.com`(详见踩坑速查#6)
    - **nanoGPT venv**:对方自己建,`pip install torch transformers datasets peft accelerate bitsandbytes tiktoken`(GPU 版按对方机器的 CUDA 版选 torch wheel)
    - **本机已 clone 但对方没有的本地路径**(仅供参考,对方需重新 clone):`F:/study/big_model/{my_gpt,tinygrad,msmodeling}`、`F:/study/AIInfraGuide`、`F:/study/big_model/models/_hf_cache`
- **当前进行中**:**GPU 硬件补课 · 消化阶段**(学生选"先消化 GPU,不急着加新课")。已讲透:CPU vs GPU、SM/Tensor Core/CUDA Core 层次(5080=84 SM×4 Tensor Core×128 CUDA Core)、显存金字塔(HBM→L2→共享内存→寄存器)、Roofline/Memory Wall、wmma 指令(load→mma→store)、block/grid/thread/warp 调度、内存行主序布局。素材来自 AIInfraGuide `docs/guides/模块一-前置知识/gpu/gpu-basics.md`。
- **学生当前卡在(下节课接着的)**:GPU 知识刚学完一轮,学生在消化。下一步等学生发问——可能方向:① 继续补 GPU(warp/SIMT/同步,模块二 2.1-2.4)② 做 GPU 小实验验证消化 ③ 转 CUDA 编程实战(模块二 1.1 起)。**M8.6 编译器路线 B 暂停**(学生定:先补 CUDA 再回编译器,因不懂 CUDA 看不懂 Triton 内核)。M8 SFT step3 仍搁置。
- **📚 学习路线对接 AIInfraGuide(2026-08-16 学生拍板,混合制)**:
  - **infra 主线走 AIInfraGuide**(`F:/study/AIInfraGuide/`,261 篇文档站):GPU 硬件(模块一第5章)→ CUDA 编程(模块二,1.1-8)→ 编译器(模块二第7章,接 M8.6 路线 B)→ 分布式(模块三,对应原 M12)→ 推理优化(模块四,对应原 M10/M11)
  - **SFT/agent 留原 PLAN**:M8 SFT(step3 切割线)、M9 LoRA、M13-15 agent 不走 AIInfraGuide(它没这些)
  - **M8.6 编译器**:先补 CUDA(模块二 1-2 章)→ 再回路线 B 对照真 inductor Triton(学生明确顺序)
  - 原 PLAN 的 M6/M8.5/M10/M11/M12 infra 部分**作废**,改用 AIInfraGuide 对应章节;原 PLAN 的 M0-M5/M7/M8/M9/M13-15 保留
- **教学规则(学生明确要求的,违反会被纠正)**:
  1. 代码**学生自己写**,老师只给骨架/hint/解释,绝不代写完整可运行代码
  2. 语法/API 细节直接给答案,不猜谜("这种语法问题可以直接告诉我")
  3. 不深入公式层,只讲 usage + intuition("太细节了""这个也太细节了吧")
  4. 不要求无意义验证("这些不用验证了,感觉验证了也没啥用")
  5. 一次一小步,不要一次抛太多("你写的太多了,一步一步来")
  6. 风格:互动式、多类比、少公式

---

## 1️⃣ 下一步行动(别家 AI 接手立刻照做)

### 当前任务:M8.6 第④步 · 填 `fuse` 的两处核心 TODO

**背景**:`mini_compiler/passes.py` 的 `fuse` 已重构为 list+循环版,骨架跑通 8→3 节点(硬编码 RMSNorm 链图测的)。两处 TODO 是逻辑命门,学生填完才算融合 pass 收尾。

**学生已理解的概念**(不用再讲):
- 为什么必须"两遍":单遍循环按插入顺序走,中间节点排在终点 MUL 前面,先搬后认删不掉
- 用"名字"区分留/删,别用 op 类型——叶子 x/w 也是 INPUT,靠 op 删会误伤
- `dead`(set,存要蒸发的中间节点名字)和 `repl`(dict,命中 MUL→(x,w))分工:一个管删谁、一个管谁换头
- 非匹配图(如 z=a*b)会崩:根因是"先伸手取 inputs[0]、后验 op",摸到叶子就 IndexError;正解是"先验 op 再摸"

**让学生填的两处**(`passes.py` fuse 函数内,代码由学生写,hint 已给):
- **TODO ①**(循环内 3 行):`if cur.op != want: hit = False; break` + `mid.append(cur)`——先验再摸,修崩溃
- **TODO ②**(1 行):`dead = {n.name for n in mid} | {mid[1].inputs[1].name}`——mid=[r,a,m,x2],mid[1]=a(ADD),a.inputs[1]=e(eps 的 CONST,不在 chain 上单独加)

**过关验证**(学生贴输出):
1. 跑 `passes.py` → 优化前 8 → 优化后 3(`INPUT x`/`INPUT w`/`RMSNORM(x,w)`)
2. 跑非匹配图 `z=a*b`(`a=INPUT; b=INPUT; z=MUL[a,b]`)→ 不崩,原样返回 3 个节点(没命中指纹,原样留)

**填完后的分支**(二选一,学生定):常量折叠(三 pass 凑齐)或串主流程(`parse→DCE→fuse` 管线,test_prog.txt 待学生补)。

### 搁置:M8 SFT Lesson2 step3 切割线函数

M8.6 收尾后回来续。概念已讲完(assistant 头 [151644,77091,198] 定位、-100 masking),代码还没跑通。详见 PROGRESS.md。**注意:M8 SFT 真训练在这台机器跑不动(无独显),只能学原理/读代码。**

**step4 预告(回 M8 时)**:SFT 训练循环——batch 多条数据、只取答案部分算 loss、`loss.backward()`、AdamW step。

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
| **M8** | **SFT 指令微调** | 🔄 **进行中**(搁置) | Qwen3-0.6B + Alpaca-zh;step3 切割线函数卡住 |
| M8.5 | 大模型端到端执行流(文件→logits 七站) | ✅ 前传完成 | 见 §7️⃣;已跑通 dynamo 96 节点 + 融合 96→84 |
| M8.6 | 手写迷你 AI 编译器(影子版) | 🔄 ①②③④✅ ⑤路线B暂停 | lower_c.py 跑通;**先补 CUDA 再回编译器对照真 inductor**(学生定) |
| M9 | LoRA | ⬜ 待学 | 不走 AIInfraGuide |
| **infra 主线** | **GPU+CUDA+编译器+分布式+推理** | 🔄 **走 AIInfraGuide** | 见下表;原 M10/M11/M12 infra 部分作废 |
| M13-M15 | agent(四零件/ReAct/RAG/框架) | ⬜ 待学 | 毕业项目;不走 AIInfraGuide |

### 📚 infra 主线对接 AIInfraGuide(2026-08-16 起,替代原 M10/M11/M12 infra)

| 阶段 | AIInfraGuide 章节 | 对应原 PLAN | 状态 |
|---|---|---|---|
| GPU 硬件 | 模块一第5章 gpu-basics.md | (原 PLAN 没系统讲) | 🔄 消化中 |
| CUDA 编程 | 模块二 1.1-2.4(环境/编程模型/内存/Warp/同步) | (原 PLAN 没讲) | ⬜ 下一步 |
| CUDA 算子 | 模块二 3.1-6.2(Reduce/GEMM/Softmax/FlashAttn) | (原 PLAN 没讲) | ⬜ 待学 |
| AI 编译器 | 模块二第7章 | M8.6 路线 B 对照真 inductor | ⬜ 补完 CUDA 后回 |
| 分布式训练 | 模块三(11 章:通信原语/数据并行/ZeRO/TP/PP/MoE/3D) | 原 M12 | ⬜ 待学 |
| 推理优化 | 模块四(11 章:LLM推理/vLLM/量化/SpecDec/PD解耦) | 原 M10/M11 | ⬜ 待学 |
| 面经 | docs/interview/(几十家公司真题) | (原 PLAN 没有专门面经) | ⬜ 面试前刷 |

**素材库**:`F:/study/AIInfraGuide/`(本地 clone,github.com/caomaolufei/AIInfraGuide)。原 msmodeling 素材库仍保留作"仿真教具"(M8.5 ⑥ runtime 调度用)。

总进度:**5/15 原里程碑完成(33%)**,infra 主线切 AIInfraGuide 重新铺。三条目标线 A大模型/B agent/C infra 同步推进。

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

## 7️⃣ M8.5 教学方案:大模型端到端执行流(已定稿,待教)

> **定位**:M8(SFT)之后、M9(LoRA)之前;前置是 M8 step3 收尾。时长 2~3 次课,一次一站,学生贴输出为过关信号。
> **可移植原则(接手 AI 必守)**:源码参考只写"获取命令 + 包内相对路径",**绝不写本机绝对路径**;接手先跑"30 秒自检脚本"。

**总地图**:磁盘文件 →①加载→ 内存 model →②tokenizer→ input_ids →③画图→ 96 节点 →④优化→ 84 节点 →⑤生成→ 内核序列 →⑥运行时调度→⑦执行→ logits

| 站 | 目标一句话 | 可变移植源码参考(怎么得到/怎么找) |
|---|---|---|
| ① 加载 | config 如何决定一个模型;登记簿→门卫→搭壳→填数 | pip 装 transformers;包内 `models/auto/modeling_auto.py`(grep `"qwen3"`)、`models/auto/auto_factory.py`(`_get_model_class`)、`models/qwen3/modeling_qwen3.py`(Qwen3Model `__init__` 的 `range(config.num_hidden_layers)`)、`modeling_utils.py`(load_state_dict)。已讲:6 站链路见 PROGRESS |
| ② tokenizer | 文本→id;三件套分工;encode 四步;词表=训练产物 | 模型目录内 `tokenizer.json`(`model.merges`/`model.vocab`/`added_tokens_decoder`);transformers 包内 `tokenization_utils_base.py`(入口);底层 BPE 在 Rust `tokenizers` 库(本地只有 .pyd,源码看 GitHub huggingface/tokenizers) |
| ③ 画图 | 套娃→摊平清单;节点=一次动作;96=拆账 | torch 包内 `_dynamo/`(export)、`fx/`(老式 tracer 为何失败做对照);教具 `my_gpt/ai_compiler_test.py`(已跑通:Qwen 一层 96 节点) |
| ④ 优化 | 图上手数学不变的手脚;模式匹配;融合=小组件→大内核;96→84 | torch 包内 `_inductor/pattern_matcher.py`(融合规则);教具 `my_gpt/_fusion_check.py`(已跑通:4 处 Norm 指纹) |
| ⑤ 生成 | lowering 表:图→tiling/循环→指令;拆→合→再拆 | `pip install triton`(本机未装,接手机器要装;装不上就纯概念讲);torch 包内 `_inductor/codegen/` |
| ⑥ 运行时调度 | 内核排队/显存池/多请求调度;**真现象用 PyTorch,真算法用 msmodeling,工业真身 vLLM** | PyTorch 演示:分配→释放→`torch.cuda.memory_reserved()` 不掉(显存池);msmodeling(任何机器 `git clone https://gitcode.com/Ascend/msmodeling`)内 `serving_cast/engine.py` BatchScheduler + `kv_cache_manager.py`(**仿真:算法真、执行假,只教概念**);vLLM `vllm/core/scheduler.py`(线上看,不下载) |
| ⑦ 执行 | 一次生成多次执行;generate 循环 | transformers 包内 `generation/utils.py` 的 `generate`(采样循环);顺带做掉 M6 欠账 `torch.compile` 对比 |

**30 秒自检脚本(接手必跑)**:
```python
import sys, pathlib, torch, transformers, tokenizers, importlib.util
print("python:", sys.executable)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__, "|", pathlib.Path(transformers.__file__).parent)
print("tokenizers:", tokenizers.__version__, "|", pathlib.Path(tokenizers.__file__).parent)
print("triton:", importlib.util.find_spec("triton") is not None)
```

**衔接**:M9 LoRA=站①的加载玩法;M10 推理加速=站④⑤⑥的底层直觉;M6 欠账 torch.compile=站⑦。

### 🔧 M8.6 教学方案:手写迷你 AI 编译器(影子版,已定稿)

> **定位**:M8.5 之后、M9 之前。**参考书**:tinygrad(github.com/tinygrad/tinygrad,线上读,每文件只读 2-3 个关键函数)。**产物**:`my_gpt/mini_compiler/` ~250 行,5 个文件。原则:一次一小步、学生写代码、每步贴输出过关。

| 步 | 目标 | 知识点 | tinygrad 参考 | 过关输出 |
|---|---|---|---|---|
| ① 摸地形 | 认识编译器源码 | 节点 = op+输入列表+参数(记账单位) | `_ops.py` 算子表;`lazy.py` 节点类 | 口述节点记录了什么 |
| ② 定 IR | 自己设计图的表示 | SSA、Op 表、Node 类;IR=编译器地基 | 对照 `_ops.py` 设计 Op 表 | `ir.py` 手工造 RMSNorm 指纹 5 节点打印成功 |
| ③ 写前端 | 文本→图 | 每行一指令的迷你语言(`%2 = mul %0, %1`),解析=按行拆词 | (tinygrad 无前端,用 Python API;我们对齐 LLVM 教程补上) | `parse.py`:文本→图,节点数正确 |
| ④ 优化管线 | 三个 pass 串起来 | 融合(指纹→rms_norm)/死代码消除/常量折叠;pass=图遍历+改写 | `engine/scheduler.py` 遍历重排 | 每 pass 前后打印节点数,样例 7→3 |
| ⑤ 后端+验收 | 图→可执行指令 | 图→VM 指令(LOAD/CALL/STORE),numpy 实现执行 | `engine/realize.py` | 数值对拍:直跑 vs 编译后跑一致;真实接入:Qwen 96 节点图→融合 pass→84 |

**验收约定**(诚实分层):结构对拍(节点数 96→84)用真模型(dynamo 导出 + adapter 40 行);数值对拍用自己样例(VM op 集不全,权重大数据不进 VM)。
**钩子**:M8.5 的 96 节点图/融合脚本=输入和参考答案;M10 学 TVM/inductor 有体感;面试口径:"手写过迷你编译器:SSA IR、fusion/DCE/常量折叠、VM 后端,接入了 dynamo 导出的真实模型图"。

### ⚠️ 接力交接:⑤ 路线 B 已定,lower_c.py 已跑通(2026-08-14,上一棒 AI 留)

**④ 已收尾**:`passes.py` 的 `fuse` 8→3 稳定跑通,两个 TODO 填完。`remove_dead_nodes` 学生自己写的,审过。**①②③④ 四步全部过关**。

**⑤ 方向学生已拍板 = 路线 B**(迷你版 + 真版并排)。学生两个硬要求:① 核心诉求是"看懂 AI 编译器怎么把大模型转成一个个内核";② 纯概念吸收不了,要代码当拐杖。三条候选里 A(造完迷你VM,离真内核远)、C(纯读真 inductor,吸收不了)被排除,选 B。

**B 当前进度:`lower_c.py` 已写完跑通** ✅
- 文件:`mini_compiler/lower_c.py`(上一棒 AI 写的,桥梁不是练习题,学生要看懂可改)
- 干啥:把融合图(3 节点)和原始图(8 节点)**分别 emit 成 C 循环代码字符串**,打印对照
- 跑通输出:原始图 emit 3 个 for 循环(POW/MEAN累加/MUL),融合图 emit 2 个(POW+MEAN合一 / MUL)→ **直观看到融合省循环 = 省内核 = 省启动开销**(④ 融合在后端的回报)
- 学生该做的:改坏几次 `lower_c.py`(改 N=8、加节点、合循环),确认懂每行

**B 下一步(未做,下棒接手)**:对照真 inductor 生成的 Triton 内核
- 让学生跑 `torch.compile` 装饰一个 rsqrt(mean(x²)+eps)*w,去系统 temp 找 `__inductor_*.py` 打开
- 和 `lower_c.py` 的 C 代码并排看:骨头一样(循环+访存+算),词汇不同(`tl.load/tl.program_id` vs `a[i]/for i`)
- 目的:学生亲眼看到"我们迷你版 emit 的 C 循环 = 真 inductor emit 的 Triton,结构同构",这就是"看懂大模型怎么转内核"
- 注意:本机 cpu torch 能跑 inductor codegen(生成 .py 不需要 GPU),但执行编译后的内核要 GPU(本机无独显,看代码即可不执行)

**踩过的坑(下棒别重蹈)**:
1. 一度把 VM 定位成"替身跳过 lowering",被学生识破讲拧了。正确定位:VM 是字节码派后端(和 Crafting Interpreters 同级),lowering 是教学正餐不跳过,真正略过的是"指令→机器码"。
2. 调研 agent 派子 agent 太多撑爆上下文(100万 token),没拿到完整结论。结论是凭知识补的(Crafting Interpreters=档①无寄存器分配栈式VM;tinygrad/TVM=档③带tiling;Kaleidoscope=档③借LLVM)。学生认可"档①和 CraftInterp 同档,不算偷懒"。
3. 学生对"多链共享中间节点"问过——答案是引用计数,讲过不写。
4. 学生骂过"方案老定不下来"——根因是上一棒把决策权抛回给学生却没逼定。教训:给推荐 + 逼学生一句话拍板,别开放性询问。

**已验证跑通**:
- `cd mini_compiler && python passes.py` → 8→3(④ 融合)
- `cd mini_compiler && python lower_c.py` → 两份 C 代码对照(⑤ lowering 迷你版)
- 本机 cpu torch 即可,不需 GPU

## 8️⃣ 已讲课程内容速查(怕忘)

- **M6**(完成):temperature/top-k/top-p 采样(手写 pipeline + 封装)、perplexity、beam search 概念、训练工程速查(warmup/AdamW/AMP 概念遗留 M8)
- **M6 补课 infra 眼镜**(2026-08-10 补):朴素 KV cache → 碎片问题 → PagedAttention 分页(block table ~ 操作系统虚拟内存)→ 对照 msmodeling kv_cache_manager(BLOCK_SIZE=128, allocate_slots/free, req_blocks=页表)
- **M7**(完成):BPE 原理(byte-level、频次合并)、tiktoken 实操、"你好" vs 英文 token 数、Qwen3 tokenizer 中文友好(你好=1 token)
- **M8 Lesson1**(完成):加载 Qwen3-0.6B 权重、验证生成("死循环重复"症状= SFT 动机)、fp16/device_map="auto"
- **M8.5 前传**(2026-08-12,完成):模型交付件四件套(config/tokenizer三件套/generation_config/safetensors);加载 6 站链路(登记簿 modeling_auto → 查表 auto_factory → import modeling_qwen3 → 搭壳 → 填 311 权重);tokenizer 三件套职责+encode 四步(词表=训练产物,推理只查不建,merges.txt 用排序定切分);AI 编译器主线:画图(dynamo.export,Qwen 一层 96 节点,已跑通教具 ai_compiler_test.py)→ 优化(模式匹配 4 处 Norm 指纹 96→84,已跑通教具 _fusion_check.py)→ 生成 → 运行时调度;节点=一次动作、图=调度员视图非最底层、msmodeling 调度=仿真(算法真执行假)