# 📚 大模型课程 · 教学计划(接手入口)

> **本文件是"现在教到哪"的唯一入口。** 接手 AI 先读 0️⃣ 区,30 秒开干。
> 历史细节/踩坑原委 → 看 `PROGRESS.md`(详细档案)。**最后更新:2026-08-16**

---

## 0️⃣ 30 秒必读(接手就做)

- **学生**:新手,已手写 nanoGPT(120M/6层/12head,GPU 训练 loss 4.5→2.5)、学过 BPE(M7)。当前在 **infra 主线**(GPU 硬件补课消化中)+ M8 SFT(搁置)。
- **两台机器并存(别搞混)**:
  - **F 盘台式(主力,有独显)**:`F:\study\big_model\nanoGPT_venv`(Python 3.14 + torch 2.11+cu128 + RTX 5080 16GB)。代码仓库 `F:/study/big_model/my_gpt/` 在这台。M8 SFT 真训练能跑。
  - **D 盘另一台(学原理,无独显)**:`D:\ai\nanoGPT_venv`(Python 3.11 + torch 2.13+cpu,Intel 核显)。SFT 真训练跑不动。
  - 接手前先确认学生在哪台(看当前 venv 路径或问)。
- 🚨 **两台是不同电脑,本机资源对方取不到**:给对方 AI 的指令**只写"获取命令 + repo 相对路径"**,绝不假设对方有本机文件。
- 📦 **对方 AI 获取资源的远程命令**(照着敲就行):
  ```bash
  git clone https://github.com/Wuweilong-Leo/nano_gpt-tutorial.git   # 主代码(PLAN/PROGRESS/mini_compiler/lesson*)
  git clone https://github.com/caomaolufei/AIInfraGuide.git           # infra 主线素材(261篇文档)
  git clone https://github.com/tinygrad/tinygrad.git                  # M8.6 编译器参考书
  git clone https://gitcode.com/Ascend/msmodeling.git                # M8.5⑥ runtime 调度仿真教具
  ```
  - Qwen3-0.6B 权重:`from_pretrained("Qwen/Qwen3-0.6B", cache_dir=...)`,国内配 `HF_ENDPOINT=https://hf-mirror.com`
  - venv:`pip install torch transformers datasets peft accelerate bitsandbytes tiktoken`(GPU 版按对方 CUDA 选 torch wheel)

- **当前进行中**:**GPU 硬件补课 · 消化阶段**。已讲透:CPU vs GPU、SM/Tensor Core/CUDA Core 层次(5080=84SM×4TC×128CC)、显存金字塔、Roofline/Memory Wall、wmma 指令(load/mma/store)、block/grid/thread/warp 调度、内存行主序。素材:`AIInfraGuide/docs/guides/模块一-前置知识/gpu/gpu-basics.md`。
- **下一步(等学生发问)**:① 继续补 GPU(warp/SIMT/同步,模块二 2.1-2.4)② GPU 小实验验证 ③ 转 CUDA 编程实战(模块二 1.1 起)。
- **教学规则(学生明确,违反会被纠正)**:
  1. 代码**学生自己写**,老师只给骨架/hint/解释,绝不代写完整可运行代码
  2. 语法/API 细节直接给答案,不猜谜
  3. 不深入公式层,只讲 usage + intuition("太细节了")
  4. 不要求无意义验证("验证了也没啥用")
  5. 一次一小步,不要一次抛太多("一步一步来")
  6. 风格:互动式、多类比、少公式

---

## 1️⃣ 课程总览

| 课 | 主题 | 状态 |
|---|---|---|
| M0-M5 | 基础 + 手写 GPT | ✅ 完成 |
| M6 | 采样/评估/训练工程 | ✅ 完成 |
| M7 | BPE + 中文 token | ✅ 完成 |
| M8 | SFT 指令微调 | 🔄 搁置(step3 切割线函数卡住) |
| M8.5 | 大模型端到端执行流(七站) | ✅ 前传完成(dynamo 96节点 + 融合 96→84) |
| M8.6 | 手写迷你 AI 编译器 | 🔄 ①②③④✅,⑤路线B暂停(先补CUDA再回) |
| **infra 主线** | **GPU+CUDA+编译器+分布式+推理** | 🔄 **走 AIInfraGuide**(见下) |
| M9 | LoRA | ⬜ 待学(不走 AIInfraGuide) |
| M13-M15 | agent | ⬜ 待学(不走 AIInfraGuide) |

### 📚 infra 主线走 AIInfraGuide(2026-08-16 起)

学生拍板混合制:infra 走 AIInfraGuide,SFT/agent 留原 PLAN。原 M10/M11/M12 infra 部分作废。

| 阶段 | AIInfraGuide 章节 | 状态 |
|---|---|---|
| GPU 硬件 | 模块一第5章 gpu-basics.md | 🔄 消化中 |
| CUDA 编程 | 模块二 1.1-2.4(环境/模型/内存/Warp/同步) | ⬜ 下一步候选 |
| CUDA 算子 | 模块二 3.1-6.2(Reduce/GEMM/Softmax/FlashAttn) | ⬜ 待学 |
| AI 编译器 | 模块二第7章(接 M8.6 路线B 对照真 inductor) | ⬜ 补完 CUDA 后回 |
| 分布式训练 | 模块三(通信/数据并行/ZeRO/TP/PP/MoE/3D) | ⬜ 待学 |
| 推理优化 | 模块四(LLM推理/vLLM/量化/SpecDec/PD解耦) | ⬜ 待学 |
| 面经 | docs/interview/(几十家公司真题) | ⬜ 面试前刷 |

---

## 2️⃣ M8 SFT 详细计划(搁置中,恢复时照做)

目标:让 Qwen3-0.6B 学会听指令。5 步走:

| 步 | 内容 | 状态 |
|---|---|---|
| 1 | 加载 Alpaca 数据(48818 条) | ✅ |
| 2 | 格式化 chat template(143 token) | ✅ |
| 3 | tokenize + loss masking | 🔄 卡在切割线定位函数 |
| 4 | SFT 训练循环 | ⬜ |
| 5 | 对比验证 | ⬜ |

**step3 hint**(恢复时给学生):找 `assistant_head = torch.tensor([151644, 77091, 198])`(`<|im_start|>assistant\n`)在 input_ids 里的位置,结束位置 = 切割线,之前全设 -100,之后保留原 id。

**Qwen3 关键 token id**:`151644=<|im_start|>` `151645=<|im_end|>` `77091=assistant` `198=\n` `assistant头三连=[151644,77091,198]`

---

## 3️⃣ M8.6 迷你编译器(①②③④✅,⑤暂停)

代码在 `mini_compiler/`(已推 git):`ir.py`(Op表+Node)/`parse.py`(文本→图)/`passes.py`(DCE+融合 8→3)/`lower_c.py`(图→C循环代码)/`test_prog.txt`。

**⑤ 路线 B 状态**:`lower_c.py` 已跑通(emit 两份 C 代码,看到融合省循环)。**先补 CUDA 再回编译器**(学生定:不懂 CUDA 看不懂 Triton 内核)。补完 CUDA 模块二 1-2 章后,回这步对照真 `torch._inductor` 生成的 Triton 内核。

### 🆕 2026-08-14 ⑤ 追加课:内核 → 调度闭环(今天教完,接上一棒)

今天学生要的是"端到端在机子上怎么跑",我们没走"装 MSVC 真跑 inductor",而是走**纯概念闭环 + 真实验证**,主线如下(全部保留在 `mini_compiler/NOTES_internals.md`):

**教了什么**(一串,不是代码练习,是概念+真代码确认):
1. **内核在代码段的真相**:`_cuda_launch_kernel`(打包参数→找内核)→ `cudaLaunchKernel(grid,block,args,stream)` = 真正入队
2. **申请内存是 CPU 的活,不在内核里**:CPU 先分配缓冲再让 GPU 跑;GPU 只是持地址的哑巴工人
3. **GPU 调度前检查三件事**:流(stream)队尾空没、资源(算/传)桶空没、依赖数据就绪没 → 三者取 max
4. **调度器选流**:贪心挑"结束最早"的流(msmodeling 的 `multistream_pass.py` 就是这么做的)
5. **bubble 气泡**:依赖没 ready 时 GPU 确实会卡,靠 多流/事件/重叠 填 → 这就是 multistream 存在理由
6. **静态 vs 动态 shape**:编译期能定内存大小 vs 运行到那步才现算

**真验证(本机已跑)**:
- `torch.fx.symbolic_trace` 确认 `matmul` = 图里的一个 `fx.Node` ✅
- `torch.cuda.memory_allocated/reserved` = N/A(本机无 GPU,诚实标注)

**修正交接单一处错误**:PLAN 之前写"本机 cpu torch 能跑 inductor codegen(只要 .py 不要 GPU)"——**实测是错的**。本机缺 MSVC `cl.exe`,`@torch.compile` 直接抛 `InvalidCxxCompiler: Compiler: cl is not found`,连生成 `.py` 都到不了(codegen 前就死)。下棒接手别再让学生试"跑 torch.compile 找 Triton"。

**素材**:`D:\ai\_downloads\msmodeling`(昇腾 NPU 性能建模,抽取了 `multistream_pass.py` 的 `_estimate_start_time_s`/`_build_schedule` 讲调度)。它需要者要时再抽,别整读。

**已验证跑通(续)**:
- `cd mini_compiler && python passes.py` → 8→3(④ 融合)
- `cd mini_compiler && python lower_c.py` → 两份 C 代码对照
- `torch.fx` 图节点验证(matmul=fx.Node)✅(上面)
- ⚠️ 不要让学生跑 `torch.compile` 找 Triton(本机缺 cl)

**dispatch table 知识点(别讲拧)**:两种表——codegen 表(编译期用,如 tinygrad `code_for_op`)/ 运行时 dispatch 表(运行时用,如 torch.fx target、PyTorch `ops_table`)。派B(生成代码)有 codegen 表,派A(解释执行)有运行时 dispatch 表。详见 PROGRESS「🖥️ GPU 硬件补课」节。

---

## 4️⃣ 踩坑速查(高频)

1. **离线 env 顺序**:`HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` 必须在 `import transformers/datasets` **之前**设
2. `torch_dtype` 新 transformers 已弃用 → 用 `dtype`
3. `tokenizer(text)` 别调两遍取 `.input_ids`,存一次复用
4. Windows 控制台 GBK 乱码 → `sys.stdout.reconfigure(encoding="utf-8")` 或 UTF-8 运行
5. HF 权重下载:直连被墙,xet 镜像 401 → 用 `from_pretrained(cache_dir=...)` 老接口 + `HF_ENDPOINT=hf-mirror.com`
6. Windows 路径反斜杠 → `open(r"...")` 加 r 前缀
7. vLLM 在 Windows/Blackwell 支持差 → 部署保底用 Ollama

---

## 5️⃣ 素材库

- **AIInfraGuide**(`F:/study/AIInfraGuide/` 或 git clone):infra 主线素材,261 篇文档,模块一(前置)/二(CUDA)/三(分布式)/四(推理)+ 面经
- **tinygrad**(github.com/tinygrad/tinygrad):M8.6 编译器参考书,每文件只读 2-3 函数
- **msmodeling**(gitcode.com/Ascend/msmodeling):华为昇腾推理仿真器,M8.5⑥ runtime 调度教具(算法真执行假,只教概念)。M10 用 `kv_cache_manager.py`(BLOCK_SIZE=128 分页)+ `engine.py` BatchScheduler

---

## 6️⃣ 已讲课程速查(怕忘)

- **M0-M5**:线性回归/梯度下降/逻辑回归/神经网络/激活/反向传播;KQV+Self-Attention、Embedding+位置编码、Block、完整 GPT 类、训练+生成(120M,loss 4.5→2.5)
- **M6**:temperature/top-k/top-p 采样、perplexity、beam search 概念、训练工程速查(AdamW/warmup/AMP 概念遗留 M8)、KV cache(学生提前自学实现)、PagedAttention 概念(+0.5 infra 眼镜)
- **M7**:BPE 原理(byte-level/频次合并)、tiktoken 实操、Qwen3 中文友好(你好=1 token vs GPT-4 的 2 token)
- **M8 Lesson1**:加载 Qwen3-0.6B、验证生成("死循环重复"=SFT 动机)
- **M8.5 前传**:模型交付件四件套、加载 6 站链路、tokenizer 三件套+encode 四步、AI 编译器主线(画图 dynamo 96 节点 → 优化融合 96→84 → 生成 → 运行时调度)
- **M8.6 ①②③④**:Op 表+Node、文本→图、DCE+融合(8→3)、lower_c(图→C 代码)
- **GPU 硬件补课**(2026-08-16):CPU vs GPU、SM/TensorCore/CUDACore 层次、显存金字塔、Roofline、wmma 指令、block/grid/thread/warp 调度、内存行主序(详见 PROGRESS)
