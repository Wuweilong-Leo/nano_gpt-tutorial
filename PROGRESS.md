# 学习进度(详细档案)

> **接手先读 `PLAN.md`**(30 秒入口)。本文件是详细历史档案,供复习/回溯。
> **最后更新:2026-08-16**

---

## 📍 当前位置

- **已完成**:M0-M7(基础+手写 GPT+采样评估+BPE)
- **进行中**:
  - **infra 主线**(走 AIInfraGuide):GPU 硬件补课 · 消化阶段
  - M8.6 迷你编译器 ①②③④✅,⑤路线B暂停(先补 CUDA 再回)
  - M8 SFT 搁置(step3 切割线函数)
- **下一步**:等学生发问。候选:① 继续补 GPU(warp/SIMT/同步)② GPU 小实验 ③ 转 CUDA 编程实战(模块二 1.1)

---

## 📊 进度仪表盘

```
原里程碑:7/15 完成(M0-M7 ✅)
infra 主线:切 AIInfraGuide,GPU 硬件补课中
三条目标线:A 通晓大模型 / B 通晓 agent / C 通晓 AI infra
```

---

## ✅ 已完成课程

### M0-M5 基础 + 手写 GPT(2026-08-02~08-05)
- 线性回归/梯度下降/逻辑回归/神经网络/激活函数/反向传播(第 1-7 课)
- KQV/Self-Attention、Embedding+位置编码、Block(attention+MLP+2LN+2残差)、完整 GPT 类(`class GPT` nn.Module)、训练+生成
- 配置:12层×1024 hidden×12 head(~120M 参数),GPU 跑通 10万步 loss 4.5→2.5
- 学生实现:KV cache 推理加速(提前自学)、Dropout、weight tying、CosineAnnealingLR、梯度裁剪、sample 生成循环
- 代码:`lesson09_attention.py`(学生自己的 GPT,含采样/评估/KV cache)

### M6 采样+评估+训练工程(2026-08-09 收官)
- temperature/top-k/top-p 采样(手写 pipeline + 封装,三温度对照实验)
- perplexity 困惑度(val_loss 2.41 → val_ppl 11.18)
- beam search 理论(确定性解码,大模型时代被采样取代)
- 训练工程速查:AdamW/warmup+cosine/CE vs MSE/LN vs BN/Pre-LN/GELU vs SwiGLU/残差(概念层,公式不深挖)
- +0.5 infra 眼镜:朴素 KV cache → 碎片问题 → PagedAttention 分页(block table ~ OS 虚拟内存)→ 对照 msmodeling kv_cache_manager
- **M6 欠账**(不紧急,面试会考):AMP 三组对比(fp32/fp16+GradScaler/bf16)、gradient checkpointing、torch.compile、profiling

### M7 BPE + 中文 token(2026-08-09 收官)
- BPE 原理(byte-level、频次合并)、tiktoken 实操
- `hello`=1 token、`unhappiness`=3 token、`你好`=2 token(GPT-4)
- **中文税实测**:`Hello, how are you?`=6 token vs `你好,今天天气怎么样?`=13 token,中文贵 2 倍
- Qwen3 中文友好:`你好`=1 token(Qwen3) vs 2 token(GPT-4),省一半
- 代码:`lesson12_bpe.py`

### M8 Lesson1(2026-08-09)
- 加载 Qwen3-0.6B 权重、验证生成("死循环重复"症状 = SFT 动机)、fp16/device_map="auto"

### M8.5 前传(2026-08-12 完成)
- 模型交付件四件套(config/tokenizer三件套/generation_config/safetensors)
- 加载 6 站链路:登记簿 modeling_auto → 查表 auto_factory → import modeling_qwen3 → 搭壳 → 填 311 权重
- tokenizer 三件套职责 + encode 四步(词表=训练产物,推理只查不建)
- AI 编译器主线:画图(dynamo.export,Qwen 一层 96 节点)→ 优化(模式匹配 4 处 Norm 指纹 96→84)→ 生成 → 运行时调度
- 教具:`ai_compiler_test.py`(96 节点)、`_fusion_check.py`(96→84),已跑通推 git

### M8.6 迷你编译器 ①②③④(2026-08-12~14)
- ① 摸地形:tinygrad UOp 浏览
- ② 定 IR:`ir.py`(10 个 Op + Node 三件套),手工造 RMSNorm 指纹打印
- ③ 写前端:`parse.py`(文本→图,8 节点)。踩两个真实编译器坑:IR 必须有显式出口(否则 DCE 把孤儿当出口)、前端登记语义与后端契约对齐
- ④ 优化管线:`passes.py` 的 `fuse` 8→3 稳定跑通(两遍结构:第一遍认指纹记 dead/repl,第二遍搬图换 RMSNORM)。`remove_dead_nodes` 学生自己写的。踩坑:单遍循环删不掉中间节点(要先认后删,用名字不用 op 类型)
- ⑤ 路线B:`lower_c.py` 已写完跑通(图→C 循环代码,8节点 emit 3 循环 vs 3节点 emit 2 循环,看到融合省循环)。**暂停,先补 CUDA**
- 代码:`mini_compiler/`(ir/parse/passes/lower_c/test_prog),已推 git

---

## 🔄 进行中:GPU 硬件补课(2026-08-16)

**起因**:学生从别的 AI 听到"dispatch table 是编译器编出来留运行时用",质疑我之前讲的"派 B 没 table"。纠正后学生想看真代码里的表 → 转 GPU 硬件(不懂硬件看不懂内核)。

**素材**:`AIInfraGuide/docs/guides/模块一-前置知识/gpu/gpu-basics.md`(493 行)

**已讲透的概念**(学生已懂,不用再讲):

- **CPU vs GPU**:CPU 法拉利(少核聪明/延迟优化),GPU 大巴车队(多核笨/吞吐优化)。深度学习=海量重复乘加=撞 GPU 枪口
- **GPU 指令四类**:整数(地址)/浮点(标量 FFMA)/Tensor Core 矩阵乘加(MMA)/访存同步
- **硬件层次(5080)**:GPU → 84 个 SM → 每 SM 4 个 Tensor Core + 128 个 CUDA Core + ~100KB 共享内存 + ~256KB 寄存器。Tensor Core 共 336,CUDA Core 共 10752(84×128)
- **Tensor Core**:一条指令算一小块矩阵乘加 D=A×B+C(如 16×16×16),比 CUDA Core 快几百倍。精度档 fp64/tf32/fp16/bf16/fp8/int8/int4,精度低=快但易丢精度(量化本质)。5080 第4代支持 fp8,无 fp64
- **数据流三跳**:HBM(显存)→共享内存(SRAM)→寄存器→Tensor Core。Tensor Core **只吃寄存器**(物理够不着显存)。共享内存是中转池,让 warp 协作搬运+复用
- **显存金字塔**:HBM(16GB/~500GB/s/几百拍)→ L2(几十MB/~2TB/s)→ 共享内存(每SM~100KB/~10TB/s)→ 寄存器(每线程~256 float/~30TB/s/1拍)
- **Roofline / Memory Wall**:算力 vs 带宽谁先撞墙。算少搬多(element-wise)→ 带宽先撞;算多搬少(matmul)→ 算力先撞。大模型推理慢常因喂不饱 Tensor Core
- **wmma 指令**:`wmma::fragment`(模板类,声明装 tile 的寄存器组,编译器决定放寄存器)→ `load_matrix_sync`(显存→寄存器,三参数:目标/源地址/leading dimension)→ `mma_sync`(触发 Tensor Core 算 D=A×B+C)→ `store_matrix_sync`(寄存器→显存)。先 load 再 mma,不能跳
- **切块 tiling**:大矩阵塞不进寄存器,切 16×16 小块循环算。沿 K 方向切多块累加(+C 是累加器)。**切块是软件干的**(cuBLAS/cutlass/inductor/自己写),硬件只算拿到的小块
- **block/grid/thread/warp**:
  - thread=最小执行单位(一个工人跑一份 kernel 副本,用 threadIdx 编号算不同数据)
  - 32 thread 捆成 warp(硬件调度最小单位,SIMT)
  - block=一组 thread 归一个 SM(不跨 SM,能共享内存/同步),一 SM 能同时塞多个 block
  - grid=所有 block 的网格,启动时定 `<<<grid, block>>>`
  - **warp 切换完全硬件自动**,程序员控不了,只能控"有多少 warp 可切"(block 大小/资源用量)。切换零开销(每 warp 寄存器物理独立)。唯一软件干预是 `__syncthreads()`
  - **软件感知 block/grid 不是为控制调度,是为表达并行结构**:block 划定数据分块+共享内存范围+同步范围,grid 表达总并行规模
- **5080 每 SM 最多 1536 thread**,整卡 84×1536≈13万 thread。塞满=占用率高=性能好(Occupancy)
- **内存行主序**:矩阵在内存一维铺开,一行一行存。地址公式 `A[i][j] = i×每行长度 + j`。同行连续(快)、跨行隔开(慢)。PyTorch/CUDA 行主序。leading dimension=每行长度=换行跳多远

**dispatch table 知识点(值钱,别讲拧)**:
- **两种表都叫 dispatch table,归属不同**:
  - codegen 表(编译期用):tinygrad `code_for_op = {Ops.ADD: lambda...}`,编译器查它拼代码
  - 运行时 dispatch 表(运行时用):torch.fx 节点 target / PyTorch `ops_table = {(op,device): impl}`,编译器编出来留 runtime 查
- 派 B(生成代码,如 tinygrad/inductor)有 codegen 表,运行时不查表;派 A(解释执行,如 torch.fx)有运行时 dispatch 表
- inductor 的"融合算子表"不是静态字典,是编译期生成的 wrapper.py 里一堆 `@triton.jit` 内核函数 + 调用序列。承载三处:①磁盘 .py 文件 ②内存 Python 模块(sys.modules,`compile_tasks.py:_reload_python_module` 的 exec)③Triton 二进制缓存(~/.triton/cache 的 cubin/ptx)

---

## 🔜 待学

### infra 主线(走 AIInfraGuide)
- CUDA 编程:模块二 1.1-2.4(环境/编程模型/内存模型/第一个 kernel/Warp/内存访问/Occupancy/同步)
- CUDA 算子:模块二 3.1-6.2(Reduce/GEMM/Softmax/FlashAttention V1V2)
- AI 编译器:模块二第7章(接 M8.6 路线B,对照真 inductor Triton)
- 分布式训练:模块三(通信原语/数据并行/ZeRO/张量并行/流水线/MoE/3D并行)
- 推理优化:模块四(LLM推理基础/推理引擎/vLLM/量化/SpecDec/PD解耦/生产部署)
- 面经:docs/interview/(几十家公司真题,面试前刷)

### 原 PLAN 保留(不走 AIInfraGuide)
- M8 SFT step3-5(恢复时照 PLAN §2 做)
- M9 LoRA(Qwen3-0.6B + peft/trl 从0写)
- M13-M15 agent(四零件/ReAct/RAG/框架对照,毕业项目)
- M6 欠账:AMP 三组对比、gradient checkpointing、torch.compile、profiling

---

## 🖥️ 环境与资产

### 两台机器并存
- **F 盘台式(主力,有独显)**:`F:\study\big_model\nanoGPT_venv`(Python 3.14.3 + torch 2.11.0+cu128 + RTX 5080 16GB + transformers 5.14)。代码仓库 `F:/study/big_model/my_gpt/`。SFT 真训练能跑。
- **D 盘另一台(学原理,无独显)**:`D:\ai\nanoGPT_venv`(Python 3.11.9 + torch 2.13.0+cpu,Intel 核显)。SFT 真训练跑不动。
- 🚨 **两台不同电脑,本机资源对方取不到**。对方 AI 获取资源走远程命令(见 PLAN 0️⃣ 区📦清单)。

### 本机路径(F 盘台式,对方需重新 clone/下载)
- 代码仓库:`F:/study/big_model/my_gpt/`(remote: github.com/Wuweilong-Leo/nano_gpt-tutorial)
- 主线代码:`lesson09_attention.py`(学生 GPT,含采样/评估/KV cache)
- SFT 代码:`lesson13_sft.py`(进行中)
- BPE 代码:`lesson12_bpe.py`
- mini_compiler:`my_gpt/mini_compiler/`(ir/parse/passes/lower_c/test_prog)
- AIInfraGuide:`F:/study/AIInfraGuide/`
- tinygrad:`F:/study/big_model/tinygrad/`
- msmodeling:`F:/study/big_model/msmodeling/`
- Qwen3 tokenizer:`F:/study/big_model/models/Qwen3-0.6B/`(~14MB)
- Qwen3 权重缓存:`F:/study/big_model/models/_hf_cache/`(596M 参数,fp16 1.5GB)
- Alpaca 数据:`F:/study/big_model/data/_hf_cache/shibing624___alpaca-zh/`(48818 条)

---

## ⚠️ 踩坑档案(下棒别重蹈)

1. **离线 env 顺序**:先设 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` 再 import,否则联网超时
2. **VM 定位别讲拧**:VM 是字节码派后端(Crafting Interpreters 同级),lowering 是教学正餐不跳,真正略过的是"指令→机器码"
3. **dispatch table 两种别混**:codegen 表(编译期)/ 运行时 dispatch 表(运行时),派A派B归属不同。别说"派B没table"(有 codegen 表)
4. **调研 agent 别派子 agent 太多**:撑爆 100万 token,没结论
5. **方案别老定不来**:给推荐 + 逼学生一句话拍板,别开放性反复询问
6. **两台机器别搞混**:F 盘有独显主力 + D 盘无独显学原理,接手先确认在哪台
7. **本机资源对方取不到**:给对方 AI 只写"获取命令+repo 相对路径",远程命令清单见 PLAN 0️⃣📦

---

## 📌 备忘

- 教学规范:互动式、少公式、多类比;代码学生自己写,AI 不代写
- 学生偏好:语法/API 细节直接给;概念/设计问题引导;不为验证而验证
- 面试口径:"手写过迷你 AI 编译器:SSA IR、fusion/DCE、图→C 代码 lowering,接过 dynamo 导出的真模型图(96→84)"
