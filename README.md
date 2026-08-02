# nano_gpt-tutorial 🎓

从 0 搭建 GPT 的小白实战教程，配套 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 项目。

> "每个知识点 → 在 nanoGPT 里找到对应代码 → 我们自己的项目往前长一层。
> 一节课 = 一个知识点 + 项目推进一格。"

## 这是什么

一个"知识点驱动 + 项目跟着长"的大模型入门课程。每节课学一个核心知识点，在 `my_gpt/` 里**亲手从 0 写**对应代码（官方 nanoGPT 只作"标准答案"对照）。每节课多一个 `lessonXX_xxx.py` 文件，最后长成完整 GPT，跑出莎士比亚。

## 目录结构

```
F:\study\big_model\
├── nanoGPT\              ← 官方仓库（只读对照，自己 clone）
├── nanoGPT_venv\         ← Python 3.14 + PyTorch CPU 环境
└── my_gpt\               ← 本仓库（你自己的 GPT，从 0 写）
    ├── README.md         ← 你在这里
    ├── PROGRESS.md       ← 学习进度（下次接着这继续）
    ├── ENV.md            ← 环境配置说明
    ├── TEACHING.md       ← 教学规范（给 AI 助手接手用）
    └── lesson09_attention.py   ← 第 9 课手写代码
```

## 整体教学计划（7 里程碑）

| 课次 | 知识点 | nanoGPT 对应 | my_gpt/ 产出 | 状态 |
|------|--------|-------------|-------------|------|
| 9 | **KQV / Self-Attention** | `CausalSelfAttention` | `lesson09_attention.py` | ✅ 完成 |
| 10 | Embedding + 位置编码 | `wte` / `wpe` | `lesson10_embedding.py` | 🔜 |
| 11 | Causal Mask + 多头 | `mask` / `n_head` | `lesson11_mask_heads.py` | ⏳ |
| 12 | Transformer Block（MLP/LN/残差） | `Block` | `lesson12_block.py` | ⏳ |
| 13 | 堆叠 = 完整 GPT | `GPT` | `lesson13_model.py` | ⏳ |
| 14 | 训练循环 / loss | `train.py` | `lesson14_train.py` | ⏳ |
| 15 | 生成与采样 | `sample.py` | `lesson15_sample.py` | ⏳ |
| 毕业 | 换中文语料 | — | 中文生成 | 🎉 |

**详细路线见 PROGRESS.md。**

## 环境快速开始

见 [ENV.md](ENV.md)。简言之：
```bash
# 激活环境
source /f/study/big_model/nanoGPT_venv/Scripts/activate
# 跑某节课的代码（注意 UTF-8 模式，Windows 防 GBK 报错）
PYTHONUTF8=1 python lesson09_attention.py
```

## 给下次接手的 AI 助手

读 [TEACHING.md](TEACHING.md) 了解教学规范（5 种互动模式、类比库、踩坑预警），
读 [PROGRESS.md](PROGRESS.md) 了解当前进度和下次该讲什么。**核心原则：代码让学生自己写，AI 只解释不代写。**
