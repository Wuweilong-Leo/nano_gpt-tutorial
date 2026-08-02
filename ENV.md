# 环境配置

## 虚拟环境
- **路径**：`F:\study\big_model\nanoGPT_venv\`
- **Python**：3.14.3
- **用途**：nanoGPT 实战专用（M1-M7），与基础课 venv 隔离

## 激活方式（Git Bash）
```bash
source /f/study/big_model/nanoGPT_venv/Scripts/activate
```
或直接用绝对路径调用（不用激活）：
```bash
/f/study/big_model/nanoGPT_venv/Scripts/python.exe 你的脚本.py
```

## ⚠️ 编码坑（必读，否则中文/emoji 报错）
Windows 终端默认 GBK 编码，输出中文或 emoji 会报 `UnicodeEncodeError`。
**所有脚本运行时必须加 UTF-8 模式**：
```bash
PYTHONUTF8=1 /f/study/big_model/nanoGPT_venv/Scripts/python.exe 你的脚本.py
```

## 已装库
| 库 | 版本 | 用途 |
|----|------|------|
| torch | 2.13.0+cpu | 核心（CPU版，无GPU） |
| torchvision | 0.28.0+cpu | 图像处理（备用） |
| numpy | 2.4.4 | 数值计算 |
| requests | 2.34.2 | prepare.py 下载数据 |
| tqdm | 4.70.0 | 训练进度条 |
| tiktoken | 0.13.0 | GPT-2 BPE tokenizer（M7中文备用） |
| transformers | 5.14.1 | 备用 |
| wandb | 0.28.1 | 训练日志（可选） |

## 未装（按计划跳过）
- **datasets**：openwebtext 数据集专用，shakespeare_char 主线用不到。M7 如需再装。

## 关键路径
| 路径 | 说明 |
|------|------|
| `F:\study\big_model\nanoGPT\` | nanoGPT 仓库代码（官方对照） |
| `…\nanoGPT\model.py` | GPT 模型定义（核心，M5 逐行读懂） |
| `…\nanoGPT\train.py` | 训练脚本 |
| `…\nanoGPT\sample.py` | 生成脚本 |
| `…\nanoGPT\data\shakespeare_char\prepare.py` | 莎士比亚数据准备（已跑，生成 train.bin/val.bin） |
| `F:\study\big_model\my_gpt\` | 本仓库，学生手写代码 |
| `F:\study\big_model\verify_env.py` | 环境验证脚本（8项全过） |

## 数据准备（已完成）
```bash
cd /f/study/big_model/nanoGPT/data/shakespeare_char
PYTHONUTF8=1 /f/study/big_model/nanoGPT_venv/Scripts/python.exe prepare.py
```
结果：
- 1,115,394 字符（莎士比亚全集）
- 65 个独特字符（词表 vocab）
- train.bin（100万 tokens）/ val.bin（11万 tokens）/ meta.pkl（编解码器）

## CPU 性能提醒
- **无 GPU，训练慢**。shakespeare_char 字符级模型小，能跑但必须降参数
- 官方 CPU 命令（README 提供，约 3 分钟）：
  ```bash
  python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
    --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
    --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
  ```
- 每课训练控制在 5 分钟内
