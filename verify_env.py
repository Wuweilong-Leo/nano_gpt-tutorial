# -*- coding: utf-8 -*-
# 环境自检脚本(对齐原 ENV.md verify_env.py 的思路)
# 用法: PYTHONUTF8=1 python verify_env.py
import sys

def line(ok, msg):
    print(f"[{'OK ' if ok else 'FAIL'}] {msg}")
    return ok

print("=" * 50)
print("环境自检")
print("=" * 50)

all_ok = True

# 1. python 版本
import sys
all_ok &= line(sys.version_info >= (3, 11), f"python {sys.version.split()[0]}")
print(f"     路径: {sys.executable}")

# 2. torch
try:
    import torch
    all_ok &= line(True, f"torch {torch.__version__}")
    all_ok &= line(not torch.cuda.is_available(), f"cuda 可用: {torch.cuda.is_available()} (CPU 模式, 预期为 False)")
    # 极小运算测试
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x @ y
    all_ok &= line(z.shape == (3, 3), f"torch 矩阵乘法测试: {z.shape}")
except Exception as e:
    all_ok &= line(False, f"torch import 失败: {e}")

# 3. numpy
try:
    import numpy as np
    all_ok &= line(True, f"numpy {np.__version__}")
except Exception as e:
    all_ok &= line(False, f"numpy 失败: {e}")

# 4. transformers
try:
    import transformers
    all_ok &= line(True, f"transformers {transformers.__version__}")
except Exception as e:
    all_ok &= line(False, f"transformers 失败: {e}")

# 5. tiktoken (M7 BPE)
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    n = len(enc.encode("hello"))
    all_ok &= line(n == 1, f"tiktoken {tiktoken.__version__} (encode 'hello' -> {n} tokens)")
except Exception as e:
    all_ok &= line(False, f"tiktoken 失败: {e}")

# 6. tqdm
try:
    import tqdm
    all_ok &= line(True, f"tqdm {tqdm.__version__}")
except Exception as e:
    all_ok &= line(False, f"tqdm 失败: {e}")

# 7. requests
try:
    import requests
    all_ok &= line(True, f"requests {requests.__version__}")
except Exception as e:
    all_ok &= line(False, f"requests 失败: {e}")

print("=" * 50)
print("全部通过!" if all_ok else "有项目失败, 见上")
print("=" * 50)
sys.exit(0 if all_ok else 1)
