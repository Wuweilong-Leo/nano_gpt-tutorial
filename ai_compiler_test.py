import os
import sys
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台防乱码

import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM

CACHE = r"F:\study\big_model\models\_hf_cache"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", cache_dir=CACHE,
    dtype=torch.float16, device_map="auto",
).eval()

layer = model.model.layers[0]      # 28 层里的第 1 层骨架(attention + mlp)

print("======== layer 结构 ========")
print(layer)

# 构造一层前向需要的全部输入(Qwen3 层的 position_embeddings 要外部传入)
x   = torch.zeros(1, 16, 1024, dtype=torch.float16, device="cuda")
mask = torch.ones(1, 1, 16, 16, dtype=torch.float16, device="cuda")
pos  = torch.arange(16, device="cuda").unsqueeze(0)
cos, sin = model.model.rotary_emb(x, pos)   # 旋转位置编码(你 M11 要学的 RoPE)

print("======== 画成计算图(dynamo.export) ========")
res = dynamo.export(layer)(x, mask, pos, None, False, (cos, sin))
nodes = list(res.graph_module.graph.nodes)
print("节点总数:", len(nodes))
print("----- 前 40 个节点 -----")
for n in nodes[:40]:
    print(f"{n.op:<14} | {n.target}")

print("----- 算子频次统计(被分解成的基本动作) -----")
from collections import Counter
c = Counter(str(n.target) for n in nodes if n.op == "call_function")
for name, cnt in c.most_common():
    print(f"{cnt:>3}  {name}")