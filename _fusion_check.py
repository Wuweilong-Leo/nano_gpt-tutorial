import os, sys
os.environ["HF_DATASETS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import torch, torch._dynamo as dynamo
from transformers import AutoModelForCausalLM
CACHE = r"F:\study\big_model\models\_hf_cache"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", cache_dir=CACHE, dtype=torch.float16, device_map="auto").eval()
layer = model.model.layers[0]
x     = torch.zeros(1, 16, 1024, dtype=torch.float16, device="cuda")
mask  = torch.ones(1, 1, 16, 16, dtype=torch.float16, device="cuda")
pos   = torch.arange(16, device="cuda").unsqueeze(0)
cos, sin = model.model.rotary_emb(x, pos)
res = dynamo.export(layer)(x, mask, pos, None, False, (cos, sin))
nodes = list(res.graph_module.graph.nodes)

# 把所有节点的"名字"提出来(pow / mean / add / rsqrt ...),按出现顺序排成一串
names = []
for n in nodes:
    t = n.target
    if n.op == "call_function" or n.op == "call_method":
        names.append(getattr(t, "__name__", str(t)))
    else:
        names.append(f"[{n.op}]")

# 优化器视角:在这串名字里找连续出现的 pow -> mean -> add -> rsqrt
PATTERN = ["pow", "mean", "add", "rsqrt"]
founds = []
i = 0
while i <= len(names) - len(PATTERN):
    if names[i:i+len(PATTERN)] == PATTERN:
        founds.append(i)
        i += len(PATTERN)
    else:
        i += 1

print("节点总数:", len(nodes))
print("找到 Norm 指纹:", len(founds), "处")
for p in founds:
    print(f"  名字串位置 {p:>3} 起: {names[p:p+7]}")
print(f"\n每个指纹 4 小节点融合成 1 个算子 -> 96 - {len(founds)}*3 = {len(nodes) - len(founds)*3} 个节点")
