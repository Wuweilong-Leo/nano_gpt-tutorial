import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
ds = load_dataset("shibing624/alpaca-zh", cache_dir=r"F:\study\big_model\data\_hf_cache")
ex = ds["train"][0]
print("instruction:", ex["instruction"])
print("input:", ex["input"])
print("output:", ex["output"][:80])

MODEL_PATH = r"F:\study\big_model\models\Qwen3-0.6B"   # tokenizer 在这
CACHE_DIR  = r"F:\study\big_model\models\_hf_cache"     # 权重缓存在这

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen3-0.6B",
      cache_dir=CACHE_DIR,
      torch_dtype=torch.float16,    # fp16,省显存
      device_map="auto",            # 自动放 GPU
  )

text = "你好，今天天气怎么样？"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 1. 把一条 Alpaca 数据拼成一问一答
prompt = ex["instruction"]
if ex["input"]:
    prompt += "\n" + ex["input"]   # 附加材料不为空时,跟在问题后面
print("prompt:", prompt)
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": ex["output"]},
]
print("messages:", messages)

# 2. 让 tokenizer 把它打扮成 Qwen3 认识的聊天格式
text = tokenizer.apply_chat_template(messages, tokenize=False)

print("--------------text start----------------")
print(text)
print("--------------text end------------------")
print(len(tokenizer(text).input_ids))
print(tokenizer(text).input_ids)