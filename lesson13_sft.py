from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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