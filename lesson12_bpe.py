import tiktoken
from transformers import AutoTokenizer

enc = tiktoken.get_encoding("cl100k_base")
print(len(enc.encode("Hello, how are you?")))         # 英文
print(len(enc.encode("你好,今天天气怎么样?")))          # 中文,意思差不多

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(len(tokenizer("你好，今天天气怎么样？").input_ids))