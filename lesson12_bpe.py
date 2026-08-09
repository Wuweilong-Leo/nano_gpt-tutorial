import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
print(len(enc.encode("Hello, how are you?")))         # 英文
print(len(enc.encode("你好,今天天气怎么样?")))          # 中文,意思差不多
