import torch

w_q = torch.nn.Linear(768, 768)
w_k = torch.nn.Linear(768, 768)
w_v = torch.nn.Linear(768, 768)

x = torch.randn(1, 5, 768) # 1个句子，5个token，768个属性

q = w_q(x)
k = w_k(x)
v = w_v(x)

q = q.reshape(1, 5, 12, -1)
k = k.reshape(1, 5, 12, -1)
v = v.reshape(1, 5, 12, -1)

q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2) # (1, 12, 5, 64)

k = k.transpose(-1, -2)

score = q @ k
score = score / ((768 / 12) ** 0.5)
score = torch.softmax(score, dim=-1) # (1, 12, 5, 5)

output = score @ v # (1, 12, 5, 64) 表示每个token的融合了上下文后的新值
output = output.transpose(1, 2)
output = output.reshape(1, 5, 768)

w_o = torch.nn.Linear(768, 768)
output = w_o(output)

print(output)

