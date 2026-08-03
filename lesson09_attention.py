import torch

hidden_size = 768
heads_num = 12

w_q = torch.nn.Linear(hidden_size, hidden_size)
w_k = torch.nn.Linear(hidden_size, hidden_size)
w_v = torch.nn.Linear(hidden_size, hidden_size)

idxs = torch.tensor([[1, 3, 4, 5, 6]])
batches_num = idxs.shape[0]
tokens_num = idxs.shape[1]
emb = torch.nn.Embedding(65, hidden_size)
x = emb(idxs)

pos = torch.arange(tokens_num).unsqueeze(0)
pos_tab = torch.nn.Embedding(1024, hidden_size)
x = x + pos_tab(pos)
# x = torch.randn(1, 5, 768) # 1个句子，5个token，768个属性

x_base = x
ln1 = torch.nn.LayerNorm(hidden_size)
x = ln1(x)

q = w_q(x)
k = w_k(x)
v = w_v(x)

q = q.reshape(batches_num, tokens_num, heads_num, -1)
k = k.reshape(batches_num, tokens_num, heads_num, -1)
v = v.reshape(batches_num, tokens_num, heads_num, -1)

q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2) # (1, 12, 5, 64)

k = k.transpose(-1, -2)

score = q @ k
score = score / ((hidden_size / heads_num) ** 0.5)
mask = torch.tril(torch.ones(tokens_num, tokens_num))
score = score.masked_fill(mask == 0, float('-inf'))
score = torch.softmax(score, dim=-1) # (1, 12, 5, 5)

output = score @ v # (1, 12, 5, 64) 表示每个token的融合了上下文后的新值
output = output.transpose(1, 2)
output = output.reshape(batches_num, tokens_num, hidden_size)

w_o = torch.nn.Linear(hidden_size, hidden_size)
output = w_o(output)
output = output + x_base

print(output)

