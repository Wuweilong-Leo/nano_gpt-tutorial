import torch

class block(torch.nn.Module):
    def __init__(self, hidden_size, head_num):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)
        self.w_q = torch.nn.Linear(hidden_size, hidden_size)
        self.w_k = torch.nn.Linear(hidden_size, hidden_size)
        self.w_v = torch.nn.Linear(hidden_size, hidden_size)
        self.w_o = torch.nn.Linear(hidden_size, hidden_size)
        self.mlp_fc = torch.nn.Linear(hidden_size, hidden_size * 4)
        self.mlp_act = torch.nn.GELU()
        self.mlp_proj = torch.nn.Linear(hidden_size * 4, hidden_size)
        self.head_num = head_num
        self.hidden_size = hidden_size

    def forward(self, x): # x:(batches_num, tokens_num, hidden_size)
        x_base = x
        x = self.ln1(x) # 归一化
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        batches_num = x.shape[0]
        tokens_num = x.shape[1]
        head_num = self.head_num

        # 切头把token切成head_num个头
        q = q.reshape(batches_num, tokens_num, head_num, -1)
        k = k.reshape(batches_num, tokens_num, head_num, -1)
        v = v.reshape(batches_num, tokens_num, head_num, -1)

        q = q.transpose(1, 2) # (batches_num, head_num, tokens_num, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.transpose(-1, -2) # (batches_num, head_num, head_dim, tokens_num)

        score = q @ k # (batches_num, head_num, tokens_num, tokens_num)

        '''
            对于一个头里score来说:
            score[0][0] = q[0][0] * k[0][0] + q[0][1] * k[1][0] + q[0][2] * k[2][0] ... + q[0][hidden_size / head_num -1] * k[hidden_size / head_num - 1][0]
            score会被加的很大，需要先缩一下, 为啥是按照这个比例缩？
            为什么 √n 是"恰到好处"——统计学直觉

            假设 q 和 k 的每个数是均值 0、方差 1 的随机数：
            - 单个乘积 q[i][d]·k[j][d] 的方差 ≈ 1
            - 64 个这样的乘积相加，方差会累加 → 总方差 ≈ 64
            - 标准差 = √方差 = √64 = 8
            
            也就是说：64 项相加后，数值的"波动幅度"是 √64 = 8 倍，不是 64 倍。
            
            所以要除以 8（标准差），把波动压回 1 倍——这就是"按 √head_dim 缩放"的数学依据。

        '''
        score = score / ((self.hidden_size / head_num) ** 0.5)

        mask = torch.tril(torch.ones(tokens_num, tokens_num, device=x.device)) # 生成一个下三角形矩阵（device=x.device：跟 x 同设备，否则 GPU 上 score 是 cuda、mask 是 cpu 会报错）
        score = score.masked_fill(mask == 0, float('-inf')) # 防止看到未来的输入
        score = torch.softmax(score, dim=-1) # 生成分数占比，表示每个token对其他token的关注度占比

        output = score @ v # 生成融合上下文信息的新值 (batches_num, head_num, tokens_num, head_dim)

        output = output.transpose(1, 2) # (batches_num, tokens_num, head_num, head_dim)
        output = output.reshape(batches_num, tokens_num, -1) # (batches_num, tokens_num, hidden_size)
        output = self.w_o(output) # 各个头的信息再融合一下 (batches_num, tokens_num, hidden_size)

        output = output + x_base # 残差连结

        output_base = output
        output = self.ln2(output) # 残差连结过，要归一化
        output = self.mlp_fc(output) # (batches_num, tokens_num, hidden_size * 4)
        output = self.mlp_act(output) # 高斯误差线性单元，去线性化
        output = self.mlp_proj(output) # 再缩回来
        output = output + output_base # (batches_num, tokens_num, hidden_size)

        return output

# hidden_size = 1024
# head_num = 16
# text = "First Citizen:\n Before we proceed any further, hear me speak."
# chars = sorted(set(text)) # 对text的字符去重加排序，获得字符列表
# vocab_size = len(chars)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
print(f'torch version: {torch.__version__}')
print(f'torch file   : {torch.__file__}')
print(f'cuda build   : {torch.version.cuda}')
import os
print(f'CVD env      : {repr(os.environ.get("CUDA_VISIBLE_DEVICES"))}')
print(f'PYTHONPATH   : {repr(os.environ.get("PYTHONPATH"))}')
print(f'PYTHONHOME   : {repr(os.environ.get("PYTHONHOME"))}')

# wte = torch.nn.Embedding(vocab_size, hidden_size).to(device) # 词嵌入表
# wpe = torch.nn.Embedding(1024, hidden_size).to(device) # 位置表
# params = list(wte.parameters()) + list(wpe.parameters()) #收集参数
#
# blocks = [block(hidden_size, head_num).to(device) for _ in range(6)]
#
# for b in blocks:
#     params += list(b.parameters()) # 收集每层的参数
#
# ln_f = torch.nn.LayerNorm(hidden_size).to(device)
# params += list(ln_f.parameters())
#
# lm_head = torch.nn.Linear(hidden_size, vocab_size).to(device) # 生成每个字符的概率
# params += list(lm_head.parameters())
#
# optimizer = torch.optim.Adam(params, lr=3e-4)
#
# stoi = {c: i for i, c in enumerate(chars)} # 形成字典 字符-》索引
# itos = {i: c for i, c in stoi.items()} # 形成字典，索引=》字符
# data = [stoi[c] for c in text] # 对于字符串里的每个字符形成索引
#
# data_input = data[:-1] # 去掉最后一个字符，形成输入
# exp = data[1:] # 从第一个字符开始，形成输出

# idxs = torch.tensor([data_input]).to(device) # 输入字符 (batches_num, token_num)
# pos = torch.arange(len(data_input)).unsqueeze(0).to(device) # 位置输入 (batches_num, token_num)
# targets = torch.tensor([exp]).to(device) # 预期值
#
# for step in range(100):
#     x = wte(idxs) + wpe(pos) # (batches_num, token_num, hidden_size)
#     for b in blocks: # 走一遍
#         x = b(x)
#     x = ln_f(x) # 归一化
#     x = lm_head(x) # (batches_num, token_num, vocab_size) 每个token下一个的vocab_size字符分别的概率
#
#     loss = torch.nn.functional.cross_entropy(x.view(-1, vocab_size), targets.view(-1)) # 搞成1维的算loss
#
#     optimizer.zero_grad() # 清梯度
#
#     loss.backward() # 算梯度
#
#     optimizer.step() # 更新参数
#
#     if step % 10 == 0:
#         print(f"step: {step}, loss: {loss.item():.4f}")

class gpt(torch.nn.Module):
    def __init__(self, vocab_size, block_size, hidden_size, head_num, n_layers):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, hidden_size).to(device)
        self.wpe = torch.nn.Embedding(block_size, hidden_size).to(device)
        self.blocks = torch.nn.ModuleList([block(hidden_size, head_num).to(device) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(hidden_size).to(device)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size).to(device)
        self.vocab_size = vocab_size

    def forward(self, idx, targets = None):
        batches_num = idx.shape[0]
        tokens_num = idx.shape[1]
        pos = torch.arange(tokens_num).unsqueeze(0).to(device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss
        return logits

with open(r"F:\study\big_model\nanoGPT\data\shakespeare_char\input.txt", "r") as f:
    text = f.read()
block_size = 64
chars = sorted(set(text))
vocab_size = len(chars)
model = gpt(vocab_size, block_size, hidden_size = 1024, head_num = 16, n_layers=12)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
stoi = {c: i for i, c in enumerate(chars)} # 形成字典 字符-》索引
itos = {i: c for i, c in stoi.items()} # 形成字典，索引=》字符
data = [stoi[c] for c in text] # 对于字符串里的每个字符形成索引
for step in range(100000):
    ix = torch.randint(len(data) - block_size, size=(1,))
    idx = torch.tensor(data[ix:ix + block_size]).unsqueeze(0).to(device)
    targets = torch.tensor(data[ix+1:ix + block_size+1]).unsqueeze(0).to(device)
    logits, loss = model(idx, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step: {step}, loss: {loss.item():.4f}")





