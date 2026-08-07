import torch

class Block(torch.nn.Module):
    def __init__(self, hidden_size, n_head):
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
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.drop = torch.nn.Dropout(p=0.1)

    def forward(self, x, kv_cache=None): # x:(B, T, hidden_size)
        x_base = x
        x = self.ln1(x) # 归一化
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        B, T = x.shape[0], x.shape[1]
        n_head = self.n_head

        # 切头把token切成n_head个头
        q = q.reshape(B, T, n_head, -1)
        k = k.reshape(B, T, n_head, -1)
        v = v.reshape(B, T, n_head, -1)

        q = q.transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_kv_cache = (k,v)

        k = k.transpose(-1, -2) # (B, n_head, head_dim, T)

        score = q @ k # (B, n_head, T, T)

        '''
            对于一个头里score来说:
            score[0][0] = q[0][0] * k[0][0] + q[0][1] * k[1][0] + q[0][2] * k[2][0] ... + q[0][hidden_size / n_head -1] * k[hidden_size / n_head - 1][0]
            score会被加的很大，需要先缩一下, 为啥是按照这个比例缩？
            为什么 √n 是"恰到好处"——统计学直觉

            假设 q 和 k 的每个数是均值 0、方差 1 的随机数：
            - 单个乘积 q[i][d]·k[j][d] 的方差 ≈ 1
            - 64 个这样的乘积相加，方差会累加 → 总方差 ≈ 64
            - 标准差 = √方差 = √64 = 8

            也就是说：64 项相加后，数值的"波动幅度"是 √64 = 8 倍，不是 64 倍。

            所以要除以 8（标准差），把波动压回 1 倍——这就是"按 √head_dim 缩放"的数学依据。

        '''
        score = score / ((self.hidden_size / n_head) ** 0.5)
        if kv_cache is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)) # 生成一个下三角形矩阵（device=x.device：跟 x 同设备，否则 GPU 上 score 是 cuda、mask 是 cpu 会报错）
            score = score.masked_fill(mask == 0, float('-inf')) # 防止看到未来的输入
        score = torch.softmax(score, dim=-1) # 生成分数占比，表示每个token对其他token的关注度占比
        score = self.drop(score) # 让矩阵每个元素又10%的概率为0，防止过拟合
        output = score @ v # 生成融合上下文信息的新值 (B, n_head, T, head_dim)

        output = output.transpose(1, 2) # (B, T, n_head, head_dim)
        output = output.reshape(B, T, -1) # (B, T, hidden_size)
        output = self.w_o(output) # 各个头的信息再融合一下 (B, T, hidden_size)

        output = output + x_base # 残差连结

        # MLP 多层感知机
        output_base = output
        output = self.ln2(output) # 残差连结过，要归一化
        output = self.mlp_fc(output) # (B, T, hidden_size * 4)
        output = self.mlp_act(output) # 高斯误差线性单元，去线性化
        output = self.mlp_proj(output) # 再缩回来
        output = self.drop(output)
        output = output + output_base # (B, T, hidden_size)

        return output, new_kv_cache

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {DEVICE}')

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size, hidden_size, n_head, n_layers):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, hidden_size).to(DEVICE)
        self.wpe = torch.nn.Embedding(block_size, hidden_size).to(DEVICE)
        self.blocks = torch.nn.ModuleList([Block(hidden_size, n_head).to(DEVICE) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(hidden_size).to(DEVICE)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size).to(DEVICE)
        self.lm_head.weight = self.wte.weight # weight type, 单位长度下向量点积越大，相似读越高
        self.vocab_size = vocab_size

    def forward(self, idx, targets = None, kv_cache=None, pos_val=None):
        B, T = idx.shape[0], idx.shape[1]
        if pos_val is None:
            pos = torch.arange(T).unsqueeze(0).to(DEVICE) # (1, T)：靠广播对齐 batch 维
        else:
            pos = torch.tensor([[pos_val]]).to(DEVICE)
        x = self.wte(idx) + self.wpe(pos)
        kv_caches = []
        for i,b in enumerate(self.blocks):
            kv_cache_tmp = None
            if kv_cache is not None:
                kv_cache_tmp = kv_cache[i]
            x, new_kv_cache = b(x, kv_cache=kv_cache_tmp)
            kv_caches.append(new_kv_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss
        return logits, kv_caches

with open(r"F:\study\big_model\nanoGPT\data\shakespeare_char\input.txt", "r") as f:
    text = f.read()
BLOCK_SIZE = 128
chars = sorted(set(text))
VOCAB_SIZE = len(chars)
model = GPT(VOCAB_SIZE, BLOCK_SIZE, hidden_size=1024, n_head=16, n_layers=12)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
stoi = {c: i for i, c in enumerate(chars)} # 形成字典 字符-》索引
itos = {i: c for c, i in stoi.items()} # 形成字典，索引=》字符（注意 stoi 是 字符→索引，反转时解包成 c, i）
data = [stoi[c] for c in text] # 对于字符串里的每个字符形成索引
n = int (0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10000)

# 训练
for step in range(100):
    ix = torch.randint(len(train_data) - BLOCK_SIZE, size=(1,))
    idx = torch.tensor(train_data[ix:ix + BLOCK_SIZE]).unsqueeze(0).to(DEVICE)
    targets = torch.tensor(train_data[ix+1:ix + BLOCK_SIZE+1]).unsqueeze(0).to(DEVICE)
    logits, loss = model(idx, targets, pos_val=None)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 防止梯度爆炸，梯度和如果高于1，就等比例缩小
    optimizer.step()
    scheduler.step()
    if step % 10 == 0:
        ix = torch.randint(len(val_data) - BLOCK_SIZE, size=(1,))
        idx = torch.tensor(val_data[ix:ix + BLOCK_SIZE]).unsqueeze(0).to(DEVICE)
        targets = torch.tensor(val_data[ix + 1:ix + BLOCK_SIZE + 1]).unsqueeze(0).to(DEVICE)
        _, val_loss = model(idx, targets)
        print(f"step: {step}, train_loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}")

# 推理
model_str = torch.tensor([[stoi["F"]]]).to(DEVICE) # (batch_size, seq_len)
model.eval()
with torch.no_grad():
    kv_cache = None
    for step in range(100):
        if step == 0:
            input_char = model_str
        else:
            input_char = next_char
        logits, new_kv_cache = model(input_char, kv_cache=kv_cache, pos_val=step)
        kv_cache = new_kv_cache
        logits = logits[:,-1,:] # 取最后一个token的评分 (batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1) # 获得概率 (batch_size, vocab_size)
        next_char = torch.multinomial(probs, 1) # 按概率抽签，1表示抽1个 (batch_size, 1)
        model_str = torch.cat([model_str, next_char], dim=1) # next_char = (1,1)

ret = ""
for i in range(model_str.shape[1]):
    ret += itos[model_str[0, i].item()]
print(ret)





