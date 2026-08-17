"""演示: torch.fx 把 model(骨架) 变成"执行表"(图) 长什么样"""
import torch
from torch.fx import symbolic_trace


class Net(torch.nn.Module):
    def forward(self, x):
        a = x @ torch.eye(4)      # matmul: x(·,4) x (4,4) -> (·,4)
        b = torch.relu(a)         # relu
        y = b @ torch.eye(4).T    # matmul
        return y


net = Net()
x = torch.randn(3, 4)

# 把骨架(forward 代码) "画" 成图
gm = torch.fx.GraphModule(net, symbolic_trace(net).graph)

print("== 执行表(图上每一个 node 就是一行) ==")
print(gm.graph)