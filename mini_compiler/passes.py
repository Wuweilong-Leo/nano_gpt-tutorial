# passes.py: 优化管线 —— 给图"瘦身"
#   [全空,你自己写] 优化 = 对图做手脚,但每一步都保证数学结果不变
#   pass = 把整张图过一遍,按一套规则改写

import sys
from pathlib import Path
from ir import Node, Op
from parse import parse

sys.stdout.reconfigure(encoding="utf-8")

# ===== 你来写 =====
# 目标:优化前 8 个节点 → 优化后 3 个节点(test_prog.txt 上的 RMSNorm 指纹)
# 验收:跑本文件,贴出"优化前 / 优化后"输出,只剩 INPUT x、INPUT w、RMSNORM(x,w)
if __name__ == "__main__":
    src = Path(__file__).parent / "test_prog.txt"
    nodes, outs = parse(src.read_text(encoding="utf-8"))

    print("====== 优化前:", len(nodes), "个节点 ======")
    for n in nodes.values():
        print("  ", n)

    opt = None          # ← 你的 optimize(nodes, outs) 替代这里
    print("====== 优化后:", len(opt), "个节点 ======")
    for n in opt.values():
        print("  ", n)


graph = {
      "x":  Node(Op.INPUT),                    # 叶子
      "w":  Node(Op.INPUT),                    # 叶子
      "x2": Node(Op.POW,   [x],  arg=2.0),     # x 的平方
      "m":  Node(Op.MEAN,  [x2], arg=-1),
      "e":  Node(Op.CONST, arg=1e-6),
      "a":  Node(Op.ADD,   [m, e]),
      "r":  Node(Op.RSQRT, [a]),
      "y":  Node(Op.MUL,   [r, w]),            # ← 最后一行,outs = ["y"]
}

def remove_dead_nodes(graph, outputs): # dfs遍历活节点
    alive = set()
    stack = list(outputs)          # 从出口 y 出发
    while stack:
        last = stack.pop()
        alive.add(last)
        for inp in graph[last].inputs:
            if inp.name not in alive:
                stack.append(inp.name)
    return {name: graph[name] for name in alive}