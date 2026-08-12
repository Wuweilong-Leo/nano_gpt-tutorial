# parse.py: 迷你前端 —— 文本 → 图
#   对应 dynamo 的活:把"源码"一行行演一遍,演到哪记到哪
#   "名字 = OP 参数"  →  一行 = 一步 = 一个节点

import sys
from pathlib import Path
from ir import Node, Op

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台防乱码(踩坑速查 #5)


def parse(text: str) -> tuple[dict[str, Node], list[str]]:
    nodes: dict[str, Node] = {}       # 名字查表:之后的行引用前面的名字 → 节点(这就是"边")
    outputs: list[str] = []           # 程序的"出口":每行的名字都记下来,最后一个 = 返回值
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()           # ["y", "=", "MUL", "r", "w"]
        name, op_name = toks[0], toks[2]
        rest = toks[3:]

        if op_name == "INPUT":
            nodes[name] = Node(Op.INPUT, name=name)        # 叶子:没有输入
        elif op_name == "CONST":
            nodes[name] = Node(Op.CONST, arg=float(rest[0]), name=name)   # 只带说明牌,无输入
        else:
            op = Op[op_name]                                # 字符串 → 枚举(查 Op 表)
            # 每组 op 的"胃"不一样:一元(1 输入) vs 二元(2 输入)
            n_in = 1 if op_name in ("POW", "MEAN", "RSQRT") else 2
            srcs = [nodes[t] for t in rest[:n_in]]          # 按规则取原料
            arg = float(rest[n_in]) if len(rest) > n_in else None   # 剩下的才是说明牌
            nodes[name] = Node(op, srcs, arg=arg, name=name)
            outputs.append(name)
    return nodes, [outputs[-1]]   # 出口 = 程序最后一行赋值的名字(迷你语言的"返回值")


if __name__ == "__main__":
    src = Path(__file__).parent / "test_prog.txt"           # 取脚本同目录的样例,不怕在哪跑
    text = src.read_text(encoding="utf-8")
    nodes, outs = parse(text)
    print("======== 优化前:", len(nodes), "个节点 ========")
    for n in nodes.values():
        print(n)