# lower_c.py: 后端 lowering 的迷你版 —— 图 → C 循环代码
#   这是"代码拐杖":让你看到计算图怎么变成一串循环代码(=内核的胚胎)
#   真编译器这一步 emit 的是 Triton/CUDA 代码,我们 emit C,骨头一样词汇不同
#
#   对照看点(写完看输出):
#     ① 原始图(8节点)要 emit 5 个循环(POW/MEAN/ADD/RSQRT/MUL 各一个)
#     ② 融合图(3节点)只 emit 1 个循环(RMSNORM 一步合一)
#     → 融合省循环 = 省内核 = 省启动开销。这就是④融合在后端的回报
#
#   假设:向量长度 N,叶子 x/w 是 float 数组,输出 y 是 float 数组

import sys
from pathlib import Path
from ir import Node, Op
from parse import parse
from passes import fuse, remove_dead_nodes

sys.stdout.reconfigure(encoding="utf-8")


def lower_to_c(graph, outs, indent="    ") -> str:
    """把图 emit 成 C 代码字符串。
    规则:每个非叶子节点 → 一个 for 循环,遍历向量做一次运算。
    叶子(INPUT/CONST)当数组/常数声明,不算循环。
    """
    lines = ["// ---- 由 lower_c.py 自动生成 ----"]
    lines.append("#include <math.h>")
    lines.append(f"#define N 1024   // 向量长度,假装 1024")
    lines.append("void kernel(float* x, float* w, float* y) {")

    # 拓扑序:从出口往回收集依赖,再反过来 = 执行序
    #   (这就是 ④ remove_dead_nodes 那招的正向版)
    order = []
    seen = set()
    def collect(name):
        if name in seen:
            return
        seen.add(name)
        for inp in graph[name].inputs:
            collect(inp.name)
        order.append(name)   # 后序:依赖在前,自己最后 → 正好执行序
    for o in outs:
        collect(o)

    for name in order:
        nd = graph[name]
        if nd.op == Op.INPUT:
            lines.append(f"{indent}// INPUT {name}:外部传入,不 emit")
        elif nd.op == Op.CONST:
            lines.append(f"{indent}float {name} = {nd.arg};   // CONST")
        elif nd.op == Op.POW:
            lines.append(f"{indent}float {name}[N];")
            lines.append(f"{indent}for (int i = 0; i < N; i++) {name}[i] = powf({nd.inputs[0].name}[i], {nd.arg});")
        elif nd.op == Op.MEAN:
            lines.append(f"{indent}float {name} = 0;")
            lines.append(f"{indent}for (int i = 0; i < N; i++) {name} += {nd.inputs[0].name}[i];   // sum")
            lines.append(f"{indent}{name} /= N;   // 再除 N = mean")
        elif nd.op == Op.ADD:
            lines.append(f"{indent}float {name} = {nd.inputs[0].name} + {nd.inputs[1].name};")
        elif nd.op == Op.RSQRT:
            lines.append(f"{indent}float {name} = 1.0f / sqrtf({nd.inputs[0].name});")
        elif nd.op == Op.MUL:
            lines.append(f"{indent}float {name}[N];")
            lines.append(f"{indent}for (int i = 0; i < N; i++) {name}[i] = {nd.inputs[0].name} * {nd.inputs[1].name}[i];")
        elif nd.op == Op.RMSNORM:
            # ★ 融合后的成品:5 步合一,一个循环搞定
            lines.append(f"{indent}// RMSNORM:融合后,POW+MEAN+ADD+RSQRT+MUL 五步缩成一个循环")
            lines.append(f"{indent}float sum = 0;")
            lines.append(f"{indent}for (int i = 0; i < N; i++) {{ float t = {nd.inputs[0].name}[i]; sum += t*t; }}   // POW+MEAN 一起")
            lines.append(f"{indent}float rs = 1.0f / sqrtf(sum/N + 1e-6f);   // ADD+RSQRT")
            lines.append(f"{indent}for (int i = 0; i < N; i++) y[i] = rs * {nd.inputs[1].name}[i];   // MUL")
        else:
            lines.append(f"{indent}// TODO: {nd.op.name} 没写 lowering")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 用硬编码图(和 passes.py 同款),保证能跑
    x  = Node(Op.INPUT, name="x")
    w  = Node(Op.INPUT, name="w")
    x2 = Node(Op.POW,   [x],  arg=2.0, name="x2")
    m  = Node(Op.MEAN,  [x2], arg=-1,  name="m")
    e  = Node(Op.CONST, arg=1e-6,        name="e")
    a  = Node(Op.ADD,   [m, e],          name="a")
    r  = Node(Op.RSQRT, [a],             name="r")
    y  = Node(Op.MUL,   [r, w],          name="y")
    graph   = {n.name: n for n in [x, w, x2, m, e, a, r, y]}
    outputs = ["y"]

    print("=" * 60)
    print("【原始图 8 节点】lower 到 C —— 数一下有几个 for 循环:")
    print("=" * 60)
    print(lower_to_c(graph, outputs))

    print()
    print("=" * 60)
    print("【融合后 3 节点】lower 到 C —— 融合省了几个循环?")
    print("=" * 60)
    fused = fuse(graph, outputs)
    print(lower_to_c(fused, outputs))
