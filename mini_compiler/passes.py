# passes.py: 优化管线 —— 给图"瘦身"
#   优化 = 对图做手脚,但每一步都保证数学结果不变
#   pass = 把整张图过一遍,按一套规则改写
#
# 当前焦点:④ 的"融合 pass" —— 认 RMSNorm 指纹,8 节点 → 3 节点
#   重打骨架:为什么必须"两遍",见 fuse 注释

import sys
from pathlib import Path
from ir import Node, Op
from parse import parse

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# 测试图:手搭一条 RMSNorm 链(8 个节点)
#   先建节点变量(带 name),再装进 dict —— name 既是变量名也是图里的 key
#   暂时绕开 test_prog.txt:用这张硬编码图测 fuse,立刻能跑
# ============================================================
x  = Node(Op.INPUT, name="x")
w  = Node(Op.INPUT, name="w")
x2 = Node(Op.POW,   [x],  arg=2.0, name="x2")   # x 的平方
m  = Node(Op.MEAN,  [x2], arg=-1,  name="m")    # 对最后一维求平均
e  = Node(Op.CONST, arg=1e-6,        name="e")   # eps
a  = Node(Op.ADD,   [m, e],          name="a")   # mean + eps
r  = Node(Op.RSQRT, [a],             name="r")   # rsqrt(mean+eps)
y  = Node(Op.MUL,   [r, w],          name="y")   # × weight  ← 出口

graph   = {n.name: n for n in [x, w, x2, m, e, a, r, y]}
outputs = ["y"]


# ============================================================
# pass 1: 死代码消除(你已写,留着)
#   注:这张测试图没有死节点,所以它在这跑不出"瘦身",
#       真正干活的是下面的 fuse。DCE 是给"有废分支"的图准备的。
# ============================================================
def remove_dead_nodes(graph, outputs):
    alive = set()
    stack = list(outputs)
    while stack:
        last = stack.pop()
        alive.add(last)
        for inp in graph[last].inputs:
            if inp.name not in alive:
                stack.append(inp.name)
    return {name: graph[name] for name in alive}


# ============================================================
# pass 2: 融合 —— 认指纹,把一条链换成一个节点
#
#  ★ 为什么必须"两遍"?
#    单遍循环按插入顺序走:x → w → x2 → m → e → a → r → y
#    中间节点(x2/m/e/a/r)排在终点 MUL(y)前面,先被搬进新图;
#    等走到 y 才认出指纹,中间节点早进门了 → 删不掉。
#    所以拆两遍:
#      第一遍:只认指纹、记"要蒸发谁"的名字(不碰图)
#      第二遍:才搬图,名字在 dead 里的跳过
#    口诀:用"名字"区分留/删,别用 op 类型 —— 叶子 x/w 也是 INPUT,
#         靠 op 删会误伤,靠名字不会。
# ============================================================
def fuse(graph, outputs):
    dead = set()      # 被融掉的中间节点名字 → 第二遍跳过
    repl = {}         # 命中 MUL 的名字 → (x, w) → 第二遍造 RMSNORM 用

    # ---- 第一遍:认指纹,只记账,不动图 ----
    #   链 = 数据:要验的四层 op(叶子 x 不在链里,它是验完 POW 后取 inputs[0] 得到)
    chain = [Op.RSQRT, Op.ADD, Op.MEAN, Op.POW]
    for name, node in graph.items():
        if node.op != Op.MUL:
            continue                       # 非 MUL 不认指纹

        cur = node                         # 从终点 MUL 开始往回走
        mid = []                           # 摸到的中间节点(命中后 = [r,a,m,x2])
        hit = True
        for want in chain:
            cur = cur.inputs[0]            # 往回摸一格
            # TODO ①(核心,3 行):
            #   先验再决定——这步写对了,前面 z=a*b 的崩溃就修好了
            #   if cur.op != want:   →   hit = False; break   (不对就否决,别硬往下摸)
            #   mid.append(cur)                                   (对了,收进 mid)
            if cur.op != want:
                hit = False
                break
            mid.append(cur)

        if not hit:
            continue                       # 没命中(如 z=a*b 这种普通乘法),原样留

        # 命中了!mid = [r, a, m, x2]
        x = mid[-1].inputs[0]              # 叶子:链底 POW 的输入
        w = node.inputs[1]                 # weight:MUL 第二个输入
        repl[name] = (x, w)                # 第二遍造 RMSNORM 要用

        # TODO ②(1 行):算 dead 名单。只记 .name 字符串,不是节点对象
        #   要蒸发的有两批:
        #     ① mid 里的四个中间节点(r/a/m/x2)—— 全进 dead
        #     ② eps 的 e:它不在 chain 上,是 ADD 的第二个输入
        #        ADD = mid[1],所以 e = mid[1].inputs[1]
        #   把这五个名字塞进 dead(提示:{n.name for n in mid} | {mid[1].inputs[1].name})
        dead |= {n.name for n in mid} | {mid[1].inputs[1].name}

    # ---- 第二遍:搬图,dead 里的跳过,MUL 格换成 RMSNORM ----
    new_graph = {}
    for name, node in graph.items():
        if name in dead:
            continue                               # 蒸发:被融掉的中间节点
        if name in repl:
            x, w = repl[name]
            new_graph[name] = Node(Op.RMSNORM, [x, w], name=name)
        else:
            new_graph[name] = node                 # 叶子(x/w)和没碰的节点原样搬
    return new_graph


if __name__ == "__main__":
    print("====== 优化前:", len(graph), "个节点 ======")
    for n in graph.values():
        print("  ", n)

    opt = fuse(graph, outputs)

    print("====== 优化后:", len(opt), "个节点 ======")
    for n in opt.values():
        print("  ", n)
