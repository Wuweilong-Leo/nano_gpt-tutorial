from enum import Enum

# ============================================================
# op 表 = 这台机器支持的指令集
#   图 = 一串"步骤",每一步做什么,种类有限 → 每种动作起个名 = op
#   分组 = 给优化/后端当 if 判断用:吃几个输入、带不带轴参数
#   叶子也要 op:没有原料(CONST/INPUT),图连不起来
#   arg = 节点的"说明牌"(POW 的指数、MEAN 的轴……塞口袋里)
# ============================================================

class Op(Enum):
    # ---- META:叶子。原料也是节点,没有它们图没有起点 ----
    INPUT = 1     # 原始输入(x、weight 这类)

    CONST = 2     # 常数(平方的 2、eps 的 1e-6)

    # ---- UNARY:吃一个输入,吐一个输出 ----
    NEG = 3       # -x

    POW = 4       # x ** k,指数 k 放 arg

    RSQRT = 5     # 1 / sqrt(x)

    # ---- BINARY:吃两个输入 ----
    ADD = 6       # a + b

    MUL = 7       # a * b

    MATMUL = 8    # a @ b。linear 的核心,大客户的常客(七次)

    # ---- REDUCE:绕轴缩维,轴放 arg ----
    MEAN = 9      # 对某轴求平均

    # ---- 大算子:融合 pass 的"成品" ----
    # ④ 的融合 pass:把 POW+MEAN+ADD+RSQRT+MUL 打包成一个节点,
    # 打包产物得有名字才存在 → RMSNORM。和 dynamo 的 SDPA 同一哲学。
    RMSNORM = 10


# ============================================================
# Node = 一次动作的记账单(三件套):
#   op     → 这是哪种动作(查 Op 表)
#   inputs → 用什么原料(上游节点们;叶子节点是空的)
#   arg    → 怎么干(说明牌:POW 的指数、MEAN 的轴……没有就 None)
#   name   → 打印和前端引用用(%0、%1 这种),可留空
# ============================================================

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Node:
    op: Op
    inputs: list["Node"] = field(default_factory=list)   # 语法:可变默认值必须 default_factory
    arg: Any = None
    name: str | None = None

    def __repr__(self):
        src = ", ".join(i.name if i.name else "?" for i in self.inputs)
        s = f"{self.op.name}({src})"
        if self.arg is not None:
            s += f" arg={self.arg}"
        return s

x      = Node(Op.INPUT, name="x")
two    = Node(Op.CONST, arg=2.0,   name="two")
x2     = Node(Op.POW, [x], arg=2.0, name="x2")     # 输入的第三个写法:把 x 包进列表
mean   = Node(Op.MEAN, [x2], arg=-1, name="mean")  # 轴 -1 = 最后一维
eps    = Node(Op.CONST, arg=1e-6, name="eps")
add    = Node(Op.ADD, [mean, eps], name="add")
r      = Node(Op.RSQRT, [add], name="r")
weight = Node(Op.INPUT, name="weight")
out    = Node(Op.MUL, [r, weight], name="out")

for nd in [x2, mean, add, r, out]:
    print(nd)