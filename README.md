# regalloc — 带调用破坏（call-clobber）约束的寄存器分配库

面向离线编译器后端的可复用寄存器分配模块。输入最多 **40 块**的三地址
CFG（每条指令显式给出 `uses`/`defs`、固定寄存器要求和 call 破坏集合，phi
输入显式绑定前驱边），在 **K = 2..8** 个物理寄存器上：

1. 计算**边敏感活跃性**（phi 源只在其绑定的前驱边上活跃；phi 目标在块
   入口诞生）；
2. 降低 phi 为边上的**并行拷贝**，自动拆分临界边；
3. 构造**冲突图**（精确的“指令点共存”规则，含固定寄存器与 call
   clobber 约束）；
4. 确定性 Chaitin 简化/选色着色，失败则**迭代溢出**并从零重建冲突图；
5. 插入 `store`/`load`（spill/reload），顺序化并行拷贝（必要时用一个
   scratch 栈槽打破置换环）；
6. 返回分配后程序、寄存器/栈槽表，以及可独立验证的原始程序活跃集合。

不追求最少溢出（经典 Chaitin 启发式：度数/出现次数最小者优先溢出）。

## 运行环境

* **Python 3.14.7**（仅使用标准库；3.10+ 语法特性，无需任何第三方依赖）
* Linux 原生运行；不需要网络、账号、密钥、数据库、中间件或 Docker。

安装 Python 后无需 `pip install` 任何内容。仓库中的 `requirements.txt`
只是一行注释。

## 准确命令

```bash
# 运行全部必要测试（独立验证器 + 确定性生成的属性测试）
python -m unittest discover -s tests -v

# 运行固定输入演示（约 0.1 秒，远小于 8 秒上限；真实计算，无 sleep、无预录）
python demo.py
```

`demo.py` 打印两部分：

1. **正常结果**：一个跨 call 活跃、K=2 必然产生溢出的程序，输出寄存器
   分配、按变量 ID 排序的栈槽、发射后的机器码，并用独立符号执行器验证
   前后程序等价；
2. **真实触发的拒绝边界**：一个值被固定到 r1，却必须活过破坏 r1 的
   call —— 抛出 `UnsatisfiableProgram`（证明无解），并说明它与“搜索
   预算耗尽”是两类不同异常。

## 公共接口

```python
from regalloc import (
    allocate, Program, Block, Phi, Instruction,
    Reg, Slot,
    Allocation, AllocatedProgram, Liveness,
    AllocatorError, InvalidArgumentError, InvalidProgramError,
    UnsatisfiableProgram, AllocationBudgetExhausted,
    check_finite_number,
)

result: Allocation = allocate(program, K)
```

### 输入数据结构（不可变 dataclass）

* `Program(entry: str, blocks: tuple[Block, ...], params: tuple[str, ...] = ())`
* `Block(id, phis: tuple[Phi] = (), instrs: tuple[Instruction] = (), succs: tuple[str] = ())`
* `Phi(dst: str, srcs: Mapping[前驱块id, 源变量], fixed_def: int | None = None)`
  —— `srcs` 的键集合必须**恰好等于**该块的前驱集合。
* `Instruction(opcode, uses=(), defs=(), fixed_use={}, fixed_def={}, clobbers=())`
  * `fixed_use` / `fixed_def`：变量 → 物理寄存器（整数 `0..K-1`），
    键必须分别出现在 `uses` / `defs` 中；
  * `clobbers`：该指令破坏的物理寄存器元组（call 破坏集合）。

### 输出

`Allocation` 字段：

| 字段 | 含义 |
| --- | --- |
| `program: AllocatedProgram` | 具体机器码，块由 `MBlock(id, instrs, succs)` 组成 |
| `assignment` | 原变量 → 寄存器号；溢出变量与“证明恒死”的变量为 `None` |
| `stack_slots` | 溢出变量 → 栈槽号；**按变量 ID 排序后连续分配 0..n-1** |
| `live: Liveness` | 原始程序的边敏感活跃性（`edge_live`/`live_in`/`live_out`/`points`，均为排序元组） |
| `K`, `iterations` | 寄存器数、溢出迭代轮数 |
| `scratch_slots` | 打破并行拷贝环所用的额外 scratch 槽（常规 phi 降低下为空元组） |

发射指令 `MInstr(opcode, reads, writes, origin, clobbers)` 的位置只能是
`Reg(i)` 或 `Slot(i)`；opcode：

* `param`：入口参数物化（仅 entry 块首，按参数声明顺序）；
* `mov` / `load` / `store`：寄存器拷贝 / 从槽装入 / 存入槽；
* 其他：原用户 opcode（语义对分配器不透明），`origin` 指向其在原块中
  的指令序号。

## 输入上限与错误语义

* 块数 `1..40`；`K` 必须是 `2..8` 的 **int**（`bool` 是 `int` 子类，
  显式拒绝：`allocate(p, True)` 抛 `InvalidArgumentError`）。
* 数值输入：`check_finite_number(x, name)` 拒绝 NaN/±Infinity 与非数值，
  同样拒绝 `bool`。本库没有其他浮点结果（分配是离散的），因此**不存在
  浮点误差口径问题**，也不存在用容差掩盖边界错误的路径。
* 非法 IR 抛 `InvalidProgramError`，错误带 `location`（块 id / 指令序号 /
  phi 序号），可定位；不做静默纠正。校验内容包括：未知变量、重复定义、
  phi 前驱集合不匹配、先定义后使用（支配关系）、entry 不得有前驱、块
  必须从 entry 可达、保留名字前缀 `..` 与保留 opcode
  （`mov/load/store/param`）禁用等。
* 输入 `Program` 永不被修改（分配器在内部副本上工作）；失败时不留部分
  变更（纯函数式构造，成功后才返回结果）。
* 两类“失败”严格分开：
  * `UnsatisfiableProgram`：**证明无解**，溢出也无法修复 —— 两个同时
    活跃值被固定进同一寄存器、固定值必须活过破坏其寄存器的指令、或单
    个使用点需要超过 K 个操作数寄存器；
  * `AllocationBudgetExhausted`：仅当有界的溢出/重着色搜索达到
    `MAX_ITERATIONS`（10000）才抛出，**不代表无解**。测试通过临时下调
    预算验证该异常确实不同于前者。

## 设计取舍

* **边敏感 phi 语义**：phi 源只在绑定边上活跃；临界边（前驱多后继、
  后继多前驱）通过插入 `..split.p->q` 单后继块放置拷贝，非临界边把
  拷贝放在唯一后继块的尾部（或唯一前驱块的头部）。
* **精确干扰**：对指令点（L=活前，A=活后，W=定义）：`L` 成团、`W`
  成团、`W` 只与真正越过该点的值 `L∩A` 连边。结果允许复用“在该点
  死亡”的操作数寄存器（标准三地址规则）；并行拷贝中死亡源与其目标
  不冲突（拷贝可同寄存器）。
* **call clobber**：只有活过指令的值（`L∩A`）被禁止使用被破坏寄存器；
  死亡操作数无所谓，结果也允许投递到被破坏的调用者保存寄存器。
* **溢出模型**：被溢出的原变量每次出现改为短寿 reload 临时量（用完即
  存储），所以一轮溢出后冲突图严格变小，迭代必然终止；参数也可溢出，
  其代价是在 entry 首物化一次后存入槽。
* **确定性**：简化阶段按名字最小选点、溢出按（度数/出现次数，名字）
  排序、迭代顺序为块声明顺序；同一输入多次分配结果逐指令相同。
* **栈槽**：按溢出变量 ID 排序连续编号；并行拷贝破环另用编号 n 的
  scratch 槽，并在 `scratch_slots` 中显式声明。

## 独立验证（测试如何建立期望值）

`tests/verifier.py` **不调用分配器算法**，它独立实现：

1. 自己的边敏感活跃性（与返回的 `Liveness` 逐点比对）；
2. “到达 token”集合 R(v)（phi 目标为各边来源的不动点并集）；
3. 对发射后程序的 may-token 抽象机：块头并集合并、`mov/load/store`
   复制 token 集、用户指令以 R(def) 替换结果、clobber 清除寄存器；
4. 逐原指令检查操作数/结果 token 精确一致、活跃变量→位置存在无冲突
   二分图完美匹配（两个同时活跃值绝不能同位）、clobber 后存活值仍有
   唯一家、固定寄存器约束、以及发射 CFG 的结构合法性。

变异测试（手工对发射码交换读、删 load/store、伪造 clobber/固定寄存器）
确认该验证器确实能抓错；“往返一致”只作补充。

`tests/proggen.py` 用固定种子 LCG 确定性生成支配合法的 SSA CFG（直线 /
diamond phi / 计数循环 phi，随机 call clobber），测试在 60 个种子 ×
4 个 K 上端到端验证（开发期另以 500 种子 × K=2..8 做过离线 fuzz，
3500 例全部通过；该规模不放在短测试中以免拖慢常规运行）。

## 已知限制（本契约的有限子集，非行业标准完整实现）

* 仅处理题面规定的三地址 IR：变量静态唯一定义（SSA 风格），不做 SSA
  解构之外的寄存器合并、不做指令调度、不模拟任何真实 ISA/调用约定；
  用户 opcode 的语义完全不透明。
* 不追求最少溢出/最少 load-store；Chaitin 启发式给出正确但非最优的
  分配。
* 并行拷贝顺序化支持“目标寄存器互不相同”的映射（本分配器的干扰规则
  保证这一点）；更一般的任意多写伪指令不在输入契约内。
* 栈槽按字一个值建模，不区分类型宽度、不对齐、不模拟寻址方式；
  `Slot(i)` 仅代表一个抽象字槽。
* `scratch_slots` 与变量槽共用线性槽号空间（scratch 编号为 n），机器
  模型假设所有槽可被 `load/store` 直接访问。

## 目录

```
regalloc/
  __init__.py     公共 API
  model.py        输入/输出数据结构与异常
  validate.py     严格、可定位的输入校验
  liveness.py     原始程序的边敏感活跃性
  allocator.py    phi 降低、冲突图、着色、迭代溢出、代码生成
  examples.py     确定性示例程序（测试与 demo 使用）
tests/
  verifier.py     独立符号执行/验证器（不调用核心算法取期望值）
  proggen.py      固定种子 SSA-CFG 生成器
  test_regalloc.py
demo.py
README.md
```
