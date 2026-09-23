# 寄存器分配后端：调用破坏 + 寄存器宽度视图 + 异常控制流（Register Allocation v2）

纯 Python 标准库实现、可导入复用的离线编译器后端模块。v1（带调用破坏约束的
寄存器分配）的接口与行为**完全保留**；v2 在同一数据模型上增加**物理寄存器
宽度视图/别名、寄存器类别、指令早破坏约束、异常控制流边、调用保存规则、
栈槽复用与对齐**，并返回带明确状态的分配结果。

- **v1 入口**：`allocate(program, K)`（行为与旧版一致，旧测试一条未删）。
- **v2 入口**：`allocate2(program, machine)`，输入显式给出
  **别名集合、允许的寄存器类别、早破坏、调用保存属性、异常后继**。
- 旧的简单目标描述仍可原样使用；`regalloc.compat` 提供 v1→v2 适配。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅标准库）。
- 无需任何第三方依赖，不需要网络 / 账号 / 密钥 / 数据库 / Docker / 硬件。
- 示例程序由仓库内固定种子代码确定性生成，不读取外部附件或在线语料。

## 准确命令

```bash
# 快速回归（v1 + v2 单元测试；本机约 1 秒）
python -m unittest discover -s tests -v

# 扩展验证（>=100 块/>=1000 指令/16 视图主规模；完整枚举小参考域；约 40 秒）
python -m unittest discover -s tests_ext -v

# v1 固定输入演示（约 0.05 秒）
python demo.py

# v2 固定输入演示（新增正常结果 + 棘手边界 + 真实触发的拒绝；约 20 秒，预算 90 秒）
python demo2.py
```

两套验证分开：`tests/` 是每次改动都跑的快速回归；`tests_ext/` 包含
主规模非退化例和小参考域的**完整枚举**（见“独立验证”），耗时较长。

---

# V1（原有接口，行为不变）

```python
from regalloc import allocate, Program, Block, Instr, PhiNode, program_from_dict
alloc = allocate(program, K)          # K 为 2..8 的 int（拒绝 bool / float）
```

输入三地址 CFG：`Instr(uid, uses, defs, fixed={var: reg}, clobbers, note)`，
phi 输入显式绑定前驱边；块上限 40、单块 10,000 指令、K∈2..8。输出
`Allocation(assignment, slots, allocated_program, liveness, iterations, history)`，
spill 槽号等于变量 id。非法输入抛 `ProgramFormatError`（带可定位路径）；
证明无解抛 `InfeasibleProgramError`；染色预算耗尽抛
`BudgetExhaustedError`（可行性未知）。完整说明见 git 历史中的旧版 README，
行为未改，74 个旧测试全部保留并通过。

---

# V2：寄存器视图 / 异常边 / 栈槽复用

## 物理机模型 `Machine`

```python
from regalloc import Machine, RegView

Machine(views, aliases, class_align, stack_align=8, name="")
RegView(name, group, lo, hi, cls, callee_saved=False)
```

| 字段 | 含义 |
| --- | --- |
| `RegView` | 物理寄存器组 `group` 上的位段 `[lo, hi)`；宽度必须是整字节 |
| `aliases` | **显式给出**的别名集合 `{view: [与之共享至少一个 bit 的其他 view]}`；校验必须对称、不能自别名，且**必须与位段重叠关系完全一致**（多报/漏报都拒绝） |
| `cls` | 视图所属寄存器类别；同一类别的所有视图宽度相同 |
| `class_align` | 每个类别的栈对齐（字节），必须显式给出 |
| `callee_saved` | 调用保存视图；该集合必须**别名封闭**（64 位保存而其中 32 位破坏自相矛盾，拒绝） |

冲突判定一律按**位段重叠**：两个值可以共存当且仅当其视图不共享任何 bit，
比较寄存器名字是不够的（`w0`/`x0` 不同名但共享低 32 位）。内置
`flat_machine(K)` 等价于旧的 K 个独立 32 位寄存器；`machine_from_dict`
接受 JSON 形状。

## v2 程序（显式选项 `var_classes`）

```python
Program(blocks, params=(), name="", var_classes={var: class_name})
```

- 只有传入 `var_classes` 的程序才是 v2 程序；v2 程序必须用 `allocate2`。
- v1 程序调用 `allocate2`、v2 程序调用 `allocate` 都会得到明确的格式错误。
- v2 块上限 **128**（v1 仍为 40）；单块仍为 10,000 条。
- 未在 `var_classes` 中出现的变量取固定缺省类：**宽度最小、类别名字典序
  最小**的类别（文档化、可复现）。

`Instr` 新增字段（缺省即旧行为）：

| 字段 | 边界含义 |
| --- | --- |
| `early_clobber=(def...)` | 这些**输出在输入被读取之前写入**；早破坏 def 不得与本指令任何 use 共享 bit |
| `may_throw=True` | 可能转到块的异常后继；异常路径上本指令的 **defs 不产生**（call 破坏照常发生） |
| `is_call=True` | 调用；破坏所有**非调用保存**视图（外加 `clobber_views`） |
| `fixed_views={var: view}` | 把操作数在**本条指令**固定到具名视图（视图类别必须与变量一致） |
| `clobber_views=(...)` | 额外显式破坏的具名视图 |

v2 不允许旧的整数 `fixed`/`clobbers`（必须用 `fixed_views`/
`is_call`+`clobber_views`），二者混用直接拒绝。

`Block` 新增 `exc_succs=(...)`：

- 正常后继与异常后继不相交；带异常后继的块必须**恰有一条** `may_throw`
  指令，反之亦然（异常边从该指令处离开，该指令不必是块末指令）；
- 异常边与正常边一起参与可达性、前驱集合和数据流；
- 异常后继块的 phi 源绑定在异常边上（与普通边同样显式）。

## 返回值 `AllocationResult`（状态，不抛“结果类”异常）

```python
result = allocate2(program, machine)
result.status        # "allocated" | "infeasible" | "budget_exhausted"
result.allocated     # True iff status == "allocated"
```

| `status` | 含义 | 程序/指派字段 |
| --- | --- | --- |
| `allocated` | 成功，已独立复核 | `assignment`/`slots`/`allocated_program` 齐全 |
| `infeasible` | **证明**无解（固定到重叠视图、同一读/写点同类视图数超过硬件等） | 全为 `None` |
| `budget_exhausted` | 精确染色预算耗尽，可行性**未知** | 全为 `None` |

预算耗尽与无解严格分开，且**绝不输出看似完整的机器程序**：非 allocated
状态下 `allocated_program/assignment/slots` 必为 `None`。检查器不信任
求解器自己的状态标志：`budget_exhausted` 却带着程序会被检查器判失败。

| 字段 | 含义 |
| --- | --- |
| `assignment` | `{变量 id: 视图名}`，确定性 |
| `slots` | `{变量 id: Slot(offset, size, align)}`；活跃范围不相交的 spill 变量**复用**字节偏移 |
| `scratch_slot` | 打拷贝环专用槽，位于所有变量槽之后，不与任何变量槽重叠 |
| `frame_size` | 栈帧字节数（= scratch 槽末端） |
| `iterations` / `history` | spill 迭代次数 / 每轮被 spill 的变量 |
| `liveness` | 边敏感活跃性报告（含每条正常边和异常边） |

寄存器级指令四类：`op reads/writes/clobbers [may_throw]`、`copy`、
`load view <- [offset]`、`store [offset] <- view`。

## 固定缺省与并列规则

- 所有迭代顺序固定：块按升序、视图按声明顺序、并列按 id/名称；
  同一输入重复运行得到同一结果（测试有断言）。
- 染色：先确定性 DSATUR 贪心；失败再跑有界精确 DSATUR（默认预算
  100,000 次赋值尝试），能证明不可染或返回预算耗尽。
- spill：每次证明不可染后，在贪心卡住的瓶颈邻域选**同类别度最大**者
  spill（并列看加权使用频次、再看 id），随后**从零重建活跃性、冲突图与
  槽位**——spill 重写后全部派生信息重算。
- 槽位分配在最终活跃性上做：按（对齐、大小、id）确定顺序，把每个 spill
  变量放到最低的、与所有同时活跃 spill 变量不重叠的对齐偏移。
- 并行拷贝顺序化按**位级“仍被读取”关系**发射；纯寄存器置换环先用
  **真实可用的临时视图**（宽度够、与该边所有存活值及所有待发射动作都不
  重叠；绝不借用仍活跃的别名位置），不存在时再用专用内存 scratch 槽。
  每条被打断的环用了哪种机制记录在
  `allocated_program.edge_breaks[(p,q)]`（`"temp"`/`"slot"`）。
- 异常边一律插入专用拆分块（异常转移从块中部出发，不能复用正常尾部代码）。

## 活跃性口径（异常边）

- 异常边 `p->q` 的活跃集 = `(in(q) - phiDefs(q)) ∪ phiSrcs(p->q)`，
  在 may_throw 指令**紧后**取；该集合向指令之前传播，故处理块需要的值
  必须在调用点存活。
- may_throw 指令的 `instr_after` 是正常延续与异常延续的并（保守 may-set，
  已在文档与测试中固定）；其后的指令只在正常路径。
- 在异常点定义或其后定义的变量不允许在异常边上活跃（构造期校验拒绝）。

# 独立验证（不调用被测求解器）

1. `regalloc/verify.py` —— **逐点逐边独立检查器**：自己重算数据流方程，
   校验每个程序点/每条边（正常与异常）上活跃值的**位级归属**、栈槽字节
   重叠、对齐、调用破坏、早破坏、固定视图物化、每条源边在分配后 CFG 中
   可实现；检查不过给出具体边/点位置，无法定位的边不允许被吞掉。
2. `regalloc/symbolic2.py` —— **按位符号执行器**：寄存器按位段建模，
   窄视图写只替换对应 bit，读别名宽视图得到组合 token，从而能抓住别名
   宽度破坏。它配对执行“原三地址程序 vs 分配后程序”，覆盖**正常出口和
   异常出口**，核对 uid 顺序、符号输入/输出、每条边上活跃 token 到家位置；
   调用破坏写入特殊 `CLOBBERED` token。循环有界展开（深度/步数是**覆盖**
   界限，结果带 `bound_reached`）。
3. `tests/_reference2.py` —— 另外两个独立参考：独立活跃性方程实现；
   **暴力枚举**小参考域的全部 var→view 指派并逐点核验约束；独立的位段
   并行赋值模拟器。
4. `tests_ext/test_exhaustive.py` —— 对 240 个确定性生成的小程序，
   暴力枚举（含早破坏/调用/混合 w/x 类）与求解器精确染色、整条流水线
   **逐个比对**；期望值只来自独立枚举，不调用被测核心算期望。
5. `tests`/`tests_ext` 中含多组**变异用例**：交换活跃值家位、重叠槽偏移、
   错齐槽位、早破坏写重叠读、删异常边拷贝、把读引到被调用破坏的视图、
   用别名宽视图替换拷贝源、伪造状态标志等，确认检查器确有判别力。

# 端到端例子（一次同时用到三组新能力 + 本题补充要求）

`regalloc/samples2.sample_v2_torture()` / `demo2.py` 第 1 节，单个 CFG 内
同时包含：

- may_throw 调用与**异常边**（带处理块 phi，异常边独立拆分块）；
- **早破坏**指令（uid 12）；
- **双宽 x 类值跨调用**存活（只能进调用保存的 x1/x2 视图）；
- 回边上的 s 类 phi **交换环**，且该边同时存活的值占满全部五个寄存器组
  的低 32 位——**没有空闲临时视图**，环只能经 scratch 栈槽打断；
- 真实 spill（调用点 w 类压力），带 4/8 字节对齐与栈槽重叠检查。

主规模非退化例：`generate_scale_program()`（固定种子，110 块、1214 指令、
16 个视图、调用/异常/早破坏/phi/回边/处理块齐全；约 5–6 秒分配）。

# 与旧行为的兼容性 / 迁移

- v1 的数据结构与 `allocate(program, K)` 行为零变化：新增字段全部可选且
  缺省为旧语义；74 个旧测试原样保留并通过，未放松任何断言。
- 唯一的新增**拒绝**（旧代码无法构造，故不破坏旧调用）：
  1. `Program(var_classes=...)` 的程序交给 v1 `allocate` → 提示用
     `allocate2`；旧程序交给 `allocate2` → 提示用 `allocate` 或经适配；
  2. v2 程序内使用旧整数 `fixed`/`clobbers` → 改用 `fixed_views`/
     `is_call`+`clobber_views`。
- 迁移办法：
  - 想要 v2 结果形状（状态对象、具名视图、槽布局）跑旧程序：
    `regalloc.compat.allocate_legacy_via_v2(program, K)`（自动把整数
    fixed/clobbers 翻译成 `flat_machine(K)` 的具名视图）；
  - 想要新硬件：定义 `Machine` + `var_classes`，改调 `allocate2`，按
    `result.status` 分支处理 allocated/infeasible/budget_exhausted。
- 旧目标描述（只给 K）无需修改即可继续用于 v1 与 compat 适配。

# 数值口径

输出是离散的视图/栈槽指派与符号 token，**不产生浮点结果**，无浮点误差或
容差；数值入口继续拒绝 NaN/Infinity、整数入口拒绝 bool。覆盖界限（符号
执行深度/步数、染色预算）是显式报告的界限，不用宽容差掩盖边界。

# 已知限制

- 有限题目子集：单一入口 CFG、SSA 风格三地址指令、每块至多一条
  may_throw（异常边从该点出发）；不做指令调度、跨函数全局分配或真实
  ABI（调用保存按显式标注执行）。
- 指令语义不透明（`f(uses)->defs`）；`fixed_views` 是操作数级约束。
- 精确视图染色最坏情况指数，受预算约束；超预算按契约返回
  `budget_exhausted`（可行性未知），不猜答案。
- 符号执行对循环/异常路径有界展开（结果带 `bound_reached`）；这是覆盖
  界限。主规模例用约 90 层/25k 步（演示）或更高（扩展验证）。
- 不追求最少 spill；启发式确定性、可解释、正确性优先。
- 不宣称行业标准兼容、不声明最优性或误差界；检查器对“成功/最优”类结论
  一律自行重算，不信任求解器输出的布尔标志。

# 目录

```
regalloc/
  model.py       v1 数据模型 + 严格校验；v2 可选字段（var_classes/早破坏/异常…）
  machine.py     物理寄存器视图、显式别名集合、类别/对齐、调用保存、flat_machine
  liveness.py    边敏感活跃性：v1 两版 + liveness_v2（异常边在 throw 点注入）
  allocator.py   v1：冲突图、染色、迭代 spill、allocate()（拒绝 v2 程序）
  allocator2.py  v2：视图/别名冲突、早破坏、调用、迭代 spill、槽复用、allocate2()
  lower.py       v1 降级（临界边、并行拷贝、scratch 槽）
  lower2.py      v2 降级：异常边专用拆分、位级并行拷贝顺序化（真临时视图/槽）
  verify.py      独立逐点逐边检查器（位级存活值、槽重叠、调用/早破坏/边可达）
  symbolic.py    v1 独立配对符号执行器
  symbolic2.py   v2 按位配对符号执行器（正常+异常出口，位段寄存器模型）
  samples.py     v1 确定性示例/生成器
  samples2.py    v2 机器与示例：组合折磨例、跨调用宽值、槽复用、主规模生成器
  compat.py      v1→v2 适配（旧程序经 allocate2 流水线）
tests/           快速回归（v1 旧测试原样保留 + v2 单元/变异测试 + 独立参考）
tests_ext/       扩展验证（主规模例 + 240 例小参考域完整枚举比对）
demo.py          v1 演示（约 0.05 秒）
demo2.py         v2 演示（正常结果 + 无空闲临时寄存器的环 + 两次真实拒绝，约 20 秒）
```
