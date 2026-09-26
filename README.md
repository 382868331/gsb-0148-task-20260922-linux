# 带调用破坏约束与可复用栈槽的寄存器分配

一个纯 Python 标准库实现、可导入复用的离题编译器后端模块：输入一个三地址 CFG
（每条指令已给出 `uses` / `defs` / 固定寄存器要求 `fixed` / call 破坏集合
`clobbers`，phi 输入显式绑定到前驱边），计算**边敏感活跃性**、构造**冲突图**、
做**确定性染色**，在寄存器不足时**迭代 spill 并重建冲突图**，最终输出寄存器级
程序、逻辑栈槽、以及可独立验证的活跃集合。新增的栈槽布局阶段默认关闭；开启后会在
**实际插入的 load/store 之后**重算内存活跃性，让生命周期不重叠的溢出值共享同一
栈区间，并输出最终帧大小、帧对齐以及每条 load/store 的字节偏移。寄存器分配本身不追求
最少 spill；小规模栈布局会枚举证明最小帧，超预算时返回合法布局并明确标记未证最优。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅使用标准库）。
- **无需任何第三方依赖**，不需要网络 / 账号 / 密钥 / 数据库 / 中间件 / Docker。
- 核心算法、测试、演示全部离线运行；示例程序由仓库内固定种子代码生成，
  不读取任何外部附件或在线语料。

## 准确命令

```bash
# 固定环境（Linux 原生 Python 3.14.7，仅标准库）
python3.14 -m venv .venv
. .venv/bin/activate
python --version  # 应为 Python 3.14.7

# 运行必要测试
python -m unittest discover -s tests -v

# 原演示（真实计算；无 sleep / 预录结果）
python demo.py

# 本轮新增栈槽布局演示（要求 90 秒内完成；通常远低于 1 秒）
python demo_iteration.py
```

## 公开接口与数据结构

```python
from regalloc import allocate, Program, Block, Instr, PhiNode, program_from_dict
from regalloc import (RegAllocError, ProgramFormatError,
                     InfeasibleProgramError, BudgetExhaustedError)

alloc = allocate(program, K)        # K 为 2..8 的 int（拒绝 bool / float）

# 可选：寄存器分配、spill/reload 插入完成后再做栈槽共享布局
alloc = allocate(
    program,
    K,
    pack_stack=True,
    slot_size=8,
    slot_alignment=8,
    frame_alignment=None,  # 默认取所有对象对齐的最大值
    stack_budget=10_000,
)
layout = alloc.stack_layout
```

- `Program(blocks, params=(), name="")`
  - `blocks`：`Block` 元组，块 0 为唯一入口；给出 `succs`（后继**块下标**）。
    所有块必须从块 0 可达。
  - `params`：入口已持值的变量 id 元组。
- `Block(label, instrs, succs=(), phis=())`
- `Instr(uid, uses=(), defs=(), fixed={var: reg}, clobbers=(), note="")`
  - `uid`：程序内唯一。
  - `uses` / `defs`：非负 int，二者在同一条指令内不得重叠。
  - `fixed`：把某操作数在**本条指令**固定到具体物理寄存器（其余生命周期
    可住在别处，由分配器在指令前后插入短 `copy` 物化）。
  - `clobbers`：call 类指令破坏的物理寄存器。
- `PhiNode(def_var, sources=((pred_block_index, src_var), ...))`
  - 每个前驱块必须恰好有一个源；源变量只在对应前驱边上活跃（边敏感）。
  - 入口块（块 0，无入边）不允许有 phi。

也可用 `program_from_dict({...})` 从纯 dict/list（如 JSON 形状）构造。

### 返回值 `Allocation`

| 字段 | 含义 |
| --- | --- |
| `k` | 使用的寄存器数（寄存器为 r0..rK-1） |
| `assignment` | `{变量 id: 物理寄存器}`，确定性 |
| `slots` | `{被 spill 的变量 id: 栈槽号}`；**栈槽号即变量 id** |
| `allocated_program` | 寄存器级 CFG（`AllocatedProgram`，可用 `.render()` 打印） |
| `liveness` | 原程序的 `LivenessReport`：块 live-in/live-out、**每条边**活跃集、每条指令 before/after 活跃集 |
| `stack_layout` | 默认 `None`；`pack_stack=True` 时为 `StackLayout` |
| `iterations` / `history` | spill 迭代次数与每轮被 spill 的变量序列 |

寄存器级指令只有四类：`op`（原指令，含 `reads/writes/clobbers`）、
`copy`、`load r <- slot`、`store slot <- r`。未布局时 `MI.offset == -1`；
布局后每条 `load/store` 的 `offset` 是从栈帧基址起算的字节偏移，渲染中显示为
`[sp+N]`。

### 栈槽布局接口

`allocate(..., pack_stack=True)` 会保持寄存器分配开关前后的 `assignment`、
`slots`（逻辑槽号仍等于被 spill 变量 id）、迭代历史不变，只在 lowering 后重写最终
`load/store` 的字节偏移。也可显式调用：

```python
from regalloc import allocate, layout_stack_slots

alloc = allocate(program, 2)
layout = layout_stack_slots(
    alloc,
    sizes=8,                  # 也可传 {var: bytes}
    alignments=8,             # 也可传 {var: alignment}，必须是 2 的幂
    frame_alignment=16,        # None 表示取对象最大对齐
    budget=10_000,
)
```

`StackLayout` 字段：

| 字段 | 含义 |
| --- | --- |
| `frame_size` / `frame_alignment` | 最终帧字节数与帧对齐，大小向上对齐到帧对齐 |
| `offsets` | `{逻辑槽变量 id: sp 相对字节偏移}`，含 phi 置换环 scratch（若使用） |
| `sizes` / `alignments` | 调用方给定的每对象大小和对齐 |
| `accesses` | 每条最终 `load/store`：块、指令下标、类型、槽、寄存器、偏移和所属 CFG 边 |
| `live_in/live_out/synthetic_blocks` | 在“实际 spill 访存事件 CFG”上重算的内存活跃性；普通指令访存在块节点，phi 访存在对应的前驱边节点 |
| `interferences` | 字节区间不可重叠的逻辑槽对 |
| `optimal` / `budget_exhausted` / `placements_checked` | 是否证明最小帧；超预算时为 `False/True`，但布局仍合法 |

## 错误语义（先于算法固定的契约）

- 所有 id / 寄存器 / `K` 必须是 `int`；**`bool` 一律拒绝**
  （`isinstance(True, int)` 在 Python 为真，故显式排除），浮点也拒绝。
- 数值入口拒绝 **NaN 与 ±Infinity**（本模块数据本身是整数；
  `check_finite_float` 提供统一的有限数校验）。
- 非法输入抛 `ProgramFormatError`（`ValueError` 子类），信息带**可定位路径**，
  如 `blocks[3].instrs[2].fixed[1]: register 3 does not exist for K=3`；
  不做静默纠正。
- `K` 必须在 **2..8**；`fixed` / `clobbers` 中的寄存器必须 `< K`。
- **输入规模上限**：最多 **40 个块**；单块最多 10,000 条指令。
- 两类“无解”严格分开：
  - `InfeasibleProgramError`：**证明无解**。例如一条指令在同一读点需要
    多于 K 个不同操作数（spill 也无法消除读点的寄存器需求）、一个输出点
    需要多于 K 个不同结果、或两个操作数被固定到同一寄存器。
  - `BudgetExhaustedError`：精确染色搜索预算耗尽（默认 100,000 次赋值尝试），
    此时**可行性未知**，绝不谎报为无解。
- 分配是无状态、纯函数式的；一次 `allocate` 失败不会改动输入对象，
  不存在“部分变更”。

## 算法与设计取舍

1. **边敏感活跃性**（`liveness.py`）：在标准 phi 数据流方程中，
   块 B 到后继 S 的边上，活跃集为
   `(in(S) - phiDefs(S)) ∪ phiSrcs(B->S)`。
   因此一个 phi 源**只**在其绑定的那条前驱边上活跃，绝不出现在其他入边。
   采用按块下标升序的不动点迭代，结果确定。

   约定：phi 的定义在拥有它的块入口“出生”，故不出现在该块的 live-in，
   而是出现在块体第一条相关指令的 live-before；phi 源在边界被并行拷贝消费。
   `instr_before/after` 是据同一反向迁移独立重算的，便于核对。

2. **冲突图**（`allocator.py`）：
   - 三地址指令的读操作数在“读点”两两冲突、并与跨指令存活值冲突；
     写结果在“写点”两两冲突、并与存活值冲突；**被消费的 use 与新 def
     不冲突**（use 死于指令边界，标准 SSA 规则）。
   - 固定寄存器操作数建模为与指令短连接的**预着色伪节点**
     （`copy` 伪节点↔变量），因此“固定”只作用于该指令点，而非整个生命周期。
   - call 的每个被破坏寄存器建模为预着色伪节点，只与**跨该 call 存活**的值
     冲突，从图层面保证跨 call 值不会落入被破坏寄存器。
   - phi 的边拷贝是每条前驱边上的并行赋值。仅当可证明合并不破坏语义时
     （源在该并行组后死亡、目标不被组内其他动作读取等）才允许同色合并；
     真交换（swap）等情形保留冲突并在降级时用内存暂存槽打破环。
   - 被 spill 的变量从图中消失：每个 use 产生独立 reload 临时量，
     每个 def 产生独立 store 临时量，二者都是短生命周期节点。

3. **染色**：先跑确定性 DSATUR 贪心；失败再跑有界的**精确 DSATUR 回溯**，
   能真正证明 K-不可染（返回 `infeasible`）或在预算耗尽时返回 `budget`。
   节点选择按（饱和度, 度, id）确定排序，故同一输入结果可复现。

4. **迭代 spill**：每次证明不可染后，按
   `(冲突度+1) / 加权使用频次`（循环内使用按 10 的循环嵌套深度次方加权）
   选最大者 spill，随后**从零重建活跃性与冲突图**。
   逻辑栈槽按变量 id 分配。超过寄存器数的参数按“栈传递 ABI”处理：
   其值入口即位于自己的栈槽。

5. **插入指令后的栈内存活跃性**（`stack.py`）：寄存器分配和 lowering 决定后，
   构造一个“块节点 + 每条 CFG 边一个边节点”的事件图。普通指令的 spill reload/store
   属于对应块节点；phi 的 load/store/置换环 scratch 动作只属于对应前驱边节点。
   然后在该事件图上做反向数据流，而不是把 phi 源/目标错误地当作块内普通指令。

6. **栈区间装箱**：调用方给定每对象大小与对齐（也可统一给定）。冲突对象的字节区间
   `[offset, offset+size)` 绝不重叠；分支互斥的生命周期可复用，汇合后仍活跃或被 phi
   边再次读写的值不能错误复用。call 只破坏模型中的寄存器，不破坏栈；若某溢出值在
   call 后仍活跃，其内存生命周期会跨越该点并阻止其他写入复用该区间。

7. **小规模精确枚举**：按冲突度/大小排序后枚举对齐坐标，寻找最小帧大小；对象数较小
   时候选坐标来自子集和，以证明最优。`budget=0` 或枚举超限时返回确定性 first-fit
   合法布局，`optimal=False`、`budget_exhausted=True`，绝不谎报最优。

8. **Phi 降级与临界边**（`lower.py`）：
   - 前驱只有一个后继时，边动作追加到该前驱尾部；否则该边为**临界边**，
     插入一个单后继的拆分块（自环临界边同样处理）。
   - 每条边的寄存器/槽并行赋值用经典算法顺序化（先发射“目标不再被需要”的
     动作；剩下的纯寄存器置换环用一个专用内存 scratch 槽打断），
     **不占用额外寄存器**，故对 K=2 也成立。

9. **不追求最少 spill**：启发式确定性、可解释；正确性优先。`pack_stack` 只改变栈偏移，
   不反过来改变染色、spill 集合或生成的寄存器/内存操作种类。

## 独立验证

- `regalloc/symbolic.py` 是**独立符号执行器**，不调用分配器核心：
  在 token（溯源）值域里，按确定性有界 DFS 配对执行“原程序 vs 分配后程序”，
  校验相同 uid 顺序、相同符号输入/输出；每条 CFG 边上，边敏感活跃集中的
  变量都带着正确 token 到达其归属位置（家寄存器或栈槽）；call 破坏后的
  寄存器变为特殊 `CLOBBERED` token，再被读取即判失败。
- `tests/_reference.py` 提供另外两个独立参考：直接验证活跃性数据流方程、
  以及用独立的并行赋值模拟器比对顺序化后的机器指令效果。
- `tests/test_stack_layout.py` 固定一个分支互斥可复用、汇合存活值不可复用的
  diamond 对照程序，枚举核对最小 16 字节帧，检查每条最终 load/store 偏移，并用原
  配对执行器覆盖两条分支的实际内存访问；另有混合大小/对齐和零预算合法回退测试。
- 测试中的期望值均为手算/独立推导，**不通过调用被测核心得到**；
  “往返一致”只作为补充。测试还包含**变异用例**（故意删 reload / 改读寄存器 /
  删 phi 边拷贝 / 写错栈槽），确认执行器确实有判别力。

覆盖要点：临界边 phi（含自环）、phi spill 访存位于前驱边、跨 call 活跃值、
固定寄存器碰撞、寄存器不足触发 spill、分支槽复用/汇合不可复用、栈槽精确最小帧与超预算
合法回退、预算耗尽与证明无解分离、严格输入拒绝、确定性与可复现。

## 数值口径

本模块的输出是离散的寄存器/栈槽指派与符号 token，**不产生任何浮点结果**，
因此不存在浮点误差或容差；唯一的数值相关契约是“拒绝 NaN/Infinity”，
不会用宽容差掩盖边界错误。

## 已知限制（不宣称完整行业标准兼容）

- 这是题目契约下的有限子集：单一入口 CFG、SSA 风格三地址指令、
  显式边绑定 phi；不做指令调度、寄存器对/宽寄存器、延迟槽、
  跨函数全局分配或真实 ABI 调用约定。
- 指令语义是不透明的（视为 `f(uses)->defs`）；`fixed` 是操作数级约束，
  不是“整个变量全程锁寄存器”。
- 精确染色在最坏情况下是指数问题；受 `COLORING_BUDGET` 限制，
  超预算时按契约抛 `BudgetExhaustedError`（可行性未知），而非给出可能错误的答案。
- 精确栈帧装箱在一般情况下也可能是昂贵搜索；超过 `stack_budget` 后返回 first-fit
  合法布局，`optimal=False`，不声称最小。当前小规模演示和测试在预算内证明最优。
- 符号执行对循环做有界展开（默认深度 80、步数 20,000），结果会标明
  `bound_reached`；这是**覆盖**界限而非数值容差。
- 块数上限 40、单块指令上限 10,000 是接口边界；演示用例规模很小，
  未穷举上限规模，也不做长时压力测试。
- 栈对象大小/对齐由调用方显式给定；模块不模拟真实平台 ABI 的 red zone、
  callee-saved 保存区、可变参数区或延迟槽。call 的语义是“破坏声明的寄存器，
  不破坏已分配栈帧内容”。

## 目录

```
regalloc/
  model.py      数据结构 + 严格输入校验 + 错误类型
  liveness.py   边敏感活跃性（含 phi / 无 phi 两种）
  allocator.py  冲突图、贪心+精确染色、迭代 spill、allocate()
  lower.py      临界边拆分、并行拷贝顺序化、寄存器/栈代码生成
  stack.py      插入 spill 后的边敏感内存活跃性、栈偏移装箱与最小帧枚举
  symbolic.py   独立配对符号执行器（前后程序对照）
  samples.py    确定性示例与固定种子程序生成器
tests/          必要测试（含独立参考 _reference.py、栈布局对照与变异用例）
demo.py         固定输入演示：正常 spill 结果 + 实际触发的拒绝边界
demo_iteration.py  本轮新增：diamond 栈槽复用、最小帧与超预算合法布局演示
```
