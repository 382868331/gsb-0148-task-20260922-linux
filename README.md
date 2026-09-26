# 带调用破坏约束的寄存器分配（Register Allocation with Call-Clobber Constraints）

一个纯 Python 标准库实现、可导入复用的离题编译器后端模块：输入一个三地址 CFG
（每条指令已给出 `uses` / `defs` / 固定寄存器要求 `fixed` / call 破坏集合
`clobbers`，phi 输入显式绑定到前驱边），计算**边敏感活跃性**、构造**冲突图**、
做**确定性染色**，在寄存器不足时**迭代 spill 并重建冲突图**，最终输出寄存器级
程序、栈槽、以及可独立验证的活跃集合。不追求最少 spill。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅使用标准库）。
- **无需任何第三方依赖**，不需要网络 / 账号 / 密钥 / 数据库 / 中间件 / Docker。
- 核心算法、测试、演示全部离线运行；示例程序由仓库内固定种子代码生成，
  不读取任何外部附件或在线语料。

## 准确命令

```bash
# 准备（可选；仓库自带 .venv 软链，Linux 原生 Python 3.14.7）
python3.14 -m venv .venv
# source .venv/bin/activate

# 运行必要测试
python -m unittest discover -s tests -v

# 运行固定输入演示（真实计算；本机约 0.05 秒，远小于 8 秒预算）
python demo.py

# 运行本轮栈槽布局演示（真实计算；本机 < 1 秒，预算 90 秒）
python demo_iteration.py
```

## 公开接口与数据结构

```python
from regalloc import allocate, Program, Block, Instr, PhiNode, program_from_dict
from regalloc import (RegAllocError, ProgramFormatError,
                     InfeasibleProgramError, BudgetExhaustedError)

alloc = allocate(program, K)        # K 为 2..8 的 int（拒绝 bool / float）
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
| `liveness` | `LivenessReport`：块 live-in/live-out、**每条边**活跃集、每条指令 before/after 活跃集 |
| `iterations` / `history` | spill 迭代次数与每轮被 spill 的变量序列 |

寄存器级指令只有四类：`op`（原指令，含 `reads/writes/clobbers`）、
`copy`、`load r <- slot`、`store slot <- r`。

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
   栈槽按变量 id 分配。超过寄存器数的参数按“栈传递 ABI”处理：
   其值入口即位于自己的栈槽。

5. **Phi 降级与临界边**（`lower.py`）：
   - 前驱只有一个后继时，边动作追加到该前驱尾部；否则该边为**临界边**，
     插入一个单后继的拆分块（自环临界边同样处理）。
   - 每条边的寄存器/槽并行赋值用经典算法顺序化（先发射“目标不再被需要”的
     动作；剩下的纯寄存器置换环用一个专用内存 scratch 槽打断），
    **不占用额外寄存器**，故对 K=2 也成立。

6. **不追求最少 spill**：启发式确定性、可解释；正确性优先。

## 栈槽布局（本轮新增，默认关闭的独立开关）

寄存器分配结束后每个 spill 值只拥有一个**逻辑槽号**（仍为变量 id）。
调用 ``allocate(program, K, pack_slots=True, ...)`` 时追加一个**后置**
栈槽布局阶段（`stack_layout.py`），它**不改变**寄存器分配结果与发射的
spill 代码，只在“插入溢出指令之后”重算活跃性并给出最终帧偏移：

1. **在 lower 后的无 phi 机器 CFG 上重算槽位活跃性**：phi 的读写在
   `lower.py` 中已经落到对应前驱边（前驱唯一后继时追加到前驱尾部，
   否则进入临界边拆分块），因此槽位活跃性只是普通的反向数据流
   （`store` 定义/杀死槽，`load` 使用槽），phi 不会被当成块内普通指令；
   栈传递参数（被 spill 的入口参数）从入口块 live-in 起就位于自己的槽。
2. **槽冲突图**：任一机器指令点 live-before 的槽两两冲突；`store` 的目标
   携带新值并与所有跨越该点存活的值相邻，写点共存由该保守团覆盖。
3. **调用方给定大小/对齐**：`slot_size` / `slot_alignment` /
   `frame_alignment` 必须是正整数（拒 bool），且
   `slot_alignment <= slot_size`、`frame_alignment` 是 `slot_alignment` 的整数倍。
   重叠活跃值占用的**字节区间**严格分开（任意偏移候选都按半开区间判重叠）。
   用于打破寄存器置换环的 scratch 槽同样进帧布局。
4. **小规模精确枚举**：按 (大小, 冲突度, id) 排序后 DFS 枚举每个槽的合法
   对齐偏移，求**最小帧大小**（末尾向上取整到帧对齐）。`layout_budget`
   （默认 200,000 次候选放置）耗尽时返回确定性贪心得到的**合法布局**，
   `proven_optimal=False`，并在 `note` 中注明“未证最优”。
5. **跨 call 活跃值不会丢失**：call 只破坏寄存器、不破坏内存；槽复用与否
   完全由 lower 后重算的活跃性决定——仍需使用的槽在 call 点与任何新值冲突，
   已死槽则允许跨 call 复用。

接口约定::

    alloc = allocate(program, K, pack_slots=True,
                     slot_size=8, slot_alignment=8,
                     frame_alignment=16, layout_budget=200_000)
    alloc.stack_layout      # StackLayout | None（开关关闭时为 None）
    alloc.slot_liveness     # lower 后 CFG 的槽活跃性 SlotLiveness

`StackLayout` 字段：`slot_offsets`（逻辑槽 -> 最终帧字节偏移）、
`slot_sizes` / `slot_alignments`、`frame_size`、`frame_alignment`、
`scratch_slot` / `scratch_offset`、`proven_optimal`、`budget` /
`budget_spent`、`note`。布局后每条 `load` / `store` 机器指令携带
`mi.offset`（其实际访存偏移），`render()` 打印为 `slotN[fp+off]`；
开关关闭时 `mi.offset == -1`、`allocated_program.frame is None`。
`allocation.slots` 与寄存器 `assignment` 不受开关影响（测试逐条断言）。

独立对照用例（`tests/test_stack_layout.py`，期望值手算/独立重推）：

- `sample_branch_slot_reuse()`：两条**互斥分支**各自的私有 spill 槽在不同
  CFG 边上存活，可共享同一偏移；而跨分支、在**汇合点**仍活跃的值
  （及汇合后产生的值）与重叠的私有槽冲突，不能共享；phi 源经边动作
  （含临界边拆分）验证；
- 小规模槽布局枚举核对最小帧（直线压力例程峰值 5 个 8 字节槽 -> 48 字节，
  16 字节对齐），`layout_budget=1` 时返回合法布局且 `proven_optimal=False`；
- 混合 8/16 字节、8/16 对齐的直接阶段 API 用例，校验字节区间不重叠与偏移对齐；
- 跨 call spill 例程 + 偏移**变异用例**（把一条 load 的偏移改成别的槽区间），
  原符号执行器按最终字节偏移建模内存，真实访存后必须判定不等价。

## 独立验证

- `regalloc/symbolic.py` 是**独立符号执行器**，不调用分配器核心：
  在 token（溯源）值域里，按确定性有界 DFS 配对执行“原程序 vs 分配后程序”，
  校验相同 uid 顺序、相同符号输入/输出；每条 CFG 边上，边敏感活跃集中的
  变量都带着正确 token 到达其归属位置（家寄存器或栈槽）；call 破坏后的
  寄存器变为特殊 `CLOBBERED` token，再被读取即判失败。
- `tests/_reference.py` 提供另外两个独立参考：直接验证活跃性数据流方程、
  以及用独立的并行赋值模拟器比对顺序化后的机器指令效果。
- 测试中的期望值均为手算/独立推导，**不通过调用被测核心得到**；
  “往返一致”只作为补充。测试还包含**变异用例**（故意删 reload / 改读寄存器 /
  删 phi 边拷贝 / 写错栈槽），确认执行器确实有判别力。开启槽布局后，执行器
  内存以**最终字节偏移**为键（共享偏移即真实别名），故改错一条 load/store 的
  偏移也会在真实访存后被抓到。

覆盖要点：临界边 phi（含自环）、跨 call 活跃值、固定寄存器碰撞、
寄存器不足触发 spill、预算耗尽与证明无解分离、严格输入拒绝、确定性与可复现。

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
- 符号执行对循环做有界展开（默认深度 80、步数 20,000），结果会标明
  `bound_reached`；这是**覆盖**界限而非数值容差。
- 块数上限 40、单块指令上限 10,000 是接口边界；演示用例规模很小，
  未穷举上限规模，也不做长时压力测试。
- 槽布局的精确偏移枚举在最坏情况下同样是指数问题；受 `layout_budget`
  限制，超预算时只保证返回**合法布局**并明确标注“未证最优”，绝不谎报最小。
  本轮槽大小/对齐是**全函数统一标量**参数（调用方给定），不建模变长/向量
  宽槽或真实 ABI 的入栈参数区重排；开关只影响帧布局，不影响寄存器分配。

## 目录

```
regalloc/
  model.py      数据结构 + 严格输入校验 + 错误类型
  liveness.py   边敏感活跃性（含 phi / 无 phi 两种）
  allocator.py  冲突图、贪心+精确染色、迭代 spill、allocate()
  lower.py      临界边拆分、并行拷贝顺序化、寄存器/栈代码生成
  stack_layout.py 插入溢出指令后的槽活跃性重算、槽冲突图、预算化精确帧布局
  symbolic.py   独立配对符号执行器（前后程序对照）
  samples.py    确定性示例与固定种子程序生成器
tests/          必要测试（含独立参考 _reference.py 与变异用例）
demo.py         固定输入演示：正常 spill 结果 + 实际触发的拒绝边界
demo_iteration.py 本轮演示：槽共享/不可共享对照、跨 call、对齐与预算边界
```
