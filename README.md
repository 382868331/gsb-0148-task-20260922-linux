# 带调用破坏约束的寄存器分配（Register Allocation with Call-Clobber Constraints）

一个纯 Python 标准库实现、可导入复用的离题编译器后端模块：输入一个三地址 CFG
（每条指令已给出 `uses` / `defs` / 固定寄存器要求 `fixed` / call 破坏集合
`clobbers`，phi 输入显式绑定到前驱边），计算**边敏感活跃性**、构造**冲突图**、
做**确定性染色**，在寄存器不足时**迭代 spill 并重建冲突图**，最终输出寄存器级
程序、栈槽、以及可独立验证的活跃集合。不追求最少 spill。

除原有的“按循环嵌套深度加权”spill 启发式外，模块还支持给每个块提供**执行频度**
（非负整数），据此按实际 use/def 点和边上 phi 用途估算**加权 reload/store 成本**
来挑选 spill 对象，把访存从热点循环挪到冷错误分支。频度**只影响启发式，不影响
程序语义**；不提供频度时分配策略与旧版完全一致。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅使用标准库）。
- **无需任何第三方依赖**，不需要网络 / 账号 / 密钥 / 数据库 / 中间件 / Docker。
- 核心算法、测试、演示全部离线运行；示例程序由仓库内固定代码构造，
  不读取任何外部附件或在线语料。

## 准确命令

```bash
# 运行必要测试（仓库根目录下；隔离环境的解释器）
./.venv/bin/python -m unittest discover -s tests -v

# 运行固定输入演示（真实计算；本机约 0.1 秒，远小于 90 秒预算）
./.venv/bin/python demo.py
```

`.venv` 指向仓库配套的 3.14.7 隔离环境；也可用任意同版本解释器直接
`python3 -m unittest discover -s tests -v` / `python3 demo.py`。

## 公开接口与数据结构

```python
from regalloc import allocate, Program, Block, Instr, PhiNode, program_from_dict
from regalloc import (RegAllocError, ProgramFormatError,
                     InfeasibleProgramError, BudgetExhaustedError)
from regalloc import spill_cost_table, actual_access_stats

alloc = allocate(program, K)        # K 为 2..8 的 int（拒绝 bool / float）
```

- `Program(blocks, params=(), name="", frequencies=None)`
  - `blocks`：`Block` 元组，块 0 为唯一入口；给出 `succs`（后继**块下标**）。
    所有块必须从块 0 可达。
  - `params`：入口已持值的变量 id 元组。
  - `frequencies`：**可选**，每个块一个**非负整数**执行频度提示
    （长度必须等于块数）。例如热循环体给大值、冷错误处理块给 `0`。
    - `None`（默认）或省略：使用原有的“循环嵌套深度加权”spill 策略，
      行为与不支持频度时完全一致。
    - 提供元组/列表：启用**按块频度加权的访存成本**启发式。
    - 边上的 phi 拷贝/spill 访存按**前驱块**频度加权；为临界边插入的
      拆分块继承其前驱边频度。
    - 频度**只改变 spill 对象的选择**，绝不改变合法性、语义或可行分配集合；
      同一份程序给任何频度（或不给）结果都通过前后等价校验。
    - dict/JSON 形式可用 `"frequencies": [int, ...]` 传入，省略等同 `None`。
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
| `iterations` / `history` | spill 迭代次数与每轮开写前的已 spill 集合 |
| `frequencies` | 本次使用的块频度（`None` 表示旧策略） |
| `spill_decisions` | 每轮 spill 决策日志 `SpillDecision(iteration, victim, mode, weighted_cost, heuristic_score)`；`mode` 为 `"frequency"` 或 `"loop-depth"` |
| `access_stats` | **实际**插入的 load/store 统计 `AccessStats`（见下），不是启发式估计 |

`regalloc.cost` 另提供两个可独立调用的核算函数：

- `spill_cost_table(program, frequencies)`：按真实 use/def 点和边上 phi 用途
  估算“某变量若被 spill”的加权 reload/store 事件数（**启发式**，供决策与对账）。
- `actual_access_stats(program, allocation)`：遍历**最终 lowered 机器码**，
  统计真正发射的 load/store（含 phi 降级为打破寄存器环所用的 scratch 槽
  访存），按所在 lowered 块频度加权。所有演示/测试里汇报的“收益”都来自这里。

寄存器级指令只有四类：`op`（原指令，含 `reads/writes/clobbers`）、
`copy`、`load r <- slot`、`store slot <- r`。

## 错误语义（先于算法固定的契约）

- 所有 id / 寄存器 / `K` / **块频度** 必须是 `int`；**`bool` 一律拒绝**
  （`isinstance(True, int)` 在 Python 为真，故显式排除），浮点也拒绝。
- 数值入口拒绝 **NaN 与 ±Infinity**（本模块数据本身是整数；
  `check_finite_float` 提供统一的有限数校验）。
- 非法输入抛 `ProgramFormatError`（`ValueError` 子类），信息带**可定位路径**，
  如 `blocks[3].instrs[2].fixed[1]: register 3 does not exist for K=3`；
  不做静默纠正。
- `K` 必须在 **2..8**；`fixed` / `clobbers` 中的寄存器必须 `< K`。
- `frequencies` 为 `None`，或长度**恰好等于块数**的非负整数序列；
  长度不符、负数、`bool`、浮点、容器类型错误（dict/str 等）都抛
  `ProgramFormatError`，路径前缀为 `frequencies` / `frequencies[i]`。
- **输入规模上限**：最多 **40 个块**；单块最多 10,000 条指令。
- 两类“无解”严格分开：
  - `InfeasibleProgramError`：**证明无解**。例如一条指令在同一读点需要
    多于 K 个不同操作数（spill 也无法消除读点的寄存器需求）、一个输出点
    需要多于 K 个不同结果、或两个操作数被固定到同一寄存器。
  - `BudgetExhaustedError`：精确染色搜索预算耗尽（默认 100,000 次赋值尝试），
    此时**可行性未知**，绝不谎报为无解。**频度模式下预算耗尽同样明确以该
    异常结束**，不返回半成品、不静默回退到旧策略。
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

4. **迭代 spill**：
   - **无频度（默认，旧策略）**：每次证明不可染后，按
     `(冲突度+1) / 加权使用频次`（循环内使用按 10 的循环嵌套深度次方加权）
     选最大者 spill。该路径的数值与选择与增加频度能力之前保持一致。
   - **提供频度（新策略）**：按真实 use/def 点与边上 phi 用途估算每个候选的
     **加权 reload/store 事件数**（`cost.py` 的 `spill_cost_table`），按
     三级字典序选择：① 冲突缓解度 `(冲突度+1)` 降序（先保证 spill 真能
     缓解当前不可染团，即“同样冲突条件”门槛）；② 加权访存成本升序（在
     缓解度相同的候选里优先冷块/冷门的变量）；③ 变量 id 升序。
   - 无论哪种模式，**每轮 spill 重写后都从零重算活跃性、冲突图与（频度模式的）
     成本表**，避免拿上一轮的旧图反复溢出同一临时量/变量；reload/store
     临时量、固定/call 伪节点及已 spill 变量不作为候选；**预算耗尽明确以
     `BudgetExhaustedError` 结束**。
   - 栈槽按变量 id 分配。超过寄存器数的参数按“栈传递 ABI”处理：
     其值入口即位于自己的栈槽。

5. **Phi 降级与临界边**（`lower.py`）：
   - 前驱只有一个后继时，边动作追加到该前驱尾部；否则该边为**临界边**，
     插入一个单后继的拆分块（自环临界边同样处理）。
   - 每条边的寄存器/槽并行赋值用经典算法顺序化（先发射“目标不再被需要”的
     动作；剩下的纯寄存器置换环用一个专用内存 scratch 槽打断），
     **不占用额外寄存器**，故对 K=2 也成立。

6. **实际收益核算与启发式严格分离**（`cost.py`）：
   - 决策用的 `spill_cost_table` 只是**启发式估计**；演示/测试报告的
     “减少了多少热点访存”一律来自 `actual_access_stats`：遍历**最终
     lowered 机器码**，对真实发射的 load/store（含 phi 降级为打破寄存器环
     所用 scratch 槽的访存）按所在 lowered 块频度加权。绝不拿估计充当收益。
   - 拆分块按其**前驱边**频度加权；无频度时所有块隐式权重 1，加权统计
     退化为原始 load/store 条数。

7. **不追求全局最优**：启发式确定性、可解释；正确性优先。频度只是把
   “哪里热”的信息喂给同一套迭代 spill 框架，不保证最少加权访存，只保证
   在给定案例上相对旧策略确实把访存移出热点（见 `demo.py` 第 3 节）。

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
  删 phi 边拷贝 / 写错栈槽），确认执行器确实有判别力。

覆盖要点：临界边 phi（含自环）、跨 call 活跃值、固定寄存器碰撞、
寄存器不足触发 spill、预算耗尽与证明无解分离、严格输入拒绝、确定性与可复现。

`tests/test_frequency.py` 针对频度能力额外覆盖：`frequencies` 的严格校验、
“频度只影响启发式”（无需 spill 的程序在任意频度下结果逐字相同）、每轮决策日志、
按纸笔推导的成本表逐值核对、直接遍历 MI 流的独立访存统计（含手工布局的
scratch 槽/拆分块加权）、热点-冷分支案例相对旧策略的真实加权访存下降与反频度
下的决策反转、固定寄存器/call 破坏/不溢出临时量等约束仍成立、以及频度模式下
预算耗尽仍抛 `BudgetExhaustedError`。

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
- 块执行频度是**相对**启发式提示（非负整数即可，不要求归一化或自洽于
  CFG 转移概率）；分配器不校验“频度之和是否符合流方程”，把它当作用户
  对冷热的声明。频度为 0 的块仍被完整分析、染色和正确性校验，只是其
  spill 访存权重为 0。

## 目录

```
regalloc/
  model.py      数据结构 + 严格输入校验（含 frequencies）+ 错误类型
  liveness.py   边敏感活跃性（含 phi / 无 phi 两种）
  allocator.py  冲突图、贪心+精确染色、两套迭代 spill 策略、allocate()
  cost.py       加权访存：启发式成本表 + 遍历最终代码的实际统计
  lower.py      临界边拆分、并行拷贝顺序化、寄存器/栈代码生成
  symbolic.py   独立配对符号执行器（前后程序对照）
  samples.py    确定性示例（含热循环/冷错误恢复夹具）与程序生成器
tests/          必要测试（_reference.py 独立参考、变异用例、test_frequency.py）
demo.py         固定输入演示：正常 spill、call 破坏、频度策略对比、拒绝边界
```
