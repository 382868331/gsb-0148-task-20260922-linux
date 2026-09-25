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

在 Linux 原生 Python 3.14.7 的隔离环境中运行（只依赖标准库）：

```bash
# 创建一次性隔离环境（无需任何第三方依赖；若已存在可跳过）
python3.14 -m venv .venv

# 运行必要测试（回归入口）
./.venv/bin/python -m unittest discover -s tests -v

# 运行固定输入演示（真实计算；本机约 0.1 秒，远小于 90 秒预算）
./.venv/bin/python demo.py
```

不使用 `.venv` 时，系统 `python3`（3.14.7）也可直接运行同样两条命令；
演示与测试均不访问网络、不读取环境变量或凭据。

## 公开接口与数据结构

```python
from regalloc import allocate, Program, Block, Instr, PhiNode, program_from_dict
from regalloc import (RegAllocError, ProgramFormatError,
                     InfeasibleProgramError, BudgetExhaustedError)

alloc = allocate(program, K)                 # K 为 2..8 的 int（拒绝 bool / float）
alloc = allocate(program, K, frequencies=f)  # 可选：每块非负整数执行频次
```

- `frequencies`：长度**恰好等于块数**的非负 `int` 序列（list/tuple，
  逐块对应，块 0 为入口）。语义是纯启发式输入——只影响“寄存器不足时
  先 spill 谁”，**不改变合法性、不改变程序语义**；非法形状给
  `ProgramFormatError`（块数不符、`bool`、负数、float、非序列都拒绝）。
  省略或传 `None` 时，走原有的循环深度加权策略，结果与旧版逐字节一致。

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
| `frequencies` / `edge_frequencies` | 实际使用的块频次与由其估计的**边频次**；未给 profile 时均为 `None` |
| `cost_history` | **每轮启发式估计**的候选 spill 加权 reload/store 成本（仅用于选受害者，不是收益） |
| `memory` | 从最终机器指令**实际清点**的访存统计：`loads/stores/total`（原始条数）、`weighted_loads/weighted_stores/weighted_total`（按块/边频次加权）、`by_site`（逐块/逐拆分边明细，可手工核对） |

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
- `frequencies` 省略或为 `None` 时走旧策略；给定时必须是长度等于块数的
  非负整数序列，拒绝 `bool` / `float` / 负数 / 长度不符 / 非序列，
  错误路径形如 `frequencies[3]: expected integer, got bool`。
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

4. **迭代 spill（旧策略，默认）**：每次证明不可染后，按
   `(冲突度+1) / 加权使用频次`（循环内使用按 10 的循环嵌套深度次方加权）
   选最大者 spill，随后**从零重建活跃性与冲突图**。
   栈槽按变量 id 分配。超过寄存器数的参数按“栈传递 ABI”处理：
   其值入口即位于自己的栈槽。

4b. **按块频次选择 spill 对象（新增，`frequencies=` 开启）**：频次是调用方
   给的非负整数（如热点循环体给大数、冷错误分支给小数），只影响启发式。
   - **边频次估计**（`profile.edge_frequencies`）：唯一后继边取整个块频次；
     唯一前驱边取 `min(前驱频次, 本块频次)`（冷块不被热前驱平均抬高，零
     永远传零）；回边 `latch -> header` 取
     `max(0, 头频次 - 已知前向入边之和)`；其余 fork/join 边把残余频次均分
     （余数按下标顺序派发）。全部整数运算，无浮点。
   - **候选成本估计**（`profile.spill_costs`，每轮重算）：按当前 spill 形式
     的真实 use/def 点计数——块内每次 use 一次 reload、每次 def 一次
     store（权=块频次）；phi 结果/源在其绑定的前驱边上各一次 store/reload
     （权=边频次）。这是**启发式估计**，仅用于选受害者。
   - **同样冲突条件优先挑代价较低者**：候选限定为冲突图的 **K-核**
     （反复剥离度数 < K 的可平凡着色节点；预着色伪节点常驻、reload/store
     临时量可剥离），核内按精确分数 `度数 / 估计加权访存成本` 取最大，
     平局按变量 id。固定寄存器伪节点、call 破坏伪节点和临时量都不是
     real node，永不被 spill；不可 spill 约束与固定/call 约束继续由图结构
     强制满足。
   - **每轮重写后从零重建活跃性、spill 形式与成本**，不会拿旧图反复溢出
     同一临时变量；染色预算耗尽仍明确抛 `BudgetExhaustedError`。
   - **收益按实际插入的 reload/store 统计**（`Allocation.memory`），从最终
     降级机器指令逐条清点（块内指令按块频次、临界边拆分块按边频次加权），
     绝不把启发式估计充当最终收益。

5. **Phi 降级与临界边**（`lower.py`）：
   - 前驱只有一个后继时，边动作追加到该前驱尾部；否则该边为**临界边**，
     插入一个单后继的拆分块（自环临界边同样处理）。
   - 每条边的寄存器/槽并行赋值用经典算法顺序化（先发射“目标不再被需要”的
     动作；剩下的纯寄存器置换环用一个专用内存 scratch 槽打断），
     **不占用额外寄存器**，故对 K=2 也成立。

6. **不追求最少 spill**：启发式确定性、可解释；正确性优先。

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

### 新增能力的独立验证（按块频次选择 spill）

- `regalloc/concrete.py` 是第二个独立验证器：用真实整数输入分别执行
  **原程序**和**分配后程序**的微型具体解释器（opcode 由 `note` 给出：
  `const/add/sub/mul/lt/br2/br3`），比较 uid 序列、块轨迹与出口可见值。
  它不调用分配器核心，也不与 `symbolic.py` 共享执行逻辑。
- 热点演示程序 `sample_hot_loop_cold_error()` 是一个带**冷错误恢复臂**的
  热循环（冷臂与热闩在静态上同循环深度，旧策略无法区分）。演示对同一
  程序分别用“热闩/冷错误”频次和**完全相反**的频次各分配一次：
  - 用真实输入独立跑两条路径（纯热路径、进入冷恢复的路径），三种分配
    （旧策略 / HOT profile / 反向 COLD profile）与原程序结果逐一一致；
  - 以同一频次把两种分配的**实际**机器指令加权清点：HOT profile 下热点
    加权访存约下降 ~50%，反向 profile 下冷臂访存下降 ~99%（精确数值以
    `./.venv/bin/python demo.py` 实跑为准，不是预录值）；
  - 另有手工布局核对：单块小程序里栈传参数的 reload 数、定义点的
    store 数与加权统计逐条对得上（`tests/test_profile.py`）。

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
- 块频次是启发式权重而非真实分支概率：边频次用结构性规则估计（唯一前驱
  /回边流平衡/均分），手写 profile 若内部流不一致，估计按裁剪规则归一，
  不会因此报错或产生浮点；该输入只影响 spill 选择，**不宣称全局最优**，
  只保证同冲突条件下倾向加权访存更便宜的候选，平局按变量 id。

## 目录

```
regalloc/
  model.py      数据结构 + 严格输入校验 + 错误类型
  liveness.py   边敏感活跃性（含 phi / 无 phi 两种）
  allocator.py  冲突图、贪心+精确染色、迭代 spill、按频次选 spill、allocate()
  profile.py    频次校验、边频次估计、加权 spill 成本估计、实际访存统计
  lower.py      临界边拆分、并行拷贝顺序化、寄存器/栈代码生成
  symbolic.py   独立配对符号执行器（前后程序对照）
  concrete.py   独立具体执行器（真实整数输入，前后程序结果对照）
  samples.py    确定性示例与固定种子程序生成器（含热循环/冷错误用例）
tests/          必要测试（含独立参考 _reference.py 与变异用例、频次用例）
demo.py         固定输入演示：正常 spill、call、频次引导 spill、拒绝边界
```
