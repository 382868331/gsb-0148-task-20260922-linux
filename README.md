# 寄存器分配：调用破坏约束（v1）＋ 同物理寄存器多宽度视图（v2）

一个纯 Python 标准库实现、可导入复用的离线编译器后端模块。它有两个**显式版本**：

- **格式 v1（原样保留）**：输入最多 40 块的三地址 CFG，每条指令给出
  `uses`/`defs`、数字固定寄存器 `fixed`、call 破坏集合 `clobbers`，phi 输入
  显式绑定到前驱边；`allocate(program, K)`，`K ∈ 2..8`。
- **格式 v2（本次新增）**：后端支持**同一物理寄存器的不同宽度视图**：
  输入显式给出**别名集合**、**允许的寄存器类别**（含双宽 pair）和指令
  **早破坏（early-clobber）**约束；把**异常控制流边**、**调用保存规则**和
  **边上的并行拷贝**一起纳入活跃性；并行拷贝环按真实可用临时寄存器或私有
  栈槽拆解，**不借用仍活跃的别名位置**；栈槽支持**复用与对齐**，spill
  重写后重建全部信息。寄存器集合来自目标描述，调用形式为 `allocate(program)`。

两个版本共用同一数据模型与公开入口；v1 的合法调用、异常类型和确定性结果
全部保留。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅标准库）。
- **无第三方依赖**，不需要网络 / 账号 / 密钥 / 数据库 / 容器 / 硬件。
- 所有示例由仓库内固定种子代码确定性生成；同输入同结果。

## 运行命令（快速回归与扩展验证分开）

```bash
# 全部测试（快速回归 v1+v2 + 扩展验证，本机约 6 秒）
python -m unittest discover -s tests -v

# 仅扩展验证：小参考域完整枚举 + 110 块主规模（约 6 秒）
python -m unittest tests.test_extended -v

# 仅快速回归（v1 旧测试 + v2 单元/变异，不含主规模，约 0.5 秒）
PYTHONPATH=. python -m unittest discover -s tests -p 'test_[!e]*.py'

# 固定输入演示（真实计算，本机约 1.5 秒，预算 90 秒）
python demo.py
```

预算：染色搜索 `COLORING_BUDGET=100_000`（v1）、
`COLORING_BUDGET_V2=200_000`（v2）；符号执行默认深度 120、步数 40_000。
这些是**搜索/覆盖**预算，不是数值容差。

---

## 格式 v2：目标描述与输入字段

### 目标描述 `TargetDesc`

```python
from regalloc import TargetDesc, RegView, RegClass, StackSpec
from regalloc import default_target, tiny_target, allocate
```

- `RegView(name, bank, lo, width, aliases)`：物理寄存器 *bank* 上的一个
  命名位窗口。`aliases` **显式**给出与它位重叠的其他视图名。构造时校验：
  别名必须对称，且**显式别名集合必须与位布局算出的重叠关系完全一致**，
  二者不一致即报错（不信任单一来源）。
- `RegClass(name, width, views=(...))`：单宽度类别，按声明顺序给出允许的
  视图（即允许的 bank）。
- `RegClass(name, width, pairs=((lo_view, hi_view), ...))`：双宽类别，给出
  显式合法的寄存器对（两个不同 bank）；一个值同时占两个 bank。
- `callee_saved=(view,...)`：调用保存视图；同 bank 的所有视图必须一致
  （ABI 属性按 bank 计）。其余 bank 为 caller-saved，call 必须在
  `clobbersv` 中列出其被破坏视图，跨 call 存活值只能放在保存 bank 或栈。
- 内置 `default_target()`：**16 个视图**（4 bank × 64/32/16/8 位，
  x86 风格 rax/eax/ax/al…），类别 GPR8/16/32/64 与双宽 `PAIR64`
  （选项 `eax:ecx`、`ebx:edx`；bank A、C caller-saved，B、D callee-saved）。
- 内置 `tiny_target()`：**6 视图 / 2 bank**，用于强制“无空闲临时寄存器”。

### 指令与块的新字段（仅 v2）

`Instr` 增加：`fixedv={var: 视图名}`、`classes={var: 类别名}`、
`clobbersv=(视图名,...)`、`early_clobber=(def 变量,...)`、
`except_to=(后继块下标,...)`。语义边界：

- `fixedv` 只在**本条指令点**固定该操作数；其余生命周期可住别处（指令前后
  插短 copy 物化）。同一指令两个操作数固定到别名视图 → 证明无解。
- `early_clobber` 中的 def 在输入被消费**之前**写入：它与所有输入、跨指令
  存活值、异常边所需值都冲突（普通 def 可与将死 use 同 bank，早破坏不行）。
- `except_to` 表示该指令可能异常转移到目标块；该路径上**本条 defs 不写入**，
  异常边所需值在指令点（排除其 def）必须可用。目标必须是后继，且该后继为
  异常专用边（块尾正常流不走它）。
- v2 用 `clobbersv`/`fixedv`，**不接受** v1 的数字 `clobbers`/`fixed`。

`Block` 增加 `edge_copies=(EdgeCopies(src, dst, ((d, s), ...)),)`：
边上的**并行拷贝**，源同时读取后再写目标；允许环。环中的目标变量在该边
“出生”（不能在别处定义）。phi 与显式边拷贝可共存，统一按边处理。

`Program` 增加：`version=2`、`target=TargetDesc(...)`、
`stack=StackSpec(align=2, min_slots=0)`、`param_classes={var: 类别}`。
默认类别取目标的 `default_class`（内置目标为 32 位 GPR）。

### 规模与边界（已固定的缺省选择）

| 边界 | v1 | v2 |
| --- | --- | --- |
| 块数上限 | 40 | 1024 |
| 单块指令上限 | 10,000 | 50,000 |
| 寄存器 | `K ∈ 2..8`（数字） | 目标描述的 bank/视图（内置 16 视图） |
| 栈槽 | 槽号=变量 id | 区间打包，支持复用与 2 的幂对齐 |
| 染色预算 | 100,000 | 200,000 |

主规模要求已满足并有**能实际跑完的非退化例子**：固定种子生成器
`regalloc.samples2.generate_large_v2()` 默认 **110 块 / 约 1101 条指令 /
16 视图**，含 3 个自然循环、1 个真实异常边、周期性 call 与固定视图；
本机分配＋全边验证约 1.4 秒。

### 返回值 `AllocationV2`

| 字段 | 含义 |
| --- | --- |
| `assignment` | `{变量: (具体视图,...)}`（pair 为 lo/hi 两个视图），确定性 |
| `assignment_banks` | 同上，以 bank 集合表示 |
| `slots` | `{被 spill 变量: (栈起始, 占用单元数)}`（区间复用、对齐） |
| `frame_slots` | 帧大小（含末尾私有 scratch 槽） |
| `allocated_program` | 视图级机器 CFG（`AllocatedProgram2`，可 `.render()`） |
| `liveness` | `LivenessReportV2`：块/边/指令前后/**指令异常点**活跃集 |
| `scratch_slots` | 实际用栈 scratch 打断拷贝环的边及槽位 |
| `verification` | 独立检查器报告（`VerificationReport`） |

机器指令四类：`op`（视图级 reads/writes/clobbersv）、`copy 视图<-视图`、
`load 视图<-slot[start+off]`、`store slot[start+off]<-视图`。

---

## 独立验证（不信任求解器的布尔声明）

`regalloc/checker2.py` 是**独立检查器**，不调用分配器核心：

1. **独立重算 v2 活跃性**（正常边、异常边、边并行拷贝），与分配器附带的
   报告逐条比对；
2. **位级机器模型**：每个 bank 按位存符号 token、每个帧单元存一个 token；
   `ax` 写、`rax` 读等位级别名按位可见；
3. **逐边符号传输检查（每条 CFG 边，而非一条执行路径）**：为该边活跃值播
   种独立位符号，其余位置播种背景符号，执行该边机器码，断言每个并行拷贝/
   phi 目标收到源的位、每个穿越值保位、未读写不相关别名位置、call 破坏位
   不携带所需值；失败带**精确位置**（边、异常指令下标、机器指令下标）；
4. **栈槽重叠检查**：两个共用单元的值只允许在从不同时活跃时复用；pair 槽
   对齐；私有 scratch 槽与任何数据槽不相交；
5. **配对位级符号执行器**：在有界 DFS 上把原三地址解释器与机器程序配对，
   覆盖**正常出口与异常出口**（异常转移时该指令 def 不生效）。期望值只由
   源程序推导，**不回调被测求解器**。

`allocate()` 在返回前强制运行该检查；任一边无法验证就抛
`VerificationFailureError` 并给出失败位置，**绝不输出看似完整、实则无法
验证的机器程序**。

### 判别力

`tests/test_v2.py` 含变异用例（篡改 op 读视图、删异常边拷贝等），确认
检查器确实拒绝；`tests/test_extended.py` 对 tiny 目标的**完整合法小参考域**
逐一枚举并行拷贝排列/穿越子集，并独立核对，而不是只跑个别样例。

## 错误语义（两版本一致）

- id/寄存器/类别必须合法；**bool 一律拒绝**，NaN/±Infinity 在数值入口拒绝。
- 非法输入抛 `ProgramFormatError`（`ValueError` 子类），信息带可定位路径。
- `InfeasibleProgramError`：**证明无解**（固定碰撞、早破坏与存活输入争用
  同一别名视图等）。
- `BudgetExhaustedError`：搜索预算耗尽，**可行性未知**，不谎报无解、不产出
  机器程序。
- `VerificationFailureError`（v2）：产物未通过独立逐边检查（带位置）。
- 分配是纯函数式的；一次失败不产生部分变更。

## 与 v1 的差异及迁移办法

| 变化 | 影响 | 迁移 |
| --- | --- | ---|
| 寄存器从数字 r0..rK-1 改为目标视图 | v1 的 `fixed:{v:r}`/`clobbers` 在 v2 被拒 | v2 用 `fixedv:{v:"eax"}`/`clobbersv`，并提供 `target=` |
| 寄存器集合来源 | v2 不传 K | `allocate(prog)`（v1 仍 `allocate(prog, K)`） |
| 块上限 40 → 1024 | 仅 v2 | 新规模用 `version=2` |
| 栈槽“号=变量 id” | v2 改为对齐区间 `{v:(start,size)}` | v1 槽语义不变；v2 读 `slots[v]` |
| 新增异常边/早破坏/边拷贝 | v1 不识别 | 这些字段仅在 `version=2` 接受 |
| v2 成功返回值类型不同 | `AllocationV2` | 旧代码若依赖 v1 字段，继续用 v1 程序即可 |

`program_from_dict({...})` 通过 `"version": 2` 与 `"target"` 字段构造 v2；
省略时仍是 v1，旧 JSON 形状无需改动。

## 已知限制（不宣称完整行业标准兼容）

- 视图位窗口以“同一 bank 内的位区间”建模；不支持跨 bank 拼接的非 pair
  寻址、子寄存器跨 bank、延迟槽或指令调度。
- phi/显式边拷贝承载的值使用默认类别（普通宽度）；双宽值的跨边传递由
  寄存器对＋spill 处理，但不为 phi 结果单独指定 pair 类别。
- 精确 bank-set 染色最坏指数，受预算限制；超预算返回“未知”。
- 配对符号执行对循环有界展开（深度 120、步数 40_000），报告会标
  `bound_reached`；逐边位级检查本身是**全边、无界**的。

## 目录

```
regalloc/
  model.py       数据模型 + 严格校验 + v1/v2 字段门控 + 错误类型
  target.py      v2 目标描述：视图/别名/类别/pair/callee-save + 内置目标
  liveness.py    v1 边敏感活跃性（保留）
  liveness2.py   v2 活跃性（异常边 + 边并行拷贝）
  allocator.py   v1 分配 + 统一公开入口 allocate(program[, K])
  allocator2.py  v2 bank-set 图、染色、迭代 spill、对齐槽、allocate() 校验
  lower.py       v1 降级（保留）
  lower2.py      v2 异常分段、视图级代码、并行拷贝环（真临时寄存器/栈槽）
  symbolic.py    v1 配对符号执行器（保留）
  checker2.py    v2 独立检查器：位级机器 + 全边验证 + 配对执行
  samples.py     v1 确定性样例（保留）
  samples2.py    v2 手算样例 + 固定种子主规模生成器
tests/           快速回归（test_v2 等）+ test_extended 扩展验证
demo.py          v1 正常结果 + v2 新能力 + 棘手边界 + 真实拒绝 + 主规模
```
