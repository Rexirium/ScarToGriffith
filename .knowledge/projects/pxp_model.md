# pxp_model：周期受限链、FSA 与热关联

入口：[pxp_basis.py](../../pxp_model/pxp_basis.py) · [谱](../../pxp_model/pxp_spectrum.py) · [参数扫描](../../pxp_model/pxp_scan.py) · [热关联](../../pxp_model/pxp_autocorrelation.py)。

| 内容 | 必须保留的约定与依据 |
|---|---|
| 物理约定 | `H=Σ P_(i-1) X_i P_(i+1)-mΣ Z_i`，`P=(1-Z)/2`，Pauli 驱动系数与 ħ 为 1；bit 1 对应 Z=+1/激发，PBC 禁止相邻 1（包括首尾）。文献若用 `-ΔΣn`，则 Δ=2m，本实现多常数 mL。 |
| 基底与对称性 | site 0 是最高位，数组顺序以 `basis.states` 为准，不按整数升序猜。无对称性空间维数 `F_(L-1)+F_(L+1)`。一步平移本征值 `exp(-2πik/L)`，反射 `i→L-1-i`；联合 k,p 仅在 k=0 或偶链 k=L/2 可用，普通 k 的反射会交换 ±k。 |
| 初态与 sector | 谱脚本将 `\|1010…〉` 投影到所选空间并归一化；k=0/p=+ 得 `\|Z2,+〉`，k=π/p=− 得差态。零投影的动力学为 NaN。扫描固定偶链 k=0,p=+；简并子空间按初态重叠旋转（能量容差 `1e-10`），不混合不同块。纠缠前投回无空间对称性的受限基底，半链熵用自然对数。 |
| 热态与时间 | 谱/扫描的热参照限制在所选 sector；热关联使用整个受限空间所有动量、两种反射宇称，配分函数全局归一化。`C_average` 对 M=ΣZ/L；`C_local` 是各格点局域 Z 自关联的平均，两者都减热平均平方。T=0 等权混合数值简并基态，T=∞ 均匀迹。虚时网格 T>0 重标为 `ts/max(ts)/(2T)`，T=0 保留输入。 |
| 数值验证 | [tests](../../pxp_model/tests/) 用独立全自旋矩阵投影核对 H、各 sector 拼回全谱、`P†P=I`、`HP=PH_sector`（常用 atol `1e-12`），并对小系统直接热谱和/时间演化核验。生产谱检查本征残差 `≤1e-9 max(1,max\|E\|)`；维数检查与约束检查不可省。 |
| FSA 决策 | [Turner 2018](../literature/Turner_2018_Weak_ergodicity_breaking.md) Eqs. (3)–(5) → `construct_fsa_eigensystem`：H+ 降低偶位激发、升高奇位空位，构造 L+1 层；hopping 求和止于 L−1。本项目增加 `-m〈ΣZ〉` 对角项；先构造全空间 FSA 再投影，保留权重 >1/2 的态并归一化。FSA 是近似，不把候选态当精确 scar。 |
| 匹配与统计 | 扫描 `matched_scar_indices` 对每个 FSA 态独立取最大重叠；`tracked_scar_indices` 从 m=0 独立追踪，允许重复，不是一一分配。能级统计先按对称性块取中央 4/5、局域展开（window=5），丢弃间隙 ≤1e-8 后归一化，再合并分布；不直接对拼接谱求间距。 |

[Choi 2019](../literature/Choi_2019_Emergent_SU2_Dynamics_and_Perfect_Quantum_Many_Body_Scars.md) 的 emergent SU(2)/复苏机制用于解释 FSA；当前 `-mΣZ` 不是其完美复苏形变的实现。`m=0.655` 是扫描中特别采样的点，不是这段代码独立确定的临界点。长寿命振荡、低纠缠异常态或幂律拟合都不能单独证明 QGP，见 [共同判据](../physics/criticality/PHYSICS.md)。

[manybodyscars 历史形变](manybodyscars.md) 使用两格点平移和交错场，与这里的一格点平移/均匀 m 不同。
