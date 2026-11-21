# 方法入口

当前实现优先：[random_tfim 的自由费米子](../projects/random_tfim.md)、[PXP 的 ED](../projects/pxp_model.md)、[经典 Ising 的 MC](../projects/random_ising.md)。新增 ED 任务读仓库 method-ed 技能与 [ED 文献](../literature/ed/INDEX.md)。

下列张量网络和量子线路资料按用户要求保留，属于方法储备，不表示已在各子项目中实现。

| 张量网络方法 | 用途 |
|---|---|
| [DMRG / iDMRG / VUMPS](dmrg/METHOD.md) | 有限链基态用 MPS 变分扫描 |
| [TEBD](tebd/METHOD.md) | 将实/虚时间演化分解成局域门，逐门 SVD 截断 MPS。近邻链最直接 |
| [TDVP / MPO 时间演化](tdvp/METHOD.md) | 把 Schrödinger 方程投影到 MPS 切空间 |
| [有限温 MPS：purification / METTS](mps-finite-t/METHOD.md) | purification 从无限温物理-辅助纠缠对出发，对物理自由度演化 exp(-βH/2)，再迹掉辅助空间 |
| [LTRG / XTRG](ltrg-xtrg/METHOD.md) | 用 MPO 表示热密度矩阵 |
| [DMRG-X](dmrg-x/METHOD.md) | 在扫描时按与当前态重叠选择局部激发态，面向低纠缠的局域化本征态。与只瞄准能量的基态 DMRG 不同。 |
| [PEPS / iPEPS](peps-ipeps/METHOD.md) | 二维有限/无限张量网络变分态 |
| [TRG / HOTRG](trg-hotrg/METHOD.md) | 对经典配分函数或离散虚时量子网络做局域截断和粗粒化。保存归一化因子及杂质张量，区分自由能与局域观测量。 |
| [MERA](mera/METHOD.md) | 由 disentangler 与 isometry 构成分层网络，适合临界态和标度算符 |

[量子线路](circuit-sim/METHOD.md) · [原文索引](../literature/README.md) · [独立验证基准](../solvable/INDEX.md)。ITensors 的版本/API 查仓库 using-itensors 技能；不以历史文献推断当前软件接口。
