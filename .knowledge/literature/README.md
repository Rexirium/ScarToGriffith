# 文献与实现

原 papers/ 的四篇全文统一存放在这里；Young–Rieger 的图 1–23 改为短文字描述，原锚点保留。论文是证据来源，项目改动见对应卡。

| 文献 | 读取任务 / 对应实现 |
|---|---|
| [Young–Rieger 1996](Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md) | [随机 TFIM](../projects/random_tfim.md)：自由费米子、G 矩阵、能隙和关联分布；注意 PBC→OBC 改动 |
| [Turner 2018](Turner_2018_Weak_ergodicity_breaking.md) | [PXP](../projects/pxp_model.md)：FSA Eqs. (3)–(5)、scar 重叠与复苏 |
| [Choi 2019](Choi_2019_Emergent_SU2_Dynamics_and_Perfect_Quantum_Many_Body_Scars.md) | SU(2)/完美复苏背景；不等同当前均匀 m 形变 |
| [Münster 2024](Munster_2024_Griffiths_phase_large_deviations.md) | [经典随机 Ising](../projects/random_ising.md)：稀有区与磁化率分布；当前代码没有其大偏差抽样 |

保留的方法原文：[ED](ed/INDEX.md)、[MPS / DMRG / TDVP / 软件](mps-based-algorithm/INDEX.md)、[PEPS / CTMRG / 自动微分](peps-based-algorithm/INDEX.md)、[LTRG / XTRG](ltrg/INDEX.md)。MERA、HOTRG、有限温 MPS、DMRG-X 的原文在对应 [方法卡](../methods/INDEX.md) 旁；量子线路的外部原文入口见 [线路方法卡](../methods/circuit-sim/METHOD.md)，精确 MPS、线路与稳定子验证见 [基准索引](../solvable/INDEX.md)。

原文就近保留一份；通用软件目录中的重复副本已合并。BibTeX 统一在 [references.bib](../references.bib)，论文自身的编号文献表仍留在原文内。
