# 项目相关的极限与基准

| 对象 | 可检查的结果 | 适用范围与入口 |
|---|---|---|
| 纯净 TFIM | J=0：E0=−Σh，C_j(τ)=exp(−2h_jτ)；h→0⁺ 有限连通链猫态的非平凡切分熵趋于 ln2 | [random_tfim 测试](../random_tfim/test/physics.jl)，前式为未减项关联 |
| 均匀临界 TFIM | h=J=1 热力学极限 e0=−4/π | [独立 PBC oracle](solvable/tfim-chain/ORACLE.md)；当前开链有边界修正 |
| 经典 Ising | 所有键 0：C=0、χ=L²/T、U4=2/(3L²) | [MC 枚举测试](../random_ising/test_observables.jl)；有限采样容差见项目卡 |
| PXP | PBC 维数 F_(L−1)+F_(L+1)，OBC 为 F_(L+2) | [项目基底测试](../pxp_model/tests/test_pxp_basis.py)；边界不能混用 |
| TN / 量子线路 | AKLT/MPS、二维纯 Ising、精确 scar、dual-unitary、稳定子代数 | [保留基准索引](solvable/INDEX.md)，先读各卡的归一化与边界 |

误差阈值属于具体测试和参数范围，不是所有尺寸、时间、无序强度的精度保证。
