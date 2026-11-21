# DMRG / iDMRG / VUMPS

有限链基态用 MPS 变分扫描；无限链用固定晶胞 iDMRG/VUMPS。有限键维 χ 是主要近似，二维圆柱还须检查宽度。

验证与限制：增加 χ、扫描数，检查能量、能量方差和目标观测量；比较不同初态。能量趋稳本身不足以证明关联或激发收敛。固定 sector/晶胞会排除候选态，必须记录。

[主要来源](../../literature/mps-based-algorithm/1008.3477_the-density-matrix-renormalization-group-in-the-age-of-matri.md) · [无限 MPS 来源](../../literature/mps-based-algorithm/1701.07035_variational-optimization-algorithms-for-uniform-matrix-produ.md) · [方法入口](../INDEX.md) · [独立基准](../../solvable/INDEX.md)。

这是保留的方法参考，不表示当前子项目已经实现；API 用仓库的 using-itensors 技能和已安装版本核对。
