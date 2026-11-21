# TEBD

将实/虚时间演化分解成局域门，逐门 SVD 截断 MPS。近邻链最直接；长程项或 PXP 投影项需要正确分组/MPO，不可直接套两体门。

验证与限制：分别减小时间步、增加 χ/降低截断阈值，检查 norm、能量及目标关联；用小链精确演化核对。纠缠增长限制可达时间，单步 discarded weight 不是总观测量误差。

[主要来源](../../literature/mps-based-algorithm/10-1103-physrevlett-93-040502.md) · [方法入口](../INDEX.md) · [独立基准](../../solvable/INDEX.md)。

这是保留的方法参考，不表示当前子项目已经实现；API 用仓库的 using-itensors 技能和已安装版本核对。
