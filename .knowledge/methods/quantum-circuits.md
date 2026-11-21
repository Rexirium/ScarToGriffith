# 量子线路：保留的方法参考

用于门序列演化、采样、可观测量与 VQE/QPE 类算法的经典验证；仓库当前没有独立 VQE/QPE 计算子项目。

| 表示 | 适用条件 | 需要核验 |
|---|---|---|
| 完整态矢 | n 比特内存 O(2ⁿ)，局域门序列约 O(门数·2ⁿ) 时间 | 比特/张量腿次序、门应用顺序、norm、幺正性、小系统矩阵对照 |
| 张量网络/MPS 收缩 | 代价由收缩图宽度、键维和纠缠决定；低深度本身不是任意几何下的效率保证 | χ、截断权重、收缩精度；若门来自 Hamiltonian 分解，还检查时间步 |
| 稳定子 | Clifford 线路可高效模拟；非 Clifford 门需额外资源 | GF(2) 对易/秩、生成元独立性、逻辑空间维数，不能用纯 Clifford 算法冒充通用门模拟 |

保留来源：原方法综述中的线路分类；[Markov–Shi](https://arxiv.org/abs/quant-ph/0511069)、[Aaronson–Gottesman](https://arxiv.org/abs/quant-ph/0406196) 的条目来自原知识库。相关本地实现资料见 [quimb](../literature/mps-based-algorithm/10-21105-joss-00819.md) 和 [TEBD](tebd/METHOD.md)。

本地可核对基准：[dual-unitary](../solvable/dual-unitary-circuits/ORACLE.md) 的 U 与空间重排 Ũ 双重幺正性、光锥关联；[kicked Ising](../solvable/kicked-ising-floquet/ORACLE.md) 的 Floquet 门序与谱形状因子；[稳定子与拓扑码](../solvable/INDEX.md) 的代数检查。各卡有限尺寸/时间窗限制必须保留；离散 Floquet 周期不等于 PXP 连续时间。
