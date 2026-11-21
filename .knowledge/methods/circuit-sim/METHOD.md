# 量子线路模拟

用于门序列演化、采样、可观测量及 VQE/QPE 的经典验证。按线路结构、初态和目标输出选择表示；这是方法参考，当前没有独立 VQE/QPE 子项目。

| 表示 | 适用条件与代价 | 精度与限制 |
|---|---|---|
| 完整态矢 | n 比特、G 个固定作用比特数的局域门：时间 O(G·2ⁿ)，内存 O(2ⁿ) | 有限精度下直接演化，不截断；先估算内存，不设通用比特数上限 |
| 张量网络 / MPS | 收缩顺序与图宽度决定一般线路的代价；MPS 还受所需键维 χ 限制 | 精确收缩与有限 χ 截断须区分；低深度或面积律本身不是任意线路高效可模拟的保证 |
| 稳定子 | 稳定子初态、Clifford 门及 Pauli 测量可用 tableau 高效模拟 | 非 Clifford 门或非稳定子初态需扩展算法；T 门数是线路属性，不是可随意调节的精度参数 |

验证：记录 bit→site、张量腿顺序、门序、初态、边界和观测量归一化；小系统用完整态矢对照 TN/稳定子结果。幺正演化检查范数；截断 MPS 增加 χ，Hamiltonian 的门分解另减小时间步。VQE 检查优化与表示误差，并与相同模型的 ED 能量比较。

项目基准：[dual-unitary](../../solvable/dual-unitary-circuits/ORACLE.md) 检查门与空间重排后的双重幺正性及光锥关联；[kicked Ising](../../solvable/kicked-ising-floquet/ORACLE.md) 检查 Floquet 门序和谱形状因子；[稳定子基准](../../solvable/INDEX.md) 检查 GF(2) 对易、秩与逻辑空间。遵守各基准的尺寸/时间窗；Floquet 周期不等于 PXP 连续时间。

来源：[Markov–Shi](https://arxiv.org/abs/quant-ph/0511069) 给出以树宽控制的模拟时间上界；[Vidal](https://arxiv.org/abs/quant-ph/0301063) 讨论受限纠缠下的模拟；[Gray–Kourtis](https://arxiv.org/abs/2002.01935) 讨论收缩路径优化；[Aaronson–Gottesman](https://arxiv.org/abs/quant-ph/0406196) 给出稳定子线路算法。此处链接外部原文，元数据见 [references.bib](../../references.bib)。

[方法入口](../INDEX.md) · [TEBD](../tebd/METHOD.md) · [quimb 资料](../../literature/mps-based-algorithm/10-21105-joss-00819.md)。
