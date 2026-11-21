# manybodyscars：历史交错形变研究

当前目录只有本地输出/缓存，没有可核对的计算源文件。依据保留的 [note0904](../../notes/note0904/scar_to_griffith.tex) 记录历史约定，不视为当前实现或可直接复现的结果。

| 内容 | 历史记录与边界 |
|---|---|
| 物理约定 | PBC、Pauli、`n=(1+Z)/2,P=(1-Z)/2`，相邻激发禁止；`H(g,r)=H_PXP+gΣ(-1)^i[Z_i+r Z_(i-1)Z_i Z_(i+1)]`。确定性交错形变，没有随机键。 |
| 基底与子空间 | 初态 `\|1010…〉`；平移只保留两格点，初态位于该平移的 k=0。笔记中的热力学汇总全部平移 sector，不能只用 scar 图的 k=0。基矢具体排序缺少源代码佐证。 |
| 验证状态 | 本地未找到对应独立测试；引用图中的数值不作为现行回归基准。复现前须找回生成代码/参数，或明确重新实现并用小系统 ED 核对。 |
| 研究决策 | 形变将高 Z2 重叠的态推向低能，探索低温慢动力学；这是工作假设，笔记也未声称已证明无无序 Griffiths 相。 |
| 文献对应 | [Turner](../literature/Turner_2018_Weak_ergodicity_breaking.md)、[Choi](../literature/Choi_2019_Emergent_SU2_Dynamics_and_Perfect_Quantum_Many_Body_Scars.md) 是 scar/FSA/SU(2) 背景；笔记自定义的 g,r 形变不等于当前 [pxp_model](pxp_model.md) 的均匀 m 项。 |
