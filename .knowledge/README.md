# 项目知识入口

只保存会影响结果解释、复现和方法选择的知识；运行参数、输出格式以各子项目 README 和源码为准。按问题打开下表的一项，不整库加载。

| 任务 | 先读 | 再查 |
|---|---|---|
| 随机键经典 Ising：MC、局域响应、临界温度 | [random_ising](projects/random_ising.md) | [源码与用法](../random_ising/README.md) |
| 随机横场链：自由费米子、能隙、关联、纠缠 | [random_tfim](projects/random_tfim.md) | [Young–Rieger](literature/Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md) |
| 纯净链：虚时关联与减项对照 | [tfim](projects/tfim.md) | 复用 random_tfim；不要沿用旧 PBC 结果 |
| PXP：受限基底、FSA、谱与热关联 | [pxp_model](projects/pxp_model.md) | [文献索引](literature/README.md) |
| 旧交错形变 PXP 结果 | [manybodyscars](projects/manybodyscars.md) | [历史笔记](../notes/note0904/scar_to_griffith.tex) |
| 张量网络或量子线路方法 | [方法索引](methods/INDEX.md) | 原文与 [独立基准](solvable/INDEX.md) |
| 单位、比较不同实现、极限核验 | [共同约定](conventions.md)、[已知极限](limits.md) | 对应项目卡与测试 |
| 临界点、Griffiths 证据或指数解释 | [判据与限制](physics/criticality/PHYSICS.md) | 对应项目卡、原始数据与文献 |

引用方向：项目卡 → 实现 / 测试 / 文献；模型卡只分派到项目卡；方法索引 → 方法摘要 → 原文 / 基准。物理定义在项目卡维护一次，不在索引重复。

`literature/` 保留原文作为证据层，索引和项目卡保持短小；同一文献仅留一份。张量网络、量子线路及其精确态/稳定子基准作为明确保留的方法储备。移除无当前用途的通用模型、杂质/电子结构方法、重复软件目录和外部项目生成页面的测试；不把这些目录当作本项目待办清单。

`src/ScarToGriffith.jl` 目前只是空模块；`julia-env/` 是运行环境；`notes/` 与 `plans/` 是历史分析/研究计划，不构成新的计算子项目。QMBS 导致 QGP 仍是待检验假设。

机制对照保留 [Anderson 局域化](models/anderson-localization/MODEL.md) 与 [MBL](models/mbl-disordered-heisenberg/MODEL.md)；[AKLT](models/aklt/MODEL.md) 是张量网络基准，均无独立计算子项目。
