# Repository Guidelines

## Research context

ScarToGriffith explores the relation between quantum many-body scars (QMBS) and
quantum Griffiths behavior (QGP). A causal link is a hypothesis, not an established
result. Distinguish model assumptions, numerical observations and physical claims.
Keep the exploratory structure; add shared APIs only when actual reuse needs them.

## 资料在哪里，什么任务需要读取

`.knowledge/` 是按需查询的项目资料，不是默认上下文、课程或任务清单。
先明确问题；通常只读入口和对应项目卡，不递归读取整个目录。

| 任务 | 必须先查的资料 |
|---|---|
| 不确定项目/资料关系 | [.knowledge/README.md](.knowledge/README.md) |
| 改 random_ising 的 MC、响应、Tc 或解释结果 | [random_ising 项目卡](.knowledge/projects/random_ising.md)，再查其 README、实现与测试 |
| 改 random_tfim 的基态、能隙、关联、熵、采样或拟合 | [random_tfim 项目卡](.knowledge/projects/random_tfim.md)；改文件输出另查其 README 的 HDF5 部分 |
| 改纯净 tfim 对照 | [tfim 项目卡](.knowledge/projects/tfim.md)，并查它复用的 random_tfim 定义 |
| 改 PXP 基底、sector、FSA、谱、初态或热关联 | [pxp_model 项目卡](.knowledge/projects/pxp_model.md) |
| 使用旧 manybodyscars 数据/笔记 | [历史项目卡](.knowledge/projects/manybodyscars.md)，不得当成当前均匀 m 模型 |
| 比较不同代码/论文的物理量 | [conventions.md](.knowledge/conventions.md)、[limits.md](.knowledge/limits.md) 与对应项目卡 |
| 选择/实现张量网络或量子线路方法 | [方法索引](.knowledge/methods/INDEX.md)，再读选中方法与来源；这些资料明确保留 |
| 使用独立解析/精确态/线路基准 | [solvable/INDEX.md](.knowledge/solvable/INDEX.md)，先核对边界与归一化，再运行对应 self_test |
| 复现文献公式/图形或补引用 | [文献索引](.knowledge/literature/README.md) → 指定原文；BibTeX 在 [.knowledge/references.bib](.knowledge/references.bib) |
| 讨论临界点、Griffiths 相、幂律或 QMBS→QGP | [证据边界](.knowledge/physics/criticality/PHYSICS.md)，结合项目卡和原始数据 |

论文原文仅在公式/证据需要时读取；`papers/` 已整合进 `.knowledge/literature/`。
不要把历史笔记、外部基准的结果或文献中的参数，描述为本项目本次已验证。
版本敏感或有争议的结论在需要时核查一手来源。引用影响结论的资料文件。
变更物理定义、基底、验证标准或研究决策时，同步更新对应项目卡及受影响链接；
卡片只记录影响正确性的信息，运行说明和完整输出字段留在子项目 README。

## Layout and commands

- Active calculations: `random_ising/`, `random_tfim/`, `tfim/`, `pxp_model/`.
- `manybodyscars/` holds historical local outputs; `notes/` and `plans/` retain research history.
- `src/ScarToGriffith.jl` is currently an empty plain Julia module, not a package API.
- Julia environments are `julia-env/local/` and `julia-env/server/`; use
  `julia --project=julia-env/local` locally and the server environment on Slurm.
  Instantiate with `julia --project=julia-env/local -e 'using Pkg; Pkg.instantiate()'`.
- Test commands and dependencies live beside each subproject; run checks appropriate
  to the change. Keep datasets, figures and machine settings untracked.

## Development and skills

- Use four-space indentation and standard Julia naming; prefer small functions.
- Record physical meaning, units, tolerances and seeds; preserve validation and errors.
- Reuse existing implementations and add focused tests for nontrivial new behavior.
- Do not manually edit Julia Manifest.toml; use Pkg in the appropriate environment.
- Repository skills live in `.agents/skills/`; use the matching quantum-model,
  method-ed, scaling-fit or software skill when relevant. Unlisted models have no
  maintained project card; do not assume deleted generic catalog entries still exist.
- Create/update repository-specific skills in place under `.agents/skills/`, not
  a staging skill directory. Cross-project skills belong under `%USERPROFILE%/.codex/skills/`.
