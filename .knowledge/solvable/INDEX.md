# 独立验证基准

保留与当前 Ising/PXP、张量网络和量子线路有关的 19 项。每项 ORACLE.md 给出边界、归一化、适用极限和 self_test；不把外部项目的历史性能记录当成本项目已运行的结果。

无序机制对照：[anderson-1d](anderson-1d/ORACLE.md)。

## 项目直接对照

[tfim-chain](tfim-chain/ORACLE.md) · [ising-2d-onsager](ising-2d-onsager/ORACLE.md) · [pxp-scars](pxp-scars/ORACLE.md) · [random-matrix-stats](random-matrix-stats/ORACLE.md)

## 张量网络精确态与能量

[aklt-chain](aklt-chain/ORACLE.md) · [aklt-honeycomb](aklt-honeycomb/ORACLE.md) · [majumdar-ghosh](majumdar-ghosh/ORACLE.md) · [motzkin-fredkin](motzkin-fredkin/ORACLE.md) · [heisenberg-xxx](heisenberg-xxx/ORACLE.md)

## 量子线路、稳定子与拓扑码

[dual-unitary-circuits](dual-unitary-circuits/ORACLE.md) · [kicked-ising-floquet](kicked-ising-floquet/ORACLE.md) · [cluster-spt](cluster-spt/ORACLE.md) · [toric-code](toric-code/ORACLE.md) · [color-code](color-code/ORACLE.md) · [quantum-double](quantum-double/ORACLE.md) · [string-net](string-net/ORACLE.md) · [haah-code](haah-code/ORACLE.md) · [x-cube](x-cube/ORACLE.md)

从仓库根目录运行（需 NumPy、SciPy、pytest；依赖见本目录 pyproject.toml）：

```sh
python .knowledge/solvable/tfim-chain/oracle.py --L 16
python -m pytest .knowledge/solvable/tests
```

引用元数据统一见 [references.bib](../references.bib)。基准之间会复用 _lib；quantum-double 还调用 toric-code，kicked-ising-floquet 调用 dual-unitary-circuits，不要单独移走依赖。
