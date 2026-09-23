# Random transverse-field Ising chain

复现 Young & Rieger (1996) 第 III–V 节的零温能隙与纵向关联计算。
采用仓库已有 `julia-env/local` 环境，无独立环境、无新增依赖。
核心只使用 Julia 标准库，运行脚本使用已有 HDF5 保存数据。

## 运行

从仓库根目录执行：

```sh
julia --project=julia-env/local random_tfim/test/runtests.jl
julia --project=julia-env/local random_tfim/test/smoke_io.jl
julia random_tfim/run.jl demo

# 小批量试跑；输出已存在时拒绝覆盖
julia random_tfim/run.jl gap 100 random_tfim/results/gap_small.h5
julia random_tfim/run.jl correlation 20 random_tfim/results/correlation_small.h5

# 论文级样本数；计算时间明显长于 demo，尤其是关联函数
julia random_tfim/run.jl gap 50000
julia random_tfim/run.jl correlation 10000
julia random_tfim/run.jl bimodal 50000
```

`run.jl` 自动激活 local 环境；直接 include 核心模块时由调用者选环境。
Windows 的 juliaup 执行别名若无法访问，可用实际安装目录的 `bin/julia.exe`。

```julia
include("random_tfim/RandomTFIM.jl")
using .RandomTFIM, Random, Statistics

J, h = sample_disorder(Xoshiro(1996), 64, 1.0)
state = ground_state(J, h)
state.gap, state.resolved
pair = correlations(state.G; rmax=24)

# pair.C[i,r+1] = <sigma_z(i) sigma_z(i+r)>，包含全部平移起点。
ensemble = disorder_ensemble(64, 1.0; nsamples=100, seed=1996)
average = vec(mean(ensemble.sample_C; dims=2))
mean_log = vec(mean(ensemble.sample_logC; dims=2))
typical = exp.(mean_log)
```

## 物理约定与公式

使用论文的 Pauli 约定及周期自旋边界：

$$H=-\sum_{i=1}^{L}J_i\sigma_i^z\sigma_{i+1}^z-\sum_{i=1}^{L}h_i\sigma_i^x.$$

要求偶数 $L\ge2$、$J_i\ge0$、$h_i>0$。$J_L$ 是跨边界键；$L=2$ 保留两条键。
默认 $J\sim U(0,1)$、$h\sim U(0,h_0)$，临界点 $h_0=1$，
$\delta=\frac12\ln h_0$。可选论文 Eq. (61) 的双峰分布。
模型约定也参见 [模型卡](../.knowledge/models/transverse-field-ising/MODEL.md)；
本项目的边界及随机模型细节以[论文](../papers/Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md)为准。

令 $K=A+B$（论文 Eqs. 40–41），其对角元为 $h_i$，次对角元为 $-J_i$。
周期费米子边界 $K_{1L}=-J_L$，反周期边界 $K_{1L}=+J_L$。
对 $K=U\operatorname{diag}(\epsilon)V^T$ 做 SVD 等价于求 $2L$ 维 BdG
矩阵的正能谱，且无需构造平方矩阵 $K^TK$。单个准粒子的能量代价是 $2\epsilon$。

$$E_0=-\sum_\mu\epsilon_\mu^{ap},\qquad
E_1=-\sum_\mu\epsilon_\mu^p+
\begin{cases}2\epsilon_{\min}^p,&\text{周期真空为偶宇称},\\0,&\text{周期真空为奇宇称}.\end{cases}$$

偶数链上周期真空宇称等于 $\operatorname{sgn}\det K_p
=\operatorname{sgn}(\prod_i h_i-\prod_i J_i)$，代码比较对数乘积。
不能按总体 $h_0$ 判断单个无序样本的宇称。乘积相等时存在零模，两种占据等能。

基态使用反周期扇区；由 $\phi=U^T,\psi=V^T$ 得
$G=-VU^T$（Eq. 56）。$i<j$ 时

$$C_{ij}=\det G_{i:j-1,\ i+1:j}.$$

跨越接缝时用反周期延拓 $G_{a+L,b}=-G_{ab}$；这样只需计算长度
$r\le L/2$ 的短路径行列式。自关联严格取 1。
`logabsdet` 保留小关联的对数；负行列式不取绝对值伪装成物理结果。

## 输出及论文对照

核心函数只返回后续计算所需的数据：

| 函数 | 返回值 |
|---|---|
| `sample_disorder` | `J, h` |
| `energy_gap` | `gap, resolved` |
| `ground_state` | `gap, resolved, G` |
| `correlations` | `C, logC` |
| `disorder_ensemble` | `gaps, resolved, sample_C, sample_logC, pair_logC` |

以上均为命名元组。`pair_logC` 默认是空数组，仅在 `keep_pairs=true` 时保留。
输入参数不再重复返回；距离、平均值、典型值和标准误在分析或保存时计算。
`C` 保留行列式的符号，`logC` 保留微小关联的对数精度，因此两者均保留。

HDF5 按 `L16/h1.0` 等分组；包含原始 gap、log gap、有效标记和距离数组。
`run.jl` 从逐样本数据计算派生统计量，保留原有 HDF5 数据集名称。
`sample_C`、`sample_logC` 的列为无序样本，行对应 `r+1`。
`average` 是所有样本和起点的 $[C]_{av}$；`mean_log` 是 $[\ln C]_{av}$，
`typical=exp(mean_log)`。`sem`、`log_sem` 按独立无序样本计算，
不会把同一样本内不同起点当作独立样本。无效对数会传播，避免静默选择偏差。
`correlation` 模式额外保留 `pair_logC[起点,r+1,样本]`，用于关联分布。

| 论文图 | 从输出构造 |
|---|---|
| 1–4 | $\ln\Delta E$ 分布；临界 $\ln\Delta E/\sqrt L$；$h_0=3$ 用 $\ln\Delta E+z\ln L$，$z\approx1.4$ |
| 6–7 | 双峰分布的临界能隙及相同激活标度 |
| 8–9 | `average` 对 $r$ 双对数图，`mean_log` 对 $\sqrt r$；排除 $r=0$ |
| 10–12 | 固定距离的 `pair_logC` 分布及除以 $\sqrt r$ 后的分布 |
| 13–16 | 非临界 `average` 除以临界值；`mean_log` 减去临界值；横轴分别为 $r\delta^2$、$2r\delta$ |
| 17–19 | $h_0=3$ 的 `pair_logC` 分布、均值及方差随距离的变化 |

Histogram 密度须按实际 bin 宽归一化；对 gap 报告删失比例，避免把过滤后的
尾部当作完整分布。跨样本的关联分布误差应按样本分块，而非对起点独立 bootstrap。
这里提供计算与原始数据，不预先拟合指数或宣称已完成论文统计精度复现。

## 数值与性能

核心函数无全局可变状态、不修改输入；局部矩阵原位分解以减少分配。
采用 Float64/LAPACK，单样本 SVD 为 $O(L^3)$。
所有起点、所有距离的独立 LU 为 $O(L r_{max}^4)$，它是完整关联扫描的主要成本；
可减少 `rmax`，或只做 gap（`rmax=0`，不计算特征向量）。
保留全部 pair 对数需要 $8L(r_{max}+1)N_s$ 字节，$L=128,N_s=10000$ 约 666 MB；
交互调用默认不保留。运行脚本固定单 BLAS 线程，顺序使用显式种子。

两个总能量的差在临界大系统会遭受消减，论文 Fig. 1 也指出此截断。
`resolution=64eps(Float64)*max(abs(E0),abs(E1),1)` 是保守诊断尺度，非严格误差界。
不超过此值标记 `resolved=false`，保留未经裁剪的原始 gap。
能量、阈值和宇称仅作为内部变量；保存时计算 `loggaps`，未分辨样本记为 `NaN`。
这不支持可信的任意小 gap 尾部；如需突破双精度截断，应另行使用高精度算法。
关联的 log determinant 防止下溢，但不能恢复收缩矩阵已经损失的相对精度。

`test/runtests.jl` 独立构造 $2^L$ 自旋哈密顿量，验证纯净/随机、
有序/临界/无序参数下的 $\Delta E$ 和全部周期起点关联，
也验证 $J=0$ 极限、可重复采样及核心返回值类型推断。
运行产物中的 `complete` 仅在所有参数完成后置 true；输出路径已存在时请另选文件。

已在 Julia 1.13.0 / local 环境中通过小系统 ED 验证及 HDF5 写入回读验证。
单 BLAS 线程的 $L=128$ 全起点、$r\le64$ 关联试跑约 0.14 秒/样本
（编译后，本机测量；不包括文件写入）。这里只实际运行了小批量验证，
未运行 50000/10000 样本的完整论文扫描。
