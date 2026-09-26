# Random transverse-field Ising chain

## 边界条件选择

能隙、空间关联、自关联和无序采样均支持 `boundary=:open` 或 `:periodic`。
开边界输入 `J` 长度为 `L-1`，周期边界为 `L`；`h` 长度均为 `L`。
单样本 `autocorrelation` 默认 `:open`，统一的 `disorder_ensemble` 及其余接口默认 `:periodic`。
同一次计算请显式传递同一个边界参数，尤其是 `ground_state` 与 `correlations`。

```julia
include("random_tfim/RandomTFIM.jl")
using .RandomTFIM, Random

boundary = :periodic # 改为 :open 即切换整套计算
J, h = sample_disorder(Xoshiro(1996), 32, 1.0; boundary)
gap = energy_gap(J, h; boundary)
state = ground_state(J, h; boundary)
pair = correlations(state.G; boundary, rmax=16)
dynamic = autocorrelation(J, h, collect(0.0:0.25:10.0); boundary)
```

两种边界使用相同 seed 时生成相同的横场和内部键；开边界仅去掉接缝键。
空间关联的默认最大距离仍为 `L÷2`；开边界可显式设 `rmax=L-1`。
开边界的 `C[i,r+1]`、`logC[i,r+1]` 仅在 `i+r≤L` 时有效，
其余位置为 `NaN`。无序样本均值按每个距离的 `L-r` 个有效起点计算，
不将越界位置计入平均，也不忽略有效格点对中的异常对数。

命令行仅保留 `demo` 和 `full`，第四个参数选择边界，两种模式均默认周期边界：

```sh
julia random_tfim/run.jl demo 20 random_tfim/results/demo_open.h5 open
julia random_tfim/run.jl full 100 random_tfim/results/full_periodic_small.h5 periodic
```

| 模式 | 默认样本数 | 链长 $L$ |
|---|---|---|
| `demo` | 20 | 16, 32 |
| `full` | 10000 | 16, 32, 64, 128 |

两种模式的横场参数均为 `h0=(1.0,1.3,1.5,1.7,2.0,2.3,3.0)`，
时间网格均为 `0:0.25:10`。每个模式都计算能隙、`r≤L÷2` 的空间关联和中点时间自关联，
并输出各自的无序平均及标准误。第二个参数可覆盖样本数，便于在 `full` 尺寸上小批量试跑。

## 中点实时间自关联（零温）

新增 `autocorrelation` 使用 Majorana 协方差矩阵与 Pfaffian 计算
$C_j(t)=\langle0|\sigma_j^z(t)\sigma_j^z(0)|0\rangle$，取 $\hbar=1$。
支持开边界和周期边界，保留偶数 `L≥2` 的约定，默认 `j=L÷2`。
开链取两个中点中的左侧；周期链的 `j` 是指定的测量格点。

这里的 `C` 同时就是 connected correlation
$A_j(t)=C_j(t)-\langle\sigma_j^z(t)\rangle\langle\sigma_j^z(0)\rangle$：
有限链、严格正横场下取确定宇称的基态，$\mathbb Z_2$ 对称性给出
$\langle\sigma_j^z\rangle=0$，无需另减磁化项。
当前接口只实现已确认的零温平稳基态，不接受热态、quench 初态或破缺对称态。
无序平均先逐样本计算关联，再平均结果，不先平均耦合常数。

```julia
include("random_tfim/RandomTFIM.jl")
using .RandomTFIM, Random, Statistics

J, h = sample_disorder(Xoshiro(1996), 32, 1.0)
times = collect(0.0:0.25:10.0)
result = autocorrelation(J[1:end-1], h, times) # 明确去掉周期接缝键
C = result.C                                # ComplexF64，含完整相位
ensemble = disorder_ensemble(32, 1.0; times, nsamples=20, seed=1996, boundary=:open)
average = vec(mean(ensemble.sample_Ct; dims=2))
```

`disorder_ensemble` 在同一个无序样本循环中只抽样一次 `J,h`，依次计算能隙、
空间关联和时间关联。返回的各个数组第 `n` 列（能隙为第 `n` 个元素）对应同一构型。
`sample_C`、`sample_logC` 保留空间关联；`sample_Ct[时间,样本]` 保存复数时间关联。
`times` 默认为空，此时跳过动态计算，`sample_Ct` 大小为 `0×nsamples`；
`rmax=0` 可跳过非平凡空间关联，两者结合即只计算能隙。
测量位置通过 `j` 指定，默认 `L÷2`。原独立的时间关联 ensemble 入口已移除。

开边界算法按以下 string 约定，选
$\gamma_{2k-1}=(\prod_{\ell<k}\sigma_\ell^x)\sigma_k^z$、
$\gamma_{2k}=(\prod_{\ell<k}\sigma_\ell^x)\sigma_k^y$，于是
$\sigma_j^z=i^{j-1}\gamma_1\cdots\gamma_{2j-1}$。定义
$H=(i/4)\gamma^T A\gamma$，其中
$A_{2k-1,2k}=-2h_k$、$A_{2k,2k+1}=-2J_k$，其余由反对称性确定。
频率因而为 `2ε`，与本项目的 Pauli 归一化一致。

令 $K=U\operatorname{diag}(\epsilon)V^T$，其中 $K_{kk}=h_k$、
$K_{k+1,k}=-J_k$。零温协方差
$\Gamma_{ab}=(i/2)\langle[\gamma_a,\gamma_b]\rangle$ 的奇偶块为
$\Gamma_{\mathrm{odd,even}}=UV^T$，偶奇块为其负转置，其余为零。
将 Majorana 重排为“全部奇指标、全部偶指标”后，
$$
A=\begin{pmatrix}0&-2K\\2K^T&0\end{pmatrix}.
$$
因此代码对 `L×L` 的 `K` 做一次 SVD，等价于对 `2L×2L` 的 `iA` 做谱分解；
后者的特征值为 $\pm2\epsilon_\mu$。文中若以 $\epsilon_\mu$ 表示 `iA`
的正特征值，则它是代码中奇异值的两倍。
这仍是 Majorana 协方差方法，满足 $\Gamma=i\operatorname{sgn}(iA)$，
且每个时间点只更新三角函数，不重复对角化。
由 $R(t)=e^{At}$ 与 $S=1:2j-1$ 构造

$$
Q(t)=[R(t)(I-i\Gamma)]_{SS},\qquad
C_j(t)=(-1)^{j-1}\operatorname{Pf}
\begin{pmatrix}
-i\Gamma_{SS}&Q(t)\\
-Q(t)^T&-i\Gamma_{SS}
\end{pmatrix}.
$$

两块等时收缩相同是因为初态为平稳基态。Pfaffian 矩阵大小为
$2(2j-1)$，中点时为 $2L-2$。计算 `Q` 时先对全部 `2L` 个模式求和，
再截取两个端点；不能将 `R` 和 `Γ` 都提前截成 `SS` 块再相乘。
算法使用带主元交换的反对称消元，不用 `sqrt(det)`，以保留符号与复数相位。

```sh
julia random_tfim/run.jl demo 20 random_tfim/results/demo_small.h5
```

该试跑计算 `L=(16,32)`、上述全部 7 个横场值、`t=0:0.25:10`，默认 20 个无序样本，
同时保存这些构型的能隙及 `r≤L÷2` 的空间关联。
任意时间网格和样本数可通过上述函数接口指定。HDF5 的各个 `L…/h…` 组保存
`times`、`sample_C_real/imag`（时间 × 样本）、`average_real/imag`、
`sem_real/imag`；误差分别对实部、虚部按独立无序样本计算，单样本时为 `NaN`。
文件元数据记录所选边界、零温实时间定义及中点位置。
实时间关联通常是复数，因此不自动取绝对值或计算 `log C`。

计算成本约为每个样本 $O(L^3+N_tL^3)$，工作内存为 $O(L^2)$，
另需保存 $N_tN_s$ 个复数样本。当前使用双精度直接 Pfaffian；极小关联可能下溢，
长时间的相位精度受频率误差限制，不保证极端无序下的任意小能标精度。

### 周期边界的宇称切换与 Pfaffian

周期自旋链基态在偶宇称（反周期费米子）扇区，而插入 $\sigma_j^z$ 后进入奇宇称
（周期费米子）扇区。分别对 $K_+$、$K_-$ 做 SVD，不能只在上述开链公式中更换接缝。
循环重排该样本的耦合和横场，使测量点成为 JW 起点，此时 $\sigma_j^z=\gamma_1$。
令 $|\phi\rangle=\gamma_1|0,+\rangle$，则

$$
C_j(t)=e^{iE_0t}\langle\phi|e^{-iH_-t}|\phi\rangle,
\qquad E_0=-\sum_\mu\epsilon_\mu^+.
$$

这是逐样本重排，不假设随机链平移对称。设 $\Gamma^+$ 的奇偶块为 $U_+V_+^T$，
对该块第一行取负即得 $\Gamma^\phi_{\mathrm{odd,even}}$。
在 $K_-=U_-\operatorname{diag}(\epsilon^-)V_-^T$ 的模式基底中，记
$B=U_-^T\Gamma^\phi_{\mathrm{odd,even}}V_-$，于是

$$
H_-=-i\sum_\mu\epsilon_\mu^-\alpha_\mu\beta_\mu,\qquad
e^{-iH_-t}=\prod_\mu
\left[\cos(\epsilon_\mu^-t)-\sin(\epsilon_\mu^-t)\alpha_\mu\beta_\mu\right].
$$

对乘积逐项使用 Wick 定理，可将期望写成一个 $2L\times2L$ Pfaffian。
按 $(\alpha_1,\beta_1,\ldots,\alpha_L,\beta_L)$ 排序，构造

$$
W_{2\mu-1,2\nu}
=\delta_{\mu\nu}\cos(\epsilon_\mu^-t)
+i\sin(\epsilon_\mu^-t)B_{\mu\nu},\qquad
W_{2\nu,2\mu-1}=-W_{2\mu-1,2\nu},
$$

同奇偶块为零，最终 $C_j(t)=e^{iE_0t}\operatorname{Pf}W(t)$。
此处使用 $\epsilon^-t$ 是多体演化算符的展开；Majorana 向量演化的频率仍是 $2\epsilon^-$。
公式保留奇宇称态和真空能量相位，不需要假设周期费米子真空本身为奇宇称，
也不使用平方根或按时间步追踪符号。负时间、乱序时间和零单粒子模式均可直接计算。

`test/runtests.jl` 对 `L=2,4,6` 的全部格点，用独立自旋哈密顿量的谱表示
检查纯净/随机及有序/临界/无序参数，并验证 `C(0)=1`、
`C(-t)=conj(C(t))`、`|C(t)|≤1` 和 `J=0` 时的 `exp(-2im*h[j]*t)`。
`test/smoke_io.jl` 检查驱动程序、数据回读、无序平均和标准误。

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
julia random_tfim/run.jl demo 20 random_tfim/results/demo_small.h5
julia random_tfim/run.jl full 2 random_tfim/results/full_smoke.h5

# 完整默认规模，同时计算全部观测量；计算时间明显长于 demo
julia random_tfim/run.jl full
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

使用论文的 Pauli 约定。周期自旋边界下：

$$H=-\sum_{i=1}^{L}J_i\sigma_i^z\sigma_{i+1}^z-\sum_{i=1}^{L}h_i\sigma_i^x.$$

要求偶数 $L\ge2$、$J_i\ge0$、$h_i>0$。$J_L$ 是跨边界键；$L=2$ 保留两条键。
开边界的相互作用求和上限为 $L-1$，不含接缝键；$L=2$ 只有一条键。
默认 $J\sim U(0,1)$、$h\sim U(0,h_0)$，临界点 $h_0=1$，
$\delta=\frac12\ln h_0$。
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

周期链基态使用反周期扇区；开链直接对无接缝的 $K$ 做 SVD，
其 $E_0=-\sum_\mu\epsilon_\mu$、$\Delta E=2\min_\mu\epsilon_\mu$，无需宇称修正。
两种边界均由 $\phi=U^T,\psi=V^T$ 得
$G=-VU^T$（Eq. 56）。$i<j$ 时

$$C_{ij}=\det G_{i:j-1,\ i+1:j}.$$

周期链跨越接缝时用反周期延拓 $G_{a+L,b}=-G_{ab}$；这样只需计算长度
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
| `disorder_ensemble` | `gaps, resolved, sample_C, sample_logC, pair_logC, sample_Ct` |
| `autocorrelation` | `C`（复数时间序列） |

以上均为命名元组。`pair_logC` 默认是空数组，仅在 `keep_pairs=true` 时保留。
输入参数不再重复返回；距离、平均值、典型值和标准误在分析或保存时计算。
`C` 保留行列式的符号，`logC` 保留微小关联的对数精度，因此两者均保留。

HDF5 按 `L16/h1.0` 等分组；包含原始 gap、log gap、有效标记和距离数组。
能隙新增 `gap_average`、`gap_sem`；对数能隙新增 `loggap_average`、`loggap_sem`。
能隙均值保留所有原始样本；若任一样本的 gap 未分辨，其 log gap 为 `NaN`，
对数能隙统计随之为 `NaN`，不通过删去样本改变平均的定义。
空间关联组的 `pair_counts` 记录每个距离的有效起点数：周期为 `L`，开边界为 `L-r`。
根属性 `boundary` 记录实际使用的边界。下表的论文对照采用周期边界。
`run.jl` 从逐样本数据计算派生统计量，保留原有 HDF5 数据集名称。
`sample_C`、`sample_logC` 的列为无序样本，行对应 `r+1`。
动态数据在函数返回中名为 `sample_Ct`；HDF5 沿用 `sample_C_real/imag`、
`average_real/imag`、`sem_real/imag`，与空间数据集 `sample_C` 区分。
`average` 是所有样本和起点的 $[C]_{av}$；`mean_log` 是 $[\ln C]_{av}$，
`typical=exp(mean_log)`。`sem`、`log_sem` 按独立无序样本计算，
不会把同一样本内不同起点当作独立样本。无效对数会传播，避免静默选择偏差。
`typical_lower/upper=exp(mean_log∓log_sem)` 表示对数均值上下一个标准误经指数映射后的范围，
并非指定置信水平的置信区间。两种模式均保留 `pair_logC[起点,r+1,样本]`，用于关联分布。
所有 SEM 都使用无序样本间的无偏样本方差，形式为 `std(samples)/sqrt(nsamples)`；
单样本的 SEM 及由其构造的范围为 `NaN`。时间关联的实部和虚部分别统计。

| 论文图 | 从输出构造 |
|---|---|
| 1–4 | $\ln\Delta E$ 分布；临界 $\ln\Delta E/\sqrt L$；$h_0=3$ 用 $\ln\Delta E+z\ln L$，$z\approx1.4$ |
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

周期边界下两个总能量的差在临界大系统会遭受消减，论文 Fig. 1 也指出此截断。
`resolution=64eps(Float64)*max(abs(E0),abs(E1),1)` 是保守诊断尺度，非严格误差界。
不超过此值标记 `resolved=false`，保留未经裁剪的原始 gap。
开链直接使用最小奇异值计算 gap，诊断阈值取 `64eps(Float64)*max(sum(ε),1)`；
该阈值同样是保守尺度，并非严格误差界。
能量、阈值和宇称仅作为内部变量；保存时计算 `loggaps`，未分辨样本记为 `NaN`。
这不支持可信的任意小 gap 尾部；如需突破双精度截断，应另行使用高精度算法。
关联的 log determinant 防止下溢，但不能恢复收缩矩阵已经损失的相对精度。

`test/runtests.jl` 独立构造 $2^L$ 自旋哈密顿量，验证纯净/随机、
有序/临界/无序参数下的 $\Delta E$、周期空间关联和开边界有效空间关联，
并检查两种边界的动态关联，以及去掉周期接缝后恢复开链结果。
也验证 $J=0$ 极限、可重复采样及核心返回值类型推断。
运行产物中的 `complete` 仅在所有参数完成后置 true；输出路径已存在时请另选文件。

已在 Julia 1.13.0 / local 环境中通过小系统 ED 验证及 HDF5 写入回读验证。
单 BLAS 线程的 $L=128$ 全起点、$r\le64$ 关联试跑约 0.14 秒/样本
（编译后，本机测量；不包括文件写入）。这里只实际运行了小批量验证，
未运行 `full` 默认 10000 样本的完整扫描。
