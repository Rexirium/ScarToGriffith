# 随机横场 Ising 链

计算有限链零温能隙、纵向空间关联，以及实时间/虚时间自关联。
核心采用自由费米子 SVD，支持开边界和周期边界；无序样本可按 Julia 线程并行。
模型与论文对照见 [Young & Rieger (1996)](../papers/Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md)
和[模型卡](../.knowledge/models/transverse-field-ising/MODEL.md)。

## 快速运行

以下命令从仓库根目录执行。核心模块仅依赖 Julia 标准库；
[run.jl](run.jl) 自动激活已有的 `julia-env/local` 环境，使用 HDF5 保存结果。

```sh
# 小批量试跑，周期边界；输出路径已存在时拒绝覆盖
julia --threads=4 random_tfim/run.jl demo 2 random_tfim/results/demo_small.h5 periodic

# full 的全部链长，先用少量样本检查
julia --threads=4 random_tfim/run.jl full 2 random_tfim/results/full_smoke.h5 open

# 完整默认规模
julia --threads=4 random_tfim/run.jl full
```

命令格式为 `run.jl [demo|full] [samples] [output.h5] [open|periodic]`。
省略参数时使用 `demo`、该模式的默认样本数、`random_tfim/results/<mode>.h5` 和周期边界。

| 模式 | 每个 `(L,h0)` 的默认样本数 | 链长 L |
|---|---:|---|
| `demo` | 20 | 16, 32 |
| `full` | 10000 | 16, 32, 64, 128 |

两种模式均扫描 `h0 = 10 .^ range(-1, 1, 101)`，计算能隙、`r=0:L÷2` 的空间关联，
以及中点 `j=L÷2`、`tau=0:0.2:20` 的虚时间自关联。
自定义链长、时间网格或实时间演化时，使用下面的函数接口。

### 绘制已有数据

绘图使用共享 `@v1.13` 环境中的 CairoMakie 和 HDF5：

```sh
julia --project=@v1.13 random_tfim/plot_results.jl random_tfim/results/full.h5 random_tfim/results/figures
```

不传参数时读取 `random_tfim/results/full.h5`；默认输出到输入文件旁的 `figures/`。
[plot_results.jl](plot_results.jl) 从同一输入文件绘制四张图，不重新采样：

- `gap_distribution.png`：2×2 子图，选取 h0≥1 中最接近 1、2、5、10 的四点，各曲线对应不同尺寸。
- `average_correlation.png`：2×3 子图，选取最接近 0.1、0.5、1、2、5、10 的六点，绘制 C 对 r 的双对数图。
- `log_correlation_sqrt_r.png`：同样六点与尺寸，绘制无序平均 ln C 对 √r 的线性图。
- `imaginary_time_autocorrelation.png`：2×2 子图分别对应四个尺寸，各含上述六个 h0 的虚时间自关联双对数曲线。

输入须含四个尺寸及足够的不同 h0 点；图中标注实际取值，完整取值和数据审计写入 `plot_audit.md`。
阴影为独立无序样本间的 ±1 SEM。能隙使用宽度为 1 的自然对数分箱，
密度按全部样本数归一化；未分辨的概率质量不重新分配。
非有限均值处断开曲线，无效 SEM 处不画阴影；对数轴另行省略非正值及跨零误差带。

### Slurm 扫描

```sh
sbatch random_tfim/submit.sh
sbatch random_tfim/submit.sh demo 4 random_tfim/results/demo_slurm.h5 open
```

[submit.sh](submit.sh) 默认运行 `full`，参数与本地脚本一致。
[run_slurm.jl](run_slurm.jl) 激活 `julia-env/server`，该环境需已安装 HDF5 和 SlurmClusterManager。
每个 worker 处理一个 `(L,h0)`，固定启动 8 个 Julia 线程；主进程独占 HDF5 写入。
提交前按集群修改账户、分区、工作目录和资源数，使分配的 CPU 与 worker 线程数匹配。

## 函数接口

### 单个无序样本

```julia
include("random_tfim/RandomTFIM.jl")
using .RandomTFIM, Random, LinearAlgebra
BLAS.set_num_threads(1)

L, h0 = 32, 1.0
boundary = :periodic                 # 改为 :open 可切换整套计算
J, h = sample_disorder(Xoshiro(1996), L, h0; boundary)
state = ground_state(J, h; boundary) # gap、resolved 和 G
pair = correlations(state.G; boundary, rmax=L÷2)

times = collect(0.0:0.2:20.0)
imaginary = autocorrelation(J, h, times; boundary, time_domain=:imaginary)
realtime = autocorrelation(J, h, times; boundary, time_domain=:real)
```

要求偶数 `L≥2`、有限的 `J≥0` 和 `h>0`。`h` 长度为 L；开边界的 `J` 长度为 L−1，周期边界为 L。
**`autocorrelation` 默认开边界，其余公开接口默认周期边界**，建议像示例一样显式传递 `boundary`。
相同 seed 在两种边界下生成相同的横场和内部键，开链仅去掉接缝键。

| 函数 | 返回值 |
|---|---|
| `sample_disorder(rng, L, h0; boundary)` | 命名元组 `(J, h)` |
| `energy_gap(J, h; boundary)` | 命名元组 `(gap, resolved)` |
| `ground_state(J, h; boundary)` | 命名元组 `(gap, resolved, G)` |
| `correlations(G; boundary, rmax)` | 直接返回 `Matrix{Float64}`，形状 `L×(rmax+1)` |
| `autocorrelation(J, h, times; boundary, j, time_domain)` | 直接返回 `Vector{Float64}`，长度 `length(times)` |

空间关联中 `C[i,r+1]=⟨σz(i)σz(i+r)⟩`，`r=0` 列为 1。
默认 `rmax=L÷2`；周期链最多取 L÷2，开链可取 L−1。
开链仅 `i+r≤L` 的位置有效，越界位置为 `NaN`。
`resolved` 仅保留在能隙接口及无序样本能隙中。

### 实时间与虚时间

两种模式都计算零温纵向自关联，取 ħ=1。内部演化使用：

$$
C_j(z)=\langle0|e^{zH}\sigma_j^z e^{-zH}\sigma_j^z|0\rangle
=\sum_n|\langle n|\sigma_j^z|0\rangle|^2e^{-(E_n-E_0)z}.
$$

| `time_domain` | z | 合法时间 | `C` 的元素类型 |
|---|---|---|---|
| `:imaginary`（默认） | τ | 有限、非负 | `Float64` |
| `:real` | it | 有限，可为负 | `Float64`，仅返回实部 |

`times` 可为空、乱序或含重复点；`j` 默认 L÷2，开链取左中点。
零时间也执行完整计算，结果在舍入误差内满足 `C(0)≈1`。
实时间返回 `Re C_j(it)=⟨{σz(j,t),σz(j,0)}⟩/2`，即对称化关联，内部仍保留复数演化。
精确算术下，虚时间关联在 [0,1] 内且单调不增；返回的实时间关联满足 `C(-t)=C(t)`，可为负。
有限链的确定宇称基态有 `⟨σz⟩=0`，因此结果同时等于 connected correlation。
接口适用于零温平稳基态，不包含热态、quench 初态或破缺对称态。

### 无序平均

沿用上例的 `L`、`h0`、`boundary` 和 `times`：

```julia
ensemble = disorder_ensemble(L, h0, times; nsamples=100, seed=1996, boundary, keep_samples=true)

average = ensemble.C_mean
mean_log = ensemble.logC_mean
typical = exp.(mean_log)
time_average = ensemble.Ct_mean
real_time_average = ensemble.real_Ct_mean

# 只保留能隙结果：显式传入空 times，再设置 rmax=0
only_gaps = disorder_ensemble(L, h0, Float64[]; nsamples=100, boundary, rmax=0)
```

| 返回字段 | 内容 / 形状 |
|---|---|
| `gap_mean, gap_sem` | 能隙均值和标准误，标量 |
| `log_gap_mean, log_gap_sem` | 对数能隙均值和标准误，标量；未分辨样本产生 NaN |
| `gaps, resolved` | 仅 `keep_samples=true` 时返回，逐样本能隙及其标记，长度 nsamples |
| `C_mean, C_sem` | 空间关联均值和标准误，长度 rmax+1 |
| `logC_mean, logC_sem` | 空间关联对数的均值和标准误，长度 rmax+1 |
| `Ct_mean, Ct_sem` | 虚时间关联均值和标准误，长度 length(times) |
| `logCt_mean, logCt_sem` | 虚时间关联对数的均值和标准误，长度 length(times) |
| `real_Ct_mean, real_Ct_sem` | 实时间自关联实部的均值和标准误，长度 length(times) |
| `real_logCt_mean, real_logCt_sem` | 实时间自关联实部对数的均值和标准误 |
| `sample_real_Ct` | 仅 `keep_samples=true` 时返回，形状 `length(times)×nsamples` |
| `sample_C` | 仅 `keep_samples=true` 时返回，形状 `(rmax+1)×nsamples` |
| `sample_Ct` | 仅 `keep_samples=true` 时返回，形状 `length(times)×nsamples` |

不返回对数样本数组；需要时由后续计算重新取对数。空间对数统计仍按逐对取 log 后平均，不能用 `log.(sample_C)` 重建。

默认 `nsamples=100`、`seed=1996`、`rmax=L÷2`、`keep_samples=false`。
`times` 是第三个必需位置参数，必须有限且非负；两种时间域共用该网格并同时计算。
显式传入空 `times` 时返回空时间统计；`rmax=0` 跳过非平凡空间关联。
先逐样本计算，再平均：空间关联先平均同一样本内的有效起点（周期 L 个、开链 L−r 个），
随后在无序样本间求均值和 SEM。同一样本的能隙与各关联数组使用相同构型。
随机数在并行前顺序生成，构型顺序与 Julia 线程数无关。

所有 SEM 使用样本标准差除以 `sqrt(nsamples)`，单样本 SEM 为 `NaN`。
虚时间使用 `sample_Ct/Ct_mean/Ct_sem`，实时间使用 `sample_real_Ct/real_Ct_mean/real_Ct_sem`，均为实数；实时间 SEM 仅统计实部的样本波动。
对数统一使用返回值的 `log(Ct)`，实时间对应原始复关联的 `log(real(Ct))`。
两类关联及其对数的均值和 SEM 均在 `disorder_ensemble` 内计算。
空间关联逐对取 log 后先平均有效起点，自关联逐时间点取 log；再在样本间求均值和 SEM。
`logC_mean/logCt_mean` 是先取对数再平均，不是均值的对数；非正或非有限实数（实时间取实部）产生的 `NaN` 会传播到对数统计，
不会通过删除格点或样本重新平均。`exp.(logC_mean)` 是典型关联；将对数均值 ±1 SEM 指数映射得到的范围不是指定置信水平的置信区间。

## HDF5 输出

本地和 Slurm 脚本均同时保存虚时间和实时间结果。根目录中的 `parameters/sizes`、`parameters/fields`
记录扫描参数，各参数组命名为 `L16/h1.0` 等。组属性仅保留观测格点 `j` 和随机种子 `seed`。
根属性仅保留 `boundary`（`open` 或 `periodic`）、`distribution`（`box`）、`nsamples` 和 `complete`，只有整个扫描完成后才设 `complete=true`。
耗时仅打印到运行日志；环境信息和固定的物理、统计定义不再写入属性，相关约定见本文档。
两个脚本共用 `results_io.jl` 写入统计量和可选样本。
计算前从 `fields` 中选出最接近 `0.1、0.5、1、2、5、10` 的 6 个值，记录在 `parameters/sample_fields`。
这些值对应索引 `[1, 36, 51, 66, 86, 101]`，约为 `[0.1, 0.501187, 1, 1.995262, 5.011872, 10]`。
仅这些参数调用 `keep_samples=true` 并保存全部返回的样本；其余参数调用 `keep_samples=false`，只保存统计量和时间网格。
`imaginary_time` 与 `real_time` 保存相同的网格；两种时间域的自关联统计均保存为 Float64 数组，实时间仅保存实部。

| 数据集 | 内容 / 形状 |
|---|---|
| `gap_samples, gap_resolved` | 仅特殊参数组：逐样本能隙及分辨标记，长度 nsamples |
| `sample_C` | 仅特殊参数组：空间关联样本，形状 `(rmax+1)×nsamples` |
| `sample_Ct, sample_real_Ct` | 仅特殊参数组：虚时间、实时间关联样本，形状 `length(times)×nsamples` |
| `gap_mean, gap_sem` | 能隙均值和标准误，标量 |
| `log_gap_mean, log_gap_sem` | 对数能隙均值和标准误，标量 |
| `correlation_mean, correlation_sem` | 空间关联均值和标准误，长度 rmax+1 |
| `log_correlation_mean, log_correlation_sem` | 空间关联对数的均值和标准误，长度 rmax+1 |
| `imaginary_time, real_time` | 共用的时间网格 |
| `autocorrelation_mean, autocorrelation_sem` | 虚时间自关联均值和标准误，与时间网格等长 |
| `log_autocorrelation_mean, log_autocorrelation_sem` | 虚时间自关联对数的均值和标准误，与时间网格等长 |
| `real_autocorrelation_mean, real_autocorrelation_sem` | 实时间自关联实部的均值和标准误 |
| `real_log_autocorrelation_mean, real_log_autocorrelation_sem` | 实时间自关联实部对数的均值和标准误 |

不保存 `log_gap_samples` 或关联的对数样本数组；对数能隙由后续分析根据 `gap_samples` 和 `gap_resolved` 重算，未分辨样本对应 `NaN`。
空间关联样本已在有效起点间平均，不能重建逐格点对的分布。
距离由数组索引恢复：空间统计量第 r+1 个元素对应 r；样本数由根属性 `nsamples` 读取。
绘图读取器兼容旧字段 `loggaps/resolved/average/sem/mean_log/log_sem` 及没有 `parameters` 的旧文件。
已有文件不会自动改写，旧实时间数据不能当作虚时间数据使用。

## 模型与算法

采用 Pauli 矩阵约定，周期自旋链的哈密顿量为

$$
H=-\sum_{i=1}^{L}J_i\sigma_i^z\sigma_{i+1}^z-\sum_{i=1}^{L}h_i\sigma_i^x.
$$

开链的键求和到 L−1；周期链的 `J[L]` 是接缝键，L=2 时仍保留两条键。
采样采用箱形分布 `J∼U(0,1)`、`h∼U(0,h0)`，实现排除零值。
该随机模型的临界点为 h0=1，控制参数 δ=½ ln h0。

### 能隙与空间关联

费米子矩阵 K 的对角元为 $h_i$、次对角元为 $-J_i$；周期费米子接缝为 $-J_L$，反周期为 $+J_L$。
对 `K=U diag(ε) Vᵀ` 做 SVD，单准粒子能量为 2ε，`G=−VUᵀ`。
开链使用双对角结构，能隙为 `2minimum(ε)`；周期自旋链基态取反周期费米子扇区：

$$
E_0=-\sum_\mu\epsilon_\mu^{ap},\qquad
E_1=-\sum_\mu\epsilon_\mu^p+
\begin{cases}2\epsilon_{\min}^p,&\prod_i h_i\ge\prod_i J_i,\\0,&\prod_i h_i<\prod_i J_i.\end{cases}
$$

代码比较对数乘积判定每个样本的真空宇称，不能用总体 h0 代替；相等时零模使占据修正为零。
对 i<j，空间关联为

$$C_{ij}=\det G_{i:j-1,\ i+1:j}.$$

周期链跨接缝时按反周期符号延拓 G。每个起点仅提取一次最大子矩阵，
用 Givens 正交旋转逐步更新各距离的 QR 因子，从对角元累加行列式对数并保留符号。
算法不除以小主元，允许奇异前缀后再次出现非奇异前缀。

### 自关联的两个计算路径

初态 SVD 记为 `U_g, ε_g, V_g`；演化 SVD 记为 `U_e, ε_e, V_e`。
开链二者相同；周期链插入 σz 后切换宇称，初态用反周期扇区，演化用周期扇区。
将 `Γ_oe=U_g V_gᵀ` 的前 j 行和前 j−1 列分别取负，得到插入 σz 后的协方差块 Γᵠ。
令 `B=U_eᵀ Γᵠ V_e`，在上文统一时间变量 z 下：

$$
R_{ab}(z)=\frac{\delta_{ab}+B_{ab}}2+
 e^{-2\epsilon_a^e z}\frac{\delta_{ab}-B_{ab}}2,\qquad
C_j(z)=e^{z\sum_a(\epsilon_a^e-\epsilon_a^g)}\det R(z).
$$

中点及一般位置使用这一 L×L 缩放行列式；开链的外部指数系数直接设为零。
虚时间矩阵为实数，实时间为复数。通过 `logabsdet` 与外部指数合并恢复结果，
不构造增长的双曲函数，也不使用 `sqrt(det)`。
非对角元共享行因子；对角上的常数项和衰减项分开计算，以保留解耦极限的微小尾部。

靠近开链端点时，令 `d=min(j,L+1-j)`；若 `4d−2≤L÷4`，改用较短的 Jordan–Wigner 字符串 Pfaffian。
右端通过反射并交换 U、V 处理。取 Majorana 约定
$\gamma_{2k-1}=(\prod_{\ell<k}\sigma_\ell^x)\sigma_k^z$、
$\gamma_{2k}=(\prod_{\ell<k}\sigma_\ell^x)\sigma_k^y$，则
$\sigma_j^z=i^{j-1}\gamma_1\cdots\gamma_{2j-1}$。在左端字符串 S=1:2j−1 上，
$M_{2a-1,\mu}=U_{a\mu}$、$M_{2a,\mu}=iV_{a\mu}$，并计算

$$
Q(z)=M_S\operatorname{diag}(e^{-2\epsilon z})M_S^\dagger,\qquad
C_j(z)=(-1)^{j-1}\operatorname{Pf}
\begin{pmatrix}-i\Gamma_{SS}&Q(z)\\-Q(z)^T&-i\Gamma_{SS}\end{pmatrix},
$$

其中 Γ 的奇偶块为 UVᵀ，偶奇块为其负转置。模式求和包含全链的 L 个 SVD 模式，
仅端点索引截取 S；每个时间点更新指数并复用缓冲区，不重复做 SVD。

## 精度与性能

### 未分辨结果

- **能隙：** 周期链的阈值为 `64eps(Float64)*max(abs(E0),abs(E1),1)`，
  开链为 `64eps(Float64)*max(sum(ε),1)`；不超过阈值时 `resolved=false`，原始 gap 不裁剪。
- **空间关联：** `correlations` 仅返回原始有符号 `C`，不再返回 `logC/resolved`，也不再接受 `logtol`。
  QR 内部仍用对数行列式重建 `C`，但最终转换为 Float64 时可能下溢为零。
  `disorder_ensemble` 从返回的 `C` 取 log，不恢复下溢前的对数，也不做条件数筛选。
- **时间关联：** 不自动取绝对值或截断负值。缩放避免指数溢出，但长虚时间的极小尾部仍可能受舍入误差、噪声底和下溢影响。

能隙标记是启发式诊断，不是严格误差界。空间关联与自关联的非正实值在取 log 时记为 NaN，
不取绝对值、不裁剪、不删除样本。对数行列式不能恢复矩阵构造时丢失的信息；可信的极小尾部需要额外精度验证。

### 计算成本与并行

空间关联的总成本约为 `O(L*rmax^3)`。
一般位置的时间关联约为 `O(L^3+Nt*L^3)`，工作矩阵占 `O(L^2)`；短字符串端点更便宜。
联合计算复用 SVD，核心不修改输入，也不使用全局可变状态。

运行脚本固定单 BLAS 线程，用 Julia 线程并行不同样本；直接调用时也建议 `BLAS.set_num_threads(1)`。
即使 `keep_samples=false`，内部仍分配关联样本数组来计算统计量，内存随样本数增长。
预先生成的构型数组约占 `16*L*nsamples` 字节，另有对象及线程工作区开销。
默认 HDF5 的关联统计量大小不随样本数增长，逐样本能隙数据除外。

性能以 [benchmark.jl](test/benchmark.jl) 在当前机器上的结果为准：固定种子、单 BLAS 线程、
预热后七次中位数，覆盖两种边界、两种时间模式、端点和强场空间关联。

## 测试与基准

```sh
# 核心物理、边界、数值诊断和无序统计：仅标准库
julia --startup-file=no --threads=2 random_tfim/test/runtests.jl

# HDF5 写入/回读，以及本地和 Slurm 输出接口；自动激活 local 环境
julia --startup-file=no --threads=2 random_tfim/test/smoke_io.jl
julia --startup-file=no random_tfim/test/summary_io.jl

# 绘图读取器：当前和旧版文件布局
julia --startup-file=no --project=@v1.13 random_tfim/test/plot_io.jl

# 性能基准；可选参数为另存的旧版源码路径
julia --startup-file=no random_tfim/test/benchmark.jl
julia --startup-file=no random_tfim/test/benchmark.jl /path/to/old/RandomTFIM.jl
```

核心测试独立构造完整自旋哈密顿量，核对能隙、空间关联和两种时间关联；
另检查宇称、断开周期接缝、解耦极限、微小尾部、奇异前缀、负时间及可重复采样。
`smoke_io.jl` 实际运行 demo/full 的少量样本扫描；`summary_io.jl` 验证输出接口，不提交 Slurm 作业。
基准对照源码须支持 `time_domain=:imaginary/:real`；空间关联核对原始 `C`。
这些小规模验证不代表已完成 full 默认 10000 样本的论文统计精度复现。

## 论文图与数据需求

以下对照采用周期边界。所有对数均为自然对数。

| 论文图 | 所需数据 / 处理 |
|---|---|
| 1–4 | ln ΔE 分布；临界时按 √L 缩放；非临界的 z 由分析阶段指定 |
| 8–9 | `correlation_mean` 对 r 的双对数图；`log_correlation_mean` 对 √r |
| 10–12 | 需要逐格点对关联分布，默认 HDF5 不包含 |
| 13–16 | 空间关联除以临界值、对数关联减去临界值；横轴分别为 rδ²、2rδ |
| 17–19 | 对数均值可由现有统计量得到；逐格点对分布与方差需要额外数据 |

提取逐格点对关联分布时需逐样本调用 `ground_state` 和 `correlations`，再对关联取对数；跨样本误差应按无序样本分块，
不要把同一样本内的起点视为独立样本。能隙直方图应报告未分辨比例，并按实际箱宽归一化。
