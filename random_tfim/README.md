# 随机横场 Ising 链

计算有限链零温能隙、纵向空间关联，以及实时间/虚时间自关联。
核心采用自由费米子 SVD，支持开边界和周期边界；无序样本可按 Julia 线程并行。
模型与论文对照见 [Young & Rieger (1996)](../papers/Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md)
和[模型卡](../.knowledge/models/transverse-field-ising/MODEL.md)。

## 快速运行

以下命令从仓库根目录执行。核心模块仅依赖 Julia 标准库；
[run.jl](run.jl) 自动激活已有的 `julia-env/local` 环境，使用 HDF5 保存结果。

```sh
# 默认读取脚本旁的 random_tfim/scan.toml
julia --threads=4 random_tfim/run.jl
# 也可以指定自己的配置文件
julia --threads=4 random_tfim/run.jl random_tfim/scan.toml
```

本地与 Slurm 入口共用 [scan.toml](scan.toml)，命令格式为 `run.jl [config.toml]`，
不再接受原来的 `demo|full` 等位置参数。相对输出路径以 TOML 所在目录为基准，
输出文件名自动追加时间戳，已有同名文件时拒绝覆盖。

配置中的 `sizes`、`nsamples`、`seed`、`boundary`、`field_distribution`、`fields`、
`times`、`sample_fields`、`worker_threads` 和 `output` 分别控制链长、样本数、主种子、
边界、横场分布、场强网格、时间网格、保存逐样本数据的目标场强、Slurm worker 线程数和输出路径。
`fields` 和 `times` 可以直接写数组，或写 `{ start = ..., stop = ..., length = ... }` 的线性网格；
增加 `scale = "log10"` 时，start/stop 表示以 10 为底的指数。
`sample_fields` 直接写目标数组，目标会匹配到最近的实际扫描点，空数组表示只保存统计量。
切换 `field_distribution` 时按需手动调整该数组；配置注释提供 fixed 的推荐目标。

`uniform` 使用 `h∼U(0,h0)`；`fixed` 使用每个格点相同的 `h=h0/e`，其中 e 为自然常数。
两种模式的临界点均为 `h0=1`，耦合均为 `J∼U(0,1)`；平均横场分别为 `h0/2` 和 `h0/e`。

默认配置保持原本地 demo 规模：链长 `[16, 32]`、每点 20 个样本。
完整扫描可将 `sizes` 改为 `[16, 32, 64, 128]`、`nsamples` 改为 `10000`，
并将 `output` 改为 `results/full_uniform.h5`（固定横场则使用 `results/full_fixed.h5`）。
Slurm 提交现在也默认使用这份配置，不再隐含 full/50000/fixed 参数。

默认扫描 `h0 = 10 .^ range(-1, 1, 101)`，计算能隙、`r=0:L÷2` 的空间关联，
以及中点 `j=L÷2`、`times=10 .^ range(-1,3,101)`（0.1 到 1000）的虚时间和实时间自关联。
每个 `(L,h0)` 使用 `seed + 1000k + L`（k 为场强索引），本地和 Slurm 使用相同规则。
在 Julia 中调用时，先用 `cfg = read_config("path/to/scan.toml")` 读取配置，
再调用 `main(cfg)`；Slurm 对应 `RandomTFIMSlurm.read_config(path)` 和 `RandomTFIMSlurm.run_scan(cfg, pids)`。

### 绘制已有数据

绘图使用共享 `@v1.13` 环境中的 CairoMakie 和 HDF5：

```sh
julia --project=@v1.13 random_tfim/plot_results.jl random_tfim/results/full.h5 random_tfim/results/figures
```

不传参数时读取 `random_tfim/results/` 下最新的 `full_uniform_<时间戳>.h5` 或 `full_fixed_<时间戳>.h5`；自定义文件名需显式传入路径。默认输出到输入文件旁的 `figures/`。
比较两种模式时请显式指定输入文件和不同输出目录，避免图像互相覆盖。各图和审计报告标注横场模式；只支持当前版本生成的 HDF5 格式。
[plot_results.jl](plot_results.jl) 从同一输入文件绘制九张图（实时间数据缺失时为八张），不重新采样：

- `gap_distribution.png`：2×2 子图，选取 h0≥1 中最接近 1、2、5、10 的四点，各曲线对应不同尺寸。
- `scaled_gap_distribution.png`：同一布局，先将样本变换为 ln ΔE/√L，再以公共箱宽 0.1 统计密度。
- `average_correlation.png`：2×3 子图，选取最接近 0.1、0.5、1、2、5、10 的六点，绘制 C 对 r 的双对数图。
- `log_correlation_sqrt_r.png`：同样六点与尺寸，绘制无序平均 ln C 对 √r 的线性图。
- `imaginary_time_autocorrelation.png`：2×2 子图分别对应四个尺寸，各含上述六个 h0 的 〈C(τ)〉 双对数曲线；直接使用已保存的均值和 SEM。
- `imaginary_time_autocorrelation_slopes.png`：h0≥该模式的临界值，横轴对数、纵轴线性；各尺寸显示 ln〈C(τ)〉 对 ln τ 的回归斜率绝对值，回归使用全部有效时间点。它是有效斜率，只有渐近幂律窗口才可解释为 1/z，固定横场的有隙区不能如此解释。
- `real_time_autocorrelation.png`：存在实时间数据时输出；同虚时图布局，纵轴线性。
- `imaginary_time_log_distribution.png`：最大尺寸（当前 L=128），四个子图沿用能隙图的 h0；选取最接近 τ=1、3、10、30、100、300 的六个网格点，用 `scatterlines!` 绘制逐样本 −ln C(τ) 的密度，图例标注实际时间。
- `imaginary_time_rescaled_distribution.png`：同一批样本和布局，绘制 x=−ln C(τ)/τ^μ 的密度；每个 h0 使用实际网格时间独立拟合 μ，并标在子图标题。

两张虚时分布图要求对应参数组保存 `sample_Ct`，每条曲线使用 60 个等宽箱。
μ 最小化六个时刻的对应分位数在对数空间中的差异：Σ[ln Q_p(−ln C)−μ ln τ−a_p]²，
p=0.05、0.10、…、0.95，每个分位数有独立截距 a_p；拟合与分箱无关。
直接对每个正样本取负对数；非正值和非有限值不取对数，其数量标在图中，密度仍按全部样本数归一化。
拟合只使用有效样本的正分位数，μ、拟合前后残差和缺失比例写入 `plot_audit.md`；
若某个时刻所有样本都因下溢等原因无效，该时刻不参与拟合且不画密度，但仍记录缺失数量；不足两个有效时刻时 μ 为 NaN，缩放图标注不可用。
该 μ 描述所选时间范围内的最佳重合，不能直接视作渐近物理指数。
虚时间均值与逐样本分布均直接使用保存的数据，不取绝对值。

输入须含四个尺寸及足够的不同 h0 点；图中标注实际取值，完整取值和数据审计写入 `plot_audit.md`。
阴影为独立无序样本间的 ±1 SEM。能隙使用宽度为 1 的自然对数分箱，
密度按全部样本数归一化；未分辨的概率质量不重新分配。
非有限均值处断开曲线，无效 SEM 处不画阴影；对数轴另行省略非正值及跨零误差带。

### 单独绘制虚时间自关联及拟合（full_fixed）

```sh
# 默认选择最新的 full_fixed_<时间戳>.h5
julia --project=@v1.13 random_tfim/plot_imaginary_time.jl
# 可指定输入、输出目录和拟合时间窗口（例如 1 <= tau <= 100）
julia --project=@v1.13 random_tfim/plot_imaginary_time.jl random_tfim/results/full_fixed_20261003_230245.h5 random_tfim/results/figures_imaginary_time_fixed 1 100
# 独立拟合检查，无额外包依赖
julia --startup-file=no random_tfim/test/imaginary_time_fit.jl
```

图 1 仅绘制 L=128，使用单个双对数子图，含 10 个 h0；`imaginary_time_fits.csv` 同样仅保存 L=128 的拟合。
在对应区域内选取最接近目标值的已存场强：铁磁区 0.2、0.6；Griffiths 区
1.1、1.4、1.7、2.1、2.5；有隙顺磁区 3、5、10。固定横场 h=h0/e，分界为 h0=1、e；
图例和结果表标出实际场强。实线和阴影沿用保存的均值及 ±1 SEM，虚线为选中的拟合。

[fit_autocorrelation.jl](fit_autocorrelation.jl) 比较 `C=A*tau^(-alpha)` 与
`C=A*exp(-lambda*tau)`（A>0，alpha/lambda≥0，不加常数项）。令 `x=ln(tau)`、`Y=ln(C)`，
幂律拟合 `Y=logA-alpha*x`，指数模型拟合 `Y=logA-lambda*exp(x)`；两者均对 logA 和衰减率做无权线性回归。
在相同有效数据点上最小化 `chi_square=sum((Y-Y_fit)^2)`，
选择 `reduced_chi_square=chi_square/(n-2)` 较小者。结果表沿用字段名，数值为对数残差平方和 SSE
及 SSE/(n−2)，不是按测量误差归一化的 χ²。
两模型使用相同数据点、各有两个参数，因此目前与比较 χ² 的选择结果相同。
图例标作 `1/z=alpha` 或 `1/ξτ=lambda`，A 保留在结果表中。
指数形式为 `A*exp(-tau/ξτ)`，ξτ 为衰减时间，图例显示其倒数。
默认使用全部有限正时间和有限正 C；1e-8 仅为图 1 的纵轴显示下限，不再截断拟合数据。
指定时间窗口仅影响拟合，原始曲线仍显示完整时间范围。SEM 仅用于误差阴影，不参与拟合或点筛选。
时间网格视为精确值；各时间点共享无序样本，未计时间协方差；结果为描述性拟合，
铁磁平台和交叉区域未必符合两种模型，选中幂律也不意味着已得到渐近指数。

默认输出到 `results/figures_imaginary_time_fixed/`：PNG、
`imaginary_time_fits.csv`（每条曲线两种模型的参数、χ²、约化 χ²、有效窗口及选择标志）
和 `imaginary_time_fit_audit.md`。不修改原始数据。

图 2 `figure2_imaginary_time_fit_parameters.png` 为 2×2 子图，各自图例位于画框内：幂律拟合的 `1/z`、
指数拟合的 `1/ξτ`、两模型中较小的 `reduced_chi_square`，以及 L=128 的两个模型各自的 `reduced_chi_square`，均随 h0 变化。
第四子图标出两条曲线的交点：在相邻场强的分数差变号区间内，按 log(h0) 线性插值；不外推。
精确落在网格点上的交点也保留。交点与左右网格场强保存至 `imaginary_time_model_crossings.csv`，仅表示模型分数相等，不代表相界。
所有横轴均为对数标度；前三子图纵轴线性，第四子图纵轴对数（非正分数不显示）。前三子图不同颜色对应 L=16、32、64、128。
与旧斜率图相同，扫描所有 h0≥1 的已存场强；前两子图始终显示各自模型的参数，不按胜出模型筛选。
拟合沿用图 1 的无权回归及命令行时间窗口。默认使用全部有效数据时，衰减曲线的子图 1 斜率与旧斜率图一致。
指定时间窗口后只拟合窗口内的数据，旧斜率图仍使用全部有效时间。
`imaginary_time_field_scan.csv` 保存全部尺寸/场强的两种拟合参数、约化 χ²、最小值、选择结果和有效时间窗口。

两个绘图脚本共享 `fit_autocorrelation.jl`：无权直线回归、幂律/指数拟合、
`autocorrelation_log_slope` 和分布缩放 `fit_time_collapse` 均集中在此文件；
后者的分位数匹配算法保持不变，`plot_results.jl` 通过 include 引用共享实现。

### Slurm 扫描

```sh
sbatch random_tfim/submit.sh
sbatch random_tfim/submit.sh
sbatch random_tfim/submit.sh random_tfim/scan.toml
```

[submit.sh](submit.sh) 默认读取 `scan.toml`，参数与本地脚本一致。
[run_slurm.jl](run_slurm.jl) 激活 `julia-env/server`，该环境需已安装 HDF5 和 SlurmClusterManager。
每个 worker 处理一个 `(L,h0)`，按配置中的 `worker_threads` 启动 Julia 线程（默认 8 个）；主进程独占 HDF5 写入。
提交前按集群修改账户、分区、工作目录和资源数，使分配的 CPU 与 worker 线程数匹配。

## 代码结构

[RandomTFIM.jl](RandomTFIM.jl) 保留模块定义、标准库依赖和导出接口，按顺序加载以下实现文件：

- [model.jl](model.jl)：链参数校验、无序采样、费米子矩阵、能隙与基态。
- [correlations.jl](correlations.jl)：空间关联及增量 QR 计算。
- [autocorrelation.jl](autocorrelation.jl)：实时间/虚时间自关联及 Pfaffian 后备算法。
- [ensemble.jl](ensemble.jl)：无序样本并行计算、均值与标准误。

这些文件共享 `RandomTFIM` 命名空间，由模块入口统一加载；使用时仍只需
`include("random_tfim/RandomTFIM.jl")` 和 `using .RandomTFIM`。
命令行运行入口仍为 `run.jl` 和 `run_slurm.jl`。
两者共用 `scan_common.jl` 的任务生成和单点计算，结果写入与完成标记统一由
`results_io.jl` 管理；本地入口顺序执行任务，Slurm 入口负责 worker 调度。

## 函数接口

### 单个无序样本

```julia
include("random_tfim/RandomTFIM.jl")
using .RandomTFIM, Random, LinearAlgebra
BLAS.set_num_threads(1)

L, h0 = 32, 1.0
boundary = :periodic                 # 改为 :open 可切换整套计算
field_distribution = :uniform        # 改为 :fixed 使用 h=h0/e
J, h = sample_disorder(Xoshiro(1996), L, h0; boundary, field_distribution)
state = ground_state(J, h; boundary) # gap、resolved 和 G
pair = correlations(state.G; boundary, rmax=L÷2)

times = collect(0.0:0.2:20.0)
imaginary = autocorrelation(J, h, times; boundary, time_domain=:imaginary)
realtime = autocorrelation(J, h, times; boundary, time_domain=:real)
```

要求偶数 `L≥2`、有限的 `J≥0` 和 `h>0`。`h` 长度为 L；开边界的 `J` 长度为 L−1，周期边界为 L。
**`autocorrelation` 默认开边界，其余公开接口默认周期边界**，建议像示例一样显式传递 `boundary`。
相同 seed 在两种边界下生成相同的横场和内部键，开链仅去掉接缝键。
两种横场模式消耗相同的随机数，因此同一 seed 的逐样本耦合也相同；默认均匀模式保持原有随机序列。

| 函数 | 返回值 |
|---|---|
| `sample_disorder(rng, L, h0; boundary, field_distribution=:uniform)` | 命名元组 `(J, h)` |
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
ensemble = disorder_ensemble(L, h0, times; nsamples=100, seed=1996, boundary, field_distribution, keep_samples=true)

average = ensemble.C_mean
mean_log = ensemble.logC_mean
typical = exp.(mean_log)
time_average = ensemble.Ct_mean
real_time_average = ensemble.real_Ct_mean

# 只保留能隙结果：显式传入空 times，再设置 rmax=0
only_gaps = disorder_ensemble(L, h0, Float64[]; nsamples=100, boundary, field_distribution, rmax=0)
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
根属性为 `boundary`（`open` 或 `periodic`）、`distribution`（`box`，描述耦合分布）、`field_distribution`（`uniform` 或 `fixed`）、`nsamples` 和 `complete`，只有整个扫描完成后才设 `complete=true`。固定模式另存 `fixed_field_divisor=e`，实际横场为 `h0/fixed_field_divisor`。扫描参数和组名中的 `h0` 始终是用户输入的尺度。
绘图要求固定模式的 `fixed_field_divisor=e`，不再兼容旧横场定义。
耗时仅打印到运行日志；环境信息和固定的物理、统计定义不再写入属性，相关约定见本文档。
两个脚本共用 `results_io.jl` 解析运行参数，并写入元数据、统计量和可选样本。
计算前将配置中的 `sample_fields` 目标匹配到扫描网格的最近点，记录在 `parameters/sample_fields`。默认目标为 `0.1、0.5、1、2、5、10`；fixed 可手动设为 `0.5、1、1.5、2、e、3`。扫描网格不变。现有六面板绘图需要保存六个不同扫描点；四点分布图使用其中最接近临界点的一点和最大的三点。
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
绘图直接读取当前字段和 `parameters` 元数据，不推断缺失字段或兼容旧字段名。

## 模型与算法

采用 Pauli 矩阵约定，周期自旋链的哈密顿量为

$$
H=-\sum_{i=1}^{L}J_i\sigma_i^z\sigma_{i+1}^z-\sum_{i=1}^{L}h_i\sigma_i^x.
$$

开链的键求和到 L−1；周期链的 `J[L]` 是接缝键，L=2 时仍保留两条键。
耦合采用箱形分布 `J∼U(0,1)`，实现排除零值。横场由 `field_distribution` 选择：

| 模式 | 横场 | 临界 h0 | 顺磁侧 |
|---|---|---|---|
| `uniform`（默认） | `h∼U(0,h0)`，排除零值 | 1 | 所有有限 h0>1 均保留 Griffiths 稀有区 |
| `fixed` | `h=h0/e` | 1 | `1<h0<e` 为 Griffiths 区；`h0>e` 为有隙顺磁区 |

临界条件为 `mean(log h)=mean(log J)`，其中 `mean(log J)=-1`。
若采用 δ=(mean(log h)−mean(log J))/(var(log h)+var(log J))，均匀模式 δ=½ ln h0，固定模式 δ=ln h0。
`h0=e≈2.718282` 是固定模式 Griffiths 区的边界，不作为严格有隙区处理。上述相区指热力学极限；有限链、有限样本和时间窗口会影响拟合。
随机横场的能隙分布选取 h0≈1、2、5、10，固定横场的新计算选取 h0≈1、2、e、3；两种模式均以 h0=1 为临界点。

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

两种边界、所有观测位置均先构造这一 L×L 缩放矩阵；开链的外部指数系数直接设为零。
虚时间矩阵为实数，实时间为复数。通过 `logabsdet` 与外部指数合并恢复结果，
不构造增长的双曲函数，也不使用 `sqrt(det)`。
非对角元共享行因子；对角上的常数项和衰减项分开计算，以保留解耦极限的微小尾部。

每个时间点在 LU 分解前计算矩阵一范数，再复用 LU 因子调用 `LAPACK.gecon!`，
估计 `rcond₁(R)=1/(‖R‖₁‖R⁻¹‖₁)`。若估计值大于 `rcond_tol`，使用行列式；
否则 OBC 使用较短一侧的 JW 字符串 Pfaffian，PBC 使用保留宇称结构的高斯态重叠 Pfaffian。
实时间监测完整的复矩阵 R，
不根据关联实部是否为负切换。时间点可乱序，每点独立判断。

`autocorrelation` 和 `disorder_ensemble` 均接受 `rcond_tol`，默认 `sqrt(eps(Float64))≈1.49e-8`。
有效范围为 [0,1]：0 强制行列式，1 强制 Pfaffian，便于交叉验证。
这是数值稳定性的启发式阈值，不是严格的相对误差保证；更大的阈值会更早切换。

PBC 后备算法在演化模式基底中，逐模式选择条件概率不小于 1/2 的参考占据态 r，
令 `S=diag(1−2r)`、`Z=(I−SB)/(I+SB)`，并显式保持 Z 的反对称性。
记 `dₐ=exp(−2εₐᵉz)`，`kₐ=rₐ ? dₐ : 1`，`qₐ=rₐ ? 1 : dₐ`，则

$$
W(z)=\begin{pmatrix}Z&-\operatorname{diag}(k)\\
\operatorname{diag}(k)&-\operatorname{diag}(q)Z\operatorname{diag}(q)\end{pmatrix},\qquad
C_j(z)=\frac{(-1)^{L(L+1)/2}\operatorname{Pf}W(z)}{\det(I+Z)}
e^{z\sum_a(\epsilon_a^e-\epsilon_a^g)}.
$$

这是 2L×2L 的高斯重叠公式，不是把病态 R 嵌入反对称矩阵。
矩阵仅使用衰减指数或单位模相位；Pfaffian 的对数幅值与归一化、扇区能量补偿合并后恢复结果。
参考占据态及工作空间只在首次切换时建立，后续复用。
PBC 虚时间 Pfaffian 使用 Float64 工作区，实时间使用 ComplexF64；每点的 q 因子预计算到复用缓冲区。
强制 Pfaffian 时跳过 R 的构造、LU 和倒条件数估计；强制行列式时跳过倒条件数估计。

OBC 后备算法使用 Jordan–Wigner 字符串 Pfaffian。令 `d=min(j,L+1-j)`，
矩阵维数为 `4d−2`；右半链通过反射并交换 U、V 处理。
仅收集倒条件数不达标的时间点，批量计算并写回原位置，工作区只建立一次；
两种时间模式均使用 ComplexF64 收缩矩阵。取 Majorana 约定
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
  `energy_gap`、`ground_state` 和 `disorder_ensemble` 统一使用完整 SVD 的奇异值，
  避免不同 SVD 路径的舍入差异使阈值附近的标记不一致；单独计算能隙也会计算奇异向量。
- **空间关联：** `correlations` 仅返回原始有符号 `C`，不再返回 `logC/resolved`，也不再接受 `logtol`。
  QR 内部仍用对数行列式重建 `C`，但最终转换为 Float64 时可能下溢为零。
  `disorder_ensemble` 从返回的 `C` 取 log，不恢复下溢前的对数，也不做条件数筛选。
- **时间关联：** 按倒条件数自动切换行列式/Pfaffian，不取绝对值或截断负值。
  切换改善行列式近奇异导致的伪尾部，但不能保证任意长时间的相对精度，仍可能受输入分解精度和下溢影响。

能隙标记是启发式诊断，不是严格误差界。空间关联与自关联的非正实值在取 log 时记为 NaN，
不取绝对值、不裁剪、不删除样本。对数行列式不能恢复矩阵构造时丢失的信息；可信的极小尾部需要额外精度验证。

### 计算成本与并行

空间关联的总成本约为 `O(L*rmax^3)`。
时间关联约为 `O(L^3+Nt*L^3)`，工作矩阵占 `O(L^2)`。
每点复用 LU 估计倒条件数；切换后 OBC 使用 `4min(j,L+1-j)−2` 维 JW-Pfaffian，
PBC 使用 2L 维高斯重叠 Pfaffian，端点也遵循同一倒条件数判据。
联合计算复用 SVD，核心不修改输入，也不使用全局可变状态。

运行脚本固定单 BLAS 线程，用 Julia 线程并行不同样本；直接调用时也建议 `BLAS.set_num_threads(1)`。
即使 `keep_samples=false`，内部仍分配关联样本数组来计算统计量，内存随样本数增长。
预先生成的构型数组约占 `16*L*nsamples` 字节，另有对象及线程工作区开销。
默认 HDF5 的关联统计量大小不随样本数增长，逐样本能隙数据除外。

性能以 [benchmark.jl](test/benchmark.jl) 在当前机器上的结果为准：固定种子、单 BLAS 线程、
预热后七次中位数，覆盖两种边界、两种时间模式、端点和强场空间关联。

## 测试与基准

```sh
# 一条命令运行核心物理、HDF5/运行脚本、绘图测试
julia --startup-file=no --threads=2 random_tfim/test/runtests.jl

# 只检查核心物理：仅标准库
julia --startup-file=no --threads=2 random_tfim/test/runtests.jl core

# 可选：当前实现的性能基准，不计入正确性测试
julia --startup-file=no random_tfim/test/benchmark.jl
```

入口依次启动四个独立 Julia 进程；某组失败仍继续运行其余组，最终返回失败状态。
`physics.jl` 保留自旋哈密顿量对照、宇称、两种边界、解耦极限、微小尾部、奇异前缀和无序统计测试。
`io.jl` 自动激活 local 环境，以小链的两个代表场强检查两种横场、两种边界、样本/摘要输出，
通过临时 TOML 运行小链扫描，并启动一个本地 worker 检查分布式输出与本地结果逐项一致；
同时检查防覆盖和失败时的未完成标记，不提交 Slurm 作业。
`plot_io.jl` 使用已有 `@v1.13` 环境，覆盖当前格式的两种横场模式、样本与摘要读取、掩码、斜率和分布缩放测试。
`imaginary_time_fit.jl` 检查无权幂律/指数拟合的参数恢复、模型选择、平台、无效点筛选、微小正尾部保留，以及与旧斜率函数和独立线性回归的一致性。
删除独立尾部对比诊断脚本（核心测试已覆盖相应数值检查）；性能基准仅测当前实现，不再加载旧源码。
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
