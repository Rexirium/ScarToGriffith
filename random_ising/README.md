# 二维随机键 Ising 的响应函数

Slurm 扫描使用 [run_slurm.jl](run_slurm.jl)、[scan.toml](scan.toml) 和 [submit.sbatch](submit.sbatch)，用法见文末。

在项目根目录启动 `julia --project=julia-env/local --threads=auto`，然后运行：

```julia
include("random_ising/observables.jl")

L, T, p, r = 16, 2.0, 0.5, 0.3
rng = MersenneTwister(10)
# 已有构型时，直接用自己的 Js 替换下面这一行。
Js = [ifelse.(rand(rng, 2L^2) .< p, 1.0, r) for _ in 1:4]
C, chi, U4, correlation = random_bond_observables(L, T, Js;
    mcs=8192, thermalization=2048, binsize=64, seed=1234, max_corr_time=100)

C[1]              # 第一个构型的总热容
chi[1]            # 第一个构型的总磁化率
U4[1]             # 第一个构型的 Binder 累积量
correlation       # 所有无序样本的平均交叠，t = 0:max_corr_time
correlation[t+1]  # 时间间隔 t 的平均关联函数
```

采用周期边界、`sᵢ=±1`、`kB=1`，哈密顿量为 `H=-Σ⟨ij⟩Jᵢⱼsᵢsⱼ`。
每个输入向量含 `2L²` 个有限非负的实际耦合值，0/1 不会被转换为键类别。
函数直接使用给定构型，不再抽样，因此无需传入分布参数 `p`、`r`。
上例仅在生成二值随机键时使用 `p`、`r`，其约定与 `critical_temp.jl` 一致：
`J=1` 的概率是 `p`，另一种键是 `J=r≥0`。

键顺序沿用 SpinMonteCarlo 的方格晶格定义。令 `i=x+L*(y-1)`，
`Js[a][2i-1]` 是 `(x,y)` 到 `(x+1,y)` 的键，`Js[a][2i]` 是到 `(x,y+1)` 的键。
坐标按周期取模。支持 `L≥2`；`L=2` 保留周期晶格产生的平行键。

令 `M=Σᵢsᵢ`，返回值定义为：

| 返回值 | 定义 | 类型 |
| --- | --- | --- |
| `C` | `(〈H²〉-〈H〉²)/T²` | `Vector{Float64}` |
| `chi` | `〈M²〉/T` | `Vector{Float64}` |
| `U4` | `1-〈M⁴〉/(3〈M²〉²)` | `Vector{Float64}`，长度 `num_disorder` |
| `correlation` | 固定起点自旋交叠的跨样本均值 | `Vector{Float64}`，长度 `max_corr_time+1` |

关联函数始终计算。令 `Ns=L²`，参考时刻为 `t0=corr_start_time`，以 SW 热化结束后的热浴 sweep 数计时，则
每个样本的 `C_a(t) = Σᵢ sᵢ(t0+t)sᵢ(t0) / Ns`，`t=0:max_corr_time` 为相对时间差。
返回 `correlation[t+1] = mean(C_a(t))`；`errors.correlation[t+1] = std(C_a(t); corrected=true)/sqrt(num_disorder)`。
`corr_start_time` 默认为 `0`，`max_corr_time` 默认为 `100`；两者均为非负整数，且其和不得超过 `mcs`。
均值和误差均为长度 `max_corr_time+1` 的向量。非空输入的首项为 `C(0)=1`，至少两个样本时其标准误为零。
单样本的标准误为 `NaN`；空输入的均值和标准误均为 `NaN` 向量。
每个无序构型仅采样一条轨迹；不做时间起点平均、不减去自旋均值。
时间单位为热浴 sweep；在 `corr_start_time` 保存参考态，计算随后 `max_corr_time` 步的交叠，静态观测量仍测量全部 `mcs` 步。
采样时只保存一个参考态并即时计算交叠，自相关计算量为 `O(Ns*max_corr_time)`，
每个活动构型保存一个参考态，内部暂存 `(max_corr_time+1,num_disorder)` 交叠矩阵，完成后归约为均值与标准误，不返回或写出逐样本交叠。
`runMC` 仍保存静态观测量的原始测量，总内存仍随测量步数增长。
构型之间仍并行；非空输入且 `max_corr_time=0` 时返回 `[1.0]`，`metadata.corr_start_time` 记录参考时刻。

这里使用有限系统零场对称系综，`〈sᵢ〉=〈M〉=0`。
`C` 和 `chi` 都是整个系统的量；
除以 `L²` 得到每格点热容和磁化率。没有减去 `〈|M|〉²`。

`U4` 使用 `1-mean(result["Binder Ratio"])/3`，与 `critical_temp.jl` 一致，逐构型计算后返回，不先做无序平均。
设置 `details=true` 可取得具名结果及热容、总磁化率、U4 的分块 jackknife 误差；`errors.U4` 是长度 `num_disorder` 的向量，计算为 `stderror(result["Binder Ratio"])/3`。`errors.correlation` 则是跨样本标准误：

```julia
out = random_bond_observables(L, T, Js; details=true)
out.heat_capacity
out.correlation # 跨样本平均 C(t)
out.errors.correlation # 各时间点的样本标准误
out.errors.heat_capacity
out.U4[1]
out.errors.U4[1]
out.metadata  # 包版本、种子、采样长度、块大小等
```

热化阶段固定使用 Swendsen–Wang 更新，共 `thermalization` 步，不计入自相关时间。
测量阶段统一使用随机单点热浴：每个 sweep 有放回地随机选点 `L²` 次，
每次按局域场对应的条件玻尔兹曼分布重新抽取该点自旋。
自相关的一个时间单位对应一个 sweep；零局域场时以等概率取 ±1。
函数不接受 `update` 关键词；metadata 和 HDF5 属性不包含 `update` 或 `thermalization_update`。
`thermalization=0` 时直接进入测量。
每个构型使用独立初始化的随机数流；相同输入、顺序和种子可复现。
每个构型通过独立的 `Parameter` 调用 `runMC`，由框架负责热化、测量、分块和 jackknife。
不同构型通过 `Threads.@threads` 并行计算，输出顺序与输入一致，种子不依赖线程调度。
可用 `--threads=4` 指定线程数，或用 `--threads=1` 串行运行；函数调用方式不变。
自定义 `Estimator` 复用字典，仅测量能量和磁化强度的矩，与 `simple_estimator` 一致。
内置后处理负责热容、总磁化率和 Binder 比的统计，返回前将热容和总磁化率乘以 `L²`。
SpinMonteCarlo v1.2.2 会先保存逐步测量值再分块；移除局域测量后，单构型内存为 `O(L²+mcs)`。
多线程运行时，每个同时计算的构型都需要这部分内存。

静态量误差描述单个构型的热采样误差，不包含无序平均的误差。
自关联误差描述跨样本平均的标准误，包含无序差异及每个样本单条轨迹的采样涨落。
正式计算前需分别增加热化步数、测量步数和块大小，并比较不同种子，检查结果与误差是否稳定。
默认步数不保证所有温度和无序强度下都已收敛。

测试命令：

```powershell
julia --project=julia-env/local --threads=4 --check-bounds=yes random_ising/test_observables.jl
```

测试用独立的 `3×3` 精确枚举核对非均匀耦合与纯模型，
并检查 U4、零耦合极限、复现性、输入不被修改和非法输入。
多构型结果及误差还会逐项对照串行 `runMC`，覆盖两种更新算法。
实现针对本项目已安装的 SpinMonteCarlo v1.2.2 源码核对；
包的接口说明见 [官方文档](https://yomichi.github.io/SpinMonteCarlo.jl/latest/)。

## 单构型局域磁化率

只需局域响应时，可用独立函数，返回两个 `L×L` 矩阵：

```julia
L, T = 16, 2.0
disorder_seed, mc_seed = 10, 1234
rng = MersenneTwister(disorder_seed)
Js = ifelse.(rand(rng, 2L^2) .< 0.5, 1.0, 0.3)
chi_local, err_local = random_bond_local_susceptibility(L, T, Js;
    mcs=8192, thermalization=1024, binsize=64, seed=mc_seed)
```

`chi_local[x,y] = 〈s[x,y]M〉/T` 是零场下该格点对均匀外场的响应，
不除以 `L²`，不减去有限采样的自旋均值。输入实际耦合 `Js` 不会被修改。
热化仍用 SW，测量每个随机单点热浴 sweep 后的局域响应，不计算其他物理量。
此处 `seed` 直接控制单条 MC 轨迹，与生成无序的 `disorder_seed` 分开。

函数自行执行 MC 循环，只保留当前块累加值及 Welford 在线统计，内存为 `O(L²)`。
`err_local` 来自等长块均值的标准误，对此线性平均等价于删除一个块的 jackknife；
要求至少两个完整块，且 `mcs` 能被 `binsize` 整除。
应增加块大小检查误差收敛；误差只描述热采样，不包含无序平均。

## Slurm 扫描

修改 `scan.toml` 设置尺寸、温度、无序构型数和 MC 参数，脚本启动时自动读取。
`output_dir` 的相对路径以 TOML 文件所在目录为基准。
修改 `submit.sbatch` 设置节点、进程和每进程的 CPU 数。
个人电脑使用 `julia-env/local/Project.toml`；服务器使用 `julia-env/server/Project.toml`，只安装 MC 扫描所需依赖。
服务器的所有节点需能访问相同的仓库和服务器 Julia 环境，worker 自动沿用主进程的环境。
`run_slurm.jl` 在加载依赖前自动激活服务器环境；独立测试脚本 `test_run_slurm.jl` 激活本地环境，路径相对于脚本位置解析。
在服务器上，从仓库根目录安装依赖并提交（Julia 1.9 或更高版本）：

```bash
julia --project=julia-env/server -e 'using Pkg; Pkg.instantiate()'
sbatch random_ising/submit.sbatch
```

每个 worker 计算一个 `(L,T)`，内部使用 `--cpus-per-task` 个线程计算不同无序构型。
结果进入有界队列后，worker 即可领取下一项任务；主进程中单独的写入任务串行写 HDF5。
`scan.toml` 中的 `result_buffer` 设置队列容量，默认 2；队列满时等待，限制缓存增长。
除队列外，内存还包括正在写入的结果及各 worker 已返回、等待入队的结果。
主进程需 `--threads=2`，提交脚本已设置；worker 线程数仍由 `--cpus-per-task` 决定。
相同尺寸在不同温度下使用同一批无序构型。
默认输出为 `random_ising/results/run_001/L_8.h5` 等文件，温度分别存入 `T_1.0` 等 group。
每组保存 `heat_capacity`、`susceptibility`、`U4`、`correlation`，`errors` 保存热容、总磁化率和 U4 误差，以及自关联样本标准误 `errors/correlation`。
随后对第一个无序构型额外调用局域响应函数，同组保存 `chi_local` 和 `errors/err_local`，两者均为 `L×L` 矩阵。
局域计算使用相同的 `L`、`T`、热化步数、测量步数和块大小，输入为第一个无序构型，MC 种子为该温度组重构出的 `realization_seeds[1]`。
因此复现第一个构型的采样轨迹，`sum(chi_local) ≈ susceptibility[1]`；`elapsed_seconds` 包含两次计算的耗时。
Julia 中 `U4` 的维度为 `(ndisorder,)`，自关联均值及其标准误均为 `(max_corr_time+1,)`，
HDF5 格式版本为 `12`，温度组不再存储 `lags`；相对时间差可由根属性重构：`lags = 0:read(attributes(file)["max_corr_time"])`，与 `correlation` 及 `errors/correlation` 的元素一一对应。公共属性 `L`、`mcs`、`thermalization`、`binsize`、`max_corr_time`、`corr_start_time`、`boundary` 和 `normalization` 仅存于文件根，其中 `corr_start_time` 记录参考时刻；读取旧版温度组中这些属性的代码需改为读取根属性。温度组仅保留 `T`、`seed`、`elapsed_seconds` 和 `complete` 属性；不再保存 `worker_id`、`worker_threads`，文件根也不再保存 `slurm_job_id`。SpinMonteCarlo 包版本记录在文件根属性 `version` 中，不使用属性记录自关联函数定义。静态量保留每个无序构型的结果，自关联仅保留均值和标准误。再次扫描时请更换 `output_dir`，脚本不会覆盖已有目录；已有文件不会自动迁移。

从版本 8 起不再保存 `disorder` 数组，也不从 worker 返回该数组。根属性保留 `disorder_seed`、`L`、`ndisorder`、`p`、`Jstrong`、`Jweak` 和 `julia_version`。需要构型时，使用写入时的 Julia/Random 版本及生成代码重构（跨版本不保证随机序列一致）：

```julia
using HDF5
include("random_ising/run_slurm.jl") # 从仓库根目录，在已配置的 Julia 环境中执行
disorder = h5open(RandomIsingScan.read_disorder, "random_ising/results/run_004/L_10.h5", "r")
# disorder 的 Julia 维度为 (2L^2, ndisorder)，第 a 列对应第 a 个无序构型。
```

`read_disorder(file)` 也支持直接读取旧文件已有的 `disorder` dataset。原先使用 `read(file["disorder"])` 的分析代码应改用该接口。从版本 12 起不再保存 `parameters` dataset；文件初始化以根属性 `format_version` 是否存在判断，该属性在公共属性及 `temperatures` 写完后设置。

从版本 11 起，温度组不再保存 `realization_seeds`。保留组属性 `seed` 和根属性 `ndisorder`，通过以下接口读取或重构采样种子（同样需要写入时的 Julia/Random 版本）：

```julia
realization_seeds = RandomIsingScan.read_realization_seeds(file, group)
# 新文件等价于：
# rand(MersenneTwister(read(attributes(group)["seed"])), UInt32,
#      read(attributes(file)["ndisorder"]))
```

该接口兼容旧文件已有的 `realization_seeds` dataset；原先直接读取该 dataset 的代码应改用此接口。

两套环境各自维护 `Manifest.toml`，在各自机器上通过 `Pkg` 生成，不复制个人电脑的 Manifest 到服务器环境。

个人电脑无需 Slurm 的小规模验证（两个进程，每进程两个线程，临时输出并与串行结果比较）：

```bash
julia --threads=2 random_ising/test_run_slurm.jl
```
