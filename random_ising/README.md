# 二维随机键 Ising 的响应函数

在项目根目录启动 `julia --project=. --threads=auto`，然后运行：

```julia
include("random_ising/observables.jl")

L, T, p, r = 16, 2.0, 0.5, 0.3
rng = MersenneTwister(10)
# 已有构型时，直接用自己的 Js 替换下面这一行。
Js = [ifelse.(rand(rng, 2L^2) .< p, 1.0, r) for _ in 1:4]
C, chi, chi_local = random_bond_observables(L, T, Js;
    mcs=8192, thermalization=2048, binsize=64, seed=1234)

C[1]              # 第一个构型的总热容
chi[1]            # 第一个构型的总磁化率
chi_local[1]      # 第一个构型的 L×L 局域磁化率矩阵
chi_local[1][x,y] # 格点 (x,y) 对均匀外场的响应
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
| `chi_local` | 每个格点为 `〈sᵢM〉/T` | `Vector{Matrix{Float64}}` |

这里使用有限系统零场对称系综，`〈sᵢ〉=〈M〉=0`。
局域矩阵是均匀外场下的空间响应图，不是所有格点对的 `χᵢⱼ` 矩阵。
`sum(chi_local[a]) ≈ chi[a]`。`C` 和 `chi` 都是整个系统的量；
除以 `L²` 得到每格点热容和磁化率。没有减去 `〈|M|〉²`。

设置 `details=true` 可取得具名结果和每个输出对应的分块 jackknife 误差：

```julia
out = random_bond_observables(L, T, Js; details=true)
out.heat_capacity
out.errors.heat_capacity
out.errors.local_susceptibility[1]
out.metadata  # 包版本、种子、采样长度、块大小等
```

默认使用 SpinMonteCarlo 的 Swendsen–Wang 更新，能处理零耦合。
`update=:local` 用于交叉检查：每步以 1/2 概率运行包的 Metropolis sweep，
随后总是执行一次随机单点热浴更新，以避免自由自旋的确定性翻转循环。
每个构型使用独立初始化的随机数流；相同输入、顺序和种子可复现。
每个构型通过独立的 `Parameter` 调用 `runMC`，由框架负责热化、测量、分块和 jackknife。
不同构型通过 `Threads.@threads` 并行计算，输出顺序与输入一致，种子不依赖线程调度。
可用 `--threads=4` 指定线程数，或用 `--threads=1` 串行运行；函数调用方式不变。
自定义 `Estimator` 在 `simple_estimator` 的基础上添加各格点的 `sᵢM/T`，
内置后处理负责热容和总磁化率所需的统计，返回前将每格点量乘以 `L²`。
SpinMonteCarlo v1.2.2 会先保存逐步测量值再分块，因此内存随 `L²*mcs` 增长。
多线程运行时，每个同时计算的构型都需要这部分内存。

误差描述单个构型的热采样误差，不包含无序平均的误差。
正式计算前需分别增加热化步数、测量步数和块大小，并比较不同种子，检查结果与误差是否稳定。
默认步数不保证所有温度和无序强度下都已收敛。

测试命令：

```powershell
julia --project=. --threads=4 --check-bounds=yes random_ising/test_observables.jl
```

测试用独立的 `3×3` 精确枚举核对非均匀耦合与纯模型，
并检查局域响应求和、零耦合极限、复现性、输入不被修改和非法输入。
多构型结果及误差还会逐项对照串行 `runMC`，覆盖两种更新算法。
实现针对本项目已安装的 SpinMonteCarlo v1.2.2 源码核对；
包的接口说明见 [官方文档](https://yomichi.github.io/SpinMonteCarlo.jl/latest/)。
