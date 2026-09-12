# ScarToGriffith
From QMBS to QGP

Julia 环境统一放在 `julia-env/`：`local/` 用于个人电脑，`server/` 用于服务器。
分别安装依赖：

```bash
julia --project=julia-env/local -e 'using Pkg; Pkg.instantiate()'
julia --project=julia-env/server -e 'using Pkg; Pkg.instantiate()'
```

交互使用或运行其他 Julia 脚本时，通过 `--project=julia-env/local` 或
`--project=julia-env/server` 选择环境。
`random_ising/run_slurm.jl` 自动激活服务器环境，带 `--test` 时自动激活本地环境；
worker 使用与主进程相同的环境。扫描用法见 [random_ising/README.md](random_ising/README.md)。
