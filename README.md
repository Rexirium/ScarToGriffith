# ScarToGriffith
From QMBS to QGP

项目物理约定、基底、验证标准和文献对应见 [.knowledge/README.md](.knowledge/README.md)；
按任务读取资料的规则见 [AGENTS.md](AGENTS.md)。论文统一放在 `.knowledge/literature/`，
张量网络与量子线路参考也保留在知识库中。

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
