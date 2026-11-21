# random_ising：二维随机键经典 Ising

入口：[用法](../../random_ising/README.md) · [响应实现](../../random_ising/observables.jl) · [临界温度](../../random_ising/critical_temp.jl)。

| 内容 | 必须保留的约定与依据 |
|---|---|
| 物理约定 | 方格 PBC，`s=±1`、`kB=1`，`H=-Σ J_ij s_i s_j`；输入是实际非负键值。二值采样 `J=1` 的概率为 `p`，否则 `J=r≥0`。`C=Var(H)/T²`、`χ=〈M²〉/T` 是总量，除以 `L²` 才是每格点量；不减 `〈\|M\|〉²`。`U4=1-〈M⁴〉/(3〈M²〉²)` 先逐构型计算再做无序平均。 |
| 构型与排序 | 无量子基底/sector。`i=x+L(y-1)`；键 `2i-1` 沿 +x、`2i` 沿 +y，坐标周期回绕。`L=2` 保留平行键。`χ_local[x,y]=〈s[x,y]M〉/T`，不除格点数。 |
| 动力学与初态 | SW 热化及静态测量结束的构型作固定参考态；随后每个热浴 sweep 有放回地选点 `L²` 次。`A(t)=Σ s_i(t0+t)s_i(t0)/L²`，`A(0)=1`；每个无序样本仅一条轨迹，不平均时间起点。SW 用于平衡采样，热浴 sweep 才是此处动力学时间。 |
| 数值验证 | [test_observables.jl](../../random_ising/test_observables.jl) 的独立 `3×3` 枚举：C、χ 偏差 `<6×估计误差+0.02`，U4 绝对误差 `0.02`；另有零键、串行/线程一致性与可复现检查。正式研究还须增加热化、测量、块长并换种子；默认参数不是收敛保证。 |
| 统计边界 | 静态量误差是单构型热采样分块 jackknife；A(t) 误差是跨无序样本 SEM，单样本为 NaN。局域响应误差来自块均值。不同温度复用同一批无序；同一样本的格点、温度、时间点不能当作独立样本。 |
| 决策与理由 | `critical_temp` 固定 Binder 目标取交点，拟合 `T_L=Tc+a L^(-1/ν)(ln L)^log_power`；默认 ν=1 是输入假设，不是测得指数。bootstrap 重采样完整无序曲线；置信区间不覆盖温度插值和标度形式误差。`r=0,p≤1/2` 用已知 Tc=0。 |
| 文献与实现 | [Münster 2024](../literature/Munster_2024_Griffiths_phase_large_deviations.md) 提供经典 Griffiths/稀有区背景，代码未实现其大偏差重要性抽样；`critical_temp` 中链接的 [随机 Ising 标度文献](https://arxiv.org/abs/0804.2788) 支持检查对数修正，不能据此把默认外推当成精确 Tc。 |

历史解释见 [note0917](../../notes/note0917/note0917.tex)：热容峰、宽分布或低温慢衰减不能单独确定 Griffiths 区；须独立定位有序转变。运行/文件版本细节留在子项目 README。
