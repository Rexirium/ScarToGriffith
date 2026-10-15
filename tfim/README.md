# Clean 横场 Ising 链的虚时自关联

采用 Pauli 约定的开边界链（OBC）：

\[
H=-\sum_{i=1}^{L-1}\sigma_i^z\sigma_{i+1}^z-h\sum_{i=1}^{L}\sigma_i^x.
\]

所有键为 `J=1`，横场为物理场强 `h>0`，不使用 `h0/e` 换算。
模型约定见[模型卡](../.knowledge/models/transverse-field-ising/MODEL.md)。

`clean_autocorrelation(h, times; L=128)` 调用
[random_tfim 的现有接口](../random_tfim/autocorrelation.jl)，传入 `L-1` 个键、
`time_domain=:imaginary, offset=true, rcond_tol=1.0`。
计算中点 `j=cld(L,2)`（偶数链取左中点）的关联；OBC 下它不等于空间平均。
支持任意整数链长 `L>=2`。内部格点使用 JW-string Pfaffian，端点使用谱和。

返回量为

\[
C_{\mathrm{offset}}(\tau)=
\langle GS_+|\sigma_j^z e^{-\tau(H-E_+)}\sigma_j^z|GS_+\rangle
-w_j e^{-\Delta\tau},\qquad
w_j=|\langle GS_-|\sigma_j^z|GS_+\rangle|^2,
\quad\Delta=E_--E_+.
\]

这里减去最低奇宇称态的完整跃迁贡献，不是常数背景拟合，也不是减去有限链的
`<sigma_z>^2`。因此 `C_offset(0)=1-w_j`，不再归一化为 1。

## 运行

```sh
julia --startup-file=no --project=@v1.13 tfim/plot_imaginary_time.jl
# 自定义输出目录、链长及计算/拟合时间窗口
julia --startup-file=no --project=@v1.13 tfim/plot_imaginary_time.jl tfim/results_obc 128 0.1 1000
```

默认 `L=128`，时间为 `0.1..10000` 的 241 个对数等距点，额外保存 `tau=0`（不参与拟合和绘图）。
场强为 `0.2, 0.6, 0.9, 1.0, 1.1, 1.4, 1.7, 2.5, 5.0, 10.0`，可修改脚本顶部的 `FIELDS`。
核心和测试只需 Julia 标准库；绘图另需 CairoMakie。

默认输出到 `tfim/results/`，重复运行更新同名文件；原周期链结果需重新计算：

- `figure1_imaginary_time_autocorrelation.png`：实线为 `C_offset`，虚线为最佳描述性拟合。
- `imaginary_time_data.csv`：原始返回值 `C`，包括零时间、微小负值和下溢值；附带 `boundary,offset,j`。
- `imaginary_time_fits.csv`：各场强的幂律及指数拟合、实际窗口、点数和模型选择。

图片和 CSV 遵循仓库 `.gitignore`，保留在本地。

## 拟合及数值精度

复用[现有拟合函数](../random_tfim/fit_autocorrelation.jl)，比较
`C_offset=A*tau^(-alpha)` 与 `C_offset=A*exp(-lambda*tau)`。
对 `log(C_offset)` 做等权最小二乘，以 `SSE/(n-2)` 较小者作为虚线。
这是描述性模型比较；图例的 `alpha` 不解释为 Griffiths 区的 `1/z`。

最低态贡献相减会在长时产生消去误差。CSV 保留原值，不裁剪或重新归一化；
横轴为对数标度，显示范围为 `0.1..1000`；纵轴为对数标度，下限为 `1e-16`、上限自动确定，跳过非正值。
拟合仅使用 `C_offset>=1e-12` 的有限正值，两个候选模型使用相同点。
该阈值是数值筛选而非严格误差界；不足三个有效点时拟合报错，应调整时间窗口。
CSV 记录实际拟合窗口。原周期链的铁磁平台、临界指数检查和拟合数值不适用于本次减项后的结果。

## 验证

```sh
julia --startup-file=no tfim/test/runtests.jl
julia --startup-file=no random_tfim/test/physics.jl
```

58 项测试覆盖奇偶链长、当前接口一致性、反射对称性、容差内正性和单调性、
双自旋单键 OBC 减项后的独立解析解、铁磁长时平台去除及非法输入。
尚未做任意精度长时尾部验证。
