# random_tfim：随机横场 Ising 开链

入口：[README](../../random_tfim/README.md) · [模块](../../random_tfim/RandomTFIM.jl)；[纯净对照](tfim.md) 直接复用此模块。

| 内容 | 必须保留的约定与依据 |
|---|---|
| 物理约定 | `H=-Σ_(i=1)^(L-1) J_i σz_i σz_(i+1)-Σ_i h_i σx_i`，Pauli、ħ=1、OBC、`L≥2,J≥0,h>0`。uniform：`J~U(0,1),h~U(0,h0)`；fixed：相同随机键、`h=h0/e`，两模式消耗相同随机数。两者临界参数 h0=1；fixed 的顺磁 Griffiths 区 `1<h0<e`，不能把 uniform 的相图直接套到 fixed。 |
| 基底与子空间 | [model.jl](../../random_tfim/model.jl)：`K=A+B=Bidiagonal(h,-J,:L)=U diag(ε) Vᵀ`，`G_ij=〈(c†_i-c_i)(c†_j+c_j)〉=-(VUᵀ)_ij`；`E0=-Σε`，自旋能隙 `Δ=2min ε`。有限链确定宇称基态，不是破缺对称乘积态；OBC 不需 PBC 的费米子边界扇区切换。 |
| 关联与熵 | [correlations](../../random_tfim/correlations.jl)：`C[i,r+1]=〈σz_i σz_(i+r)〉`，无效 OBC 起点为 NaN。[entropy](../../random_tfim/entanglement.jl)：切分 `1:cut`，由 `G[A,A]` 奇异值 ν 取 `p=(1-ν)/2`，每对 Majorana 只计一次二元熵，单位 nats；空/全链为 0，铁磁猫态极限为 ln2。 |
| 时间与减项 | [autocorrelation](../../random_tfim/autocorrelation.jl) 默认中点 `j=cld(L,2)`；`C_j(z)=Σ_n \|〈n\|σz_j\|0〉\|² exp[-(En-E0)z]`，z=τ 或 it；实时间只返回实部。API 默认 `offset=false`；扫描默认 true。减项为 `w_j exp(-Δz)`（实时间实部为 `w_j cos Δt`），故 `C_offset(0)=1-w_j`，不是减去 `〈σz〉²` 或常数平台。最低模简并时只扣 SVD 选中的一个态。 |
| 数值标准 | [physics.jl](../../random_tfim/test/physics.jl)：小链自旋 ED 检查 Δ（atol `1e-11`）、空间/时间关联、offset、Schmidt 熵（通常 atol `2e-10`）。解耦自旋 `C_j(τ)=exp(-2h_jτ)`；另检验反射、零时刻、虚时单调性。`resolved` 阈值 `64eps(Float64) max(Σε,1)` 只标记能隙分辨能力，不是严格误差界。 |
| 数值决策 | SVD 避免平方 K 丢失小奇异值；空间关联用增量 QR，避免奇异前缀破坏 Schur 更新。完整自关联按倒条件数选择行列式或 JW-string Pfaffian。虚时 offset 优先 [pairing](../../random_tfim/pairing_autocorrelation.jl) 避免两个近等数相减；`a=‖Zτ‖²_F/2≤1e-8` 用正项首阶尾部，截断相对界约 `5e-9`（不含输入舍入误差）。病态时回退到全关联再相减；非负不代表尾部准确。 |
| 无序与拟合 | 先逐样本、再无序平均；空间 log 先对每一对取 log 再平均起点，不能由 `log(sample_C)` 重建。非正/非有限值及未分辨能隙产生 NaN，不删样本后重归一化。SEM 按独立无序样本计算；`fit_autocorrelation` 比较等权 log 残差 SSE/(n−2)，字段虽叫 χ²，却不是测量误差归一化的 χ²，未计时间协方差。 |

JW 约定：`γ_(2j−1)=(∏_(l<j)σx_l)σz_j`、`γ_(2j)=(∏_(l<j)σx_l)σy_j`；协方差按全部奇 Majorana、再全部偶 Majorana 排列，`Γ=[0 −Gᵀ; G 0]`。改字符串方向或排序必须同步核对符号。

文献映射：[Young–Rieger 1996](../literature/Young_Rieger_1996_Numerical_study_random_transverse_field_Ising_chain.md) §III，尤其 Eq. (56) → `ground_state` 的 G 与空间关联行列式；原文的 PBC/宇称处理改为 OBC、SVD。§IV–V → 能隙/平均与典型关联诊断，不能照搬原文有限尺寸图形作为 OBC 验收值。纠缠公式的 [Peschel–Eisler 来源](https://arxiv.org/abs/0906.1663) 已在 README 引用；offset 和 pairing 尾部是项目扩展，不是 Young–Rieger 原文公式。

研究理由见 [note1006](../../notes/note1006/random_tfim.tex)：SEM 加权会让早时点压倒尾部，故现用等权 log 拟合；选中幂律或拟合分数交叉不等于相界/渐近 1/z。旧数据必须按自身 boundary/offset 元数据解释。当前扫描格式与早期 `disorder_ensemble` 输出不同，读取前查 README 的 HDF5 部分。
