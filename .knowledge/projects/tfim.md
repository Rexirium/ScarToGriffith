# tfim：纯净开链对照

入口：[用法](../../tfim/README.md) · [clean_tfim.jl](../../tfim/clean_tfim.jl)。共用 [random_tfim 约定](random_tfim.md)，这里只记差异。

| 内容 | 必须保留的差异 |
|---|---|
| 物理约定 | OBC、Pauli、所有 `J=1`，输入 h 就是实际横场，不作 `h0/e` 换算。ħ=1，有限链零温确定宇称基态。 |
| 格点与减项 | `j=cld(L,2)`（偶数链左中点），不是空间平均；调用 `autocorrelation(...; time_domain=:imaginary,offset=true,rcond_tol=1.0)`。扣最低奇宇称跃迁后 `C(0)=1-w_j`，不重新归一化。当前底层优先 pairing，`rcond_tol=1` 控制回退的全关联 Pfaffian 路径。 |
| 数值验证 | [runtests.jl](../../tfim/test/runtests.jl)：双自旋单键 OBC 独立解析解 atol `1e-14`，并检查奇偶链、接口一致性、反射、容差内正性/单调性及铁磁平台去除。 |
| 决策与限制 | CSV 保留原始负值/下溢；绘图跳过非正值。拟合仅用有限且 `C≥1e-12` 的相同点，比较 log 残差 SSE/(n−2)。阈值不是严格误差界；指数仅描述窗口内衰减，不能称作随机 Griffiths 的 1/z。 |
| 文献与实现 | 自由费米子与 offset 的对应关系继承 random_tfim；这是无序研究的纯净对照，不是 Young–Rieger 随机分布的复现。旧 PBC 平台、指数和拟合值不用于当前 OBC 减项结果验收。 |

尚未完成任意精度长时尾部核验；缩小步长或曲线看起来平滑不能消除消去误差。
