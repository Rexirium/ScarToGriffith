# Imaginary-time fits with a constant background

Input: `D:\Documents\mywork\ScarToGriffith\random_tfim\results\full_fixed_20261003_230245.h5`; h=h0/e; sizes=[16, 32, 64, 128]; samples=50000.
Figure 1: L=128, representative FM/Griffiths/PM fields; Figure 2: all sizes, h0>=1.
Figure 1 subtracts the selected model's B from the stored mean and fitted curve.
SEM is unchanged and excludes uncertainty in fitted B; nonpositive shifted values are masked on the log axis.
Background subtraction is for display only; fitting and model selection still use the original C.
Models: C=A*tau^alpha+B, C=A*exp(-lambda*tau)+B.
Constraints: A>0, B>=0, alpha<=0, lambda>=0. Power rate=-alpha; exponential rate=lambda.
Window: [1.0, 1000.0]; identical finite positive times and C for both models.
Minimize sum((log(C)-log(C_fit))^2), retaining the original script's unweighted log-space objective.
SEM is used only for plot shading. Tiny positive tails remain in the fit; plot floor is 1e-12.
Compare log SSE/(N-3), with three nominal parameters for both models; smaller wins (power wins exact ties).
This is not measurement-error-normalized chi-square, a fit probability, or a phase-boundary estimator.
A damped Gauss-Newton solver uses analytic derivatives, scaled times and log-sum-exp predictions.
Three positive-B starts times three rate starts and the exact B=0 boundary are compared.
Convergence uses projected gradient or jointly stable score and parameters; CSV records the flag.
Flat curves make A/B and the decay rate unidentifiable; N-3 is only the nominal degrees of freedom.
No covariance correction, bootstrap uncertainty, or global-optimum guarantee is supplied.
Figure 2 panels: -alpha, lambda, best log SSE/(N-3), and both scores at L=128.
Model-score crossings are interpolated linearly in log(h0), without extrapolation: Any[(h0 = 1.7473320309184552, reduced_chi_square = 1.7213924611191866, left = 1.7378008287493754, right = 1.8197008586099834)].
Nonconverged scan candidates: Tuple{Any, Any, Any}[].
CSV files contain both candidate parameters/scores and model selections.
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4.
