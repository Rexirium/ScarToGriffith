# Imaginary-time fits

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\full_fixed_20261003_230245.h5`; h=h0/e; sizes=[16, 32, 64, 128]; samples=50000.
Figure 1 and imaginary_time_fits.csv contain only L=128; Figure 2 scans all four sizes.
Selected fields: FM: 0.19952623149688797; FM: 0.6025595860743578; Griffiths: 1.096478196143185; Griffiths: 1.3803842646028848; Griffiths: 1.7378008287493754; Griffiths: 2.0892961308540396; Griffiths: 2.51188643150958; PM: 3.019951720402016; PM: 5.011872336272722; PM: 10.0.
FM: h0<1; Griffiths: 1<h0<e; gapped PM: h0>e.
Requested fit window: [0.0, Inf]. All finite positive times and finite positive C enter both models.
The 1e-8 vertical plot floor does not truncate the fits. SEM affects scoring and shading, not fitted parameters.
Actual windows and point counts appear in imaginary_time_fits.csv.
Models: C=A*tau^(-alpha), C=A*exp(-lambda*tau); A>0, alpha/lambda>=0; no offset.
Legend notation: 1/z=alpha; 1/xi_tau=lambda; C=A*exp(-tau/xi_tau).
Fit natural-log coordinates x=ln(tau), Y=ln(C) with equal weights.
Power: Y=logA-alpha*x; exponential: Y=logA-lambda*exp(x).
Both models are linear in logA and the decay rate; use unweighted linear regression with nonnegative rate.
Fit parameters minimize log_sse=sum((Y-Y_fit)^2) with no SEM weights.
At those fixed parameters evaluate chi_square=sum(((Y-Y_fit)/(SEM/C))^2).
Both models use the same scoring subset: fit points with finite positive SEM and SEM/C.
Select the model with smaller reduced_chi_square=chi_square/(n_score-2).
n counts fitting points; n_score counts scoring points. Invalid SEM never removes a point from fitting.
Both models use identical points and have two parameters, so this is equivalent to comparing chi_square.
CSV records both fits and the selected flag.
SEM/C is a first-order approximation to log-space error. No time covariance or resampling is used.
Scores evaluate unweighted fits; parameters do not minimize weighted chi-square. Do not assume a chi-square reference distribution.
Model selection is descriptive, not a goodness-of-fit probability or asymptotic exponent claim.
In particular FM plateaus and crossover regions need not follow either simple decay law.
Plot uses stored means/SEM directly, with the existing uncertainty masking; zero time omitted.
Figure 2: 2x2 panels with in-axis legends, log h0 axes; panels 1-3 have linear vertical axes, panel 4 has a log vertical axis (nonpositive scores omitted); all 51 stored fields h0>=1.
Each curve is one size. Panel 1: power-law rate 1/z; panel 2: exponential rate 1/xi_tau.
Both rate panels show the corresponding candidate model regardless of which model wins.
Panel 3: min(power.reduced_chi_square, exponential.reduced_chi_square) at each (L,h0).
Panel 4: both candidate reduced chi-squares at L=128. Crossings: Any[(h0 = 1.7492717646876603, reduced_chi_square = 7.013670423256463e7, left = 1.7378008287493754, right = 1.8197008586099834)].
Intersections are interpolated linearly in log(h0) between adjacent fields with opposite signs of the model-score difference.
Exact equal scores on grid points are retained; no extrapolation. These are numerical model-score crossings, not phase boundaries.
Crossing coordinates and bracketing fields are saved in imaginary_time_model_crossings.csv.
Figure 2 uses the same unweighted fits, masks, and requested time window as Figure 1.
Both scripts share fit_autocorrelation.jl; default power-law rates match the old slope plot for decaying data.
A custom time window affects Figures 1/2; the old slope plot continues to use all valid positive times.
All field-scan rates, both reduced chi-squares, selected models, and actual windows are in imaginary_time_field_scan.csv.
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4.
