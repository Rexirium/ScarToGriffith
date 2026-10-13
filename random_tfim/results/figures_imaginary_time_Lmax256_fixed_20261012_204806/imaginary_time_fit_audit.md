# Imaginary-time fits

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\Lmax256_fixed_20261012_204806.h5`; h=h0/e; sizes=[32, 64, 128, 256]; samples=50000; offset=true.
Lowest odd-parity transition subtracted; fit the stored arithmetic mean without further subtraction.
Figure 1 and imaginary_time_fits.csv contain only L=256; Figure 2 scans all 4 sizes.
Selected fields: FM: 0.19952623149688797; FM: 0.6025595860743578; Griffiths: 1.096478196143185; Griffiths: 1.3803842646028848; Griffiths: 1.7378008287493754; Griffiths: 2.0892961308540396; Griffiths: 2.51188643150958; PM: 3.019951720402016; PM: 5.011872336272722; PM: 10.0.
FM: h0<1; Griffiths: 1<h0<e; gapped PM: h0>e.
Requested fit window: [1.0, 1000.0]. All finite positive times and finite positive C enter both models.
Plot uses the same time window, with a 1e-12 vertical plot floor that does not truncate the fits. SEM only controls uncertainty shading.
Actual windows and point counts appear in imaginary_time_fits.csv.
Models: C=A*tau^(-alpha), C=A*exp(-lambda*tau); A>0, alpha/lambda>=0; no offset.
Legend notation: 1/z=alpha; 1/xi_tau=lambda; C=A*exp(-tau/xi_tau).
Fit natural-log coordinates x=ln(tau), Y=ln(C) with equal weights.
Power: Y=logA-alpha*x; exponential: Y=logA-lambda*exp(x).
Both models are linear in logA and the decay rate; use unweighted linear regression with nonnegative rate.
Minimize chi_square=sum((Y-Y_fit)^2) in log space.
For fixed n, minimizing this also minimizes reduced_chi_square; the horizontal time grid is exact.
Select the model with smaller reduced_chi_square=chi_square/(n-2).
Both models use identical points and have two parameters, so this is equivalent to comparing chi_square.
CSV records both fits and the selected flag.
Historical CSV names chi_square/reduced_chi_square denote SSE and SSE/(n-2), equivalent to unit log errors.
These are not measurement-error-normalized chi-squares. No time covariance or resampling is used.
Model selection is descriptive, not a goodness-of-fit probability or asymptotic exponent claim.
In particular FM plateaus and crossover regions need not follow either simple decay law.
Plot uses stored means/SEM directly, with the existing uncertainty masking; zero time omitted.
Figure 2: 2x2 panels with in-axis legends, log h0 axes; panels 1 and 3 have linear vertical axes, panels 2 and 4 have log vertical axes (nonpositive values omitted); all 51 stored fields h0>=1.
Each curve is one size. Panel 1: power-law rate 1/z; panel 2: exponential rate 1/xi_tau.
Both rate panels show the corresponding candidate model regardless of which model wins.
Panel 3: min(power.reduced_chi_square, exponential.reduced_chi_square) at each (L,h0).
Panel 4: both candidate reduced chi-squares at L=256. Crossings: Any[(h0 = 1.6352229270894332, reduced_chi_square = 5.4077187987450595, left = 1.5848931924611136, right = 1.6595869074375607)].
Intersections are interpolated linearly in log(h0) between adjacent fields with opposite signs of the model-score difference.
Exact equal scores on grid points are retained; no extrapolation. These are numerical model-score crossings, not phase boundaries.
Crossing coordinates and bracketing fields are saved in imaginary_time_model_crossings.csv.
Figure 2 uses the same unweighted fits, masks, and requested time window as Figure 1.
All field-scan rates, both reduced chi-squares, selected models, and actual windows are in imaginary_time_field_scan.csv.
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4.
