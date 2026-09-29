# Plot audit

Input: `D:\Documents\mywork\ScarToGriffith\random_tfim\results\full_20260929_163025.h5`
Boundary: periodic
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4
Sizes: [16, 32, 64, 128]
Gap fields (2×2): [1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Correlation fields (2×3): [0.1, 0.5011872336272722, 1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Autocorrelation: 2×2 size panels, the same six fields, from the same input file.

Real-time autocorrelation: included; linear vertical axis preserves negative values.
Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.
Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.
Scaled gap density: transform each sample to x = ln(ΔE)/√L, then histogram with common bin width 0.1 for every L; normalize by all samples.
Imaginary-time mean curves use |stored autocorrelation_mean|, retaining the stored SEM; this is not mean(|sample_Ct|). Real-time curves keep their signs.
Autocorrelation slopes (1/z): one curve per L, all 51 fields with h0 >= 1; absolute unweighted OLS slope of ln(|autocorrelation_mean|) versus ln(time), with intercept, over all finite positive times and nonzero finite means.
Mean log spatial correlations are mean(log C), not log(mean C). No resampling.
Time distributions: L=128, fields=[1.0, 1.9952623149688795, 5.011872336272722, 10.0], target times=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], actual nearest times=[1.0, 3.019951720402016, 10.0, 30.19951720402016, 100.0, 301.9951720402016]; 60 equal-width bins per curve, normalized by all samples.
Collapse minimizes sum over times and quantiles of [ln Q_p(-ln |C|) - μ ln τ - a_p]^2, with independent intercept a_p and p=0.05:0.05:0.95; only positive quantiles of valid samples enter the fit.
Time distributions use the absolute value of each sample before the negative logarithm. Zero/nonfinite samples are omitted; missing mass remains in the density. μ and log-quantile RMS are descriptive fits, not an asymptotic exponent determination.
Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.
Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.

| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |
|---|---|---|---|---|---|
| 0.1 | 16 | 10000 | 9770 | 0 | 0 |
| 0.5011872336272722 | 16 | 10000 | 28 | 0 | 0 |
| 1.0 | 16 | 10000 | 0 | 0 | 0 |
| 1.9952623149688795 | 16 | 10000 | 0 | 0 | 0 |
| 5.011872336272722 | 16 | 10000 | 0 | 0 | 0 |
| 10.0 | 16 | 10000 | 0 | 0 | 0 |
| 0.1 | 32 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 32 | 10000 | 4013 | 0 | 0 |
| 1.0 | 32 | 10000 | 14 | 0 | 0 |
| 1.9952623149688795 | 32 | 10000 | 0 | 0 | 0 |
| 5.011872336272722 | 32 | 10000 | 0 | 8 | 0 |
| 10.0 | 32 | 10000 | 0 | 9 | 0 |
| 0.1 | 64 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 64 | 10000 | 9850 | 0 | 0 |
| 1.0 | 64 | 10000 | 440 | 0 | 0 |
| 1.9952623149688795 | 64 | 10000 | 1 | 21 | 0 |
| 5.011872336272722 | 64 | 10000 | 0 | 25 | 0 |
| 10.0 | 64 | 10000 | 0 | 25 | 6 |
| 0.1 | 128 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 128 | 10000 | 10000 | 0 | 0 |
| 1.0 | 128 | 10000 | 2280 | 8 | 0 |
| 1.9952623149688795 | 128 | 10000 | 0 | 54 | 0 |
| 5.011872336272722 | 128 | 10000 | 0 | 56 | 14 |
| 10.0 | 128 | 10000 | 0 | 59 | 14 |

## Time-distribution collapse

| h0 | L | μ | log-quantile RMS before | after | invalid samples at each τ |
|---|---|---|---|---|---|
| 1.0 | 128 | 0.2543517451391797 | 0.5275535346604103 | 0.17287403583343677 | 1.0: 0/10000; 3.02: 0/10000; 10.0: 0/10000; 30.2: 0/10000; 100.0: 0/10000; 302.0: 0/10000 |
| 1.9952623149688795 | 128 | 0.48071541101978754 | 0.9494867944880188 | 0.11896500700260149 | 1.0: 0/10000; 3.02: 0/10000; 10.0: 0/10000; 30.2: 0/10000; 100.0: 0/10000; 302.0: 0/10000 |
| 5.011872336272722 | 128 | 0.4186658287253467 | 0.8694183454348533 | 0.2877690355748932 | 1.0: 0/10000; 3.02: 0/10000; 10.0: 0/10000; 30.2: 0/10000; 100.0: 0/10000; 302.0: 0/10000 |
| 10.0 | 128 | 0.3033337463423415 | 0.6791758230731665 | 0.32856851247973384 | 1.0: 0/10000; 3.02: 0/10000; 10.0: 0/10000; 30.2: 0/10000; 100.0: 0/10000; 302.0: 0/10000 |
