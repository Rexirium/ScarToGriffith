# Plot audit

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\full_fixed_20261003_230245.h5`
Boundary: periodic
Field distribution: fixed; h = h₀/e; critical h0=1.0
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4
Sizes: [16, 32, 64, 128]
Gap fields (2×2): [1.5135612484362082, 1.9952623149688795, 2.7542287033381663, 3.019951720402016]
Correlation fields (2×3): [0.5011872336272722, 1.0, 1.5135612484362082, 1.9952623149688795, 2.7542287033381663, 3.019951720402016]
Autocorrelation: 2×2 size panels, the same six fields, from the same input file.

Real-time autocorrelation: included; linear vertical axis preserves negative values.
Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.
Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.
Scaled gap density: transform each sample to x = ln(ΔE)/√L, then histogram with common bin width 0.1 for every L; normalize by all samples.
Imaginary-time mean curves use stored autocorrelation_mean directly, retaining the stored SEM. Real-time curves keep their signs.
Autocorrelation effective log-log slopes: one curve per L, all 51 fields with h0 >= 1.0; absolute unweighted OLS slope of ln(autocorrelation_mean) versus ln(time), with intercept, over all finite positive times and positive finite means.
Slopes are descriptive, not automatically 1/z. Fixed fields are gapped for h0 > 2.718281828459045.
Mean log spatial correlations are mean(log C), not log(mean C). No resampling.
Time distributions: L=128, fields=[1.5135612484362082, 1.9952623149688795, 2.7542287033381663, 3.019951720402016], target times=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], actual nearest times=[1.0, 3.019951720402016, 10.0, 30.19951720402016, 100.0, 301.9951720402016]; 60 equal-width bins per curve, normalized by all samples.
Collapse minimizes sum over times and quantiles of [ln Q_p(-ln C) - μ ln τ - a_p]^2, with independent intercept a_p and p=0.05:0.05:0.95; only positive quantiles of valid samples enter the fit.
Times with no valid samples are excluded from collapse fitting; fewer than two valid times gives unavailable (NaN) μ. Empty densities are omitted, with missing counts retained.
Time distributions use each sample directly in the negative logarithm. Nonpositive/nonfinite samples are omitted; missing mass remains in the density. μ and log-quantile RMS are descriptive fits, not an asymptotic exponent determination.
Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.
Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.

| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |
|---|---|---|---|---|---|
| 0.5011872336272722 | 16 | 50000 | 0 | 0 | 0 |
| 1.0 | 16 | 50000 | 0 | 0 | 0 |
| 1.5135612484362082 | 16 | 50000 | 0 | 0 | 0 |
| 1.9952623149688795 | 16 | 50000 | 0 | 0 | 0 |
| 2.7542287033381663 | 16 | 50000 | 0 | 0 | 0 |
| 3.019951720402016 | 16 | 50000 | 0 | 0 | 0 |
| 0.5011872336272722 | 32 | 50000 | 16467 | 0 | 0 |
| 1.0 | 32 | 50000 | 0 | 0 | 0 |
| 1.5135612484362082 | 32 | 50000 | 0 | 0 | 0 |
| 1.9952623149688795 | 32 | 50000 | 0 | 0 | 0 |
| 2.7542287033381663 | 32 | 50000 | 0 | 0 | 0 |
| 3.019951720402016 | 32 | 50000 | 0 | 0 | 0 |
| 0.5011872336272722 | 64 | 50000 | 49862 | 0 | 0 |
| 1.0 | 64 | 50000 | 18 | 0 | 0 |
| 1.5135612484362082 | 64 | 50000 | 0 | 1 | 0 |
| 1.9952623149688795 | 64 | 50000 | 0 | 19 | 0 |
| 2.7542287033381663 | 64 | 50000 | 0 | 21 | 0 |
| 3.019951720402016 | 64 | 50000 | 0 | 23 | 0 |
| 0.5011872336272722 | 128 | 50000 | 50000 | 0 | 0 |
| 1.0 | 128 | 50000 | 2200 | 0 | 0 |
| 1.5135612484362082 | 128 | 50000 | 0 | 47 | 0 |
| 1.9952623149688795 | 128 | 50000 | 0 | 51 | 0 |
| 2.7542287033381663 | 128 | 50000 | 0 | 56 | 4 |
| 3.019951720402016 | 128 | 50000 | 0 | 56 | 11 |

## Time-distribution collapse

| h0 | L | μ | log-quantile RMS before | after | invalid samples at each τ |
|---|---|---|---|---|---|
| 1.5135612484362082 | 128 | 0.6653482446985981 | 1.3057454983309835 | 0.07108843793717916 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 1.9952623149688795 | 128 | 0.7963830948911305 | 1.5611850372821305 | 0.04333954440230469 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 2.7542287033381663 | 128 | 0.8748002213228366 | 1.7146611597577563 | 0.0376082548578306 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 3.019951720402016 | 128 | 0.8896713045437405 | 1.7437507226602833 | 0.03547264700419634 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
