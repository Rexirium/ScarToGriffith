# Plot audit

Input: `D:\Documents\mywork\ScarToGriffith\random_tfim\results\full_uniform_20260930_001900.h5`
Boundary: periodic
Field distribution: uniform; h ∼ U(0, h₀); critical h0=1.0
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4
Sizes: [16, 32, 64, 128]
Gap fields (2×2): [1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Correlation fields (2×3): [0.1, 0.5011872336272722, 1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Autocorrelation: 2×2 size panels, the same six fields, from the same input file.

Real-time autocorrelation: included; linear vertical axis preserves negative values.
Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.
Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.
Scaled gap density: transform each sample to x = ln(ΔE)/√L, then histogram with common bin width 0.1 for every L; normalize by all samples.
Imaginary-time mean curves use stored autocorrelation_mean directly, retaining the stored SEM. Real-time curves keep their signs.
Autocorrelation effective log-log slopes: one curve per L, all 51 fields with h0 >= 1.0; absolute unweighted OLS slope of ln(autocorrelation_mean) versus ln(time), with intercept, over all finite positive times and positive finite means.
Slopes are descriptive, not automatically 1/z. Uniform fields retain a Griffiths region for all finite h0 > 1.
Mean log spatial correlations are mean(log C), not log(mean C). No resampling.
Time distributions: L=128, fields=[1.0, 1.9952623149688795, 5.011872336272722, 10.0], target times=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], actual nearest times=[1.0, 3.019951720402016, 10.0, 30.19951720402016, 100.0, 301.9951720402016]; 60 equal-width bins per curve, normalized by all samples.
Collapse minimizes sum over times and quantiles of [ln Q_p(-ln C) - μ ln τ - a_p]^2, with independent intercept a_p and p=0.05:0.05:0.95; only positive quantiles of valid samples enter the fit.
Times with no valid samples are excluded from collapse fitting; fewer than two valid times gives unavailable (NaN) μ. Empty densities are omitted, with missing counts retained.
Time distributions use each sample directly in the negative logarithm. Nonpositive/nonfinite samples are omitted; missing mass remains in the density. μ and log-quantile RMS are descriptive fits, not an asymptotic exponent determination.
Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.
Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.

| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |
|---|---|---|---|---|---|
| 0.1 | 16 | 50000 | 48881 | 0 | 0 |
| 0.5011872336272722 | 16 | 50000 | 161 | 0 | 0 |
| 1.0 | 16 | 50000 | 0 | 0 | 0 |
| 1.9952623149688795 | 16 | 50000 | 0 | 0 | 0 |
| 5.011872336272722 | 16 | 50000 | 0 | 0 | 0 |
| 10.0 | 16 | 50000 | 0 | 0 | 0 |
| 0.1 | 32 | 50000 | 50000 | 0 | 0 |
| 0.5011872336272722 | 32 | 50000 | 19748 | 0 | 0 |
| 1.0 | 32 | 50000 | 86 | 0 | 0 |
| 1.9952623149688795 | 32 | 50000 | 0 | 0 | 0 |
| 5.011872336272722 | 32 | 50000 | 0 | 8 | 0 |
| 10.0 | 32 | 50000 | 0 | 11 | 0 |
| 0.1 | 64 | 50000 | 50000 | 0 | 0 |
| 0.5011872336272722 | 64 | 50000 | 49267 | 0 | 0 |
| 1.0 | 64 | 50000 | 2175 | 0 | 0 |
| 1.9952623149688795 | 64 | 50000 | 1 | 21 | 0 |
| 5.011872336272722 | 64 | 50000 | 0 | 25 | 0 |
| 10.0 | 64 | 50000 | 0 | 27 | 2 |
| 0.1 | 128 | 50000 | 50000 | 0 | 0 |
| 0.5011872336272722 | 128 | 50000 | 50000 | 0 | 0 |
| 1.0 | 128 | 50000 | 11528 | 24 | 0 |
| 1.9952623149688795 | 128 | 50000 | 0 | 54 | 0 |
| 5.011872336272722 | 128 | 50000 | 0 | 57 | 13 |
| 10.0 | 128 | 50000 | 0 | 59 | 14 |

## Time-distribution collapse

| h0 | L | μ | log-quantile RMS before | after | invalid samples at each τ |
|---|---|---|---|---|---|
| 1.0 | 128 | 0.2553815962037307 | 0.5286608598084461 | 0.17040893608881003 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 1.9952623149688795 | 128 | 0.48542518353792335 | 0.9568989364656236 | 0.1039709939123944 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 5.011872336272722 | 128 | 0.5146430005149945 | 1.0149273068136184 | 0.11414075710559998 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
| 10.0 | 128 | 0.48316977758550644 | 0.9552367048757371 | 0.12657131048493417 | 1.0: 0/50000; 3.02: 0/50000; 10.0: 0/50000; 30.2: 0/50000; 100.0: 0/50000; 302.0: 0/50000 |
