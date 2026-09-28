# Plot audit

Input: `D:\Documents\mywork\ScarToGriffith\random_tfim\results\full.h5`
Boundary: periodic spins; even L; Pauli normalization
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4
Sizes: [16, 32, 64, 128]
Gap fields (2×2): [1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Correlation fields (2×3): [0.1, 0.5011872336272722, 1.0, 1.9952623149688795, 5.011872336272722, 10.0]
Autocorrelation: 2×2 size panels, the same six fields, from the same input file.

Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.
Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.
Mean log correlations are mean(log C), not log(mean C). No refitting or resampling.
Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.
Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.

| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |
|---|---|---|---|---|---|
| 0.1 | 16 | 10000 | 9772 | 0 | 0 |
| 0.5011872336272722 | 16 | 10000 | 30 | 0 | 0 |
| 1.0 | 16 | 10000 | 0 | 0 | 0 |
| 1.9952623149688795 | 16 | 10000 | 0 | 0 | 0 |
| 5.011872336272722 | 16 | 10000 | 0 | 0 | 0 |
| 10.0 | 16 | 10000 | 0 | 0 | 0 |
| 0.1 | 32 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 32 | 10000 | 4009 | 0 | 0 |
| 1.0 | 32 | 10000 | 14 | 0 | 0 |
| 1.9952623149688795 | 32 | 10000 | 0 | 0 | 0 |
| 5.011872336272722 | 32 | 10000 | 0 | 8 | 0 |
| 10.0 | 32 | 10000 | 0 | 9 | 0 |
| 0.1 | 64 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 64 | 10000 | 9850 | 0 | 0 |
| 1.0 | 64 | 10000 | 442 | 0 | 0 |
| 1.9952623149688795 | 64 | 10000 | 1 | 20 | 0 |
| 5.011872336272722 | 64 | 10000 | 0 | 25 | 0 |
| 10.0 | 64 | 10000 | 0 | 25 | 5 |
| 0.1 | 128 | 10000 | 10000 | 0 | 0 |
| 0.5011872336272722 | 128 | 10000 | 10000 | 0 | 0 |
| 1.0 | 128 | 10000 | 2279 | 9 | 0 |
| 1.9952623149688795 | 128 | 10000 | 0 | 54 | 0 |
| 5.011872336272722 | 128 | 10000 | 0 | 56 | 9 |
| 10.0 | 128 | 10000 | 0 | 59 | 18 |
