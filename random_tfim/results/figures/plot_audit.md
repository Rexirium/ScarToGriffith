# Plot audit

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\full_sample1000.h5`
Julia: 1.13.0; project: `c:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\julia-env\local\Project.toml`
CairoMakie: 0.15.14; HDF5: 0.17.3
Boundary: periodic spins; even L; Pauli normalization

Natural logarithms. Error bands are ±1 SEM across independent disorder samples.
Gap densities use counts / (total samples × bin width), so missing probability is not renormalized away.
Log correlations use mean(sample_logC), not log(mean(sample_C)). Distances with any nonfinite log sample are omitted.
The average-C plot retains the supplied arithmetic averages; tiny negative individual correlations are counted below as numerical artifacts.

| h0 | L | samples | unresolved gaps | nonfinite log entries | negative C entries | last complete log r |
|---|---|---|---|---|---|---|
| 1.0 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 1.0 | 32 | 1000 | 3 | 0 | 0 | 16 |
| 1.0 | 64 | 1000 | 44 | 0 | 0 | 32 |
| 1.0 | 128 | 1000 | 242 | 0 | 0 | 64 |
| 1.3 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 1.3 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 1.3 | 64 | 1000 | 0 | 0 | 0 | 32 |
| 1.3 | 128 | 1000 | 7 | 539 | 0 | 25 |
| 1.5 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 1.5 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 1.5 | 64 | 1000 | 0 | 0 | 0 | 32 |
| 1.5 | 128 | 1000 | 1 | 5592 | 66 | 18 |
| 1.7 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 1.7 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 1.7 | 64 | 1000 | 0 | 32 | 0 | 20 |
| 1.7 | 128 | 1000 | 0 | 14895 | 450 | 14 |
| 2.0 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 2.0 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 2.0 | 64 | 1000 | 0 | 150 | 0 | 19 |
| 2.0 | 128 | 1000 | 0 | 29975 | 2633 | 16 |
| 2.3 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 2.3 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 2.3 | 64 | 1000 | 0 | 878 | 3 | 15 |
| 2.3 | 128 | 1000 | 0 | 35388 | 5809 | 14 |
| 3.0 | 16 | 1000 | 0 | 0 | 0 | 8 |
| 3.0 | 32 | 1000 | 0 | 0 | 0 | 16 |
| 3.0 | 64 | 1000 | 0 | 6271 | 163 | 11 |
| 3.0 | 128 | 1000 | 0 | 42457 | 11822 | 12 |
