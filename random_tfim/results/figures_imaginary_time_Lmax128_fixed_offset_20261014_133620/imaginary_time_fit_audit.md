# Imaginary-time fits

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\Lmax128_fixed_offset_20261014_133620.h5`; h=h0/e; sizes=[16, 32, 64, 128]; samples=50000; offset=true.
Lowest odd-parity transition subtracted; fit the stored arithmetic mean without further subtraction.
Figure 1 and imaginary_time_fits.csv contain only L=128; Figure 2 scans all 4 sizes.
Selected fields: FM: 0.19952623149688797; FM: 0.6025595860743578; Griffiths: 1.096478196143185; Griffiths: 1.3803842646028848; Griffiths: 1.7378008287493754; Griffiths: 2.0892961308540396; Griffiths: 2.51188643150958; PM: 3.019951720402016; PM: 5.011872336272722; PM: 10.0.
FM: h0<1; Griffiths: 1<h0<e; gapped PM: h0>e.
Requested fit window: [1.0, end of stored time grid]. All finite positive times and finite positive C enter both models.
Plot uses the same time window, with a 1e-12 vertical plot floor that does not truncate the fits. SEM only controls uncertainty shading.
Actual windows and point counts appear in imaginary_time_fits.csv.
Models: C=A*tau^(-alpha) and C=A*exp(-lambda*tau); A>0, alpha/lambda>=0.
Legend notation: 1/z=alpha; 1/xi_tau=lambda.
Unweighted linear regression of log(C), with nonnegative decay rate.
Minimize chi_square=sum((log(C)-log(C_model))^2), using the same finite positive data for both models.
Select the model with smaller reduced_chi_square=chi_square/(n-2).
Both models have 2 nominal parameters, so this is equivalent to comparing chi_square.
CSV records both fits and the selected flag.
Historical CSV names chi_square/reduced_chi_square denote SSE and SSE/(n-2), equivalent to unit log errors.
These are not measurement-error-normalized chi-squares. No time covariance or resampling is used.
Model selection is descriptive, not a goodness-of-fit probability or asymptotic exponent claim.
In particular FM plateaus and crossover regions need not follow either simple decay law.
Plot uses stored means/SEM directly, with the existing uncertainty masking; zero time omitted.
Figure 2: 2x2 panels with in-axis legends, log h0 axes spanning [0.1, 10]; panels 1 and 3 have linear vertical axes, panels 2 and 4 have log vertical axes (nonpositive values omitted); 101 field-grid points. Absent groups remain gaps in curves and are excluded from fitting and crossing interpolation.
Each curve is one size. Panel 1: power-law rate 1/z; panel 2: exponential rate 1/xi_tau.
Both rate panels show the corresponding candidate model regardless of which model wins.
Panel 3: min(power.reduced_chi_square, exponential.reduced_chi_square) at each (L,h0).
Panel 4: both candidate reduced chi-squares at L=128. Crossings: Any[(h0 = 0.10660375782721705, reduced_chi_square = 7.883675775820859, left = 0.10471285480508996, right = 0.1096478196143185), (h0 = 0.11394972528775203, reduced_chi_square = 5.591793899850368, left = 0.1096478196143185, right = 0.1148153621496883), (h0 = 0.11851842213823205, reduced_chi_square = 8.929811899588328, left = 0.1148153621496883, right = 0.12022644346174127), (h0 = 0.13046632304793412, reduced_chi_square = 7.963205699407534, left = 0.12589254117941673, right = 0.1318256738556407), (h0 = 0.13424687905129723, reduced_chi_square = 7.8403308541531676, left = 0.1318256738556407, right = 0.1380384264602885), (h0 = 0.14085320241797356, reduced_chi_square = 6.913675507130728, left = 0.1380384264602885, right = 0.14454397707459277), (h0 = 0.14710035476378297, reduced_chi_square = 8.267680188067636, left = 0.14454397707459277, right = 0.15135612484362082), (h0 = 0.17737361573761892, reduced_chi_square = 8.051788225253983, left = 0.17378008287493754, right = 0.18197008586099836), (h0 = 0.18429811926738252, reduced_chi_square = 8.086725132706315, left = 0.18197008586099836, right = 0.19054607179632474), (h0 = 0.19587175266640391, reduced_chi_square = 8.46539141418099, left = 0.19054607179632474, right = 0.19952623149688797), (h0 = 0.205494314744932, reduced_chi_square = 5.108970250035234, left = 0.19952623149688797, right = 0.20892961308540392), (h0 = 0.2572203095735006, reduced_chi_square = 6.019913639099006, left = 0.251188643150958, right = 0.2630267991895382), (h0 = 0.26648874721857724, reduced_chi_square = 6.7468532084645645, left = 0.2630267991895382, right = 0.2754228703338166), (h0 = 1.4632499916848332, reduced_chi_square = 9.708939569246503, left = 1.4454397707459274, right = 1.5135612484362082), (h0 = 1.5995739832986264, reduced_chi_square = 30.912257585473835, left = 1.5848931924611136, right = 1.6595869074375607), (h0 = 1.7596328438619673, reduced_chi_square = 121.75079723491639, left = 1.7378008287493754, right = 1.8197008586099834)].
Intersections are interpolated linearly in log(h0) between adjacent fields with opposite signs of the model-score difference.
Exact equal scores on grid points are retained; no extrapolation. These are numerical model-score crossings, not phase boundaries.
Crossing coordinates and bracketing fields are saved in imaginary_time_model_crossings.csv.
Figure 2 uses the same unweighted fits, masks, and requested time window as Figure 1.
All field-scan rates, both reduced chi-squares, selected models, and actual windows are in imaginary_time_field_scan.csv.
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4.
