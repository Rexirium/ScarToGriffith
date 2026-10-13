# Imaginary-time fits

Input: `C:\Users\徐子浩_物理学院\Documents\myworks\ScarToGriffith\random_tfim\results\Lmax128_fixed_offset_20261013_180213.h5`; h=h0/e; sizes=[16, 32, 64, 128]; samples=50000; offset=true.
Lowest odd-parity transition subtracted; fit the stored arithmetic mean without further subtraction.
Figure 1 and imaginary_time_fits.csv contain only L=128; Figure 2 scans all 4 sizes.
Selected fields: FM: 0.19952623149688797; FM: 0.6025595860743578; Griffiths: 1.096478196143185; Griffiths: 1.3803842646028848; Griffiths: 1.7378008287493754; Griffiths: 2.0892961308540396; Griffiths: 2.51188643150958; PM: 3.019951720402016; PM: 5.011872336272722; PM: 10.0.
FM: h0<1; Griffiths: 1<h0<e; gapped PM: h0>e.
Requested fit window: [1.0, 10000.0]. All finite positive times and finite positive C enter both models.
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
Panel 4: both candidate reduced chi-squares at L=128. Crossings: Any[(h0 = 0.10793031963276138, reduced_chi_square = 10.920388507848653, left = 0.10471285480508996, right = 0.1096478196143185), (h0 = 0.11395156205811806, reduced_chi_square = 5.588978233293467, left = 0.1096478196143185, right = 0.1148153621496883), (h0 = 0.11756514115532968, reduced_chi_square = 8.10154431374911, left = 0.1148153621496883, right = 0.12022644346174127), (h0 = 0.12773203500193278, reduced_chi_square = 10.960169178628933, left = 0.12589254117941673, right = 0.1318256738556407), (h0 = 0.13694365180026188, reduced_chi_square = 9.357108777266493, left = 0.1318256738556407, right = 0.1380384264602885), (h0 = 0.13911619079204182, reduced_chi_square = 7.905689197127661, left = 0.1380384264602885, right = 0.14454397707459277), (h0 = 0.1493185917570929, reduced_chi_square = 11.167595003800045, left = 0.14454397707459277, right = 0.15135612484362082), (h0 = 0.17454281023063084, reduced_chi_square = 10.42207439137393, left = 0.17378008287493754, right = 0.18197008586099836), (h0 = 0.1882391253115532, reduced_chi_square = 13.596873261696455, left = 0.18197008586099836, right = 0.19054607179632474), (h0 = 0.1946835013148408, reduced_chi_square = 10.111663315037456, left = 0.19054607179632474, right = 0.19952623149688797), (h0 = 0.20659167742293452, reduced_chi_square = 5.375187784918596, left = 0.19952623149688797, right = 0.20892961308540392), (h0 = 0.25725606609474355, reduced_chi_square = 6.019143204688232, left = 0.251188643150958, right = 0.2630267991895382), (h0 = 0.2664587504857371, reduced_chi_square = 6.739530556614376, left = 0.2630267991895382, right = 0.2754228703338166), (h0 = 1.588358923352369, reduced_chi_square = 97.33547534929866, left = 1.5848931924611136, right = 1.6595869074375607), (h0 = 2.745974334389112, reduced_chi_square = 7937.56284429235, left = 2.6302679918953817, right = 2.7542287033381663), (h0 = 2.7623107194851433, reduced_chi_square = 7951.237828940138, left = 2.7542287033381663, right = 2.884031503126606), (h0 = 3.4277511729304617, reduced_chi_square = 9089.691875473654, left = 3.311311214825911, right = 3.4673685045253166), (h0 = 3.505898843022867, reduced_chi_square = 9198.833764777002, left = 3.4673685045253166, right = 3.630780547701014), (h0 = 3.8929967070382787, reduced_chi_square = 9524.436884518256, left = 3.8018939632056115, right = 3.9810717055349722), (h0 = 4.072790936936692, reduced_chi_square = 9385.744121585387, left = 3.9810717055349722, right = 4.168693834703354), (h0 = 4.52877149983555, reduced_chi_square = 6848.625776627125, left = 4.36515832240166, right = 4.570881896148751), (h0 = 4.923787057641788, reduced_chi_square = 11661.450406282205, left = 4.786300923226384, right = 5.011872336272722), (h0 = 6.4386103226323925, reduced_chi_square = 8630.291107271341, left = 6.309573444801933, right = 6.606934480075959), (h0 = 6.777369188811692, reduced_chi_square = 8759.070373111801, left = 6.606934480075959, right = 6.918309709189365), (h0 = 9.457640229948812, reduced_chi_square = 6918.325461054823, left = 9.120108393559097, right = 9.549925860214358), (h0 = 9.643278127561649, reduced_chi_square = 6910.74371965104, left = 9.549925860214358, right = 10.0)].
Intersections are interpolated linearly in log(h0) between adjacent fields with opposite signs of the model-score difference.
Exact equal scores on grid points are retained; no extrapolation. These are numerical model-score crossings, not phase boundaries.
Crossing coordinates and bracketing fields are saved in imaginary_time_model_crossings.csv.
Figure 2 uses the same unweighted fits, masks, and requested time window as Figure 1.
All field-scan rates, both reduced chi-squares, selected models, and actual windows are in imaginary_time_field_scan.csv.
Julia: 1.13.0; CairoMakie: 0.15.15; HDF5: 0.17.4.
