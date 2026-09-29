# Run: julia --startup-file=no random_tfim/test/compare_imaginary_tails.jl [output_dir]
# Compare existing OBC implementations; neither is a high-precision reference.
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM, LinearAlgebra, Random

function main(args)
    length(args) <= 1 || error("expected at most one output directory")
    output = isempty(args) ? joinpath(@__DIR__, "..", "results", "imaginary_tail_comparison") : abspath(args[1])
    points_path = joinpath(output, "points.csv")
    summary_path = joinpath(output, "summary.csv")
    any(ispath, (points_path, summary_path)) && error("comparison output already exists")
    BLAS.set_num_threads(1)

    # Exact decoupled-spin check also warms up both implementations.
    check_times = [0.0, 1.0, 10.0, 100.0]
    F = svd!(RandomTFIM.fermion_matrix(zeros(7), ones(8), Val(:open)))
    for values in (RandomTFIM.autocorrelation(F, F, check_times, 4, 1.0; rcond_tol=0.0),
                   RandomTFIM.string_autocorrelation(F, check_times, 4, false))
        @assert all(isapprox.(values, exp.(-2 .* check_times); rtol=1e-12, atol=0))
    end

    times = [0.0; 10.0 .^ range(-1, 3; length=41)]
    first_time(mask) = (k = findfirst(mask); isnothing(k) ? NaN : times[k])
    mkpath(output)
    open(points_path, "w") do points
        open(summary_path, "w") do summary
            println(points, "L,h0,seed,j,tau,C_det,C_pf,absolute_difference,relative_difference_vs_pf")
            println(summary, "L,h0,seed,j,det_seconds,pf_seconds,first_relative_difference_gt_0p001,first_negative_det,first_negative_pf,first_zero_pf")
            # Same factorization, site and time grid isolate the contraction algorithm.
            for L in (16, 64, 128), h0 in (0.5, 1.0, 3.0), seed in 1:3
                J, h = sample_disorder(Xoshiro(seed), L, h0; boundary=:open)
                F = svd!(RandomTFIM.fermion_matrix(J, h, Val(:open)))
                j = L ÷ 2
                det_seconds = @elapsed C_det = RandomTFIM.autocorrelation(F, F, times, j, 1.0; rcond_tol=0.0)
                pf_seconds = @elapsed C_pf = RandomTFIM.string_autocorrelation(F, times, j, false)
                @assert isapprox(C_det[1], 1; atol=1e-10)
                @assert isapprox(C_pf[1], 1; atol=1e-10)
                differences = abs.(C_det .- C_pf)
                # A zero Pfaffian gives no usable relative reference (including underflow).
                relative = [isfinite(p) && !iszero(p) ? d / abs(p) : NaN
                            for (d, p) in zip(differences, C_pf)]
                for k in eachindex(times)
                    println(points, join((L, h0, seed, j, times[k], C_det[k], C_pf[k], differences[k], relative[k]), ','))
                end
                disagreement = first_time(relative .> 1e-3)
                println(summary, join((L, h0, seed, j, det_seconds, pf_seconds, disagreement,
                    first_time(C_det .< 0), first_time(C_pf .< 0), first_time(iszero.(C_pf))), ','))
                println("L=$L h0=$h0 seed=$seed: first >0.1% difference at tau=$disagreement; det=$(round(det_seconds; digits=4))s pf=$(round(pf_seconds; digits=4))s")
                flush(points)
                flush(summary)
            end
        end
    end
    println("Saved $points_path and $summary_path")
    println("Times exclude SVD and compilation; single measurements, not benchmark medians.")
    println("NaN in first-event columns means no event on this grid, not proof of accuracy.")
end

main(ARGS)
