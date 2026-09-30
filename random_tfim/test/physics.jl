using Test, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM
# Test-only scalar wrapper for the logarithmic Pfaffian kernel.
function pfaffian!(A::Matrix{ComplexF64})
    logabs, phase = RandomTFIM.logabspfaffian!(A)
    return phase * exp(logabs)
end
BLAS.set_num_threads(1)

@testset "Selectable transverse fields" begin
    for boundary in (:open, :periodic)
        rng_uniform, rng_fixed = Xoshiro(91), Xoshiro(91)
        for _ in 1:3
            uniform = sample_disorder(rng_uniform, 6, 3.0; boundary)
            fixed = sample_disorder(rng_fixed, 6, 3.0; boundary, field_distribution=:fixed)
            @test uniform.J == fixed.J
            @test all(0 .< uniform.h .<= 3)
            @test fixed.h == fill(3.0 / exp(1), 6)
        end
        result = disorder_ensemble(6, 3.0, [0.0, 0.4]; boundary,
            field_distribution=:fixed, nsamples=3, seed=91, keep_samples=true)
        rng = Xoshiro(91)
        for n in 1:3
            J, h = sample_disorder(rng, 6, 3.0; boundary, field_distribution=:fixed)
            @test result.sample_Ct[:, n] ≈ autocorrelation(J, h, [0.0, 0.4]; boundary)
            @test result.sample_real_Ct[:, n] ≈ autocorrelation(J, h, [0.0, 0.4]; boundary, time_domain=:real)
        end
    end
    @test_throws ArgumentError sample_disorder(Xoshiro(1), 6, 1.0; field_distribution=:bad)
    @test_throws ArgumentError disorder_ensemble(6, 1.0, [0.0]; field_distribution=:bad)
end

@testset "Ensemble summary and optional samples" begin
    for boundary in (:open, :periodic)
        kwargs = (; nsamples=3, seed=71, boundary)
        times = [0.0, 0.4]
        summary = disorder_ensemble(6, 1.0, times; kwargs...)
        samples = disorder_ensemble(6, 1.0, times; kwargs..., keep_samples=true)
        @test !hasproperty(summary, :pair_logC)
        @test !hasproperty(samples, :pair_logC)
        @test !hasproperty(summary, :gaps)
        @test !hasproperty(summary, :resolved)
        for field in (:sample_logC, :sample_logCt, :sample_real_logCt, :loggaps)
            @test !hasproperty(samples, field)
        end
        for field in keys(summary)
            @test isequal(getproperty(summary, field), getproperty(samples, field))
        end
        loggaps = map((gap, ok) -> ok ? log(gap) : NaN, samples.gaps, samples.resolved)
        for (values, avg, sem) in ((samples.gaps, :gap_mean, :gap_sem),
                (loggaps, :log_gap_mean, :log_gap_sem))
            @test getproperty(summary, avg) ≈ mean(values) nans=true
            @test getproperty(summary, sem) ≈ std(values) / sqrt(3) nans=true
        end
        for (raw, avg, sem) in ((:sample_C, :C_mean, :C_sem),
                (:sample_Ct, :Ct_mean, :Ct_sem),
                (:sample_real_Ct, :real_Ct_mean, :real_Ct_sem))
            @test !hasproperty(summary, raw)
            values = getproperty(samples, raw)
            @test getproperty(summary, avg) ≈ vec(mean(values; dims=2))
            @test getproperty(summary, sem) ≈ vec(std(values; dims=2)) / sqrt(3)
        end
        @test summary.Ct_sem isa Vector{Float64}
        @test summary.Ct_mean isa Vector{Float64}
        @test summary.logCt_mean isa Vector{Float64}
        @test summary.logCt_sem isa Vector{Float64}
        @test samples.logCt_mean ≈ vec(mean(log.(samples.sample_Ct); dims=2))
        @test size(samples.sample_Ct) == (length(times), kwargs.nsamples)
        single = disorder_ensemble(6, 1.0, [0.4]; boundary, nsamples=1)
        @test isnan(single.gap_sem) && isnan(single.log_gap_sem)
        @test all(isnan, single.C_sem)
        @test all(isnan, single.logC_sem)
        @test all(isnan, single.Ct_sem)
        @test all(isnan, single.logCt_sem)
        empty_time = disorder_ensemble(6, 1.0, Float64[]; boundary, nsamples=3, rmax=0)
        @test isempty(empty_time.Ct_mean) && isempty(empty_time.Ct_sem)
        @test isempty(empty_time.logCt_mean) && isempty(empty_time.logCt_sem)
        @test empty_time.C_mean == [1.0]
        @test empty_time.C_sem == [0.0]
    end
end

# Independent oracle: Pauli matrices in the sigma-z product basis.
function spin_oracle(J, h; periodic=true)
    L = length(h)
    H = zeros(2^L, 2^L)
    for s in 0:2^L-1, i in 1:L
        j = mod1(i+1, L)
        zi, zj = 1-2*((s >> (i-1)) & 1), 1-2*((s >> (j-1)) & 1)
        if periodic || i < L
            H[s+1, s+1] -= J[i] * zi * zj
        end
        H[xor(s, 1 << (i-1))+1, s+1] -= h[i]
    end
    F = eigen(Symmetric(H))
    psi = F.vectors[:, 1]
    @test norm(H * psi - F.values[1] * psi) < 1e-10
    Z = [1-2*((s >> (j-1)) & 1) for s in 0:2^L-1, j in 1:L]
    C = Z' * (abs2.(psi) .* Z)
    return (; E0=F.values[1], E1=F.values[2], C, Z, energies=F.values, vectors=F.vectors)
end

# Independent spectral sum, shared by imaginary, real and near-zero-time checks.
function spin_autocorrelation(exact, j, times, scale=1.0)
    psi, z = @view(exact.vectors[:, 1]), @view(exact.Z[:, j])
    weights = abs2.(exact.vectors' * (z .* psi))
    gaps = exact.energies .- exact.E0
    C = [sum(weights .* exp.(-scale .* gaps .* t)) for t in times]
    return (; C, magnetization=sum(z .* abs2.(psi)))
end

@testset "Optimized kernels: decay, singular prefixes and short strings" begin
    rng = Xoshiro(3209)
    # Independent Pfaffian oracle: expansion along the first row.
    function pfaffian_expansion(A)
        n = size(A, 1)
        n == 0 && return 1.0 + 0im
        return sum(2:n) do j
            keep = [k for k in 2:n if k != j]
            (-1)^j * A[1, j] * pfaffian_expansion(A[keep, keep])
        end
    end
    for n in (2, 4, 6, 8)
        X = randn(rng, ComplexF64, n, n)
        A = X - transpose(X)
        @test pfaffian!(copy(A)) ≈ pfaffian_expansion(A) rtol=1e-12
        D = randn(rng, ComplexF64, n, n)
        W = zeros(ComplexF64, 2n, 2n)
        W[1:2:2n, 2:2:2n] = D
        W[2:2:2n, 1:2:2n] = -transpose(D)
        @test pfaffian!(W) ≈ det(D) rtol=1e-12
    end
    tiny = zeros(ComplexF64, 4, 4)
    tiny[1, 2], tiny[2, 1] = 1e-310, -1e-310
    tiny[3, 4], tiny[4, 3] = 1e300, -1e300
    @test pfaffian!(tiny) ≈ 1e-10 rtol=1e-12

    # Test all prefixes against fresh pivoted LU, including a singular first
    # prefix followed by a nonsingular second one, and determinant underflow.
    singular_prefix = zeros(8, 8)
    singular_prefix[1, 3] = singular_prefix[2, 2] = 1
    small = zeros(32, 32)
    for i in 1:31
        small[i, i+1] = 1e-20
    end
    for G in (randn(rng, 16, 16), singular_prefix, small), boundary in (:open, :periodic)
        L = size(G, 1)
        rmax = boundary == :open ? L-1 : L÷2
        saved = copy(G)
        actual = correlations(G; boundary, rmax)
        @test G == saved
        for i in 1:L, r in 1:rmax
            if boundary == :open && i+r > L
                @test isnan(actual[i, r+1])
                continue
            end
            A = [((xor(i+a-1 > L, i+b > L)) ? -1 : 1) *
                G[mod1(i+a-1, L), mod1(i+b, L)] for a in 1:r, b in 1:r]
            logabs, phase = logabsdet(A)
            @test actual[i, r+1] ≈ phase * exp(logabs) rtol=1e-10 atol=1e-12
        end
    end
    @test correlations(small; boundary=:open, rmax=31)[1, end] == 0

    # Check raw strong-disorder correlations against independent determinants.
    for h0 in (1.5, 10.0), boundary in (:open, :periodic)
        J, h = sample_disorder(rng, 128, h0; boundary)
        G = ground_state(J, h; boundary).G
        pair = correlations(G; boundary)
        for i in (1, 33, 64), r in (16, 32, 64)
            logabs, phase = logabsdet(G[i:i+r-1, i+1:i+r])
            @test pair[i, r+1] ≈ phase * exp(logabs) atol=1e-10
        end
    end

    times = [13.2, 0.0, 0.7, 13.2]
    for h0 in (0.4, 1.0, 3.0)
        J, h = sample_disorder(rng, 32, h0; boundary=:open)
        Jcopy, hcopy = copy(J), copy(h)
        F = svd!(RandomTFIM.fermion_matrix(J, h, Val(:open)))
        for j in (1, 2, 3, 8, 16, 25, 30, 31, 32)
            actual = autocorrelation(J, h, times; j)
            @test actual isa Vector{Float64}
            # The determinant and JW-string algorithms contract different
            # matrices, including reflection at the right end of the chain.
            reference = RandomTFIM.string_autocorrelation(F, times, j, false)
            @test actual ≈ reference atol=2e-10
            @test actual[2] ≈ 1 atol=1e-11
        end
        @test J == Jcopy && h == hcopy
    end
    @test_throws ArgumentError autocorrelation(ones(7), ones(8), [floatmax(Float64)])
    @test_throws ArgumentError autocorrelation(ones(8), ones(8), [floatmax(Float64)]; boundary=:periodic)
end

@testset "Spatial C-only interface and ensemble logarithms" begin
    # Return raw Float64 values, including underflow and invalid open-chain pairs.
    G = zeros(6, 6)
    G[1:4, 2:5] = Matrix(Diagonal(fill(1e-100, 4)))
    pair = correlations(G; boundary=:open, rmax=4)
    @test pair isa Matrix{Float64}
    @test pair[1, 5] == 0
    @test isnan(pair[6, 2])

    # Small nonzero correlations remain unfiltered.
    G[1:4, 2:5] = Matrix(Diagonal([1.0, 1e-10, 1.0, 1.0]))
    @test correlations(G; boundary=:open, rmax=4)[1, 3] ≈ 1e-10

    # A singular prefix must not prevent recovery at the next distance.
    G .= 0
    G[1:2, 2:3] = [0.0 1.0; 1.0 0.0]
    pair = correlations(G; boundary=:open, rmax=2)
    @test pair[1, 2] == 0
    @test pair[1, 3] ≈ -1
    @test correlations(G; rmax=0) == ones(6, 1)
    @test_throws ArgumentError correlations(fill(NaN, 4, 4))

    values = [1.0, exp(-2), 0.0, -1.0, NaN, Inf]
    @test isequal(RandomTFIM.correlation_log.(values), [0.0, -2.0, NaN, NaN, NaN, NaN])
    # Negative real parts must propagate through temporal statistics, not be dropped.
    ensemble = disorder_ensemble(8, 1.8, [0.0, 1.7, 13.2]; nsamples=3, seed=92, keep_samples=true)
    expected = map(x -> x > 0 ? log(x) : NaN, ensemble.sample_real_Ct)
    @test any(isnan, expected)
    @test isequal(ensemble.real_logCt_mean, vec(mean(expected; dims=2)))
    @test isequal(ensemble.real_logCt_sem, vec(std(expected; dims=2)) / sqrt(3))
end

@testset "Both boundaries: Majorana/Pfaffian dynamics vs spin ED" begin
    rng = Xoshiro(942)
    times = [0.0, 0.13, 0.8, 3.1, 12.0, 40.0]
    for boundary in (:open, :periodic), L in (2, 4, 6), h0 in (0.4, 1.0, 3.0), random in (false, true)
        J, h = random ? sample_disorder(rng, L, h0) : (ones(L), fill(h0, L))
        bonds = boundary == :open ? J[1:end-1] : J
        exact = spin_oracle(bonds, h; periodic=(boundary == :periodic))
        for j in 1:L
            reference = spin_autocorrelation(exact, j, times)
            expected = reference.C
            @test abs(reference.magnetization) < 1e-10
            actual = @inferred autocorrelation(bonds, h, times; j, boundary)
            @test actual ≈ expected .- reference.magnetization^2 atol=2e-10
            @test actual[1] ≈ 1 atol=1e-12
            @test actual isa Vector{Float64}
            @test all(-1e-12 .<= actual .<= 1+1e-12)
            @test all(diff(actual) .<= 1e-12)
            # Relative accuracy is only asserted above the roundoff floor.
            resolved = expected .> 1e-8
            @test all(isapprox.(actual[resolved], expected[resolved]; rtol=1e-6, atol=0))
        end
    end
    h = collect(0.5:0.5:3.0)
    for j in 1:6
        @test autocorrelation(zeros(5), h, times; j) ≈ exp.(-2 .* h[j] .* times) rtol=1e-12 atol=0
    end
    empty_C = @inferred autocorrelation(ones(3), ones(4), Float64[])
    @test empty_C isa Vector{Float64}
    @test isempty(empty_C)
    @test_throws DimensionMismatch autocorrelation(ones(4), ones(4), times)
    @test_throws ArgumentError autocorrelation(ones(3), ones(4), times; j=0)
    for t in (NaN, -0.1, Inf)
        @test_throws ArgumentError autocorrelation(ones(3), ones(4), [t])
    end
    @test_throws ArgumentError autocorrelation(ones(3), zeros(4), times)
    @test_throws ArgumentError disorder_ensemble(4, 1.0, times; keep_samples=true, nsamples=0)
    @test pfaffian!(zeros(ComplexF64, 4, 4)) == 0
    # Requires a pivot swap; Pf(A)=a12*a34-a13*a24+a14*a23=-6.
    A = ComplexF64[0 0 2 0; 0 0 0 3; -2 0 0 0; 0 -3 0 0]
    @test pfaffian!(A) == -6
end

@testset "Real-time autocorrelation vs spin ED" begin
    times = [0.0, 0.13, 1.7, -1.7, 13.2, 0.13]
    for boundary in (:open, :periodic), L in (2, 6, 8)
        J, h = sample_disorder(Xoshiro(1942+L), L, 1.8; boundary)
        exact = spin_oracle(J, h; periodic=boundary == :periodic)
        for j in unique([1, L÷2, L])
            expected = spin_autocorrelation(exact, j, times, 1im).C
            actual = autocorrelation(J, h, times; boundary, j, time_domain=:real)
            @test actual isa Vector{Float64}
            @test actual ≈ real.(expected) atol=2e-10
            @test actual[1] ≈ 1 atol=1e-12
            @test actual[3] ≈ actual[4] atol=1e-12
        end
    end
    for boundary in (:open, :periodic), j in (1, 4, 8)
        h = collect(0.25:0.25:2.0)
        J = zeros(boundary == :open ? 7 : 8)
        @test autocorrelation(J, h, times; boundary, j, time_domain=:real) ≈
            cos.(2 .* h[j] .* times) atol=1e-12
    end
    times = abs.(times)
    for boundary in (:open, :periodic)
        result = disorder_ensemble(8, 1.8, times; keep_samples=true, boundary, nsamples=3, seed=92)
        @test result.sample_real_Ct isa Matrix{Float64}
        rng = Xoshiro(92)
        reference = Matrix{ComplexF64}(undef, length(times), 3)
        for n in 1:3
            J, h = sample_disorder(rng, 8, 1.8; boundary)
            exact = spin_oracle(J, h; periodic=boundary == :periodic)
            reference[:, n] = spin_autocorrelation(exact, 4, times, 1im).C
            @test result.sample_real_Ct[:, n] ≈ autocorrelation(J, h, times;
                boundary, time_domain=:real)
        end
        @test result.real_Ct_mean ≈ vec(mean(real.(reference); dims=2)) atol=2e-10
        @test result.real_Ct_sem ≈ vec(std(real.(reference); dims=2)) / sqrt(3) atol=2e-10
        empty_result = disorder_ensemble(8, 1.8, Float64[]; keep_samples=true, boundary, nsamples=1)
        @test size(empty_result.sample_real_Ct) == (0, 1)
        @test eltype(empty_result.sample_real_Ct) == Float64
    end
    @test autocorrelation(ones(3), ones(4), Float64[]; time_domain=:real) isa Vector{Float64}
    @test_throws ArgumentError autocorrelation(ones(3), ones(4), times; time_domain=:bad)
    @test_throws MethodError disorder_ensemble(4, 1.0)
    for t in (NaN, Inf, -Inf, floatmax(Float64))
        @test_throws ArgumentError autocorrelation(ones(3), ones(4), [t]; time_domain=:real)
    end
end

@testset "Imaginary-time tails: decoupled spins without cancellation" begin
    # Both endpoints and the midpoint use the condition-number fallback.
    # Pointwise relative checks detect loss of tiny tails hidden by array norms.
    times = [20.0, 100.0, 400.0]
    for boundary in (:open, :periodic), j in (1, 4)
        J = zeros(boundary == :open ? 7 : 8)
        actual = autocorrelation(J, ones(8), times; boundary, j)
        @test all(isapprox.(actual, exp.(-2 .* times); rtol=1e-12, atol=0))
        @test all(isfinite, actual)
    end
end

@testset "Condition-controlled determinant and boundary-specific Pfaffians" begin
    # Force each path as well as the adaptive path against independent spin ED.
    for boundary in (:open, :periodic), L in (2, 6), h0 in (0.4, 1.0, 3.0)
        J, h = sample_disorder(Xoshiro(480+L), L, h0; boundary)
        exact = spin_oracle(J, h; periodic=boundary == :periodic)
        for j in unique([1, L÷2, L]), time_domain in (:imaginary, :real)
            times = time_domain == :imaginary ? [0.0, 0.3, 1.7, 13.2] : [0.0, -1.7, 0.3, 1.7, 13.2]
            scale = time_domain == :imaginary ? 1.0 : 1.0im
            expected = real.(spin_autocorrelation(exact, j, times, scale).C)
            for rcond_tol in (0.0, sqrt(eps(Float64)), 1.0)
                actual = @inferred autocorrelation(J, h, times; j, boundary, time_domain, rcond_tol)
                @test actual ≈ expected atol=2e-10
            end
        end
    end

    # Critical PBC has a zero fermion mode; disconnected spins have exact
    # occupied modes. Neither permits assuming an empty-vacuum Thouless chart.
    for J in (ones(6), zeros(6)), time_domain in (:imaginary, :real)
        h, times = ones(6), [0.0, 0.4, 7.0]
        exact = spin_oracle(J, h)
        scale = time_domain == :imaginary ? 1.0 : 1.0im
        expected = real.(spin_autocorrelation(exact, 3, times, scale).C)
        @test autocorrelation(J, h, times; boundary=:periodic, j=3,
            time_domain, rcond_tol=1.0) ≈ expected atol=2e-10
    end

    # A coarse threshold makes both branches accessible in both time domains.
    # Time order is arbitrary: fallback cannot become a permanent time cutoff.
    for boundary in (:open, :periodic), time_domain in (:imaginary, :real)
        J, h = sample_disorder(Xoshiro(483), 6, 1.8; boundary)
        times = [13.2, 0.0, 1.7, 13.2]
        det = autocorrelation(J, h, times; boundary, time_domain, rcond_tol=0.0)
        pf = autocorrelation(J, h, times; boundary, time_domain, rcond_tol=1.0)
        adaptive = autocorrelation(J, h, times; boundary, time_domain, rcond_tol=0.99)
        @test adaptive[2] == det[2]
        @test adaptive[[1, 3, 4]] == pf[[1, 3, 4]]
    end

    # Regression: the old determinant reaches a +/- roundoff plateau, while
    # the selected JW fallback retains a decaying OBC tail.
    J, h = sample_disorder(Xoshiro(3), 16, 3.0; boundary=:open)
    F = svd!(RandomTFIM.fermion_matrix(J, h, Val(:open)))
    times = [100.0, 0.0, 10.0, 1000.0, 100.0]
    reference = RandomTFIM.string_autocorrelation(F, times, 8, false)
    actual = autocorrelation(J, h, times; j=8)
    @test all(isapprox.(actual, reference; rtol=1e-8, atol=0))
    @test all(actual .> 0)
    @test actual[[1, 4, 5]] == autocorrelation(J, h, times; j=8, rcond_tol=1.0)[[1, 4, 5]]

    # OBC uses the shorter reflected string on the right half. Compare both
    # sides with an unreflected string and the independent Gaussian overlap.
    for j in (2, 7, 9, 15), scale in (1.0, 1.0im)
        times = scale isa Real ? [100.0, 0.0, 0.3, 10.0, 100.0] : [13.2, 0.0, -1.7, 1.7, 13.2]
        actual = RandomTFIM.autocorrelation(F, F, times, j, scale; rcond_tol=1.0)
        reference = RandomTFIM.string_autocorrelation(F, times, j, false, scale)
        @test actual ≈ reference atol=1e-12
        G = F.U * F.V'
        G[1:j, :] .*= -1
        G[:, 1:j-1] .*= -1
        chart = RandomTFIM.pfaffian_chart(F.U' * G * F.V, typeof(scale))
        overlap = [RandomTFIM.gaussian_pfaffian!(chart, exp.(-2scale .* F.S .* t), zero(scale)) for t in times]
        @test actual ≈ overlap atol=1e-12
        adaptive = RandomTFIM.autocorrelation(F, F, times, j, scale; rcond_tol=0.99)
        @test adaptive[[1, 5]] == actual[[1, 5]]
    end

    # Independent two-spin parity blocks avoid forbidden-parity ED roundoff
    # contaminating the reference at exponentially small imaginary-time tails.
    for boundary in (:open, :periodic)
        J = boundary == :open ? [0.3] : [0.1, 0.2]
        h = [0.8, 1.1]
        bond = sum(J)
        even = eigen(Symmetric([-bond -sum(h); -sum(h) bond]))
        odd = eigen(Symmetric([-bond h[1]-h[2]; h[1]-h[2] bond]))
        weights = abs2.(odd.vectors' * even.vectors[:, 1])
        gaps = odd.values .- even.values[1]
        times = [0.0, 20.0, 100.0, 300.0]
        reference = [sum(weights .* exp.(-gaps .* t)) for t in times]
        for rcond_tol in (sqrt(eps(Float64)), 1.0)
            actual = autocorrelation(J, h, times; boundary, j=1, rcond_tol)
            @test all(isapprox.(actual, reference; rtol=1e-9, atol=0))
        end
    end

    # Ensemble forwarding must use the same threshold for both time domains.
    for boundary in (:open, :periodic), rcond_tol in (0.0, 1.0)
        times = [0.0, 0.7, 100.0]
        result = disorder_ensemble(6, 3.0, times; boundary, rcond_tol,
            nsamples=2, seed=49, rmax=0, keep_samples=true)
        rng = Xoshiro(49)
        for n in 1:2
            J, h = sample_disorder(rng, 6, 3.0; boundary)
            @test result.sample_Ct[:, n] == autocorrelation(J, h, times; boundary, rcond_tol)
            @test result.sample_real_Ct[:, n] == autocorrelation(J, h, times;
                boundary, rcond_tol, time_domain=:real)
        end
    end
    for tolerance in (-1.0, 1.01, NaN, Inf)
        @test_throws ArgumentError autocorrelation(ones(3), ones(4), Float64[]; rcond_tol=tolerance)
        @test_throws ArgumentError disorder_ensemble(4, 1.0, Float64[]; rcond_tol=tolerance)
    end
end

@testset "Time-loop optimizations: zero and near-zero times" begin
    # Both endpoint and midpoint calculations must retain phase
    # and diagonal decay tails when the off-diagonal row factor is tiny.
    times = [0.0, 1e-14, 1e-8, 0.3, 0.0]
    for boundary in (:open, :periodic)
        J, h = sample_disorder(Xoshiro(9428), 8, 2.0; boundary)
        exact = spin_oracle(J, h; periodic=boundary == :periodic)
        for (time_domain, scale) in ((:imaginary, 1.0), (:real, 1im)), j in (1, 4, 8)
            expected = spin_autocorrelation(exact, j, times, scale).C
            actual = autocorrelation(J, h, times; boundary, j, time_domain)
            @test actual ≈ real.(expected) atol=2e-12
            @test actual[[1, end]] ≈ ones(2) atol=2e-12
        end
    end
end

@testset "Free fermions vs spin ED" begin
    rng = Xoshiro(734)
    for L in (2, 4, 6, 8), h0 in (0.4, 1.0, 3.0), random in (false, true)
        J, h = random ? sample_disorder(rng, L, h0) : (ones(L), fill(h0, L))
        exact = spin_oracle(J, h)
        result = @inferred ground_state(J, h)
        gap = @inferred energy_gap(J, h)
        @test result.gap ≈ exact.E1-exact.E0 atol=1e-11
        @test gap.gap ≈ result.gap atol=1e-11
        @test result.G * result.G' ≈ I atol=1e-12
        pair = @inferred correlations(result.G)
        for r in 0:L÷2, i in 1:L
            @test pair[i, r+1] ≈ exact.C[i, mod1(i+r, L)] atol=2e-10
        end
        @test pair isa Matrix{Float64}
    end
end

@testset "Open-chain gaps, spatial pairs and disorder means" begin
    rng = Xoshiro(934)
    for L in (2, 4, 6, 8), h0 in (0.4, 1.0, 3.0)
        J, h = sample_disorder(rng, L, h0; boundary=:open)
        exact = spin_oracle(J, h; periodic=false)
        state = @inferred ground_state(J, h; boundary=:open)
        @test state.gap ≈ exact.E1-exact.E0 atol=1e-11
        @test energy_gap(J, h; boundary=:open).gap ≈ state.gap atol=1e-12
        pair = correlations(state.G; boundary=:open, rmax=L-1)
        for r in 0:L-1, i in 1:L
            if i+r <= L
                @test pair[i, r+1] ≈ exact.C[i, i+r] atol=2e-10
            else
                @test isnan(pair[i, r+1])
            end
        end
        # Cutting the periodic seam must recover OBC, including time evolution.
        periodic_J = vcat(J, 0.0)
        @test energy_gap(periodic_J, h).gap ≈ state.gap atol=1e-11
        times = [17.3, 0.0, 2.1, 17.3, 0.14]
        @test autocorrelation(periodic_J, h, times; boundary=:periodic) ≈
            autocorrelation(J, h, times; boundary=:open) atol=2e-10
    end
    for boundary in (:open, :periodic)
        @test autocorrelation(zeros(boundary == :open ? 5 : 6), ones(6), [0.0, 1.2]; boundary) ≈ exp.(-2 .* [0.0, 1.2])
    end
    @test all(disorder_ensemble(6, 3.0, Float64[]; keep_samples=true, nsamples=3, boundary=:open, rmax=0).resolved)
    Jp, hp = sample_disorder(Xoshiro(8), 6, 1.0)
    Jo, ho = sample_disorder(Xoshiro(8), 6, 1.0; boundary=:open)
    @test Jo == Jp[1:end-1] && ho == hp
    @test_throws ArgumentError sample_disorder(Xoshiro(1), 4, 1.0; boundary=:bad)
    @test_throws ArgumentError autocorrelation(ones(4), ones(4), [0.0]; boundary=:bad)
    @test_throws ArgumentError correlations(zeros(4, 4); boundary=:bad)
    @test_throws ArgumentError correlations(zeros(4, 4); boundary=:periodic, rmax=3)
    @test_throws DimensionMismatch energy_gap(ones(4), ones(4); boundary=:open)
end

@testset "Unified ensemble uses the same realization for all observables" begin
    times = [0.0, 0.3, 0.7]
    nsamples = 17 # More samples than test threads; check serial RNG ordering.
    for boundary in (:open, :periodic)
        result = @inferred NamedTuple disorder_ensemble(6, 1.0, times; keep_samples=true, boundary,
            nsamples, seed=83, j=4, rmax=2)
        @test result.sample_Ct isa Matrix{Float64}
        expected_logC = zeros(3, nsamples)
        rng = Xoshiro(83)
        for n in 1:nsamples
            J, h = sample_disorder(rng, 6, 1.0; boundary)
            state = ground_state(J, h; boundary)
            pair = correlations(state.G; boundary, rmax=2)
            @test result.gaps[n] ≈ state.gap
            @test result.resolved[n] == state.resolved
            for r in 0:2
                origins = 1:(boundary == :open ? 6-r : 6)
                @test result.sample_C[r+1, n] ≈ mean(pair[origins, r+1])
                expected_logC[r+1, n] = mean(log.(pair[origins, r+1]))
            end
            @test result.sample_Ct[:, n] ≈ autocorrelation(J, h, times; boundary, j=4)
        end
        @test result.logC_mean ≈ vec(mean(expected_logC; dims=2))
        @test result.logC_sem ≈ vec(std(expected_logC; dims=2)) / sqrt(nsamples)
        without_space = disorder_ensemble(6, 1.0, times; keep_samples=true, boundary, nsamples, seed=83, j=4, rmax=0)
        @test without_space.gaps ≈ result.gaps
        @test without_space.sample_Ct ≈ result.sample_Ct
        without_time = disorder_ensemble(6, 1.0, Float64[]; keep_samples=true, boundary, nsamples, seed=83, rmax=2)
        @test without_time.sample_C ≈ result.sample_C
        @test size(without_time.sample_Ct) == (0, nsamples)
        @test without_time.sample_Ct isa Matrix{Float64}

        # Include tiny gaps near the resolution threshold (periodic sample 105).
        # All three APIs must use the same SVD path and retain serial order.
        gap_only = disorder_ensemble(32, 0.4, Float64[]; keep_samples=true, boundary, nsamples=129, seed=84, rmax=0)
        @test any(!, gap_only.resolved)
        @test isnan(gap_only.log_gap_mean) && isnan(gap_only.log_gap_sem)
        rng = Xoshiro(84)
        for n in 1:129
            J, h = sample_disorder(rng, 32, 0.4; boundary)
            expected = energy_gap(J, h; boundary)
            state = ground_state(J, h; boundary)
            @test gap_only.gaps[n] == expected.gap == state.gap
            @test gap_only.resolved[n] == expected.resolved
            @test state.resolved == expected.resolved
        end
    end
    for t in (NaN, -0.1, Inf)
        @test_throws ArgumentError disorder_ensemble(6, 1.0, [t])
    end
    @test_throws ArgumentError disorder_ensemble(6, 1.0, times; keep_samples=true, j=0)
end

@testset "Limits and sampling" begin
    J, h = zeros(6), collect(0.5:0.5:3.0)
    state = ground_state(J, h)
    @test state.gap ≈ 2minimum(h)
    @test correlations(state.G)[:, 2:end] == zeros(6, 3)
    @test !energy_gap(ones(32), fill(0.1, 32)).resolved
    a = @inferred NamedTuple disorder_ensemble(8, 1.0, Float64[]; keep_samples=true, nsamples=5)
    b = disorder_ensemble(8, 1.0, Float64[]; keep_samples=true, nsamples=5)
    @test isequal(a, b)
    @test all(exp.(a.logC_mean) .<= a.C_mean .+ 1e-14)
    @test all(disorder_ensemble(4, 3.0, Float64[]; keep_samples=true, nsamples=3, rmax=0).resolved)
    @test_throws ArgumentError energy_gap(ones(3), ones(3))
    @test_throws ArgumentError energy_gap(ones(4), zeros(4))
end
