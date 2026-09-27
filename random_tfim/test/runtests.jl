using Test, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM
BLAS.set_num_threads(1)

@testset "Ensemble summary and optional samples" begin
    for boundary in (:open, :periodic), time_domain in (:imaginary, :real)
        kwargs = (; nsamples=3, seed=71, boundary, time_domain, times=[0.0, 0.4])
        summary = disorder_ensemble(6, 1.0; kwargs...)
        samples = disorder_ensemble(6, 1.0; kwargs..., keep_samples=true)
        @test summary.gaps == samples.gaps
        @test summary.resolved == samples.resolved
        for (raw, avg, sem) in ((:sample_C, :C_mean, :C_sem),
                (:sample_logC, :logC_mean, :logC_sem), (:sample_Ct, :Ct_mean, :Ct_sem))
            @test !hasproperty(summary, raw)
            values = getproperty(samples, raw)
            @test getproperty(summary, avg) ≈ vec(mean(values; dims=2))
            @test getproperty(summary, sem) ≈ vec(std(values; dims=2)) / sqrt(3)
            @test getproperty(summary, avg) == getproperty(samples, avg)
            @test getproperty(summary, sem) == getproperty(samples, sem)
        end
        @test summary.Ct_sem isa Vector{Float64}
        @test size(samples.sample_Ct) == (length(kwargs.times), kwargs.nsamples)
        single = disorder_ensemble(6, 1.0; boundary, time_domain, nsamples=1, times=[0.4])
        @test all(isnan, single.C_sem)
        @test all(isnan, single.logC_sem)
        @test all(isnan, single.Ct_sem)
        empty_time = disorder_ensemble(6, 1.0; boundary, time_domain, nsamples=3, rmax=0)
        @test isempty(empty_time.Ct_mean) && isempty(empty_time.Ct_sem)
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
        @test RandomTFIM.pfaffian!(copy(A)) ≈ pfaffian_expansion(A) rtol=1e-12
        D = randn(rng, ComplexF64, n, n)
        W = zeros(ComplexF64, 2n, 2n)
        W[1:2:2n, 2:2:2n] = D
        W[2:2:2n, 1:2:2n] = -transpose(D)
        @test RandomTFIM.pfaffian!(W) ≈ det(D) rtol=1e-12
    end
    tiny = zeros(ComplexF64, 4, 4)
    tiny[1, 2], tiny[2, 1] = 1e-310, -1e-310
    tiny[3, 4], tiny[4, 3] = 1e300, -1e300
    @test RandomTFIM.pfaffian!(tiny) ≈ 1e-10 rtol=1e-12

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
                @test isnan(actual.C[i, r+1]) && isnan(actual.logC[i, r+1])
                continue
            end
            A = [((xor(i+a-1 > L, i+b > L)) ? -1 : 1) *
                G[mod1(i+a-1, L), mod1(i+b, L)] for a in 1:r, b in 1:r]
            logabs, phase = logabsdet(A)
            @test actual.C[i, r+1] ≈ phase * exp(logabs) rtol=1e-10 atol=1e-12
            expected_log = actual.resolved[i, r+1] && phase > 0 ? logabs : NaN
            @test isequal(actual.logC[i, r+1], expected_log) ||
                isapprox(actual.logC[i, r+1], expected_log; atol=1e-10)
        end
    end
    @test correlations(small; boundary=:open, rmax=31).logC[1, end] ≈ 31log(1e-20)

    # Strong disorder must flag unresolved tails rather than refactor them.
    for h0 in (1.5, 10.0), boundary in (:open, :periodic)
        J, h = sample_disorder(rng, 128, h0; boundary)
        G = ground_state(J, h; boundary).G
        pair = correlations(G; boundary)
        for i in (1, 33, 64), r in (16, 32, 64)
            logabs, phase = logabsdet(G[i:i+r-1, i+1:i+r])
            expected = pair.resolved[i, r+1] && phase > 0 ? logabs : NaN
            @test isequal(pair.logC[i, r+1], expected) ||
                isapprox(pair.logC[i, r+1], expected; atol=1e-6)
        end
    end

    times = [13.2, 0.0, 0.7, 13.2]
    for h0 in (0.4, 1.0, 3.0)
        J, h = sample_disorder(rng, 32, h0; boundary=:open)
        Jcopy, hcopy = copy(J), copy(h)
        F = svd!(RandomTFIM.fermion_matrix(J, h, Val(:open)))
        for j in (1, 2, 3, 8, 16, 25, 30, 31, 32)
            actual = autocorrelation(J, h, times; j).C
            @test actual isa Vector{Float64}
            # The determinant and JW-string algorithms contract different
            # matrices, including reflection at the right end of the chain.
            reference = RandomTFIM.string_autocorrelation(F, times, j, false).C
            @test actual ≈ reference atol=2e-10
            @test actual[2] ≈ 1 atol=1e-11
        end
        @test J == Jcopy && h == hcopy
    end
    @test_throws ArgumentError autocorrelation(ones(7), ones(8), [floatmax(Float64)])
    @test_throws ArgumentError autocorrelation(ones(8), ones(8), [floatmax(Float64)]; boundary=:periodic)
end

@testset "Spatial QR condition diagnostics" begin
    # A tiny determinant need not be ill-conditioned: uniform scaling keeps
    # its logarithm meaningful even when exp(logC) underflows.
    G = zeros(6, 6)
    G[1:4, 2:5] = Matrix(Diagonal(fill(1e-100, 4)))
    pair = correlations(G; boundary=:open, rmax=4)
    @test pair.resolved[1, 5]
    @test pair.C[1, 5] == 0
    @test pair.logC[1, 5] ≈ 4log(1e-100)
    @test !pair.resolved[6, 2] && isnan(pair.logC[6, 2])

    # The same small pivot is acceptable or unresolved according to logtol.
    G[1:4, 2:5] = Matrix(Diagonal([1.0, 1e-10, 1.0, 1.0]))
    strict = correlations(G; boundary=:open, rmax=4)
    loose = correlations(G; boundary=:open, rmax=4, logtol=0.1)
    @test !strict.resolved[1, 3] && isnan(strict.logC[1, 3])
    @test loose.resolved[1, 3] && loose.logC[1, 3] ≈ log(1e-10)
    @test isequal(strict.C, loose.C)

    # Off-diagonal entries matter: reading the wrong triangle would accept
    # this prefix at 8e-4, whereas its lower-triangular estimate rejects it.
    G .= 0
    G[1:2, 2:3] = [1.0 1.0; 0.0 1e-10]
    @test !correlations(G; boundary=:open, rmax=2, logtol=8e-4).resolved[1, 3]

    # A singular prefix must not prevent recovery at the next distance.
    G .= 0
    G[1:2, 2:3] = [0.0 1.0; 1.0 0.0]
    pair = correlations(G; boundary=:open, rmax=2)
    @test !pair.resolved[1, 2] && isnan(pair.logC[1, 2])
    @test pair.resolved[1, 3] && pair.C[1, 3] ≈ -1
    @test isnan(pair.logC[1, 3]) # Preserve sign; do not turn negative C positive.
    @test all(correlations(G; rmax=0).resolved)
    for tol in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError correlations(G; logtol=tol)
        @test_throws ArgumentError disorder_ensemble(6, 1.0; logtol=tol)
    end
    @test_throws ArgumentError correlations(fill(NaN, 4, 4))
    # Unresolved logs propagate through averaging; no sample filtering.
    ensemble = disorder_ensemble(16, 5.0; nsamples=2, logtol=1e-18, keep_samples=true)
    @test any(isnan, ensemble.sample_logC)
    @test isequal(ensemble.logC_mean, vec(mean(ensemble.sample_logC; dims=2)))
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
            @test actual.C ≈ expected .- reference.magnetization^2 atol=2e-10
            @test actual.C[1] ≈ 1 atol=1e-12
            @test actual.C isa Vector{Float64}
            @test all(-1e-12 .<= actual.C .<= 1+1e-12)
            @test all(diff(actual.C) .<= 1e-12)
            # Relative accuracy is only asserted above the roundoff floor.
            resolved = expected .> 1e-8
            @test all(isapprox.(actual.C[resolved], expected[resolved]; rtol=1e-6, atol=0))
        end
    end
    h = collect(0.5:0.5:3.0)
    for j in 1:6
        @test autocorrelation(zeros(5), h, times; j).C ≈ exp.(-2 .* h[j] .* times) rtol=1e-12 atol=0
    end
    empty_C = @inferred autocorrelation(ones(3), ones(4), Float64[])
    @test empty_C.C isa Vector{Float64}
    @test isempty(empty_C.C)
    @test_throws DimensionMismatch autocorrelation(ones(4), ones(4), times)
    @test_throws ArgumentError autocorrelation(ones(3), ones(4), times; j=0)
    for t in (NaN, -0.1, Inf)
        @test_throws ArgumentError autocorrelation(ones(3), ones(4), [t])
    end
    @test_throws ArgumentError autocorrelation(ones(3), zeros(4), times)
    @test_throws ArgumentError disorder_ensemble(4, 1.0; keep_samples=true, times, nsamples=0)
    @test RandomTFIM.pfaffian!(zeros(ComplexF64, 4, 4)) == 0
    # Requires a pivot swap; Pf(A)=a12*a34-a13*a24+a14*a23=-6.
    A = ComplexF64[0 0 2 0; 0 0 0 3; -2 0 0 0; 0 -3 0 0]
    @test RandomTFIM.pfaffian!(A) == -6
end

@testset "Real-time autocorrelation vs spin ED" begin
    times = [0.0, 0.13, 1.7, -1.7, 13.2, 0.13]
    for boundary in (:open, :periodic), L in (2, 6, 8)
        J, h = sample_disorder(Xoshiro(1942+L), L, 1.8; boundary)
        exact = spin_oracle(J, h; periodic=boundary == :periodic)
        for j in unique([1, L÷2, L])
            expected = spin_autocorrelation(exact, j, times, 1im).C
            actual = autocorrelation(J, h, times; boundary, j, time_domain=:real).C
            @test actual isa Vector{ComplexF64}
            @test actual ≈ expected atol=2e-10
            @test actual[1] ≈ 1 atol=1e-12
            @test actual[3] ≈ conj(actual[4]) atol=1e-12
        end
    end
    for boundary in (:open, :periodic), j in (1, 4, 8)
        h = collect(0.25:0.25:2.0)
        J = zeros(boundary == :open ? 7 : 8)
        @test autocorrelation(J, h, times; boundary, j, time_domain=:real).C ≈
            exp.(-2im .* h[j] .* times) atol=1e-12
    end
    for boundary in (:open, :periodic)
        result = disorder_ensemble(8, 1.8; keep_samples=true, times, boundary, nsamples=3, seed=92,
            time_domain=:real)
        @test result.sample_Ct isa Matrix{ComplexF64}
        rng = Xoshiro(92)
        for n in 1:3
            J, h = sample_disorder(rng, 8, 1.8; boundary)
            @test result.sample_Ct[:, n] ≈ autocorrelation(J, h, times;
                boundary, time_domain=:real).C
        end
        empty_result = disorder_ensemble(8, 1.8; keep_samples=true, boundary, nsamples=1, time_domain=:real)
        @test size(empty_result.sample_Ct) == (0, 1)
        @test eltype(empty_result.sample_Ct) == ComplexF64
    end
    @test autocorrelation(ones(3), ones(4), Float64[]; time_domain=:real).C isa Vector{ComplexF64}
    @test_throws ArgumentError autocorrelation(ones(3), ones(4), times; time_domain=:bad)
    @test_throws ArgumentError disorder_ensemble(4, 1.0; time_domain=:bad)
    for t in (NaN, Inf, -Inf, floatmax(Float64))
        @test_throws ArgumentError autocorrelation(ones(3), ones(4), [t]; time_domain=:real)
    end
end

@testset "Imaginary-time tails: decoupled spins without cancellation" begin
    # L=8,j=1 exercises the short-string branch; j=4 uses the determinant.
    # Pointwise relative checks detect loss of tiny tails hidden by array norms.
    times = [20.0, 100.0, 400.0]
    for boundary in (:open, :periodic), j in (1, 4)
        J = zeros(boundary == :open ? 7 : 8)
        actual = autocorrelation(J, ones(8), times; boundary, j).C
        @test all(isapprox.(actual, exp.(-2 .* times); rtol=1e-12, atol=0))
        @test all(isfinite, actual)
    end
end

@testset "Time-loop optimizations: zero and near-zero times" begin
    # Both short-string endpoints and determinant paths must retain phase
    # and diagonal decay tails when the off-diagonal row factor is tiny.
    times = [0.0, 1e-14, 1e-8, 0.3, 0.0]
    for boundary in (:open, :periodic)
        J, h = sample_disorder(Xoshiro(9428), 8, 2.0; boundary)
        exact = spin_oracle(J, h; periodic=boundary == :periodic)
        for (time_domain, scale) in ((:imaginary, 1.0), (:real, 1im)), j in (1, 4, 8)
            expected = spin_autocorrelation(exact, j, times, scale).C
            actual = autocorrelation(J, h, times; boundary, j, time_domain).C
            @test actual ≈ expected atol=2e-12
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
            @test pair.C[i, r+1] ≈ exact.C[i, mod1(i+r, L)] atol=2e-10
        end
        @test exp.(pair.logC) ≈ pair.C atol=1e-12
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
                @test pair.C[i, r+1] ≈ exact.C[i, i+r] atol=2e-10
                @test exp(pair.logC[i, r+1]) ≈ pair.C[i, r+1] atol=1e-12
            else
                @test isnan(pair.C[i, r+1]) && isnan(pair.logC[i, r+1])
            end
        end
        # Cutting the periodic seam must recover OBC, including time evolution.
        periodic_J = vcat(J, 0.0)
        @test energy_gap(periodic_J, h).gap ≈ state.gap atol=1e-11
        times = [17.3, 0.0, 2.1, 17.3, 0.14]
        @test autocorrelation(periodic_J, h, times; boundary=:periodic).C ≈
            autocorrelation(J, h, times; boundary=:open).C atol=2e-10
    end
    for boundary in (:open, :periodic)
        @test autocorrelation(zeros(boundary == :open ? 5 : 6), ones(6), [0.0, 1.2]; boundary).C ≈ exp.(-2 .* [0.0, 1.2])
    end
    a = disorder_ensemble(6, 1.0; keep_samples=true, nsamples=3, boundary=:open, rmax=5, keep_pairs=true)
    for r in 0:5
        @test vec(mean(a.pair_logC[1:6-r, r+1, :]; dims=1)) ≈ a.sample_logC[r+1, :]
    end
    @test a.sample_C ≈ disorder_ensemble(6, 1.0; keep_samples=true, nsamples=3, boundary=:open, rmax=5).sample_C
    @test all(disorder_ensemble(6, 3.0; keep_samples=true, nsamples=3, boundary=:open, rmax=0).resolved)
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
        result = @inferred NamedTuple disorder_ensemble(6, 1.0; keep_samples=true, times, boundary,
            nsamples, seed=83, j=4, rmax=2, keep_pairs=true)
        @test result.sample_Ct isa Matrix{Float64}
        rng = Xoshiro(83)
        for n in 1:nsamples
            J, h = sample_disorder(rng, 6, 1.0; boundary)
            state = ground_state(J, h; boundary)
            pair = correlations(state.G; boundary, rmax=2)
            @test result.gaps[n] ≈ state.gap
            @test result.resolved[n] == state.resolved
            @test isequal(result.pair_logC[:, :, n], pair.logC)
            for r in 0:2
                origins = 1:(boundary == :open ? 6-r : 6)
                @test result.sample_C[r+1, n] ≈ mean(pair.C[origins, r+1])
                @test result.sample_logC[r+1, n] ≈ mean(pair.logC[origins, r+1])
            end
            @test result.sample_Ct[:, n] ≈ autocorrelation(J, h, times; boundary, j=4).C
        end
        without_space = disorder_ensemble(6, 1.0; keep_samples=true, times, boundary, nsamples, seed=83, j=4, rmax=0)
        @test without_space.gaps ≈ result.gaps
        @test without_space.sample_Ct ≈ result.sample_Ct
        without_time = disorder_ensemble(6, 1.0; keep_samples=true, boundary, nsamples, seed=83, rmax=2)
        @test without_time.sample_C ≈ result.sample_C
        @test size(without_time.sample_Ct) == (0, nsamples)
        @test without_time.sample_Ct isa Matrix{Float64}

        # Gap-only results also retain serial order, including resolution flags.
        gap_only = disorder_ensemble(32, 0.4; keep_samples=true, boundary, nsamples=129, seed=84, rmax=0)
        rng = Xoshiro(84)
        for n in 1:129
            J, h = sample_disorder(rng, 32, 0.4; boundary)
            expected = energy_gap(J, h; boundary)
            @test gap_only.gaps[n] == expected.gap
            @test gap_only.resolved[n] == expected.resolved
        end
    end
    for t in (NaN, -0.1, Inf)
        @test_throws ArgumentError disorder_ensemble(6, 1.0; times=[t])
    end
    @test_throws ArgumentError disorder_ensemble(6, 1.0; keep_samples=true, times, j=0)
end

@testset "Limits and sampling" begin
    J, h = zeros(6), collect(0.5:0.5:3.0)
    state = ground_state(J, h)
    @test state.gap ≈ 2minimum(h)
    @test correlations(state.G).C[:, 2:end] == zeros(6, 3)
    @test !energy_gap(ones(32), fill(0.1, 32)).resolved
    a = @inferred NamedTuple disorder_ensemble(8, 1.0; keep_samples=true, nsamples=5)
    b = disorder_ensemble(8, 1.0; keep_samples=true, nsamples=5)
    @test isequal(a, b)
    @test all(exp.(mean(a.sample_logC; dims=2)) .<= mean(a.sample_C; dims=2) .+ 1e-14)
    pairs = disorder_ensemble(4, 1.0; keep_samples=true, nsamples=3, keep_pairs=true)
    @test size(pairs.pair_logC) == (4, 3, 3)
    @test dropdims(mean(pairs.pair_logC; dims=1); dims=1) ≈ pairs.sample_logC
    @test all(disorder_ensemble(4, 3.0; keep_samples=true, nsamples=3, rmax=0).resolved)
    @test_throws ArgumentError energy_gap(ones(3), ones(3))
    @test_throws ArgumentError energy_gap(ones(4), zeros(4))
end
