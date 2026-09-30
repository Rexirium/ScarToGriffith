using Test

# Isolate modules and environments; report every suite even if one fails.
mode = isempty(ARGS) ? "all" : only(ARGS)
mode in ("all", "core") || error("Usage: runtests.jl [all|core]")
suites = mode == "core" ? ["physics.jl"] : ["physics.jl", "io.jl", "plot_io.jl"]
@testset "RandomTFIM" begin
    for suite in suites
        @testset "$suite" begin
            project = suite == "plot_io.jl" ? ["--project=@v1.13"] : String[]
            cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) $project $(joinpath(@__DIR__, suite))`
            @test success(pipeline(cmd; stdout, stderr))
        end
    end
end
