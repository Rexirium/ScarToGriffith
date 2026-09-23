using Test, HDF5, LinearAlgebra
include(joinpath(@__DIR__, "..", "run.jl"))

# Exercise the actual driver, including metadata and overwrite protection.
mktempdir() do directory
    output = joinpath(directory, "smoke.h5")
    main(["correlation", "2", output])
    h5open(output, "r") do file
        @test read(attributes(file)["complete"])
        @test length(read(file["L128/h1.0/gaps"])) == 2
        @test size(read(file["L128/h3.0/pair_logC"])) == (128, 65, 2)
        @test all(isfinite, read(file["L128/h1.0/average"]))
        group = file["L16/h1.0"]
        sample_C = read(group["sample_C"])
        sample_logC = read(group["sample_logC"])
        @test read(group["average"]) ≈ vec(mean(sample_C; dims=2))
        @test read(group["typical"]) ≈ exp.(vec(mean(sample_logC; dims=2)))
        @test read(group["sem"]) ≈ vec(std(sample_C; dims=2)) / sqrt(2)
        gaps, resolved = read(group["gaps"]), read(group["resolved"])
        @test isequal(read(group["loggaps"]),
            map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved))
    end
    @test_throws ArgumentError main(["demo", "2", output])
end
