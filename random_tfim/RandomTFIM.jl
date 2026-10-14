module RandomTFIM

using LinearAlgebra
using Random
using Statistics
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt, libblastrampoline

export sample_disorder, energy_gap, ground_state, correlations, disorder_ensemble
export autocorrelation
export entanglement_entropy

# Implementation files share the RandomTFIM namespace.
include("model.jl")
include("correlations.jl")
include("entanglement.jl")
include("autocorrelation.jl")
include("pairing_autocorrelation.jl")
include("ensemble.jl")

end
