# Test-only scalar wrapper for the logarithmic Pfaffian kernel.
function pfaffian!(A::Matrix{ComplexF64})
    logabs, phase = RandomTFIM.logabspfaffian!(A)
    return phase * exp(logabs)
end

