using Revise
using LinearAlgebra
using ExactDiagonalize
# For plotting
using CairoMakie

let
    set_systype(:Spin)
    Ls = 12
    g = -0.3

    basis = SpinBasis(Ls)

    initvec = product_state(basis, n -> isodd(n) ? :Up : :Dn)

    opsum = OpSum(Float64)
    for j in 1 : Ls
        jpprev = mod1(j - 2, Ls)
        jprev = mod1(j - 1, Ls)
        jpost = mod1(j + 1, Ls)
        jppost = mod1(j + 2, Ls)

        opsum += 1.0, :Pdn, jprev, :X, j, :Pdn, jpost
        opsum += g*(-1)^j, :Z, j
        #opsum += g, :X, j, :X, jpost
        #opsum += -g, :iY, j, :iY, jpost
    end

    eigenergies, eigstates = spectrum(opsum, basis; retvecs=true)

    entropies = zeros(basis.dim)

    for n in 1 : basis.dim
        entropy= ent_entropy(basis, eigstates[:, n], Ls ÷ 2)
        entropies[n] = entropy
        
    end

    
    overlaps = abs2.(transpose(eigstates) * initvec)
    marksizes = [overlaps[n] > 1e-2 ? 12 : 5 for n in 1 : basis.dim]

    set_theme!(Axis = (
        titlesize = 20, 
        titlefont = "Times New Roman",  
        xtickalign = 1, xgridvisible=false, 
        ytickalign = 1, ygridvisible=false, 
        xlabelsize = 18, 
        ylabelsize = 18
    ))

    fig = Figure()
    ax = Axis(fig[1, 1], 
        title="L = $(Ls), Z₂ gz = $g", 
        yscale=identity, 
        xlabel=L"E_n", ylabel=L"S(L/2)",  
        # limits=(nothing, (0.0, 5))
    )

    scatter!(ax, eigenergies, entropies, 
        color = overlaps, colormap=:plasma, 
        markersize=marksizes
    )
    fig
    #save("manybodyscars/pxp_unconstrained_z2_L=$(Ls)_g=$(g).png", fig)
    
end