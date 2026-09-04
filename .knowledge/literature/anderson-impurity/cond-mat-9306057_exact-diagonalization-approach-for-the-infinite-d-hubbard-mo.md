---
source: "https://arxiv.org/abs/cond-mat/9306057"
type: "arxiv"
canonical_id: "cond-mat/9306057"
title: "Exact Diagonalization Approach for the infinite D Hubbard Model"
authors: "M. Caffarel, W. Krauth"
year: "1993"
venue: "arXiv: Condensed Matter"
arxiv_id: "cond-mat/9306057"
full_text: yes
---

# Exact Diagonalization Approach for the infinite D Hubbard Model

**Authors:** M. Caffarel, W. Krauth

**Citation:** arXiv: Condensed Matter, 1993

**arXiv:** [cond-mat/9306057](https://arxiv.org/abs/cond-mat/9306057)

## Abstract

We present a powerful method for calculating the thermodynamic properties of the Hubbard model in infinite dimensions, using an exact diagonalization of an Anderson model with a finite number of sites. At finite temperatures, the explicit diagonalization of the Anderson Hamiltonian allows the calculation of Green's functions with a resolution far superior to that of Quantum Monte Carlo calculations. At zero temperature, the Lancz\`os method is used and yields the essentially exact zero-temperature solution of the model, except in a region of very small frequencies. Numerical results for the half-filled case in the paramagnetic phase (quasi-particle weight, self-energy, and also real-frequency spectral densities) are presented.

## Full Text

Exact Diagonalization Approach for the D = ∞ Hubbard Model
                                                             Michel Caffarel∗,+ and Werner Krauth∗∗
                                                            ∗ CNRS-Laboratoire de Physique Quantique1

                                                                IRSAMC, Université Paul Sabatier
                                                     118, route de Narbonne; F-31062 Toulouse Cedex; France
                                                                   e-mail: mc@tolosa.ups-tlse.fr
                                                      ∗∗ CNRS-Laboratoire de Physique Statistique de l’ENS

                                                         24, rue Lhomond; 75231 Paris Cedex 05; France
arXiv:cond-mat/9306057v1 28 Jun 1993




                                                                  e-mail: krauth@physique.ens.fr
                                                                           (June 1993)
                                                  We present a powerful method for calculating the thermodynamic prop-
                                            erties of the Hubbard model in infinite dimensions, using an exact diagonal-
                                            ization of an Anderson model with a finite number of sites. At finite temper-
                                            atures, the explicit diagonalization of the Anderson Hamiltonian allows the
                                            calculation of Green’s functions with a resolution far superior to that of Quan-
                                            tum Monte Carlo calculations. At zero temperature, the Lanczòs method is
                                            used and yields the essentially exact zero-temperature solution of the model,
                                            except in a region of very small frequencies. Numerical results for the half-
                                            filled case in the paramagnetic phase (quasi-particle weight, self-energy, and
                                            also real-frequency spectral densities) are presented.
                                            PACS numbers: 71.10+x,75.10 Lp, 71.45 Lr, 75.30 Fv




                                                                                   1
    Following the pioneering work of Metzner and Vollhardt [1], the limit of large dimensions
for models of strongly correlated fermions has received much attention. In this limit, the
highly intricate quantum many-body problem simplifies considerably and leads to a non-
trivial mean-field theory [2]. Remarkably, this limit captures many features of the physics
in finite dimensions and gives a very successful description of quantum fluctuations.
    In spite of the considerable simplification obtained in taking the large D limit, the mean-
field equations still have to be solved numerically. Up to now, all calculations [3], [4], [5] have
relied on the Hirsch-Fye Quantum Monte Carlo (QMC) algorithm [6]. A major limitation of
this scheme is the difficulty of accessing the low-temperature regime, where statistical and
finite time-step discretization errors of the QMC algorithm become very important.
    In this paper, we present a powerful method for solving these mean-field equations, which
leads to an essentially exact solution in the imaginary frequency domain. As an example, we
consider the Hubbard model on a lattice of infinite connectivity z → ∞ which, after proper
rescaling of the kinetic energy, is written as
                                     X   1                   X
                           H =−         √ c+i c j + h.c. + U   ni↑ ni↓ ,                         (1)
                                  <ij>σ  2z                  i

The calculation of the single-site properties of the Hubbard model in this limit reduces to
the self-consistent determination of the on-site Green’s function G(ω) of the Hubbard model
and of a bath Green’s function G0 (ω), which describes the interaction on the single site
with the external environment. G(ω) and G0 (ω) are related by a self-consistency condition
which, on the Bethe lattice, reads:
                                                     1
                                     G−1
                                      0 (ω) = ω + µ − G(ω)                                       (2)
                                                     2
It is for simplicity only that we restrict our attention in this paper to the z → ∞ Bethe
lattice.
    As is well known [7] [8], the on-site Green’s function of the Hubbard model may be
interpreted as the Green’s function of an Anderson model
                      X              X
                                     ns                                X
                                                                       ns
          HAnd = ǫd        d+
                            σ dσ +           ǫk a+
                                                 kσ akσ + Und↑ nd↓ +        (Vk a+
                                                                                 kσ dσ + h.c.)   (3)
                       σ             σ,k=2                             σ,k=2

in which the function G0 (iωn ) is given by the U = 0 Green’s function of the impurity
                                                                   X
                                                                   ns
                                                                         Vk2
                  G0 (iωn ) = GAnd
                               0   (iωn ) = [iωn − ǫd − µ −                     ]−1              (4)
                                                                   k=2
                                                                       iωn − ǫk
Given the infinite number of degrees of freedom of the models defined in eq. (1) and eq. (3),
it is evident that strict self-consistency can only be obtained with a continuous Anderson
model, i. e. with ns = ∞. The main result of the present paper is that a systematic
approximation of G0 (iω) with a finite-ns Anderson model gives extremely good results. We
stress from the beginning that we are interested in an approximation of the imaginary-
frequency Green’s functions only.
                                                               −1 And
    In practice, we approximate any G−1  0 (iω) by a function G0      (iω) with a finite number
ns of sites. This can be cast into a minimization problem in the variables ǫk and Vk . For
this paper, we choose the following cost function:

                                                      2
                                       nX
                                   1    max
                                                                             2
                      χ2 =                    |G−1          −1 And
                                                0 (iωn ) − G0      (iωn )|                  (5)
                             nmax + 1 n=0

where nmax is chosen sufficiently large (ωnmax >> maxk (ǫk )) [9]. We search for the param-
eters ǫk and Vk minimizing the χ2 in eq. (5) with a standard conjugate gradient method
[10].
    For a small number of sites, ns ≤ 6, the Green’s function G(iωn ) can be obtained exactly
from the complete set of eigenvectors and eigenvalues of the Anderson Hamiltonian eq. (3).
The procedure
                             eq. (5)
                               −1 And             eq. (3)       eq. (2)
                 G−1
                  0 (iωn ) −→ G0      (iωn ) −→ G(iωn ) −→ G−1
                                                            0 (iωn )                        (6)

is then iterated to convergence.
    The following observations are made:
                                                                              −1 And
    1) We notice in general very small differences between G−1  0 (iω) and G0        (iω) as ex-
                                       2               2
pressed by small minimal values of χ in eq. (5). χ decreases by approximately a constant
factor each time we add one more site.
    2) The extensive comparisons with QMC results [5] which we have undertaken indicate
that exact diagonalization is by far the superior method for this problem. As an example,
we show in fig. 1 QMC and exact diagonalization data for the half-filled Hubbard model at
β = 32 and U = 3. The Monte Carlo data are shown for a imaginary-time discretization of
∆τ = 1, .5, and .25 (cf, e. g. [5]), and the exact diagonalization data for ns = 3, 5. It may
be worthy of notice that the diagonalization calculations can be obtained in a few minutes
on a work station, while for the QMC data acquisition (at ∆τ = 0.5) several hours were
needed (several days for ∆τ = 0.25).
    Beyond ns = 6, the size of the Hilbert space becomes too large for an explicit diagonal-
ization of the Anderson Hamiltonian. However, the calculation of zero-temperature Green’s
functions is still possible by means of the Lanczòs algorithm [11], which allows us to easily
calculate G(iω) and G0 (iω) up to ns ∼ 10 on a workstation [12]. The fit with the An-
derson model is performed as before. We simply replace the Matsubara frequencies by a
fine grid of imaginary frequencies, which correspond to a “fictitious” inverse temperature
β (ωn = (2n + 1)π/β). β introduces a low-frequency cutoff in an obvious way. In fig. 2
                                            −1 And
we display the functions G−1  0 (iω) and G0        (iω) for U = 2 and U = 4.8. At the scale
of the figure the two curves can hardly be distinguished, and an essentially perfect fit (i.e.
perfect self-consistency) is obtained in the whole range of frequencies. The inset in the figure
shows a blow-up of the small frequency regime as the number of sites is increased. Notice
the systematic amelioration of the fit. Furthermore, the ’physical’ Green’s function G−1    0 is
extremely independent of ns , especially at high frequency. Already at ω = 0.11, e.g., G−1    0
varies by less than 0.0001 between ns = 6, 8, and 10.
    For the data at U = 4.8, the quality of the fit is excellent even with a small number
of sites. This is easily explained by the existence of a physical cutoff in frequency, which
results from the Mott gap.
    We now pass to the calculation of other physical quantities and present in fig. 3 some
results for the quasi-particle spectral weight Z calculated from the slope of the self-energy
Σ = G−1       −1
       0 −G . In the inset of fig. 3 we present the raw data of ImΣ(iω) at small frequencies
from which the spectral weight is extracted (ImΣ(iω) ∼ (1 − 1/Z)ω + . . .). To get a truly

                                                  3
stabilized slope of Σ we have found it to be necessary to reach very low temperatures. The
main plot compares the results at ns = 10 with the “iterated perturbation theory” (IPT)
result. This method is based essentially on the use of a weak coupling calculation to second-
order in U of Σ and has shown to give a satisfactory interpolation between the small and
large U limits (exclusively at half filling and in the paramagnetic phase) [8], [13], [14]. On a
few points we give in addition the results of the exact diagonalization at ns = 6 and ns = 8.
Given the extremely good agreement between the values of Z calculated with ns = 8 and
10, we are very confident of the numerical values presented.
    As discussed in ref [13], the IPT approximation leads to a first-order Mott-Hubbard
transition (cf fig. 3), and the quasi-particle weight Z jumps discontinuously at U ∼ 3.6. We
have only found limited evidence for such a scenario within the present approach. At ns = 6,
we are unable to stabilize two solutions at the same values of the physical parameters (the
coexistence of two solutions is indicative of a first-order phase transition). At ns = 8, and
using a fictitious temperature of β = 120, we find a coexistence region within a very small
interval of U: 4.45 ≤ U ≤ 4.60 [15]. Even though the question of the order of the transition
will have to await a more detailed investigation, it seems to us to be difficult to reconcile
our numerical results with a abrupt transition anywhere close to U = 3.6.
    Finally, we show some data concerning the one-particle spectral densities ρ(ω) =
−ImG(ω +iǫ)/π as obtained from the Lanczòs calculation together with IPT-approximation
solutions [13]. Fig. 4 shows the spectral density (ns =10) for different values of U. In the
Fermi-liquid regime the spectrum of our finite-size Anderson model consists of a large num-
ber of peaks, while in the insulating phase we systematically observe a simpler structure
made of only a few peaks. As U is increased we see that ρ(ω) develops three well-separated
structures: a central quasi-particle feature and two broad high-energy satellite features cor-
responding to the formation of the upper Hubbard band. At U = 4.8 a gap is observed in
good agreement with the approximate IPT solution. In the insets of Fig. 4 we also present
the integrated single particle density of states corresponding to Lanczòs and IPT solutions.
The agreement between both curves is seen to be very good, provided we average over a
frequency interval of ω ∼ 0.5. This indicates that the calculated spectral density contains
coarse-grained information about the exact solution, as it should be. Due to the discrete
nature of the Anderson model used, the fine details of the spectrum are poorly reproduced.
    A remark is in order here: As is well known, the continuation of numerical data from the
imaginary axis onto the real-frequency domain is a very difficult problem and constitutes
for example one of the major limitations of the QMC method. Here we encountered the
analytic continuation problem in the ’easy’ direction. Indeed, in the present work very
precise imaginary frequency data (cf fig, 1, fig. 2) can be obtained with a representation
in ω , which very cleary has its limits (cf fig. 4). It is crucial in this context that we only
attempt to satisfy the self-consistency condition on the imaginary axis.
    In conclusion, we have presented a powerful numerical method for simulating the D = ∞
models for strongly correlated fermions based on a self-consistent single-impurity model
treated by exact diagonalization. At the temperatures reachable by quantum Monte Carlo
calculations we get essentially the exact solution of the model. At lower temperatures
unreachable by QMC, we get also an amazingly good solution, except in the region of very
small frequencies where some difficulties appear due to the finite degrees of freedom in the
                                              And
representation of the free propagator (G−1  0     ) of the impurity.


                                               4
    Elsewhere [16] we present a study of the instability of the normal phase with respect
to superconductivity of an infinite-D two-band Hubbard model [17]. That work and addi-
tional calculations on the Hubbard model away from half-filling clearly show that the exact
diagonalization method presented in this work is in no way limited to the particle-hole sym-
metric point of the Hubbard model. Broken-symmetry phases, magnetic fields, etc, as well
as the calculation of susceptibilities [18] can be easily handled within this approach, which
we expect to rapidly become a standard tool for the investigation of D = ∞ systems.


                                ACKNOWLEDGMENTS

   We acknowledge helpful discussions with J. Bellissard, A. Georges, G. Kotliar, D. Poil-
blanc and T. Ziman. This work was supported by DRET contract no 921479.

+
 Permanent address: Laboratoire Dynamique des Interactions Moléculaires, Tour 22 Uni-
versité Paris VI; 4 place Jussieu F-75252 Paris Cedex 05, France




                                             5
 [1] W. Metzner and D. Vollhardt, Phys. Rev. Lett. 62 324 (1989)
 [2] For a recent review and references, see e.g. D.Vollhardt, to appear in ’Correlated Electron
     Systems’, proceedings of the Jerusalem Winter School of Theoretical Physics, V. J. Emery ed.
     (World Scientific). (preprint RWTH/ITP-C 6/92)
 [3] M. Jarrell, Phys. Rev. Lett. 69, 168 (1992)
 [4] M. Rozenberg, X. Y. Zhang and G. Kotliar, Phys. Rev. Lett. 69, 1236 (1992)
 [5] A. Georges and W. Krauth, Phys. Rev. Lett. 69, 1240 (1992)
 [6] J. E. Hirsch and R. M. Fye, Phys. Rev. Lett. 56 2521 (1986)
 [7] P. van Dongen and D. Vollhardt, Phys. Rev. Lett. 65 1663 (1990)
 [8] A. Georges and G. Kotliar Phys. Rev. B 45 6479 (1992)
 [9] The precise form of the function which is minimized plays no role, as it should be. Practically
     the same results can be obtained by fitting G0 and GAnd   0  instead of their inverses cf [16], or
     by putting in ω-dependent factors.
[10] In order to force the particle-hole symmetry a symmetric distribution of the ǫl ’s around zero
     with one energy kept fixed at zero has been chosen (ns − 1 independent parameters) in most
     of our simulations.
[11] R. Haydock, V. Heine, and M. J. Kelly J. Phys. C 8, 2591 (1975)
[12] Higher ns (say, up to ns ∼ 16) could be handled on a supercomputer using efficiently de-
     signed algorithms (that was not the purpose of this work). In any case, this would not change
     quantitatively our well-converged results (see below).
[13] A. Georges and W. Krauth, LPTENS preprint 92/24 (1992) to appear in Phys. Rev. B (Sept.
     1993)
[14] X. Y. Zhang, M. Rosenberg and G. Kotliar, Preprint
[15] Convergence is very slow in the transition region. There it may take several hundred iterations
     to destroy an apparently stable Fermi-liquid or Mott insulating phase. However, we always end
     up with a fully stabilized solution. Such a situation is almost impossible to handle correctly in
     expensive QMC calculations at low temperatures.
[16] W. Krauth and M. Caffarel, LPSENS preprint 93/17 (1993)
[17] A. Georges, G. Kotliar, and W. Krauth, LPTENS preprint 93/10, to appear in Z. Phys. (1993)
[18] work in progress, in collaboration with A. Georges.




                                                  6
Figure Captions

1. Comparison of G(τ ) for U = 3, and β = 32 between exact diagonalization (ns = 3, 5
   (bottom solid lines) the results cannot be distinguished on the scale plotted) and
   Quantum Monte Carlo (solid: ∆τ = 1, dashed: ∆τ = 0.5). Inset: G(τ = 4) vs. ∆τ 2
   (the QMC algorithm converges roughly in ∆τ 2 ). At ∆τ = 0. exact diagonalization
   results for ns = 3, 5.
                             −1 And
2. Plot of G−1
             0 (iω) and of G0       for two values of the interaction, U = 2 (Fermi liquid-
   regime) and U = 4.8 (Mott insulator regime) at ns = 10. The inset gives G0 (iω)−1
            And
   and G−1
         0      at small frequency for ns = 6, 8, 10. Note the systematic improvement of
   the solution. The misfit between the two functions is the only source of error of the
   exact diagonalization approach.

3. Quasi-particle weight Z as a function of U for the half-filled Hubbard model. The curve
   gives the IPT approximation, which predicts a first-order transition. The crosses give
   the results for ns = 10, with the corresponding results for ns = 6, 8 at two points.
   The inset shows the small-ω behavior of ImΣ(iω) for ns = 6, 8, 10 from which the
   quasi-particle weight is calculated. Note the excellent convergence with ns .

4. Density of states ρ(ω) for different values of U (a value of ǫ = 0.01 is used). We
   compare with the IPT density of states [13]. Insets: Comparison of the integrated
   densities of states between exact diagonalization and IPT.




                                           7
