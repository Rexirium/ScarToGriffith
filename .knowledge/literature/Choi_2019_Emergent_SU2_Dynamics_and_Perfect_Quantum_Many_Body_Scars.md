# Emergent SU(2) Dynamics and Perfect Quantum Many-Body Scars

Soonwon Choi<sup>1</sup>, Christopher J. Turner<sup>2</sup>, Hannes Pichler<sup>3,4</sup>, Wen Wei Ho<sup>3</sup>, Alexios A. Michailidis<sup>5</sup>, Zlatko Papić<sup>2</sup>, Maksym Serbyn<sup>5</sup>, Mikhail D. Lukin<sup>3</sup>, and Dmitry A. Abanin<sup>6</sup>

<sup>1</sup> Department of Physics, University of California Berkeley, Berkeley, California 94720, USA  
<sup>2</sup> School of Physics and Astronomy, University of Leeds, Leeds LS2 9JT, United Kingdom  
<sup>3</sup> Department of Physics, Harvard University, Cambridge, Massachusetts 02138, USA  
<sup>4</sup> ITAMP, Harvard-Smithsonian Center for Astrophysics, Cambridge, Massachusetts 02138, USA  
<sup>5</sup> IST Austria, Am Campus 1, 3400 Klosterneuburg, Austria  
<sup>6</sup> Department of Theoretical Physics, University of Geneva, 1211 Geneva, Switzerland

*Received 7 January 2019; published 7 June 2019*

## Abstract

Motivated by recent experimental observations of coherent many-body revivals in a constrained Rydberg atom chain, we construct a weak quasilocal deformation of the Rydberg-blockaded Hamiltonian, which makes the revivals virtually perfect. Our analysis suggests the existence of an underlying nonintegrable Hamiltonian which supports an emergent SU(2)-spin dynamics within a small subspace of the many-body Hilbert space. We show that such perfect dynamics necessitates the existence of atypical, nonergodic energy eigenstates---quantum many-body scars. Furthermore, using these insights, we construct a toy model that hosts exact quantum many-body scars, providing an intuitive explanation of their origin. Our results offer specific routes to enhancing coherent many-body revivals and provide a step toward establishing the stability of quantum many-body scars in the thermodynamic limit.

DOI: [10.1103/PhysRevLett.122.220603](https://doi.org/10.1103/PhysRevLett.122.220603)

Remarkable experimental advances have recently enabled studies of nonequilibrium dynamics of isolated, strongly interacting quantum systems [1-3]. In such systems, it is commonly believed that a generic state far from equilibrium eventually thermalizes, whereupon any initial local information becomes unrecoverable [4-6]. While this process of thermalization provides the basis of statistical mechanics, it also poses challenges for building large-scale quantum devices. Hence, it is of fundamental interest to understand mechanisms to evade thermalization. Two well-studied possibilities include many-body localization in strongly disordered systems, and fine-tuned integrable systems [7-9].

Recently, quench experiments with Rydberg atom arrays [10-12] have discovered nonthermalizing dynamics of a new kind [12]. Initialized in a high-energy Néel state, the system exhibited unexpectedly long-lived, periodic revivals, failing to thermalize on experimentally accessible timescales; in contrast, other high-energy product states exhibited thermalizing dynamics consistent with conventional expectations.

These surprising observations have stimulated strong theoretical interest [13-17]. Reference [13] showed that the oscillatory dynamics stems from a small number of nonthermal many-body eigenstates, which are embedded in a sea of thermal eigenstates that generically obey the eigenstate thermalization hypothesis (ETH) [4-6]. These atypical, ergodicity-breaking eigenstates were named “quantum many-body scars” (QMBS) in analogy to quantum scars in single-particle quantum systems, which are similarly nonergodic wave functions that concentrate along the unstable, periodic trajectories of the counterpart classical system [18]. Reference [14] firmed up this analogy by showing that the long-lived revivals were also closely related to an unstable periodic orbit in a variational, “semiclassical” description of the quantum many-body dynamics.

Despite much theoretical effort, several key questions regarding the nature of QMBS remain open. In particular, owing to the slow decay, the ultimate fate of the revivals at very long times in the thermodynamic limit is not fully understood. Another outstanding challenge is to understand the physical mechanism protecting scars in the Rydberg chain and beyond. Reference [16] conjectured that the observed revivals arise due to proximity to a putative integrable point. They demonstrated a nontrivial deformation of the Rydberg-blockaded Hamiltonian that results in a substantial modification of the many-body level statistics with the entire spectrum becoming accompanyingly nonthermal, which could be interpreted as proximity to integrability. We note that earlier works [19,20] have demonstrated the coexistence of ETH-violating states in a generically ergodic spectrum, by explicitly constructing many-body eigenstates with low entanglement at arbitrary energy densities in a nonintegrable Affleck-Kennedy-Lieb-Tasaki model. Also, it has been reported that quantum Ising models with longitudinal field can exhibit weak thermalization at low-energy densities [21-23].

In the present work, we show that the periodic many-body revivals of the effective model describing the experiment [12] become extremely stable with a suitable weak, quasilocal deformation, with the return probability of the Néel state approaching unity within $10^{-6}$ for systems with more than 30 particles. Remarkably, despite such manifestly nonergodic dynamics at infinite temperature and the strongly nonthermal character of the associated scarred eigenstates, the bulk of the spectrum remains well thermal, in contrast to the conjecture in Ref. [16]. Rather than being integrable, the revival dynamics can be understood as the coherent rotation of an emergent, large SU(2) spin that lives within a special subspace of the many-body Hilbert space.

Our results strongly suggest the existence of a “parent” Hamiltonian with perfect oscillatory dynamics. We prove that, under generic settings, such perfect revivals impose strong constraints on the structure of energy eigenstates, necessitating the presence of some ETH-violating eigenstates. This result directly relates observable nonequilibrium dynamics to properties of energy eigenstates and parallels the mechanism behind quantum scarring in single-particle quantum chaos [18]. Finally, guided by the emergent SU(2)-spin structure, we construct a solvable toy model that explicitly hosts the phenomenology of QMBS, which provides an intuitive explanation of their origin in the constrained model.

## Model and revivals

The one-dimensional array of Rydberg atoms in the experiments [12] is described by a kinetically constrained [24,25] spin-$1/2$ chain with Hamiltonian

$$
H_0 = \sum_{i=1}^{N} C\sigma_i^x C. \tag{1}
$$

Here $\sigma_i^\mu$ ($\mu \in \{x,y,z\}$) are Pauli operators at site $i$, and

$$
C = \prod_i \left[1 - \frac{(1+\sigma_i^z)(1+\sigma_{i+1}^z)}{4}\right]
$$

is a global projector constraining the Hilbert space to spin configurations without two adjacent up spins, $|\uparrow\uparrow\rangle$, corresponding to the regime of a strong nearest-neighbor Rydberg blockade [26] in the experiments [12]. The dynamics is such that a spin may flip only when both of its neighbors are in the $|\downarrow\rangle$ state, and the model is thus strongly interacting [27-29]. For simplicity, we assume periodic boundary conditions and only consider the constrained Hilbert space defined by $C=1$, whose dimensionality grows asymptotically as $\phi^N$, where $\phi=(1+\sqrt{5})/2$ is the golden ratio.

The model in Eq. (1) exhibits unexpected, long-lived periodic revivals when initialized in the Néel state $|Z_2\rangle=|\uparrow\downarrow\uparrow\downarrow\cdots\rangle$. Despite its large energy density (corresponding to infinite temperature), quench dynamics from this initial state exhibits large recurrences of the Loschmidt echo $g_0(t)\equiv|\langle Z_2|e^{-iH_0t}|Z_2\rangle|^2$ at multiples of a period $\tau$ with a slow overall decay [Fig. 1(a)] [12-16]. This is accompanied by a generally linear growth of the bipartite entanglement entropy [Fig. 1(b)], which is slower compared to thermalizing initial states. As shown in Ref. [13], such dynamics arise due to the existence of a band of special, nonthermal “quantum many-body scarred” eigenstates that have large overlaps with $|Z_2\rangle$. Furthermore, these special eigenstates can be approximately constructed using an analytical framework that was dubbed the forward scattering approximation (FSA) [13,15]. In essence, FSA relies on decomposing the Hamiltonian into a “raising” and “lowering” part, $H_0=H_0^+ + H_0^-$, with

$$
H_0^\pm = \sum_{i\in\mathrm{even}} C\sigma_i^\pm C + \sum_{i\in\mathrm{odd}} C\sigma_i^\mp C.
$$

Then, $N+1$ vectors $|k\rangle_0=\beta_k H_0^+|k-1\rangle_0$ can be recursively defined from $|0\rangle_0=|Z_2\rangle$, where $k\in\{0,1,2,\ldots,N\}$ and $\beta_k$ is the normalization coefficient, spanning a subspace $\mathcal K$. It has been shown that eigenstates belonging to the special band predominantly live in $\mathcal K$ [13,15].

### Stabilizing revivals

In order to stabilize the revivals of $|Z_2\rangle$, we consider various perturbations that preserve the particle-hole, time-reversal, and inversion symmetries of the system (thus, pinning the energy of $|Z_2\rangle$). Generically, most perturbations weaken the revivals. However, we find that the following range-four deformation

$$
\delta H_2=-\sum_i h_2 C\sigma_i^x C(\sigma_{i+2}^z+\sigma_{i-2}^z). \tag{2}
$$

with $h_2\approx0.05$ (derived below) significantly improves fidelities of the revivals. Physically, this perturbation corresponds to raising or lowering of a spin, depending on the configuration of nearby spins. We note that this form has been previously considered in Ref. [16], which numerically found that at $h_2\approx0.024$, the entire spectrum becomes least thermal. In contrast, our value of $h_2$ is approximately twice larger, and the spectrum remains thermal, aside from the scarred eigenstates (see below).

Our key observation is that $\delta H_2$ partially cancels errors arising in the FSA analysis. More specifically, the precision of FSA, and therefore the fidelity of revivals, relies on the dynamics from the $|Z_2\rangle$ initial state generated by $H_0^\pm$ to be (nearly) closed in $\mathcal K$. This condition would be exactly achieved if the $|k\rangle$ were eigenstates of the operator $H_0^z\equiv[H_0^+,H_0^-]$ but is generically not satisfied for $2\leq k\leq N-2$. We find that this error can be reduced by adding $\delta H_2$ to the Hamiltonian, and redefining the raising (lowering) operators $H_0^\pm\mapsto H_2^\pm$ (hence also $H_0^z\mapsto H_2^z$), and the subspace $\mathcal K$ by replacing $\sigma_i^\pm\mapsto\sigma_i^\pm[1+h_2(\sigma_{i+2}^z+\sigma_{i-2}^z)]$. For example, one can analytically show that the component of $H_2^z|2\rangle$ perpendicular to $|2\rangle$ is significantly decreased when $h_2=1/2-1/\sqrt5\approx0.053$ [30]. Indeed, this perturbation strongly improves many-body revivals, leading to fidelity $g(\tau)\approx0.998$ at its first maximum for $N=32$. Furthermore, the linear growth of entanglement entropy is significantly slowed.

The dramatic increase in revival fidelities owing to $\delta H_2$ suggests that it might be possible to further enhance the oscillations, making them perfect. It is natural to consider longer-range perturbations of the form

$$
\delta H_R=-\sum_i\sum_{d=2}^{R}h_d C\sigma_i^xC(\sigma_{i-d}^z+\sigma_{i+d}^z). \tag{3}
$$

which describe additional interactions between pairs of spins separated by a distance $d$, with strengths $h_d$. We numerically optimize $h_d$ by maximizing the fidelity $g(t)$ under $H=H_0+\delta H_R$ at its first revival; see Fig. 1(c) for $N=20$ with $R=10$. In Ref. [30], we show that qualitatively similar results are obtained from other optimization methods, e.g., minimizing errors in FSA, etc. We find that the optimized $h_d$ decay exponentially at large $d$ and can be very well approximated by the expression

$$
h_d^{\mathrm{ansatz}}=h_0\left(\phi^{d-1}-\phi^{-(d-1)}\right)^{-2}. \tag{4}
$$

where $\phi$ is the golden ratio, and $h_0$ is a single parameter determining the overall strength. Henceforth, we will use $h_d$ from Eq. (4) truncated at the maximum distance $R=N/2$, which allows us to perform a meaningful finite-size scaling analysis. Numerical optimization of the ansatz yields $h_0\approx0.051$. Below, we will derive this value from certain algebraic relations among $H^\pm$, $H^z$ within the subspace $\mathcal K$, which are appropriately redefined quantities from $H_0^\pm$, $H_0^z$, $\mathcal K$ accounting for the long-range terms in an analogous fashion as the case for $R=2$ above.

Dynamics under the Hamiltonian $H=H_0+\delta H_R$ makes the $|Z_2\rangle$ revivals even more stable, with $1-g(\tau)\approx10^{-6}$ for $N=32$ at the first revival [Fig. 1(a)]. Simultaneously, we observe that the linear growth of the bipartite entanglement entropy is significantly reduced and is barely discernible [Fig. 1(b)]. A scaling analysis in [30] suggests that the average rate of local thermalization, defined by the decay of $g(t)^{1/N}$, at late times vanishes in the thermodynamic limit.

### Dynamics constrains eigenstate properties

The possible existence of a parent Hamiltonian leading to perfect oscillatory dynamics, strongly and quantifiably constrains the nonergodic nature of the quantum many-body scars. Specifically, we can appeal to the following general relation, whose proof is simple and given in Ref. [30].

**Lemma.** Consider a generic many-body Hamiltonian $H$ with extensive energy, $\|H\|=O(N)$. If an initial state $|\Psi_0\rangle$ under time evolution perfectly comes back to itself after some finite time $\tau$, independent of the system size $N$, i.e., $|\langle\Psi_0|e^{-iH\tau}|\Psi_0\rangle|=1$, then $|\Psi_0\rangle$ can be decomposed into $O(N)$ energy eigenstates, and at least one of them, $|\epsilon\rangle$, has a large overlap, $|\langle\epsilon|\Psi_0\rangle|^2\geq O(1/N)$.

If the periodic revival occurs for a physical state $|\Psi_0\rangle$ with a finite energy density (that obeys the cluster decomposition, so that the energy variance goes as $N$), such as $|Z_2\rangle$ in our case, this Lemma dictates the presence of a high-energy eigenstate with a large overlap $\sim1/N$ with a low-entangled state. This constitutes a violation of the ergodic scenario, where a high-energy eigenstate is expected to be similar to a random vector in the exponentially large Hilbert space.

In accordance with this result, the decomposition of the Néel state $|Z_2\rangle$ is seen to be dominated by $N+1$ special eigenstates of $H$ [Fig. 2(a)], which are much better separated from the bulk than in the case of unperturbed Hamiltonian. We also confirm that these eigenstates exhibit nonergodic properties, such as the logarithmic scaling of entanglement entropy, and can furthermore be constructed by a straightforward extension of FSA with significantly improved accuracy [15,30].

Importantly, while the deformed model shows very stable revivals, the bulk of the spectrum remains thermal. To illustrate this, we compute the $r$ parameter associated to the level repulsion of the energy levels $E_i$, $\langle r_i\rangle=\langle\min(\delta_i,\delta_{i+1})/\max(\delta_i,\delta_{i+1})\rangle$, where $\delta_i=E_{i+1}-E_i$ is the level spacing and $\langle\cdot\rangle$ indicates averaging over a symmetry-resolved Hilbert space sector [31]. Figure 2(c) shows a clear flow in system size toward $\langle r_i\rangle\approx0.53$, the Wigner-Dyson value associated with quantum chaotic Hamiltonians, implying the coexistence of nonergodic dynamics and an ergodic bulk. In addition, the distribution $P(s)$ of the unfolded level spacing $s$ is consistent with the Wigner-Dyson class of the Gaussian orthogonal ensemble [Fig. 2(b)].

### Algebraic structure in the subspace $\mathcal K$

The almost perfect fidelity revivals of the deformed Hamiltonian imply $H^\pm$ and $H^z$ form a closed algebra within the subspace $\mathcal K$. Indeed, we find numerically that

$$
P_{\mathcal K}[H^z,H^\pm]P_{\mathcal K}\approx\pm\Delta P_{\mathcal K}H^\pm P_{\mathcal K}, \tag{5}
$$

where $P_{\mathcal K}=\sum_k|k\rangle\langle k|$ is the projector onto $\mathcal K$, and $\Delta$ is a constant. As $|0\rangle=|Z_2\rangle$ is an eigenstate of $H^z$, $|k\rangle$ are also approximate eigenvectors of $H^z$ with harmonically spaced eigenvalues $H_k^z=\langle k|H^z|k\rangle$ with $\Delta\approx H_{k+1}^z-H_k^z$. Thus, upon a suitable rescaling, $H^z$ plays the role of $S^z$ in the $\mathfrak{su}(2)$ algebra, and $H^\pm$ play the role of raising and lowering operators within $\mathcal K$. As the dimensionality of $\mathcal K$ is $N+1$, this implies that the operators form a spin $s=N/2$ irreducible representation of the $\mathfrak{su}(2)$ algebra, with $|Z_2\rangle$ and $|Z'_2\rangle=|\downarrow\uparrow\downarrow\uparrow\ldots\rangle$ being the lowest and highest weight states, respectively. To check this, we explicitly evaluated the matrix elements $\langle k+1|H^+|k\rangle$. Figure 3(a) confirms that up to an overall multiplicative factor, the matrix elements of $H^+$ reproduce those of the spin-raising operator $S^+$ in this representation.

Thus, the virtually perfect oscillatory dynamics of $|Z_2\rangle$ can be understood as a large spin ($s=N/2$) pointing initially in an emergent “$z$ direction,” undergoing coherent Rabi oscillations under the Hamiltonian $H=H^++H^-$, which is akin to the $S^x$ operator, with period $\tau=2\pi/\sqrt{2\Delta}$. We stress that the emergence of this SU(2) structure within $\mathcal K$ is nontrivial, since the Hamiltonian $H$ does not have any rotational symmetry.

The identification of this emergent algebra allows us to fix $h_0$ of our ansatz for $h_d$ analytically. In particular, $H_k^z$ can be explicitly calculated for $k=0,1$ in the thermodynamic limit. Imposing a harmonic spacing, i.e., $H_k^z=\Delta(k-N/2)$, leads to a nontrivial constraint [30]

$$
(1-h)\left[1-h-16\sum_{n=1}^{\infty}h_{2n}\right]=16\sum_{n=1}^{\infty}h_{2n}^2. \tag{6}
$$

where $h\equiv2\sum_{n\geq2}h_n(-1)^n$. This fixes $h_0\approx0.0506656$ in our ansatz Eq. (4), which agrees very well with the numerically optimized value. Furthermore, Eq. (6) determines the harmonic gap $\Delta=(1-h)^2\approx0.835845$, and, correspondingly, the oscillation period $\tau\approx4.85962$, which are also in excellent agreement with those from exact numerical simulations [30].

### Toy model

The above investigations reveal that an emergent SU(2) structure within a special subspace underpins the many-body revivals. Motivated by this, we construct a (solvable) toy model that exhibits similar phenomenology: in this model, there is a band of nonthermal eigenstates supporting perfect oscillatory dynamics and exhibiting logarithmic entanglement, embedded in an otherwise thermal spectrum.

Consider a system of $N$ spin-$1/2$ particles on a ring. The special subspace $\mathcal V$ of our model is defined as the common null space of $N$ projection operators $P_{i,i+1}=(1-\vec\sigma_i\cdot\vec\sigma_{i+1})/4$ onto neighboring pairs of singlets and is spanned by the $N+1$ states of the largest spin representation $s=N/2$ of the $\mathfrak{su}(2)$ algebra. We enumerate the basis states $|s=N/2;S^x=m_x\rangle$ of $\mathcal V$ by eigenvalues of the $S^x=\sum_i\sigma_i^x/2$ operator, $m_x\in\{-s,\ldots,s\}$.

Now, we take any Hamiltonian of the form

$$
H_{\mathrm{toy}}=\frac{\Omega}{2}\sum_i\sigma_i^x+\sum_i V_{i-1,i+2}P_{i,i+1}. \tag{7}
$$

Here $V_{ij}$ is a generic two-spin operator acting on spins $(i,j)$, e.g., $V_{i,j}=\sum_{\mu\nu}J_{ij}^{\mu\nu}\sigma_i^\mu\sigma_j^\nu$ with arbitrary coefficients $J_{ij}^{\mu\nu}$. Note that $H_{\mathrm{toy}}$ does not commute with $P_{i,i+1}$ nor $S^x$; thus, it does not have any obvious local symmetries. However, it can be easily verified that $|s=N/2;S^x=m_x\rangle$ are eigenstates with harmonically spaced energies $E=\Omega m_x$. On the other hand, states in the Hilbert space outside of $\mathcal V$ are affected by $V_{i-1,i+2}$ terms and hybridize to form ergodic eigenstates [30]. Now, initializing our system, e.g., in the lowest weight state $|N/2;S^z=-N/2\rangle$ leads to rotations of a large spin around the $x$ axis with frequency $\Omega$, whose motion remains in $\mathcal V$. We note that our construction is reminiscent of work by Shiraishi and Mori [32] where a set of local projectors was used to embed certain nonergodic energy eigenstates into the bulk of a many-body spectrum.

Clearly, $H_{\mathrm{toy}}$ exhibits all the features of perfect quantum many-body scarring and is appealing as an intuitive understanding of the origin of scars in the constrained spin models. However, there remain many open questions: first, the explicit relationship between the constrained spin model Eqs. (1)-(3) and the toy model Eq. (7) is not obvious. The nonisomorphic Hilbert spaces, as well as the nontrivial entanglement dynamics in the constrained model [Fig. 1(b)], suggest that the mapping between these two models, if exists, cannot be strictly local. Second, it is highly desirable to find an analytic derivation of the deformation, Eq. (3), that leads to the emergent $\mathfrak{su}(2)$ algebra in the constrained spin model, and understand when such deformations exist for other local models. We note that this emergent algebra is reminiscent of the $\eta$-pairing symmetry that holds exactly in the Hubbard model [33], which allows to construct exact eigenstates at finite energy density with logarithmic [34] and subthermal entanglement [35].

## Summary and outlook

To summarize, we have constructed a constrained spin model which exhibits nearly perfect QMBS. The remarkably long-lived oscillatory dynamics suggests that quantum scars remain stable in the thermodynamic limit. We showed that the dynamics can be understood in terms of a large, precessing SU(2) spin, and used this intuition to introduce a family of toy models with perfect scarring. In future work, it would be highly desirable to find an analytical mapping between the toy models and the constrained spin model. Moreover, the approach developed here may be applied to stabilize other types of quantum scars, in particular the ones originating from the $|Z_3\rangle$ state in the model (1) [15], as well as the ones found in higher-spin constrained models [14].

Another exciting challenge is to find models in which the MPS-based description of quantum scars trajectory becomes exact [14]. In a broader context, special nonthermalizing trajectories may have intriguing connections to revivals or slow thermalization in strongly rotating gravitational systems [36,37]. To understand the origin of this nonthermalizing dynamics, it would be valuable to establish whether QMBS can emerge from a dynamic that goes through states with high entanglement.

Statement of compliance with EPSRC policy framework on research data: this publication is theoretical work that does not require supporting research data.

## Figure captions

**Figure 1. Nonthermalizing dynamics in constrained spin Hamiltonians.** (a) Many-body fidelity $g(t)$ as a function of time for the Hamiltonian $H_0$ without any perturbations and with optimal perturbations Eqs. (3) and (4). The inset shows the infidelity, $1-g(t)$. (b) Half-chain bipartite entanglement entropy (EE) dynamics. At the optimal perturbation point, the EE shows bounded, oscillatory dynamics. The inset shows the eigenvalues $p_\mu(t)$ of the half-chain reduced density matrix. Numerical simulations are performed with $N=32$ starting from the Néel state. (c) Optimized perturbation strengths $h_d$ decay exponentially. Solid line indicates the analytical ansatz function (4).

**Figure 2.** (a) The overlap of $|Z_2\rangle$ with energy eigenstates of $H$ is dominated by $N+1$ special ones (red circles), well separated from the bulk. The density of data points is color coded. (b) Eigenvalue level statistics of both $H_0$ and $H$ for $N=32$ closely follow that of Wigner-Dyson class of the Gaussian orthogonal ensemble. (c) Level statistics indicator $\langle r_i\rangle$ as a function of system size $N$ flows to its value in the Wigner-Dyson ensemble, indicating that the bulk of the system remains ergodic. All data is for the momentum $k=0$, inversion even sector.

**Figure 3. Emergent SU(2) structure in the subspace $\mathcal K$.** (a) Matrix elements of the operator $H^+$ between consecutive vectors $|k\rangle$ are in excellent agreement with that of an appropriately rescaled raising operator $S^+$ in the $s=N/2$ representation of the $\mathfrak{su}(2)$ algebra shown as the solid curve. (b) The FSA basis vectors $|k\rangle$ are approximate eigenstates of the operator $H^z$ with harmonically spaced eigenvalues. The inset shows the residual of the eigenvalue spacing $\Delta_k\equiv H^z_{k+1}-H^z_k$ away from its mean value. The error bars are extracted from variances in the expectation values of $H^z$ in $|k\rangle$.

## Acknowledgments

We thank E. Altman, D. Jafferis, V. Khemani, O. Motrunich, S. Shenker, and especially Xiaoliang Qi for useful discussions. This Letter was supported through the National Science Foundation (NSF), the Center for Ultracold Atoms, the Air Force Office of Scientific Research via the MURI, and the Vannevar Bush Faculty Fellowship. We are grateful to the Kavli Institute for Theoretical Physics, which is supported by the National Science Foundation under Grant No. NSF PHY-1748958, and the program “The Dynamics of Quantum Information,” where part of this work was completed. S. C. acknowledges supports from the Miller Institute for Basic Research in Science. H. P. is supported by the NSF through a grant for the Institute for Theoretical Atomic, Molecular, and Optical Physics at Harvard University and the Smithsonian Astrophysical Observatory. W. W. H. is supported by the Moore Foundation’s EPiQS Initiative through Grant No. GBMF4306. D. A. A. acknowledges support by the Swiss NSF. C. J. T. and Z. P. acknowledge support by EPSRC Grants No. EP/P009409/1, No. EP/R020612/1, and No. EP/M50807X/1.

S. C. and C. J. T. contributed equally to this work.

## References

1. I. Bloch, J. Dalibard, and S. Nascimbène, “Quantum simulations with ultracold quantum gases,” *Nat. Phys.* **8**, 267 (2012).
2. T. D. Ladd, J. Jelezko, R. Laflamme, Y. Nakamura, C. Monroe, and J. L. O’Brien, “Quantum computers,” *Nature (London)* **464**, 45 (2010).
3. A. Polkovnikov, K. Sengupta, A. Silva, and M. Vengalattore, “Colloquium: Nonequilibrium dynamics of closed interacting quantum systems,” *Rev. Mod. Phys.* **83**, 863 (2011).
4. J. M. Deutsch, “Quantum statistical mechanics in a closed system,” *Phys. Rev. A* **43**, 2046 (1991).
5. M. Srednicki, “Chaos and quantum thermalization,” *Phys. Rev. E* **50**, 888 (1994).
6. M. Rigol, V. Dunjko, and M. Olshanii, “Thermalization and its mechanism for generic isolated quantum systems,” *Nature (London)* **452**, 854 (2008).
7. R. Nandkishore and D. A. Huse, “Many-body localization and thermalization in quantum statistical mechanics,” *Annu. Rev. Condens. Matter Phys.* **6**, 15 (2015).
8. D. A. Abanin, E. Altman, I. Bloch, and M. Serbyn, “Ergodicity, Entanglement and many-body localization,” arXiv:1804.11065.
9. B. Sutherland, *Beautiful Models: 70 Years of Exactly Solved Quantum Many-Body Problems* (World Scientific, Singapore, 2004).
10. P. Schauß *et al.*, “Observation of spatially ordered structures in a two-dimensional Rydberg gas,” *Nature (London)* **491**, 87 (2012).
11. H. Labuhn *et al.*, “Tunable two-dimensional arrays of single Rydberg atoms for realizing quantum Ising models,” *Nature (London)* **534**, 667 (2016).
12. H. Bernien *et al.*, “Probing many-body dynamics on a 51-atom quantum simulator,” *Nature (London)* **551**, 579 (2017).
13. C. J. Turner, A. A. Michailidis, D. A. Abanin, M. Serbyn, and Z. Papić, “Weak ergodicity breaking from quantum many-body scars,” *Nat. Phys.* **14**, 745 (2018).
14. W. W. Ho, S. Choi, H. Pichler, and M. D. Lukin, “Periodic Orbits, Entanglement, and Quantum Many-Body Scars in Constrained Models: Matrix Product State Approach,” *Phys. Rev. Lett.* **122**, 040603 (2019).
15. C. J. Turner, A. A. Michailidis, D. A. Abanin, M. Serbyn, and Z. Papić, “Quantum scarred eigenstates in a Rydberg atom chain: Entanglement, breakdown of thermalization, and stability to perturbations,” *Phys. Rev. B* **98**, 155134 (2018).
16. V. Khemani, C. R. Laumann, and A. Chandran, “Signatures of integrability in the dynamics of Rydberg-blockaded chains,” *Phys. Rev. B* **99**, 161101 (2019).
17. C.-J. Lin and O. I. Motrunich, “Exact strong-ETH violating eigenstates in the Rydberg-blockaded atom chain,” arXiv:1810.00888.
18. E. J. Heller, “Bound-State Eigenfunctions of Classically Chaotic Hamiltonian Systems: Scars of Periodic Orbits,” *Phys. Rev. Lett.* **53**, 1515 (1984).
19. S. Moudgalya, S. Rachel, B. A. Bernevig, and N. Regnault, “Exact excited states of non-integrable models,” *Phys. Rev. B* **98**, 235155 (2018).
20. S. Moudgalya, N. Regnault, and B. A. Bernevig, “Entanglement of exact excited states of AKLT models: Exact results, many-body scars and the violation of strong ETH,” *Phys. Rev. B* **98**, 235156 (2018).
21. M. C. Bañuls, J. I. Cirac, and M. B. Hastings, “Strong and Weak Thermalization of Infinite Nonintegrable Quantum Systems,” *Phys. Rev. Lett.* **106**, 050405 (2011).
22. C.-J. Lin and O. I. Motrunich, “Quasiparticle explanation of the weak-thermalization regime under quench in a non-integrable quantum spin chain,” *Phys. Rev. A* **95**, 023621 (2017).
23. A. J. A. James, R. M. Konik, and N. J. Robinson, “Non-thermal States Arising from Confinement in One and Two Dimensions,” *Phys. Rev. Lett.* **122**, 130603 (2019).
24. G. H. Fredrickson and H. C. Andersen, “Kinetic Ising Model of the Glass Transition,” *Phys. Rev. Lett.* **53**, 1244 (1984).
25. R. G. Palmer, D. L. Stein, E. Abrahams, and P. W. Anderson, “Models of Hierarchically Constrained Dynamics for Glassy Relaxation,” *Phys. Rev. Lett.* **53**, 958 (1984).
26. D. Jaksch *et al.*, “Fast Quantum Gates for Neutral Atoms,” *Phys. Rev. Lett.* **85**, 2208 (2000).
27. B. Sun and F. Robicheaux, “Numerical study of two-body correlation in a 1d lattice with perfect blockade,” *New J. Phys.* **10**, 045032 (2008).
28. B. Olmos, R. González-Férez, and I. Lesanovsky, “Collective Rydberg excitations of an atomic gas confined in a ring lattice,” *Phys. Rev. A* **79**, 043419 (2009).
29. B. Olmos, R. González-Férez, I. Lesanovsky, and L. Velázquez, “Universal time evolution of a Rydberg lattice gas with perfect blockade,” *J. Phys. A* **45**, 325301 (2012).
30. See Supplemental Material at <http://link.aps.org/supplemental/10.1103/PhysRevLett.122.220603> for detailed information on the optimization of the deformation parameters, additional numerical simulation results, scaling analysis, and the proof of the lemma.
31. Because our Hamiltonian has spatial translation and inversion symmetries, we only diagonalize the symmetry sector with total momentum zero and even inversion parity. We exclude the degenerate energy eigenstates at zero energy since they originate from symmetry considerations.
32. N. Shiraishi and T. Mori, “Systematic Construction of Counterexamples to the Eigenstate Thermalization Hypothesis,” *Phys. Rev. Lett.* **119**, 030601 (2017).
33. C. N. Yang, “$\eta$ Pairing and Off-Diagonal Long-Range Order in a Hubbard Model,” *Phys. Rev. Lett.* **63**, 2144 (1989).
34. O. Vafek, N. Regnault, and B. Andrei Bernevig, “Entanglement of exact excited eigenstates of the Hubbard model in arbitrary dimension,” *SciPost Phys.* **3**, 043 (2017).
35. T. Veness, F. H. L. Essler, and M. P. A. Fisher, “Quantum disentangled liquid in the half-filled Hubbard model,” *Phys. Rev. B* **96**, 195153 (2017).
36. E. da Silva, E. Lopez, J. Mas, and A. Serantes, “Collapse and revival in holographic quenches,” *J. High Energy Phys.* **04** (2015) 038.
37. D. Jafferis (private communication).
