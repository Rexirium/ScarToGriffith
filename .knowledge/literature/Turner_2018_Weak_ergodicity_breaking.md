# Weak ergodicity breaking from quantum many-body scars

**C. J. Turner¹, A. A. Michailidis¹˒², D. A. Abanin³, M. Serbyn² and Z. Papić¹\***

1. School of Physics and Astronomy, University of Leeds, Leeds, UK.
2. IST Austria, Klosterneuburg, Austria.
3. Department of Theoretical Physics, University of Geneva, Geneva, Switzerland.

\*e-mail: z.papic@leeds.ac.uk

**Nature Physics**, Vol. 14, July 2018, pp. 745–749. Articles. DOI: [10.1038/s41567-018-0137-5](https://doi.org/10.1038/s41567-018-0137-5).

> **Transcription note (not part of the article).** This Markdown transcribes all six pages of the supplied published PDF, including Methods and references 1–55. The separate Supplementary Information is not included in that PDF. Running headers, footers, page breaks, ORCID icons and reader highlighting are omitted. Citation superscripts are represented by bracketed reference numbers. Mathematical expressions use LaTeX; the density-wave symbol is transcribed as $\mathbb{Z}$. Original figure captions are retained, followed by explicitly labelled descriptions added for this transcription. A short editorial note at the end records apparent source inconsistencies without silently rewriting the authors' equations.

## Abstract

The thermodynamic description of many-particle systems rests on the assumption of ergodicity, the ability of a system to explore all allowed configurations in the phase space. Recent studies on many-body localization have revealed the existence of systems that strongly violate ergodicity in the presence of quenched disorder. Here, we demonstrate that ergodicity can be weakly broken by a different mechanism, arising from the presence of special eigenstates in the many-body spectrum that are reminiscent of quantum scars in chaotic non-interacting systems. In the single-particle case, quantum scars correspond to wavefunctions that concentrate in the vicinity of unstable periodic classical trajectories. We show that many-body scars appear in the Fibonacci chain, a model with a constrained local Hilbert space that has recently been experimentally realized in a Rydberg-atom quantum simulator. The quantum scarred eigenstates are embedded throughout the otherwise thermalizing many-body spectrum but lead to direct experimental signatures, as we show for periodic recurrences that reproduce those observed in the experiment. Our results suggest that scarred many-body bands give rise to a new universality class of quantum dynamics, opening up opportunities for the creation of novel states with long-lived coherence in systems that are now experimentally realizable.

<!-- The following introductory text has no section heading in the PDF. -->

Controllable, quantum-coherent systems of ultracold atoms [1,2], trapped ions [3] and nitrogen-vacancy spins in diamond [4] have emerged as platforms for realizing and probing highly non-equilibrium quantum matter. In particular, these systems have opened the door to the investigation of non-ergodic dynamics in isolated quantum systems. Such an unusual kind of dynamics is now known to occur when there is an emergence of many conserved quantities, such as in integrable systems [5] and many-body localization [6–8]. In both cases, the system strongly violates the ‘eigenstate thermalization hypothesis’ (ETH) [9,10], which was conjectured to govern the properties of ergodic systems and their approach to thermal equilibrium. This motivates the question: are there systems that only weakly break ergodicity? In particular, are there systems in which some eigenstates are atypical and dynamics strongly depends on the initial conditions? The existing theoretical studies, which tested the ETH numerically in systems of spins, fermions and bosons in one-dimensional (1D) and two-dimensional (2D) systems [11,12] seem to rule out such a possibility. In particular, an earlier study [13] found that in an ergodic spin chain, all highly excited states were typical and thermal, obeying the strong version of the ETH.

In this paper we demonstrate that weak breaking of ergodicity can occur in kinetically constrained 1D models that are reminiscent of the effective models describing the interactions between anyon excitations in 2D topological phases of matter, such as fractional quantum Hall states [14]. Topological order in these systems is connected with emergent gauge fields, such that their many-body Hilbert space cannot be decomposed as a tensor product of local Hilbert spaces. Although models of this type have been theoretically investigated [15–22], recent works [23–25] demonstrate that they can also be realized in experiments with Rydberg atoms in 1D or 2D traps. We focus on the simple example of a 1D chain (with the Hamiltonian defined in equation (1) below) whose constrained Hilbert space grows in the Fibonacci sequence, and therefore below we refer to this system as a ‘Fibonacci chain’.

Our results for the non-ergodic dynamics in the Fibonacci chain can be summarized as follows. First, based on the energy level statistics, we find that the model exhibits level repulsion and is non-integrable. Second, we show that the model has a band of special eigenstates coexisting with thermalizing eigenstates in the middle of the many-body band. Surprisingly, even though the special eigenstates comprise only a vanishing fraction of all states in the thermodynamic limit, they have direct physical manifestations, and they can be accessed by preparing the system in specific product states. In particular, the band of special eigenstates underlies the unexpected long-duration oscillations observed experimentally [25]. Finally, we shed light on the structure of special eigenstates by introducing an effective tight-binding model. This allows us to obtain accurate numerical approximations of special eigenstates by solving the problem of a single particle hopping on a Hilbert space graph.

The existence of a band of special eigenstates is strongly reminiscent of the phenomenon of quantum scars in single-particle chaotic billiards [26], which have been observed in microwave cavities [27] and quantum dots [28]. In the context of single-particle quantum chaos, scars represent a concentration of some eigenfunctions along the trajectory of unstable classical periodic orbits. In our study of a many-body system, strong revivals arise due to the existence of an analogous trajectory in the Hilbert space that includes two particular product states. We show that this periodic orbit ‘scars’ a subset of eigenstates that form a special band. Analogous to the single-particle case, these eigenstates are concentrated in parts of the Hilbert space. Our effective tight-binding approach sheds light on the structure of scarred eigenstates and, correspondingly, the structure of the periodic orbit.

## Model

In the experiment [25], a chain of Rydberg atoms was realized in which, effectively, excitations were created/annihilated with equal amplitudes. In the limit where the nearest-neighbour interaction is much larger than the detuning and the Rabi frequency, the system is modelled by the following spin-1/2 Hamiltonian [16],

$$
H=\sum_{i=1}^{L}P_iX_{i+1}P_{i+2}
\tag{1}
$$

where $X_i,Y_i,Z_i$ are the Pauli operators, $L$ denotes the length of the chain and we work in units $\hbar=1$. In what follows, we use $|\circ\rangle$ to refer the ground state and $|\bullet\rangle$ to refer to the Rydberg state of a single atom. The operator $X_i=|\circ\rangle\langle\bullet|+|\bullet\rangle\langle\circ|$ creates or removes an excitation at a given site, and projectors $P_i=|\circ\rangle\langle\circ|=(1-Z_i)/2$, written in terms of $Z_i=|\bullet\rangle\langle\bullet|-|\circ\rangle\langle\circ|$, ensure that the nearby atoms are not simultaneously in the excited state. For example, $P_1X_2P_3$ acting on $|\circ\circ\circ\rangle$ gives $|\circ\bullet\circ\rangle$ (and vice versa), while it annihilates any of the configurations $|\bullet\circ\circ\rangle$, $|\circ\circ\bullet\rangle$, $|\bullet\circ\bullet\rangle$.

The presence of projectors in the Hamiltonian (equation (1)) does not allow for relaxation of several adjacent Rydberg atoms. In other words, configurations $|\ldots\bullet\bullet\ldots\rangle$ are ‘dark states’ breaking our chain into two disconnected parts. Hence, such configurations are excluded in what follows and we consider a constrained Hilbert space without any adjacent Rydberg atoms. Such a constraint makes the Hilbert space identical to that of chains of non-Abelian Fibonacci anyons, rather than spins-1/2 or fermions. For periodic boundary conditions (PBC), the Hilbert space dimension is equal to $\mathcal{D}=F_{L-1}+F_{L+1}$, where $F_n$ is the $n$th Fibonacci number. For instance, in the case of the $L=6$ chain we have $\mathcal{D}=18$, as shown in Fig. 1. For open boundary conditions (OBC), $\mathcal{D}$ scales as $F_{L+2}$. Thus, the Hilbert space is evidently very different from, for example, the spin-$\frac{1}{2}$ chain where the number of states grows as $2^L$.

### Fig. 1 | The Hilbert space graph of the Fibonacci chain with $L=6$ sites.

**Original caption.** The nodes of the graph label the allowed product states, while the edges connect configurations that result from a given product state due to the action of the Hamiltonian. Nodes of the graph are grouped according to the Hamming distance $D_{\mathbb{Z}_2}$ from the $|\mathbb{Z}_2\rangle$ state.

**Image description (added).** Eighteen blue nodes are arranged in seven vertical groups, labelled $D_{\mathbb{Z}_2}=0,1,2,3,4,5,6$, over progressively darker shaded strips. Small strings of filled and empty circles indicate the six-site product configurations. Edges join allowed configurations in neighbouring distance groups. The two alternating configurations lie at the opposite ends of the graph.

The model in equation (1) is particle–hole symmetric: an operator $\mathcal{P}=\prod_i Z_i$ anticommutes with the Hamiltonian, $\mathcal{P}H=-H\mathcal{P}$, and therefore each eigenstate $|\psi\rangle$ with energy $E\ne0$ has a partner $\mathcal{P}|\psi\rangle$ with energy $-E$. Furthermore, the model has spatial inversion symmetry $I$ which maps $i\to L-i+1$. In addition, with PBC, this model has translation symmetry. In what follows, unless specified otherwise, we restrict ourselves to PBC (thus identifying $i=L+1$ and $i=1$) and explicitly resolve translation and inversion symmetries which allow us to fully diagonalize systems of up to $L=32$ sites (with $\mathcal{D}_{0+}=77{,}436$ states in the zero-momentum inversion-symmetric sector).

Experiment [25] and numerical simulations on small systems [29] revealed that the relaxation under unitary dynamics specified by the Hamiltonian (equation (1)) strongly depends on the initial state of the system. In particular, starting from period-2 charge density wave states

$$
|\mathbb{Z}_2\rangle=|\bullet\circ\bullet\circ\ldots\rangle,
\qquad |\mathbb{Z}'_2\rangle=|\circ\bullet\circ\bullet\ldots\rangle
\tag{2}
$$

that are related by a translation by one lattice period, the system shows surprising long-time oscillations of local observables for long chains of up to $L=51$ sites. Although this might suggest that the system is non-ergodic, it was also observed that the initial state with all atoms in the state $|\circ\rangle$ shows fast relaxation and no revivals, characteristic of thermalizing systems. Given that the model in equation (1) is translation invariant and has no disorder, many-body localization cannot be at play. Below we explain the origin of the observed oscillations and the apparent non-ergodic dynamics.

## Dynamics

We start by characterizing the dynamical evolution of the model in equation (1) for different initial conditions. Motivated by experiment [25], we consider a family of charge density wave states $|\mathbb{Z}_k\rangle=|\ldots\bullet\circ\ldots\circ\bullet\ldots\rangle$, where the atoms in excited states are separated by $k-1$ atoms in the ground state, as well as the fully polarized state $|\ldots\circ\circ\circ\ldots\rangle\equiv|0\rangle$. We use the infinite time evolving block decimation (iTEBD) method, which provides results valid in the thermodynamic limit up to some finite time [30]. The bond dimension used is 400, which limits the evolution time to $t\sim30$.

Figure 2a reveals linear growth of entanglement entropy evaluated for the midpoint bipartition for all considered initial states. Yet, the slope of entanglement growth strongly depends on the initial state, with the slowest growth observed when the system is prepared in the period-2 density wave state, $|\mathbb{Z}_2\rangle$, in equation (2). In addition, the entanglement growth has weak oscillations on top of the linear growth. Relative to the magnitude of entropy, the oscillations are most significant for the $|\mathbb{Z}_2\rangle$ initial state. Figure 2b illustrates the oscillations in entanglement by subtracting the linear component. We note that the oscillations are periodic with the period $T_{\mathbb{Z}_2}\approx2.35$, in agreement with ref. [25]. Similarly, periodic oscillations are clearly visible in the local correlation function, $\langle Z_i Z_{i+1}\rangle$ (Fig. 2c). The oscillations that persist for long times when the entanglement light-cone reaches a distance of $\gtrsim20$ sites, as evidenced by the correlation function, are highly unusual. Although experimental work [25] presented a variational ansatz capturing these oscillations, below we demonstrate that the oscillations actually arise due to the existence of special eigenstates within the rest of the many-body spectrum.

### Fig. 2 | Periodic revivals in the dynamics of entanglement entropy and local correlation function.

**Original caption.** **a,** Entanglement entropy for the midpoint bipartition displays linear growth starting from various initial density-wave product states, as well as the fully polarized $|0\rangle=|\ldots\circ\circ\circ\ldots\rangle$ state. **b,c,** For the $|\mathbb{Z}_2\rangle$ initial state the entanglement entropy oscillates around the linear growth with the same frequency as the local correlation functions.

**Image description (added).** Three panels share a horizontal time axis $t$. Panel a plots entropy $S$ for $|0\rangle$ (orange), $|\mathbb{Z}_2\rangle$ (blue), $|\mathbb{Z}_3\rangle$ (green) and $|\mathbb{Z}_4\rangle$ (red); the blue curve grows most slowly. Panel b plots the oscillating residual $\Delta S$ for $|\mathbb{Z}_2\rangle$. Panel c plots $\langle Z_iZ_{i+1}\rangle$ for the same initial state, showing repeated oscillations over the displayed time interval. Vertical guide lines highlight the recurrence pattern.

## Special states

The special eigenstates become clearly visible when one arranges the entire many-body spectrum according to the overlap with the density-wave $|\mathbb{Z}_2\rangle$ state, as shown in Fig. 3a. This reveals the ‘$\mathbb{Z}_2$-band’ of special eigenstates, which are distinguished by atypically high overlaps with the $|\mathbb{Z}_2\rangle$ product state. The energy separation between states stays approximately constant near the centre of the band and is given by $\Omega\approx1.33$. This energy separation matches half the frequency of the real-time oscillations observed in the iTEBD numerical simulations in Fig. 2. The factor of two comes from the fact that the measured correlator does not distinguish between the $|\mathbb{Z}_2\rangle$ and $|\mathbb{Z}'_2\rangle$ states.

Next, we show that it is possible to construct accurate approximations to the entire band of special states. This is surprising because the model in equation (1) is not frustration free; hence even its ground state cannot be exactly expressed as a matrix product state with bond dimension equal to 2 (ref. [31]). Remarkably, the eigenstates in the $|\mathbb{Z}_2\rangle$-band can still be accurately described within an effective tight-binding approximation. In this effective description, a ‘site’ will turn out to be a superposition of product states at a fixed Hamming distance $D_{\mathbb{Z}_2}$ from the $|\mathbb{Z}_2\rangle$ product state, which is defined as the minimum number of spin flips required to transform those states into $|\mathbb{Z}_2\rangle$ (see Fig. 1). Despite the apparent simplicity of this representation, special eigenstates have larger than area-law entanglement entropy and cannot be expressed as matrix product states of finite bond dimension, in contrast to the states considered in ref. [32].

We start by splitting the Hamiltonian as $H=H^++H^-$, where we have introduced the operator

$$
H^+=\sum_{i\in\mathrm{even}}P_{i-1}\sigma_i^+P_{i+1}
+\sum_{i\in\mathrm{odd}}P_{i-1}\sigma_i^-P_{i+1}
\tag{3}
$$

with $\sigma_i^+=|\bullet\rangle\langle\circ|$ and $\sigma_i^-=|\circ\rangle\langle\bullet|$. This decomposition reflects the fact that $H^+$ increases $D_{\mathbb{Z}_2}$ by one (similarly, $H^-$ lowers $D_{\mathbb{Z}_2}$). To derive the tight-binding model, we want to construct a basis $\{|n\rangle\}$ in which the Hamiltonian in equation (1) is tridiagonal. We proceed iteratively, starting from the initial state $|0\rangle=|\mathbb{Z}_2\rangle$, and propagating forward by the application of $H^+$; that is, $|n\rangle=(H^+)^n|\mathbb{Z}_2\rangle/\|(H^+)^n|\mathbb{Z}_2\rangle\|$. We dub this method the ‘forward scattering approximation’ (FSA). As suggested by the notation, the basis vectors of the FSA can be labelled by the Hamming distance $n$ from the initial $|\mathbb{Z}_2\rangle$ state. This is possible because, for the model in equation (1) and the chosen initial state, there is no intersection between the product state supports of different FSA vectors $|n\rangle$, $|m\rangle$ for $m\ne n$. Another special feature, due to the fact that the Hamming distance must be $n\le L$, is that the FSA basis must contain exactly $L+1$ vectors.

The FSA results in a tridiagonal matrix which is the effective tight-binding Hamiltonian that describes the band of special states in Fig. 3a,

$$
H_{\mathrm{FSA}}=\sum_{n=0}^{L}\beta_n\bigl(|n\rangle\langle n+1|+\mathrm{h.c.}\bigr)
\tag{4}
$$

where h.c. stands for Hermitian conjugate and the hopping amplitude is given by

$$
\beta_n=\langle n+1|H^+|n\rangle=\langle n|H^-|n+1\rangle
\tag{5}
$$

Although equations (4) and (5) are formally reminiscent of the Lanczos recurrence, in the latter case the initial vector can be arbitrary and the propagation is instead performed by the full Hamiltonian. In the Methods section we discuss the approximation involved in the FSA and estimate the associated errors to be less than 1% for chains of $L=32$ sites.

Finally, we compare the eigenstates of $H_{\mathrm{FSA}}$ with exact eigenstates from the special band obtained numerically in the $L=32$ chain with PBC. In Fig. 3b we observe that the lowest-energy special state has exactly the same overlaps with the basis states $|n\rangle$ as the FSA eigenstate. For the special eigenstates in the middle of the many-body band, such as the one shown in Fig. 3c, the FSA overestimates the overlap, yet captures the oscillations. The agreement between the FSA and exact eigenstates is highly surprising, and it further supports the unusual nature of the special eigenstates. Indeed, a basis that has only $L+1$ states, each concentrated in small parts of the Hilbert space, would provide an extremely poor approximation for a generic highly excited eigenstate of a thermalizing system of size $L$.

### Fig. 3 | Scarred many-body states in the FSA approximation.

**Original caption.** **a,** Scatter plot of the overlap of many-body eigenstates of the Hamiltonian equation (1) with the $|\mathbb{Z}_2\rangle$ product state reveals a band of special eigenstates separated from the remaining eigenstates. Crosses denote overlaps with eigenstates from the FSA approximation, which agree very well with exact results. The density of data points (shown in the middle of the graph) illustrates the tower structure in the overlaps. **b,c,** Squared overlap between the basis vectors of the FSA approximation $|n\rangle$ and the exact eigenstates (black) or approximate FSA eigenstates (red) for the ground state (**b**) and for the state in the special band adjacent to energy $E=0$ (**c**). **d,** Participation ratios of special eigenstates decay parametrically slower compared to the average participation ratio of all states within the same energy range. Dashed line shows the inverse Hilbert space dimension. All data are for $L=32$ in the inversion-symmetric, zero-momentum sector (equivalent results are obtained in the other symmetry sector where $|\mathbb{Z}_2\rangle$ has support).

**Image description (added).** Panel a plots $\log|\langle\mathbb{Z}_2|\psi\rangle|^2$ against energy $E$. The special states form a sparse upper envelope above a dense tower-like cloud; red FSA crosses follow that envelope. Panels b and c plot $L|\langle n|\psi\rangle|^2$ against FSA index $n$: the ground-state curves nearly coincide, while the near-zero-energy curves oscillate and differ in amplitude. Panel d uses a logarithmic vertical scale for $\langle\mathrm{PR}_2\rangle$ versus system size $L$; blue triangles denote the special band, orange squares other states, and a grey dashed line $1/\mathcal{D}_{0+}$. The special-band curve decreases much more slowly. Although the caption says “All data are for $L=32$”, panel d explicitly varies $L$.

To provide further insights into the structure of special eigenstates, we study their participation ratios in the product state basis. The second participation ratio, $\mathrm{PR}_2$, of the eigenstate $|\psi\rangle$ is defined as a sum of all wavefunction coefficients, $\mathrm{PR}_2=\sum_\alpha|\langle\alpha|\psi\rangle|^4$, where $\alpha$ labels all distinct product states in the inversion symmetric zero-momentum sector. For ergodic states, one expects that $\mathrm{PR}_2$ decreases as the inverse Hilbert space dimension of the corresponding sector. This is indeed what we observe in Fig. 3d for $\langle\mathrm{PR}_2\rangle_{\mathrm{av}}$ averaged over all eigenstates (with $E\ne0$, see below) in the middle two-thirds of the full energy band. At the same time, $\mathrm{PR}_2$ averaged over special eigenstates from the same energy interval also decreases exponentially with the system size, yet being exponentially enhanced compared to $\langle P_2\rangle_{\mathrm{av}}$. The exponential enhancement of $\mathrm{PR}_2$ is evident in Fig. 3d and its persistence for chains of up to $L=32$ sites provides strong evidence for the existence of special states even in the thermodynamic limit.

In addition, the enhancement of the participation ratio for special eigenstates evidences their concentration in some subregions of the Hilbert space. This suggests that special eigenstates are equivalent to the many-body version of quantum scarred wavefunctions that concentrate in the vicinity of unstable periodic classical orbits in the single-particle quantum chaos. The success of the FSA in yielding a good approximation to special eigenstates shows that the ‘special trajectory’ in the present case connects the two different charge density wave states, $\mathbb{Z}_2$ and $\mathbb{Z}'_2$. This provides a natural explanation for the unusual dynamics observed in Fig. 2 and in experiment [25], including the revivals in the many-body fidelity starting from the $\mathbb{Z}_2$ initial state shown in Supplementary Information. Finally, we have found evidence that other states, for example, $|\bullet\circ\circ\bullet\circ\circ\ldots\rangle$, lead to similar behaviour to the $\mathbb{Z}_2$ state, although at present it is not clear how to determine all such configurations.

## Level statistics and zero modes

Above we suggested a connection between special states, non-ergodic dynamics and quantum scars. However, an alternative explanation for the unusual behaviour observed could be some proximate integrability [33,34]. Indeed, the Hamiltonian equation (1) can be seen as a deformation of the ‘golden chain’ introduced in ref. [15], which is Yang–Baxter integrable. However, the present model cannot be treated as a weak deformation of the ‘golden chain’ [15], as the two Hamiltonians differ by several terms with $O(1)$ coefficients.

To investigate possible integrability, in Fig. 4 we studied the level statistics of the model in equation (1), which is a common diagnostic in the context of both single-particle and many-body quantum systems [35,36]. Fig. 4 reveals that even for relatively small system size, there is a pronounced level repulsion and the distribution of the energy level spacing is close to the semi-Poisson distribution, $P(s)=4se^{-2s}$ (ref. [37]), characterized by the level repulsion at $s\to0$ and an exponential tail. This is in sharp contrast with integrable systems, which always have Poisson level statistics. Moreover, on increasing the system size to $L=32$, we observe that the level statistics steadily approaches the Wigner–Dyson distribution. The Wigner–Dyson level statistics, along with ballistic growth of entanglement, rule out the integrability-based explanation of the non-ergodic dynamics in the model in equation (1).

Another feature prominent in the inset of Fig. 4 is the peak in the otherwise Gaussian density of states. This peak is caused by a large number of states annihilated by the Hamiltonian, $H|\psi\rangle=0$, which form a degenerate subspace of zero modes. We find that the number of zero modes is given by a Fibonacci number; that is, their number grows exponentially with system size. For open boundaries and even system size $L$, the zero energy degeneracy is given by $\mathcal{Z}_L=F_{L/2+1}$. When $L$ is instead odd, we have instead $\mathcal{Z}_L=F_{(L-1)/2}$. In the Methods section we demonstrate that the emergence of zero modes and their count can be understood from the interplay between inversion symmetry and sublattice structure present in the graph representation of the Hamiltonian (Fig. 1). This additional special property of the model in equation (1) appears to be compatible with the presence of scars: as can be seen in Fig. 3a, one of the special states is also a zero mode.

We note that zero modes are stable with respect to introducing potential energy as long as it commutes with particle-hole symmetry $\mathcal{P}$ and anticommutes with the inversion operator. An example of such potential energy is provided by the staggered chemical potential, $\sum_{i=1}^{L}(-)^iZ_i$, which does not change the number of zero modes in the model in equation (1) with OBC for even system sizes. Although our symmetry arguments allow us to enumerate all zero modes, the complete understanding of their algebraic properties and the precise relation to scars appear to be much more difficult problems.

### Fig. 4 | Level statistics and zero modes in the Fibonacci chain.

**Original caption.** Level statistics $P(s)$ interpolates between semi-Poisson (SP) and Wigner–Dyson (WD) distributions with increasing system size. Data are inconsistent with the Poisson statistics (P). $P(s)$ is calculated for the zero-momentum inversion-symmetric ($0+$) sector of the model in equation (1) from the unfolded energy levels $E_i$ in the bulk of the spectrum excluding the zero modes, that is, for $i\in[|\mathcal{D}_{0+}\rangle/5,|\mathcal{D}_{0+}\rangle/2-500]$. The inset shows the density of states for $L=32$, illustrating that it has a Gaussian form without any anomalies, except for the spike at $E=0$ due to zero modes.

**Image description (added).** The main plot shows the spacing distribution $P(s)$ against unfolded spacing $s$. Orange squares, green circles and blue triangles represent $L=28,30,32$, respectively. Reference curves are blue dashed (Poisson), red dash-dotted (semi-Poisson), and black dashed (Wigner–Dyson). The finite-size data show level repulsion near $s=0$ and move toward the Wigner–Dyson curve as $L$ increases. An inset plots $\rho(E)$ against $E$, showing a broad, approximately Gaussian density with a narrow central zero-energy spike.

## Discussion

In summary, we have demonstrated the weak breakdown of eigenstate thermalization in the Fibonacci chain. This breakdown is associated with a band of special eigenstates that we identified as ‘quantum many-body scars’. These eigenstates are analogues of single-particle chaotic wavefunctions, but with scars concentrated in parts of the Hilbert space. Moreover, as shown transparently by our tight-binding method, scars can be experimentally probed by initializing the system in special states, such as $|\bullet\circ\bullet\circ\ldots\rangle$ and $|\circ\bullet\circ\bullet\ldots\rangle$. The ensuing quantum dynamics then remains concentrated on a very specific subset of the Hilbert space, giving rise to robust oscillations even in very large systems.

We emphasize that our findings are qualitatively different from previous proposals of many-body localization in translation-invariant models [22,38–48], where the potential energy was designed to make most of the processes off-resonant, leading to ergodicity breaking. In contrast, the model in equation (1) does not have any potential energy, and features ballistic propagation of entanglement. Our study thus suggests the existence of a new universality class of quantum dynamics, which is neither fully thermalizing nor many-body localized, and which we attribute to the presence of a local dynamical constraint. This opens many exciting research directions that could lead to a better understanding of weakly non-ergodic systems. In particular, the analogy with quantum scars should be put on a firmer footing. This requires a more rigorous generalization of the concept of ‘trajectory’ (in the sense of single-particle quantum scars) to the many-body case. Full classification and understanding of all such trajectories and their ‘parent states’ also remains an open problem. The FSA presented here may be regarded as a first step in these directions. Another open question concerns the precise relationship between quantum scars and other unusual aspects of the model, such as the existence of zero modes. We have demonstrated the compatibility of these two phenomena in the model of equation (1), but we expect scars to be more generic than the zero modes, as their existence does not appear to rely crucially on a symmetry of the model. In Supplementary Information we illustrate the stability of scarred eigenstates to various perturbations.

While the questions formulated above may be addressed in the context of the specific model of equation (1), our work motivates the search for similar behaviour in different kinetically constrained models. It would be highly desirable to understand the features of constrained models that give rise to the non-ergodic dynamics. In particular, the model of equation (1) is a ‘projection’ of a trivial paramagnet Hamiltonian, thus the projection of other non-interacting models might be a promising direction in such a search. Provided one identifies a broader class of ‘quantum scarred’ models, these can be used to engineer many-body states with long coherence times in the existing quantum simulators [49,50]. An obvious direction is to exploit the compatibility between scars and zero modes, which would provide an exceptionally long-lived storage and possible manipulation of quantum information. Such endeavours would hopefully also uncover the interesting connections between quantum dynamics and the spectral theory of graphs, and lead to a better understanding of different universality classes of thermalizing systems.

Received: 20 November 2017; Accepted: 13 March 2018; Published online: 14 May 2018

## References

1. Kinoshita, T., Wenger, T. & Weiss, D. S. A quantum Newton’s cradle. Nature 440, 900–903 (2006).

2. Schreiber, M. et al. Observation of many-body localization of interacting fermions in a quasirandom optical lattice. Science 349, 842–845 (2015).

3. Smith, J. et al. Many-body localization in a quantum simulator with programmable random disorder. Nat. Phys. 12, 907–911 (2016).

4. Kucsko, G. et al. Critical thermalization of a disordered dipolar spin system in diamond. Preprint at [http://arxiv.org/abs/1609.08216](http://arxiv.org/abs/1609.08216) (2016).

5. Sutherland, B. Beautiful Models: 70 Years of Exactly Solved Quantum Many-body Problems (World Scientific, River Edge, NJ, 2004).

6. Basko, D., Aleiner, I. & Altshuler, B. Metal–insulator transition in a weakly interacting many-electron system with localized single-particle states. Ann. Phys. 321, 1126–1205 (2006).

7. Serbyn, M., Papić, Z. & Abanin, D. A. Local conservation laws and the structure of the many-body localized states. Phys. Rev. Lett. 111, 127201 (2013).

8. Huse, D. A., Nandkishore, R. & Oganesyan, V. Phenomenology of fully many-body-localized systems. Phys. Rev. B 90, 174202 (2014).

9. Deutsch, J. M. Quantum statistical mechanics in a closed system. Phys. Rev. A. 43, 2046 (1991).

10. Srednicki, M. Chaos and quantum thermalization. Phys. Rev. E 50, 888–901 (1994).

11. Rigol, M., Dunjko, V., Yurovsky, V. & Olshanii, M. Relaxation in a completely integrable many-body quantum system: An ab initio study of the dynamics of the highly excited states of 1D lattice hard-core bosons. Phys. Rev. Lett. 98, 050405 (2007).

12. Rigol, M., Dunjko, V. & Olshanii, M. Thermalization and its mechanism for generic isolated quantum systems. Nature 452, 854–858 (2008).

13. Kim, H., Ikeda, T. N. & Huse, D. A. Testing whether all eigenstates obey the eigenstate thermalization hypothesis. Phys. Rev. E 90, 052105 (2014).

14. Read, N. & Rezayi, E. Beyond paired quantum Hall states: Parafermions and incompressible states in the first excited Landau level. Phys. Rev. B 59, 8084–8092 (1999).

15. Feiguin, A. et al. Interacting anyons in topological quantum liquids: The Golden Chain. Phys. Rev. Lett. 98, 160409 (2007).

16. Lesanovsky, I. & Katsura, H. Interacting Fibonacci anyons in a Rydberg gas. Phys. Rev. A. 86, 041601 (2012).

17. Lindner, N. H., Berg, E., Refael, G. & Stern, A. Fractionalizing Majorana fermions: Non-Abelian statistics on the edges of Abelian quantum Hall states. Phys. Rev. X 2, 041002 (2012).

18. Glaetzle, A. W. et al. Quantum spin-ice and dimer models with Rydberg atoms. Phys. Rev. X 4, 041037 (2014).

19. Vasseur, R., Potter, A. C. & Parameswaran, S. A. Quantum criticality of hot random spin chains. Phys. Rev. Lett. 114, 217201 (2015).

20. Chandran, A., Schulz, M. D. & Burnell, F. J. The eigenstate thermalization hypothesis in constrained Hilbert spaces: A case study in non-Abelian anyon chains. Phys. Rev. B 94, 235122 (2016).

21. Lan, Z. & Powell, S. Eigenstate thermalization hypothesis in quantum dimer models. Phys. Rev. B 96, 115140 (2017).

22. Lan, Z., van Horssen, M., Powell, S. & Garrahan, J. P. Quantum slow relaxation and metastability due to dynamical constraints. Preprint at [http://arxiv.org/abs/1706.02603](http://arxiv.org/abs/1706.02603) (2017).

23. Schauß, P. et al. Observation of spatially ordered structures in a two-dimensional Rydberg gas. Nature 491, 87–91 (2012).

24. Labuhn, H. et al. Tunable two-dimensional arrays of single Rydberg atoms for realizing quantum Ising models. Nature 534, 667–670 (2016).

25. Bernien, H. et al. Probing many-body dynamics on a 51-atom quantum simulator. Nature 551, 579–584 (2017).

26. Heller, E. J. Bound-state eigenfunctions of classically chaotic Hamiltonian systems: Scars of periodic orbits. Phys. Rev. Lett. 53, 1515–1518 (1984).

27. Sridhar, S. Experimental observation of scarred eigenfunctions of chaotic microwave cavities. Phys. Rev. Lett. 67, 785–788 (1991).

28. Marcus, C. M., Rimberg, A. J., Westervelt, R. M., Hopkins, P. F. & Gossard, A. C. Conductance fluctuations and chaotic scattering in ballistic microstructures. Phys. Rev. Lett. 69, 506–509 (1992).

29. Sun, B. & Robicheaux, F. Numerical study of two-body correlation in a 1D lattice with perfect blockade. New. J. Phys. 10, 045032 (2008).

30. Vidal, G. Classical simulation of infinite-size quantum lattice systems in one spatial dimension. Phys. Rev. Lett. 98, 070201 (2007).

31. Lesanovsky, I. Liquid ground state, gap, and excited states of a strongly correlated spin chain. Phys. Rev. Lett. 108, 105301 (2012).

32. Moudgalya, S., Rachel, S., Bernevig, B. A. & Regnault, N. Exact excited states of non-integrable models. Preprint at [http://arxiv.org/abs/1708.05021](http://arxiv.org/abs/1708.05021) (2017).

33. Fendley, P., Sengupta, K. & Sachdev, S. Competing density-wave orders in a one-dimensional hard-boson model. Phys. Rev. B 69, 075106 (2004).

34. Fendley, P. Strong zero modes and eigenstate phase transitions in the XYZ/ interacting Majorana chain. J. Phys. A 49, 30LT01 (2016).

35. Berry, M. V. & Tabor, M. Level clustering in the regular spectrum. Proc. R. Soc. A 356, 375–394 (1977).

36. Pal, A. & Huse, D. A. Many-body localization phase transition. Phys. Rev. B 82, 174411 (2010).

37. Bogomolny, E. B., Gerland, U. & Schmit, C. Models of intermediate spectral statistics. Phys. Rev. E 59, R1315–R1318 (1999).

38. Carleo, G., Becca, F., Schiró, M. & Fabrizio, M. Localization and glassy dynamics of many-body quantum systems. Sci. Rep. 2, 243 (2012).

39. De Roeck, W. & Huveneers, F. Asymptotic quantum many-body localization from thermal disorder. Commun. Math. Phys. 332, 1017–1082 (2014).

40. Schiulaz, M. & Müller, M. Ideal quantum glass transitions: Many-body localization without quenched disorder. AIP Conf. Proc. 1610, 11–23 (2014).

41. Yao, N. Y., Laumann, C. R., Cirac, J. I., Lukin, M. D. & Moore, J. E. Quasi-many-body localization in translation-invariant systems. Phys. Rev. Lett. 117, 240601 (2016).

42. van Horssen, M., Levi, E. & Garrahan, J. P. Dynamics of many-body localization in a translation-invariant quantum glass model. Phys. Rev. B 92, 100305 (2015).

43. Veness, T., Essler, F. H. L. & Fisher, M. P. A. Quantum disentangled liquid in the half-filled Hubbard model. Phys. Rev. B 96, 195153 (2017).

44. Kim, I. H. & Haah, J. Localization from superselection rules in translationally invariant systems. Phys. Rev. Lett. 116, 027202 (2016).

45. Yarloo, H., Langari, A. & Vaezi, A. Anyonic self-induced disorder in a stabilizer code: Quasi many-body localization in a translational invariant model. Phys. Rev. B 97, 054304 (2018).

46. Michailidis, A. A. et al. Slow dynamics in translation-invariant quantum lattice models. Phys. Rev. B 97, 104307 (2018).

47. Smith, A., Knolle, J., Moessner, R. & Kovrizhin, D. L. Absence of ergodicity without quenched disorder: From quantum disentangled liquids to many-body localization. Phys. Rev. Lett. 119, 176601 (2017).

48. Brenes, M., Dalmonte, M., Heyl, M. & Scardicchio, A. Many-body localization dynamics from gauge invariance. Phys. Rev. Lett. 120, 030601 (2018).

49. Bloch, I., Dalibard, J. & Nascimbène, S. Quantum simulations with ultracold quantum gases. Nat. Phys. 8, 267–276 (2012).

50. Blatt, R. & Roos, C. F. Quantum simulations with trapped ions. Nat. Phys. 8, 277–284 (2012).

## Acknowledgements

We acknowledge insightful discussions with M. Lukin and W.W. Ho. C.J.T., A.M. and Z.P. acknowledge support from EPSRC grants EP/P009409/1 and EP/M50807X/1, and Royal Society Research Grant RG160635. D.A. acknowledges support from the Swiss National Science Foundation. This work was initiated during ‘Conference on Many-Body-Localization: Advances in the Theory and Experimental Progress’ at ICTP Trieste.

## Author contributions

All authors contributed to developing the ideas, analysing the results and writing the manuscript. C.J.T., A.A.M. and M.S. performed the numerical simulations.

## Competing interests

The authors declare no competing interests.

## Additional information

**Supplementary information** is available for this paper at [https://doi.org/10.1038/s41567-018-0137-5](https://doi.org/10.1038/s41567-018-0137-5).

**Reprints and permissions information** is available at [www.nature.com/reprints](https://www.nature.com/reprints).

**Correspondence and requests for materials** should be addressed to Z.P.

**Publisher’s note:** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

## Methods

### FSA and error analysis

To better understand the approximation involved in the FSA, we compare it with the Lanczos algorithm [51], where the initial vector can be arbitrary and the propagation is instead performed by the full Hamiltonian, $H^++H^-$. By contrast, equation (5) implies that in the exact FSA the backward propagation results in a previous vector in the basis, $H^-|n+1\rangle=\beta_n|n\rangle$. Although this holds exactly for the non-interacting Hamiltonian $\sum_i X_i$, the presence of projectors makes this relation approximate. To quantify the error per iteration of the FSA we take the difference of the two terms and multiply from the left with $\langle n|H^+$ to express the error as a scalar quantity. The error function can be written as

$$
\operatorname{err}(n)=\left|\frac{\langle n|H^+H^-|n\rangle}{\beta_n^2}-1\right|
\tag{6}
$$

where $\operatorname{err}(n)=0$ is equivalent to $H^-|n+1\rangle=\beta_n|n\rangle$. The division with $\beta_n^2$ makes the error independent of the magnitude of $\beta_n$. Numerically we find that $\operatorname{err}(n)\approx0.2\%$ for $L=32$ and has a decreasing trend as we increase the system size, which is promising in terms of scaling the method to the thermodynamic limit. As an additional error measure, the average energy difference between the exact eigenstates in the $\mathbb{Z}_2$-band and the eigenstates of $H_{\mathrm{FSA}}$ for $L=32$ is $\overline{\Delta E}/E\approx1\%$, which further supports the accuracy of the FSA scheme.

### Derivation of the zero mode count

To derive the zero mode count and to understand its origin, we reformulate the problem of finding eigenstates as a hopping problem on a graph with vertices corresponding to product states in the constrained Hilbert space. An example of such a graph for $L=6$ was illustrated in Fig. 1. The quantum many-body problem becomes equivalent to a single-particle hopping on this graph. Since the application of any term in the Hamiltonian changes the number of excitations ($\bullet$) by $\pm1$, the graph has a bipartite structure with even/odd sublattices corresponding to the product states with an even/odd number of excitations.

It is well known that bipartite lattices support zero modes, with their number lower-bounded by the difference between the number of sites in two sublattices [52,53]. However, in the present case it is crucial to account for inversion symmetry $I$, which splits the Hilbert space into two sectors with $I=\pm1$ and is compatible with a bipartite structure. The inversion-odd sector does not include any inversion-invariant product states, for example, the basis state $\circ\bullet\circ\circ\bullet\circ$ does not belong to the $I=-1$ sector of the $L=6$ chain. When the system size $L$ is even, all inversion-symmetric states have an even number of excitations, and the number of such states is given by $\mathcal{Z}_L=F_{L/2+1}$, where $F_n$ denotes the $n$th Fibonacci number. Hence in the $I=-1$ sector the even sublattice of the graph will have a deficit of inversion-symmetric states, leading to the presence of $\mathcal{Z}_L$ zero modes. In this way, we can directly classify the zero-mode states for other cases. For example, when $L$ is odd, the number of zero modes is $\mathcal{Z}_L=F_{(L-1)/2}$. The counting of zero modes for PBC can be determined similarly.

Interestingly, the many-body wavefunctions of the zero mode eigenstates may be chosen such that they all have integer coefficients in the product state basis (modulo overall normalization). This might be anticipated from the fact that the Hamiltonian matrix elements are all integers, and the solutions to the zero mode condition can be obtained by Gaussian elimination. Nevertheless, the decomposition into integers is suggestive of the existence of a recursive relation, which bears curious similarity to the Jack polynomials that appear in the Calogero–Sutherland model [54] and the fractional quantum Hall effect [55]. In the latter case, model Hamiltonians can be defined that enforce local constraints in direct analogy with equation (1), and the quasihole excitations (whose number grows exponentially, determined by the constrained Hilbert space) appear as exact zero modes of the Hamiltonian, with integer coefficients in the product state basis. Although we do not believe there are direct physical similarities between these models, it would be interesting to elucidate their mathematical relations.

### Data availability

The data that support the plots within this paper and other findings of this study are available at [https://doi.org/10.5518/335](https://doi.org/10.5518/335).

## References (Methods)

51. Lanczos, C. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. J. Res. Natl Bur. Stand. 45, 255–282 (1950).

52. Sutherland, B. Localization of electronic wave functions due to local topology. Phys. Rev. B 34, 5208–5211 (1986).

53. Inui, M., Trugman, S. A. & Abrahams, E. Unusual properties of midband states in systems with off-diagonal disorder. Phys. Rev. B 49, 3190–3196 (1994).

54. Sutherland, B. Exact results for a quantum many-body problem in one dimension. Phys. Rev. A. 4, 2019–2021 (1971).

55. Bernevig, B. A. & Haldane, F. D. M. Model fractional quantum Hall states and Jack polynomials. Phys. Rev. Lett. 100, 246802 (2008).

---

## Editorial notes on transcription (not part of the article)

1. Equations (1)–(6) were checked against the rendered PDF. Equation (4) retains the printed upper limit $L$, although the stated basis contains $L+1$ vectors indexed $0,\ldots,L$. Equation (6) retains the printed denominator $\beta_n^2$ and numerator $\langle n|H^+H^-|n\rangle$; its indexing is not silently reconciled with the preceding recurrence. These are source-level issues, not PDF-extraction substitutions.
2. The source uses $|0\rangle$ both for the fully polarized state in the dynamics discussion and for the initial FSA basis vector $|\mathbb{Z}_2\rangle$ later on. Both usages are preserved.
3. The Fig. 3 caption's symmetry-sector wording is preserved, including its parenthetical statement about the other sector. The caption's “All data are for $L=32$” is also retained, even though panel d plots several sizes.
4. The Fig. 4 caption visibly prints ket-like delimiters around $\mathcal{D}_{0+}$ in its index interval. These are retained in the original-caption transcription. The dimension-based interval is naturally read as $i\in[\mathcal{D}_{0+}/5,\mathcal{D}_{0+}/2-500]$.
5. The source prints $\langle P_2\rangle_{\mathrm{av}}$ once where the surrounding discussion uses $\langle\mathrm{PR}_2\rangle_{\mathrm{av}}$; the printed expression is retained. Reference 35's missing opening parenthesis before 1977 and obvious missing spaces in journal/volume fields have been repaired. Reference details otherwise follow the supplied PDF and have not been independently checked against external bibliographic databases.
