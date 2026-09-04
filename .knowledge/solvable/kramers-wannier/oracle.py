"""Kramers–Wannier duality as a checkable identity (pure cross-load, no engines).

Two faces of the same self-duality of the Ising model, each verified by loading
a sibling oracle — no new physics is computed here.

(i) 1D QUANTUM (TFIM chain, cross-load `tfim-chain`). The transverse-field Ising
    chain H = −J Σ σ^z σ^z − h Σ σ^x has single-fermion dispersion
    ε(k) = 2√(J² + h² − 2Jh cos k), which is SYMMETRIC under the exchange
    (J,h) ↔ (h,J). Hence the ground-state energy obeys e0(J,h) = e0(h,J) exactly,
    at finite L and in the thermodynamic limit. This spectral symmetry is the
    finite-chain shadow of the Kramers–Wannier bond-algebra duality: the duality
    maps σ^z_i σ^z_{i+1} (bond, ordering) to σ^x on the dual site (field,
    disordering) and exchanges the ordered/disordered phases, i.e. J ↔ h. The
    self-dual point h = J is the chain's quantum critical point.

(ii) 2D CLASSICAL (square-lattice Ising, cross-load `ising-2d-onsager`). With the
    dual coupling K* fixed by  sinh(2K) sinh(2K*) = 1, the reduced free energy
    ψ(K) = (1/N) ln Z obeys the exact Kramers–Wannier relation

        ψ(K) − ½ ln sinh(2K)  =  ψ(K*) − ½ ln sinh(2K*)        (KW duality)

    — i.e. g(K) ≡ ψ(K) − ½ ln sinh(2K) is a self-dual invariant. This is DERIVED
    below (see `kw_free_energy_invariant`) and verified against the Onsager
    double-integral ψ(K) at K ∈ {0.3, 0.6} to 1e-8. The self-dual fixed point
    sinh(2K_c) = 1 gives K_c = ½ ln(1+√2), identical to the Onsager card's
    critical temperature via K_c = J/T_c.

Reference: [@KramersWannier1941] (Phys. Rev. 60, 252 — Part I).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402


def _load(sibling):
    p = Path(__file__).resolve().parents[1] / sibling / "oracle.py"
    spec = importlib.util.spec_from_file_location(f"{sibling.replace('-', '_')}_oracle", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- (ii) 2D classical Kramers–Wannier free-energy duality -----------------

def dual_coupling(K):
    """K* fixed by sinh(2K) sinh(2K*) = 1  ⇔  K* = ½ arcsinh(1/sinh 2K)."""
    return 0.5 * np.arcsinh(1.0 / np.sinh(2 * K))


def kw_free_energy_invariant(psi_of_K, K):
    """The self-dual invariant g(K) = ψ(K) − ½ ln sinh(2K).

    Derivation. High-T (polygon) and low-T (domain-wall) expansions of the
    square-lattice partition function are the SAME polygon sum P evaluated at
    tanh K and at e^{-2K} respectively. Writing the dual coupling via
    tanh K* = e^{-2K} (equivalently sinh 2K sinh 2K* = 1) and using the high-T
    form Z(K) = (2 cosh²K)^N P(tanh K) on both sides gives, in the thermodynamic
    limit, ψ(K) − ψ(K*) = ½ ln[sinh(2K)/sinh(2K*)]. Since sinh(2K*) = 1/sinh(2K),
    the right side is ln sinh(2K), so ψ(K) − ½ ln sinh(2K) is invariant under
    K ↔ K*. (Verified numerically here against the Onsager ψ.)
    """
    return psi_of_K(K) - 0.5 * np.log(np.sinh(2 * K))


def kc_selfdual():
    """Self-dual fixed point sinh(2K_c) = 1  ⇒  K_c = ½ ln(1+√2)."""
    return 0.5 * np.log(1.0 + np.sqrt(2.0))


def compute(J=1.0, h=0.7, K=0.3):
    """Kramers–Wannier duality residuals (all should be ~0)."""
    tfim = _load("tfim-chain")
    ons = _load("ising-2d-onsager")
    Ks = dual_coupling(K)
    g = kw_free_energy_invariant(ons.lnZ_per_site, K)
    gs = kw_free_energy_invariant(ons.lnZ_per_site, Ks)
    return {
        "tfim_e0_duality_residual": abs(tfim.e0_thermo(J, h) - tfim.e0_thermo(h, J)),
        "dual_coupling_product": np.sinh(2 * K) * np.sinh(2 * dual_coupling(K)),
        "kw_free_energy_residual": abs(g - gs),
        "Kc_selfdual": kc_selfdual(),
        "Kc_matches_onsager_Tc": abs(kc_selfdual() - 1.0 / ons.tc()),
    }


def self_test():
    tfim = _load("tfim-chain")
    ons = _load("ising-2d-onsager")

    # anchor 1: TFIM spectral duality e0(J,h) == e0(h,J), exact, at three pairs,
    # finite-L and thermodynamic limit.
    for J, h in [(1.0, 0.5), (0.7, 1.3), (2.0, 0.4)]:
        assert abs(tfim.e0_finite(16, J, h) - tfim.e0_finite(16, h, J)) < 1e-12
        assert abs(tfim.e0_thermo(J, h) - tfim.e0_thermo(h, J)) < 1e-12
    # the self-dual point h = J is the chain's critical point (gap closes there).
    assert tfim.compute(h=1.0, J=1.0)["gap_single_fermion"] == 0.0

    # anchor 2: the DERIVED KW free-energy relation holds against Onsager's ψ.
    for K in (0.3, 0.6):
        Ks = dual_coupling(K)
        assert abs(np.sinh(2 * K) * np.sinh(2 * Ks) - 1.0) < 1e-12          # duality curve
        g = kw_free_energy_invariant(ons.lnZ_per_site, K)
        gs = kw_free_energy_invariant(ons.lnZ_per_site, Ks)
        assert abs(g - gs) < 1e-8, (K, g, gs)                              # KW invariant
    # the invariant is non-trivial (K and K* actually differ, ψ actually changes).
    assert abs(dual_coupling(0.3) - 0.3) > 0.1
    assert abs(ons.lnZ_per_site(0.3) - ons.lnZ_per_site(dual_coupling(0.3))) > 0.1

    # anchor 3: self-dual point sinh(2K_c)=1 ⇒ K_c=½ln(1+√2) == Onsager K_c=J/T_c.
    assert abs(np.sinh(2 * kc_selfdual()) - 1.0) < 1e-12
    assert abs(kc_selfdual() - 1.0 / ons.tc()) < 1e-12                     # arithmetic identity


if __name__ == "__main__":
    oracle_main(compute, {"J": (float, 1.0), "h": (float, 0.7), "K": (float, 0.3)})
