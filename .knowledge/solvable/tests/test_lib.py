import numpy as np
from _lib import ed, gf2, bethe


def _xxx_ed_energy(N):
    """Brute-force ED ground energy of H = sum_i S_i . S_{i+1} (PBC, S=sigma/2)."""
    sx, sy, sz = ed.spin_ops(N)
    H = sum(sx[i] @ sx[(i + 1) % N] + sy[i] @ sy[(i + 1) % N]
            + sz[i] @ sz[(i + 1) % N] for i in range(N))
    return ed.ground_energy(H)


def test_bethe_xxx_ground_energy_matches_ed():
    # GROUND TRUTH: Bethe-ansatz roots reproduce brute-force ED to 1e-10.
    for N in (8, 10, 12):
        roots = bethe.xxx_ground_roots(N)
        assert len(roots) == N // 2, (N, len(roots))
        e_bethe = bethe.xxx_energy(roots, N)
        e_ed = _xxx_ed_energy(N)
        assert abs(e_bethe - e_ed) < 1e-10, (N, e_bethe, e_ed)


def test_bethe_roots_are_real_and_symmetric():
    for N in (8, 10, 12):
        r = bethe.xxx_ground_roots(N)
        # roots are symmetric about 0 as a SET: {r} == {-r}
        assert np.allclose(np.sort(r), np.sort(-r), atol=1e-12), N
        # roots genuinely solve the logarithmic Bethe equations
        M = N // 2
        I = np.arange(M) - (M - 1) / 2.0
        diff = r[:, None] - r[None, :]
        residual = (2 * np.arctan(2 * r)
                    - 2 * np.pi * I / N
                    - 2 * np.arctan(diff).sum(axis=1) / N)
        assert np.max(np.abs(residual)) < 1e-12, (N, np.max(np.abs(residual)))


def test_ed_spin_tfim_l2():
    # H = -sz.sz - h(sx1+sx2) with PAULIS, h=0: E0 = -1, doubly degenerate
    sx, sy, sz = ed.spin_ops(2)
    H = -4 * (sz[0] @ sz[1]) # Pauli = 2*S
    assert abs(ed.ground_energy(H) + 1.0) < 1e-12
    assert ed.ground_states(H) == 2


def test_gf2_rank():
    assert gf2.rank(np.eye(3, dtype=np.uint8)) == 3
    assert gf2.rank(np.array([[1, 1], [1, 1]], dtype=np.uint8)) == 1
    assert gf2.rank(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)) == 2
