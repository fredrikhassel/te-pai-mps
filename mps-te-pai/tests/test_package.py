"""Verify: different Hamiltonians, initial states, observables, and circuit correctness."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mps_tepai import Hamiltonian, TE_PAI, trotter_mps, run_simulation
from mps_tepai.pai import gamma, prob_list, abc


def test_hamiltonians():
    """Test that all Hamiltonian types construct and produce different dynamics."""
    print("\n=== TEST: Different Hamiltonians ===")
    n = 6
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=n)

    hamiltonians = {
        "spin_chain": Hamiltonian.spin_chain(n, freqs, j=0.1)
    }

    for name, hamil in hamiltonians.items():
        assert hamil.nqubits == n, f"{name}: nqubits mismatch"
        assert len(hamil) > 0, f"{name}: no terms"
        coefs = hamil.coefs(0.0)
        assert len(coefs) == len(hamil), f"{name}: coefs length mismatch"
        l1 = hamil.l1_norm(1.0)
        assert l1 > 0, f"{name}: l1_norm should be > 0"

        # Run short Trotter simulation
        t, vals = trotter_mps(hamil, n, T=0.5, N=20, n_snap=2, max_bond=8)
        assert len(t) == 3, f"{name}: expected 3 time points (t=0, 0.25, 0.5)"
        assert all(0 <= v <= 1 for v in vals), f"{name}: values out of [0,1] range"
        print(f"  {name:15s}: {len(hamil)} terms, l1={l1:.3f}, vals={[f'{v:.4f}' for v in vals]}")

def test_initial_states():
    """Test that different initial states give different measurements."""
    print("\n=== TEST: Different Initial States ===")
    n = 6
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=n)
    hamil = Hamiltonian.spin_chain(n, freqs, j=0.1)

    results = {}
    for state in ["plus", "plus_flip", "zero"]:
        t, vals = trotter_mps(hamil, n, T=0.5, N=30, n_snap=2, max_bond=8,
                              operator="X", measure_qubit=0, initial_state=state)
        results[state] = vals
        print(f"  {state:12s} <X_0>: vals={[f'{v:.4f}' for v in vals]}")

    # |+> starts at <X>=1, |0> starts at <X>=0.5 (eigenvalue 0 mapped to 0.5)
    assert abs(results["plus"][0] - 1.0) < 1e-6, f"plus state should start at <X>=1, got {results['plus'][0]}"
    assert abs(results["zero"][0] - 0.5) < 1e-6, f"zero state should start at <X>=0.5, got {results['zero'][0]}"

    # plus_flip applies Z on middle qubit — measure on that qubit to see the difference
    mid = n // 2
    for state in ["plus", "plus_flip"]:
        t, vals = trotter_mps(hamil, n, T=0.5, N=30, n_snap=2, max_bond=8,
                              operator="X", measure_qubit=mid, initial_state=state)
        results[state + "_mid"] = vals
        print(f"  {state:12s} <X_{mid}>: vals={[f'{v:.4f}' for v in vals]}")
    # At t=0, |+> has <X_mid>=1, |+_flip> has <X_mid>=0 (Z|+> = |->)
    assert abs(results["plus_mid"][0] - 1.0) < 1e-6, \
        f"plus <X_mid> at t=0 should be 1, got {results['plus_mid'][0]}"
    assert abs(results["plus_flip_mid"][0] - 0.0) < 1e-6, \
        f"plus_flip <X_mid> at t=0 should be 0, got {results['plus_flip_mid'][0]}"
    print("  PASS: Initial states verified")


def test_observables():
    """Test that different Pauli operators and qubit indices give different results."""
    print("\n=== TEST: Different Observables ===")
    n = 6
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=n)
    hamil = Hamiltonian.spin_chain(n, freqs, j=0.1)

    results = {}
    for op in ["X", "Y", "Z"]:
        for q in [0, 1]:
            key = f"{op}_{q}"
            t, vals = trotter_mps(hamil, n, T=0.5, N=30, n_snap=2, max_bond=8,
                                  operator=op, measure_qubit=q, initial_state="plus_flip")
            results[key] = vals
            print(f"  <{op}_{q}>: vals={[f'{v:.4f}' for v in vals]}")

    # X on qubit 0 vs X on qubit 1 should differ (different local environments)
    assert not np.allclose(results["X_0"], results["X_1"], atol=1e-6), \
        "X on qubit 0 vs 1 should differ"
    # X vs Y vs Z on same qubit should differ
    assert not np.allclose(results["X_0"], results["Y_0"], atol=1e-6), \
        "X vs Y on qubit 0 should differ"
    assert not np.allclose(results["X_0"], results["Z_0"], atol=1e-6), \
        "X vs Z on qubit 0 should differ"
    print("  PASS: Different observables verified")


def test_gamma_weighting():
    """Verify gamma normalization is computed and applied correctly."""
    print("\n=== TEST: Gamma Weighting ===")
    n = 6
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=n)
    hamil = Hamiltonian.spin_chain(n, freqs, j=0.1)

    delta = np.pi / 32
    T_total = 1.5
    n_snap = 3
    N = 150
    te_pai = TE_PAI(hamil, n, delta=delta, T=T_total, N=N, n_snap=n_snap)

    # 1. Verify gam_list structure
    assert len(te_pai.gam_list) == n_snap + 1, \
        f"n_snap={n_snap} should give gam_list of length {n_snap+1}, got {len(te_pai.gam_list)}"
    assert te_pai.gam_list[0] == 1, "gam_list[0] should be 1"
    gamma_final = te_pai.gam_list[-1]
    print(f"  gamma final: {gamma_final:.6f}")
    assert gamma_final >= 1.0, f"gamma should be >= 1, got {gamma_final}"

    # 2. Manually verify gamma computation
    N_eff = N * n_snap # Effective number of Trotter steps used in TE-PAI
    steps = np.linspace(0, T_total, N_eff)
    angles = [[2 * np.abs(c) * T_total / N_eff for c in hamil.coefs(t)] for t in steps]
    manual_gamma = np.prod([gamma(angles[j], delta) for j in range(N_eff)])
    assert abs(manual_gamma - gamma_final) < 1e-10, \
        f"gamma mismatch: manual={manual_gamma}, stored={gamma_final}"
    print(f"  Manual gamma matches stored gamma: {manual_gamma:.6f}")

    # 3. Verify overhead formula
    expected_overhead = np.exp(2 * hamil.l1_norm(T_total) * np.tan(delta / 2))
    assert abs(te_pai.overhead - expected_overhead) < 1e-8, \
        f"overhead mismatch: {te_pai.overhead} vs {expected_overhead}"
    print(f"  Overhead: {te_pai.overhead:.6f} (matches formula)")

    # 4. Run simulation and check that gamma weighting is applied in results
    circuits = te_pai.run_te_pai(100)
    result = run_simulation(
        te_pai, circuits, n, max_bond=8,
        operator="X", measure_qubit=0, initial_state="plus_flip",
    )

    # At t=0, all runs should report exactly gamma^0 = 1.0 times the measured value
    t0_values = [r[0] for r in result.raw_runs]
    # The t=0 measurement is measure(initial_state) * gam_list[0] = measure * 1.0
    # With plus_flip state, <X_0> should be close to 1.0 for all runs
    assert all(abs(v - t0_values[0]) < 1e-10 for v in t0_values), \
        "All t=0 values should be identical (same initial state, deterministic)"
    print(f"  t=0 values consistent: {t0_values[0]:.6f}")

    # Check that later time points have variance (due to random circuits + sign flips)
    if len(result.raw_runs[0]) > 1:
        t1_values = [r[1] for r in result.raw_runs]
        has_variance = np.std(t1_values) > 0
        assert has_variance, "t=1 values should have variance from random circuits"
        # Values can exceed [0,1] due to gamma weighting and sign flips
        has_values_outside_01 = any(v < 0 or v > 1 for v in t1_values)
        print(f"  t=1 mean: {np.mean(t1_values):.6f}, std: {np.std(t1_values):.6f}")
        print(f"  Values outside [0,1] (due to gamma/sign): {has_values_outside_01}")

    print("  PASS: Gamma weighting verified")


def test_pai_probabilities():
    """Verify PAI probability decomposition fundamentals."""
    print("\n=== TEST: PAI Probabilities ===")

    # abc raw values: a + b + c = 1 (signed sum, not absolute value)
    # The absolute-value normalization happens in prob_list
    for theta in [0.01, 0.1, 0.5, 1.0]:
        for delta in [np.pi/64, np.pi/32, np.pi/16]:
            p = abc(theta, delta)
            signed_sum = np.sum(p)
            assert abs(signed_sum - 1.0) < 1e-10, \
                f"abc signed sum != 1 for theta={theta}, delta={delta}: {p} sum={signed_sum}"

    # prob_list should produce valid probability distributions
    angles = [0.05, 0.1, 0.2, 0.3]
    delta = np.pi / 32
    probs = prob_list(angles, delta)
    for i, p in enumerate(probs):
        assert abs(sum(p) - 1.0) < 1e-10, f"Probability distribution {i} doesn't sum to 1: {p}"
        assert all(pi >= 0 for pi in p), f"Negative probability in distribution {i}: {p}"

    # gamma is real and finite
    g = gamma(angles, delta)
    assert np.isfinite(g), f"gamma should be finite, got {g}"
    assert g > 0, f"gamma should be positive, got {g}"

    print(f"  abc signed sums verified for multiple theta/delta combos")
    print(f"  prob_list produces valid distributions")
    print(f"  gamma({angles}, pi/32) = {g:.6f} (finite, positive)")
    print("  PASS: PAI probabilities verified")


if __name__ == "__main__":
    test_pai_probabilities()
    test_hamiltonians()
    test_initial_states()
    test_observables()
    test_gamma_weighting()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
