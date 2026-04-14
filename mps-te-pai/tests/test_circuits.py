"""TE-PAI circuit structure tests.

Demonstrates:
  1. Gate count saturates as Trotter depth N -> infinity
  2. Gate counts form a normal distribution around their expected value
  3. Effect of Delta on circuit lengths
  4. Gate validity (angles, Paulis, qubit indices) and sign-consistency
  5. Pi-gate prevalence increases with larger Delta
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mps_tepai import Hamiltonian, TE_PAI
from mps_tepai.pai import abc


# ---------- shared setup ----------
def _make_hamil(n=10, seed=0, j=0.1):
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-1, 1, size=n)
    return Hamiltonian.spin_chain(n, freqs, j=j)


def _count_gates(circuits):
    """Return list of per-circuit gate counts."""
    counts = []
    for _, gates_arr in circuits:
        n = sum(len(snap) for snap in gates_arr)
        counts.append(n)
    return counts


def _classify_gates(circuits, delta):
    """Count pi-gates and delta-gates across all circuits."""
    n_pi = 0
    n_delta = 0
    for _, gates_arr in circuits:
        for snap in gates_arr:
            for _, angle, _ in snap:
                if abs(abs(angle) - np.pi) < 1e-10:
                    n_pi += 1
                elif abs(abs(angle) - delta) < 1e-10:
                    n_delta += 1
    return n_pi, n_delta


# =====================================================================
# 1. Gate-count saturation: E[gates] converges as N -> inf
# =====================================================================
def test_gate_count_saturation():
    """The expected gate count converges to ((3-cos(D))/sin(D)) * l1_norm(T)
    as the Trotter depth N grows.  Show that it saturates."""
    print("\n=== TEST: Gate-Count Saturation with Trotter Depth ===")
    hamil = _make_hamil()
    delta = np.pi / 256
    T = 0.5

    analytic = ((3 - np.cos(delta)) / np.sin(delta)) * hamil.l1_norm(T)

    N_values = [10, 50, 100, 500, 1000, 5000]
    print(f"  Analytic limit: {analytic:.4f}")
    print(f"  {'N':>6s}  {'E[gates]':>10s}  {'|error|':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

    prev_expected = None
    for N in N_values:
        te = TE_PAI(hamil, hamil.nqubits, delta=delta, T=T, N=N, n_snap=1)
        err = abs(te.expected_num_gates - analytic)
        print(f"  {N:6d}  {te.expected_num_gates:10.4f}  {err:10.6f}")
        prev_expected = te.expected_num_gates

    # The last value should be very close to the analytic limit
    assert abs(prev_expected - analytic) < 0.01, \
        f"At N={N_values[-1]}, expected_num_gates should converge to analytic limit"
    # Also verify monotone convergence: error shrinks
    errors = []
    for N in N_values:
        te = TE_PAI(hamil, hamil.nqubits, delta=delta, T=T, N=N, n_snap=1)
        errors.append(abs(te.expected_num_gates - analytic))
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i-1] + 1e-10, \
            f"Error should decrease: N={N_values[i-1]}->{N_values[i]}, err={errors[i-1]:.6f}->{errors[i]:.6f}"
    print("  PASS: Gate count saturates to analytic limit")


# =====================================================================
# 2. Gate counts form a normal distribution
# =====================================================================
def test_gate_count_distribution():
    """Sample many circuits, verify gate counts are roughly normally
    distributed around the expected value."""
    print("\n=== TEST: Gate-Count Distribution ===")
    hamil = _make_hamil()
    delta = np.pi / 32
    T = 0.5
    N = 200
    n_circuits = 2000

    te = TE_PAI(hamil, hamil.nqubits, delta=delta, T=T, N=N, n_snap=1)
    circuits = te.run_te_pai(n_circuits)
    counts = np.array(_count_gates(circuits), dtype=float)

    mean = np.mean(counts)
    std = np.std(counts, ddof=1)
    expected = te.expected_num_gates

    print(f"  Expected (analytic):  {expected:.2f}")
    print(f"  Sample mean:          {mean:.2f}")
    print(f"  Sample std:           {std:.2f}")
    print(f"  |mean - expected|:    {abs(mean - expected):.2f}")

    # Mean should be close to expected (within a few standard errors)
    se = std / np.sqrt(n_circuits)
    z = abs(mean - expected) / se
    print(f"  z-score (mean vs expected): {z:.2f}")
    assert z < 5, f"Mean deviates too far from expected: z={z:.2f}"

    # Check normality: fraction within 1-sigma and 2-sigma of the mean
    within_1s = np.mean(np.abs(counts - mean) < std)
    within_2s = np.mean(np.abs(counts - mean) < 2 * std)
    print(f"  Within 1-sigma: {100*within_1s:.1f}% (expected ~68%)")
    print(f"  Within 2-sigma: {100*within_2s:.1f}% (expected ~95%)")
    assert within_1s > 0.55, f"Too few samples within 1-sigma: {within_1s:.2f}"
    assert within_2s > 0.88, f"Too few samples within 2-sigma: {within_2s:.2f}"

    # Histogram summary (text-based)
    bins = np.linspace(counts.min(), counts.max(), 12)
    hist, edges = np.histogram(counts, bins=bins)
    max_bar = max(hist)
    print(f"\n  Gate-count histogram ({n_circuits} circuits):")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / max_bar) if max_bar > 0 else ""
        lo, hi = edges[i], edges[i+1]
        print(f"    [{lo:6.0f}-{hi:6.0f}) {hist[i]:4d} {bar}")

    print("  PASS: Gate counts are normally distributed")


# =====================================================================
# 3. Effect of Delta on circuit lengths
# =====================================================================
def test_delta_effect_on_length():
    r"""Smaller Delta -> more gates (finer rotation granularity).
    Larger Delta -> fewer gates but larger overhead."""
    print("\n=== TEST: Effect of Delta on Circuit Lengths ===")
    hamil = _make_hamil()
    T = 0.5
    N = 200
    n_circuits = 500

    deltas = [np.pi / 128, np.pi / 64, np.pi / 32, np.pi / 16, np.pi / 8, np.pi / 4]
    results = []

    print(f"  {'Delta':>12s}  {'E[gates]':>10s}  {'mean(sampled)':>14s}  {'overhead':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*10}")

    for d in deltas:
        te = TE_PAI(hamil, hamil.nqubits, delta=d, T=T, N=N, n_snap=1)
        circuits = te.run_te_pai(n_circuits)
        counts = _count_gates(circuits)
        mean_count = np.mean(counts)
        results.append((d, te.expected_num_gates, mean_count, te.overhead))
        label = f"pi/{np.pi/d:.0f}"
        print(f"  {label:>12s}  {te.expected_num_gates:10.2f}  {mean_count:14.2f}  {te.overhead:10.4f}")

    # Verify: smaller delta -> more gates
    expected_gates = [r[1] for r in results]
    for i in range(1, len(expected_gates)):
        assert expected_gates[i] < expected_gates[i-1], \
            f"Larger delta should give fewer gates: {deltas[i-1]:.4f}->{deltas[i]:.4f}"

    # Verify: smaller delta -> smaller overhead
    overheads = [r[3] for r in results]
    for i in range(1, len(overheads)):
        assert overheads[i] > overheads[i-1], \
            f"Larger delta should give larger overhead: {deltas[i-1]:.4f}->{deltas[i]:.4f}"

    print("  PASS: Smaller Delta -> more gates, less overhead (and vice versa)")


# =====================================================================
# 4. Gate validity: angles, Paulis, qubit indices, sign consistency
# =====================================================================
def test_gate_validity():
    """Every gate must have angle = +/-Delta or +/-pi, valid Pauli type,
    valid qubit indices, and sign = (-1)^(number of pi-gates)."""
    print("\n=== TEST: Gate Validity ===")
    n = 6
    hamil = _make_hamil(n=n)
    delta = np.pi / 16
    te = TE_PAI(hamil, n, delta=delta, T=0.5, N=20, n_snap=1)
    circuits = te.run_te_pai(500)

    assert len(circuits) == 500, f"Expected 500 circuits, got {len(circuits)}"

    n_pi_gates = 0
    n_delta_gates = 0
    n_sign_flips = 0
    all_paulis = set()

    for signs, gates_arr in circuits:
        # signs is a list of per-snapshot cumulative signs
        assert all(s in (1, -1) for s in signs), f"Signs must be +/-1, got {signs}"
        if signs[-1] == -1:
            n_sign_flips += 1

        for snapshot in gates_arr:
            for gate in snapshot:
                pauli, angle, qubits = gate
                all_paulis.add(pauli)
                assert pauli in ("XX", "YY", "ZZ", "Z"), f"Unknown Pauli: {pauli}"

                if abs(abs(angle) - np.pi) < 1e-10:
                    n_pi_gates += 1
                elif abs(abs(angle) - delta) < 1e-10:
                    n_delta_gates += 1
                else:
                    raise AssertionError(
                        f"Gate angle {angle} is neither pi ({np.pi}) nor +/-delta ({delta})"
                    )

                for q in qubits:
                    assert 0 <= q < n, f"Qubit index {q} out of range [0, {n})"

    total_gates = n_pi_gates + n_delta_gates
    print(f"  Total gates: {total_gates}")
    print(f"  Pi-gates: {n_pi_gates} ({100*n_pi_gates/total_gates:.1f}%)")
    print(f"  Delta-gates: {n_delta_gates} ({100*n_delta_gates/total_gates:.1f}%)")
    print(f"  Sign flips: {n_sign_flips}/{len(circuits)} ({100*n_sign_flips/len(circuits):.1f}%)")
    print(f"  Pauli types seen: {sorted(all_paulis)}")

    assert n_pi_gates > 0, "Should have at least some pi-gates with delta=pi/16"
    assert n_delta_gates > 0, "Should have delta-rotation gates"
    assert n_sign_flips > 0, "Some circuits should have sign=-1 (from pi-gates)"
    assert all_paulis == {"XX", "YY", "ZZ", "Z"}, f"Expected all Pauli types, got {all_paulis}"

    # Verify per-snapshot sign consistency: signs[k] = (-1)^(pi-gates through snapshot k)
    for signs, gates_arr in circuits:
        counted_pi = 0
        for snap_idx, snapshot in enumerate(gates_arr):
            for gate in snapshot:
                if abs(abs(gate[1]) - np.pi) < 1e-10:
                    counted_pi += 1
            expected_sign = (-1) ** counted_pi
            assert signs[snap_idx] == expected_sign, \
                f"Sign {signs[snap_idx]} at snapshot {snap_idx} doesn't match " \
                f"{counted_pi} cumulative pi-gates (expected {expected_sign})"

    print("  PASS: All gates valid, per-snapshot signs consistent with pi-gate count")


# =====================================================================
# 5. Pi-gate prevalence increases with larger Delta
# =====================================================================
def test_pi_gate_prevalence():
    r"""The fraction of pi-gates among all non-identity gates grows with Delta.

    Analytic insight: for a single term with angle theta, the PAI decomposition
    gives three outcomes (identity, +/-Delta rotation, pi-rotation). Among the
    non-identity outcomes, the ratio c/(|b|+|c|) grows with Delta for small
    theta. Across a full Hamiltonian (many terms, varying angles), the net
    effect is that larger Delta -> higher pi-gate fraction empirically.
    """
    print("\n=== TEST: Pi-Gate Prevalence vs Delta ===")
    hamil = _make_hamil()
    T = 0.5
    N = 100
    n_circuits = 1000

    # Analytic: for small theta, c/(|b|+|c|) grows cleanly with Delta
    theta = 0.01
    print(f"\n  Analytic: pi-fraction among non-identity gates (theta={theta})")
    print(f"  {'Delta':>10s}  {'p(identity)':>12s}  {'p(delta)':>12s}  {'p(pi)':>12s}  {'pi/(d+pi)':>10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    prev_frac = 0
    for d in [np.pi/128, np.pi/64, np.pi/32, np.pi/16, np.pi/8, np.pi/4]:
        p = abc(theta, d)
        p_norm = np.abs(p) / np.sum(np.abs(p))
        frac = p_norm[2] / (p_norm[1] + p_norm[2])
        label = f"pi/{np.pi/d:.0f}"
        print(f"  {label:>10s}  {p_norm[0]:12.6f}  {p_norm[1]:12.6f}  {p_norm[2]:12.6f}  {frac:10.6f}")
        assert frac >= prev_frac - 1e-12, \
            f"pi-fraction should increase with Delta: {prev_frac:.6f} -> {frac:.6f}"
        prev_frac = frac

    # Empirical: sample circuits at different Delta values
    deltas = [np.pi / 64, np.pi / 32, np.pi / 16, np.pi / 8, np.pi / 4]
    pi_fractions = []

    print(f"\n  Empirical: pi-gate fraction in {n_circuits} sampled circuits")
    print(f"  {'Delta':>10s}  {'total gates':>12s}  {'pi-gates':>10s}  {'pi-fraction':>12s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*12}")

    for d in deltas:
        te = TE_PAI(hamil, hamil.nqubits, delta=d, T=T, N=N, n_snap=1)
        circuits = te.run_te_pai(n_circuits)
        n_pi, n_delta = _classify_gates(circuits, d)
        total = n_pi + n_delta
        frac = n_pi / total if total > 0 else 0
        pi_fractions.append(frac)
        label = f"pi/{np.pi/d:.0f}"
        print(f"  {label:>10s}  {total:12d}  {n_pi:10d}  {frac:12.4f}")

    # Verify: pi-gate fraction strictly increases with Delta
    for i in range(1, len(pi_fractions)):
        assert pi_fractions[i] > pi_fractions[i-1], \
            f"Pi-fraction should increase: Delta=pi/{np.pi/deltas[i-1]:.0f} ({pi_fractions[i-1]:.4f}) " \
            f"-> pi/{np.pi/deltas[i]:.0f} ({pi_fractions[i]:.4f})"

    print("  PASS: Pi-gate prevalence increases with Delta")


# =====================================================================
if __name__ == "__main__":
    test_gate_count_saturation()
    test_gate_count_distribution()
    test_delta_effect_on_length()
    test_gate_validity()
    test_pi_gate_prevalence()
    print("\n" + "=" * 50)
    print("ALL CIRCUIT TESTS PASSED")
    print("=" * 50)
