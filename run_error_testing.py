#!/usr/bin/env python
"""run_error_testing.py — Multi-observable error comparison: TE-PAI vs Trotter.

Measures a broad set of observables (single-qubit X/Y/Z on all qubits, and
optionally nearest-neighbour XX/YY/ZZ) to compare TE-PAI estimation accuracy
against Trotterization over time.

Produces a multi-panel figure:
  (1) TE-PAI vs Trotter for a highlighted observable whose ⟨O⟩ starts near zero
  (2) Average |error| across all observables over time
  (3) Average scaled magnitude of TE-PAI samples over time

Usage (via runner.py):
    Add experiments with "type": "error_testing" in config.json.

Direct usage:
    python run_error_testing.py --n-qubits 10 --total-time 1.0 \
        --N-samples 100 --N-tepai 50 --delta 0.003
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Package import — works whether mps-te-pai is installed or run from source
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "mps-te-pai", "src"))

from mps_tepai import (
    Hamiltonian,
    TE_PAI,
    create_mps_circuit,
    apply_gates,
)

import quimb as qu

from run_experiment import (
    DATA_DIR,
    _experiment_folder,
)


# ===================================================================
#  Observable specification
# ===================================================================

def _build_observable_specs(n_qubits, max_locality=3):
    """Generate list of observable specs at multiple locality levels.

    Each spec is a tuple: (name, pauli_str, qubit_indices)
      - name: human-readable label, e.g. "X_0", "ZZ_0_1", "XYZ_0_1_2"
      - pauli_str: Pauli string, e.g. "X", "ZZ", "XYZ"
      - qubit_indices: tuple of qubit indices

    Parameters
    ----------
    n_qubits : int
    max_locality : int
        Maximum number of qubits in an observable (1, 2, 3, ...).
        Observables are built on contiguous nearest-neighbour chains.
    """
    paulis_1 = ("X", "Y", "Z")
    paulis_2 = ("XX", "YY", "ZZ")
    # Representative sample of 3-qubit Pauli strings
    paulis_3 = ("XXX", "YYY", "ZZZ", "XYZ", "ZXZ")
    # Representative sample of 4-qubit Pauli strings
    paulis_4 = ("XXXX", "ZZZZ", "XYZX")

    specs = []
    # 1-qubit
    for q in range(n_qubits):
        for p in paulis_1:
            specs.append((f"{p}_{q}", p, (q,)))

    # 2-qubit (nearest-neighbour)
    if max_locality >= 2:
        for q in range(n_qubits - 1):
            for pp in paulis_2:
                specs.append((f"{pp}_{q}_{q+1}", pp, (q, q + 1)))

    # 3-qubit (contiguous triples)
    if max_locality >= 3 and n_qubits >= 3:
        for q in range(n_qubits - 2):
            for pp in paulis_3:
                qubits = (q, q + 1, q + 2)
                name = f"{pp}_{'_'.join(str(i) for i in qubits)}"
                specs.append((name, pp, qubits))

    # 4-qubit (contiguous quadruples)
    if max_locality >= 4 and n_qubits >= 4:
        for q in range(n_qubits - 3):
            for pp in paulis_4:
                qubits = (q, q + 1, q + 2, q + 3)
                name = f"{pp}_{'_'.join(str(i) for i in qubits)}"
                specs.append((name, pp, qubits))

    return specs


def _locality_of(obs_spec):
    """Return the locality (number of qubits) of an observable spec."""
    return len(obs_spec[2])


def _build_operator(pauli_str):
    """Build the matrix operator from a Pauli string like 'X', 'ZZ', 'XY'."""
    op = qu.pauli(pauli_str[0])
    for ch in pauli_str[1:]:
        op = op & qu.pauli(ch)
    return op


def _measure_all_observables(circuit, obs_specs):
    """Measure all observables on the MPS circuit.

    Returns array of physical expectation values in [-1, 1].
    """
    vals = np.empty(len(obs_specs))
    for i, (_name, pauli_str, qubits) in enumerate(obs_specs):
        op = _build_operator(pauli_str)
        vals[i] = np.real(circuit.local_expectation(op, qubits))
    return vals


def _format_obs_label(name):
    """Convert 'Z_0' → r'$\\langle Z_0 \\rangle$', 'XYZ_0_1_2' → ..., etc."""
    parts = name.split("_")
    pauli = parts[0]
    qubits = parts[1:]
    if len(pauli) == len(qubits):
        terms = " ".join(f"{pauli[i]}_{{{qubits[i]}}}" for i in range(len(pauli)))
        return rf"$\langle {terms} \rangle$"
    return name


# ===================================================================
#  Exact Hamiltonian exponentiation (small qubit counts)
# ===================================================================

def _build_hamiltonian_matrix(hamil, t):
    """Build the full 2^n × 2^n Hamiltonian matrix at time t.

    Constructs H = sum_i c_i(t) * P_i where P_i are Pauli tensor products
    embedded on the appropriate qubits.
    """
    from scipy.sparse import eye as speye, kron as spkron

    n = hamil.nqubits
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    _pauli = {
        "X": qu.pauli("X"),
        "Y": qu.pauli("Y"),
        "Z": qu.pauli("Z"),
    }
    I2 = np.eye(2, dtype=complex)

    for pauli_str, qubits, coef_fn in hamil.terms:
        c = coef_fn(t)
        if abs(c) < 1e-30:
            continue
        # Build the full tensor product operator
        ops = [I2] * n
        for k, ch in enumerate(pauli_str):
            ops[qubits[k]] = _pauli[ch]
        full = ops[0]
        for j in range(1, n):
            full = np.kron(full, ops[j])
        H += c * full

    return H


def _build_initial_statevector(n_qubits, initial_state):
    """Build the initial state as a dense 2^n vector."""
    if initial_state == "zero":
        psi = np.zeros(2 ** n_qubits, dtype=complex)
        psi[0] = 1.0
    elif initial_state in ("plus", "plus_flip"):
        # |+>^n = H^n |0>^n = uniform superposition
        psi = np.ones(2 ** n_qubits, dtype=complex) / np.sqrt(2 ** n_qubits)
        if initial_state == "plus_flip":
            # Apply Z on middle qubit: flips sign when that qubit is |1>
            middle = int(np.floor(n_qubits / 2))
            for i in range(2 ** n_qubits):
                if (i >> (n_qubits - 1 - middle)) & 1:
                    psi[i] *= -1
    else:
        raise ValueError(f"Unknown initial_state: {initial_state}")
    return psi


def _measure_obs_statevector(psi, obs_specs, n_qubits):
    """Measure all observables on a statevector. Returns array of real values."""
    I2 = np.eye(2, dtype=complex)
    _pauli = {
        "X": qu.pauli("X"),
        "Y": qu.pauli("Y"),
        "Z": qu.pauli("Z"),
    }
    vals = np.empty(len(obs_specs))
    for idx, (_name, pauli_str, qubits) in enumerate(obs_specs):
        ops = [I2] * n_qubits
        for k, ch in enumerate(pauli_str):
            ops[qubits[k]] = _pauli[ch]
        full = ops[0]
        for j in range(1, n_qubits):
            full = np.kron(full, ops[j])
        vals[idx] = np.real(psi.conj() @ full @ psi)
    return vals


def _run_exact_multi_obs(hamil, nq, T, n_snap, init, obs_specs):
    """Exact time evolution via Hamiltonian exponentiation.

    For time-independent H (or piecewise), we exponentiate at each snapshot:
      |ψ(t_k)> = exp(-i H t_k) |ψ(0)>

    Since the Hamiltonian coefficients can be time-dependent, we use a
    fine-grained product of short-time propagators.

    Returns (times, obs_matrix) where obs_matrix has shape (n_obs, n_snap+1).
    """
    from scipy.linalg import expm

    psi = _build_initial_statevector(nq, init)
    n_obs = len(obs_specs)
    obs_matrix = np.zeros((n_obs, n_snap + 1))
    obs_matrix[:, 0] = _measure_obs_statevector(psi, obs_specs, nq)

    dt = T / n_snap
    # Sub-steps per snapshot for accuracy with time-dependent H
    n_sub = 10
    dt_sub = dt / n_sub

    for k in range(n_snap):
        for s in range(n_sub):
            t_mid = k * dt + (s + 0.5) * dt_sub
            H = _build_hamiltonian_matrix(hamil, t_mid)
            psi = expm(-1j * H * dt_sub) @ psi
        obs_matrix[:, k + 1] = _measure_obs_statevector(psi, obs_specs, nq)

    times = np.linspace(0, T, n_snap + 1)
    return times, obs_matrix


def _exact_et_file():
    return "exact_multiobs.npz"


def load_or_run_exact_multi_obs(folder, hamil, nq, T, n_snap, init, obs_specs):
    """Run exact multi-observable simulation, or load from cache."""
    fname = _exact_et_file()
    path = os.path.join(folder, fname)

    obs_names = [s[0] for s in obs_specs]

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        cached_names = list(data.get("obs_names", []))
        if cached_names == obs_names:
            print(f"  [cache] {fname}")
            return data["times"], data["obs_matrix"]
        else:
            print(f"  [cache STALE] {fname} — observable set changed")

    print(f"  Running exact simulation ({nq}q, {n_snap} snapshots, "
          f"{len(obs_specs)} observables) ...")
    t0 = time.time()
    times, obs_matrix = _run_exact_multi_obs(
        hamil, nq, T, n_snap, init, obs_specs,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    np.savez_compressed(
        path, times=times, obs_matrix=obs_matrix,
        obs_names=np.array(obs_names),
    )
    print(f"  Saved -> {path}")
    return times, obs_matrix


# ===================================================================
#  Trotter with multi-observable measurement
# ===================================================================

def _run_trotter_multi_obs(hamil, nq, T, N, n_snap, chi, init, obs_specs):
    """Run fixed-N Trotter and measure all observables at each snapshot.

    Uses N total Trotter steps distributed uniformly across the simulation,
    with measurements taken every N/n_snap steps.

    Returns (times, obs_matrix) where obs_matrix has shape (n_obs, n_snap+1).
    """
    circ = create_mps_circuit(nq, init, chi)
    n_obs = len(obs_specs)
    obs_matrix = np.zeros((n_obs, n_snap + 1))
    obs_matrix[:, 0] = _measure_all_observables(circ, obs_specs)

    ts = np.linspace(0, T, N, endpoint=False)
    terms = [hamil.get_term(t) for t in ts]
    per_snap = N // n_snap

    for i in range(N):
        gates = [
            (p, 2 * c * T / N, idx)
            for p, idx, c in terms[i]
        ]
        apply_gates(circ, gates)

        if (i + 1) % per_snap == 0:
            snap_idx = (i + 1) // per_snap
            obs_matrix[:, snap_idx] = _measure_all_observables(circ, obs_specs)

    times = np.linspace(0, T, n_snap + 1)
    return times, obs_matrix


# ===================================================================
#  TE-PAI multi-observable worker
# ===================================================================

def _et_worker(args):
    """Error-testing worker: run one TE-PAI circuit, measure all observables.

    Returns raw physical measurements array of shape (n_obs, n_timesteps).
    Sign/gamma weighting is applied in post-processing.
    """
    nq, gates_arr, chi, init, obs_specs = args
    circ = create_mps_circuit(nq, init, chi)

    n_obs = len(obs_specs)
    n_ts = len(gates_arr) + 1
    raw = np.zeros((n_obs, n_ts))
    raw[:, 0] = _measure_all_observables(circ, obs_specs)

    for si, snap_gates in enumerate(gates_arr):
        apply_gates(circ, snap_gates)
        raw[:, si + 1] = _measure_all_observables(circ, obs_specs)

    return raw


def _run_tepai_multi_obs(te_pai, circuits, nq, chi, init, nw, obs_specs):
    """Run all TE-PAI circuits and collect per-sample, per-observable measurements.

    Returns
    -------
    times : (n_ts,)
    all_raw : (n_samples, n_obs, n_ts) — raw physical values in [-1, 1]
    all_signs : (n_samples, n_snap)
    gam_arr : (n_ts,) — gamma values
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = len(circuits)
    inputs = [
        (nq, gates_arr, chi, init, obs_specs)
        for _signs, gates_arr in circuits
    ]

    results = [None] * n
    if nw > 1 and n > 1:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_et_worker, inp): i
                    for i, inp in enumerate(inputs)}
            for f in as_completed(futs):
                results[futs[f]] = f.result()
    else:
        for i, inp in enumerate(inputs):
            results[i] = _et_worker(inp)

    all_raw = np.array(results)                                  # (S, n_obs, n_ts)
    all_signs = np.array([signs for signs, _ in circuits])       # (S, n_snap)
    gam_arr = np.array(te_pai.gam_list)                          # (n_ts,)

    dT = te_pai.T / te_pai.n_snap
    times = np.arange(te_pai.n_snap + 1) * dT

    return times, all_raw, all_signs, gam_arr


# ===================================================================
#  Cache helpers
# ===================================================================

def _trotter_et_file(N):
    return f"trotter_multiobs_N{N}.npz"


def _tepai_et_file(pi_over_delta, N, S):
    return f"tepai_et_d{pi_over_delta}_N{N}_S{S}.npz"


def load_or_run_trotter_multi_obs(folder, hamil, nq, T, N, n_snap,
                                   chi, init, obs_specs):
    """Run multi-observable Trotter, or load from cache."""
    fname = _trotter_et_file(N)
    path = os.path.join(folder, fname)

    obs_names = [s[0] for s in obs_specs]

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        cached_names = list(data.get("obs_names", []))
        if cached_names == obs_names:
            print(f"  [cache] {fname}")
            return data["times"], data["obs_matrix"]
        else:
            print(f"  [cache STALE] {fname} — observable set changed")

    print(f"  Running Trotter (N={N}, fixed, "
          f"{len(obs_specs)} observables) ...")
    t0 = time.time()
    times, obs_matrix = _run_trotter_multi_obs(
        hamil, nq, T, N, n_snap, chi, init, obs_specs,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    np.savez_compressed(
        path, times=times, obs_matrix=obs_matrix,
        obs_names=np.array(obs_names),
    )
    print(f"  Saved -> {path}")
    return times, obs_matrix


def load_or_run_tepai_multi_obs(folder, hamil, nq, T, dt, delta, Nt, S,
                                 chi, init, nw, seed, obs_specs):
    """Run multi-observable TE-PAI, or load from cache."""
    pi_over_delta = round(np.pi / delta)
    fname = _tepai_et_file(pi_over_delta, Nt, S)
    path = os.path.join(folder, fname)

    obs_names = [s[0] for s in obs_specs]

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        cached_names = list(data.get("obs_names", []))
        cached_seed = int(data["seed"]) if "seed" in data else -1
        if cached_names == obs_names and cached_seed == seed:
            print(f"  [cache] {fname}")
            return (data["times"], data["all_raw"],
                    data["all_signs"], data["gam_arr"])
        else:
            print(f"  [cache STALE] {fname}")

    dur = T
    n_snap = int(round(dur / dt))

    print(f"  Building TE-PAI (delta={delta:.4e}, N={Nt}, "
          f"snaps={n_snap}) ...")
    te = TE_PAI(hamil, nq, delta=delta, T=dur, N=Nt, n_snap=n_snap)

    if S < te.overhead ** 2:
        print(f"  Note: overhead={te.overhead:.1f} -> suggest "
              f"S>={int(np.ceil(te.overhead ** 2))}")

    print(f"  Generating {S} circuits ...")
    t0 = time.time()
    circuits = te.run_te_pai(S, n_workers=nw, seed=seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    print(f"  Executing ({nw} workers, {len(obs_specs)} observables) ...")
    t0 = time.time()
    times, all_raw, all_signs, gam_arr = _run_tepai_multi_obs(
        te, circuits, nq, chi, init, nw, obs_specs,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    np.savez_compressed(
        path, times=times, all_raw=all_raw, all_signs=all_signs,
        gam_arr=gam_arr, obs_names=np.array(obs_names), seed=seed,
    )
    print(f"  Saved -> {path}")
    return times, all_raw, all_signs, gam_arr


# ===================================================================
#  Post-processing
# ===================================================================

def _compute_weighted_obs(all_raw, all_signs, gam_arr):
    """Apply sign × γ weighting to raw TE-PAI measurements.

    Parameters
    ----------
    all_raw   : (S, n_obs, n_ts) — raw physical values in [-1, 1]
    all_signs : (S, n_snap)
    gam_arr   : (n_ts,)

    Returns
    -------
    all_weighted  : (S, n_obs, n_ts)
    mean_weighted : (n_obs, n_ts) — per-observable mean estimates
    """
    S, n_obs, n_ts = all_raw.shape
    all_weighted = np.zeros_like(all_raw)

    # t=0: weight = 1 (gamma_0 = 1, no sign yet)
    all_weighted[:, :, 0] = all_raw[:, :, 0]

    # t>0: weight = sign_k × γ_k
    for k in range(1, n_ts):
        # all_signs[:, k-1] has shape (S,), broadcast over n_obs
        all_weighted[:, :, k] = (all_raw[:, :, k]
                                  * all_signs[:, k - 1, np.newaxis]
                                  * gam_arr[k])

    mean_weighted = np.mean(all_weighted, axis=0)   # (n_obs, n_ts)
    return all_weighted, mean_weighted


def _compute_errors(mean_weighted, trotter_obs_matrix):
    """Compute per-observable and average errors.

    Returns
    -------
    per_obs_error : (n_obs, n_ts) — absolute error per observable
    avg_error     : (n_ts,)       — mean absolute error across observables
    """
    per_obs_error = np.abs(mean_weighted - trotter_obs_matrix)
    avg_error = np.mean(per_obs_error, axis=0)
    return per_obs_error, avg_error


# ===================================================================
#  Plotting
# ===================================================================

_LOCALITY_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:purple",
    4: "tab:brown",
}


def plot_error_testing(times, ref_obs, tepai_mean, tepai_se,
                       per_obs_error, per_sample_error,
                       hi_rms, hi_empirical_err,
                       hi_scaled_mag, overhead_arr,
                       highlight_idx, obs_specs,
                       out_main, out_locality, n_samples,
                       ref_label="Trotter"):
    """Produce two figures for error testing.

    Figure 1 (out_main) — 4-panel:
      (1) Highlighted observable TE-PAI vs reference
      (2) Per-shot RMS + mean |error| + avg |sample| + overhead
      (3) Best observable (smallest RMS): measurement over time
      (4) Best observable: RMS vs overhead

    Figure 2 (out_locality) — standalone:
      Average |error| broken down by locality with variance shading
    """
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    hi_name = obs_specs[highlight_idx][0]
    hi_label = _format_obs_label(hi_name)

    # Pre-compute best observable: smallest time-averaged empirical error
    # per_obs_error = |mean(samples) - ref|, shape (n_obs, n_ts)
    rms_per_obs = np.sqrt(np.mean(per_sample_error ** 2, axis=0))  # (n_obs, n_ts)
    avg_empirical_per_obs = np.mean(per_obs_error, axis=1)  # (n_obs,)
    best_idx = int(np.argmin(avg_empirical_per_obs))
    best_name = obs_specs[best_idx][0]
    best_label = _format_obs_label(best_name)

    # Standard error for best observable
    S = per_sample_error.shape[0]
    if S > 1:
        best_se = np.std(
            per_sample_error[:, best_idx, :] + ref_obs[best_idx],  # reconstruct weighted
            axis=0, ddof=1,
        ) / np.sqrt(S)
    else:
        best_se = np.zeros(len(times))
    # Simpler: recompute from all_weighted passed implicitly via tepai_mean/tepai_se
    best_se = tepai_se[best_idx]

    # ================================================================
    # Figure 1: 4-panel
    # ================================================================
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(8, 14), sharex=True,
    )

    # ---- Panel 1: Highlighted observable TE-PAI vs reference ----
    ax1.plot(times, ref_obs[highlight_idx],
             color="black", label=ref_label, lw=2)
    ax1.plot(times, tepai_mean[highlight_idx],
             color="tab:green", label="TE-PAI", lw=2)
    ax1.fill_between(
        times,
        tepai_mean[highlight_idx] - tepai_se[highlight_idx],
        tepai_mean[highlight_idx] + tepai_se[highlight_idx],
        color="tab:green", alpha=0.2,
    )
    ax1.set_ylabel(hi_label)
    ax1.legend(loc="best")
    ax1.set_title(
        f"Observable comparison — {hi_name}  ($S = {n_samples}$)",
        fontsize=14,
    )

    # ---- Panel 2: RMS, empirical error, avg magnitude, overhead ----
    ax2.plot(times, hi_rms, color="tab:green", ls="--", lw=1.5,
             label="Per-shot RMS")
    ax2.plot(times, hi_empirical_err, color="tab:blue", lw=2,
             label="Mean |error|")
    ax2.plot(times, hi_scaled_mag, color="tab:orange", lw=1.5,
             label="Avg. |weighted sample|")
    ax2.plot(times, overhead_arr, color="red", ls="--", lw=1.5,
             label="Theoretical upper bound")
    ax2.set_yscale("log")
    ax2.legend(loc="best")

    # ---- Panel 3: Best observable — measurement over time ----
    ax3.plot(times, ref_obs[best_idx],
             color="black", label=ref_label, lw=2)
    ax3.plot(times, tepai_mean[best_idx],
             color="tab:green", label="TE-PAI", lw=2)
    ax3.fill_between(
        times,
        tepai_mean[best_idx] - best_se,
        tepai_mean[best_idx] + best_se,
        color="tab:green", alpha=0.2,
    )
    ax3.set_ylabel(best_label)
    ax3.legend(loc="best")
    ax3.set_title(f"Best observable (lowest RMS) — {best_name}")

    # ---- Panel 4: Best observable — RMS vs overhead ----
    ax4.plot(times, rms_per_obs[best_idx], color="tab:green", lw=2,
             label=f"Per-shot RMS — {best_name}")
    ax4.plot(times, overhead_arr, color="red", ls="--", lw=1.5,
             label="Theoretical upper bound")
    ax4.set_xlabel("Time")
    ax4.set_yscale("log")
    ax4.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(out_main, bbox_inches="tight")
    print(f"\nPlot -> {out_main}")
    print(f"  Best observable (lowest error): {best_name} "
          f"(avg |error| = {avg_empirical_per_obs[best_idx]:.6f})")

    # ================================================================
    # Figure 2: Locality RMS error breakdown (standalone)
    # ================================================================
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 5))

    # per_sample_error shape: (S, n_obs, n_ts)
    # Per-shot RMS per obs (in overhead units): sqrt(mean_s(error^2))
    localities = sorted(set(_locality_of(s) for s in obs_specs))
    for loc in localities:
        idxs = [i for i, s in enumerate(obs_specs) if _locality_of(s) == loc]
        loc_rms_per_obs = np.sqrt(
            np.mean(per_sample_error[:, idxs, :] ** 2, axis=0),
        )  # (n_obs_loc, n_ts)
        loc_rms_mean = np.mean(loc_rms_per_obs, axis=0)  # (n_ts,)
        loc_rms_std = np.std(loc_rms_per_obs, axis=0, ddof=1)  # (n_ts,)

        color = _LOCALITY_COLORS.get(loc, "tab:gray")
        zo = max(localities) - loc + 2  # lower locality renders on top
        ax3.plot(times, loc_rms_mean, color=color, lw=2,
                 label=f"Variance ({len(idxs)} {loc}-qubit obs.)", zorder=zo)
        ax3.fill_between(times,
                         loc_rms_mean - loc_rms_std,
                         loc_rms_mean + loc_rms_std,
                         color=color, alpha=0.15, zorder=zo)

    # Overall RMS with variance across observables
    overall_rms_per_obs = np.sqrt(
        np.mean(per_sample_error ** 2, axis=0),
    )  # (n_obs, n_ts)
    avg_rms = np.mean(overall_rms_per_obs, axis=0)
    avg_rms_std = np.std(overall_rms_per_obs, axis=0, ddof=1)
    ax3.plot(times, avg_rms, color="black", ls="--", lw=1.5,
             label=f"Variance (mean {len(obs_specs)} obs.)")
    ax3.fill_between(times,
                     avg_rms - avg_rms_std,
                     avg_rms + avg_rms_std,
                     color="black", alpha=0.1)

    # Analytical overhead reference
    ax3.plot(times, overhead_arr, color="red", ls="--",
             lw=1.5, label=r"Theoretical overhead")

    ax3.set_xlabel("Time")
    ax3.set_yscale("log")
    ax3.set_xlim(times[0], times[-1])
    ax3.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig(out_locality, bbox_inches="tight")
    print(f"Plot -> {out_locality}")

    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  CLI + main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-observable error testing: TE-PAI vs Trotter.",
    )
    # Physics / simulation
    p.add_argument("--n-qubits",      type=int,   default=10)
    p.add_argument("--total-time",    type=float, default=0.5)
    p.add_argument("--dt",            type=float, default=0.1)
    p.add_argument("--j",             type=float, default=0.1)
    p.add_argument("--max-bond",      type=int,   default=16)
    p.add_argument("--initial-state", type=str,   default="plus_flip")
    p.add_argument("--seed",          type=int,   default=0)

    # Algorithm
    p.add_argument("--N-trotter", type=int,   default=20)
    p.add_argument("--N-tepai",   type=int,   default=50)
    p.add_argument("--N-samples", type=int,   default=100)
    p.add_argument("--delta",     type=float, default=np.pi / 1024)

    # Error-testing specific
    p.add_argument("--max-locality", type=int, default=3,
                   help="Max qubit count per observable (1=single only, "
                        "2=+pairs, 3=+triples, 4=+quadruples)")
    p.add_argument("--highlight-observable", type=str, default=None,
                   help="Observable to highlight (e.g. 'Z_0'). "
                        "Auto-selects first Y or Z observable if omitted.")
    p.add_argument("--exact-threshold", type=int, default=14,
                   help="Use exact Hamiltonian exponentiation instead of "
                        "Trotter when n_qubits <= this value (default: 14)")

    # Legacy alias accepted for backwards compatibility
    p.add_argument("--include-two-qubit", action="store_true", default=False,
                   help=argparse.SUPPRESS)

    # Runtime
    p.add_argument("--n-cores",   type=int, default=None)
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--plot-name", type=str, default=None,
                   help="Filename for the output plot (saved inside data folder)")

    # Accepted for runner.py compatibility but unused by this experiment
    p.add_argument("--operator",          type=str,   default="X")
    p.add_argument("--measure-qubit",     type=int,   default=0)
    p.add_argument("--tepai-start-time",  type=float, nargs="+", default=[0.0])
    p.add_argument("--pi-over-delta",     type=int,   default=None)

    return p.parse_args()


def main():
    args = parse_args()

    nw = args.n_cores or max(1, (mp.cpu_count() or 4) - 2)
    T = args.total_time
    dt = args.dt
    n_snap = int(round(T / dt))
    assert n_snap >= 1, f"total-time must be >= dt={dt}"

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    # Build observable list
    max_loc = args.max_locality
    if args.include_two_qubit and max_loc < 2:
        max_loc = 2
    obs_specs = _build_observable_specs(args.n_qubits, max_loc)

    # Resolve highlight observable
    highlight_idx = 0
    if args.highlight_observable:
        names = [s[0] for s in obs_specs]
        if args.highlight_observable in names:
            highlight_idx = names.index(args.highlight_observable)
        else:
            print(f"  Warning: '{args.highlight_observable}' not in "
                  f"observable set, using auto-select")
            args.highlight_observable = None

    if not args.highlight_observable:
        # Auto-select: first Y or Z single-qubit observable (starts at ~0
        # for the plus_flip initial state)
        for i, (name, pauli_str, qubits) in enumerate(obs_specs):
            if pauli_str in ("Y", "Z") and qubits == (0,):
                highlight_idx = i
                break

    print(f"Error testing: {args.n_qubits}q, chi={args.max_bond}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, "
          f"S={args.N_samples}, workers={nw}")
    localities = sorted(set(_locality_of(s) for s in obs_specs))
    loc_counts = {l: sum(1 for s in obs_specs if _locality_of(s) == l)
                  for l in localities}
    loc_str = ", ".join(f"{c}×{l}q" for l, c in loc_counts.items())
    print(f"  Observables: {len(obs_specs)} ({loc_str})")
    print(f"  Highlight: {obs_specs[highlight_idx][0]}")

    # Data directory (shared with other experiment types)
    folder = os.path.join(
        DATA_DIR,
        _experiment_folder(
            args.n_qubits, args.max_bond, args.j, args.seed,
            T, dt, args.operator, args.measure_qubit, args.initial_state,
        ),
    )
    os.makedirs(folder, exist_ok=True)
    print(f"  Data dir: {folder}\n")

    # ---- Reference simulation (exact or Trotter) ----
    use_exact = args.n_qubits <= args.exact_threshold
    if use_exact:
        print("Exact reference (Hamiltonian exponentiation):")
        ref_times, ref_obs_matrix = load_or_run_exact_multi_obs(
            folder, hamil, args.n_qubits, T, n_snap,
            args.initial_state, obs_specs,
        )
        ref_label = "Exact"
    else:
        print("Trotter reference (multi-observable):")
        ref_times, ref_obs_matrix = load_or_run_trotter_multi_obs(
            folder, hamil, args.n_qubits, T, args.N_trotter, n_snap,
            args.max_bond, args.initial_state, obs_specs,
        )
        ref_label = "Trotter"

    # ---- TE-PAI (multi-observable) ----
    print("\nTE-PAI (multi-observable):")
    tepai_times, all_raw, all_signs, gam_arr = load_or_run_tepai_multi_obs(
        folder, hamil, args.n_qubits, T, dt, args.delta,
        args.N_tepai, args.N_samples, args.max_bond, args.initial_state,
        nw, args.seed, obs_specs,
    )

    # ---- Post-process ----
    all_weighted, mean_weighted = _compute_weighted_obs(
        all_raw, all_signs, gam_arr,
    )

    # Align reference to TE-PAI times (should already match for tstart=0)
    ref_aligned = np.zeros_like(mean_weighted)
    for o in range(len(obs_specs)):
        ref_aligned[o] = np.interp(
            tepai_times, ref_times, ref_obs_matrix[o],
        )

    # Per-observable absolute error and average error
    per_obs_error, avg_error = _compute_errors(mean_weighted, ref_aligned)

    # Per-sample per-observable absolute error: (S, n_obs, n_ts)
    per_sample_error = np.abs(all_weighted - ref_aligned[np.newaxis, :, :])

    # Standard error per observable: std across samples / sqrt(S)
    S = all_weighted.shape[0]
    if S > 1:
        tepai_se = np.std(all_weighted, axis=0, ddof=1) / np.sqrt(S)
    else:
        tepai_se = np.zeros_like(mean_weighted)

    # Per-shot RMS for highlighted observable: sqrt(mean((sample_i - trotter)^2))
    hi_deviations = all_weighted[:, highlight_idx, :] - ref_aligned[highlight_idx]
    hi_rms = np.sqrt(np.mean(hi_deviations ** 2, axis=0))  # (n_ts,)

    # Empirical error for highlighted observable: mean(|sample_i - ref|)
    hi_empirical_err = np.mean(
        np.abs(all_weighted[:, highlight_idx, :] - ref_aligned[highlight_idx]),
        axis=0,
    )  # (n_ts,)

    # Average scaled magnitude for highlighted observable
    hi_scaled_mag = np.mean(
        np.abs(all_weighted[:, highlight_idx, :]), axis=0,
    )  # (n_ts,)

    # Analytical overhead γ(t)
    delta = args.delta
    dT = T / n_snap
    tan_half_delta = np.tan(delta / 2)
    overhead_arr = np.array([
        np.exp(2 * hamil.l1_norm(k * dT) * tan_half_delta)
        for k in range(n_snap + 1)
    ])
    # NOTE: overhead_arr stays as raw γ(t). Division by √S is done
    # locally in the middle panel where estimation-error quantities are plotted.

    # ---- Summary ----
    avg_error = np.mean(per_obs_error, axis=0)
    hi_name = obs_specs[highlight_idx][0]

    # Per-obs time-averaged empirical error for ranking
    avg_empirical_per_obs = np.mean(per_obs_error, axis=1)  # (n_obs,)
    best_idx_summary = int(np.argmin(avg_empirical_per_obs))
    worst_idx_summary = int(np.argmax(avg_empirical_per_obs))
    best_name_summary = obs_specs[best_idx_summary][0]
    worst_name_summary = obs_specs[worst_idx_summary][0]

    # Per-shot RMS per obs for overhead comparison
    rms_per_obs_summary = np.sqrt(
        np.mean(per_sample_error ** 2, axis=0),
    )  # (n_obs, n_ts)

    # Per-locality stats
    localities = sorted(set(_locality_of(s) for s in obs_specs))
    loc_counts = {l: sum(1 for s in obs_specs if _locality_of(s) == l)
                  for l in localities}

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Qubits:                 {args.n_qubits}")
    print(f"  Max bond dim (χ):       {args.max_bond}")
    print(f"  Coupling j:             {args.j}")
    print(f"  Seed:                   {args.seed}")
    print(f"  Initial state:          {args.initial_state}")
    print(f"  Total time T:           {T}")
    print(f"  Time step dt:           {dt}")
    print(f"  Snapshots:              {n_snap}")
    print(f"  δ (delta):              {args.delta:.6e}  (π/{int(round(np.pi/args.delta))})")
    print(f"  N_tepai:                {args.N_tepai}")
    print(f"  N_samples (S):          {S}")
    print(f"  Reference:              {ref_label}")
    if ref_label == "Trotter":
        print(f"  N_trotter:              {args.N_trotter}")
    print(f"  Max locality:           {max_loc}")
    obs_str = ", ".join(f"{c}×{l}q" for l, c in loc_counts.items())
    print(f"  Observables:            {len(obs_specs)} ({obs_str})")
    print(f"  Highlight:              {hi_name}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Final overhead γ(T):    {overhead_arr[-1]:.4f}")
    print(f"  Final avg |error|:      {avg_error[-1]:.6f}")
    print(f"  Time-avg |error|:       {np.mean(avg_error):.6f}")
    print(f"")
    print(f"  Highlighted obs ({hi_name}):")
    print(f"    Final |error|:        {per_obs_error[highlight_idx, -1]:.6f}")
    print(f"    Time-avg |error|:     {avg_empirical_per_obs[highlight_idx]:.6f}")
    print(f"    Final per-shot RMS:   {hi_rms[-1]:.6f}")
    print(f"")
    print(f"  Best obs ({best_name_summary}):")
    print(f"    Time-avg |error|:     {avg_empirical_per_obs[best_idx_summary]:.6f}")
    print(f"    Final |error|:        {per_obs_error[best_idx_summary, -1]:.6f}")
    print(f"    Final per-shot RMS:   {rms_per_obs_summary[best_idx_summary, -1]:.6f}")
    print(f"")
    print(f"  Worst obs ({worst_name_summary}):")
    print(f"    Time-avg |error|:     {avg_empirical_per_obs[worst_idx_summary]:.6f}")
    print(f"    Final |error|:        {per_obs_error[worst_idx_summary, -1]:.6f}")
    print(f"    Final per-shot RMS:   {rms_per_obs_summary[worst_idx_summary, -1]:.6f}")

    print(f"\n  Error by locality (time-averaged):")
    for loc in localities:
        idxs = [i for i, s in enumerate(obs_specs) if _locality_of(s) == loc]
        loc_avg_err = np.mean(avg_empirical_per_obs[idxs])
        loc_std_err = np.std(avg_empirical_per_obs[idxs], ddof=1) if len(idxs) > 1 else 0.0
        loc_avg_rms = np.mean(np.mean(rms_per_obs_summary[idxs], axis=1))
        print(f"    {loc}-qubit ({loc_counts[loc]:3d} obs): "
              f"avg |error| = {loc_avg_err:.6f} ± {loc_std_err:.6f}, "
              f"avg RMS = {loc_avg_rms:.6f}")

    print(f"\n  Overhead vs RMS sanity check (final time):")
    n_rms_above = np.sum(rms_per_obs_summary[:, -1] > overhead_arr[-1])
    print(f"    γ(T) = {overhead_arr[-1]:.4f}")
    print(f"    Obs with RMS > γ:     {n_rms_above}/{len(obs_specs)}")
    print(f"    Max RMS at T:         {np.max(rms_per_obs_summary[:, -1]):.6f}")
    print(f"    Min RMS at T:         {np.min(rms_per_obs_summary[:, -1]):.6f}")
    print(f"{'=' * 60}")

    # ---- Plot ----
    if not args.no_plot:
        base_name = args.plot_name or "error_testing.pdf"
        stem, ext = os.path.splitext(base_name)
        out_main = os.path.join(folder, base_name)
        out_locality = os.path.join(folder, f"{stem}_locality{ext}")
        plot_error_testing(
            tepai_times, ref_aligned, mean_weighted, tepai_se,
            per_obs_error, per_sample_error,
            hi_rms, hi_empirical_err, hi_scaled_mag,
            overhead_arr, highlight_idx, obs_specs,
            out_main, out_locality, args.N_samples,
            ref_label=ref_label,
        )


if __name__ == "__main__":
    main()
