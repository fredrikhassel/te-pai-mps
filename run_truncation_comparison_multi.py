#!/usr/bin/env python
"""run_truncation_comparison_multi.py — Truncation comparison aggregated over
all single-qubit observables.

Runs three simulations:
  1. Reference Trotter at high bond dimension (max_bond)
  2. Truncated Trotter at low bond dimension (max_bond_truncated)
  3. TE-PAI at the same low bond dimension, aggregated over N_samples

Measures X, Y, Z on every qubit and plots the average absolute error over time
for truncated Trotter and TE-PAI relative to the reference Trotter.  The TE-PAI
line includes a shaded standard-error band.

Config-driven (via runner.py):
    Add experiments with "type": "truncation_comparison_multi" in config.json.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Package import
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

from run_experiment import DATA_DIR, _experiment_folder

from run_error_testing import (
    _measure_all_observables,
    _run_trotter_multi_obs,
    _et_worker,
    _run_tepai_multi_obs,
    _compute_weighted_obs,
)


# ===================================================================
#  Observable specification — single-qubit only
# ===================================================================

def _build_single_qubit_specs(n_qubits):
    """Generate observable specs for X, Y, Z on every qubit.

    Returns list of (name, pauli_str, qubit_indices) tuples.
    """
    specs = []
    for q in range(n_qubits):
        for p in ("X", "Y", "Z"):
            specs.append((f"{p}_{q}", p, (q,)))
    return specs


# ===================================================================
#  Cache helpers
# ===================================================================

def _trotter_tc_file(N, chi):
    return f"trotter_multiobs_tc_N{N}_chi{chi}.npz"


def _tepai_tc_file(pi_over_delta, N, S, chi):
    return f"tepai_multiobs_tc_d{pi_over_delta}_N{N}_S{S}_chi{chi}.npz"


def _data_folder(nq, j, seed, T, dt, init):
    """Folder name for multi-observable truncation comparison (no single
    operator/qubit in the path since we measure all)."""
    return (
        f"nq{nq}_j{j:.2f}_seed{seed}"
        f"_T{T:.2f}_dt{dt:.2f}_{init}_truncation_multi"
    )


# ===================================================================
#  Cache-aware runners
# ===================================================================

def load_or_run_trotter_tc(folder, hamil, nq, T, N, n_snap,
                           chi, init, obs_specs):
    """Run multi-observable Trotter at a given chi, or load from cache."""
    fname = _trotter_tc_file(N, chi)
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

    print(f"  Running Trotter (N={N}, chi={chi}, "
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


def load_or_run_tepai_tc(folder, hamil, nq, T, dt, delta, Nt, S,
                         chi, init, nw, seed, obs_specs):
    """Run multi-observable TE-PAI at a given chi, or load from cache."""
    pi_over_delta = round(np.pi / delta)
    fname = _tepai_tc_file(pi_over_delta, Nt, S, chi)
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
          f"snaps={n_snap}, chi={chi}) ...")
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
#  Plotting
# ===================================================================

def plot_truncation_comparison_multi(times, trunc_err, tepai_err, tepai_err_se,
                                     chi_ref, chi_trunc, n_samples, n_obs,
                                     out_path):
    """Plot average |error| over time for truncated Trotter and TE-PAI."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Truncated Trotter error (black solid)
    ax.plot(times, trunc_err, color="black", lw=1.5,
            label=rf"Trotter $\chi={chi_trunc}$")

    # TE-PAI error (green solid with SE band)
    ax.plot(times, tepai_err, color="tab:green", lw=1.5,
            label=rf"TE-PAI $\chi={chi_trunc}$")
    if n_samples > 1:
        ax.fill_between(
            times,
            tepai_err - tepai_err_se,
            tepai_err + tepai_err_se,
            color="tab:green", alpha=0.2,
        )
    else:
        print("  Note: N_samples=1 -> no SE band plotted")

    ax.set_xlim(0, times[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel(
        rf"Truncation error"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


def plot_best_observable(times, ref_obs, trunc_obs, tepai_obs, tepai_se,
                         chi_ref, chi_trunc, n_samples, obs_name,
                         out_path):
    """Single-observable plot (same style as run_truncation_comparison) for the
    observable where TE-PAI has the largest advantage over truncated Trotter."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Reference Trotter (black dashed)
    ax.plot(times, ref_obs, color="black", ls="--", lw=1.5,
            label=rf"Trotter $\chi={chi_ref}$ (reference)")

    # Truncated Trotter (black solid)
    ax.plot(times, trunc_obs, color="black", lw=1.5,
            label=rf"Trotter $\chi={chi_trunc}$")

    # TE-PAI (green solid with error band)
    ax.plot(times, tepai_obs, color="tab:green", lw=1.5,
            label=rf"TE-PAI $\chi={chi_trunc}$")
    if n_samples > 1:
        ax.fill_between(
            times, tepai_obs - tepai_se, tepai_obs + tepai_se,
            color="tab:green", alpha=0.2,
        )

    ax.set_xlim(0, times[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel(rf"$\langle {obs_name} \rangle$")
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


def plot_truncation_comparison_multi_minimal(times, trunc_err, tepai_err,
                                              tepai_err_se, chi_ref,
                                              chi_trunc, n_samples, n_obs,
                                              out_path):
    """Wide, minimal version of the multi-observable truncation comparison."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fs = 18
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    lw = 2.5

    ax.plot(times, trunc_err, color="black", lw=lw,
            label=rf"Trotter $\chi={chi_trunc}$")
    ax.plot(times, tepai_err, color="tab:green", lw=lw,
            label=rf"TE-PAI $\chi={chi_trunc}$")
    if n_samples > 1:
        ax.fill_between(
            times,
            tepai_err - tepai_err_se,
            tepai_err + tepai_err_se,
            color="tab:green", alpha=0.2,
        )

    ax.set_xlim(0, times[-1])
    ax.set_xlabel("Time", fontsize=fs)
    ax.set_ylabel("Truncation error", fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs - 2)
    ax.tick_params(axis="y", labelsize=fs - 2)
    ax.legend(loc="best", fontsize=fs - 2)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Plot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  CLI + main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Truncation comparison aggregated over all "
                    "single-qubit observables.",
    )
    # Physics / simulation
    p.add_argument("--n-qubits",    type=int,   default=10)
    p.add_argument("--total-time",  type=float, default=0.5)
    p.add_argument("--dt",          type=float, default=0.1)
    p.add_argument("--j",           type=float, default=0.1)
    p.add_argument("--max-bond",    type=int,   default=16,
                   help="Bond dimension for reference Trotter")
    p.add_argument("--max-bond-truncated", type=int, default=4,
                   help="Bond dimension for truncated Trotter and TE-PAI")
    p.add_argument("--initial-state", type=str, default="plus_flip")
    p.add_argument("--seed",        type=int,   default=0)

    # Algorithm
    p.add_argument("--N-trotter",   type=int,   default=20)
    p.add_argument("--N-tepai",     type=int,   default=50)
    p.add_argument("--N-samples",   type=int,   default=100)
    p.add_argument("--delta",       type=float, default=np.pi / 1024)

    # Runtime
    p.add_argument("--n-cores",     type=int,   default=None)
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("--plot-name",   type=str,   default=None,
                   help="Filename for the plot (saved inside data folder)")

    # Accepted for runner.py compatibility but unused
    p.add_argument("--operator",         type=str,   default="X")
    p.add_argument("--measure-qubit",    type=int,   default=0)
    p.add_argument("--tepai-start-time", type=float, nargs="+", default=[0.0])

    return p.parse_args()


def main():
    args = parse_args()

    nw = args.n_cores or max(1, (mp.cpu_count() or 4) - 2)
    T = args.total_time
    dt = args.dt
    n_snap = int(round(T / dt))
    assert n_snap >= 1, f"total-time must be >= dt={dt}"

    chi_ref = args.max_bond
    chi_trunc = args.max_bond_truncated

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    # All single-qubit observables
    obs_specs = _build_single_qubit_specs(args.n_qubits)
    n_obs = len(obs_specs)

    print(f"Truncation comparison (multi-observable): {args.n_qubits}q, "
          f"chi_ref={chi_ref}, chi_trunc={chi_trunc}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, "
          f"S={args.N_samples}, workers={nw}")
    print(f"  Observables: {n_obs} (X/Y/Z on {args.n_qubits} qubits)")

    # Shared data directory
    folder = os.path.join(
        DATA_DIR,
        _data_folder(args.n_qubits, args.j, args.seed, T, dt,
                     args.initial_state),
    )
    os.makedirs(folder, exist_ok=True)
    print(f"  Data dir: {folder}\n")

    # ---- Reference Trotter (high chi) ----
    print(f"Reference Trotter (chi={chi_ref}):")
    ref_t, ref_obs = load_or_run_trotter_tc(
        folder, hamil, args.n_qubits, T, args.N_trotter, n_snap,
        chi_ref, args.initial_state, obs_specs,
    )

    # ---- Truncated Trotter (low chi) ----
    print(f"\nTruncated Trotter (chi={chi_trunc}):")
    trunc_t, trunc_obs = load_or_run_trotter_tc(
        folder, hamil, args.n_qubits, T, args.N_trotter, n_snap,
        chi_trunc, args.initial_state, obs_specs,
    )

    # ---- TE-PAI (low chi) ----
    print("\nTE-PAI:")
    tepai_t, all_raw, all_signs, gam_arr = load_or_run_tepai_tc(
        folder, hamil, args.n_qubits, T, dt, args.delta,
        args.N_tepai, args.N_samples, chi_trunc, args.initial_state,
        nw, args.seed, obs_specs,
    )

    # ---- Post-process ----
    all_weighted, mean_weighted = _compute_weighted_obs(
        all_raw, all_signs, gam_arr,
    )

    # Align reference and truncated Trotter to TE-PAI times
    ref_aligned = np.zeros_like(mean_weighted)
    trunc_aligned = np.zeros_like(mean_weighted)
    for o in range(n_obs):
        ref_aligned[o] = np.interp(tepai_t, ref_t, ref_obs[o])
        trunc_aligned[o] = np.interp(tepai_t, trunc_t, trunc_obs[o])

    # Average |error| across observables: truncated Trotter vs reference
    trunc_err = np.mean(np.abs(trunc_aligned - ref_aligned), axis=0)  # (n_ts,)

    # Average |error| across observables: TE-PAI vs reference
    tepai_err = np.mean(np.abs(mean_weighted - ref_aligned), axis=0)  # (n_ts,)

    # Standard error of the TE-PAI average error via bootstrap over samples:
    # For each sample, compute its average |error| across observables,
    # then take SE = std / sqrt(S) across samples.
    S = all_weighted.shape[0]
    per_sample_avg_err = np.mean(
        np.abs(all_weighted - ref_aligned[np.newaxis, :, :]), axis=1,
    )  # (S, n_ts)
    if S > 1:
        tepai_err_se = np.std(per_sample_avg_err, axis=0, ddof=1) / np.sqrt(S)
    else:
        tepai_err_se = np.zeros(len(tepai_t))

    # ---- Find best-advantage observable ----
    # Per-observable |error|: (n_obs, n_ts)
    trunc_per_obs_err = np.abs(trunc_aligned - ref_aligned)
    tepai_per_obs_err = np.abs(mean_weighted - ref_aligned)
    # Advantage = trunc_err - tepai_err (positive means TE-PAI is better)
    advantage = trunc_per_obs_err - tepai_per_obs_err  # (n_obs, n_ts)
    # Average advantage over time for each observable
    avg_advantage = np.mean(advantage, axis=1)  # (n_obs,)
    best_idx = int(np.argmax(avg_advantage))
    best_name = obs_specs[best_idx][0]

    # SE for the best observable's TE-PAI estimate
    if S > 1:
        best_tepai_se = (np.std(all_weighted[:, best_idx, :], axis=0, ddof=1)
                         / np.sqrt(S))
    else:
        best_tepai_se = np.zeros(len(tepai_t))

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Reference chi:   {chi_ref}")
    print(f"  Truncated chi:   {chi_trunc}")
    print(f"  Observables:     {n_obs}")
    print(f"  N_samples:       {args.N_samples}")
    print(f"  Trunc. Trotter avg |error| (final): {trunc_err[-1]:.6f}")
    print(f"  TE-PAI avg |error| (final):         {tepai_err[-1]:.6f}")
    print(f"  Best TE-PAI advantage observable:    {best_name} "
          f"(avg advantage = {avg_advantage[best_idx]:.6f})")
    print(f"{'=' * 60}")

    # ---- Plot ----
    if not args.no_plot:
        out = os.path.join(
            folder,
            args.plot_name or "truncation_comparison_multi.pdf",
        )
        plot_truncation_comparison_multi(
            tepai_t, trunc_err, tepai_err, tepai_err_se,
            chi_ref, chi_trunc, args.N_samples, n_obs,
            out,
        )

        base, ext = os.path.splitext(out)
        out_minimal = base + "_minimal" + ext
        plot_truncation_comparison_multi_minimal(
            tepai_t, trunc_err, tepai_err, tepai_err_se,
            chi_ref, chi_trunc, args.N_samples, n_obs,
            out_minimal,
        )

        # Best-advantage single-observable plot
        out_best = base + "_best" + ext
        plot_best_observable(
            tepai_t,
            ref_aligned[best_idx],
            trunc_aligned[best_idx],
            mean_weighted[best_idx],
            best_tepai_se,
            chi_ref, chi_trunc, args.N_samples, best_name,
            out_best,
        )


if __name__ == "__main__":
    main()
