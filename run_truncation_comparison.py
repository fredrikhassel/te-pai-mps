#!/usr/bin/env python
"""run_truncation_comparison.py — Compare truncated Trotterization vs TE-PAI.

Runs three simulations:
  1. Reference Trotter at high bond dimension (max_bond)
  2. Truncated Trotter at low bond dimension (max_bond_truncated)
  3. TE-PAI at the same low bond dimension, aggregated over N_samples

Produces a single-panel plot:
  - Reference Trotter: black dashed line
  - Truncated Trotter (chi=low): black solid line
  - TE-PAI (chi=low): tab:green solid line

Config-driven (via runner.py):
    Add experiments with "type": "truncation_comparison" in config.json.
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

from mps_tepai import Hamiltonian

from run_experiment import (
    DATA_DIR,
    _experiment_folder,
    _trotter_file,
    _tepai_file,
    load_or_run_trotter,
    load_or_run_tepai,
)


# ===================================================================
#  Plotting
# ===================================================================

def plot_truncation_comparison(ref_t, ref_obs,
                                trunc_t, trunc_obs,
                                tepai_t, tepai_obs, tepai_se,
                                chi_ref, chi_trunc, n_samples,
                                out_path):
    """Single-panel plot comparing reference Trotter, truncated Trotter, and TE-PAI."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Reference Trotter (black dashed)
    ax.plot(ref_t, ref_obs, color="black", ls="--", lw=1.5,
            label=rf"Trotter $\chi={chi_ref}$ (reference)")

    # Truncated Trotter (black solid)
    ax.plot(trunc_t, trunc_obs, color="black", lw=1.5,
            label=rf"Trotter $\chi={chi_trunc}$")

    # TE-PAI (green solid with error band)
    ax.plot(tepai_t, tepai_obs, color="tab:green", lw=1.5,
            label=rf"TE-PAI $\chi={chi_trunc}$")
    if n_samples > 1:
        ax.fill_between(
            tepai_t, tepai_obs - tepai_se, tepai_obs + tepai_se,
            color="tab:green", alpha=0.2,
        )

    ax.set_xlim(0, ref_t[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle X_0 \rangle$")
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


def plot_truncation_comparison_minimal(ref_t, ref_obs,
                                       trunc_t, trunc_obs,
                                       tepai_t, tepai_obs, tepai_se,
                                       chi_ref, chi_trunc, n_samples,
                                       out_path):
    """Wide, minimal version of the truncation comparison plot."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fs = 18  # base font size for this plot

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    lw = 2.5

    ax.plot(ref_t, ref_obs, color="black", ls="--", lw=lw,
            label=rf"Trotter $\chi={chi_ref}$ (ref.)")
    ax.plot(trunc_t, trunc_obs, color="black", lw=lw,
            label=rf"Trotter $\chi={chi_trunc}$")
    ax.plot(tepai_t, tepai_obs, color="tab:green", lw=lw,
            label=rf"TE-PAI $\chi={chi_trunc}$")
    if n_samples > 1:
        ax.fill_between(
            tepai_t, tepai_obs - tepai_se, tepai_obs + tepai_se,
            color="tab:green", alpha=0.2,
        )

    ax.set_xlim(0, ref_t[-1])
    ax.set_ylabel(r"$\langle X_0 \rangle (t)$", fontsize=fs)
    ax.set_xlabel("Time", fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs - 2)
    ax.tick_params(axis="y", labelsize=fs - 2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

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
        description="Truncation comparison: reference Trotter vs "
                    "truncated Trotter vs TE-PAI.",
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
    p.add_argument("--operator",    type=str,   default="X")
    p.add_argument("--measure-qubit", type=int, default=0)
    p.add_argument("--initial-state", type=str, default="plus_flip")
    p.add_argument("--seed",        type=int,   default=0)

    # Algorithm
    p.add_argument("--N-trotter",   type=int,   default=20)
    p.add_argument("--N-tepai",     type=int,   default=50)
    p.add_argument("--N-samples",   type=int,   default=100)
    p.add_argument("--delta",       type=float, default=np.pi / 1024)
    p.add_argument("--tepai-start-time", type=float, nargs='+', default=[0.0])

    # Runtime
    p.add_argument("--n-cores",     type=int,   default=None)
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("--plot-name",   type=str,   default=None,
                   help="Filename for the plot SVG (saved inside data folder)")
    return p.parse_args()


def main():
    args = parse_args()

    nw = args.n_cores or max(1, (mp.cpu_count() or 4) - 2)
    T = args.total_time
    dt = args.dt
    n_snap = int(round(T / dt))
    assert n_snap >= 1, f"total-time must be >= dt={dt}"

    tstart = args.tepai_start_time[0]

    chi_ref = args.max_bond
    chi_trunc = args.max_bond_truncated

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    print(f"Truncation comparison: {args.n_qubits}q, "
          f"chi_ref={chi_ref}, chi_trunc={chi_trunc}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, "
          f"S={args.N_samples}, workers={nw}")

    # ---- Reference Trotter (high chi) ----
    ref_folder = os.path.join(
        DATA_DIR,
        _experiment_folder(
            args.n_qubits, chi_ref, args.j, args.seed,
            T, dt, args.operator, args.measure_qubit, args.initial_state,
        ),
    )
    os.makedirs(ref_folder, exist_ok=True)

    trot_fname = _trotter_file(args.N_trotter, False)
    print(f"\nReference Trotter (chi={chi_ref}):")
    ref_t, ref_obs, _, _, _ = load_or_run_trotter(
        ref_folder, trot_fname, hamil, args.n_qubits, T,
        args.N_trotter, n_snap, chi_ref,
        args.operator, args.measure_qubit, args.initial_state, False,
    )

    # ---- Truncated Trotter (low chi) ----
    trunc_folder = os.path.join(
        DATA_DIR,
        _experiment_folder(
            args.n_qubits, chi_trunc, args.j, args.seed,
            T, dt, args.operator, args.measure_qubit, args.initial_state,
        ),
    )
    os.makedirs(trunc_folder, exist_ok=True)

    print(f"\nTruncated Trotter (chi={chi_trunc}):")
    trunc_t, trunc_obs, _, _, _ = load_or_run_trotter(
        trunc_folder, trot_fname, hamil, args.n_qubits, T,
        args.N_trotter, n_snap, chi_trunc,
        args.operator, args.measure_qubit, args.initial_state, False,
    )

    # ---- TE-PAI (low chi) ----
    pi_over_delta = round(np.pi / args.delta)
    tepai_fname = _tepai_file(
        pi_over_delta, args.N_tepai, args.N_samples, tstart,
    )
    print("\nTE-PAI:")
    tepai_t, tepai_obs, tepai_rms, _, _, _, _ = load_or_run_tepai(
        trunc_folder, tepai_fname, hamil, args.n_qubits, T, dt,
        args.delta, args.N_tepai, args.N_samples,
        chi_trunc, args.operator, args.measure_qubit,
        args.initial_state, nw, args.seed,
        trotter_obs=ref_obs, trotter_times=ref_t,
        tstart=tstart, N_prefix=args.N_trotter, n_snap_total=n_snap,
    )

    tepai_se = tepai_rms / np.sqrt(args.N_samples)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Reference chi:   {chi_ref}")
    print(f"  Truncated chi:   {chi_trunc}")
    print(f"  N_samples:       {args.N_samples}")
    print(f"{'=' * 60}")

    # ---- Plot ----
    if not args.no_plot:
        out = os.path.join(
            trunc_folder,
            args.plot_name or "truncation_comparison.pdf",
        )
        plot_truncation_comparison(
            ref_t, ref_obs,
            trunc_t, trunc_obs,
            tepai_t, tepai_obs, tepai_se,
            chi_ref, chi_trunc, args.N_samples,
            out,
        )

        # Wide minimal version
        base, ext = os.path.splitext(out)
        out_minimal = base + "_minimal" + ext
        plot_truncation_comparison_minimal(
            ref_t, ref_obs,
            trunc_t, trunc_obs,
            tepai_t, tepai_obs, tepai_se,
            chi_ref, chi_trunc, args.N_samples,
            out_minimal,
        )


if __name__ == "__main__":
    main()
