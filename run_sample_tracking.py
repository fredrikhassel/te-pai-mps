#!/usr/bin/env python
"""run_sample_tracking.py — TE-PAI sample-level tracking experiment.

Like run_experiment.py but tracks every individual TE-PAI sample's weighted
observable for detailed visualization of the sampling process.

Produces a 2×1 figure:
  (1) Trotter + TE-PAI time series with individual sample scatter (gray dots)
      and analytical overhead envelope (red dashed + fill).
  (2) Per-circuit RMS error (green dashed) alongside analytical overhead (red
      dashed).

Usage (toy defaults):
    python run_sample_tracking.py

Production example:
    python run_sample_tracking.py --n-qubits 10 --total-time 1.0 \
        --N-samples 200 --N-tepai 50 --delta 0.003 --n-cores 8

Config-driven (via runner.py):
    Add experiments with "type": "sample_tracking" in config.json.
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
    build_trotter_prefix_gates,
    create_mps_circuit,
    apply_gates,
    measure,
)

from run_experiment import (
    DATA_DIR,
    _experiment_folder,
    _save_csv,
    _load_csv,
    _check_meta,
    load_or_run_trotter,
    _tepai_naive_worker,
    _tepai_naive_hybrid_worker,
)


# ===================================================================
#  File naming
# ===================================================================

def _tepai_st_basename(pi_over_delta, N, S, tstart):
    """File basename (no extension) for sample-tracking TE-PAI data."""
    return f"tepai_st_d{pi_over_delta}_N{N}_S{S}_tstart{tstart:.2f}"


# ===================================================================
#  Per-sample TE-PAI execution
# ===================================================================

def _run_tepai_sample_tracked(te_pai, circuits, nq, chi, op, mq, init, nw,
                               trotter_obs_aligned, tstart=0.0, prefix=None):
    """Execute all TE-PAI circuits and return per-sample weighted observables.

    Uses naive (unweighted) workers to get raw measurements, then applies
    the correct  sign × γ  weighting:

        weighted_obs[i, k] = (2·raw[i,k] − 1) · sign_i[k−1] · γ[k]

    Returns
    -------
    times : (n_timesteps,)
    mean_weighted : (n_timesteps,)
    rms : (n_timesteps,) — RMS deviation from Trotter
    mean_gates : (n_timesteps,)
    all_weighted : (n_samples, n_timesteps) — per-sample weighted observables
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gam = te_pai.gam_list
    n = len(circuits)

    # Use naive workers — raw measurements without sign/gamma pre-weighting
    if tstart > 0 and prefix is not None:
        fn = _tepai_naive_hybrid_worker
        inputs = [
            (nq, s, g, gam, chi, init, op, mq, prefix)
            for s, g in circuits
        ]
    else:
        fn = _tepai_naive_worker
        inputs = [
            (nq, s, g, gam, chi, init, op, mq)
            for s, g in circuits
        ]

    results = [None] * n
    if nw > 1 and n > 1:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(fn, inp): i for i, inp in enumerate(inputs)}
            for f in as_completed(futs):
                results[futs[f]] = f.result()
    else:
        for i, inp in enumerate(inputs):
            results[i] = fn(inp)

    all_raw = np.array([r[0] for r in results])            # (n, n_ts) [0,1]
    all_gates = np.array([r[2] for r in results], dtype=float)

    # Physical observables (unweighted)
    all_phys = 2 * all_raw - 1                              # (n, n_ts) [-1,1]

    # Per-sample signs from circuit tuples
    all_signs = np.array([signs for signs, _ in circuits])   # (n, n_snap)

    # Gamma values
    gam_arr = np.array(gam)                                  # (n_snap+1,)

    # Weighted observables: (2·raw − 1) × sign × γ
    n_ts = all_phys.shape[1]
    all_weighted = np.zeros_like(all_phys)
    all_weighted[:, 0] = all_phys[:, 0]                      # t=0: weight=1
    for k in range(1, n_ts):
        all_weighted[:, k] = (all_phys[:, k]
                              * all_signs[:, k - 1]
                              * gam_arr[k])

    mean_weighted = np.mean(all_weighted, axis=0)
    mean_gates = np.mean(all_gates, axis=0)

    # RMS deviation from Trotter reference
    deviations = all_weighted - trotter_obs_aligned[np.newaxis, :]
    rms = np.sqrt(np.mean(deviations ** 2, axis=0))

    dT = te_pai.T / te_pai.n_snap
    pts = te_pai.n_snap + 1
    times = tstart + np.arange(pts) * dT

    return times, mean_weighted, rms, mean_gates, all_weighted


# ===================================================================
#  Cache-aware runner
# ===================================================================

def load_or_run_tepai_samples(folder, basename, hamil, nq, T, dt, delta,
                               Nt, S, chi, op, mq, init, nw, seed,
                               trotter_obs, trotter_times,
                               tstart=0.0, N_prefix=None, n_snap_total=None):
    """Run TE-PAI with sample tracking, or load from cache.

    Saves aggregate data as CSV and per-sample weighted observables as NPZ.
    """
    csv_path = os.path.join(folder, basename + ".csv")
    npz_path = os.path.join(folder, basename + "_samples.npz")

    # Check cache
    if os.path.isfile(csv_path) and os.path.isfile(npz_path):
        meta, d = _load_csv(csv_path)
        expected = dict(type="tepai_st", delta=delta, N_tepai=Nt,
                        n_samples=S, tepai_start_time=tstart,
                        initial_state=init, seed=seed)
        bad = _check_meta(meta, expected)
        if bad:
            print(f"  [cache STALE] {basename}  — mismatch: "
                  + ", ".join(f"{k}: {g!s}≠{w!s}" for k, g, w in bad))
            print(f"  Re-running ...")
        else:
            print(f"  [cache] {basename}")
            npz = np.load(npz_path)
            return (d["time"], d["observable"], d["rms_error"],
                    d["gate_count"], d["overhead"], npz["all_weighted"])

    # Build TE-PAI
    dur = T - tstart
    n_snap_tepai = int(round(dur / dt))

    print(f"  Building TE-PAI (delta={delta:.4e}, N={Nt}, "
          f"snaps={n_snap_tepai}) ...")
    te = TE_PAI(hamil, nq, delta=delta, T=dur, N=Nt, n_snap=n_snap_tepai)

    if S < te.overhead ** 2:
        print(f"  Note: overhead={te.overhead:.1f} -> suggest "
              f"S>={int(np.ceil(te.overhead ** 2))}")

    # Generate circuits (no rejection sampling — keep all parities)
    print(f"  Generating {S} circuits ...")
    t0 = time.time()
    circuits = te.run_te_pai(S, n_workers=nw, seed=seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    # Build Trotter prefix for hybrid mode
    prefix = None
    if tstart > 0 and N_prefix is not None and n_snap_total is not None:
        ns_pre = int(round(tstart / dt))
        prefix = build_trotter_prefix_gates(
            hamil, T, N_prefix, n_snap_total, ns_pre,
        )

    # Align Trotter observable to TE-PAI timesteps
    tepai_times = tstart + np.arange(n_snap_tepai + 1) * dt
    trotter_obs_aligned = np.interp(tepai_times, trotter_times, trotter_obs)

    # Execute
    print(f"  Executing ({nw} workers) ...")
    t0 = time.time()
    times, mean_obs, rms, mean_gates, all_weighted = \
        _run_tepai_sample_tracked(
            te, circuits, nq, chi, op, mq, init, nw,
            trotter_obs_aligned, tstart, prefix,
        )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Per-timestep analytical overhead
    dT = dur / n_snap_tepai
    tan_half_delta = np.tan(delta / 2)
    oh_arr = np.array([
        np.exp(2 * hamil.l1_norm(k * dT) * tan_half_delta)
        for k in range(n_snap_tepai + 1)
    ])

    # Save aggregate CSV
    meta = dict(
        type="tepai_st", delta=delta, N_tepai=Nt, n_samples=S,
        tepai_start_time=tstart, initial_state=init, seed=seed,
        overhead_final=te.overhead, gamma_final=te.gamma_final,
        expected_num_gates=te.expected_num_gates,
    )
    _save_csv(
        csv_path, meta,
        ["time", "observable", "rms_error", "gate_count", "overhead"],
        [times, mean_obs, rms, mean_gates, oh_arr],
    )
    print(f"  Saved -> {csv_path}")

    # Save per-sample data
    np.savez_compressed(npz_path, all_weighted=all_weighted)
    print(f"  Saved -> {npz_path}")

    return times, mean_obs, rms, mean_gates, oh_arr, all_weighted


# ===================================================================
#  Plotting
# ===================================================================

def plot_sample_tracking(trot_t, trot_obs,
                          tepai_t, tepai_obs, tepai_rms,
                          all_weighted, overhead_arr,
                          trotter_obs_aligned, tstart, out_path,
                          n_samples=1, hist_time=8.0):
    """3×1 subplot: sample scatter + RMS vs overhead + histogram at hist_time."""
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, axes = plt.subplot_mosaic(
        [["A", "A"], ["B", "C"]],
        figsize=(14, 9),
        gridspec_kw={"hspace": 0.3, "wspace": 0.2,
                     "height_ratios": [3, 2]},
    )
    ax1 = axes["A"]   # top: time series (full width)
    ax3 = axes["B"]   # bottom-left: histogram
    ax2 = axes["C"]   # bottom-right: variance / overhead

    # Bold panel labels above top-left corner of each axis
    for label, ax in (("A)", ax1), ("B)", ax3), ("C)", ax2)):
        ax.text(0.01, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    # ---- Top: time series with sample scatter and overhead envelope ----

    # Overhead envelope around Trotter reference (aligned to TE-PAI times)
    upper = trotter_obs_aligned + overhead_arr
    lower = trotter_obs_aligned - overhead_arr
    ax1.fill_between(tepai_t, lower, upper, color="red", alpha=0.08)
    ax1.plot(tepai_t, upper, "r--", lw=1.2)
    ax1.plot(tepai_t, lower, "r--", lw=1.2)

    # Individual sample scatter (gray dots with low alpha; overlapping → darker)
    alpha = 0.12
    t_tiled = np.tile(tepai_t, (n_samples, 1))       # (S, n_ts)

    # Overhead-normalized sample scatter (blue dots) — plotted first for legend order
    all_normalized = all_weighted / overhead_arr[np.newaxis, :]
    ax1.scatter(
        t_tiled.ravel(), all_normalized.ravel(),
        color="tab:blue", alpha=alpha, s=6, edgecolors="none",
        rasterized=True, zorder=2, label="Expected value in circuit sample",
    )

    ax1.scatter(
        t_tiled.ravel(), all_weighted.ravel(),
        color="gray", alpha=alpha, s=6, edgecolors="none",
        rasterized=True, zorder=1, label="Scaled samples",
    )

    # Trotter reference (black solid)
    ax1.plot(trot_t, trot_obs, color="black", lw=2.0, zorder=3, label="Trotter")

    # TE-PAI aggregate mean (green solid)
    ax1.plot(tepai_t, tepai_obs, color="tab:green", lw=2.0, zorder=4, label="TE-PAI")

    if tstart > 0:
        ax1.axvline(tstart, color="tab:green", ls=":", lw=1)

    ax1.set_ylim(-10, 10)
    ax1.set_xlim(0, trot_t[-1])
    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.set_xlabel(r"Time ($T$)")
    leg1 = ax1.legend(loc="upper left", markerscale=4)
    for lh in leg1.legend_handles:
        lh.set_alpha(0.8)

    # ---- Bottom: per-circuit RMS error + analytical overhead ----

    ax2.plot(tepai_t, tepai_rms, color="tab:green", ls="--", lw=2.0,
             label="Empirical variance")

    ax2.plot(tepai_t, overhead_arr, color="red", ls="--", lw=2.0,
             label="Theoretical overhead")

    # Configuration standard deviation: RMS / overhead
    config_std = tepai_rms / np.where(overhead_arr != 0, overhead_arr, 1.0)
    ax2.plot(tepai_t, config_std, color="tab:blue", ls="--", lw=2.0,
             label="Configuration variance")

    if tstart > 0:
        ax2.axvline(tstart, color="tab:green", ls=":", lw=1)

    ax2.set_xlabel(r"Time ($T$)")
    ax2.set_xlim(0, trot_t[-1])
    ax2.set_yticklabels([])
    ax2.set_yscale("log")
    ax2.legend(loc="upper left")

    # ---- Middle: histogram of TE-PAI shot distribution at hist_time ----

    # Find the time index closest to hist_time
    t_idx = np.argmin(np.abs(tepai_t - hist_time))
    actual_t = tepai_t[t_idx]

    # Extract per-sample values at this time
    sample_vals = all_weighted[:, t_idx]

    # Trotter canonical value at this time
    trotter_val = trotter_obs_aligned[t_idx]

    ax3.hist(sample_vals, bins=30, color="gray", edgecolor="darkgray",
             alpha=0.7, label="Scaled samples")
    sample_mean = np.mean(sample_vals)
    ax3.axvline(sample_mean, color="tab:green", ls="-", lw=2.0, label="Sample mean")
    ax3.axvline(trotter_val, color="black", ls=":", lw=2.0, label="Trotter")
    ax3.set_ylabel("Sample count")
    ax3.set_xlabel("Scaled sample value")
    ax3.text(0.02, 0.95, rf"$T={actual_t:.1f}$", transform=ax3.transAxes,
             va="top", ha="left")
    ax3.legend(loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.96)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  CLI + main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="TE-PAI sample tracking experiment with data caching.",
    )
    # Physics / simulation
    p.add_argument("--n-qubits",    type=int,   default=10)
    p.add_argument("--total-time",  type=float, default=0.5)
    p.add_argument("--dt",          type=float, default=0.1)
    p.add_argument("--j",           type=float, default=0.1)
    p.add_argument("--max-bond",    type=int,   default=16)
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

    # Single start time only (take first if multiple provided)
    tstart = args.tepai_start_time[0]
    if len(args.tepai_start_time) > 1:
        print(f"  Warning: sample tracking uses single start time only, "
              f"using tstart={tstart}")

    if tstart > 0:
        assert abs(tstart / dt - round(tstart / dt)) < 1e-9, \
            f"tepai-start-time ({tstart}) must be a multiple of dt={dt}"
        assert tstart < T, f"tepai-start-time {tstart} must be < total-time {T}"

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    print(f"Sample tracking: {args.n_qubits}q, chi={args.max_bond}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, "
          f"S={args.N_samples}, workers={nw}")

    # Data directory (shared with run_experiment.py)
    folder = os.path.join(
        DATA_DIR,
        _experiment_folder(
            args.n_qubits, args.max_bond, args.j, args.seed,
            T, dt, args.operator, args.measure_qubit, args.initial_state,
        ),
    )
    os.makedirs(folder, exist_ok=True)
    print(f"  Data dir: {folder}\n")

    # ---- Trotter reference (reuses cache from run_experiment.py) ----
    trot_fname = f"trotter_N{args.N_trotter}_adaptiveTrue.csv"
    print("Trotter reference:")
    trot_t, trot_obs, _trot_bonds, _trot_costs, _trot_gates = load_or_run_trotter(
        folder, trot_fname, hamil, args.n_qubits, T,
        args.N_trotter, n_snap, args.max_bond,
        args.operator, args.measure_qubit, args.initial_state, True,
    )

    # ---- TE-PAI with sample tracking ----
    pi_over_delta = round(np.pi / args.delta)
    basename = _tepai_st_basename(
        pi_over_delta, args.N_tepai, args.N_samples, tstart,
    )
    print("\nTE-PAI (sample tracking):")
    tepai_t, tepai_obs, tepai_rms, tepai_gates, oh_arr, all_weighted = \
        load_or_run_tepai_samples(
            folder, basename, hamil, args.n_qubits, T, dt,
            args.delta, args.N_tepai, args.N_samples,
            args.max_bond, args.operator, args.measure_qubit,
            args.initial_state, nw, args.seed,
            trotter_obs=trot_obs, trotter_times=trot_t,
            tstart=tstart, N_prefix=args.N_trotter, n_snap_total=n_snap,
        )

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Samples tracked:  {all_weighted.shape[0]}")
    print(f"  Timesteps:        {all_weighted.shape[1]}")
    print(f"  Final overhead:   {oh_arr[-1]:.4f}")
    print(f"  Final RMS error:  {tepai_rms[-1]:.6f}")
    trot_gate_final = _trot_gates[-1]
    tepai_gate_final = tepai_gates[-1]
    ratio = trot_gate_final / tepai_gate_final if tepai_gate_final > 0 else float("inf")
    print(f"  Trotter gates:    {trot_gate_final:.0f}")
    print(f"  TE-PAI gates:     {tepai_gate_final:.0f}")
    print(f"  Ratio (Trot/PAI): {ratio:.2f}x")
    print(f"{'=' * 60}")

    # ---- Plot ----
    if not args.no_plot:
        # Align Trotter to TE-PAI timesteps for overhead envelope
        dur = T - tstart
        n_snap_tepai = int(round(dur / dt))
        tepai_times_arr = tstart + np.arange(n_snap_tepai + 1) * dt
        trotter_aligned = np.interp(tepai_times_arr, trot_t, trot_obs)

        out = os.path.join(folder, args.plot_name or "sample_tracking.pdf")
        plot_sample_tracking(
            trot_t, trot_obs,
            tepai_t, tepai_obs, tepai_rms,
            all_weighted, oh_arr,
            trotter_aligned, tstart, out,
            n_samples=args.N_samples,
        )


if __name__ == "__main__":
    main()
