#!/usr/bin/env python
"""run_resource_estimation.py — Resource estimation: Trotter depth vs TE-PAI shots.

Compares the accuracy of Trotterization at different depths against the number
of TE-PAI shots needed to achieve the same accuracy.

Pipeline:
  1. Run a deep canonical (non-adaptive) Trotter simulation as reference truth.
  2. Run 10 shallower non-adaptive Trotter simulations and compute their final
     error relative to the canonical value.
  3. Run a TE-PAI simulation with many samples (sample tracking) to obtain the
     per-shot RMS error at the final timestep.
  4. Plot: TE-PAI expected error (RMS/√N_s) vs N_s alongside horizontal Trotter
     error lines, colour-coded by depth (viridis).

Config-driven (via runner.py):
    Add experiments with "type": "resource_estimation" in config.json.

Quick toy test:
    python run_resource_estimation.py --n-qubits 10 --total-time 1.0 \
        --N-trotter-canon 500 --N-samples 100 --j 1.0 --max-bond 8
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
    measure,
)

from run_experiment import (
    DATA_DIR,
    _experiment_folder,
    _save_csv,
    _load_csv,
    _check_meta,
)

from run_sample_tracking import (
    _tepai_st_basename,
    load_or_run_tepai_samples,
)

import glob as _glob
import re as _re


# ===================================================================
#  Find largest cached N_samples for a given delta
# ===================================================================

def _find_best_samples(folder, pi_over_delta, N_tepai, tstart):
    """Find the cached tepai_st file with the largest S for this delta.

    Returns the best S (int) if found, else None.
    """
    pattern = os.path.join(
        folder,
        f"tepai_st_d{pi_over_delta}_N{N_tepai}_S*_tstart{tstart:.2f}.csv",
    )
    matches = _glob.glob(pattern)
    if not matches:
        return None
    best_S = 0
    for m in matches:
        base = os.path.basename(m)
        hit = _re.search(r"_S(\d+)_", base)
        if hit:
            best_S = max(best_S, int(hit.group(1)))
    return best_S if best_S > 0 else None


# ===================================================================
#  Non-adaptive Trotter — full time series, cached
# ===================================================================

def _run_trotter_timeseries(hamil, nq, T, N, chi, op, mq, init):
    """Non-adaptive linear Trotter with measurement at every step.

    Returns (times, obs) arrays of length N+1 (including t=0).
    """
    circ = create_mps_circuit(nq, init, chi)

    ts_eval = np.linspace(0, T, N)
    terms = [hamil.get_term(t) for t in ts_eval]

    times = [0.0]
    obs = [2 * measure(circ, op, mq) - 1]

    for i in range(N):
        gates = [(p, 2 * c * T / N, idx) for p, idx, c in terms[i]]
        apply_gates(circ, gates)
        times.append((i + 1) * T / N)
        obs.append(2 * measure(circ, op, mq) - 1)

    return np.array(times), np.array(obs)


def load_or_run_re_trotter(folder, N, hamil, nq, T, chi, op, mq, init):
    """Cache-aware wrapper: run full time series, cache as NPZ.

    Returns (times, obs) arrays of length N+1.
    """
    fname = f"re_trotter_N{N}.npz"
    path = os.path.join(folder, fname)

    if os.path.isfile(path):
        npz = np.load(path)
        meta_N = int(npz["N"])
        meta_init = str(npz["initial_state"])
        if meta_N == N and meta_init == init:
            print(f"  [cache] {fname}")
            return npz["times"], npz["obs"]
        print(f"  [cache STALE] {fname}")

    print(f"  Running Trotter N={N} ...", end="", flush=True)
    t0 = time.time()
    times, obs = _run_trotter_timeseries(
        hamil, nq, T, N, chi, op, mq, init,
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s  final obs={obs[-1]:.6f}")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    np.savez_compressed(path, times=times, obs=obs,
                        N=np.array(N), initial_state=np.array(init))
    print(f"  Saved -> {path}")
    return times, obs


# ===================================================================
#  Resource-estimation summary cache
# ===================================================================

def _save_re_summary(path, canon_N, canon_obs, trotter_Ns, trotter_errors,
                     rms_per_shot, n_samples):
    """Save resource-estimation summary to a CSV for reproducibility."""
    meta = dict(
        type="re_summary",
        canon_N=canon_N,
        canon_obs=canon_obs,
        rms_per_shot=rms_per_shot,
        n_samples=n_samples,
    )
    _save_csv(
        path, meta,
        ["trotter_N", "trotter_error"],
        [np.array(trotter_Ns, dtype=float),
         np.array(trotter_errors, dtype=float)],
    )


# ===================================================================
#  Plotting
# ===================================================================

def plot_resource_estimation(trotter_Ns, trotter_errors, tepai_curves,
                              out_path):
    """Plot TE-PAI expected error curves with Trotter accuracy lines.

    Parameters
    ----------
    trotter_Ns : list[int]
    trotter_errors : list[float]
    tepai_curves : list[dict]
        Each dict has keys: pi_over_delta, rms_per_shot, N_samples
    out_path : str
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Determine N_s range from all curves ---
    min_err = min(e for e in trotter_errors if e > 0)
    all_rms = [c["rms_per_shot"] for c in tepai_curves]
    N_s_max = max(
        int(np.ceil((rms / min_err) ** 2)) * 3
        for rms in all_rms
    )
    N_s_max = max(N_s_max, 200)
    N_s = np.logspace(0, np.log10(N_s_max), 500)

    # --- TE-PAI 1/√N_s curves (one per delta) ---
    tepai_colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:purple"]
    for idx, curve in enumerate(tepai_curves):
        color = tepai_colors[idx % len(tepai_colors)]
        epsilon = curve["rms_per_shot"] / np.sqrt(N_s)
        ax.plot(N_s, epsilon, color=color, lw=2, zorder=5)

    # --- Trotter horizontal lines (viridis: shallowest=0 → deepest=1) ---
    n_levels = len(trotter_Ns)
    norm = Normalize(vmin=0, vmax=max(n_levels - 1, 1))
    from matplotlib.colors import LinearSegmentedColormap

    def truncated_cmap(cmap, minval=0.0, maxval=0.9, n=256):
        return LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )

    cmap = truncated_cmap(plt.get_cmap("viridis"), 0.0, 0.85)

    for i, (N_trot, err) in enumerate(zip(trotter_Ns, trotter_errors)):
        color = cmap(norm(i))
        ax.axhline(err, color=color, ls="--", lw=2, alpha=1.0)

    # --- Axis formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_s$")
    ax.set_ylabel(r"$\epsilon$")

    # --- Annotate Trotter lines with N_trot on the right edge ---
    for i, (N_trot, err) in enumerate(zip(trotter_Ns, trotter_errors)):
        color = cmap(norm(i))
        err = err*1.15
        if i != len(trotter_Ns) - 1:
            ax.annotate(
                rf"$N={N_trot}$",
                xy=(0.7, err), xycoords=("axes fraction", "data"),
                va="center", ha="left", fontsize=14, color=color,
            )
        else:
            ax.annotate(
                rf"$N={N_trot}$",
                xy=(0.2, err), xycoords=("axes fraction", "data"),
                va="center", ha="left", fontsize=14, color=color,
            )


    # --- Annotate TE-PAI curves ---
    for idx, curve in enumerate(tepai_curves):
        color = tepai_colors[idx % len(tepai_colors)]
        epsilon = curve["rms_per_shot"] / np.sqrt(N_s)
        # Place label near the middle of the curve
        mid = len(N_s) // 2 + 140
        ax.annotate(
            r"$\mathrm{TE\text{-}PAI}$ error",
            xy=(N_s[mid], epsilon[mid]),
            xytext=(40, 4), textcoords="offset points",
            va="bottom", ha="center", fontsize=14, color=color,
        )

    ax.set_xlim(1, N_s_max)
    plt.savefig(out_path, bbox_inches="tight")
    plt.tight_layout()
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


def plot_timeseries(canon_t, canon_obs, shallow_runs, tepai_t, tepai_obs,
                    tepai_rms=None):
    """Quick diagnostic: all Trotters + TE-PAI mean as time series.

    Top : observable time series
        canon  : black solid
        shallow: viridis colour-coded shallowest → deepest
        TE-PAI : red solid
    Bottom : per-circuit RMS error over time (if tepai_rms provided)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    has_rms = tepai_rms is not None
    if has_rms:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                       height_ratios=[3, 1],
                                       sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_levels = len(shallow_runs)
    norm = Normalize(vmin=0, vmax=max(n_levels - 1, 1))
    cmap = plt.get_cmap("viridis")

    # Shallow Trotters (draw first so canonical sits on top)
    for i, (N, t, obs) in enumerate(shallow_runs):
        color = cmap(norm(i))
        ax.plot(t, obs, color=color, lw=1, alpha=0.8)

    # Canonical Trotter
    ax.plot(canon_t, canon_obs, color="black", lw=2, label="Canonical Trotter")

    # TE-PAI sample mean
    ax.plot(tepai_t, tepai_obs, color="red", lw=2, label="TE-PAI mean")

    ax.set_ylabel(r"$\langle X_0 \rangle$")
    if not has_rms:
        ax.set_xlabel("Time")

    # Legend: canonical + TE-PAI + viridis gradient summary
    handles = [
        Line2D([0], [0], color="black", lw=2),
        Line2D([0], [0], color="red", lw=2),
    ]
    labels = ["Canonical Trotter", "TE-PAI mean"]
    for i, (N, _, _) in enumerate(shallow_runs):
        handles.append(Line2D([0], [0], color=cmap(norm(i)), lw=1.5))
        labels.append(f"$N={N}$")

    ax.legend(handles, labels, loc="best", fontsize=9, ncol=2)

    # --- Bottom: per-circuit RMS error over time ---
    if has_rms:
        ax2.plot(tepai_t, tepai_rms, color="red", lw=1.5)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Per-circuit RMS")

    fig.tight_layout()
    plt.show()


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Resource estimation: Trotter depth vs TE-PAI shots.",
    )
    # Physics / simulation
    p.add_argument("--n-qubits",       type=int,   default=20)
    p.add_argument("--total-time",     type=float, default=10.0)
    p.add_argument("--dt",             type=float, default=0.1)
    p.add_argument("--j",              type=float, default=1.0)
    p.add_argument("--max-bond",       type=int,   default=8)
    p.add_argument("--operator",       type=str,   default="X")
    p.add_argument("--measure-qubit",  type=int,   default=0)
    p.add_argument("--initial-state",  type=str,   default="plus_flip")
    p.add_argument("--seed",           type=int,   default=0)

    # Trotter
    p.add_argument("--N-trotter",      type=int,   default=20,
                   help="(ignored, kept for runner.py compatibility)")
    p.add_argument("--N-trotter-canon", type=int,  default=5000,
                   help="Deep canonical (non-adaptive) Trotter N (reference)")
    p.add_argument("--trotter-levels", type=int, nargs='+', default=[10],
                   help="Shallow Trotter depths: a single int N auto-generates "
                        "N geom-spaced levels; multiple ints are used directly")

    # TE-PAI
    p.add_argument("--N-tepai",        type=int,   default=50)
    p.add_argument("--N-samples",      type=int,   default=500)
    p.add_argument("--delta",          type=float, default=None,
                   help="(Legacy) single delta value; prefer --pi-over-delta")
    p.add_argument("--pi-over-delta",  type=int, nargs='+', default=None,
                   help="One or more pi/delta values (e.g. 1024 2048)")
    p.add_argument("--tepai-start-time", type=float, nargs='+', default=[0.0])

    # Runtime
    p.add_argument("--n-cores",        type=int,   default=None)
    p.add_argument("--no-plot",        action="store_true")
    p.add_argument("--plot-name",      type=str,   default=None,
                   help="Filename for the plot SVG")
    p.add_argument("--plot-timeseries", action="store_true",
                   help="Show a diagnostic time-series plot of all Trotters "
                        "and TE-PAI (display only, nothing saved)")
    p.add_argument("--final-step-only", action="store_true",
                   help="(Deprecated) Compare accuracy at the final time step "
                        "only instead of using time-series RMSE. The final-step "
                        "method can produce inconsistent results because a single "
                        "time point is a noisy estimator of overall accuracy.")
    p.add_argument("--empirical-error", action="store_true",
                   help="Use the empirical aggregate error at the final time "
                        "step instead of per-shot RMS. The per-shot sigma is "
                        "back-derived as sigma = |mean - canon| * sqrt(N_samples) "
                        "so that sigma/sqrt(N_s) reproduces the observed error "
                        "at N_s = N_samples.")
    return p.parse_args()


# ===================================================================
#  Main
# ===================================================================

def main():
    args = parse_args()

    nw = args.n_cores or max(1, (mp.cpu_count() or 4) - 2)
    T = args.total_time
    dt = args.dt
    n_snap = int(round(T / dt))
    assert n_snap >= 1, f"total-time must be >= dt={dt}"

    N_canon = args.N_trotter_canon

    # Resolve pi_over_delta list
    if args.pi_over_delta is not None:
        pi_over_deltas = args.pi_over_delta
    elif args.delta is not None:
        pi_over_deltas = [round(np.pi / args.delta)]
    else:
        pi_over_deltas = [128]  # fallback default

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    print(f"Resource Estimation: {args.n_qubits}q, chi={args.max_bond}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, workers={nw}")
    print(f"  Canonical Trotter N={N_canon}, "
          f"pi/delta={pi_over_deltas}, N_tepai={args.N_tepai}")

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

    # ================================================================
    # 1. Canonical deep Trotter (non-adaptive) — reference truth
    # ================================================================
    print("=" * 60)
    print("Canonical Trotter (non-adaptive, reference):")
    print("=" * 60)
    canon_t, canon_obs = load_or_run_re_trotter(
        folder, N_canon, hamil, args.n_qubits, T,
        args.max_bond, args.operator, args.measure_qubit, args.initial_state,
    )
    canon_final = canon_obs[-1]
    print(f"  Canonical final obs = {canon_final:.8f}")

    # ================================================================
    # 2. Shallow Trotters
    # ================================================================
    levels_arg = args.trotter_levels
    if len(levels_arg) == 1:
        # Single int → auto-generate that many geom-spaced levels
        n_levels = levels_arg[0]
        N_min = max(5, N_canon // 200)
        N_max = N_canon // 2
        N_levels = np.unique(
            np.geomspace(N_min, N_max, n_levels).astype(int)
        )
        if len(N_levels) < n_levels:
            N_levels = np.unique(
                np.geomspace(max(2, N_min // 2), N_max, n_levels + 5
                             ).astype(int)
            )[:n_levels]
    else:
        # Explicit list of N values
        N_levels = np.unique(np.array(levels_arg))

    print(f"\n{'=' * 60}")
    print(f"Shallow Trotters ({len(N_levels)} levels): "
          f"{[int(n) for n in N_levels]}")
    print("=" * 60)

    trotter_Ns = []
    trotter_errors = []
    trotter_runs = []              # (N, times, obs) for timeseries plot
    for N in N_levels:
        N = int(N)
        t_arr, obs_arr = load_or_run_re_trotter(
            folder, N, hamil, args.n_qubits, T,
            args.max_bond, args.operator, args.measure_qubit,
            args.initial_state,
        )
        if args.final_step_only or args.empirical_error:
            err = abs(obs_arr[-1] - canon_final)
        else:
            # Time-series RMSE: interpolate to common grid with canonical
            common_t = np.linspace(0, T, n_snap + 1)
            obs_interp = np.interp(common_t, t_arr, obs_arr)
            canon_interp = np.interp(common_t, canon_t, canon_obs)
            err = np.sqrt(np.mean((obs_interp - canon_interp) ** 2))
        trotter_Ns.append(N)
        trotter_errors.append(err)
        trotter_runs.append((N, t_arr, obs_arr))

    gates_per_step = len(hamil.get_term(0))
    print(f"\n  {'N':>6s}  {'gates':>10s}  {'error':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}")
    for N, err in zip(trotter_Ns, trotter_errors):
        print(f"  {N:6d}  {N * gates_per_step:10d}  {err:12.6e}")

    # ================================================================
    # 3. TE-PAI sample tracking — one run per pi/delta value
    # ================================================================
    tstart = args.tepai_start_time[0]
    ref_times = np.linspace(0, T, n_snap + 1)
    ref_obs = np.full_like(ref_times, canon_final)

    tepai_curves = []  # list of dicts for plotting & summary
    first_tepai_run = None  # store first run for timeseries plot

    for pod in pi_over_deltas:
        delta = np.pi / pod

        # Find biggest cached N_samples for this delta.
        # Use the cache only if it meets or exceeds the requested N_samples;
        # otherwise honour the user's request (which will trigger a new run).
        best_S = _find_best_samples(folder, pod, args.N_tepai, tstart)
        if best_S is not None and best_S >= args.N_samples:
            S_use = best_S
            print(f"\n  [auto] Found cached S={S_use} for pi/delta={pod}")
        elif best_S is not None:
            S_use = args.N_samples
            print(f"\n  [upgrade] Cache has S={best_S} for pi/delta={pod}, "
                  f"but requested S={S_use} — will run new")
        else:
            S_use = args.N_samples
            print(f"\n  [fallback] No cache for pi/delta={pod}, "
                  f"using supplied S={S_use}")

        basename = _tepai_st_basename(pod, args.N_tepai, S_use, tstart)

        print(f"\n{'=' * 60}")
        print(f"TE-PAI sample tracking (pi/delta={pod}, S={S_use}):")
        print("=" * 60)
        tepai_t, tepai_obs, tepai_rms, tepai_gates, oh_arr, all_weighted = \
            load_or_run_tepai_samples(
                folder, basename, hamil, args.n_qubits, T, dt,
                delta, args.N_tepai, S_use,
                args.max_bond, args.operator, args.measure_qubit,
                args.initial_state, nw, args.seed,
                trotter_obs=ref_obs, trotter_times=ref_times,
                tstart=tstart, N_prefix=None, n_snap_total=n_snap,
            )

        # Compute per-shot RMS error vs canonical Trotter
        if args.empirical_error:
            tepai_mean_final = np.mean(all_weighted[:, -1])
            empirical_err = abs(tepai_mean_final - canon_final)
            rms = empirical_err * np.sqrt(S_use)
        elif args.final_step_only:
            per_sample_final = all_weighted[:, -1]
            rms = np.sqrt(np.mean((per_sample_final - canon_final) ** 2))
        else:
            canon_at_tepai = np.interp(tepai_t, canon_t, canon_obs)
            deviations = all_weighted - canon_at_tepai[np.newaxis, :]
            rms = np.sqrt(np.mean(deviations ** 2))

        # Mean TE-PAI gate count (at final timestep)
        tepai_mean_gates = float(tepai_gates[-1]) if len(tepai_gates) > 0 else 0.0

        tepai_curves.append(dict(
            pi_over_delta=pod,
            delta=delta,
            rms_per_shot=rms,
            N_samples=S_use,
            mean_gates=tepai_mean_gates,
            tepai_t=tepai_t,
            tepai_obs=tepai_obs,
            tepai_rms=tepai_rms,
        ))

        if first_tepai_run is None:
            first_tepai_run = (tepai_t, tepai_obs, tepai_rms)

    # ================================================================
    # 4. Summary prints for all deltas
    # ================================================================
    if args.empirical_error:
        mode_label = "empirical"
    elif args.final_step_only:
        mode_label = "final t"
    else:
        mode_label = "time-series"

    print(f"\n{'=' * 60}")
    print(f"  Resource Estimation Summary  ({mode_label} comparison)")
    print(f"{'=' * 60}")
    print(f"  Canonical Trotter (N={N_canon}, "
          f"gates={N_canon * gates_per_step}): obs = {canon_final:.8f}")

    for curve in tepai_curves:
        pod = curve["pi_over_delta"]
        rms = curve["rms_per_shot"]
        S_use = curve["N_samples"]
        tepai_g = curve["mean_gates"]
        print(f"\n  --- pi/delta = {pod}  (S={S_use}) ---")
        print(f"  TE-PAI per-shot RMS ({mode_label}):   sigma = {rms:.6f}")
        print(f"  TE-PAI mean gate count:          {tepai_g:.0f}")
        print()
        print(f"  {'N_Trot':>8s}  {'Trot gates':>10s}  {'eps_Trot':>12s}  "
              f"{'N_s to match':>14s}  {'Trot/TEPAI':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*10}")
        for N, err in zip(trotter_Ns, trotter_errors):
            Ns_cross = (rms / err) ** 2 if err > 0 else np.inf
            trotter_g = N * gates_per_step
            if tepai_g > 0:
                ratio = round(trotter_g / tepai_g)
                ratio_str = f"{ratio}x"
            else:
                ratio_str = "N/A"
            print(f"  {N:8d}  {trotter_g:10d}  {err:12.4e}  "
                  f"{Ns_cross:14.1f}  {ratio_str:>10s}")

    print(f"{'=' * 60}")

    # Save summary
    summary_path = os.path.join(folder, "re_summary.csv")
    _save_re_summary(
        summary_path, N_canon, canon_final,
        trotter_Ns, trotter_errors,
        tepai_curves[0]["rms_per_shot"], tepai_curves[0]["N_samples"],
    )
    print(f"  Summary -> {summary_path}")

    # ================================================================
    # 5. Plot
    # ================================================================
    if not args.no_plot:
        out = os.path.join(
            folder, args.plot_name or "resource_estimation.pdf",
        )
        plot_resource_estimation(
            trotter_Ns, trotter_errors, tepai_curves, out,
        )

    # ================================================================
    # 6. Optional diagnostic time-series plot (all data already cached)
    # ================================================================
    if args.plot_timeseries and first_tepai_run is not None:
        tepai_t, tepai_obs, tepai_rms = first_tepai_run
        plot_timeseries(canon_t, canon_obs, trotter_runs,
                        tepai_t, tepai_obs, tepai_rms)


if __name__ == "__main__":
    main()
