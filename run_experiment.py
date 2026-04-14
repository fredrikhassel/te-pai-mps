#!/usr/bin/env python
"""run_experiment.py — TE-PAI vs Trotter experiment with persistent data caching.

Replicates the advantage.py experiment from the mps-te-pai package but saves
Trotter and TE-PAI results as CSV files under data/<experiment>/. Subsequent
runs with matching parameters load from cache instead of re-computing.

Works on macOS and Linux with full multiprocessing support.

Usage (toy defaults — quick test):
    python run_experiment.py

Production example:
    python run_experiment.py --total-time 1.0 --N-trotter 2000 --N-tepai 2000 \
                             --N-samples 200 --delta 0.05 --n-cores 8
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
    build_trotter_prefix_gates,
    create_mps_circuit,
    apply_gates,
    measure,
)

DATA_DIR = os.path.join(_ROOT, "data")


# ===================================================================
#  Naming conventions
# ===================================================================

def _experiment_folder(nq, chi, j, seed, T, dt, op, mq, init):
    """Folder name encoding all shared experiment parameters."""
    return (
        f"nq{nq}_chi{chi}_j{j:.2f}_seed{seed}"
        f"_T{T:.2f}_dt{dt:.2f}_{op}{mq}_{init}"
    )


def _trotter_file(N, adaptive):
    return f"trotter_N{N}_adaptive{adaptive}.csv"


def _tepai_file(pi_over_delta, N, S, tstart):
    return f"tepai_d{pi_over_delta}_N{N}_S{S}_tstart{tstart:.2f}.csv"


# ===================================================================
#  CSV I/O
# ===================================================================

def _save_csv(path, meta, cols, arrays):
    """Write a CSV with ``# key=value`` header lines."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write(",".join(cols) + "\n")
        for row in zip(*arrays):
            f.write(",".join(f"{v}" for v in row) + "\n")


def _load_csv(path):
    """Read a CSV written by ``_save_csv``.

    Returns ``(meta_dict, {col_name: np.array})``.
    """
    meta = {}
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].strip().split("=", 1)
                    meta[k.strip()] = v.strip()
            elif header is None:
                header = line.split(",")
            else:
                rows.append([float(x) for x in line.split(",")])
    data = {}
    arr = np.array(rows)
    for i, col in enumerate(header):
        data[col] = arr[:, i]
    return meta, data


# ===================================================================
#  Helper: max bond dimension of an MPS circuit
# ===================================================================

def _max_bond_dim(circ, nq):
    """Return the current maximum bond dimension of the MPS."""
    if nq <= 1:
        return 1
    sizes = circ.psi.bond_sizes()
    return max(sizes) if sizes else 1

def _cost(psi):
    """Estimate TEBD cost as sum of chi^3 over all nearest-neighbor bonds."""
    bond_dims = np.asarray(psi.bond_sizes(), dtype=float)
    return float(np.sum(bond_dims ** 3))


# ===================================================================
#  Trotter with bond-dimension + gate-count tracking
# ===================================================================

def _run_trotter_tracked(hamil, nq, T, N, n_snap, chi, op, mq, init, adaptive):
    """Run a single Trotter circuit, recording observable, bond dim, cost,
    and cumulative gate count at every snapshot."""
    circ = create_mps_circuit(nq, init, chi)

    vals = [measure(circ, op, mq)]
    bonds = [_max_bond_dim(circ, nq)]
    costs = [_cost(circ.psi)]
    gcounts = [0]
    cum_gates = 0

    if adaptive:
        dT = T / n_snap
        for k in range(1, n_snap + 1):
            N_cum = N * k * k
            N_prev = N * (k - 1) * (k - 1)
            N_int = N_cum - N_prev

            t0 = (k - 1) * dT
            t1 = k * dT
            dt_int = t1 - t0

            steps = np.linspace(t0, t1, N_int, endpoint=False)
            terms = [hamil.get_term(t) for t in steps]

            for i in range(N_int):
                gates = [
                    (p, 2 * c * dt_int / N_int, idx)
                    for p, idx, c in terms[i]
                ]
                apply_gates(circ, gates)
                cum_gates += len(gates)

            vals.append(measure(circ, op, mq))
            bonds.append(_max_bond_dim(circ, nq))
            costs.append(_cost(circ.psi))
            gcounts.append(cum_gates)
    else:
        ts = np.linspace(0, T, N)
        terms = [hamil.get_term(t) for t in ts]
        per_snap = N // n_snap

        for i in range(N):
            gates = [(p, 2 * c * T / N, idx) for p, idx, c in terms[i]]
            apply_gates(circ, gates)
            cum_gates += len(gates)

            if (i + 1) % per_snap == 0:
                vals.append(measure(circ, op, mq))
                bonds.append(_max_bond_dim(circ, nq))
                costs.append(_cost(circ.psi))
                gcounts.append(cum_gates)

    times = np.linspace(0, T, n_snap + 1)
    obs = 2 * np.array(vals) - 1          # map [0,1] → [-1,1]
    return (times, obs, np.array(bonds, dtype=float),
            np.array(costs, dtype=float), np.array(gcounts, dtype=float))


# ===================================================================
#  TE-PAI per-circuit workers  (module-level for pickling with spawn)
# ===================================================================

def _tepai_worker(args):
    """Execute one TE-PAI circuit → (obs, bonds, costs, gates)."""
    nq, signs, gates_arr, gam_list, chi, init, op, mq = args
    circ = create_mps_circuit(nq, init, chi)

    obs = [measure(circ, op, mq)]
    bonds = [_max_bond_dim(circ, nq)]
    costs = [_cost(circ.psi)]
    gcounts = [0]
    cum = 0

    for si, snap_gates in enumerate(gates_arr):
        apply_gates(circ, snap_gates)
        cum += len(snap_gates)
        val = measure(circ, op, mq)
        w = signs[si] * gam_list[si + 1]
        obs.append(val * w)
        bonds.append(_max_bond_dim(circ, nq))
        costs.append(_cost(circ.psi))
        gcounts.append(cum)

    return obs, bonds, costs, gcounts


def _tepai_naive_worker(args):
    """Execute one TE-PAI circuit without gamma/sign weighting (naive mean)."""
    nq, _signs, gates_arr, _gam_list, chi, init, op, mq = args
    circ = create_mps_circuit(nq, init, chi)

    obs = [measure(circ, op, mq)]
    bonds = [_max_bond_dim(circ, nq)]
    costs = [_cost(circ.psi)]
    gcounts = [0]
    cum = 0

    for _si, snap_gates in enumerate(gates_arr):
        apply_gates(circ, snap_gates)
        cum += len(snap_gates)
        obs.append(measure(circ, op, mq))
        bonds.append(_max_bond_dim(circ, nq))
        costs.append(_cost(circ.psi))
        gcounts.append(cum)

    return obs, bonds, costs, gcounts


def _tepai_hybrid_worker(args):
    """Execute one hybrid Trotter-prefix + TE-PAI circuit."""
    nq, signs, gates_arr, gam_list, chi, init, op, mq, prefix = args
    circ = create_mps_circuit(nq, init, chi)

    cum = 0
    for pg in prefix:
        apply_gates(circ, pg)
        cum += len(pg)

    obs = [measure(circ, op, mq)]
    bonds = [_max_bond_dim(circ, nq)]
    costs = [_cost(circ.psi)]
    gcounts = [cum]

    for si, snap_gates in enumerate(gates_arr):
        apply_gates(circ, snap_gates)
        cum += len(snap_gates)
        val = measure(circ, op, mq)
        w = signs[si] * gam_list[si + 1]
        obs.append(val * w)
        bonds.append(_max_bond_dim(circ, nq))
        costs.append(_cost(circ.psi))
        gcounts.append(cum)

    return obs, bonds, costs, gcounts


def _tepai_naive_hybrid_worker(args):
    """Execute one hybrid circuit without gamma/sign weighting (naive mean)."""
    nq, _signs, gates_arr, _gam_list, chi, init, op, mq, prefix = args
    circ = create_mps_circuit(nq, init, chi)

    cum = 0
    for pg in prefix:
        apply_gates(circ, pg)
        cum += len(pg)

    obs = [measure(circ, op, mq)]
    bonds = [_max_bond_dim(circ, nq)]
    costs = [_cost(circ.psi)]
    gcounts = [cum]

    for _si, snap_gates in enumerate(gates_arr):
        apply_gates(circ, snap_gates)
        cum += len(snap_gates)
        obs.append(measure(circ, op, mq))
        bonds.append(_max_bond_dim(circ, nq))
        costs.append(_cost(circ.psi))
        gcounts.append(cum)

    return obs, bonds, costs, gcounts


def _run_tepai_tracked(te_pai, circuits, nq, chi, op, mq, init, nw,
                       trotter_obs_aligned, tstart=0.0, prefix=None,
                       naive_mean=False):
    """Execute all TE-PAI circuits and aggregate results.

    When *naive_mean* is True (≤10 samples with all-positive parity),
    observables are aggregated as a simple mean without gamma/sign weighting.

    Otherwise, observable values are aggregated with gamma/sign weighting
    (mean across circuits).  Bond dimensions and gate counts are always
    simple means.

    Error is computed as the per-shot RMS deviation from the Trotter reference
    at each timestep:  rms[t] = sqrt( mean_i( (sample_i[t] - trotter[t])^2 ) )
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gam = te_pai.gam_list
    n = len(circuits)

    if tstart > 0 and prefix is not None:
        fn = _tepai_naive_hybrid_worker if naive_mean else _tepai_hybrid_worker
        inputs = [
            (nq, s, g, gam, chi, init, op, mq, prefix)
            for s, g in circuits
        ]
    else:
        fn = _tepai_naive_worker if naive_mean else _tepai_worker
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

    all_obs = np.array([r[0] for r in results])
    all_bonds = np.array([r[1] for r in results], dtype=float)
    all_costs = np.array([r[2] for r in results], dtype=float)
    all_gates = np.array([r[3] for r in results], dtype=float)

    # Per-sample physical observable: map [0,1] → [-1,1]
    all_obs_phys = 2 * all_obs - 1          # shape (n, n_timesteps)
    mean_obs_phys = np.mean(all_obs_phys, axis=0)

    # Per-shot RMS error against the Trotter reference
    deviations = all_obs_phys - trotter_obs_aligned[np.newaxis, :]
    rms_per_shot = np.sqrt(np.mean(deviations ** 2, axis=0))

    mean_bonds = np.mean(all_bonds, axis=0)
    mean_costs = np.mean(all_costs, axis=0)
    mean_gates = np.mean(all_gates, axis=0)

    dT = te_pai.T / te_pai.n_snap
    pts = te_pai.n_snap + 1
    times = tstart + np.arange(pts) * dT

    return times, mean_obs_phys, rms_per_shot, mean_bonds, mean_costs, mean_gates


# ===================================================================
#  Cache-aware runners
# ===================================================================

def _check_meta(meta, expected):
    """Return list of (key, cached_val, expected_val) for any mismatches.

    Values are compared as strings (since CSV metadata is stored as text).
    Floating-point values are rounded to 10 significant figures before
    comparison to avoid false positives from repr noise.
    """
    mismatches = []
    for key, want in expected.items():
        got = meta.get(key)
        if got is None:
            mismatches.append((key, "<missing>", want))
            continue
        # Normalise both sides: try float comparison first, fall back to str
        try:
            g, w = float(got), float(want)
            if round(g, 10) != round(w, 10):
                mismatches.append((key, got, want))
        except (ValueError, TypeError):
            if str(got) != str(want):
                mismatches.append((key, got, want))
    return mismatches


def load_or_run_trotter(folder, fname, hamil, nq, T, N, n_snap,
                        chi, op, mq, init, adaptive):
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        meta, d = _load_csv(path)
        # Validate cached metadata matches requested parameters
        expected = dict(type="trotter", N=N, adaptive=adaptive,
                        initial_state=init)
        bad = _check_meta(meta, expected)
        if bad:
            print(f"  [cache STALE] {fname}  — mismatch: "
                  + ", ".join(f"{k}: {g!s}≠{w!s}" for k, g, w in bad))
            print(f"  Re-running ...")
        else:
            print(f"  [cache] {fname}")
            cost = d["cost"] if "cost" in d else np.ones(len(d["time"]))
            return (d["time"], d["observable"],
                    d["max_bond_dim"], cost, d["gate_count"])

    print(f"  Running Trotter (N={N}, adaptive={adaptive}) ...")
    t0 = time.time()
    times, obs, bonds, costs, gates = _run_trotter_tracked(
        hamil, nq, T, N, n_snap, chi, op, mq, init, adaptive,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    _save_csv(
        path,
        dict(type="trotter", N=N, adaptive=adaptive, initial_state=init),
        ["time", "observable", "max_bond_dim", "cost", "gate_count"],
        [times, obs, bonds, costs, gates],
    )
    print(f"  Saved -> {path}")
    return times, obs, bonds, costs, gates


def load_or_run_tepai(folder, fname, hamil, nq, T, dt, delta, Nt, S,
                      chi, op, mq, init, nw, seed,
                      trotter_obs, trotter_times,
                      tstart=0.0, N_prefix=None, n_snap_total=None):
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        meta, d = _load_csv(path)
        # Validate cached metadata matches requested parameters
        expected = dict(type="tepai", delta=delta, N_tepai=Nt,
                        n_samples=S, tepai_start_time=tstart,
                        initial_state=init, seed=seed)
        bad = _check_meta(meta, expected)
        if bad:
            print(f"  [cache STALE] {fname}  — mismatch: "
                  + ", ".join(f"{k}: {g!s}≠{w!s}" for k, g, w in bad))
            print(f"  Re-running ...")
        else:
            print(f"  [cache] {fname}")
            if "overhead" in d:
                oh_arr = d["overhead"]
            else:
                oh_arr = np.ones(len(d["time"]))
            cost = d["cost"] if "cost" in d else np.ones(len(d["time"]))
            return (d["time"], d["observable"], d["rms_error"],
                    d["max_bond_dim"], cost, d["gate_count"], oh_arr)

    dur = T - tstart
    n_snap_tepai = int(round(dur / dt))

    print(f"  Building TE-PAI (delta={delta:.4e}, N={Nt}, snaps={n_snap_tepai}) ...")
    te = TE_PAI(hamil, nq, delta=delta, T=dur, N=Nt, n_snap=n_snap_tepai)

    if S < te.overhead ** 2:
        print(f"  Note: overhead={te.overhead:.1f} -> suggest "
              f"S>={int(np.ceil(te.overhead ** 2))}")

    naive_mean = S <= 10

    if naive_mean:
        # Rejection-sample: keep only circuits with +1 parity throughout.
        # Generate in batches until we have S good circuits.
        print(f"  Generating {S} +1-parity circuits (rejection sampling) ...")
        t0 = time.time()
        accepted = []
        batch_seed = seed
        max_attempts = S * 200          # safety cap
        total_generated = 0
        while len(accepted) < S and total_generated < max_attempts:
            batch_size = min((S - len(accepted)) * 5, max_attempts - total_generated)
            batch = te.run_te_pai(batch_size, n_workers=nw, seed=batch_seed)
            total_generated += batch_size
            for signs, gates_arr in batch:
                if all(s == 1 for s in signs):
                    accepted.append((signs, gates_arr))
                    if len(accepted) >= S:
                        break
            batch_seed = batch_seed + 1 if batch_seed is not None else None
        if len(accepted) < S:
            print(f"  Warning: only found {len(accepted)}/{S} +1-parity "
                  f"circuits after {total_generated} attempts")
        else:
            print(f"  Accepted {S}/{total_generated} circuits "
                  f"in {time.time() - t0:.1f}s")
        circuits = accepted
    else:
        print(f"  Generating {S} circuits ...")
        t0 = time.time()
        circuits = te.run_te_pai(S, n_workers=nw, seed=seed)
        print(f"  Generated in {time.time() - t0:.1f}s")

    prefix = None
    if tstart > 0 and N_prefix is not None and n_snap_total is not None:
        ns_pre = int(round(tstart / dt))
        prefix = build_trotter_prefix_gates(
            hamil, T, N_prefix, n_snap_total, ns_pre,
        )

    # Align Trotter observable to TE-PAI timesteps
    dur = T - tstart
    n_snap_tepai = int(round(dur / dt))
    tepai_times = tstart + np.arange(n_snap_tepai + 1) * dt
    trotter_obs_aligned = np.interp(tepai_times, trotter_times, trotter_obs)

    print(f"  Executing ({nw} workers) ...")
    t0 = time.time()
    times, obs, rms, bonds, costs, gates = _run_tepai_tracked(
        te, circuits, nq, chi, op, mq, init, nw,
        trotter_obs_aligned, tstart, prefix,
        naive_mean=naive_mean,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Per-timestep analytical overhead: exp(2 * l1_norm(t_elapsed) * tan(Δ/2))
    dT = dur / n_snap_tepai
    tan_half_delta = np.tan(delta / 2)
    oh_arr = np.array([
        np.exp(2 * hamil.l1_norm(k * dT) * tan_half_delta)
        for k in range(n_snap_tepai + 1)
    ])

    meta = dict(
        type="tepai", delta=delta, N_tepai=Nt, n_samples=S,
        tepai_start_time=tstart, initial_state=init, seed=seed,
        overhead_final=te.overhead, gamma_final=te.gamma_final,
        expected_num_gates=te.expected_num_gates,
    )
    _save_csv(
        path, meta,
        ["time", "observable", "rms_error", "max_bond_dim", "cost",
         "gate_count", "overhead"],
        [times, obs, rms, bonds, costs, gates, oh_arr],
    )
    print(f"  Saved -> {path}")
    return times, obs, rms, bonds, costs, gates, oh_arr


# ===================================================================
#  Plotting
# ===================================================================

def plot_results(trot_t, trot_obs, trot_gates,
                 tepai_t, tepai_obs, tepai_se, tepai_gates,
                 tstart, out_path,
                 trot_bonds=None, tepai_bonds=None, chi=None,
                 n_samples=1):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    tepai_final = tepai_gates[-1]
    total_trotter = trot_gates[-1]
    has_advantage = tepai_final < total_trotter

    t_cross = None
    if has_advantage:
        for k in range(1, len(trot_gates)):
            if trot_gates[k] >= tepai_final:
                frac = (
                    (tepai_final - trot_gates[k - 1])
                    / (trot_gates[k] - trot_gates[k - 1])
                )
                t_cross = (
                    trot_t[k - 1]
                    + frac * (trot_t[k] - trot_t[k - 1])
                )
                break

    # Decide layout: chi > 16 with bond data → 4-row figure
    high_chi = (chi is not None and chi > 16
                and trot_bonds is not None and tepai_bonds is not None)

    if high_chi:
        fig, (ax1, axA, axB, axC) = plt.subplots(
            4, 1, figsize=(8, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 2, 2]},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # -- expectation values --
    # Trotter is always a solid black line
    ax1.plot(trot_t, trot_obs, color="black", label="Trotter")

    if has_advantage and t_cross is not None:
        ax1.axvline(t_cross, color="tab:green", ls="--", lw=1)
        ax1.axvspan(
            t_cross, trot_t[-1], color="tab:green", alpha=0.10,
            label="TE-PAI advantage",
        )

    lbl = "Trotter + TE-PAI" if tstart > 0 else "TE-PAI"
    if n_samples <= 10:
        # Empirical error: deviation from Trotter reference
        trotter_interp = np.interp(tepai_t, trot_t, trot_obs)
        empirical_err = np.abs(tepai_obs - trotter_interp)
        ax1.errorbar(tepai_t, tepai_obs, yerr=empirical_err,
                     fmt="x", color="tab:green", markersize=5,
                     capsize=3, capthick=1, elinewidth=1,
                     zorder=5, label=lbl)
    else:
        ax1.plot(tepai_t, tepai_obs, color="tab:green", label=lbl)
        ax1.fill_between(
            tepai_t, tepai_obs - tepai_se, tepai_obs + tepai_se,
            color="tab:green", alpha=0.2,
        )
    if tstart > 0:
        ax1.axvline(
            tstart, color="tab:green", ls=":", lw=1,
            label=f"TE-PAI start ($t={tstart}$)",
        )
    ax1.set_ylabel(r"$\langle X_0 \rangle$")
    ax1.legend(loc="upper right")

    # -- Error inset for t=0 TE-PAI with few samples --
    if n_samples <= 10 and tstart == 0:
        import matplotlib.ticker as mticker
        axins_err = ax1.inset_axes([0.08, 0.15, 0.38, 0.35])
        trotter_interp = np.interp(tepai_t, trot_t, trot_obs)
        err = np.abs(tepai_obs - trotter_interp)
        axins_err.plot(tepai_t, err, color="tab:green", lw=1.2)
        axins_err.set_xlabel("$t$", fontsize=10, labelpad=2)
        axins_err.set_ylabel("Error", fontsize=10, labelpad=2)
        axins_err.tick_params(labelsize=8, pad=2)
        axins_err.set_xlim(tepai_t[0], tepai_t[-1])

        # Scale y-ticks to single-digit with ×10^exp annotation
        err_max = np.max(err)
        if err_max > 0:
            exp = int(np.floor(np.log10(err_max)))
            sc = 10 ** exp
            axins_err.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _, s=sc: f"{v / s:.0f}")
            )
            # Force ticks at integer multiples of the scale
            axins_err.yaxis.set_major_locator(
                mticker.MultipleLocator(sc)
            )
            axins_err.text(
                0.02, 0.95, rf"$\times 10^{{{exp}}}$",
                transform=axins_err.transAxes, fontsize=8,
                va="top", ha="left",
            )

    # -- Zoomed inset on last 6 datapoints (single nonzero tstart only) --
    if tstart > 0:
        from matplotlib.patches import Rectangle

        n_zoom = min(6, len(trot_t))
        t_zoom_start = trot_t[-n_zoom]
        t_zoom_end = trot_t[-1]
        dt_plot = trot_t[1] - trot_t[0] if len(trot_t) > 1 else 0.1

        # Collect y-values in the zoom window for axis limits
        zoom_ys = list(trot_obs[-n_zoom:])
        tepai_mask = (tepai_t >= t_zoom_start - 1e-9) & (
            tepai_t <= t_zoom_end + 1e-9
        )
        if np.any(tepai_mask):
            zoom_ys.extend(tepai_obs[tepai_mask])

        y_min, y_max = min(zoom_ys), max(zoom_ys)
        y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.1

        axins = ax1.inset_axes([0.02, 0.02, 0.38, 0.38])
        axins.plot(trot_t[-n_zoom:], trot_obs[-n_zoom:], "k-", lw=2.5)

        if n_samples <= 10:
            if np.any(tepai_mask):
                axins.scatter(
                    tepai_t[tepai_mask], tepai_obs[tepai_mask],
                    marker="x", color="tab:green", s=80,
                    linewidths=2, zorder=5,
                )
        else:
            if np.any(tepai_mask):
                axins.plot(
                    tepai_t[tepai_mask], tepai_obs[tepai_mask],
                    color="tab:green",
                )

        x0 = t_zoom_start - 0.3 * dt_plot
        x1 = t_zoom_end + 0.3 * dt_plot
        y0 = y_min - y_pad
        y1 = y_max + y_pad
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_xticks([])
        axins.set_yticks([])

        # Dashed rectangle on main plot around zoom region
        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor="black", facecolor="none",
            linestyle="--",
        )
        ax1.add_patch(rect)

    if high_chi:
        # -- Build aligned Trotter prefix for TE-PAI subplots when tstart > 0 --
        # For tstart > 0 the TE-PAI arrays start at tstart, so we prepend
        # the Trotter values up to tstart so the curves connect visually.
        if tstart > 0:
            # Trotter indices up to and including tstart
            prefix_mask = trot_t <= tstart + 1e-12
            trot_prefix_t = trot_t[prefix_mask]
            trot_prefix_bonds = trot_bonds[prefix_mask]
            trot_prefix_gates = trot_gates[prefix_mask]

            # TE-PAI curves: prepend Trotter prefix so the line is continuous
            tepai_plot_t = np.concatenate([trot_prefix_t[:-1], tepai_t])
            tepai_plot_bonds = np.concatenate(
                [trot_prefix_bonds[:-1], tepai_bonds])
            tepai_plot_gates = np.concatenate(
                [trot_prefix_gates[:-1], tepai_gates])
        else:
            tepai_plot_t = tepai_t
            tepai_plot_bonds = tepai_bonds
            tepai_plot_gates = tepai_gates

        # (A) Bond dimension over time
        axA.plot(trot_t, trot_bonds, color="black", label="Trotter")
        axA.plot(tepai_plot_t, tepai_plot_bonds, color="tab:green", label=lbl)
        if tstart > 0:
            axA.axvline(tstart, color="tab:green", ls=":", lw=1)
        axA.set_ylabel(r"Bond dimension $\chi$")
        axA.legend()

        # (B) Gate count over time
        axB.plot(trot_t, trot_gates, color="black", label="Trotter")
        axB.plot(tepai_plot_t, tepai_plot_gates, color="tab:green", label=lbl)
        if tstart > 0:
            axB.axvline(tstart, color="tab:green", ls=":", lw=1)
        axB.set_ylabel("Cumulative gate count")
        axB.legend()

        # (C) Circuit cost: gate_count × bond_dim^3
        trot_cost = trot_gates * trot_bonds ** 3
        tepai_cost = tepai_plot_gates * tepai_plot_bonds ** 3
        axC.plot(trot_t, trot_cost, color="black", label="Trotter")
        axC.plot(tepai_plot_t, tepai_cost, color="tab:green", label=lbl)
        if tstart > 0:
            axC.axvline(tstart, color="tab:green", ls=":", lw=1)
        axC.set_xlabel("Time")
        axC.set_ylabel(r"$C_{\mathrm{circ}}$ (gates $\times\;\chi^3$)")
        axC.legend()

    else:
        # -- gate counts (original behaviour for chi <= 16) --
        mx = max(total_trotter, tepai_final)
        exp = int(np.floor(np.log10(mx))) if mx > 0 else 0
        sc = 10 ** exp if exp > 0 else 1

        ax2.plot(trot_t, trot_gates / sc, color="black")
        ax2.plot(tepai_t, tepai_gates / sc, color="tab:green")

        if has_advantage and t_cross is not None:
            ax2.axhline(tepai_final / sc, color="tab:green", ls="--", lw=1)
            above = trot_gates >= tepai_final
            if np.any(above):
                idx0 = np.argmax(above)
                tr = np.insert(trot_t[idx0:], 0, t_cross)
                gr = np.insert(trot_gates[idx0:], 0, tepai_final)
                ax2.fill_between(
                    tr, tepai_final / sc, gr / sc,
                    color="gray", alpha=0.3,
                    label="Additional Trotter gates",
                )

        if tstart > 0:
            ax2.axvline(tstart, color="tab:green", ls=":", lw=1)

        ax2.set_xlabel("Time")
        ax2.set_xlim(0, trot_t[-1])
        if exp > 0:
            ax2.set_ylabel(
                rf"Cumulative gate count $(\times 10^{{{exp}}})$"
            )
        else:
            ax2.set_ylabel("Cumulative gate count")
        ax2.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.2g}")
        )
        ax2.legend()

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  Multi-start-time plotting
# ===================================================================

_TAB_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:olive", "tab:cyan",
]


def plot_multi_start(trot_t, trot_obs, trot_gates, tepai_runs, out_path):
    """Plot multi-start TE-PAI experiment.

    tepai_runs: list of dicts with keys:
        tstart, times, obs, overhead, gates
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Rectangle

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ---- Top: expectation values ----
    ax1.plot(trot_t, trot_obs, color="black", label="Trotter", lw=2.5)

    for i, run in enumerate(tepai_runs):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        ts = run["tstart"]
        t = run["times"]
        obs = run["obs"]
        err = run["overhead"] - 1       # per-timestep error bar (starts at 0)

        lbl = f"$t_0={ts:.1f}$"
        ax1.errorbar(
            t, obs, yerr=err,
            fmt="none",
            color=color,
            capsize=5, capthick=1, elinewidth=3,
            label=lbl, zorder=3,
        )

    ax1.set_ylabel(r"$\langle X_0 \rangle$")
    ax1.grid(True)

    # ---- Inset: last 5 datapoints ----
    n_zoom = min(5, len(trot_t))
    t_zoom_start = trot_t[-n_zoom]
    t_zoom_end = trot_t[-1]
    dt_plot = trot_t[1] - trot_t[0] if len(trot_t) > 1 else 0.1

    zoom_ys = list(trot_obs[-n_zoom:])
    for run in tepai_runs:
        mask = (run["times"] >= t_zoom_start - 1e-9) & (
            run["times"] <= t_zoom_end + 1e-9
        )
        if np.any(mask):
            zoom_ys.extend(run["obs"][mask])

    y_min, y_max = min(zoom_ys), max(zoom_ys)
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.1

    axins = ax1.inset_axes([0.02, 0.02, 0.38, 0.38])
    axins.plot(trot_t[-n_zoom:], trot_obs[-n_zoom:], "k-", lw=2.5)

    for i, run in enumerate(tepai_runs):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        mask = (run["times"] >= t_zoom_start - 1e-9) & (
            run["times"] <= t_zoom_end + 1e-9
        )
        if np.any(mask):
            axins.scatter(
                run["times"][mask], run["obs"][mask],
                marker="x", color=color, s=80, linewidths=2, zorder=5,
            )

    ax1.set_xlim(0,3)
    x0 = t_zoom_start - 0.3 * dt_plot
    x1 = t_zoom_end + 0.3 * dt_plot
    y0 = y_min - y_pad
    y1 = y_max + y_pad
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_xticks([])
    axins.set_yticks([])

    # Dashed rectangle on main plot around zoom region
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--",
    )
    ax1.add_patch(rect)

    # ---- Bottom: gate counts ----
    all_final_gates = [trot_gates[-1]] + [r["gates"][-1] for r in tepai_runs]
    mx = max(all_final_gates)
    exp = int(np.floor(np.log10(mx))) if mx > 0 else 0
    sc = 10 ** exp if exp > 0 else 1

    ax2.plot(trot_t, trot_gates / sc, color="black", label="Trotter", lw=2.5)
    for i, run in enumerate(tepai_runs):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        ts = run["tstart"]
        lbl = f"$t_0={ts:.1f}$"
        ax2.plot(run["times"], run["gates"] / sc, color=color, label=lbl,
                 lw=2.5)

    ax2.set_xlabel("Time")
    if exp > 0:
        ax2.set_ylabel(
            rf"Cumulative gate count $(\times 10^{{{exp}}})$"
        )
    else:
        ax2.set_ylabel("Cumulative gate count")
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2g}")
    )
    ax2.grid(True)

    # Shared legend between subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.tight_layout(h_pad=2.5)
    # Place legend centered between the two subplots
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    legend_y = (pos1.y0 + pos2.y1) / 2
    fig.legend(handles=handles, labels=labels, loc="center",
               ncol=4, frameon=False,
               bbox_to_anchor=(0.5, legend_y))
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  CLI + main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="TE-PAI vs Trotter experiment with data caching.",
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
    p.add_argument("--N-samples",   type=int,   default=1)
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

    start_times = args.tepai_start_time          # list of floats
    multi_start = len(start_times) > 1

    for ts in start_times:
        if ts > 0:
            assert abs(ts / dt - round(ts / dt)) < 1e-9, \
                f"tepai-start-time ({ts}) must be a multiple of dt={dt}"
            assert ts < T, f"tepai-start-time {ts} must be < total-time {T}"

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    print(f"Experiment: {args.n_qubits}q, chi={args.max_bond}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, workers={nw}")
    if multi_start:
        print(f"  Multi-start TE-PAI: start times = {start_times}")

    # Data directory
    folder = os.path.join(
        DATA_DIR,
        _experiment_folder(
            args.n_qubits, args.max_bond, args.j, args.seed,
            T, dt, args.operator, args.measure_qubit, args.initial_state,
        ),
    )
    os.makedirs(folder, exist_ok=True)
    print(f"  Data dir: {folder}\n")

    # ---- Trotter reference (always first) ----
    trot_fname = _trotter_file(args.N_trotter, True)
    print("Trotter reference:")
    trot_t, trot_obs, trot_bonds, _trot_costs, trot_gates = load_or_run_trotter(
        folder, trot_fname, hamil, args.n_qubits, T,
        args.N_trotter, n_snap, args.max_bond,
        args.operator, args.measure_qubit, args.initial_state, True,
    )

    pi_over_delta = round(np.pi / args.delta)

    if multi_start:
        # ----- Multi-start mode: S=1, overhead error bars -----
        tepai_runs = []
        for ts in start_times:
            tepai_fname = _tepai_file(pi_over_delta, args.N_tepai, 1, ts)
            print(f"\nTE-PAI (tstart={ts}):")
            t, obs, _rms, _bonds, _costs, gates, oh_arr = load_or_run_tepai(
                folder, tepai_fname, hamil, args.n_qubits, T, dt,
                args.delta, args.N_tepai, 1,                  # force S=1
                args.max_bond, args.operator, args.measure_qubit,
                args.initial_state, nw, args.seed,
                trotter_obs=trot_obs, trotter_times=trot_t,
                tstart=ts, N_prefix=args.N_trotter, n_snap_total=n_snap,
            )
            tepai_runs.append(dict(
                tstart=ts, times=t, obs=obs, overhead=oh_arr, gates=gates,
            ))
            print(f"    Overhead Γ(final) = {oh_arr[-1]:.4f}")

        # ---- Plot ----
        if not args.no_plot:
            out = os.path.join(folder, args.plot_name or "multi_start.pdf")
            plot_multi_start(trot_t, trot_obs, trot_gates, tepai_runs, out)

    else:
        # ----- Single-start mode (original advantage experiment) -----
        tstart = start_times[0]
        tepai_fname = _tepai_file(
            pi_over_delta, args.N_tepai, args.N_samples, tstart,
        )
        print("\nTE-PAI:")
        tepai_t, tepai_obs, tepai_rms, tepai_bonds, _tepai_costs, tepai_gates, _ = \
            load_or_run_tepai(
                folder, tepai_fname, hamil, args.n_qubits, T, dt,
                args.delta, args.N_tepai, args.N_samples,
                args.max_bond, args.operator, args.measure_qubit,
                args.initial_state, nw, args.seed,
                trotter_obs=trot_obs, trotter_times=trot_t,
                tstart=tstart, N_prefix=args.N_trotter, n_snap_total=n_snap,
            )

        # Standard error of the mean from per-shot RMS
        tepai_se = tepai_rms / np.sqrt(args.N_samples)

        # ---- Summary ----
        print(f"\n{'=' * 60}")
        print(f"  Trotter total gates:   {trot_gates[-1]:.0f}")
        print(f"  TE-PAI avg gates:      {tepai_gates[-1]:.0f}")
        ratio = (
            trot_gates[-1] / tepai_gates[-1]
            if tepai_gates[-1] > 0 else float("inf")
        )
        print(f"  Gate ratio:            {ratio:.1f}x")
        print(f"{'=' * 60}")

        # ---- Plot ----
        if not args.no_plot:
            out = os.path.join(folder, args.plot_name or "advantage.pdf")
            plot_results(
                trot_t, trot_obs, trot_gates,
                tepai_t, tepai_obs, tepai_se, tepai_gates,
                tstart, out,
                trot_bonds=trot_bonds, tepai_bonds=tepai_bonds,
                chi=args.max_bond, n_samples=args.N_samples,
            )


if __name__ == "__main__":
    main()
