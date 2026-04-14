#!/usr/bin/env python
"""run_bond_dimension.py — Bond-dimension tracking experiments.

Two options:
  Option 1: Trotter and TE-PAI both from t=0, supports multiple deltas
  Option 2: TE-PAI starts halfway through simulation time

Both use linear (non-adaptive) Trotterization and produce 2x2 subplots:
  A: expectation value     B: gate counts
  C: max bond dimension    D: cost metric (N_gates × Σχ_i³)
"""

import argparse
import math
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

from mps_tepai import Hamiltonian, TE_PAI
from run_experiment import (
    _experiment_folder,
    _save_csv,
    _load_csv,
    _check_meta,
    _run_trotter_tracked,
    _run_tepai_tracked,
    DATA_DIR,
)


# ===================================================================
#  File naming
# ===================================================================

def _bd_trotter_file(N):
    return f"trotter_N{N}_adaptiveFalse.csv"


def _bd_tepai_file(pi_over_delta, N, S, tstart):
    return f"bd_tepai_d{pi_over_delta}_N{N}_S{S}_tstart{tstart:.2f}.csv"


# ===================================================================
#  Linear Trotter prefix for hybrid mode
# ===================================================================

def build_linear_prefix_gates(hamil, T_total, N_total, n_snap_total, n_snap_prefix):
    """Build Trotter prefix gates using linear (non-adaptive) stepping."""
    per_snap = N_total // n_snap_total
    ts = np.linspace(0, T_total, N_total)
    terms = [hamil.get_term(t) for t in ts]

    prefix_gates = []
    for k in range(n_snap_prefix):
        snap_gates = []
        for i in range(k * per_snap, (k + 1) * per_snap):
            for pauli, ind, coef in terms[i]:
                snap_gates.append((pauli, 2 * coef * T_total / N_total, ind))
        prefix_gates.append(snap_gates)

    return prefix_gates


# ===================================================================
#  Cache-aware runners
# ===================================================================

def load_or_run_trotter_linear(folder, fname, hamil, nq, T, N, n_snap,
                                chi, op, mq, init):
    """Load or run non-adaptive Trotter with caching."""
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        meta, d = _load_csv(path)
        expected = dict(type="trotter", N=N, adaptive=False, initial_state=init)
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

    print(f"  Running linear Trotter (N={N}) ...")
    t0 = time.time()
    times, obs, bonds, costs, gates = _run_trotter_tracked(
        hamil, nq, T, N, n_snap, chi, op, mq, init, adaptive=False,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    _save_csv(
        path,
        dict(type="trotter", N=N, adaptive=False, initial_state=init),
        ["time", "observable", "max_bond_dim", "cost", "gate_count"],
        [times, obs, bonds, costs, gates],
    )
    print(f"  Saved -> {path}")
    return times, obs, bonds, costs, gates


def load_or_run_tepai_bd(folder, fname, hamil, nq, T, dt, delta, Nt, S,
                          chi, op, mq, init, nw, seed,
                          trotter_obs, trotter_times,
                          tstart=0.0, N_trotter=None, n_snap_total=None):
    """Load or run TE-PAI for bond-dimension experiments.

    Uses linear (non-adaptive) prefix for hybrid mode (tstart > 0).
    Returns (times, obs, rms, bonds, costs, gates, overhead) — same
    signature as load_or_run_tepai in run_experiment.py.
    """
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        meta, d = _load_csv(path)
        expected = dict(type="bd_tepai", delta=delta, N_tepai=Nt,
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

    print(f"  Building TE-PAI (delta={delta:.4e}, N={Nt}, "
          f"snaps={n_snap_tepai}) ...")
    te = TE_PAI(hamil, nq, delta=delta, T=dur, N=Nt, n_snap=n_snap_tepai)

    if S < te.overhead ** 2:
        print(f"  Note: overhead={te.overhead:.1f} -> suggest "
              f"S>={int(np.ceil(te.overhead ** 2))}")

    naive_mean = S <= 10

    if naive_mean:
        print(f"  Generating {S} +1-parity circuits (rejection sampling) ...")
        t0_gen = time.time()
        accepted = []
        batch_seed = seed
        max_attempts = S * 200
        total_generated = 0
        while len(accepted) < S and total_generated < max_attempts:
            batch_size = min((S - len(accepted)) * 5,
                             max_attempts - total_generated)
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
                  f"in {time.time() - t0_gen:.1f}s")
        circuits = accepted
    else:
        print(f"  Generating {S} circuits ...")
        t0_gen = time.time()
        circuits = te.run_te_pai(S, n_workers=nw, seed=seed)
        print(f"  Generated in {time.time() - t0_gen:.1f}s")

    # Build linear prefix for hybrid mode
    prefix = None
    if tstart > 0 and N_trotter is not None and n_snap_total is not None:
        ns_pre = int(round(tstart / dt))
        prefix = build_linear_prefix_gates(
            hamil, T, N_trotter, n_snap_total, ns_pre,
        )

    # Align Trotter observable to TE-PAI timesteps
    n_snap_tepai = int(round(dur / dt))
    tepai_times = tstart + np.arange(n_snap_tepai + 1) * dt
    trotter_obs_aligned = np.interp(tepai_times, trotter_times, trotter_obs)

    print(f"  Executing ({nw} workers) ...")
    t0_exec = time.time()
    times, obs, rms, bonds, costs, gates = _run_tepai_tracked(
        te, circuits, nq, chi, op, mq, init, nw,
        trotter_obs_aligned, tstart, prefix,
        naive_mean=naive_mean,
    )
    print(f"  Done in {time.time() - t0_exec:.1f}s")

    # Per-timestep analytical overhead: exp(2 * l1_norm(t_elapsed) * tan(Δ/2))
    dT = dur / n_snap_tepai
    tan_half_delta = np.tan(delta / 2)
    oh_arr = np.array([
        np.exp(2 * hamil.l1_norm(k * dT) * tan_half_delta)
        for k in range(n_snap_tepai + 1)
    ])

    meta = dict(
        type="bd_tepai", delta=delta, N_tepai=Nt, n_samples=S,
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

_TAB_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]


def plot_bond_dimension(trot_t, trot_obs, trot_bonds, trot_costs, trot_gates,
                        tepai_datasets, out_path, tstart=0.0,
                        n_samples=1):
    """Plot 2x2 subplots with shared bottom legend.

    A — Expectation value      B — Gate count
    C — Max bond dimension     D — Cost metric

    tepai_datasets: list of dicts with keys:
        label, times, obs, se, bonds, costs, gates, overhead
    """
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _ROOT)
        from plot_style import apply_style
        apply_style()
    except ImportError:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axA, axB = axes[0]
    axC, axD = axes[1]

    # -- helper: prepend Trotter prefix for hybrid mode --
    def _prepend(trot_data, tepai_data):
        if tstart > 0:
            mask = trot_t <= tstart + 1e-12
            return (np.concatenate([trot_t[mask][:-1], tepai_data[0]]),
                    np.concatenate([trot_data[mask][:-1], tepai_data[1]]))
        return tepai_data

    # Collect handles/labels from axA for shared legend
    handles, labels = [], []

    # ---- A: Expectation values ----
    h, = axA.plot(trot_t, trot_obs, color="black", lw=2)
    handles.append(h); labels.append("Trotter")

    for i, ds in enumerate(tepai_datasets):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        t, obs, se = ds["times"], ds["obs"], ds["se"]

        if tstart > 0:
            prefix_mask = trot_t <= tstart + 1e-12
            plot_t = np.concatenate([trot_t[prefix_mask][:-1], t])
            plot_obs = np.concatenate([trot_obs[prefix_mask][:-1], obs])
            plot_se = np.concatenate([
                np.zeros(np.sum(prefix_mask) - 1), se])
        else:
            plot_t, plot_obs, plot_se = t, obs, se

        if n_samples <= 10:
            h = axA.scatter(plot_t, plot_obs, color=color, marker="x",
                            s=30, zorder=5)
        else:
            h, = axA.plot(plot_t, plot_obs, color=color)
            axA.fill_between(
                plot_t, plot_obs - plot_se, plot_obs + plot_se,
                color=color, alpha=0.15,
            )
        handles.append(h); labels.append(ds["label"])

    if tstart > 0:
        h = axA.axvline(tstart, color="gray", ls=":", lw=1)
        handles.append(h); labels.append(f"TE-PAI start ($t={tstart}$)")

    axA.set_xlabel("Time")
    axA.set_ylabel(r"$\langle X_0 \rangle$")
    axA.set_title(r"$\bf{A}$  Expectation value", loc="left", fontsize=16)

    # ---- B: Gate counts ----
    axB.plot(trot_t, trot_gates, color="black", lw=2)
    for i, ds in enumerate(tepai_datasets):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        pt, pg = _prepend(trot_gates, (ds["times"], ds["gates"]))
        axB.plot(pt, pg, color=color)
    if tstart > 0:
        axB.axvline(tstart, color="gray", ls=":", lw=1)
    axB.set_xlabel("Time")
    axB.set_ylabel("Cumulative gate count")
    axB.set_title(r"$\bf{B}$  Gate count", loc="left", fontsize=16)

    # ---- C: Max bond dimension ----
    axC.plot(trot_t, trot_bonds, color="black", lw=2)
    for i, ds in enumerate(tepai_datasets):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        pt, pb = _prepend(trot_bonds, (ds["times"], ds["bonds"]))
        axC.plot(pt, pb, color=color)
    if tstart > 0:
        axC.axvline(tstart, color="gray", ls=":", lw=1)
    axC.set_xlabel("Time")
    axC.set_ylabel(r"Max bond dimension $\chi$")
    axC.set_title(r"$\bf{C}$  Bond dimension", loc="left", fontsize=16)

    # ---- D: Cost metric (N_gates × Σχ_i³) ----
    trot_cost_metric = trot_costs * trot_gates
    axD.plot(trot_t, trot_cost_metric, color="black", lw=2)
    for i, ds in enumerate(tepai_datasets):
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        cost_metric = ds["costs"] * ds["gates"]
        pt, pc = _prepend(trot_cost_metric, (ds["times"], cost_metric))
        axD.plot(pt, pc, color=color)
    if tstart > 0:
        axD.axvline(tstart, color="gray", ls=":", lw=1)
    axD.set_xlabel("Time")
    axD.set_ylabel(r"Cost $\;N_g \times \sum \chi_i^3$")
    axD.set_title(r"$\bf{D}$  Cost metric", loc="left", fontsize=16)

    # Absorb scientific-notation offset (e.g. "1e7") into the y-axis label
    # so it doesn't collide with the left-aligned subplot title
    fig.canvas.draw()
    for ax in (axA, axB, axC, axD):
        offset = ax.yaxis.get_offset_text().get_text()
        if offset:
            current = ax.get_ylabel()
            ax.set_ylabel(f"{current}  [{offset}]")
            ax.yaxis.get_offset_text().set_visible(False)

    # ---- Shared legend below all subplots ----
    fig.legend(handles, labels, loc="lower center",
               ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


# ===================================================================
#  CLI + main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Bond-dimension tracking experiments (linear Trotter).",
    )
    # Physics / simulation
    p.add_argument("--n-qubits",       type=int,   default=10)
    p.add_argument("--total-time",     type=float, default=1.0)
    p.add_argument("--dt",             type=float, default=0.1)
    p.add_argument("--j",              type=float, default=0.1)
    p.add_argument("--max-bond",       type=int,   default=16)
    p.add_argument("--operator",       type=str,   default="X")
    p.add_argument("--measure-qubit",  type=int,   default=0)
    p.add_argument("--initial-state",  type=str,   default="plus_flip")
    p.add_argument("--seed",           type=int,   default=0)

    # Algorithm
    p.add_argument("--N-trotter",      type=int,   default=100)
    p.add_argument("--N-tepai",        type=int,   default=100)
    p.add_argument("--N-samples",      type=int,   default=100)
    p.add_argument("--bd-option",      type=int,   default=1, choices=[1, 2])
    p.add_argument("--pi-over-deltas", type=int,   nargs="+", default=[256])

    # Runtime
    p.add_argument("--n-cores",        type=int,   default=None)
    p.add_argument("--no-plot",        action="store_true")
    p.add_argument("--plot-name",      type=str,   default=None)
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()

    nw = args.n_cores or max(1, (mp.cpu_count() or 4) - 2)
    T = args.total_time
    dt = args.dt
    n_snap = int(round(T / dt))
    assert n_snap >= 1, f"total-time must be >= dt={dt}"

    bd_option = args.bd_option
    pi_over_deltas = args.pi_over_deltas

    if bd_option == 2:
        tstart = T / 2
        assert abs(tstart / dt - round(tstart / dt)) < 1e-9, \
            f"T/2={tstart} must be a multiple of dt={dt}"
    else:
        tstart = 0.0

    # Deterministic Hamiltonian from seed
    rng = np.random.default_rng(args.seed)
    freqs = rng.uniform(-1, 1, size=args.n_qubits)
    hamil = Hamiltonian.spin_chain(args.n_qubits, freqs, j=args.j)

    print(f"Bond-dimension experiment (option {bd_option})")
    print(f"  {args.n_qubits}q, chi={args.max_bond}, "
          f"j={args.j}, seed={args.seed}")
    print(f"  T={T}, dt={dt}, {n_snap} snapshots, workers={nw}")
    print(f"  N_trotter={args.N_trotter} (linear), N_tepai={args.N_tepai}")
    delta_strs = ", ".join(f"π/{d}" for d in pi_over_deltas)
    print(f"  Deltas: [{delta_strs}]")
    if bd_option == 2:
        print(f"  TE-PAI start: t={tstart}")

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

    # ---- Non-adaptive Trotter reference ----
    trot_fname = _bd_trotter_file(args.N_trotter)
    print("Linear Trotter reference:")
    trot_t, trot_obs, trot_bonds, trot_costs, trot_gates = \
        load_or_run_trotter_linear(
            folder, trot_fname, hamil, args.n_qubits, T,
            args.N_trotter, n_snap, args.max_bond,
            args.operator, args.measure_qubit, args.initial_state,
        )

    # ---- TE-PAI for each delta ----
    tepai_datasets = []
    for pod in pi_over_deltas:
        delta = math.pi / pod
        fname = _bd_tepai_file(pod, args.N_tepai, args.N_samples, tstart)
        print(f"\nTE-PAI (Δ=π/{pod}):")

        t, obs, rms, bonds, costs, gates, oh_arr = load_or_run_tepai_bd(
            folder, fname, hamil, args.n_qubits, T, dt,
            delta, args.N_tepai, args.N_samples,
            args.max_bond, args.operator, args.measure_qubit,
            args.initial_state, nw, args.seed,
            trotter_obs=trot_obs, trotter_times=trot_t,
            tstart=tstart, N_trotter=args.N_trotter, n_snap_total=n_snap,
        )

        se = rms / np.sqrt(args.N_samples)
        tepai_datasets.append(dict(
            label=rf"TE-PAI ($\Delta=\pi/{pod}$)",
            times=t, obs=obs, se=se, bonds=bonds, costs=costs,
            gates=gates, overhead=oh_arr,
        ))

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Trotter total gates:   {trot_gates[-1]:.0f}")
    for ds in tepai_datasets:
        print(f"  {ds['label']} avg gates: {ds['gates'][-1]:.0f}")
    print(f"{'=' * 60}")

    # ---- Plot ----
    if not args.no_plot:
        default_name = f"bond_dim_opt{bd_option}.pdf"
        out = os.path.join(folder, args.plot_name or default_name)
        plot_bond_dimension(
            trot_t, trot_obs, trot_bonds, trot_costs, trot_gates,
            tepai_datasets, out, tstart=tstart,
            n_samples=args.N_samples,
        )


if __name__ == "__main__":
    main()
