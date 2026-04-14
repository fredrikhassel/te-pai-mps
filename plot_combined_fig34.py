#!/usr/bin/env python
"""plot_combined_fig34.py — 2x2 combined figure for figure_3 / figure_4.

Loads the cached Trotter and TE-PAI CSVs for the two experiments:

    figure_3_q100_N20_d4096   — 100 qubit hybrid TE-PAI (tstart = 3, T = 4)
    figure_4_q100_N20_d4096   — 100 qubit TE-PAI from start (tstart = 0, T = 5)

Produces a 2x2 figure:

    ┌─────────────────┬─────────────────┐
    │ A: figure_3 <X> │ B: figure_4 <X> │   (shared y axis + label)
    ├─────────────────┼─────────────────┤
    │ C: figure_3 gt  │ D: figure_4 gt  │
    └─────────────────┴─────────────────┘

No recomputation is performed — this script only reads the cached CSV
data written by run_experiment.py.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from plot_style import apply_style  # noqa: E402


def _load_csv(path):
    """Read a CSV written by run_experiment._save_csv."""
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
    arr = np.array(rows)
    data = {col: arr[:, i] for i, col in enumerate(header)}
    return meta, data


DATA_DIR = os.path.join(_ROOT, "data")

# (folder, trotter csv, tepai csv, tstart, title)
FIG3 = (
    "nq100_chi16_j0.10_seed0_T4.00_dt0.10_X0_plus_flip",
    "trotter_N20_adaptiveTrue.csv",
    "tepai_d4096_N1000_S10_tstart3.00.csv",
    3.0,
    "Hybrid Trotter + TE-PAI",
)
FIG4 = (
    "nq100_chi16_j0.10_seed0_T5.00_dt0.10_X0_plus_flip",
    "trotter_N20_adaptiveTrue.csv",
    "tepai_d4096_N1000_S10_tstart0.00.csv",
    0.0,
    "TE-PAI from start",
)


def _load_experiment(folder, trot_file, tepai_file):
    base = os.path.join(DATA_DIR, folder)
    _, trot = _load_csv(os.path.join(base, trot_file))
    _, tepai = _load_csv(os.path.join(base, tepai_file))
    return trot, tepai


def _advantage_crossover(trot_t, trot_gates, tepai_final):
    """Return time at which Trotter cumulative gates would exceed the
    final TE-PAI gate count, or None if no crossover."""
    if tepai_final >= trot_gates[-1]:
        return None
    for k in range(1, len(trot_gates)):
        if trot_gates[k] >= tepai_final:
            frac = ((tepai_final - trot_gates[k - 1])
                    / (trot_gates[k] - trot_gates[k - 1]))
            return trot_t[k - 1] + frac * (trot_t[k] - trot_t[k - 1])
    return None


def _plot_expectation(ax, trot, tepai, tstart):
    ax.plot(trot["time"], trot["observable"], color="black", label="Trotter")

    tepai_t = tepai["time"]
    tepai_obs = tepai["observable"]

    # TE-PAI advantage region (green shading) — same logic as plot_results
    t_cross = _advantage_crossover(trot["time"], trot["gate_count"],
                                    tepai["gate_count"][-1])
    if t_cross is not None:
        ax.axvline(t_cross, color="tab:green", ls="--", lw=1)
        ax.axvspan(t_cross, trot["time"][-1],
                   color="tab:green", alpha=0.10,
                   label="TE-PAI advantage")

    # Empirical error = |TE-PAI mean - Trotter reference|
    trotter_interp = np.interp(tepai_t, trot["time"], trot["observable"])
    emp_err = np.abs(tepai_obs - trotter_interp)

    lbl = "Trotter + TE-PAI" if tstart > 0 else "TE-PAI"
    ax.errorbar(tepai_t, tepai_obs, yerr=emp_err,
                fmt="x", color="tab:green", markersize=5,
                capsize=3, capthick=1, elinewidth=1,
                zorder=5, label=lbl)

    if tstart > 0:
        ax.axvline(tstart, color="tab:green", ls=":", lw=1,
                   label=f"TE-PAI start ($t={tstart}$)")

    ax.set_xlim(0, trot["time"][-1])
    ax.legend(loc="upper right", fontsize=10)

    # --- Insets: error (tstart == 0) or zoom on last 6 points (tstart > 0) ---
    _add_expectation_inset(ax, trot, tepai, tstart)


def _add_expectation_inset(ax, trot, tepai, tstart):
    """Replicates the two inset styles used in run_experiment.plot_results."""
    import matplotlib.ticker as mticker
    from matplotlib.patches import Rectangle

    trot_t = trot["time"]
    trot_obs = trot["observable"]
    tepai_t = tepai["time"]
    tepai_obs = tepai["observable"]

    if tstart == 0:
        # Error inset: |TE-PAI - Trotter| vs t
        axins = ax.inset_axes([0.08, 0.15, 0.38, 0.35])
        trotter_interp = np.interp(tepai_t, trot_t, trot_obs)
        err = np.abs(tepai_obs - trotter_interp)
        axins.plot(tepai_t, err, color="tab:green", lw=1.2)
        axins.set_xlabel("$t$", fontsize=10, labelpad=2)
        axins.set_ylabel("Error", fontsize=10, labelpad=2)
        axins.tick_params(labelsize=8, pad=2)
        axins.set_xlim(tepai_t[0], tepai_t[-1])

        err_max = float(np.max(err))
        if err_max > 0:
            exp = int(np.floor(np.log10(err_max)))
            sc = 10 ** exp
            axins.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _, s=sc: f"{v / s:.0f}")
            )
            axins.yaxis.set_major_locator(mticker.MultipleLocator(sc))
            axins.text(
                0.02, 0.95, rf"$\times 10^{{{exp}}}$",
                transform=axins.transAxes, fontsize=8,
                va="top", ha="left",
            )
        return

    # tstart > 0 — zoomed inset on last 6 datapoints with dashed rectangle
    n_zoom = min(6, len(trot_t))
    t_zoom_start = trot_t[-n_zoom]
    t_zoom_end = trot_t[-1]
    dt_plot = trot_t[1] - trot_t[0] if len(trot_t) > 1 else 0.1

    zoom_ys = list(trot_obs[-n_zoom:])
    tepai_mask = (tepai_t >= t_zoom_start - 1e-9) & (
        tepai_t <= t_zoom_end + 1e-9
    )
    if np.any(tepai_mask):
        zoom_ys.extend(tepai_obs[tepai_mask])

    y_min, y_max = min(zoom_ys), max(zoom_ys)
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.1

    axins = ax.inset_axes([0.02, 0.02, 0.38, 0.38])
    axins.plot(trot_t[-n_zoom:], trot_obs[-n_zoom:], "k-", lw=2.5)
    if np.any(tepai_mask):
        axins.scatter(
            tepai_t[tepai_mask], tepai_obs[tepai_mask],
            marker="x", color="tab:green", s=80,
            linewidths=2, zorder=5,
        )

    x0 = t_zoom_start - 0.3 * dt_plot
    x1 = t_zoom_end + 0.3 * dt_plot
    y0 = y_min - y_pad
    y1 = y_max + y_pad
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_xticks([])
    axins.set_yticks([])

    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="black", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def _plot_gates(ax, trot, tepai, tstart):
    import matplotlib.ticker as ticker

    trot_gates = trot["gate_count"]
    tepai_gates = tepai["gate_count"]
    tepai_final = tepai_gates[-1]
    total_trotter = trot_gates[-1]
    mx = max(total_trotter, tepai_final)
    exp = int(np.floor(np.log10(mx))) if mx > 0 else 0
    sc = 10 ** exp if exp > 0 else 1

    ax.plot(trot["time"], trot_gates / sc, color="black", label="Trotter")
    ax.plot(tepai["time"], tepai_gates / sc, color="tab:green",
            label="Trotter + TE-PAI" if tstart > 0 else "TE-PAI")

    # Advantage shading: region where Trotter would need more gates than
    # the TE-PAI final cost.
    t_cross = _advantage_crossover(trot["time"], trot_gates, tepai_final)
    if t_cross is not None:
        ax.axhline(tepai_final / sc, color="tab:green", ls="--", lw=1)
        above = trot_gates >= tepai_final
        idx0 = int(np.argmax(above))
        tr = np.insert(trot["time"][idx0:], 0, t_cross)
        gr = np.insert(trot_gates[idx0:], 0, tepai_final)
        ax.fill_between(tr, tepai_final / sc, gr / sc,
                        color="gray", alpha=0.3,
                        label="Additional Trotter gates")

    if tstart > 0:
        ax.axvline(tstart, color="tab:green", ls=":", lw=1)

    ax.set_xlim(0, trot["time"][-1])
    ax.set_xlabel("Time")
    if exp > 0:
        ax.set_ylabel(rf"Cumulative gate count $(\times 10^{{{exp}}})$")
    else:
        ax.set_ylabel("Cumulative gate count")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2g}")
    )
    ax.legend(loc="upper left", fontsize=10)


def main():
    apply_style()
    import matplotlib.pyplot as plt

    trot3, tepai3 = _load_experiment(*FIG3[:3])
    trot4, tepai4 = _load_experiment(*FIG4[:3])

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7),
        gridspec_kw={"hspace": 0.30, "wspace": 0.25},
    )
    (axA, axB), (axC, axD) = axes

    # --- Top row: expectation values (share y axis + label) ---
    _plot_expectation(axA, trot3, tepai3, FIG3[3])
    _plot_expectation(axB, trot4, tepai4, FIG4[3])
    axA.set_ylabel(r"$\langle X_0 \rangle$")

    # Share y between the two top panels
    y_lo = min(axA.get_ylim()[0], axB.get_ylim()[0])
    y_hi = max(axA.get_ylim()[1], axB.get_ylim()[1])
    axA.set_ylim(y_lo, y_hi)
    axB.set_ylim(y_lo, y_hi)
    axB.sharey(axA)
    plt.setp(axB.get_yticklabels(), visible=False)

    # --- Bottom row: cumulative gate counts ---
    _plot_gates(axC, trot3, tepai3, FIG3[3])
    _plot_gates(axD, trot4, tepai4, FIG4[3])

    # Drop duplicate y label on D (bottom-right gate-count panel)
    axD.set_ylabel("")

    # --- Bold panel labels above top-left of each axis ---
    for label, ax in (("A)", axA), ("B)", axB), ("C)", axC), ("D)", axD)):
        ax.text(0.01, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    out_path = os.path.join(DATA_DIR, "figure_3_4_combined.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Plot -> {out_path}")
    if sys.platform == "darwin":
        plt.show()


if __name__ == "__main__":
    main()
