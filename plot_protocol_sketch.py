"""Plot a sketch figure comparing the standard Trotterization protocol
with the MPS TE-PAI protocol (circuits + stylized complexity plots).

The two circuits are drawn manually (not via qiskit) so that:
  * colors stay vivid and identical across both circuits,
  * the TE-PAI circuit keeps the same layout as the Trotter one, with
    "removed" gates rendered as empty dashed boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch

try:
    from plot_style import apply_style
    apply_style()
except ImportError:
    pass


# ---- color palette -----------------------------------------------------------
RZ_FILL = "#2196F3"   # vivid blue
RZ_TEXT = "#ffffff"
XX_FILL = "#A51C54"   # deep magenta
XX_TEXT = "#ffffff"

GATE_COLORS = {
    "rz":  (RZ_FILL, RZ_TEXT),
    "rxx": (XX_FILL, XX_TEXT),
    "ryy": (XX_FILL, XX_TEXT),
    "rzz": (XX_FILL, XX_TEXT),
}

# ---- circuit canvas layout ---------------------------------------------------
N_Q = 4
WIRE_Y = [3, 2, 1, 0]          # q0 at top
GATE_W = 0.58
GATE_H = 0.58
EDGE_LW = 1.4

# column x-positions
X_RZ = 0.6
X_EVEN = [1.7, 2.4, 3.1]       # bonds (0,1) and (2,3): Rxx,Ryy,Rzz
X_ODD  = [4.2, 4.9, 5.6]       # bond (1,2): Rxx,Ryy,Rzz
X_WRAP = [6.7, 7.4, 8.1]       # bond (0,3): Rxx,Ryy,Rzz
X_BARRIERS = [1.1, 3.65, 6.15, 8.6]
X_MIN, X_MAX = -0.6, 8.9
Y_MIN, Y_MAX = -1.1, 3.6


def _wire_y(q):
    return WIRE_Y[q]


def draw_wires(ax):
    for q in range(N_Q):
        y = _wire_y(q)
        ax.plot([X_MIN + 0.2, X_MAX - 0.1], [y, y],
                color="black", lw=1.2, zorder=1)
        ax.text(X_MIN + 0.15, y, f"$q_{{{q}}}$",
                ha="right", va="center", fontsize=14)


def draw_barriers(ax):
    for x in X_BARRIERS:
        ax.plot([x, x], [-0.35, 3.35],
                color="gray", linestyle=(0, (1, 2)), lw=1.0, zorder=1)


def _gate_rect(x_center, y_bot, height, fill, dashed):
    return Rectangle(
        (x_center - GATE_W / 2, y_bot),
        GATE_W, height,
        facecolor="none" if dashed else fill,
        edgecolor="black",
        linewidth=EDGE_LW,
        linestyle=(0, (3, 2)) if dashed else "-",
        zorder=3,
    )


def draw_single(ax, x, qubit, label, kind, dashed=False, fontsize=12):
    y = _wire_y(qubit)
    fill, text_color = GATE_COLORS[kind]
    ax.add_patch(_gate_rect(x, y - GATE_H / 2, GATE_H, fill, dashed))
    if not dashed:
        ax.text(x, y, label, ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight="bold", zorder=4)


def draw_multi(ax, x, qubits, label, kind, dashed=False, fontsize=12):
    ys = [_wire_y(q) for q in qubits]
    y_top = max(ys) + GATE_H / 2
    y_bot = min(ys) - GATE_H / 2
    height = y_top - y_bot
    y_center = (y_top + y_bot) / 2
    fill, text_color = GATE_COLORS[kind]
    ax.add_patch(_gate_rect(x, y_bot, height, fill, dashed))
    if not dashed:
        ax.text(x, y_center, label, ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight="bold", zorder=4)


def setup_circuit_axes(ax, title):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=10,
                 loc="left", x=0.02)


# ---- circuit specifications --------------------------------------------------
def draw_trotter(ax):
    setup_circuit_axes(ax, "Standard Trotterization protocol")
    draw_wires(ax)
    draw_barriers(ax)

    # Rz on every qubit
    for q in range(N_Q):
        draw_single(ax, X_RZ, q, "Rz", "rz")

    # even-bond column: (0,1) and (2,3)
    for x, name in zip(X_EVEN, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [0, 1], name, name.lower())
        draw_multi(ax, x, [2, 3], name, name.lower())

    # odd bond (1,2)
    for x, name in zip(X_ODD, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [1, 2], name, name.lower())

    # wrap-around bond (0,3)
    for x, name in zip(X_WRAP, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [0, 3], name, name.lower())

    # blue dashed "x N" box around the whole gate region
    box_color = "#1E88E5"
    x0, x1 = X_RZ - 0.55, X_WRAP[-1] + 0.55
    y0, y1 = -0.45, 3.45
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor="none", edgecolor=box_color,
        linewidth=2.0, linestyle=(0, (5, 3)), zorder=5,
    ))
    ax.text((x0 + x1) / 2, y0 - 0.25, r"$\times\,N$",
            ha="center", va="top", color=box_color,
            fontsize=18, fontweight="bold")


def draw_tepai(ax):
    setup_circuit_axes(ax, "MPS TE-PAI protocol")
    draw_wires(ax)
    draw_barriers(ax)

    # Rz column: q1 -> Rz/(+Δ), q2 -> Rz/(-Δ), q0 & q3 removed
    draw_single(ax, X_RZ, 0, "", "rz", dashed=True)
    draw_single(ax, X_RZ, 1, "Rz\n(+Δ)", "rz", fontsize=7)
    draw_single(ax, X_RZ, 2, "Rz\n(-Δ)", "rz", fontsize=7)
    draw_single(ax, X_RZ, 3, "", "rz", dashed=True)

    # even bonds: only Rxx on (0,1), only Rzz on (2,3)
    even_keep = {
        (0, "rxx"): "Rxx\n(+Δ)",
        (1, "rzz"): "Rzz\n(-Δ)",
    }
    for col_idx, name in enumerate(["Rxx", "Ryy", "Rzz"]):
        kind = name.lower()
        x = X_EVEN[col_idx]
        lbl = even_keep.get((0, kind))
        draw_multi(ax, x, [0, 1], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)
        lbl = even_keep.get((1, kind))
        draw_multi(ax, x, [2, 3], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)

    # odd bond (1,2): Ryy, Rzz(π); Rxx removed
    odd_keep = {"ryy": "Ryy\n(-Δ)", "rzz": "Rzz\n(π)"}
    for col_idx, name in enumerate(["Rxx", "Ryy", "Rzz"]):
        kind = name.lower()
        lbl = odd_keep.get(kind)
        draw_multi(ax, X_ODD[col_idx], [1, 2], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)

    # wrap bond (0,3): Rxx, Ryy; Rzz removed
    wrap_keep = {"rxx": "Rxx\n(+Δ)", "ryy": "Ryy\n(+Δ)"}
    for col_idx, name in enumerate(["Rxx", "Ryy", "Rzz"]):
        kind = name.lower()
        lbl = wrap_keep.get(kind)
        draw_multi(ax, X_WRAP[col_idx], [0, 3], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)


# ---- stylized plots ----------------------------------------------------------
def style_stub_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.3)


def plot_complexity(ax):
    t = np.linspace(0.0, 2.5, 500)
    trot_color = "black"
    tepai_color = "tab:green"
    limit_color = "#c0362c"

    y_trot = np.exp(2.0 * t) - 1
    # TE-PAI only from t=1 onward (where max(t²,1) diverges from 1)
    mask = t >= 1.0
    t_te = t[mask]
    y_tepai = (np.exp(2.0 * t_te) - 1) / t_te ** 2
    limit = 20.0

    ax.plot(t, y_trot, color=trot_color, lw=2.2, label="Trotter")
    ax.plot(t_te, y_tepai, color=tepai_color, lw=2.2,
            label="TE-PAI (fewer gates)")
    ax.axhline(limit, color=limit_color, lw=1.6, label="Computational limit")

    for y_line, t_arr, col in [(y_trot, t, trot_color),
                                (y_tepai, t_te, tepai_color)]:
        hits = np.where(y_line >= limit)[0]
        if hits.size:
            t_cross = t_arr[hits[0]]
            ax.vlines(t_cross, 0, limit,
                      colors=col, linestyles="--", lw=1.6)

    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Complexity")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 55)
    style_stub_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=11)


def plot_gate_count(ax):
    t = np.linspace(0, 1, 300)
    trot = 9 * t ** 2
    ax.plot(t, trot, color="black", lw=2.2, label="Trotter")
    t0 = 0.5
    y0 = 9 * t0 ** 2
    slope = 2 * 9 * t0 * 0.55
    t2 = np.linspace(t0, 1, 150)
    y2 = y0 + slope * (t2 - t0)
    ax.plot(t2, y2, color="#2ca05a", lw=2.2, label="TE-PAI")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Gate-count")
    style_stub_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=13)


def plot_overhead(ax):
    t = np.linspace(0, 1, 300)
    overhead = np.exp(3.5 * t)    # starts at 1
    shots = 2.8 * np.sqrt(t)      # starts at 0, sub-exponential, < overhead
    ax.plot(t, overhead, color="#c0362c", lw=2.2, label="Overhead")
    ax.plot(t, shots, color="#2ca05a", lw=2.2, linestyle="--",
            label="TE-PAI shots")
    ax.set_xlabel("Simulation time")
    ax.set_ylim(bottom=0)
    style_stub_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=13)


# ---- assembly ----------------------------------------------------------------
def main(out_path="data/protocol_sketch.pdf"):
    fig = plt.figure(figsize=(14, 9.0))
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.18, wspace=0.02,
        width_ratios=[1.7, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.05, right=0.98, top=0.95, bottom=0.08,
    )

    ax_tl = fig.add_subplot(outer[0, 0])
    ax_tr = fig.add_subplot(outer[0, 1])
    ax_bl = fig.add_subplot(outer[1, 0])

    inner_br = outer[1, 1].subgridspec(1, 2, wspace=0.22)
    ax_br1 = fig.add_subplot(inner_br[0, 0])
    ax_br2 = fig.add_subplot(inner_br[0, 1])

    draw_trotter(ax_tl)
    draw_tepai(ax_bl)

    plot_complexity(ax_tr)
    plot_gate_count(ax_br1)
    plot_overhead(ax_br2)

    for ax, letter in [(ax_tr, "A)"), (ax_br1, "B)"), (ax_br2, "C)")]:
        ax.text(-0.02, 1.01, letter, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=16, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved -> {out_path}")
    return fig


if __name__ == "__main__":
    main()
