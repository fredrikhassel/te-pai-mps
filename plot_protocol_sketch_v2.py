"""Protocol sketch v2: circuits on the left, single complexity plot on the right.

Helvetica font, Oxford Blue titles, no B)/C) plots, no A) annotation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

try:
    from plot_style import apply_style
    apply_style()
except ImportError:
    pass

# ---- global font override: Helvetica ----------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

# ---- colors ------------------------------------------------------------------
# Oxford Blue in CMYK {1, .87, .42, .51} -> RGB
# R = 255*(1-1)*(1-0.51) = 0
# G = 255*(1-0.87)*(1-0.51) ≈ 16
# B = 255*(1-0.42)*(1-0.51) ≈ 72
OXFORD_BLUE = "#001048"

RZ_FILL = "#2196F3"
RZ_TEXT = "#ffffff"
XX_FILL = "#A51C54"
XX_TEXT = "#ffffff"

GATE_COLORS = {
    "rz":  (RZ_FILL, RZ_TEXT),
    "rxx": (XX_FILL, XX_TEXT),
    "ryy": (XX_FILL, XX_TEXT),
    "rzz": (XX_FILL, XX_TEXT),
}

# ---- circuit canvas layout ---------------------------------------------------
N_Q = 4
WIRE_Y = [3, 2, 1, 0]
GATE_W = 0.58
GATE_H = 0.58
EDGE_LW = 1.4

X_RZ = 0.6
X_EVEN = [1.7, 2.4, 3.1]
X_ODD  = [4.2, 4.9, 5.6]
X_WRAP = [6.7, 7.4, 8.1]
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
                 loc="left", x=0.02, color=OXFORD_BLUE)


# ---- circuit specifications --------------------------------------------------
def draw_trotter(ax):
    setup_circuit_axes(ax, "Standard Trotterization protocol")
    draw_wires(ax)
    draw_barriers(ax)

    for q in range(N_Q):
        draw_single(ax, X_RZ, q, "Rz", "rz")

    for x, name in zip(X_EVEN, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [0, 1], name, name.lower())
        draw_multi(ax, x, [2, 3], name, name.lower())

    for x, name in zip(X_ODD, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [1, 2], name, name.lower())

    for x, name in zip(X_WRAP, ["Rxx", "Ryy", "Rzz"]):
        draw_multi(ax, x, [0, 3], name, name.lower())

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

    draw_single(ax, X_RZ, 0, "", "rz", dashed=True)
    draw_single(ax, X_RZ, 1, "Rz\n(+Δ)", "rz", fontsize=7)
    draw_single(ax, X_RZ, 2, "Rz\n(-Δ)", "rz", fontsize=7)
    draw_single(ax, X_RZ, 3, "", "rz", dashed=True)

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

    odd_keep = {"ryy": "Ryy\n(-Δ)", "rzz": "Rzz\n(π)"}
    for col_idx, name in enumerate(["Rxx", "Ryy", "Rzz"]):
        kind = name.lower()
        lbl = odd_keep.get(kind)
        draw_multi(ax, X_ODD[col_idx], [1, 2], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)

    wrap_keep = {"rxx": "Rxx\n(+Δ)", "ryy": "Ryy\n(+Δ)"}
    for col_idx, name in enumerate(["Rxx", "Ryy", "Rzz"]):
        kind = name.lower()
        lbl = wrap_keep.get(kind)
        draw_multi(ax, X_WRAP[col_idx], [0, 3], lbl or "", kind,
                   dashed=(lbl is None), fontsize=11)


# ---- stylized plot -----------------------------------------------------------
def style_stub_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.3)


T_TEPAI_START = 1.0  # TE-PAI begins where max(t²,1) kicks in


def plot_complexity(ax):
    t = np.linspace(0.0, 2.5, 500)
    trot_color = "black"
    tepai_color = "tab:green"
    limit_color = "#c0362c"

    y_trot = np.exp(2.0 * t) - 1
    # TE-PAI only from T_TEPAI_START onward
    mask = t >= T_TEPAI_START
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

    ax.set_ylabel("Complexity", fontsize=16)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 55)
    style_stub_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=15)


def plot_expectation(ax):
    t = np.linspace(0.0, 2.5, 500)
    trot_color = "black"
    tepai_color = "tab:green"

    # damped oscillation (Trotterization)
    y_trot = np.exp(-0.6 * t) * np.cos(6 * t)
    ax.plot(t, y_trot, color=trot_color, lw=2.2, label="Trotterization")

    # TE-PAI scatter starting at T_TEPAI_START
    t_pts = np.append(np.arange(T_TEPAI_START, 2.35, 0.15), 2.35)
    y_pts = np.exp(-0.6 * t_pts) * np.cos(6 * t_pts)
    errs = 0.04 * np.exp(1.0 * (t_pts - T_TEPAI_START))

    ax.errorbar(t_pts, y_pts, yerr=errs,
                fmt="x", color=tepai_color, markersize=7, markeredgewidth=2,
                ecolor=tepai_color, elinewidth=1.6, capsize=3, capthick=1.4,
                label="TE-PAI (with overhead)", zorder=5)

    ax.set_ylabel("Expectation value", fontsize=16)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-1, 1)
    style_stub_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=15)


# ---- assembly ----------------------------------------------------------------
def main(out_path="data/protocol_sketch_v2.pdf"):
    fig = plt.figure(figsize=(14, 9.0))
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.18, wspace=0.02,
        width_ratios=[1.7, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.05, right=0.98, top=0.95, bottom=0.08,
    )

    ax_tl = fig.add_subplot(outer[0, 0])
    ax_bl = fig.add_subplot(outer[1, 0])

    # two right-column plots sharing x-axis
    inner_r = outer[:, 1].subgridspec(2, 1, hspace=0.08)
    ax_r1 = fig.add_subplot(inner_r[0])
    ax_r2 = fig.add_subplot(inner_r[1], sharex=ax_r1)
    plt.setp(ax_r1.get_xticklabels(), visible=False)

    draw_trotter(ax_tl)
    draw_tepai(ax_bl)
    plot_expectation(ax_r1)
    plot_complexity(ax_r2)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved -> {out_path}")
    return fig


if __name__ == "__main__":
    main()
