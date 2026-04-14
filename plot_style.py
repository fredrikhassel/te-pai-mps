"""
Central plot style configuration for uniform text sizes across all figures.

Usage:
    from plot_style import apply_style
    apply_style()  # Call once at module level or before creating figures

All font sizes are controlled here so that every figure in the paper has
consistent text rendering.  Change the values below and re-run to update
all plots at once.
"""
import matplotlib.pyplot as plt

# ── Font sizes (change these to resize text globally) ────────────────────────
FONT_SIZE = 20
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14
LEGEND_TITLE_SIZE = 14

# ── Full rcParams for plot style ─────────────────────────────────────────────
STYLE = {
    # Font
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    # Axes
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    # Ticks
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    # Legend
    'legend.fontsize': LEGEND_SIZE,
    'legend.title_fontsize': LEGEND_TITLE_SIZE,
    # Figure
    'figure.titlesize': TITLE_SIZE,
    # Grid
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    # Lines
    'lines.linewidth': 2,
    # Error bars
    'errorbar.capsize': 3,
    # Save
    'savefig.dpi': 300,
}


def apply_style():
    """Apply the standard plot style. Call once before creating figures."""
    plt.rcParams.update(STYLE)