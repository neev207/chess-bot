"""Shared, restrained plot style: one accent colour, thin marks, light grid, direct labels."""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2933"       # text
MUTED = "#6b7785"     # secondary text / axes
GRID = "#e4e8ec"
ACCENT = "#2a6fb0"    # single-series bars / lines
ACCENT2 = "#d9822b"   # second series (training vs validation)
DIV_LOW, DIV_HIGH = "#d9822b", "#2a6fb0"  # diverging pair; neutral midpoint is white/grey


def style_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    return fig, ax
