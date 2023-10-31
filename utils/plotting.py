import numpy as np


def fill_between_segments(ax, x, lower, upper, mask, alpha=0.3, color="tab:orange"):
    change_points = np.where(mask[:-1] != mask[1:])[0]
    start = 0
    for cp in change_points:
        if not mask[start]:
            ax.fill_between(
                x[start : cp + 1],
                lower[start : cp + 1],
                upper[start : cp + 1],
                alpha=alpha,
                color=color,
            )
        start = cp + 1
    if not mask[start]:
        ax.fill_between(
            x[start:], lower[start:], upper[start:], alpha=alpha, color=color
        )


def plot_segments(ax, x, y, mask, color1, color2, linestyle1, linestyle2, alpha=0.8):
    change_points = np.where(mask[:-1] != mask[1:])[0]
    start = 0
    for cp in change_points:
        ax.plot(
            x[start : cp + 1],
            y[start : cp + 1],
            color=color1 if mask[start] else color2,
            linestyle=linestyle1 if mask[start] else linestyle2,
            alpha=alpha,
        )
        start = cp + 1
    ax.plot(
        x[start:],
        y[start:],
        color=color1 if mask[start] else color2,
        linestyle=linestyle1 if mask[start] else linestyle2,
        alpha=alpha,
    )
