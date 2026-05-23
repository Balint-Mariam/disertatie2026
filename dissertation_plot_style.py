import pathlib

import matplotlib.pyplot as plt


PALETTE = {
    "navy": "#1f3b73",
    "teal": "#1f7a8c",
    "gray": "#6c757d",
    "light_gray": "#ced4da",
    "orange": "#c97a2b",
    "burgundy": "#7a1f2b",
    "green": "#2a9d8f",
}

MODEL_COLORS = {
    "Naive": PALETTE["navy"],
    "ARIMA": PALETTE["gray"],
    "XGBoost": PALETTE["teal"],
}

STRATEGY_COLORS = {
    "Unhedged": PALETTE["navy"],
    "Delta Hedged": PALETTE["teal"],
    "Delta-Gamma Hedged": PALETTE["burgundy"],
    "Simple": PALETTE["gray"],
}

SIGNAL_COLORS = {
    "LONG": PALETTE["teal"],
    "SHORT": PALETTE["burgundy"],
    "FLAT": PALETTE["gray"],
}


def apply_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.6),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#d0d0d0",
            "legend.fontsize": 10,
            "grid.alpha": 0.25,
            "grid.color": "#9aa0a6",
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        }
    )


def save_figure(fig, out_png: pathlib.Path, dpi: int = 300, save_pdf: bool = True) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    if save_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)

