"""Generează figura fluxului metodologic în limba română."""

from __future__ import annotations

import argparse
import pathlib
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dissertation_plot_style import apply_theme


def _wrap(text: str, width: int = 44) -> str:
    return textwrap.fill(text, width=width, break_long_words=False)


def build_figure(outdir: pathlib.Path) -> list[pathlib.Path]:
    apply_theme()

    fig = plt.figure(figsize=(15.5, 9.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Fluxul metodologic al analizei empirice",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#1f3c88",
    )

    groups = [
        {
            "title": "A. Pregătirea Datelor",
            "color": "#dfe3eb",
            "steps": [
                "1) Date brute de opțiuni",
                "2) Curățare și filtrare a datelor",
                "3) Extracția volatilității implicite",
                "4) Suprafața IV standardizată\nGrilă fixă moneyness-maturitate",
            ],
        },
        {
            "title": "B. Prognoză",
            "color": "#d9e7ec",
            "steps": [
                "5) Împărțire train / validation / test",
                "6) Modele de prognoză\nNaive / Persistence, ARIMA, XGBoost",
                "7) Evaluarea prognozei\nRMSE, MAE",
                "8) Selecția modelului\nXGBoost",
            ],
        },
        {
            "title": "C. Strategie De Tranzacționare\nȘi Construcția Portofoliului",
            "color": "#e4e6ea",
            "steps": [
                "9) Construcția semnalului de tranzacționare\nIV prognozată vs IV observată, z-score\nlong / short / flat",
                "10) Construcția portofoliului\nBacktest sintetic în spațiul IV\nPortofoliu realist strict pe opțiuni",
                "11) Calculul sensibilităților (Greeks)\nDelta, gamma, vega, theta",
            ],
        },
        {
            "title": "D. Managementul Riscului\nȘi Evaluare",
            "color": "#eadfe3",
            "steps": [
                "12) Strategii de acoperire\nDelta hedge, delta-gamma hedge",
                "13) Evaluarea performanței\nPnL, Sharpe ratio, max drawdown\nEficiența acoperirii",
            ],
        },
    ]

    panel_x = 0.025
    panel_w = 0.95
    left_col_w = 0.235
    group_gap = 0.012
    top_y = 0.93
    bottom_y = 0.05
    available_h = top_y - bottom_y

    weights = [len(g["steps"]) + 0.75 for g in groups]
    total_weight = sum(weights)
    total_gaps = group_gap * (len(groups) - 1)
    usable_h = available_h - total_gaps
    group_heights = [usable_h * (w / total_weight) for w in weights]

    box_edge = "#334155"
    arrow_color = "#58677d"

    cursor_top = top_y
    for gi, (group, g_h) in enumerate(zip(groups, group_heights)):
        group_bottom = cursor_top - g_h
        group_box = FancyBboxPatch(
            (panel_x, group_bottom),
            panel_w,
            g_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.3,
            edgecolor="#c6cfdb",
            facecolor=group["color"],
            alpha=0.95,
        )
        ax.add_patch(group_box)

        ax.text(
            panel_x + 0.018,
            cursor_top - 0.015,
            group["title"],
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#111827",
        )

        step_x = panel_x + left_col_w + 0.02
        step_w = panel_w - left_col_w - 0.035
        step_n = len(group["steps"])
        top_inner = cursor_top - 0.015
        bottom_inner = group_bottom + 0.018
        step_gap = 0.008
        step_h = (top_inner - bottom_inner - step_gap * (step_n - 1)) / step_n

        for si, step in enumerate(group["steps"]):
            y_top = top_inner - si * (step_h + step_gap)
            y = y_top - step_h

            step_patch = FancyBboxPatch(
                (step_x, y),
                step_w,
                step_h,
                boxstyle="round,pad=0.009,rounding_size=0.011",
                linewidth=1.2,
                edgecolor=box_edge,
                facecolor="#f8fafc",
            )
            ax.add_patch(step_patch)

            text = step if "\n" in step else _wrap(step, width=50)
            ax.text(
                step_x + step_w / 2,
                y + step_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=11.5,
                color="#1f2937",
            )

            center_x = step_x + step_w / 2
            if si < step_n - 1:
                next_y_top = y - step_gap
                arrow = FancyArrowPatch(
                    (center_x, y - 0.001),
                    (center_x, next_y_top + 0.001),
                    arrowstyle="-|>",
                    mutation_scale=24,
                    linewidth=1.5,
                    color=arrow_color,
                )
                ax.add_patch(arrow)

        if gi < len(groups) - 1:
            next_top = group_bottom - group_gap
            center_x = step_x + step_w / 2
            inter_arrow = FancyArrowPatch(
                (center_x, group_bottom + 0.003),
                (center_x, next_top - 0.002),
                arrowstyle="-|>",
                mutation_scale=24,
                linewidth=1.6,
                color=arrow_color,
            )
            ax.add_patch(inter_arrow)

        cursor_top = group_bottom - group_gap

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "figure_20_flux_metodologic.png"
    out_pdf = outdir / "figure_20_flux_metodologic.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    return [out_png, out_pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generează figura fluxului metodologic în limba română.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs_ro/figures"))
    args = parser.parse_args()
    outputs = build_figure(args.outdir)
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
