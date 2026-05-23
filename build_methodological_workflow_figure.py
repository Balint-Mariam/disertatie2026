import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dissertation_plot_style import PALETTE, apply_theme


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str = "#2f2f2f",
    text_size: int = 10,
    weight: str = "normal",
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=text_size,
        color="#1f2937",
        weight=weight,
        zorder=4,
    )
    return x, y, w, h


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#4b5563",
) -> None:
    arr = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color=color,
        zorder=5,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)


def box_bottom_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, _ = box
    return x + w / 2.0, y


def box_top_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h


def draw_group_background(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    color: str,
    ) -> None:
    group = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=color,
        facecolor=color,
        alpha=0.10,
        zorder=0,
    )
    ax.add_patch(group)
    ax.text(
        x + 0.01,
        y + h - 0.012,
        title,
        ha="left",
        va="top",
        fontsize=11,
        color="#111827",
        weight="semibold",
        zorder=1,
    )


def build_figure(outdir: pathlib.Path, save_svg: bool = True) -> list[pathlib.Path]:
    apply_theme()
    fig, ax = plt.subplots(figsize=(13, 12.5))
    ax.set_position([0.03, 0.05, 0.94, 0.90])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Methodological workflow of the empirical analysis",
        fontsize=16,
        fontweight="semibold",
        color=PALETTE["navy"],
        y=0.992,
    )

    groups = [
        {
            "title": "A. Data preparation",
            "color": PALETTE["navy"],
            "box_fc": "#e8eef8",
            "steps": [
                "1) Raw option data",
                "2) Data cleaning and filtering",
                "3) Implied volatility extraction",
                "4) Standardized IV surface\nfixed moneyness-maturity grid",
            ],
        },
        {
            "title": "B. Forecasting",
            "color": PALETTE["teal"],
            "box_fc": "#e6f3f5",
            "steps": [
                "5) Train / validation / test split",
                "6) Forecasting models\nNaive / Persistence, ARIMA, XGBoost",
                "7) Forecast evaluation\nRMSE, MAE",
                "8) Model selection\nXGBoost",
            ],
        },
        {
            "title": "C. Trading strategy and\nportfolio construction",
            "color": PALETTE["gray"],
            "box_fc": "#f2f4f7",
            "steps": [
                "9) Trading signal construction\nforecast IV vs observed IV, z-score\nlong / short / flat",
                "10) Portfolio construction\nsynthetic IV-space backtest\nrealistic strict option portfolio",
                "11) Greeks computation\ndelta, gamma, vega, theta",
            ],
        },
        {
            "title": "D. Risk management\nand evaluation",
            "color": PALETTE["burgundy"],
            "box_fc": "#f8eaed",
            "steps": [
                "12) Hedging strategies\ndelta hedge, delta-gamma hedge",
                "13) Performance evaluation\nPnL, Sharpe ratio, max drawdown\nhedge effectiveness",
            ],
        },
    ]

    box_x, box_w = 0.20, 0.76
    box_h = 0.045
    intra_gap = 0.014
    inter_group_gap = 0.028
    top = 0.97

    placed_boxes: list[tuple[float, float, float, float]] = []
    y_cursor = top
    for g in groups:
        n = len(g["steps"])
        group_h = n * box_h + (n - 1) * intra_gap
        group_bottom = y_cursor - group_h

        draw_group_background(
            ax,
            x=0.02,
            y=group_bottom - 0.010,
            w=0.96,
            h=group_h + 0.020,
            title=g["title"],
            color=g["color"],
        )

        y_step = y_cursor - box_h
        for step_text in g["steps"]:
            b = add_box(
                ax,
                x=box_x,
                y=y_step,
                w=box_w,
                h=box_h,
                text=step_text,
                fc=g["box_fc"],
                ec="#364152",
                text_size=10,
            )
            placed_boxes.append(b)
            y_step -= box_h + intra_gap

        y_cursor = group_bottom - inter_group_gap

    # Main sequential arrows: 1 -> 13.
    for upper, lower in zip(placed_boxes[:-1], placed_boxes[1:]):
        s = box_bottom_center(upper)
        e = box_top_center(lower)
        add_arrow(ax, (s[0], s[1] - 0.003), (e[0], e[1] + 0.003), color="#475569")

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "figure_20_methodological_workflow.png"
    out_pdf = outdir / "figure_20_methodological_workflow.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    out_paths = [out_png, out_pdf]
    if save_svg:
        out_svg = outdir / "figure_20_methodological_workflow.svg"
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        out_paths.append(out_svg)
    plt.close(fig)
    return out_paths


def update_manifest(manifest_path: pathlib.Path) -> None:
    if not manifest_path.exists():
        return
    text = manifest_path.read_text(encoding="utf-8")
    entry = (
        "- `figures/figure_20_methodological_workflow.png`: End-to-end empirical workflow diagram "
        "(**Methodology chapter / Presentation slides**)\n"
    )
    if "figure_20_methodological_workflow.png" in text:
        return
    insert_point = "### Presentation-only Media\n"
    if insert_point in text:
        text = text.replace(insert_point, entry + "\n" + insert_point)
    else:
        marker = "## Figures\n\n"
        if marker in text:
            text = text.replace(marker, marker + entry)
        else:
            text = text + "\n\n## Figures\n\n" + entry
    manifest_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dissertation methodological workflow flowchart figure.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs/figures"))
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("dissertation_outputs/manifests/README_dissertation_outputs.md"))
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export.")
    args = parser.parse_args()

    paths = build_figure(outdir=args.outdir, save_svg=(not args.no_svg))
    update_manifest(args.manifest)
    print("Workflow figure generated:")
    for p in paths:
        print(f"- {p}")
    print(f"Manifest updated: {args.manifest}")


if __name__ == "__main__":
    main()
