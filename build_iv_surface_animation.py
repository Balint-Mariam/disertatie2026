import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, cm, colors
import matplotlib as mpl

from dissertation_plot_style import PALETTE, apply_theme


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def get_sorted_dates(csv_path: pathlib.Path, chunksize: int) -> list[pd.Timestamp]:
    dates = set()
    for chunk in pd.read_csv(csv_path, usecols=["quote_date"], chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"]).dt.normalize()
        for d in qd.dropna().unique():
            dates.add(pd.Timestamp(d))
    return sorted(dates)


def select_frame_dates(
    dates: list[pd.Timestamp],
    frame_step: int,
    max_frames: int | None,
) -> list[pd.Timestamp]:
    if not dates:
        return []
    step = max(frame_step, 1)
    picked = dates[::step]
    if max_frames is not None and max_frames > 0:
        picked = picked[:max_frames]
    if len(picked) == 1 and len(dates) > 1:
        picked = [dates[0], dates[-1]]
    return picked


def load_selected_frames(
    csv_path: pathlib.Path,
    selected_dates: set[pd.Timestamp],
    chunksize: int,
) -> pd.DataFrame:
    parts = []
    usecols = ["quote_date", "log_moneyness", "T", "iv_grid"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"]).dt.normalize()
        chunk = chunk[chunk["quote_date"].isin(selected_dates)]
        if chunk.empty:
            continue
        chunk["log_moneyness"] = pd.to_numeric(chunk["log_moneyness"], errors="coerce")
        chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        chunk["iv_grid"] = pd.to_numeric(chunk["iv_grid"], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=["quote_date", "log_moneyness", "T", "iv_grid"])
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def build_surface_cube(
    df: pd.DataFrame,
    frame_dates: list[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    x_vals = np.sort(df["log_moneyness"].unique())
    t_vals = np.sort(df["T"].unique())
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)

    surfaces: list[np.ndarray] = []
    for d in frame_dates:
        day = df[df["quote_date"] == d]
        pivot = day.pivot_table(index="T", columns="log_moneyness", values="iv_grid", aggfunc="mean")
        pivot = pivot.reindex(index=t_vals, columns=x_vals)
        surfaces.append(pivot.to_numpy(dtype=float))
    return x_grid, t_grid, surfaces


def save_animation(
    frame_dates: list[pd.Timestamp],
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    surfaces: list[np.ndarray],
    out_mp4: pathlib.Path,
    out_gif: pathlib.Path | None,
    fps: int,
) -> tuple[bool, bool]:
    apply_theme()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    all_vals = np.concatenate([z[np.isfinite(z)] for z in surfaces if np.isfinite(z).any()])
    if all_vals.size == 0:
        raise ValueError("Nu exista valori finite pentru iv_grid in frame-urile selectate.")

    vmin = float(np.nanpercentile(all_vals, 1))
    vmax = float(np.nanpercentile(all_vals, 99))
    if vmin >= vmax:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(float(np.nanmin(t_grid)), float(np.nanmax(t_grid)))
    ax.set_ylim(float(np.nanmin(x_grid)), float(np.nanmax(x_grid)))
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("T (years)")
    ax.set_ylabel("log(K/S)")
    ax.set_zlabel("Implied Volatility")
    ax.view_init(elev=24, azim=36)
    ax.xaxis.label.set_color(PALETTE["navy"])
    ax.yaxis.label.set_color(PALETTE["navy"])
    ax.zaxis.label.set_color(PALETTE["navy"])

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("Implied Volatility")

    surface_holder = {"obj": None}

    def draw_frame(i: int):
        if surface_holder["obj"] is not None:
            surface_holder["obj"].remove()
        z = surfaces[i]
        z_masked = np.ma.masked_invalid(z)
        surface_holder["obj"] = ax.plot_surface(
            t_grid,
            x_grid,
            z_masked,
            cmap="viridis",
            norm=norm,
            rstride=1,
            cstride=1,
            linewidth=0.15,
            edgecolor=(0, 0, 0, 0.12),
            antialiased=True,
            alpha=0.96,
        )
        day_str = frame_dates[i].strftime("%Y-%m-%d")
        ax.set_title(f"Implied Volatility Surface Evolution | {day_str}")
        return (surface_holder["obj"],)

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(frame_dates),
        interval=max(1000 // max(fps, 1), 50),
        blit=False,
    )

    mp4_ok = False
    gif_ok = False
    try:
        configure_ffmpeg_if_needed()
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200)
        anim.save(out_mp4, writer=writer, dpi=180)
        mp4_ok = True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: nu am putut salva MP4 ({exc}).")

    if out_gif is not None:
        try:
            gif_writer = animation.PillowWriter(fps=fps)
            anim.save(out_gif, writer=gif_writer, dpi=140)
            gif_ok = True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: nu am putut salva GIF ({exc}).")

    plt.close(fig)
    return mp4_ok, gif_ok




def configure_ffmpeg_if_needed() -> None:
    if animation.writers.is_available("ffmpeg"):
        return
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IV surface time animation from existing standardized grid output.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("iv_grid_long.csv"))
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs/figures"))
    parser.add_argument("--frame-step", type=int, default=5, help="Use one frame every N dates (default: 5).")
    parser.add_argument("--fps", type=int, default=8, help="Video frames per second (default: 8).")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on number of frames (0 = no cap).")
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--save-gif", action="store_true", help="Also save GIF output.")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul de intrare: {args.csv}")

    header = pd.read_csv(args.csv, nrows=0)
    required = {"quote_date", "log_moneyness", "T", "iv_grid"}
    missing = required - set(header.columns)
    if missing:
        sys.exit(f"Lipsesc coloane necesare in {args.csv}: {', '.join(sorted(missing))}")

    all_dates = get_sorted_dates(args.csv, chunksize=args.chunksize)
    if not all_dates:
        sys.exit("Nu exista date valide in iv_grid_long.csv.")

    selected = select_frame_dates(
        dates=all_dates,
        frame_step=args.frame_step,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
    )
    if len(selected) < 2:
        sys.exit("Prea putine frame-uri selectate pentru animatie.")

    df = load_selected_frames(args.csv, set(selected), chunksize=args.chunksize)
    if df.empty:
        sys.exit("Nu exista observatii valide pentru frame-urile selectate.")

    x_grid, t_grid, surfaces = build_surface_cube(df, selected)
    if not surfaces:
        sys.exit("Nu am putut construi suprafetele pentru animatie.")

    out_mp4 = args.outdir / "iv_surface_animation.mp4"
    out_gif = args.outdir / "iv_surface_animation.gif" if args.save_gif else None
    mp4_ok, gif_ok = save_animation(
        frame_dates=selected,
        x_grid=x_grid,
        t_grid=t_grid,
        surfaces=surfaces,
        out_mp4=out_mp4,
        out_gif=out_gif,
        fps=max(args.fps, 1),
    )

    duration = len(selected) / max(args.fps, 1)
    print("\n=== Animation Summary ===")
    print(f"Input dates available: {len(all_dates)}")
    print(f"Frames used: {len(selected)} (step={args.frame_step}, max_frames={args.max_frames})")
    print(f"FPS: {args.fps}")
    print(f"Approx duration: {duration:.1f} sec")
    print(f"MP4 saved: {'yes' if mp4_ok else 'no'} -> {out_mp4}")
    if args.save_gif:
        print(f"GIF saved: {'yes' if gif_ok else 'no'} -> {out_gif}")


if __name__ == "__main__":
    main()
