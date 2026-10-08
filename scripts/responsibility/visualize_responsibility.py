"""A video of an agent's counterfactual responsibility through one scene: one
frame per context step t_k, stitched into a GIF and an MP4 (the MP4 needs
OpenCV).

Each frame shows
  scene     the map, every agent at t_k with its last second of history, the
            queried agent's motion set (DenseTNT goal samples, the first 2 s
            solid and coloured by goal probability, the rest faint), its logged
            future, and the neighbours the metrics compared it with; the
            neighbour behind beta_s is outlined in red, the one behind beta_c
            in blue
  courtesy  that neighbour's goal distribution with the agent in the scene and
            without it (the two sides of the KL)
  timeline  beta_s (m) and beta_c (nats) over the clip, the current step marked,
            and, given a levels.csv from fit_levels.py, the level of every window
            as background colour (aggressive levels hatched)

Two ways to run it:

  live      --scene N: computes the frames with DenseTNT, or --model mtr (the same code as
            compute_responsibility.py; --run takes a run's settings so the
            values equal its windows.csv) and saves them as record.pkl
  offline   --record path/to/<scene>.pkl: renders a record written by
            compute_responsibility.py --save-records (or by a live run) --
            no DenseTNT, TensorFlow or scene files needed

Examples (from the repository root):
    python -m scripts.responsibility.visualize_responsibility --scene 17 \\
        --run logs/responsibility/sdc --levels logs/responsibility/levels/levels.csv \\
        --out-dir logs/responsibility/video_17
    python -m scripts.responsibility.visualize_responsibility \\
        --record logs/responsibility/sdc/records/17.pkl --out-dir logs/responsibility/video_17
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from responsibility.interaction import InteractionConfig  # noqa: E402
from responsibility.metrics import MOTION_SETS, ResponsibilityConfig  # noqa: E402
from responsibility.records import load_record, save_record, scene_of  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

AGENT_COLOR = "#d62728"
NEIGHBOUR_COLOR = "#1f77b4"
OTHER_COLOR = "#9a9a9a"
SAFETY_COLOR = "#d62728"
COURTESY_COLOR = "#1f77b4"
MAP_STYLE = {  # type prefix -> (colour, width)
    "ROAD_EDGE": ("#505050", 1.0),
    "ROAD_LINE": ("#b0b0b0", 0.6),
    "LANE": ("#dcdcdc", 0.8),
    "CROSSWALK": ("#c9b458", 0.8),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="Live: scene file index (e.g. 17) or path to a .pkl scene.")
    src.add_argument("--record", help="Offline: a record from compute_responsibility.py --save-records.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--levels", default=None, help="levels.csv from fit_levels.py, for the timeline.")
    p.add_argument("--view-radius", type=float, default=None, help="m; default: fitted to the scene (30-90 m).")
    p.add_argument("--ego-heatmap", action="store_true", help="Also draw the agent's own goal distribution.")
    p.add_argument("--fps", type=float, default=2.0)
    live = p.add_argument_group("live mode")
    live.add_argument("--scenes", default="raw_scenes_500")
    live.add_argument("--agent", default="sdc", help="sdc, adv or a track id.")
    live.add_argument("--run", default=None, help="compute_responsibility.py output whose settings to use.")
    d = ResponsibilityConfig()
    live.add_argument("--stride", type=int, default=d.window_stride)
    live.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    live.add_argument("--motion-set", default=d.motion_set, choices=MOTION_SETS,
                      help="As compute_responsibility.py's; the video draws exactly the scored set.")
    live.add_argument("--horizon", type=int, default=d.metric_horizon)
    live.add_argument("--seed", type=int, default=d.seed)
    live.add_argument("--device", default=None, help="Default: cuda if available.")
    from responsibility.models import add_model_arguments

    add_model_arguments(p)  # --model densetnt|mtr --checkpoint ... (as for the run being shown)
    return p.parse_args()


# ----------------------------------------------------------------------------
# live capture
# ----------------------------------------------------------------------------


def config_from(args) -> ResponsibilityConfig:
    if args.run is None:
        return ResponsibilityConfig(n_safety_samples=args.n_samples, metric_horizon=args.horizon,
                                    window_stride=args.stride, seed=args.seed, motion_set=args.motion_set)
    saved = json.loads((Path(args.run) / "config.json").read_text())["responsibility"]
    saved["interaction"] = InteractionConfig(**saved["interaction"])
    return ResponsibilityConfig(**saved)


def pick_agent(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        return [i for i in scene.objects_of_interest if i != scene.sdc][0]
    return scene.index(which)


def live_record(args):
    import torch

    from responsibility.models import load_model
    from responsibility.records import run_scene

    path = {p.stem: p for p in scene_files(args.scenes)}[args.scene] if args.scene.isdigit() else Path(args.scene)
    scene = Scene.load(path)
    agent = pick_agent(scene, args.agent)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_from(args)
    observations, record = run_scene(load_model(args, device), scene, agent, cfg)
    record["scene_file"] = path.stem
    for obs in observations:
        print(f"t={obs.step / 10:.1f}s: safety {obs.safety:+.3f} m, courtesy {obs.courtesy:.4f} nats, "
              f"{len(obs.per_neighbour)} neighbour(s)", flush=True)
    return record


# ----------------------------------------------------------------------------
# rendering (from records only)
# ----------------------------------------------------------------------------


def map_segments(scene: Scene):
    """(colour, width) -> polylines, per map feature type group."""
    groups = {}
    for feature in scene.map_features.values():
        kind = str(feature.get("type", ""))
        style = next((v for k, v in MAP_STYLE.items() if kind.startswith(k)), None)
        pts = feature.get("polyline", feature.get("polygon"))
        if style is None or pts is None:
            continue
        pts = np.asarray(pts)[:, :2]
        if kind.startswith("CROSSWALK") and len(pts) > 2:
            pts = np.concatenate([pts, pts[:1]])
        if len(pts) > 1:
            groups.setdefault(style, []).append(pts)
    return groups


def box(center, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    dx = np.array([1, 1, -1, -1]) * length / 2
    dy = np.array([1, -1, -1, 1]) * width / 2
    return np.stack([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy], -1)


def worst(obs, key):
    items = [(k, v[key]) for k, v in obs["per_neighbour"].items() if v.get(key) is not None]
    return max(items, key=lambda kv: kv[1], default=(None, None))


def fit_radius(scene, agent, frames, horizon):
    need = 25.0
    for f in frames:
        centre = scene.position[agent, f["step"], :2]
        pts = [f["samples"][:, :horizon].reshape(-1, 2)]
        pts += [scene.position[scene.index(b), f["step"], :2][None] for b in f["observation"]["per_neighbour"]]
        need = max(need, float(np.abs(np.concatenate(pts) - centre).max()) + 8.0)
    return float(np.clip(need, 30.0, 90.0))


def draw_scene(ax, scene, agent, frame, map_groups, radius, horizon, ego_heatmap, weighted=False):
    step, obs = frame["step"], frame["observation"]
    centre = scene.position[agent, step, :2]
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    for (colour, width), lines in map_groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))
    if ego_heatmap:
        g = frame["goals"]
        ax.scatter(g["points"][:, 0], g["points"][:, 1], c=np.log(g["prob"]), cmap="Greys", s=4, alpha=0.5,
                   zorder=2, linewidths=0)

    neighbours = {scene.index(b) for b in obs["per_neighbour"]}
    s_id, _ = worst(obs, "safety")
    c_id, _ = worst(obs, "courtesy")
    fut = slice(step + 1, step + 1 + horizon)
    for i in range(scene.n_agents):
        if not scene.valid[i, step]:
            continue
        colour = AGENT_COLOR if i == agent else NEIGHBOUR_COLOR if i in neighbours else OTHER_COLOR
        hist = slice(max(0, step - 10), step + 1)
        h = scene.position[i, hist, :2][scene.valid[i, hist]]
        if len(h) > 1:
            ax.plot(*h.T, color=colour, lw=0.8, alpha=0.5, zorder=3)
        length, width = scene.shape_at(i, step)
        tid = scene.track_ids[i]
        edge, lw = "black", 0.5
        if tid == s_id:
            edge, lw = SAFETY_COLOR, 2.2
        if tid == c_id:
            edge, lw = (COURTESY_COLOR, 2.2) if tid != s_id else ("#7b2d8e", 2.6)
        ax.add_patch(Polygon(box(scene.position[i, step, :2], scene.heading[i, step], length, width), closed=True,
                             facecolor=colour, edgecolor=edge, lw=lw,
                             alpha=0.9 if i == agent or i in neighbours else 0.45, zorder=6))
        if i in neighbours:
            f = scene.position[i, fut, :2][scene.valid[i, fut]]
            if len(f) > 1:
                ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.3, ls="--", zorder=5)
            ax.annotate(tid, scene.position[i, step, :2], fontsize=6, color=NEIGHBOUR_COLOR,
                        xytext=(3, 3), textcoords="offset points", zorder=7)

    samples = frame["samples"]
    lp = frame["sample_log_prob"] if frame["sample_log_prob"] is not None else np.zeros(len(samples))
    norm = plt.Normalize(lp.min(), lp.max() if lp.max() > lp.min() else lp.min() + 1e-6)
    cmap = plt.get_cmap("plasma")
    for j in np.argsort(lp):
        ax.plot(*samples[j, horizon - 1:].T, color=cmap(norm(lp[j])), lw=0.6, alpha=0.25, zorder=4)
        ax.plot(*samples[j, :horizon].T, color=cmap(norm(lp[j])), lw=1.4, alpha=0.8, zorder=4)
    logged = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(logged) > 1:
        ax.plot(*logged.T, color="black", lw=1.6, ls="--", zorder=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    weighted = weighted and frame.get("metric_samples", True)  # display-only frames are always samples
    ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label(
        "log weight of trajectory" if weighted else "log p(goal) of sample", fontsize=7)
    label = "motion set (2 s solid)" if frame.get("metric_samples", True) else "motion set (display only)"
    handles = [
        plt.Line2D([], [], color=cmap(0.8), lw=1.5, label=label),
        plt.Line2D([], [], color="black", ls="--", lw=1.5, label="agent, logged"),
        plt.Line2D([], [], color=NEIGHBOUR_COLOR, ls="--", lw=1.3, label="neighbours, logged"),
        plt.Line2D([], [], color=SAFETY_COLOR, lw=2.2, label="beta_s from"),
        plt.Line2D([], [], color=COURTESY_COLOR, lw=2.2, label="beta_c toward"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_courtesy(axes, scene, agent, frame, map_groups, radius, horizon):
    obs, step = frame["observation"], frame["step"]
    c_id, value = worst(obs, "courtesy")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if c_id is None or c_id not in frame["courtesy"]:
        axes[0].text(0.5, 0.5, "no vehicle neighbour\n(no courtesy)", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=8)
        axes[1].axis("off")
        return
    b = scene.index(c_id)
    centre = scene.position[b, step, :2]
    r = min(radius, 60.0)
    fut = slice(step + 1, step + 1 + horizon)
    pair = frame["courtesy"][c_id]
    for ax, goals, title, present in ((axes[0], pair["with"], "with the agent", True),
                                      (axes[1], pair["without"], "without the agent", False)):
        for (colour, width), lines in map_groups.items():
            ax.add_collection(LineCollection(lines, colors=colour, linewidths=width * 0.8, zorder=1))
        # each panel on its own scale: a spread-out distribution would vanish next to a peaked one
        p = goals["prob"]
        keep = p > p.max() * 1e-3
        pts = goals["points"][keep]
        ax.scatter(pts[:, 0], pts[:, 1], c=np.log10(p[keep]), cmap="Blues", s=5, linewidths=0, zorder=2)
        length, width = scene.shape_at(b, step)
        ax.add_patch(Polygon(box(centre, scene.heading[b, step], length, width), closed=True,
                             facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5, zorder=4))
        f = scene.position[b, fut, :2][scene.valid[b, fut]]
        if len(f) > 1:
            ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.2, ls="--", zorder=3)
        a_len, a_wid = scene.shape_at(agent, step)
        ax.add_patch(Polygon(box(scene.position[agent, step, :2], scene.heading[agent, step], a_len, a_wid),
                             closed=True, facecolor=AGENT_COLOR if present else "none",
                             edgecolor="black" if present else AGENT_COLOR, ls="-" if present else "--",
                             lw=0.8, zorder=4))
        ax.set_xlim(centre[0] - r, centre[0] + r)
        ax.set_ylim(centre[1] - r, centre[1] + r)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfbfb")
        ax.set_title(title, fontsize=8)
    axes[0].text(1.0, 1.13, f"courtesy: goals of vehicle {c_id} (blue), KL = {value:.3f} nats",
                 transform=axes[0].transAxes, ha="center", fontsize=8.5)


def load_levels(path, scene_stem, agent_id):
    if path is None:
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["scene"] == scene_stem and r["agent_id"] == agent_id:
                out[int(r["step"])] = (int(r["level"]), int(r["aggressive"]))
    return out


def draw_timeline(ax, frames, n, levels):
    t = np.array([f["step"] / 10 for f in frames])
    s = np.array([f["observation"]["safety"] for f in frames])
    c = np.array([f["observation"]["courtesy"] for f in frames])
    title = "responsibility over the clip"
    if levels:
        n_levels = max(z for z, _ in levels.values()) + 1
        cmap = plt.get_cmap("viridis", max(n_levels, 2))
        half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.25
        for f in frames:
            if f["step"] in levels:
                z, aggressive = levels[f["step"]]
                ax.axvspan(f["step"] / 10 - half, f["step"] / 10 + half, color=cmap(z), alpha=0.18, lw=0,
                           hatch="///" if aggressive else None)
        seen = sorted({z: a for z, a in levels.values()}.items())
        ax.legend(handles=[Patch(facecolor=cmap(z), alpha=0.35, hatch="///" if a else None, label=f"level {z}")
                           for z, a in seen], fontsize=6.5, loc="upper right", ncol=len(seen))
        title += " (background: level; hatched: aggressive)"
    ax.axhline(0, color="#999999", lw=0.7)
    ax.plot(t, s, color=SAFETY_COLOR, marker="o", ms=3, lw=1.3)
    ax.plot(t[n], s[n], "o", color="black", ms=6)
    ax.set_ylabel(r"$\beta_s$ [m]", color=SAFETY_COLOR)
    ax.set_xlabel("context time t_k [s]")
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    tx = ax.twinx()
    tx.plot(t, c, color=COURTESY_COLOR, marker="s", ms=3, lw=1.3)
    tx.plot(t[n], c[n], "s", color="black", ms=6)
    tx.set_ylabel(r"$\beta_c$ [nats]", color=COURTESY_COLOR)
    tx.set_ylim(0, max(0.05, c.max() * 1.15))
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.3)


def render(scene, agent, frame, n, frames, map_groups, radius, horizon, levels, ego_heatmap, weighted=False):
    fig = plt.figure(figsize=(12, 8.2), dpi=100)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], height_ratios=[3, 1.2], hspace=0.28, wspace=0.08)
    ax_scene = fig.add_subplot(grid[0, 0])
    ax_with, ax_without = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    ax_time = fig.add_subplot(grid[1, :])
    draw_scene(ax_scene, scene, agent, frame, map_groups, radius, horizon, ego_heatmap, weighted)
    draw_courtesy((ax_with, ax_without), scene, agent, frame, map_groups, radius, horizon)
    draw_timeline(ax_time, frames, n, levels)
    obs = frame["observation"]
    s_id, _ = worst(obs, "safety")
    c_id, _ = worst(obs, "courtesy")
    level = levels.get(frame["step"])
    fig.suptitle(
        f"scene {scene.scenario_id}  agent {scene.track_ids[agent]}  t_k = {frame['step'] / 10:.1f} s  "
        f"speed {obs['speed']:.1f} m/s\n"
        f"safety {obs['safety']:+.3f} m" + (f" (vs {s_id})" if s_id else "")
        + f"    courtesy {obs['courtesy']:.4f} nats" + (f" (toward {c_id})" if c_id else "")
        + (f"    level {level[0]}{' - aggressive' if level[1] else ''}" if level else ""),
        fontsize=10)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_record(record, out_dir: Path, levels_csv=None, view_radius=None, ego_heatmap=False, fps=2.0):
    """Renders every frame of a record and writes the frames, a GIF and (with
    OpenCV) an MP4 to ``out_dir``. Returns the written paths."""
    scene = scene_of(record)
    agent = record["agent"]
    frames = record["frames"]
    if not frames:
        raise ValueError("the record has no frames")
    horizon = record["config"]["metric_horizon"]
    weighted = record["config"].get("motion_set", "sampled") != "sampled"  # colours are weights, not goal log p
    radius = view_radius or fit_radius(scene, agent, frames, horizon)
    levels = load_levels(levels_csv, record.get("scene_file", ""), record["agent_id"])
    groups = map_segments(scene)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    images = []
    for n, frame in enumerate(frames):
        image = render(scene, agent, frame, n, frames, groups, radius, horizon, levels, ego_heatmap, weighted)
        plt.imsave(out_dir / "frames" / f"t_{frame['step']:03d}.png", image)
        images.append(image)
    return write_video(images, out_dir, fps), radius


def write_video(images, out_dir: Path, fps: float):
    from PIL import Image

    gif = out_dir / "responsibility.gif"
    pil = [Image.fromarray(im) for im in images]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)
    try:
        import cv2
    except ImportError:
        return gif, None
    h, w = images[0].shape[:2]
    mp4 = out_dir / "responsibility.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return gif, None
    for im in images:
        writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    writer.release()
    return gif, mp4


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.record:
        record = load_record(args.record)
    else:
        record = live_record(args)
        save_record(record, out / "record.pkl")
    (gif, mp4), radius = render_record(record, out, args.levels, args.view_radius, args.ego_heatmap, args.fps)
    print(f"wrote {len(record['frames'])} frames to {out / 'frames'} (view half-width {radius:.0f} m), {gif}"
          + (f", {mp4}" if mp4 else " (no OpenCV: GIF only)")
          + ("" if args.record else f"; record: {out / 'record.pkl'}"))


if __name__ == "__main__":
    main()
