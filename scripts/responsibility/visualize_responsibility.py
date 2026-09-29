"""A video of an agent's counterfactual responsibility through one scene: one
frame per context step t_k, stitched into a GIF (and an MP4 if OpenCV is
installed).

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

The values are computed by the same code as compute_responsibility.py; with
--run the run's settings are used, so they match its windows.csv.

Example (from the repository root):
    python -m scripts.responsibility.visualize_responsibility --scene 17 \\
        --run logs/responsibility/sdc --levels logs/responsibility/levels/levels.csv \\
        --out-dir logs/responsibility/video_17
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
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from responsibility.interaction import InteractionConfig  # noqa: E402
from responsibility.metrics import ResponsibilityConfig, responsibility_at, window_steps  # noqa: E402
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
    p.add_argument("--scene", required=True, help="Scene file index (e.g. 17) or path to a .pkl.")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--agent", default="sdc", help="sdc, adv or a track id.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run", default=None, help="compute_responsibility.py output whose settings to use.")
    p.add_argument("--levels", default=None, help="levels.csv from fit_levels.py, for the timeline.")
    d = ResponsibilityConfig()
    p.add_argument("--stride", type=int, default=d.window_stride)
    p.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    p.add_argument("--horizon", type=int, default=d.metric_horizon)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--view-radius", type=float, default=None, help="m; default: fitted to the scene (30-90 m).")
    p.add_argument("--ego-heatmap", action="store_true", help="Also draw the agent's own goal distribution.")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def config_from(args) -> ResponsibilityConfig:
    if args.run is None:
        return ResponsibilityConfig(n_safety_samples=args.n_samples, metric_horizon=args.horizon,
                                    window_stride=args.stride, seed=args.seed)
    saved = json.loads((Path(args.run) / "config.json").read_text())["responsibility"]
    saved["interaction"] = InteractionConfig(**saved["interaction"])
    return ResponsibilityConfig(**saved)


def scene_path(args) -> Path:
    if args.scene.isdigit():
        return {p.stem: p for p in scene_files(args.scenes)}[args.scene]
    return Path(args.scene)


def pick_agent(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        return [i for i in scene.objects_of_interest if i != scene.sdc][0]
    return scene.index(which)


def map_segments(scene: Scene):
    """(segments, colour, width) per map feature type group."""
    groups = {}
    for feature in scene.map_features.values():
        kind = str(feature.get("type", ""))
        style = next((v for k, v in MAP_STYLE.items() if kind.startswith(k)), None)
        if style is None:
            continue
        pts = feature.get("polyline", feature.get("polygon"))
        if pts is None:
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


def goal_points(dist, top_mass=0.99):
    """The most probable goals covering ``top_mass`` of the distribution, in
    the scene frame, with their probabilities."""
    p = dist.log_prob.exp().cpu().numpy()
    order = np.argsort(-p)
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_mass)) + 1]
    return dist.to_global(dist.goals[keep]), p[keep]


def collect(model, scene, agent, cfg):
    generator = torch.Generator().manual_seed(cfg.seed)
    frames = []
    for step in window_steps(scene, agent, cfg):
        record = {}
        obs = responsibility_at(model, scene, agent, step, cfg, generator, record=record)
        if obs is None:
            print(f"t={step / 10:.1f}s: no prediction for the agent, skipped", flush=True)
            continue
        frame = {"step": step, "obs": obs, "record": record}
        if record.get("samples") is None:  # no neighbours: still show the motion set
            _, _, frame["samples"] = model.sample(record["distribution"], min(cfg.n_safety_samples, 24),
                                                  generator=generator)
        else:
            frame["samples"] = record["samples"]
        idx = np.random.default_rng(0).permutation(len(frame["samples"]))  # log-prob per sample via its goal
        frame["sample_order"] = idx
        frames.append(frame)
        print(f"t={step / 10:.1f}s: safety {obs.safety:+.3f} m, courtesy {obs.courtesy:.4f} nats, "
              f"{len(obs.per_neighbour)} neighbour(s)", flush=True)
    return frames


def worst(obs, key):
    items = [(k, v[key]) for k, v in obs.per_neighbour.items() if v.get(key) is not None]
    return max(items, key=lambda kv: kv[1], default=(None, None))


def fit_radius(scene, agent, frames, horizon):
    need = 25.0
    for f in frames:
        centre = scene.position[agent, f["step"], :2]
        pts = [f["samples"][:, :horizon].reshape(-1, 2)]
        pts += [scene.position[scene.index(b), f["step"], :2][None] for b in f["obs"].per_neighbour]
        need = max(need, float(np.abs(np.concatenate(pts) - centre).max()) + 8.0)
    return float(np.clip(need, 30.0, 90.0))


def draw_scene(ax, scene, agent, frame, map_groups, radius, horizon, ego_heatmap):
    step, obs, record = frame["step"], frame["obs"], frame["record"]
    centre = scene.position[agent, step, :2]
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    for (colour, width), lines in map_groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))

    if ego_heatmap:
        pts, p = goal_points(record["distribution"])
        ax.scatter(pts[:, 0], pts[:, 1], c=np.log(p), cmap="Greys", s=4, alpha=0.5, zorder=2, linewidths=0)

    neighbours = {scene.index(b) for b in obs.per_neighbour}
    s_id, _ = worst(obs, "safety")
    c_id, _ = worst(obs, "courtesy")
    fut = slice(step + 1, step + 1 + horizon)
    for i in range(scene.n_agents):
        if not scene.valid[i, step]:
            continue
        role_colour = AGENT_COLOR if i == agent else NEIGHBOUR_COLOR if i in neighbours else OTHER_COLOR
        hist = slice(max(0, step - 10), step + 1)
        h = scene.position[i, hist, :2][scene.valid[i, hist]]
        if len(h) > 1:
            ax.plot(*h.T, color=role_colour, lw=0.8, alpha=0.5, zorder=3)
        length, width = scene.shape_at(i, step)
        edge = "black"
        lw = 0.5
        tid = scene.track_ids[i]
        if tid == s_id:
            edge, lw = SAFETY_COLOR, 2.2
        if tid == c_id:
            edge, lw = (COURTESY_COLOR, 2.2) if tid != s_id else ("#7b2d8e", 2.6)
        ax.add_patch(Polygon(box(scene.position[i, step, :2], scene.heading[i, step], length, width), closed=True,
                             facecolor=role_colour, edgecolor=edge, lw=lw,
                             alpha=0.9 if i == agent or i in neighbours else 0.45, zorder=6))
        if i in neighbours:
            f = scene.position[i, fut, :2][scene.valid[i, fut]]
            if len(f) > 1:
                ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.3, ls="--", zorder=5)
            ax.annotate(tid, scene.position[i, step, :2], fontsize=6, color=NEIGHBOUR_COLOR,
                        xytext=(3, 3), textcoords="offset points", zorder=7)

    samples = frame["samples"]
    dist = record["distribution"]
    # colour each sample by the probability of the goal it was completed from
    ends = samples[:, -1]
    goals = dist.goals_global
    nearest = np.argmin(((ends[:, None] - goals[None]) ** 2).sum(-1), axis=1)
    lp = dist.log_prob.cpu().numpy()[nearest]
    norm = plt.Normalize(lp.min(), lp.max() if lp.max() > lp.min() else lp.min() + 1e-6)
    cmap = plt.get_cmap("plasma")
    for j in np.argsort(lp):
        ax.plot(*samples[j, horizon - 1:].T, color=cmap(norm(lp[j])), lw=0.6, alpha=0.25, zorder=4)
        ax.plot(*samples[j, :horizon].T, color=cmap(norm(lp[j])), lw=1.4, alpha=0.8, zorder=4)
    logged = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(logged) > 1:
        ax.plot(*logged.T, color="black", lw=1.6, ls="--", zorder=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label("log p(goal) of sample", fontsize=7)

    handles = [
        plt.Line2D([], [], color=cmap(0.8), lw=1.5, label="motion set (2 s solid)"),
        plt.Line2D([], [], color="black", ls="--", lw=1.5, label="agent, logged"),
        plt.Line2D([], [], color=NEIGHBOUR_COLOR, ls="--", lw=1.3, label="neighbours, logged"),
        plt.Line2D([], [], color=SAFETY_COLOR, lw=2.2, label="beta_s from"),
        plt.Line2D([], [], color=COURTESY_COLOR, lw=2.2, label="beta_c toward"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_courtesy(axes, scene, agent, frame, map_groups, radius, horizon):
    obs, record = frame["obs"], frame["record"]
    c_id, value = worst(obs, "courtesy")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if c_id is None or scene.index(c_id) not in record["courtesy"]:
        axes[0].text(0.5, 0.5, "no vehicle neighbour\n(no courtesy)", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=8)
        axes[1].axis("off")
        return
    b = scene.index(c_id)
    step = frame["step"]
    with_a, without_a = record["courtesy"][b]
    centre = scene.position[b, step, :2]
    r = min(radius, 60.0)
    fut = slice(step + 1, step + 1 + horizon)
    for ax, dist, title, agent_present in ((axes[0], with_a, "with the agent", True),
                                          (axes[1], without_a, "without the agent", False)):
        for (colour, width), lines in map_groups.items():
            ax.add_collection(LineCollection(lines, colors=colour, linewidths=width * 0.8, zorder=1))
        # each panel on its own scale: a spread-out distribution would vanish next to a peaked one
        p = dist.log_prob.exp().cpu().numpy()
        keep = p > p.max() * 1e-3
        pts = dist.to_global(dist.goals[keep])
        ax.scatter(pts[:, 0], pts[:, 1], c=np.log10(p[keep]), cmap="Blues", s=5, linewidths=0, zorder=2)
        length, width = scene.shape_at(b, step)
        ax.add_patch(Polygon(box(centre, scene.heading[b, step], length, width), closed=True,
                             facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5, zorder=4))
        f = scene.position[b, fut, :2][scene.valid[b, fut]]
        if len(f) > 1:
            ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.2, ls="--", zorder=3)
        a_len, a_wid = scene.shape_at(agent, step)
        ax.add_patch(Polygon(box(scene.position[agent, step, :2], scene.heading[agent, step], a_len, a_wid),
                             closed=True, facecolor=AGENT_COLOR if agent_present else "none",
                             edgecolor=AGENT_COLOR if not agent_present else "black",
                             ls="-" if agent_present else "--", lw=0.8, zorder=4))
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
    s = np.array([f["obs"].safety for f in frames])
    c = np.array([f["obs"].courtesy for f in frames])
    if levels:
        n_levels = max(z for z, _ in levels.values()) + 1
        cmap = plt.get_cmap("viridis", max(n_levels, 2))
        half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.25
        for f in frames:
            if f["step"] in levels:
                z, aggressive = levels[f["step"]]
                ax.axvspan(f["step"] / 10 - half, f["step"] / 10 + half, color=cmap(z), alpha=0.18, lw=0,
                           hatch="///" if aggressive else None)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.plot(t, s, color=SAFETY_COLOR, marker="o", ms=3, lw=1.3, label=r"$\beta_s$ (m)")
    ax.plot(t[n], s[n], "o", color="black", ms=6)
    ax.set_ylabel(r"$\beta_s$ [m]", color=SAFETY_COLOR)
    ax.set_xlabel("context time t_k [s]")
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    tx = ax.twinx()
    tx.plot(t, c, color=COURTESY_COLOR, marker="s", ms=3, lw=1.3)
    tx.plot(t[n], c[n], "s", color="black", ms=6)
    tx.set_ylabel(r"$\beta_c$ [nats]", color=COURTESY_COLOR)
    tx.set_ylim(0, max(0.05, c.max() * 1.15))
    title = "responsibility over the clip"
    if levels:
        title += " (background: level; hatched: aggressive)"
        from matplotlib.patches import Patch

        seen = sorted({z: a for z, a in levels.values()}.items())
        ax.legend(handles=[Patch(facecolor=cmap(z), alpha=0.35, hatch="///" if a else None, label=f"level {z}")
                           for z, a in seen], fontsize=6.5, loc="upper right", ncol=len(seen))
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.3)


def render(scene, agent, frame, n, frames, map_groups, radius, horizon, levels, ego_heatmap):
    fig = plt.figure(figsize=(12, 8.2), dpi=100)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], height_ratios=[3, 1.2], hspace=0.28, wspace=0.08)
    ax_scene = fig.add_subplot(grid[0, 0])
    ax_with, ax_without = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    ax_time = fig.add_subplot(grid[1, :])
    draw_scene(ax_scene, scene, agent, frame, map_groups, radius, horizon, ego_heatmap)
    draw_courtesy((ax_with, ax_without), scene, agent, frame, map_groups, radius, horizon)
    draw_timeline(ax_time, frames, n, levels)
    obs = frame["obs"]
    s_id, s_val = worst(obs, "safety")
    c_id, c_val = worst(obs, "courtesy")
    level = levels.get(frame["step"])
    fig.suptitle(
        f"scene {scene.scenario_id}  agent {scene.track_ids[agent]}  t_k = {frame['step'] / 10:.1f} s  "
        f"speed {obs.speed:.1f} m/s\n"
        f"safety {obs.safety:+.3f} m" + (f" (vs {s_id})" if s_id else "")
        + f"    courtesy {obs.courtesy:.4f} nats" + (f" (toward {c_id})" if c_id else "")
        + (f"    level {level[0]}{' - aggressive' if level[1] else ''}" if level else ""),
        fontsize=10)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


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
    from responsibility.densetnt import DenseTNT

    cfg = config_from(args)
    path = scene_path(args)
    scene = Scene.load(path)
    agent = pick_agent(scene, args.agent)
    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    model = DenseTNT(device=args.device)
    frames = collect(model, scene, agent, cfg)
    if not frames:
        raise SystemExit("no windows to show")
    radius = args.view_radius or fit_radius(scene, agent, frames, cfg.metric_horizon)
    levels = load_levels(args.levels, path.stem, scene.track_ids[agent])
    groups = map_segments(scene)
    images = []
    for n, frame in enumerate(frames):
        image = render(scene, agent, frame, n, frames, groups, radius, cfg.metric_horizon, levels, args.ego_heatmap)
        plt.imsave(out / "frames" / f"t_{frame['step']:03d}.png", image)
        images.append(image)
    gif, mp4 = write_video(images, out, args.fps)
    print(f"wrote {len(images)} frames to {out / 'frames'} (view half-width {radius:.0f} m), {gif}"
          + (f", {mp4}" if mp4 else " (no OpenCV: GIF only)"))


if __name__ == "__main__":
    main()
