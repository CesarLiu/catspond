"""Pictures of make_go_first.py's scene sets, to check by eye what
forced_yield.py measures: one figure per scene and set, and an index.md
gallery that lists them by the forced demand.

Each figure has three panels:
  log        the source scene: the map, the pair's tracks (agent blue, other
             orange; a dot every second, labelled with the time), the
             conflict point P* and each agent's conflict zone (thick), the
             two footprints at the agent's worst entry into its zone, and
             every other agent there in grey
  rewritten  the same for the rewritten scene (the set's)
  time       each agent's distance along its path to P* (m) over time, the
             zone shaded; the log dashed, the rewritten solid; the agent's
             worst entry marked, with the braking it demanded of the other
             and the PET (s) between the agent leaving its zone and the
             other entering its own

For the log sets (log, log_yielder_first) there is no rewrite and the
figure shows the scene once. A rewritten scene is judged with its log's
right of way, as evaluate_forced_yield.py does; the title gives its own
reading too. With --gif each rewritten scene also gets an
animation, a frame every --gif-stride steps.

Example:
    python -m scripts.responsibility.visualize_forced_yield \\
        --scenes logs/responsibility/forced_yield/validation_interactive/scenes \\
        --sets delay_pet2 delay_pet1 retime_pet1 log_yielder_first --limit 20 \\
        --out-dir logs/responsibility/forced_yield/validation_interactive/figures
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from responsibility.forced_yield import carry_priority, forced_yield, pair_priority  # noqa: E402
from responsibility.right_of_way import RightOfWayParams, _Path, _signals, _window  # noqa: E402
from responsibility.scene import Scene, sidecar_index  # noqa: E402
from scripts.responsibility.visualize_responsibility import box, map_segments  # noqa: E402

DT = 0.1  # s per step
AGENT, OTHER, GREY = "#1f77b4", "#ff7f0e", "#9a9a9a"
LOG_SETS = ("log", "log_yielder_first")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True, help="make_go_first.py's --out-dir.")
    p.add_argument("--sets", nargs="+", default=None, help="Sets to draw (default: all but log).")
    p.add_argument("--scene", nargs="+", default=None, help="Only these scenario ids.")
    p.add_argument("--limit", type=int, default=None, help="At most this many scenes per set.")
    p.add_argument("--radius", type=float, default=40.0, help="m around P* shown on the maps.")
    p.add_argument("--gif", action="store_true", help="Also animate each rewritten scene.")
    p.add_argument("--gif-stride", type=int, default=2)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def geometry(scene: Scene, agent: int, other: int, prio=None):
    """What the figures show of one pair: the forced-yield result, P*, and
    per agent its steps, distance along its path to P* (m), zone half
    length h (m) and zone polyline (None without a conflict). ``prio``:
    the right of way to judge with, by default the scene's own."""
    p = RightOfWayParams()
    prio = prio if prio is not None else pair_priority(scene, agent, other, params=p)
    fy = forced_yield(scene, agent, other, params=p, prio=prio)
    if prio is None or prio.conflict is None:
        return fy, None
    c = prio.conflict
    window = _window(scene, agent, other, scene.n_steps - 1)
    out = {"point": c.point, "kind": c.kind}
    for x, o, s_star in ((agent, other, c.s[0]), (other, agent, c.s[1])):
        path = _Path(scene, x, scene.n_steps - 1, 0.0)
        s = path.s[:path.n_real] - s_star
        h = np.nan
        if window.size:
            sig_path = _Path(scene, x, int(window[-1]), p.extend)
            in_zone, _, _ = _signals(scene, x, o, sig_path, s_star, c.angle, window, p)
            h = float(in_zone[0] + abs(sig_path.s_at(window[:1])[0] - s_star))
        zone = path.points[:path.n_real][np.abs(s) <= h] if np.isfinite(h) else np.zeros((0, 2))
        out[x] = {"steps": path.steps, "s": s, "h": h, "zone": zone}
    return fy, out


def pet(geo, agent: int, other: int):
    """s between the first of the two leaving its zone and the other
    entering its own (positive: the agent went first)."""
    if geo is None:
        return None
    a, o = geo[agent], geo[other]
    a_out = a["steps"][a["s"] > a["h"]]
    o_in = o["steps"][o["s"] >= -o["h"]]
    o_out = o["steps"][o["s"] > o["h"]]
    a_in = a["steps"][a["s"] >= -a["h"]]
    if a_out.size and o_in.size and a_out[0] <= o_in[0]:
        return (o_in[0] - a_out[0]) * DT
    if o_out.size and a_in.size and o_out[0] <= a_in[0]:
        return -(a_in[0] - o_out[0]) * DT
    return None


def fmt(x):
    return "–" if x is None else "inf" if np.isinf(x) else f"{x:.2f}"


def draw_map(ax, scene, agent, other, geo, fy, step, radius, title, groups):
    centre = geo["point"] if geo is not None else scene.position[agent, step, :2]
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    for (colour, width), lines in groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))
    for i in range(scene.n_agents):
        if i in (agent, other) or not scene.valid[i, step]:
            continue
        length, width = scene.shape_at(i, step)
        ax.add_patch(Polygon(box(scene.position[i, step, :2], scene.heading[i, step], length, width),
                             closed=True, facecolor=GREY, edgecolor="none", alpha=0.45, zorder=3))
    for x, colour, name in ((agent, AGENT, "agent"), (other, OTHER, "other")):
        seen = scene.valid[x]
        pts = scene.position[x, seen, :2]
        ax.plot(*pts.T, color=colour, lw=1.2, alpha=0.8, zorder=4)
        steps = np.flatnonzero(seen)
        for k in steps[steps % 10 == 0]:
            ax.plot(*scene.position[x, k, :2], "o", ms=3, color=colour, zorder=5)
            ax.annotate(f"{k * DT:.0f}", scene.position[x, k, :2], fontsize=6, color=colour,
                        xytext=(2, 2), textcoords="offset points", zorder=5)
        if geo is not None and len(geo[x]["zone"]) > 1:
            ax.plot(*geo[x]["zone"].T, color=colour, lw=7, alpha=0.25, solid_capstyle="butt", zorder=2)
        if seen[step]:
            length, width = scene.shape_at(x, step)
            ax.add_patch(Polygon(box(scene.position[x, step, :2], scene.heading[x, step], length, width),
                                 closed=True, facecolor=colour, edgecolor="black", lw=0.8, alpha=0.9, zorder=6))
            ax.annotate(name, scene.position[x, step, :2], fontsize=7, color="black", xytext=(4, -9),
                        textcoords="offset points", zorder=7)
    if geo is not None:
        ax.plot(*geo["point"], marker="*", ms=12, color="crimson", zorder=8)
    ax.set_title(f"{title}: t = {step * DT:.1f} s (agent's worst entry)" if fy.entry is not None
                 else f"{title}: t = {step * DT:.1f} s", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_time(ax, curves, agent, other, fy, cursor=None):
    """curves: [(label, geo, linestyle)]."""
    for label, geo, ls in curves:
        if geo is None:
            continue
        for x, colour, name in ((agent, AGENT, "agent"), (other, OTHER, "other")):
            g = geo[x]
            ax.plot(g["steps"] * DT, g["s"], color=colour, ls=ls, lw=1.5, label=f"{name}, {label}")
    geo = next((g for _, g, _ in reversed(curves) if g is not None), None)
    if geo is not None:
        for x, colour in ((agent, AGENT), (other, OTHER)):
            if np.isfinite(geo[x]["h"]):
                ax.axhspan(-geo[x]["h"], geo[x]["h"], color=colour, alpha=0.08)
    ax.axhline(0, color="crimson", lw=0.8)
    if fy.entry is not None:
        ax.axvline(fy.entry * DT, color="black", lw=0.8, ls=":")
        ax.annotate(f"agent enters: other must brake {fmt(fy.demand)} m/s²", (fy.entry * DT, 0),
                    fontsize=7, xytext=(4, 12), textcoords="offset points")
    if cursor is not None:
        ax.axvline(cursor * DT, color="grey", lw=1.0)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(-60, 40)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance past P* along own path (m)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3)


def figure(name, sid, meta, scene, log, out: Path, radius):
    agent, other = scene.index(meta["agent"]), scene.index(meta["other"])
    rewritten = name not in LOG_SETS and log is not None
    own = pair_priority(scene, agent, other)
    if rewritten:
        prio_log = pair_priority(log, agent, other)
        fy_log, geo_log = geometry(log, agent, other, prio_log)
        fy, geo = geometry(scene, agent, other, carry_priority(prio_log, scene, agent, other))
    else:
        fy, geo = geometry(scene, agent, other, own)
    panels = 3 if rewritten else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.2 * panels, 6.2))
    step = fy.entry if fy.entry is not None else int(np.flatnonzero(scene.valid[agent])[0])
    if rewritten:
        log_step = fy_log.entry if fy_log.entry is not None else step
        draw_map(axes[0], log, agent, other, geo_log, fy_log, log_step, radius, "log", map_segments(log))
    draw_map(axes[-2], scene, agent, other, geo, fy, step, radius, name, map_segments(scene))
    curves = ([("log", geo_log, "--")] if rewritten else []) + [(name, geo, "-")]
    draw_time(axes[-1], curves, agent, other, fy)
    rewrite = ", ".join(f"{k} {meta[k]}" for k in ("delay", "accel", "t0") if k in meta)
    forced = f"forced {fmt(fy_log.forced)} → {fmt(fy.forced)} m/s²" if rewritten else f"forced {fmt(fy.forced)} m/s²"
    p = pet(geo, agent, other)
    own_reading = ""
    if rewritten and own is not None:
        holder = "none" if own.holder is None else "agent" if own.holder == agent else "other"
        own_reading = f" · own reading: {own.case}, holder {holder}"
    fig.suptitle(f"{name} · {sid} · rule {fy.case} ({fy.kind}), holder: {fy.holder} · {forced} · "
                 f"PET {fmt(p)} s" + (f" · {rewrite}" if rewrite else "") + own_reading, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return fy, p


def animation(name, meta, scene, log, out: Path, radius, stride):
    from PIL import Image

    agent, other = scene.index(meta["agent"]), scene.index(meta["other"])
    prio = carry_priority(pair_priority(log, agent, other), scene, agent, other) if log is not None else None
    fy, geo = geometry(scene, agent, other, prio)
    groups = map_segments(scene)
    frames = []
    for step in range(0, scene.n_steps, stride):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_map(axes[0], scene, agent, other, geo, fy, step, radius, name, groups)
        axes[0].set_title(f"{name}: t = {step * DT:.1f} s", fontsize=9)
        draw_time(axes[1], [(name, geo, "-")], agent, other, fy, cursor=step)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * DT * stride), loop=0)


def main(argv=None):
    args = parse_args(argv)
    root, out = Path(args.scenes), Path(args.out_dir)
    sets = args.sets or sorted(p.name[:-len(".index.json")] for p in root.glob("*.index.json")
                               if p.name != "log.index.json")
    lines = ["# Forced yields: figures", "", f"Scenes: `{root}`. Sorted by the forced demand (m/s²), largest first.", ""]
    for name in sets:
        if not sidecar_index(root / name).exists():
            print(f"{name}: no such set, skipped")
            continue
        index = json.loads(sidecar_index(root / name).read_text())
        keys = [k for k in sorted(index) if args.scene is None or k in args.scene][:args.limit]
        (out / name).mkdir(parents=True, exist_ok=True)
        rows = []
        for sid in keys:
            scene = Scene.load(root / name / f"{sid}.pkl")
            log_path = root / "log" / f"{sid}.pkl"
            log = Scene.load(log_path) if log_path.exists() else None
            fy, p = figure(name, sid, index[sid], scene, log, out / name / f"{sid}.png", args.radius)
            if args.gif and name not in LOG_SETS:
                animation(name, index[sid], scene, log, out / name / f"{sid}.gif", args.radius, args.gif_stride)
            rows.append((sid, fy, p))
            print(f"{name}/{sid}: {fy.case}, holder {fy.holder}, forced {fmt(fy.forced)}, PET {fmt(p)}", flush=True)
        rows.sort(key=lambda r: -1.0 if r[1].forced is None else r[1].forced, reverse=True)
        lines += [f"## {name} ({len(rows)} scenes)", "", "| scene | rule | holder | forced (m/s²) | PET (s) |",
                  "|---|---|---|---|---|"]
        lines += [f"| [{sid}]({name}/{sid}.png) | {fy.case} | {fy.holder} | {fmt(fy.forced)} | {fmt(p)} |"
                  for sid, fy, p in rows]
        lines += [""] + [f"### {sid}\n\n![{sid}]({name}/{sid}.png)\n" for sid, _, _ in rows]
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'index.md'}")


if __name__ == "__main__":
    main()
