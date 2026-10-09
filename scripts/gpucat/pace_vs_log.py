"""How fast trained MetaDrive policies drive compared with the logged ego,
from the rollouts in rollouts_eval/<policy>/<adv_mode>/ (test scenes
400-499). Per run: the speed ratio (distance the ego covers over the
distance the logged ego covers in the same steps), the share of steps
more than 2 m/s slower than the log, the share of needless stops (ego
below 0.5 m/s while the log moves above 3 m/s), and the share of steps the
ego is more than 5 m behind or ahead of the logged ego along its path.

    python scripts/gpucat/pace_vs_log.py none
"""
import pickle, glob, sys, numpy as np
sys.path.insert(0, "/home/Cesar/catspond")
from responsibility.scene import Scene
from responsibility.rollouts import load_rollout

scenes = {}
def logged(i):
    if i not in scenes:
        sc = Scene.from_description(pickle.load(open(f"raw_scenes_500/{i}.pkl", "rb")))
        v = sc.valid[sc.sdc]
        scenes[i] = (sc.position[sc.sdc, :, :2].astype(float), v)
    return scenes[i]

def arc(path, pts):
    seg = np.diff(path, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    a = path[:-1]
    t = np.clip(((pts[:, None] - a) * seg).sum(-1) / np.maximum(L ** 2, 1e-9), 0, 1)
    d = np.linalg.norm(pts[:, None] - (a + t[..., None] * seg), axis=-1)
    k = d.argmin(1)
    return cum[k] + t[np.arange(len(pts)), k] * L[k]

groups = {"replay(TD3)": "td3_replay_s", "cat": "td3_cat_s", "fair2_0.1": "td3_cat_fair2_0.1_s",
          "cat_share": "td3_cat_share_s", "cat_rss": "td3_cat_rss_s"}
mode = sys.argv[1] if len(sys.argv) > 1 else "none"
print(f"test adversary: {mode}")
print(f"{'group':12s} {'seed':>4s} {'speed ratio':>11s} {'slower>2m/s':>11s} {'needless stop':>13s} {'>5m behind':>10s} {'>5m ahead':>9s} {'ego speed':>9s} {'log speed':>9s}")
for g, pre in groups.items():
    for s in range(3):
        files = sorted(glob.glob(f"rollouts_eval/{pre}{s}/{mode}/*.pkl"))
        if not files:
            continue
        de = dl = 0.0; slow = stop = behind = ahead = n = 0; ve_all = []; vl_all = []
        for f in files:
            r = load_rollout(f)
            i = int(r["scene_file"].split("/")[-1].split(".")[0]) if r.get("scene_file") else int(f.split("/")[-1][:-4])
            lp, lv = logged(i)
            ep = np.asarray(r["ego"]["position"], float)
            T = min(len(ep), len(lp))
            ok = ~np.isnan(ep[:T, 0]) & lv[:T]
            ok[0] = False
            idx = np.flatnonzero(ok[1:] & ok[:-1]) + 1
            if len(idx) < 5:
                continue
            ve = np.linalg.norm(ep[idx] - ep[idx - 1], axis=1) / 0.1
            vl = np.linalg.norm(lp[idx] - lp[idx - 1], axis=1) / 0.1
            de += ve.sum(); dl += vl.sum()
            slow += (ve < vl - 2).sum(); stop += ((ve < 0.5) & (vl > 3)).sum(); n += len(idx)
            path = lp[lv]
            se, sl = arc(path, ep[idx]), arc(path, lp[idx])
            behind += (se < sl - 5).sum(); ahead += (se > sl + 5).sum()
            ve_all.append(ve.mean()); vl_all.append(vl.mean())
        print(f"{g:12s} {s:>4d} {de / dl:>11.2f} {slow / n:>11.2f} {stop / n:>13.3f} {behind / n:>10.2f} {ahead / n:>9.2f} {np.mean(ve_all):>9.1f} {np.mean(vl_all):>9.1f}")
