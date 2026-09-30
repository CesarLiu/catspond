"""Collision penalties weighted by responsibility, for CAT's RL training
(cat_RLtrain.py --blame_weighting share).

In CAT's training environment a collision does not end the episode: at
every step the ego touches another vehicle, MetaDrive replaces the step's
driving reward with -crash_vehicle_penalty (ScenarioEnv.reward_function;
only CAT's evaluation sets crash_vehicle_done). A collision the ego could
not have prevented -- an adversary driving into it -- then teaches it to
avoid whatever led there, often just driving on. Here every collision is
attributed (responsibility/blame.py) and the ego keeps only its share of
the penalty:

  event    a run of consecutive steps in contact with the same vehicle. It
           is attributed once, at its first step c, in the scene rebuilt
           from the episode up to c: the ego (and the adversary) as
           simulated, the other objects as spawned
           (recording.Recorder.crash_rollout, rollouts.scene_from_rollout)
  weight   w = the ego's share beta_ego+ / (beta_ego+ + beta_other+) when
           the attribution tells the two sides apart (verdict "ego" or
           "other": they differ by more than ``margin`` metres), otherwise
           w = 1, the full penalty: no attribution (collision after the end
           of the log, partner not predicted, no window with both), or one
           too close to call. Doubt keeps the penalty, so that noise in the
           attribution does not become noise in the reward
  reward   at every penalised step of the event (in contact, and the reward
           is the crash penalty: arriving and leaving the road take
           precedence in MetaDrive's reward)
               r' = w * (-penalty) + (1 - w) * step_reward
           where step_reward is the driving reward the penalty replaced
           (info["step_reward"]). w = 1 is MetaDrive's reward; w = 0 the
           step as if nothing had been hit

An episode's transitions are held back and released by ``end`` with the
weighted rewards, so the replay buffer never holds an unweighted collision.
Every attributed event is appended to ``log_path`` (CSV, LOG_FIELDS).
"""

import csv
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from responsibility.blame import MARGIN, Blame, rollout_blame
from responsibility.metrics import ResponsibilityConfig
from responsibility.recording import Recorder, SceneIndex, current_scenario_id
from responsibility.rollouts import scene_from_rollout
from responsibility.scene import Scene

SIDES = ("ego", "other")
LOG_FIELDS = ["episode", "total_steps", "scene", "scenario_index", "adversary", "crash_step", "penalised_steps",
              "other_id", "other_type", "window", "beta_ego", "beta_other", "share", "verdict", "rule", "weight",
              "seconds"]


@dataclass
class Event:
    step: int  # recorded state (scene step) of the first contact
    other: Optional[str]  # the logged object in contact (closest to the ego)
    transitions: List[int] = field(default_factory=list)  # the episode's penalised transitions


def contact_events(contacts: Sequence[bool], partners: Sequence[Optional[str]],
                   penalised: Sequence[bool]) -> List[Event]:
    """Contact runs of an episode. Entry i describes the state after
    transition i (recorded state i + 1): in contact or not, with whom, and
    whether transition i's reward is the crash penalty."""
    events: List[Event] = []
    current = None
    for i, (contact, other, pen) in enumerate(zip(contacts, partners, penalised)):
        if not contact:
            current = None
            continue
        if current is None or other != current.other:
            current = Event(i + 1, other)
            events.append(current)
        if pen:
            current.transitions.append(i)
    return events


def penalty_weight(blame: Optional[Blame]) -> float:
    """The share of the collision penalty the ego keeps (module docstring)."""
    if blame is None or blame.share is None or blame.verdict not in SIDES:
        return 1.0
    return float(blame.share)


def weighted_reward(reward: float, step_reward: float, weight: float, penalty: float) -> float:
    return weight * -penalty + (1.0 - weight) * step_reward


class BlameWeighting:
    """Weights the collision penalties of CAT's training episodes (module
    docstring). Per episode: ``begin`` after the reset (and set_adv_info),
    ``add`` for every transition instead of the replay buffer, ``end`` when
    it is done, returning the transitions to store."""

    def __init__(self, model, scenes, penalty: float, cfg: Optional[ResponsibilityConfig] = None,
                 margin: float = MARGIN, log_path=None, cache: int = 8, verbose: bool = True):
        self.model = model
        self.scenes = Path(scenes)
        self.index = SceneIndex(scenes)
        self.penalty = float(penalty)
        self.cfg = cfg or ResponsibilityConfig()
        self.margin = margin
        self.log_path = None if log_path is None else Path(log_path)
        self.cache: "OrderedDict[str, Scene]" = OrderedDict()
        self.cache_size = cache
        self.verbose = verbose
        self.episodes = 0
        self.weights: List[float] = []  # of every attributed event so far
        self.env = None

    def begin(self, env, adversary=None) -> None:
        self.env = env
        self.adversary = None if adversary is None else str(adversary)
        self.recorder = Recorder(env, self.adversary)
        self.recorder.record()
        self.transitions: List[list] = []
        self.contacts: List[bool] = []
        self.partners: List[Optional[str]] = []
        self.penalised: List[bool] = []
        self.step_rewards: List[float] = []

    def add(self, state, action, next_state, reward, done, info) -> None:
        self.recorder.record()
        contact = bool(self.env.vehicle.crash_vehicle)
        self.contacts.append(contact)
        self.partners.append(self.recorder.nearest() if contact else None)
        self.penalised.append(contact and bool(np.isclose(reward, -self.penalty)))
        self.step_rewards.append(float(info.get("step_reward", 0.0)))
        self.transitions.append([state, action, next_state, reward, done])

    def _scene(self, stem: str) -> Scene:
        if stem in self.cache:
            self.cache.move_to_end(stem)
            return self.cache[stem]
        scene = Scene.load(self.scenes / f"{stem}.pkl")
        self.cache[stem] = scene
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return scene

    def attribute(self, event: Event, stem: str, scenario_id: str) -> Optional[Blame]:
        rollout = self.recorder.crash_rollout(event.step, event.other, stem, scenario_id)
        scene = scene_from_rollout(self._scene(stem), rollout)
        return rollout_blame(self.model, scene, rollout, self.cfg, margin=self.margin)

    def end(self, total_steps: int = 0) -> List[Tuple]:
        """The episode's transitions, collision penalties weighted."""
        self.episodes += 1
        events = [e for e in contact_events(self.contacts, self.partners, self.penalised) if e.transitions]
        if events:
            seed = int(self.env.current_seed)
            scenario_id = current_scenario_id(self.env)
            stem = self.index.stem(seed, scenario_id)
            for event in events:
                t0 = time.time()
                try:
                    blame = self.attribute(event, stem, scenario_id)
                    error = None
                except Exception as e:  # a long training run must not stop on one attribution
                    blame, error = None, e
                w = penalty_weight(blame)
                self.weights.append(w)
                for i in event.transitions:
                    self.transitions[i][3] = weighted_reward(self.transitions[i][3], self.step_rewards[i], w,
                                                             self.penalty)
                row = self._log(event, blame, w, stem, seed, total_steps, time.time() - t0, error)
                if self.verbose:
                    print(f"collision at step {event.step} with {event.other}: {row['verdict']} "
                          f"(beta ego {row['beta_ego']}, other {row['beta_other']}), penalty weight {w:.2f} "
                          f"over {len(event.transitions)} steps ({row['seconds']} s)", flush=True)
        out = [tuple(t) for t in self.transitions]
        self.transitions = []
        return out

    def _log(self, event, blame, w, stem, seed, total_steps, seconds, error) -> dict:
        row = {"episode": self.episodes, "total_steps": total_steps, "scene": stem, "scenario_index": seed,
               "adversary": self.adversary or "", "crash_step": event.step,
               "penalised_steps": len(event.transitions), "other_id": event.other, "other_type": "",
               "window": "", "beta_ego": "", "beta_other": "", "share": "",
               "verdict": "error" if error is not None else "unknown", "rule": "n/a", "weight": round(w, 4),
               "seconds": round(seconds, 2)}
        if blame is not None:
            b = blame.as_row()
            row.update({k: b[k] for k in ("other_id", "other_type", "window", "verdict", "rule")})
            row.update({k: "" if b[k] is None else round(b[k], 4) for k in ("beta_ego", "beta_other", "share")})
        if error is not None:
            print(f"warning: collision at step {event.step} of scene {stem} not attributed "
                  f"({type(error).__name__}: {error}); full penalty kept", flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists()
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
        return row

    def summary(self) -> str:
        if not self.weights:
            return "no collisions attributed yet"
        w = np.array(self.weights)
        return (f"{len(w)} collisions attributed: mean penalty weight {w.mean():.2f}, "
                f"{100 * np.mean(w < 0.5):.0f}% mostly the other's fault, {100 * np.mean(w == 1.0):.0f}% full penalty")
