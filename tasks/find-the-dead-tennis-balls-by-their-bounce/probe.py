"""Physics check for the dead-tennis-ball task (not used by the task or its demo): does a ball come back as high as its
coefficient of restitution says?

    python probe.py task.py [readings per second, default 120]

A good and a flat ball are placed in the air at rest, their bottoms 15, 25, 35 and 45 cm above the table, and left to
fall. Their heights are read after every control step and the top of the first bounce is compared with
bounce squared times the drop. Placing a ball in the air directly is something only this check does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DROPS = (0.15, 0.25, 0.35, 0.45)
WATCH_S = 3.0


def first_bounce(z: np.ndarray) -> int:
    """Index of the top of the first bounce in a series of heights of the ball's bottom."""
    i = int(np.argmax(z < 0.01))                 # the first reading at the table
    while i + 1 < len(z) and z[i + 1] <= z[i]:
        i += 1
    while i + 1 < len(z) and z[i + 1] >= z[i]:
        i += 1
    return i


def main(task_path: str, hz: int = 120) -> None:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import READY
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=hz, cam_width=320, cam_height=176, headless=True)
    T = emb.task.module
    K = T.K
    emb.reset(Scene(id="ball-probe", instruction=T.INSTRUCTION))
    from yam_arena.scene import object_states   # importable only once the simulator has started

    rest = np.r_[np.array(READY, float), 1.0]    # both arms stay in their rest pose
    step = lambda: emb.step(Action(data=pack(rest, rest)))   # noqa: E731
    for _ in range(hz // 2):
        step()
    balls = {T.GOOD[0]: (K.BOUNCE_GOOD, (T.TRAY_XY[0], T.TRAY_XY[1] + 0.18)),
             T.DEAD[0]: (K.BOUNCE_DEAD, (T.TRAY_XY[0] + 0.15, T.TRAY_XY[1] + 0.18))}
    for drop in DROPS:
        for name, (_, (x, y)) in balls.items():
            K.write_pose(emb._env, name, (x, y, T.TOP + drop + K.BALL_R))
        heights = {name: [] for name in balls}
        ends = {}
        for _ in range(int(WATCH_S * hz)):
            step()
            st = object_states(emb._env, list(balls))
            for name in balls:
                heights[name].append(float(st[name].pos[2]) - T.TOP - K.BALL_R)
                ends[name] = st[name].pos[:2]
        for name, (bounce, spot) in balls.items():
            z = np.array(heights[name])
            top = first_bounce(z)
            rule = K.rebound(drop, bounce)
            still = next((i for i in range(len(z) - 1, 0, -1) if abs(z[i] - z[-1]) > 0.002), 0)
            print(json.dumps({"ball": name, "kind": "good" if name in T.GOOD else "flat", "drop_cm": round(100 * drop, 1),
                              "came_back_cm": round(100 * float(z[top]), 2), "rule_cm": round(100 * rule, 2),
                              "error_pct": round(100 * (float(z[top]) - rule) / rule, 1),
                              "top_of_bounce_s": round(top / hz, 3), "bouncing_until_s": round(still / hz, 2),
                              "moved_sideways_mm": round(1000 * float(np.hypot(*(np.asarray(ends[name]) - np.asarray(spot)))), 1)}),
                  flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120)
