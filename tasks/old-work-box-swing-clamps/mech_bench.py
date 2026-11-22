"""Test bench for the box's swing clamps alone (CPU PhysX, about 3 minutes).

    python mech_bench.py [--out bench_out]

The bare box, without its cable, starts flush in the cut-out (its ears on the wall) and the arms stay at their ready poses. A steady torque is
put on each clamp screw in turn, as a screwdriver would (the PhysX joint's effort), and the clamp's state is logged:
the turns until its arm is swung out, until it touches the back of the drywall, and the force it presses with at
each torque. Then the screw is turned back out one turn, and the box is pulled straight out with 20 N.
Writes bench.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

os.environ.setdefault("OWB_NO_CABLE", "1")     # the bench is of the bare box: no cable, the box free
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH_TASK = '''
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from task import *  # noqa: F401,F403
import task as _t
import boxparts as _bp
OBJECTS = [o if o.name != "box" else _bp.part("box", "box", tuple(_t.HOLE + [0.0002, 0.0, 0.0])) for o in _t.OBJECTS]
'''


def main(out: str) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    sys.path.insert(0, str(HERE))
    path = HERE / "_bench_task.py"
    path.write_text(BENCH_TASK)
    try:
        emb = yam_arena_embodiment(task=str(path), object_jitter=0, control_hz=20, cam_width=64, cam_height=48,
                                   device="cpu")
        emb.reset(Scene(id="bench", instruction="bench"))
    finally:
        path.unlink(missing_ok=True)
    import boxparts as bp
    import torch
    from yam_arena.constants import READY

    env = emb._env
    box = env.unwrapped.scene["box"]
    names = list(box.joint_names)
    hold = np.r_[READY, 1.0]

    def step():
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack
        emb.step(Action(data=pack(hold, hold)))

    def state():
        st = bp.part_state(env, "box")
        return st, bp.mechanism(st.joints)

    def effort(t, torque):
        e = torch.zeros((1, len(names)), device=box.device)
        if t is not None:
            e[0, names.index(f"screw_{t}")] = torque
        box.set_joint_effort_target(e)

    log = {"rows": []}
    for _ in range(20):
        step()
    st0, _ = state()
    for t in ("t", "b"):
        swung = touched = None
        for torque, n in ((0.05, 400), (0.10, 60), (0.20, 60), (0.40, 60)):
            effort(t, torque)
            for k in range(n):
                step()
                st, m = state()
                c = m[t]
                if swung is None and c["swing"] >= 80:
                    swung = c["turns"]
                if touched is None and c["force"] > 1.0:
                    touched = c["turns"]
                if k % 20 == 0:
                    row = {"clamp": t, "torque": torque, "turns": round(c["turns"], 3), "swing": round(c["swing"], 1),
                           "arm_x_mm": round(1000 * c["arm_x"], 2), "force": round(c["force"], 1),
                           "box_moved_mm": round(1000 * float(np.linalg.norm(st.pos - st0.pos)), 2)}
                    log["rows"].append(row)
                    print(row, flush=True)
            log[f"{t}_force_at_{torque}"] = round(c["force"], 1)
        log[f"{t}_turns_swung"], log[f"{t}_turns_touch"] = swung, touched
        effort(None, 0.0)
        for _ in range(40):
            step()
        log[f"{t}_force_left_alone"] = round(state()[1][t]["force"], 1)
    # pull the box straight out (toward the arms) with 20 N, then back the top screw out one turn
    body = box.body_names.index("body")
    f = torch.zeros((1, len(box.body_names), 3), device=box.device)
    f[0, body, 0] = -20.0
    box.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=None)
    for _ in range(40):
        step()
    st, m = state()
    log["pulled_20N_moved_mm"] = round(1000 * float(np.linalg.norm(st.pos - st0.pos)), 2)
    box.set_external_force_and_torque(torch.zeros_like(f), torch.zeros_like(f), body_ids=None)
    start = state()[1]["t"]["turns"]
    effort("t", -0.05)
    for _ in range(200):
        step()
        if state()[1]["t"]["turns"] <= start - 1.0:
            break
    effort(None, 0.0)
    for _ in range(20):
        step()
    m = state()[1]["t"]
    log["backed_out_one_turn"] = {k: round(v, 3) for k, v in m.items()}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "bench.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "rows"}), flush=True)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_out")
    main(ap.parse_args().out)
