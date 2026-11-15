"""On the station table: a 2-quart hand-crank salad spinner with washed lettuce in its basket, its loose lid on.

The lettuce has to be spun dry: the basket kept at 5 turns a second (300 rpm) or more for 5 seconds, with the lid on.
The crank on the lid turns the basket through 1:5 gears and a one-way clutch, so the basket spins up while the crank
is turned clockwise and coasts when it stops. The gears resist the crank, and that resistance turns the whole bowl on
the worktop unless something holds it: the bowl's plastic base slides easily. So one hand holds the bowl by its side
handle while the other turns the crank. The lid only lies on the bowl: a hand that pulls up on the knob, or drags it
sideways, lifts or tips the lid off, and then the crank drives nothing.

Run it:    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:      python demo.py task.py out_dir
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spinnerkit as K  # noqa: E402
from spinnerkit import Lid, Spinner, finger_points, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Spin the washed lettuce dry in the salad spinner: turn its crank clockwise fast enough to keep the basket "
               "spinning at 5 turns a second or more for 5 seconds, keep the lid on, then let go of the crank.")

# -- layout ---------------------------------------------------------------------------------------------------------
LOT = "A"                                # one fixed layout
SPINNER_XY = np.array([0.320, -0.070])   # the spinner's base, centred here
BOWL_MASS = 0.13                         # the bowl; the basket with its lettuce adds 0.27 kg
LID_MASS = 0.20                          # the lid with its gears; the crank and knob add 0.03 kg
NAME, LID = "spinner", "lid"

# -- scoring -----------------------------------------------------------------------------------------------------------
# Drying: water leaves the leaves only while the basket spins fast. 5 turns a second (300 rpm) is the slow end of the
# speeds salad spinners are run at; at this basket's wall (8.8 cm out) it presses water outward at 8.9 times gravity.
# Nothing slower earns any credit: every score above 0 is for time spent at 300 rpm or more with the lid on the bowl.
DRY_RPS = 5.0                            # the basket at or above this many turns a second is drying the lettuce
DRY_S = 5.0                              # ... and has dried it after this long at it, in all
HALF_S = 2.5                             # half of that: 0.5
SOME_S = 1.0                             # the basket got up to drying speed and stayed there a second: 0.25
KNOCKED_MOVE = 0.08                      # m: the bowl shoved this far across the table at any time ...
KNOCKED_TILT = math.radians(15.0)        # ... or tilted this far: knocked about, 0 for the trial
CLEAR = 0.040                            # both grippers' tool points this far from the knob: let go


def spinner_start() -> np.ndarray:
    return np.array([SPINNER_XY[0], SPINNER_XY[1], TOP])


def wall_g(rps: float) -> float:
    """How hard the basket's wall presses water outward at ``rps`` turns a second, in multiples of gravity."""
    return (2 * math.pi * rps) ** 2 * K.BASKET_R / 9.81


OBJECTS = [Spinner(NAME, pos=tuple(float(v) for v in spinner_start()), size=(0.20, 0.26, 0.11), mass=BOWL_MASS, lid=LID),
           Lid(LID, pos=tuple(float(v) for v in spinner_start()), size=(0.21, 0.21, 0.17), mass=LID_MASS)]


# -- what happened during the trial ---------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(knocked=False, lid_off=False, crank_turns=0.0, basket_turns=0.0, basket_rps=0.0, peak_rps=0.0,
                  dry_steps=0, dry_s=0.0, let_go=False, steps=0, moved=0.0, yaw_turns=0.0, lift=0.0, tilt=0.0)


_clear()


def tilt(quat) -> float:
    return math.acos(max(-1.0, min(1.0, float(K._rotation(quat)[2, 2]))))


def yaw(quat) -> float:
    r = K._rotation(quat)
    return math.atan2(r[1, 0], r[0, 0])


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    dt = float(environment.unwrapped.step_dt)
    states = object_states(environment, [NAME, LID])
    st, lid = states[NAME], states[LID]
    pos, quat = np.asarray(st.pos), st.quat_xyzw
    moved = float(np.linalg.norm(pos[:2] - spinner_start()[:2]))
    s["moved"] = max(s["moved"], moved)
    s["yaw_turns"] = yaw(quat) / (2 * math.pi)
    t = K.turns(NAME)
    s["lift"], s["tilt"] = max(s["lift"], t["lift"]), max(s["tilt"], t["tilt"])
    if s["steps"] > 5:
        if moved > KNOCKED_MOVE or tilt(quat) > KNOCKED_TILT:
            s["knocked"] = True
        if not t["on"]:
            s["lid_off"] = True                                           # the lid came off the bowl
    s["crank_turns"], s["basket_turns"], s["basket_rps"] = t["crank"], t["basket"], t["basket_rps"]
    if t["on"] and not s["lid_off"]:
        s["peak_rps"] = max(s["peak_rps"], t["basket_rps"])
        if t["basket_rps"] >= DRY_RPS:
            s["dry_steps"] += 1
            s["dry_s"] = s["dry_steps"] * dt
    knob = K.knob_point(lid.pos, lid.quat_xyzw, lid.joints["crank"], up=K.KNOB_H / 2)
    s["let_go"] = all(float(np.linalg.norm(knob - h)) > CLEAR for h in finger_points(environment))


# -- scoring --------------------------------------------------------------------------------------------------------
def score_from(dry_s: float, knocked: bool, lid_off: bool, let_go: bool) -> float:
    """Pure scoring rule. ``dry_s``: seconds in all that the basket turned at DRY_RPS or faster with the lid on the
    bowl. ``knocked``: the spinner was shoved or tipped. ``lid_off``: the lid came off the bowl at some time.
    ``let_go``: both grippers are clear of the knob now. A basket that never reaches DRY_RPS scores 0, however long
    it is cranked."""
    if knocked or lid_off:
        return 0.0
    if dry_s >= DRY_S and let_go:
        return 1.0
    if dry_s >= HALF_S:
        return 0.5
    return 0.25 if dry_s >= SOME_S else 0.0


def score(objects) -> float:
    """0 if the spinner was ever knocked about (shoved 8 cm across the table or tilted 15 degrees) or its lid ever came
    off the bowl (lifted 5 mm, tipped 8 degrees from the rim, or 10.5 cm off its axis). Otherwise, by the time in all
    that the basket turned at 5 turns a second or more with the lid on: 1.0 for 5 seconds or more with, at the end,
    both grippers 4 cm or more from the crank's knob; 0.5 for 2.5 seconds or more; 0.25 for 1 second or more; else 0."""
    s = _STATE
    return score_from(s["dry_s"], s["knocked"], s["lid_off"], s["let_go"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    st = objects[NAME]
    s = _STATE
    return {"score": score(objects), "crank_turns": round(s["crank_turns"], 2), "basket_turns": round(s["basket_turns"], 1),
            "basket_rps": round(s["basket_rps"], 2), "peak_rps": round(s["peak_rps"], 2), "dry_s": round(s["dry_s"], 2),
            "knocked": s["knocked"], "lid_off": s["lid_off"], "lid_lift_mm": round(1000 * s["lift"], 1),
            "lid_tilt_deg": round(math.degrees(s["tilt"]), 1), "let_go": s["let_go"],
            "moved_cm": round(100 * s["moved"], 2), "bowl_yaw_deg": round(360 * s["yaw_turns"], 1),
            "spinner_cm": [round(100 * float(v), 2) for v in st.pos], "tilt_deg": round(math.degrees(tilt(st.quat_xyzw)), 1)}


def demo(objects, seed):
    return {}
