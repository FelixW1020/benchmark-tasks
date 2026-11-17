"""On the station table: a 2-quart hand-crank salad spinner with washed lettuce in its basket, its loose lid on.

The lettuce has to be spun dry: at least two-thirds of the water it carries spun out of it, with the lid on. Water
leaves only when the basket spins fast (well over 5 turns a second for about 6 seconds does it; see "The drying rule"
in the README). The crank on the lid turns the basket through 1:5 gears and a one-way clutch, so the basket spins up while the crank
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

INSTRUCTION = ("Spin the washed lettuce dry in the salad spinner: crank it clockwise fast enough and long enough to spin "
               "out at least two-thirds of the water the lettuce carries (the basket well over 5 turns a second for "
               "about 6 seconds), keep the lid on, then let go of the crank.")

LOT = "A"                                # one fixed layout
SPINNER_XY = np.array([0.320, -0.070])   # the spinner's base, centred here
BOWL_MASS = 0.13                         # the bowl; the basket with its lettuce adds 0.27 kg
LID_MASS = 0.20                          # the lid with its gears; the crank and knob add 0.03 kg
NAME, LID = "spinner", "lid"

# -- scoring -----------------------------------------------------------------------------------------------------------
# What is scored is water: the share of the water the washed lettuce carried in that has been spun out of it. On a real
# spinner that is weighed (lettuce before and after). The simulator has no water, so it works that share out each step
# from how hard the basket's wall presses outward, by the rule below (README, "The drying rule"). The rule is in terms
# of that push, in multiples of gravity, so it does not depend on the size of the machine. Its two numbers are a
# physical argument and an estimate; neither has been measured on a spinner.
HOLD_EXP = 0.5                           # leaves pressed outward at n times gravity can hold n ** -HOLD_EXP of the water
                                         # they held dripping at 1 g (surface tension: the capillary length goes as n ** -0.5)
DRAIN_S = 2.0                            # water above that leaves with this time constant, s (an estimate)
DRY = 2.0 / 3.0                          # this share of the water spun out: dry, 1.0
HALF = 1.0 / 2.0                         # this share: 0.5
SOME = 1.0 / 3.0                         # this share: 0.25
KNOCKED_MOVE = 0.08                      # m: the bowl shoved this far across the table at any time ...
KNOCKED_TILT = math.radians(15.0)        # ... or tilted this far: knocked about, 0 for the trial
CLEAR = 0.040                            # both grippers' tool points this far from the knob: let go


def spinner_start() -> np.ndarray:
    return np.array([SPINNER_XY[0], SPINNER_XY[1], TOP])


def wall_g(rps: float, radius: float | None = None) -> float:
    """How hard the basket's wall presses water outward at ``rps`` turns a second, in multiples of gravity."""
    return (2 * math.pi * rps) ** 2 * (K.BASKET_R if radius is None else radius) / 9.81


def holds(n: float) -> float:
    """The share of its water that washed, dripped lettuce can still hold when pressed outward at ``n`` times gravity.
    At 1 g or less it holds all of it (it has already dripped at 1 g)."""
    return 1.0 if n <= 1.0 else n ** -HOLD_EXP


def drain(water: float, rps: float, dt: float) -> float:
    """The share of its water still on the lettuce after ``dt`` more seconds with the basket at ``rps`` turns a second,
    given the share ``water`` on it now. Water only ever leaves."""
    keep = holds(wall_g(rps))
    if water <= keep:
        return water
    return keep + (water - keep) * math.exp(-dt / DRAIN_S)


def turns_a_second_for(share: float) -> float:
    """The basket speed (turns a second) that, kept up for long enough, spins out ``share`` of the water."""
    n = (1.0 - share) ** (-1.0 / HOLD_EXP)
    return math.sqrt(n * 9.81 / K.BASKET_R) / (2 * math.pi)


OBJECTS = [Spinner(NAME, pos=tuple(float(v) for v in spinner_start()), size=(0.20, 0.26, 0.11), mass=BOWL_MASS, lid=LID),
           Lid(LID, pos=tuple(float(v) for v in spinner_start()), size=(0.21, 0.21, 0.17), mass=LID_MASS)]


# -- what happened during the trial ---------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(knocked=False, lid_off=False, crank_turns=0.0, basket_turns=0.0, basket_rps=0.0, peak_rps=0.0,
                  water=1.0, fast_s=0.0, let_go=False, steps=0, moved=0.0, yaw_turns=0.0, lift=0.0, tilt=0.0)


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
    if not s["lid_off"]:                                                  # the basket spins water out, driven or coasting
        rps = abs(float(t["basket_rps"]))
        s["peak_rps"] = max(s["peak_rps"], rps)
        s["water"] = drain(s["water"], rps, dt)
        if rps >= 5.0:
            s["fast_s"] += dt
    knob = K.knob_point(lid.pos, lid.quat_xyzw, lid.joints["crank"], up=K.KNOB_H / 2)
    s["let_go"] = all(float(np.linalg.norm(knob - h)) > CLEAR for h in finger_points(environment))


# -- scoring --------------------------------------------------------------------------------------------------------
def score_from(spun_out: float, knocked: bool, lid_off: bool, let_go: bool) -> float:
    """Pure scoring rule. ``spun_out``: the share (0 to 1) of the water the lettuce carried in that has left it.
    ``knocked``: the spinner was shoved or tipped. ``lid_off``: the lid came off the bowl at some time. ``let_go``:
    both grippers are clear of the knob now."""
    if knocked or lid_off:
        return 0.0
    if spun_out >= DRY - 1e-9 and let_go:
        return 1.0
    if spun_out >= HALF - 1e-9:
        return 0.5
    return 0.25 if spun_out >= SOME - 1e-9 else 0.0


def score(objects) -> float:
    """0 if the spinner was ever knocked about (shoved 8 cm across the table or tilted 15 degrees) or its lid ever came
    off the bowl (lifted 5 mm, tipped 8 degrees from the rim, or 10.5 cm off its axis). Otherwise, by the share of the
    water the lettuce carried in that has been spun out of it: 1.0 for two-thirds or more with, at the end, both
    grippers 4 cm or more from the crank's knob; 0.5 for half or more; 0.25 for a third or more; else 0."""
    s = _STATE
    return score_from(1.0 - s["water"], s["knocked"], s["lid_off"], s["let_go"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    st = objects[NAME]
    s = _STATE
    return {"score": score(objects), "crank_turns": round(s["crank_turns"], 2), "basket_turns": round(s["basket_turns"], 1),
            "basket_rps": round(s["basket_rps"], 2), "peak_rps": round(s["peak_rps"], 2),
            "peak_wall_g": round(wall_g(s["peak_rps"]), 1), "s_at_5_rps": round(s["fast_s"], 2),
            "water_spun_out": round(1.0 - s["water"], 3),
            "knocked": s["knocked"], "lid_off": s["lid_off"], "lid_lift_mm": round(1000 * s["lift"], 1),
            "lid_tilt_deg": round(math.degrees(s["tilt"]), 1), "let_go": s["let_go"],
            "moved_cm": round(100 * s["moved"], 2), "bowl_yaw_deg": round(360 * s["yaw_turns"], 1),
            "spinner_cm": [round(100 * float(v), 2) for v in st.pos], "tilt_deg": round(math.degrees(tilt(st.quat_xyzw)), 1)}


def demo(objects, seed):
    return {}
