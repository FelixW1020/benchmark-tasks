"""On the station table: a small brass candlestick accent lamp, 11 inches tall with its shade on, stands with its
candelabra bulb in and its shade off. The shade, a six-sided clip-on shade, stands on the table beside it.

The job is to put the shade on the lamp. A clip-on shade has no screw or harp: inside its top, a wire fitter carries
two sprung wire loops that slide down over the bulb, and a small ring between them that comes to rest on the bulb's
tip. The shade has to be lifted over the bulb, lowered straight down the bulb's axis, and pressed lightly until the
ring is home. The lamp is narrow-based for its height (a 3 1/4 inch base under a bulb whose tip is 27 cm up): a trial
in which it goes over scores 0.

The lamp's and the shade's places change with ``SHADE_LOT`` (A or B).

Run it:    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:      python demo.py task.py out_dir
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import shadekit as K  # noqa: E402
from shadekit import Jaw, Lamp, Shade, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the shade on the lamp."

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("SHADE_LOT", "A")
# the lamp's place; the shade's place and the direction its panel 0 faces (degrees from y, counter-clockwise from above)
LAYOUT = {"A": ((0.300, -0.130), (0.300, -0.330), 0.0), "B": ((0.280, -0.105), (0.325, -0.320), -10.0)}[LOT]
LAMP_XY, SHADE_XY, SHADE_YAW = LAYOUT[0], LAYOUT[1], math.radians(LAYOUT[2])
GAP = 0.0004                       # the parts start this far above the table

LAMP, SHADE = "accent_lamp", "lamp_shade"
JAWS = ("loop_a", "loop_b")

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_UP = 0.004          # "seated": the fitter's ring no more than this far above its place on the bulb's tip ...
SEAT_OFF = 0.004         # ... and no further than this from the lamp's axis
LEVEL = 5.0              # degrees between the shade's axis and the lamp's for full score
CROOKED = 12.0           # ... and for 0.75
ON_UP = 0.045            # "on the bulb": the ring no more than this far above its place ...
ON_OFF = 0.010           # ... within this of the lamp's axis ...
ON_TILT = 30.0           # ... and the shade no more than this far from the lamp's axis
UPRIGHT = 3.0            # the lamp counts as upright within this many degrees of vertical
MOVED = 0.030            # ... and as in its place within this distance of where it started
FELL = 20.0              # the lamp has gone over once it leans this far
CLEAR = 0.030            # both grippers this far from the shade and from the lamp at the end
STILL, STILL_TURN = 0.02, 0.5    # m/s and rad/s


def lamp_start() -> tuple:
    return np.array([LAMP_XY[0], LAMP_XY[1], TOP + GAP]), (0.0, 0.0, 0.0, 1.0)


def shade_start() -> tuple:
    return np.array([SHADE_XY[0], SHADE_XY[1], TOP + GAP]), K.turn_z(SHADE_YAW)


def jaw_start(which: int) -> tuple:
    pos, quat = shade_start()
    hinge, rot = K.jaw_rest(which)
    return K.place(pos, quat, hinge), K._quat(K._rotation(quat) @ rot)


def _objects() -> list:
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    lp, lq = lamp_start()
    sp, sq = shade_start()
    wide = K.BOT_AF / math.cos(math.pi / 6)
    objs = [Lamp(LAMP, pos=tup(lp), quat=lq, size=(2 * K.BASE_R, 2 * K.BASE_R, K.TIP_Z)),
            Shade(SHADE, pos=tup(sp), quat=sq, size=(wide, wide, K.HS))]
    for which, name in enumerate(JAWS):
        jp, jq = jaw_start(which)
        objs.append(Jaw(name, pos=tup(jp), quat=jq, size=(2 * K.JAW_WIRE, 2 * K.JAW_HALF, K.JAW_L), shade=SHADE, which=which))
    return objs


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, fell=False, fell_step=0, over=False, on=False, seated=False, tilt=0.0, up=0.0, off=0.0,
                  lamp_lean=0.0, lamp_moved=0.0, lamp_ok=True, released=False, still=False, loops=[0.0, 0.0],
                  near_shade=0.0, near_lamp=0.0)


_clear()


def lamp_pose_facts(lamp_pos, lamp_quat) -> dict:
    """``lean``: the lamp's axis from the vertical (deg). ``moved``: how far the middle of its base is from where it
    started, along the table (m)."""
    lean = math.degrees(math.acos(float(np.clip(K._rotation(lamp_quat)[2, 2], -1.0, 1.0))))
    start = lamp_start()[0]
    return {"lean": lean, "moved": float(np.hypot(*(np.asarray(lamp_pos, float)[:2] - start[:2])))}


def fitted(shade_pos, shade_quat, lamp_pos, lamp_quat) -> dict:
    """Where the shade is on the lamp: ``seated`` the fitter's ring is home on the bulb's tip; ``on`` the ring is over
    the bulb and no more than 4.5 cm above its place; ``over`` the bulb's tip is inside the shade."""
    s = K.seat(shade_pos, shade_quat, lamp_pos, lamp_quat)
    seated = bool(s["up"] <= SEAT_UP and s["off"] <= SEAT_OFF and s["tilt"] <= ON_TILT)
    on = bool(s["up"] <= ON_UP and s["off"] <= ON_OFF and s["tilt"] <= ON_TILT)
    return {"seated": seated, "on": on, "over": s["over"], "tilt": s["tilt"], "up": s["up"], "off": s["off"]}


def score_from(fell: bool, over: bool, on: bool, seated: bool, tilt: float, lamp_ok: bool, released: bool, still: bool) -> float:
    """Pure scoring rule. ``fell``: the lamp leaned 20 degrees or more at any time in the trial. ``over``: the bulb's
    tip is inside the shade. ``on``: the fitter's ring is over the bulb, within 1 cm of the lamp's axis and no more
    than 4.5 cm above its place. ``seated``: the ring is home (within 4 mm of its place and of the axis). ``tilt``:
    degrees between the shade's axis and the lamp's. ``lamp_ok``: the lamp is upright within 3 degrees and within 3 cm
    of where it started. ``released``: both grippers are clear of the shade and the lamp. ``still``: both are at rest."""
    if fell:
        return 0.0
    at_rest = released and still
    if seated and tilt <= LEVEL and lamp_ok and at_rest:
        return 1.0
    if seated and tilt <= CROOKED and at_rest:
        return 0.75
    if on and at_rest:
        return 0.5
    if over:
        return 0.25
    return 0.0


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    st = object_states(environment, [LAMP, SHADE, *JAWS])
    lamp, shade = st[LAMP], st[SHADE]
    lf = lamp_pose_facts(lamp.pos, lamp.quat_xyzw)
    s["lamp_lean"], s["lamp_moved"] = lf["lean"], lf["moved"]
    if lf["lean"] >= FELL and not s["fell"]:
        s["fell"], s["fell_step"] = True, s["steps"]
    s["lamp_ok"] = bool(lf["lean"] <= UPRIGHT and lf["moved"] <= MOVED)
    s.update(fitted(shade.pos, shade.quat_xyzw, lamp.pos, lamp.quat_xyzw))
    s["loops"] = [math.degrees(K.jaw_angle(i, shade.quat_xyzw, st[n].quat_xyzw)) for i, n in enumerate(JAWS)]
    hands = np.array([p for side in K.gripper_points(environment) for p in side]) - np.array([0.0, 0.0, 0.0])
    s["near_shade"] = K.shade_distance(hands, shade.pos, shade.quat_xyzw)
    s["near_lamp"] = K.lamp_distance(hands, lamp.pos, lamp.quat_xyzw)
    s["released"] = bool(min(s["near_shade"], s["near_lamp"]) >= CLEAR)
    slow = lambda o: float(np.linalg.norm(o.lin_vel)) < STILL and float(np.linalg.norm(o.ang_vel)) < STILL_TURN   # noqa: E731
    s["still"] = bool(slow(lamp) and slow(shade))


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial. 0 if the lamp leaned 20 degrees or more at any time (it went over). Otherwise
    1.0: the shade is seated (the ring of its fitter within 4 mm of its place on the bulb's tip and of the lamp's
    axis), level (its axis within 5 degrees of the lamp's), the lamp upright within 3 degrees and within 3 cm of where
    it started, both grippers at least 3 cm from the shade and the lamp, and both at rest (under 2 cm/s and
    0.5 rad/s). 0.75: seated, let go and at rest, but the shade is 5 to 12 degrees off level or the lamp has been
    moved or left leaning. 0.5: the shade hangs on the bulb without being home (its ring within 1 cm of the lamp's
    axis and no more than 4.5 cm above its place, the shade within 30 degrees of the lamp's axis), or is home but
    12 to 30 degrees off level; let go and at rest. 0.25: the bulb's tip is inside the shade, in
    any other state (still held, or resting crooked). 0 otherwise."""
    s = _STATE
    return score_from(s["fell"], s["over"], s["on"], s["seated"], s["tilt"], s["lamp_ok"], s["released"], s["still"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    mm = lambda v: round(1000 * float(v), 1)   # noqa: E731
    return {"score": score(objects), "lot": LOT, "fell": s["fell"], "fell_step": s["fell_step"], "over": s["over"], "on": s["on"],
            "seated": s["seated"], "tilt_deg": round(s["tilt"], 1), "up_mm": mm(s["up"]), "off_mm": mm(s["off"]),
            "lamp_lean_deg": round(s["lamp_lean"], 1), "lamp_moved_mm": mm(s["lamp_moved"]), "lamp_ok": s["lamp_ok"],
            "released": s["released"], "still": s["still"], "loops_deg": [round(v, 1) for v in s["loops"]],
            "near_shade_mm": mm(s["near_shade"]), "near_lamp_mm": mm(s["near_lamp"])}


def demo(objects=None, seed=None):
    return {}
