"""On the station table, which stands in for a kitchen counter: a wind-up kitchen timer, a steel cylinder 6 cm across
and 6 cm tall in two parts. The lower part carries the minute scale, the upper part an index mark. It reads zero.

The job is to set it for 10 minutes so that it will ring. With a clockwork timer that takes two things the
instruction does not say:

- it has to be turned all the way to the 55-minute mark first, which winds the bell, and only then back to 10.
  Turned straight to 10 the timer counts down and does not ring;
- one part has to be held while the other is turned. Turned by one hand where it stands, the whole timer goes round
  on the counter and the reading falls far short of the mark.

Once set, the timer runs: its reading falls by a minute every minute.

The timer's place, and the way its zero mark faces, change with ``TIMER_LOT`` (A or B).

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

import timerkit as K  # noqa: E402
from timerkit import Base, Top, Train, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Set the kitchen timer for 10 minutes."

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("TIMER_LOT", "A")
# where the timer stands, and the direction its zero mark faces (degrees from x; 180 is toward the robot)
LAYOUT = {"A": ((0.300, -0.060), 180.0), "B": ((0.260, 0.050), 120.0)}[LOT]
REST = 0.0003                 # the parts are set down this far above the table

TIMER, DIAL, TRAIN = "timer", "timer_top", "timer_train"

# -- scoring -----------------------------------------------------------------------------------------------------------
SET_MIN = 10.0                # the time asked for
EXACT, NEAR = 1.0, 3.0        # minutes either side of it: full marks, and the tier below
WOUND_MIN = 54.0              # "turned to the 55-minute mark": the reading has reached this
RUNG_MIN = 0.5                # back at zero the bell goes off, and the winding is spent
UPRIGHT = 10.0                # degrees the timer's axis may lean and still count as standing
DOWN = 0.005                  # ... with its underside no more than this far above the table
CLEAR = 0.030                 # both grippers this far from the timer at the end
STILL, STILL_TURN = 0.02, 0.5   # m/s and rad/s


def timer_start() -> tuple:
    """The lower part's position and quaternion at the start."""
    (x, y), deg = LAYOUT
    return np.array([x, y, TOP + REST]), K.turn_z(math.radians(deg))


def _objects() -> list:
    pos, quat = timer_start()
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    fixed = os.environ.get("TIMER_BENCH") == "1"              # bench check of the mechanism: the lower part is held
    return [Base(TIMER, pos=tup(pos), quat=quat, size=(2 * K.RADIUS, 2 * K.RADIUS, K.SPLIT), fixed=fixed),
            Train(TRAIN, pos=tup(K.place(pos, quat, (0.0, 0.0, K.SPLIT + K.SEAM / 2))), quat=quat,
                  size=(0.010, 0.010, K.SEAM), base=TIMER),
            Top(DIAL, pos=tup(K.place(pos, quat, (0.0, 0.0, K.SPLIT + K.SEAM))), quat=quat,
                size=(2 * K.RADIUS, 2 * K.RADIUS, K.TOP_H), base=TIMER, train=TRAIN)]


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, deg=0.0, deg_max=0.0, wound=False, wound_step=0, rung_step=0, standing=True, on_table=True,
                  released=True, still=True, tilt=0.0, up=0.0, near=1.0, moved=0.0)


_clear()


def wound_after(wound: bool, minute: float) -> bool:
    """Whether the bell is wound after a step in which the dial reads ``minute``: reaching the 55-minute mark winds
    it, and running (or being turned) back to zero rings it and spends the winding."""
    if minute >= WOUND_MIN:
        return True
    if minute <= RUNG_MIN:
        return False
    return wound


def standing(base_pos, base_quat) -> dict:
    """``standing``: the timer stands on the table the right way up; ``on_table``: it has not fallen off the table."""
    lean, up = K.tilt(base_quat), float(base_pos[2] - TOP)
    return {"standing": bool(lean <= UPRIGHT and -DOWN <= up <= DOWN), "on_table": bool(up > -0.05), "tilt": lean, "up": up}


def score_from(wound: bool, minute: float, standing: bool, released: bool, still: bool, on_table: bool = True) -> float:
    """Pure scoring rule. ``wound``: the dial has been at the 55-minute mark (54 or more) and not back at zero since.
    ``minute``: the dial's reading now. ``standing``: the timer stands upright on the table. ``released``: both
    grippers are clear of it. ``still``: it is at rest. ``on_table``: it has not fallen off the table."""
    if not on_table:
        return 0.0
    left = standing and released and still
    off = abs(minute - SET_MIN)
    if wound and left and off <= EXACT:
        return 1.0
    if wound and left and off <= NEAR:
        return 0.75
    if wound and left:
        return 0.5
    if wound or (left and off <= EXACT):
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
    st = object_states(environment, [TIMER, DIAL])
    base, top = st[TIMER], st[DIAL]
    s["deg"] = K.dial(DIAL)
    s["deg_max"] = max(s["deg_max"], s["deg"])
    minute = K.minutes(s["deg"])
    was = s["wound"]
    s["wound"] = wound_after(was, minute)
    if s["wound"] and not was:
        s["wound_step"] = s["steps"]
    if was and not s["wound"]:
        s["rung_step"] = s["steps"]
    s.update(standing(base.pos, base.quat_xyzw))
    hands = [p for side in K.gripper_points(environment) for p in side]
    s["near"] = min(K.axis_distance(p, base.pos, base.quat_xyzw) for p in hands)
    s["released"] = bool(s["near"] >= CLEAR)
    s["still"] = bool(max(float(np.linalg.norm(base.lin_vel)), float(np.linalg.norm(top.lin_vel))) < STILL
                      and max(float(np.linalg.norm(base.ang_vel)), float(np.linalg.norm(top.ang_vel))) < STILL_TURN)
    start, _ = timer_start()
    s["moved"] = float(np.linalg.norm(np.asarray(base.pos, float)[:2] - start[:2]))


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial. The timer's reading is the angle the lower part has been turned clockwise
    (seen from above) against the upper part from zero, 6 degrees to the minute. "Wound": the reading has reached 54 minutes or more (the
    55-minute mark the maker asks for) and has not been back to 0.5 or less since. "Left standing": the timer stands
    on the table with its axis within 10 degrees of upright, both grippers at least 3 cm from it, at rest (under
    2 cm/s and 0.5 rad/s). 1.0: wound, left standing, reading 9 to 11 minutes. 0.75: wound, left standing, reading
    7 to 13. 0.5: wound and left standing at any other reading. 0.25: wound but not left standing (still held,
    lying down); or never wound, but left standing reading 9 to 11 (it will count down and not ring). 0 otherwise,
    and whenever the timer has left the table."""
    s = _STATE
    return score_from(s["wound"], K.minutes(s["deg"]), s["standing"], s["released"], s["still"], s["on_table"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    return {"score": score(objects), "lot": LOT, "minutes": round(K.minutes(s["deg"]), 2), "max_minutes": round(K.minutes(s["deg_max"]), 2),
            "wound": s["wound"], "wound_step": s["wound_step"], "rung_step": s["rung_step"], "standing": s["standing"],
            "on_table": s["on_table"], "released": s["released"], "still": s["still"], "tilt_deg": round(s["tilt"], 1),
            "up_mm": round(1000 * s["up"], 1), "near_mm": round(1000 * s["near"], 1), "moved_mm": round(1000 * s["moved"], 1)}


def demo(objects=None, seed=None):
    return {}
