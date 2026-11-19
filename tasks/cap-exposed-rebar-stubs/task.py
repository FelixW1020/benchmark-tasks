"""On the station table, which stands in for the top of a footing on a building site: three reinforcing-bar stubs
stand in a row out of a board, 14 cm tall. One already wears an impalement-rated safety cap (a 4 in square
steel-reinforced top on a round neck). The other two are bare. On the table in front of them lie two more rated caps,
on their sides as they came out of the box, and three plain plastic mushroom caps.

Bare bar ends are an impalement hazard to anyone who trips or falls onto them, and the rule is that they be guarded.
A mushroom cap is not a guard: it is the tempting wrong choice here, small, light and easy to pick up. A rated cap
guards a bar only once it has been pushed down until the bar's end is against the steel insert inside it; set on
loosely it sits high on its fins, or tips off. The instruction says none of this.

The 4 in top is wider than the gripper opens, so a rated cap cannot be taken from above. It can be taken by its neck.

Which bar is capped at the start, and where the loose caps lie, change with ``REBAR_LOT`` (A or B).

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

import rebarkit as K  # noqa: E402
from rebarkit import Cap, Fin, Footing, Mushroom, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "The crew is about to start work around this footing. Make the rebar safe."

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("REBAR_LOT", "A")
BOARD_X = 0.330                    # the board's centre line, and the bars'
BAR_Y = (-0.130, 0.000, 0.130)     # the three bars, right to left as the robot sees them
BAR_TOP = K.BOARD[2] + K.BAR_UP    # the bars' ends above the table (0.160)
GAP = 0.0004                       # loose parts start this far above what they rest on
# capped: the bar that wears a cap at the start. caps: where the middle of each loose rated cap's mouth is (x, y), and
# which way its mouth points (degrees from x; 180 is toward the robot). up: mushroom caps standing on their heads.
# flat: the one mushroom cap that lies on its side, placed like the rated caps.
LAYOUT = {
    "A": {"capped": 2, "caps": (((0.170, -0.150), 180.0), ((0.170, -0.010), 180.0)),
          "up": ((0.150, 0.160), (0.215, 0.215)), "flat": ((0.175, 0.105), 180.0)},
    "B": {"capped": 0, "caps": (((0.165, -0.020), 180.0), ((0.185, 0.125), 180.0)),
          "up": ((0.150, -0.265), (0.220, -0.300)), "flat": ((0.170, -0.160), 180.0)},
}[LOT]
CAPPED = LAYOUT["capped"]
BARE = tuple(i for i in range(len(BAR_Y)) if i != CAPPED)

FOOTING = "footing"
CAPS = ("cap_a", "cap_b", "cap_c")           # cap_a and cap_b lie loose; cap_c is on its bar at the start
MUSHROOMS = ("mushroom_a", "mushroom_b", "mushroom_c")   # a and b stand on their heads; c lies on its side
FINS = {c: tuple(f"{c}_fin{k}" for k in range(K.FINS)) for c in CAPS}

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_GAP = 0.002         # "seated": the bar's end within this of the steel insert ...
LEVEL = 5.0              # ... and the cap's axis within this many degrees of the bar's
ON_DEPTH = 0.010         # "on the bar": the bar's end at least this far up the cap's bore ...
ON_TILT = 30.0           # ... and the cap within this of upright
CLEAR = 0.030            # both grippers this far from a cap for it to count as let go
STILL, STILL_TURN = 0.02, 0.5    # m/s and rad/s
TOP_SEATED = BAR_TOP + K.CAP_H - K.INSERT_Z      # a seated cap's top above the table (0.1779)


def bar_tip(i: int) -> np.ndarray:
    return np.array([BOARD_X, BAR_Y[i], TOP + BAR_TOP])


def lying_start(xy, heading_deg: float, mushroom: bool = False) -> tuple:
    """Position (the middle of the mouth) and quaternion of a cap lying on its side at ``xy``."""
    dims = (K.M_STEM_R, K.M_HEAD_R, K.M_STEM_L) if mushroom else (K.NECK_R, K.PLATE / 2, K.NECK_L)
    z, quat, _ = K.lying(math.radians(heading_deg), *dims)
    return np.array([xy[0], xy[1], TOP + z + GAP]), quat


def cap_start(name: str) -> tuple:
    if name == "cap_c":
        return bar_tip(CAPPED) - np.array([0.0, 0.0, K.INSERT_Z - GAP]), (0.0, 0.0, 0.0, 1.0)
    xy, heading = LAYOUT["caps"][CAPS.index(name)]
    return lying_start(xy, heading)


def fin_start(cap: str, which: int) -> tuple:
    pos, quat = cap_start(cap)
    origin, rot = K.fin_rest(which)
    out = (K.BAR_R - K.FIN_IN + 0.0002) if cap == "cap_c" else 0.0      # on its bar a cap's fins start pushed aside
    return K.place(pos, quat, origin + rot[:, 0] * out), K._quat(K._rotation(quat) @ rot)


def mushroom_start(name: str) -> tuple:
    i = MUSHROOMS.index(name)
    if i == 2:
        return lying_start(*LAYOUT["flat"], mushroom=True)
    x, y = LAYOUT["up"][i]
    return np.array([x, y, TOP + K.M_STEM_L + K.M_HEAD_T + GAP]), (1.0, 0.0, 0.0, 0.0)     # head down, mouth up


def _objects() -> list:
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    objs = [Footing(FOOTING, pos=(BOARD_X, 0.0, TOP), size=K.BOARD, bars=BAR_Y, top=BAR_TOP)]
    for c in CAPS:
        cp, cq = cap_start(c)
        objs.append(Cap(c, pos=tup(cp), quat=cq, size=(K.PLATE, K.PLATE, K.CAP_H)))
        for k, f in enumerate(FINS[c]):
            fp, fq = fin_start(c, k)
            objs.append(Fin(f, pos=tup(fp), quat=fq, size=(2 * K.FIN_ROD, 2 * K.FIN_ROD, K.FIN_EDGE), cap=c, which=k))
    for m in MUSHROOMS:
        mp, mq = mushroom_start(m)
        objs.append(Mushroom(m, pos=tup(mp), quat=mq, size=(2 * K.M_HEAD_R, 2 * K.M_HEAD_R, K.M_STEM_L + K.M_HEAD_T)))
    return objs


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, bars=["bare"] * len(BAR_Y), which=[""] * len(BAR_Y), depth=[0.0] * len(BAR_Y),
                  tilt=[0.0] * len(BAR_Y), off=[0.0] * len(BAR_Y), near={}, fins={})


_clear()


def cap_on_bar(cap_pos, cap_quat, i: int) -> dict:
    """A rated cap against bar ``i``: ``on`` the bar's end is at least 1 cm up its bore, the cap within 30 degrees of
    upright; ``seated`` the end is within 2 mm of the steel insert and the cap within 5 degrees of the bar's axis."""
    f = K.bar_in_cap(cap_pos, cap_quat, bar_tip(i))
    on = bool(f["inside"] and f["depth"] >= ON_DEPTH and f["tilt"] <= ON_TILT)
    seated = bool(on and f["depth"] >= K.INSERT_Z - SEAT_GAP and f["tilt"] <= LEVEL)
    return {"on": on, "seated": seated, **f}


def bar_state(rated: list, mushroom: bool) -> str:
    """One bar's state from what is on it. ``rated``: for each rated cap whose bore holds the bar's end,
    (seated, let go and at rest). "guarded": a rated cap is seated on it, let go and at rest. "loose": a rated cap is
    on it, let go and at rest, but not seated. "mushroom": it wears a mushroom cap. "bare": anything else (a cap that
    is still held or still moving guards nothing yet)."""
    if any(seated and free for seated, free in rated):
        return "guarded"
    if any(free for _, free in rated):
        return "loose"
    return "mushroom" if mushroom else "bare"


def score_from(bars, capped: int) -> float:
    """Pure scoring rule. ``bars``: each bar's state, "guarded", "loose", "mushroom" or "bare". ``capped``: the bar
    that wore a cap at the start. 0 if that bar is no longer guarded. Otherwise each of the other two bars adds 0.5
    if it is guarded and 0.25 if a rated cap is on it but not seated; a bare bar and a bar under a mushroom cap add
    nothing."""
    if bars[capped] != "guarded":
        return 0.0
    worth = {"guarded": 0.5, "loose": 0.25}
    return float(sum(worth.get(s, 0.0) for i, s in enumerate(bars) if i != capped))


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    names = [*CAPS, *MUSHROOMS, *(f for c in CAPS for f in FINS[c])]
    st = object_states(environment, names)
    hands = np.array([p for side in K.gripper_points(environment) for p in side])
    slow = lambda o: float(np.linalg.norm(o.lin_vel)) < STILL and float(np.linalg.norm(o.ang_vel)) < STILL_TURN   # noqa: E731
    free, near = {}, {}
    for c in CAPS:
        near[c] = K.cap_distance(hands, st[c].pos, st[c].quat_xyzw)
        free[c] = bool(near[c] >= CLEAR and slow(st[c]))
    for i in range(len(BAR_Y)):
        rated, deepest = [], None
        for c in CAPS:
            f = cap_on_bar(st[c].pos, st[c].quat_xyzw, i)
            if f["on"]:
                rated.append((f["seated"], free[c]))
                if deepest is None or f["depth"] > deepest[1]["depth"]:
                    deepest = (c, f)
        mush = ""
        for m in MUSHROOMS:
            if K.bar_in_mushroom(st[m].pos, st[m].quat_xyzw, bar_tip(i))["inside"]:
                mush = m
        s["bars"][i] = bar_state(rated, bool(mush))
        s["which"][i] = deepest[0] if deepest else mush
        s["depth"][i], s["tilt"][i], s["off"][i] = (deepest[1]["depth"], deepest[1]["tilt"], deepest[1]["off"]) if deepest else (0.0, 0.0, 0.0)
    s["near"] = near
    s["fins"] = {c: [K.fin_travel(k, st[c].pos, st[c].quat_xyzw, st[f].pos) for k, f in enumerate(FINS[c])] for c in CAPS}


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial. A bar is guarded when a rated cap is seated on it (the bar's end within 2 mm of
    the cap's steel insert, the cap's axis within 5 degrees of the bar's), both grippers are at least 3 cm from that
    cap, and the cap is at rest (under 2 cm/s and 0.5 rad/s). 0 if the bar that was capped at the start is no longer
    guarded. Otherwise each of the two bars that were bare adds 0.5 if it is guarded, and 0.25 if a rated cap is on
    it, let go and at rest, but not seated (the bar's end at least 1 cm up the bore, the cap within 30 degrees of
    upright). A bar left bare or wearing a mushroom cap adds nothing."""
    return score_from(_STATE["bars"], CAPPED)


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    mm = lambda v: round(1000 * float(v), 1)   # noqa: E731
    return {"score": score(objects), "lot": LOT, "capped": CAPPED, "bars": list(s["bars"]), "which": list(s["which"]),
            "depth_mm": [mm(v) for v in s["depth"]], "tilt_deg": [round(float(v), 1) for v in s["tilt"]],
            "off_mm": [mm(v) for v in s["off"]], "near_mm": {c: mm(v) for c, v in s["near"].items()},
            "fins_mm": {c: [mm(v) for v in vs] for c, vs in s["fins"].items()}}


def demo(objects=None, seed=None):
    return {}
