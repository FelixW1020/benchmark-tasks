"""On the station table (standing in for a clinic's treatment-room counter): a Mayo instrument tray under a sterile
drape (the sterile field being set up for a minor procedure), and beside it a sterile supply pack that has been
opened flat, its wrapper's inside up, with two stainless medicine cups on it. A pair of sterile thumb dressing forceps
lies across the near edge of the field: working end on the field, handle sticking out over the edge.

The job is the one a nurse or medical assistant does with bare (unsterile) hands while setting up: use the sterile
forceps to move both cups from the opened pack onto the sterile field, then lay the forceps back down.

The robot's grippers are not sterile. Only the working end of the forceps may touch the cups, the field and the
inside of the pack. The outer 2.5 cm of the field and of the pack's wrapper count as unsterile, and so does the table.
No part of either arm may pass over a sterile area. While they are held, the forceps' tips must not be raised above
their handle. Anything sterile that touches something unsterile is contaminated, and passes that on to what it touches
next.

The cups' places change with ``XFER_LOT`` (A or B).

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

import xferkit as K  # noqa: E402
from xferkit import Blade, Cup, Forceps, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Using the sterile forceps, move both medicine cups from the opened sterile pack onto the sterile field "
               "without contaminating anything, then lay the forceps back across the field's edge and let go.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("XFER_LOT", "A")
BORDER = 0.025                                   # the outer 2.5 cm (1 in) of a sterile field counts as unsterile
NEAR = 0.330                                     # x of the near edge of both the field and the pack
TRAY = (0.325, 0.490, 0.019)                     # the Mayo tray: depth (x), width (y), height
DRAPE_T = 0.001                                  # the drape over it
FIELD_Y = (-0.170, 0.320)                        # the tray's edges in y
FIELD_TOP = TRAY[2] + DRAPE_T                    # the field's surface above the table
PACK = (0.305, 0.305, 0.002)                     # the opened wrapper: 12 in square, lying flat
PACK_Y = (-0.515, -0.210)
PACK_TOP = PACK[2]
# the sterile part of each: its top surface inside the border (x0, x1, y0, y1)
FIELD_IN = (NEAR + BORDER, NEAR + TRAY[0] - BORDER, FIELD_Y[0] + BORDER, FIELD_Y[1] - BORDER)
PACK_IN = (NEAR + BORDER, NEAR + PACK[0] - BORDER, PACK_Y[0] + BORDER, PACK_Y[1] - BORDER)
CUP_XY = {"A": ((0.385, -0.345), (0.385, -0.440)), "B": ((0.395, -0.320), (0.385, -0.420))}[LOT]   # on the pack
FORCEPS_Y = {"A": -0.115, "B": -0.105}[LOT]      # the forceps lie along x at this y
OVERHANG = 0.050                                 # their handle sticks out this far over the field's near edge
BLUE, WRAP = (0.20, 0.42, 0.66), (0.36, 0.60, 0.74)

FORCEPS, BLADE, CUPS = "forceps", "blade", ("cup_a", "cup_b")
AREAS = {"field": (FIELD_IN, FIELD_TOP), "pack": (PACK_IN, PACK_TOP)}

# -- scoring -----------------------------------------------------------------------------------------------------------
TOUCH = 0.0015           # within this of a surface or of another thing counts as touching it
NEAR_ARM = 0.004         # ... and within this of any point of an arm
TIPS_UP = 0.005          # while the forceps are held, their tips may not be more than this above the handle's end
UPRIGHT = math.radians(10.0)   # a cup stands if its axis is within this of upright
CLEAR = 0.030            # both arms this far from a thing for it to be let go
TIP_CLEAR = 0.020        # ... and the forceps' tips this far from a cup
STILL = 0.02             # m/s


def tail_start() -> np.ndarray:
    """Where the forceps' tail is at the start (world): on edge, the blades' lower edges on the field."""
    return np.array([NEAR - OVERHANG, FORCEPS_Y, TOP + FIELD_TOP + K.half_width(0.05) + 0.0005])


def cup_start(i: int) -> np.ndarray:
    return np.array([CUP_XY[i][0], CUP_XY[i][1], TOP + PACK_TOP + 0.0005])


def _objects() -> list:
    half = K.F_ANGLE / 2
    t = tail_start()
    tray = Box("field", pos=(NEAR + TRAY[0] / 2, sum(FIELD_Y) / 2, TOP + FIELD_TOP / 2), size=(TRAY[0], TRAY[1], FIELD_TOP),
               color=BLUE)
    pack = Box("pack", pos=(NEAR + PACK[0] / 2, sum(PACK_Y) / 2, TOP + PACK_TOP / 2), size=PACK, color=WRAP)
    fa = Forceps(FORCEPS, pos=tuple(float(v) for v in t), size=(0.14, 0.01, 0.012), quat=K.turn_z(half), blade=BLADE)
    fb = Blade(BLADE, pos=tuple(float(v) for v in t), size=(0.14, 0.01, 0.012), quat=K.turn_z(-half))
    cups = [Cup(n, pos=tuple(float(v) for v in cup_start(i)), size=(0.052, 0.052, 0.045)) for i, n in enumerate(CUPS)]
    return [tray, pack, fa, fb, *cups]


OBJECTS = _objects()


# -- where things are, and what touches what (pure) ----------------------------------------------------------------------

def _inside(points, rect) -> np.ndarray:
    p = np.atleast_2d(points)
    return (p[:, 0] >= rect[0]) & (p[:, 0] <= rect[1]) & (p[:, 1] >= rect[2]) & (p[:, 1] <= rect[3])


_FIELD_RECT = (NEAR, NEAR + TRAY[0], FIELD_Y[0], FIELD_Y[1])
_PACK_RECT = (NEAR, NEAR + PACK[0], PACK_Y[0], PACK_Y[1])


def surface_height(points) -> np.ndarray:
    """The height (above the table) of the surface under each point: the field's top, the pack's, or the table."""
    p = np.atleast_2d(points)
    h = np.zeros(len(p))
    h[_inside(p, _PACK_RECT)] = PACK_TOP
    h[_inside(p, _FIELD_RECT)] = FIELD_TOP
    return h


def over(points, area: str) -> np.ndarray:
    """For each point: is it above the sterile part of ``area`` ("field" or "pack"), at any height?"""
    return _inside(points, AREAS[area][0])


def touches(points, area: str) -> bool:
    """Does any point touch the sterile part of ``area``?"""
    p = np.atleast_2d(points)
    return bool(np.any(over(p, area) & (p[:, 2] - TOP - AREAS[area][1] < TOUCH)))


def touches_unsterile(points) -> bool:
    """Does any point touch the table, a border, or the side of the tray?"""
    p = np.atleast_2d(points)
    z = p[:, 2] - TOP
    sterile = over(p, "field") | over(p, "pack")
    if np.any(~sterile & (z - surface_height(p) < TOUCH)):
        return True
    grown = (_FIELD_RECT[0] - TOUCH, _FIELD_RECT[1] + TOUCH, _FIELD_RECT[2] - TOUCH, _FIELD_RECT[3] + TOUCH)
    return bool(np.any(_inside(p, grown) & ~_inside(p, _FIELD_RECT) & (z < FIELD_TOP)))


def _apart(a, b) -> float:
    """The least distance between two sets of points."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    best = math.inf
    for i in range(0, len(a), 256):
        d = np.linalg.norm(a[i:i + 256, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def new_state() -> dict:
    s = {"steps": 0, "dirty": {n: False for n in ("forceps", "field", "pack", *CUPS)}, "why": {},
         "cups_on": 0, "on": {n: False for n in CUPS}, "down": False, "ready": False, "tips_up_mm": 0.0,
         "gap_mm": 0.0, "arm_over": False}
    return s


def _soil(s: dict, name: str, why: str) -> None:
    if not s["dirty"][name]:
        s["dirty"][name] = True
        s["why"][name] = f"step {s['steps']}: {why}"


def advance(s: dict, arm, blades, cups, speeds) -> dict:
    """One control step of the record. ``arm``: points on both arms (world). ``blades``: ((pos, quat), (pos, quat))
    of the two blades. ``cups``: {name: (pos, quat)}. ``speeds``: {name: m/s} for "forceps" and each cup."""
    s["steps"] += 1
    arm = np.atleast_2d(arm)
    f = K.forceps_points(*blades[0], *blades[1])
    work, handle = f["work"], f["handle"]
    s["tips_up_mm"], s["gap_mm"] = round(1000 * f["up"], 1), round(1000 * f["gap"], 1)
    s["arm_over"] = False
    # an arm above a sterile area contaminates it
    for area in AREAS:
        if np.any(over(arm, area)):
            s["arm_over"] = True
            _soil(s, area, "an arm passed over it")
    # the forceps
    held = _apart(arm, np.vstack([work, handle])) < CLEAR
    if held and f["up"] > TIPS_UP:
        _soil(s, "forceps", "tips raised above the handle")
    if _apart(arm, work) < NEAR_ARM:
        _soil(s, "forceps", "an arm touched the working end")
    if touches_unsterile(work):
        _soil(s, "forceps", "the working end touched an unsterile surface")
    cup_pts = {n: K.cup_points(*cups[n]) for n in CUPS}
    for n in CUPS:
        if np.any(K.in_cup(arm, *cups[n], NEAR_ARM)):
            _soil(s, n, "an arm touched it")
        if touches_unsterile(cup_pts[n]):
            _soil(s, n, "it touched an unsterile surface")
        if np.any(K.in_cup(handle, *cups[n], TOUCH)):
            _soil(s, n, "the forceps' handle touched it")
    for area in AREAS:
        if touches(handle, area):
            _soil(s, area, "the forceps' handle touched it")
    # contamination passes between sterile things that touch
    for _ in range(3):
        for area in AREAS:
            if touches(work, area):
                if s["dirty"][area]:
                    _soil(s, "forceps", f"the working end touched the contaminated {area}")
                elif s["dirty"]["forceps"]:
                    _soil(s, area, "contaminated forceps touched it")
            for n in CUPS:
                if touches(cup_pts[n], area):
                    if s["dirty"][area]:
                        _soil(s, n, f"it stood on the contaminated {area}")
                    elif s["dirty"][n]:
                        _soil(s, area, f"contaminated {n} touched it")
        for n in CUPS:
            if np.any(K.in_cup(work, *cups[n], TOUCH)):
                if s["dirty"]["forceps"]:
                    _soil(s, n, "contaminated forceps touched it")
                elif s["dirty"][n]:
                    _soil(s, "forceps", f"the working end touched contaminated {n}")
        a, b = CUPS
        if np.any(K.in_cup(cup_pts[a], *cups[b], TOUCH)) or np.any(K.in_cup(cup_pts[b], *cups[a], TOUCH)):
            if s["dirty"][a] != s["dirty"][b]:
                _soil(s, a if s["dirty"][b] else b, "a contaminated cup touched it")
    # where things have ended up
    rect, top = AREAS["field"]
    shrunk = (rect[0] + K.C_TOP_R, rect[1] - K.C_TOP_R, rect[2] + K.C_TOP_R, rect[3] - K.C_TOP_R)
    for n in CUPS:
        pos, quat = cups[n]
        standing = bool(K.cup_tilt(quat) < UPRIGHT and abs(pos[2] - TOP - top) < 0.003 and _inside(pos, shrunk)[0])
        let_go = bool(not np.any(K.in_cup(arm, pos, quat, CLEAR)) and not np.any(K.in_cup(f["tip"], pos, quat, TIP_CLEAR)))
        s["on"][n] = bool(standing and let_go and speeds[n] < STILL and not s["dirty"][n])
    s["cups_on"] = sum(s["on"].values())
    s["down"] = bool(not held and speeds["forceps"] < STILL)
    resting = any(bool(np.all(over(work, area))) and touches(work, area) and not s["dirty"][area] for area in AREAS)
    s["ready"] = bool(s["down"] and resting and not s["dirty"]["forceps"])
    return s


def score_from(field_clean: bool, cups_on: int, down: bool, ready: bool) -> float:
    """Pure scoring rule. ``field_clean``: the sterile field has not been contaminated. ``cups_on``: how many cups
    stand on the field's sterile part, uncontaminated, let go and at rest. ``down``: the forceps have been put down
    and let go and are at rest. ``ready``: ... uncontaminated, with their working end resting on a sterile area."""
    if not field_clean or cups_on <= 0:
        return 0.0
    if cups_on < 2:
        return 0.25
    if not down:
        return 0.5
    return 1.0 if ready else 0.75


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = new_state()


def _clear() -> None:
    _STATE.clear()
    _STATE.update(new_state())


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    st = object_states(environment, [FORCEPS, BLADE, *CUPS])
    blades = tuple((st[n].pos, st[n].quat_xyzw) for n in (FORCEPS, BLADE))
    cups = {n: (st[n].pos, st[n].quat_xyzw) for n in CUPS}
    speeds = {n: float(np.linalg.norm(st[n].lin_vel)) for n in CUPS}
    speeds["forceps"] = max(float(np.linalg.norm(st[n].lin_vel)) for n in (FORCEPS, BLADE))
    advance(_STATE, K.arm_points(environment), blades, cups, speeds)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial from what was touched during it. Touching means within 1.5 mm (4 mm for any
    part of an arm). The sterile things are the two cups, the forceps' working end (the 5 cm from the tips) and the
    tops of the field and of the pack's wrapper more than 2.5 cm from their edges. A sterile thing is contaminated
    when it touches the table, a 2.5 cm border, the tray's side, an arm, the forceps' handle (the 6.5 cm from the
    tail) or anything already contaminated; the field or the pack also when any part of an arm is above its sterile
    part, at any height; the forceps also when, while they are held (an arm within 3 cm of them), their tips are
    more than 5 mm higher than the end of their handle.
    0 if the field is contaminated. Otherwise 0.25 for one cup, 0.5 for both, that stand on the field's sterile part
    (upright within 10 degrees, the whole cup inside the border), uncontaminated, let go (both arms 3 cm away, the
    forceps' tips 2 cm away) and at rest (under 2 cm/s). 0.75: both cups, and the forceps put down, let go (both
    arms 3 cm away) and at rest. 1.0: both cups, and the forceps put down uncontaminated with their whole working
    end resting on the sterile part of the field or of the pack."""
    s = _STATE
    return score_from(not s["dirty"]["field"], s["cups_on"], s["down"], s["ready"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    return {"score": score(objects), "lot": LOT, "dirty": dict(s["dirty"]), "why": dict(s["why"]),
            "cups_on": s["cups_on"], "on": dict(s["on"]), "forceps_down": s["down"], "forceps_ready": s["ready"],
            "tips_up_mm": s["tips_up_mm"], "tip_gap_mm": s["gap_mm"], "arm_over": s["arm_over"]}


def demo(objects=None, seed=None):
    return {}
