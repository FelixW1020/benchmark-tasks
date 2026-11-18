"""On the station table (standing in for a kitchen counter): a walnut cutting board with a 20 cm chef's knife lying on
it on its side, handle to the left, cutting edge toward the robot. The knife's plastic edge guard (a folded PVC
channel, 21.6 cm long and 4.9 cm wide) stands on its open side in front of the board, leaning against the board's
front edge with its folded edge up.

The job is the one a cook does before a knife goes into a drawer or a knife roll: put the guard on. The guard's slot
has to come onto the bare edge until the whole edge, heel to tip, is inside it and the edge sits at the slot's
bottom. The knife is then left lying on the board.

No part of either arm may come within 1 cm of any part of the cutting edge that is not inside the guard, at any time.

The knife's and the guard's places change with ``GUARD_LOT`` (A or B).

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

import guardkit as K  # noqa: E402
from guardkit import Guard, Knife, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Put the edge guard on the chef's knife without touching the knife's cutting edge, then leave the "
               "knife lying on the cutting board.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("GUARD_LOT", "A")
BOARD = (0.305, 0.457, 0.038)                    # the cutting board: depth (x), width (y), thickness (12 x 18 x 1.5 in)
BOARD_X0 = 0.215                                 # x of its front edge
BOARD_RECT = (BOARD_X0, BOARD_X0 + BOARD[0], -BOARD[1] / 2, BOARD[1] / 2)
BOARD_TOP = BOARD[2]
HEEL_XY = {"A": (0.285, 0.055), "B": (0.293, 0.040)}[LOT]      # the knife's heel, on its edge line
GUARD_Y0 = {"A": 0.060, "B": 0.035}[LOT]         # y of the guard's heel end (the end nearer the knife's handle)
GUARD_LEAN = math.radians(8.0)                   # the guard leans this far from upright, its top toward the board
WALNUT = (0.42, 0.29, 0.19)

KNIFE, GUARD = "knife", "guard"

# -- scoring -----------------------------------------------------------------------------------------------------------
EDGE_NEAR = 0.010        # no part of an arm may come this near a bare part of the edge
PART = 0.25              # this share of the edge inside the guard counts as "partly on"
SEAT = 0.004             # seated: the straight part of the edge within this of the slot's bottom
CLEAR = 0.030            # both arms this far from the knife and the guard for them to be let go
STILL, TURN = 0.02, 0.5  # at rest: under this many m/s and rad/s
LIE = 0.004              # lying on the board: the knife's lowest point within this of the board's top


def knife_start() -> tuple:
    """Where the knife is at the start (its origin and quaternion): on its side, handle to the left, edge toward the
    robot, its handle flat on the board."""
    pos = np.array([HEEL_XY[0], HEEL_XY[1], TOP + BOARD_TOP + K.HANDLE_T / 2 + 0.0005])
    return pos, K.turn_z(-math.pi / 2)


def guard_frame(lean: float = GUARD_LEAN) -> np.ndarray:
    """The leaning guard's axes in the world (columns): its length along -y, its fold up, its top toward the board."""
    return np.column_stack([(0.0, -1.0, 0.0), (-math.sin(lean), 0.0, -math.cos(lean)), (math.cos(lean), 0.0, -math.sin(lean))])


def guard_start() -> tuple:
    """Where the guard is at the start (its origin and quaternion): standing on its open side on the table, leaning
    against the top of the board's front edge."""
    s, c = math.sin(GUARD_LEAN), math.cos(GUARD_LEAN)
    z = K.GW * c + K.G_T / 2 * s + 0.0005
    x = BOARD_X0 - (K.G_T / 2 + (BOARD_TOP - z) * s) / c - 0.0003
    return np.array([x, GUARD_Y0, TOP + z]), K._quat(guard_frame())


def _objects() -> list:
    board = Box("board", pos=(BOARD_X0 + BOARD[0] / 2, 0.0, TOP + BOARD_TOP / 2), size=BOARD, color=WALNUT)
    kp, kq = knife_start()
    gp, gq = guard_start()
    knife = Knife(KNIFE, pos=tuple(float(v) for v in kp), size=(K.BL + K.HL, K.BH, K.HANDLE_T), quat=kq)
    guard = Guard(GUARD, pos=tuple(float(v) for v in gp), size=(K.GL, K.GW, K.G_T), quat=gq)
    return [board, knife, guard]


OBJECTS = _objects()


# -- where things are (pure) ---------------------------------------------------------------------------------------------

def _apart(a, b) -> float:
    """The least distance between two sets of points."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if not len(a) or not len(b):
        return math.inf
    best = math.inf
    for i in range(0, len(a), 256):
        d = np.linalg.norm(a[i:i + 256, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def on_board(points) -> bool:
    """Are all these points over the board?"""
    p = np.atleast_2d(points)
    r = BOARD_RECT
    return bool(np.all((p[:, 0] >= r[0]) & (p[:, 0] <= r[1]) & (p[:, 1] >= r[2]) & (p[:, 1] <= r[3])))


def new_state() -> dict:
    return {"steps": 0, "touched": False, "why": "", "nearest_mm": None, "covered": 0.0, "all": False,
            "seated": False, "edge_mm": None, "lying": False, "let_go": False, "still": False}


def advance(s: dict, arm, knife, guard, speeds) -> dict:
    """One control step of the record. ``arm``: points on both arms (world). ``knife``, ``guard``: (pos, quat).
    ``speeds``: {name: (m/s, rad/s)} for the knife and the guard."""
    s["steps"] += 1
    arm = np.atleast_2d(arm)
    edge = K.edge_points(*knife)
    inside, loc = K.in_slot(edge, *guard)
    # the bare edge: no part of an arm may come near it
    near = _apart(arm, edge[~inside])
    if near < math.inf and (s["nearest_mm"] is None or 1000 * near < s["nearest_mm"]):
        s["nearest_mm"] = round(1000 * near, 1)
    if near < EDGE_NEAR and not s["touched"]:
        s["touched"] = True
        s["why"] = f"step {s['steps']}: an arm came within {1000 * near:.1f} mm of the bare edge"
    # how far the guard is on
    s["covered"] = round(float(inside.mean()), 3)
    s["all"] = bool(inside.all())
    deep = float(loc[K.STRAIGHT_EDGE, 1].max()) - K.G_FOLD            # the straight edge's worst distance from the slot's bottom
    s["edge_mm"] = round(1000 * deep, 1) if s["all"] else None
    s["seated"] = bool(s["all"] and deep <= SEAT)
    # how things have been left
    kp, gp = K.knife_points(*knife), K.guard_points(*guard)
    low = float(kp[:, 2].min()) - TOP - BOARD_TOP
    s["lying"] = bool(on_board(kp) and on_board(gp) and abs(low) <= LIE)
    s["let_go"] = bool(_apart(arm, np.vstack([kp, gp])) >= CLEAR)
    s["still"] = bool(all(speeds[n][0] < STILL and speeds[n][1] < TURN for n in (KNIFE, GUARD)))
    return s


def score_from(touched: bool, covered: float, whole: bool, seated: bool, lying: bool, let_go: bool, still: bool) -> float:
    """Pure scoring rule. ``touched``: an arm came within 1 cm of a bare part of the edge at some time in the trial.
    ``covered``: the share of the edge that is inside the guard's slot at the end. ``whole``: all of it is.
    ``seated``: ... and the straight part of the edge is within 4 mm of the slot's bottom. ``lying``: the knife lies
    on the board, with the guard, nothing over the board's edge. ``let_go``: both arms are 3 cm from the knife and
    the guard. ``still``: both are at rest."""
    if touched:
        return 0.0
    if whole and seated:
        return 1.0 if (lying and let_go and still) else 0.75
    if whole:
        return 0.5
    return 0.25 if covered >= PART else 0.0


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

    st = object_states(environment, [KNIFE, GUARD])
    poses = {n: (st[n].pos, st[n].quat_xyzw) for n in (KNIFE, GUARD)}
    speeds = {n: (float(np.linalg.norm(st[n].lin_vel)), float(np.linalg.norm(st[n].ang_vel))) for n in (KNIFE, GUARD)}
    advance(_STATE, K.arm_points(environment), poses[KNIFE], poses[GUARD], speeds)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial; the rule about the bare edge is watched through the whole trial.
    The edge is the cutting edge from 4 mm in front of the heel to the tip. A part of it is covered when it is inside
    the guard's slot (between the two walls, within the guard's length); otherwise it is bare.
    0 if any part of either arm (upper arm to fingertips) came within 1 cm of a bare part of the edge at any time.
    Otherwise, from the end state: 0.25 if at least a quarter of the edge is covered; 0.5 if all of it is; 0.75 if
    all of it is and the guard is seated (the straight 9 cm of the edge next to the heel within 4 mm of the slot's
    bottom, so that the spine stands no more than 11 mm out of the guard at the heel); 1.0 if, as well, the knife
    lies on the board (its lowest point within 4 mm of the board's top, no part of the knife or the guard over the
    board's edge), both arms are at least 3 cm from the knife and the guard, and both are at rest (under 2 cm/s and
    0.5 rad/s)."""
    s = _STATE
    return score_from(s["touched"], s["covered"], s["all"], s["seated"], s["lying"], s["let_go"], s["still"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    return {"score": score(objects), "lot": LOT, "touched_edge": s["touched"], "why": s["why"],
            "nearest_to_bare_edge_mm": s["nearest_mm"], "edge_covered": s["covered"], "whole_edge_covered": s["all"],
            "edge_from_slot_bottom_mm": s["edge_mm"], "seated": s["seated"], "lying_on_board": s["lying"],
            "let_go": s["let_go"], "at_rest": s["still"]}


def demo(objects=None, seed=None):
    return {}
