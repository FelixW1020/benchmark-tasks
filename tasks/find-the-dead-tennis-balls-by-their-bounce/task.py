"""On the station table: four tennis balls in a row in a shallow wooden tray, a grey bin and a blue box. Two of the
balls have lost their pressure ("gone flat", "dead"); the other two are good. All four look and weigh the same. They
differ in one thing: dropped onto the table, a good ball comes back 56% of the height it fell (the rule of tennis:
135 to 147 cm from a 254 cm drop), a flat one 36% (a punctured ball keeps 80% of its coefficient of restitution).

The job is to find the two flat balls by their bounce, put them in the grey bin, and put the two good ones in the blue
box. Each ball carries one to four pen dots, so that whoever scores the trial can tell which ball is which; which two
are flat changes with ``BALL_LOT`` (A or B), and so do the places of the tray, the bin and the box.

Run it:    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:      python demo.py task.py out_dir
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ballkit as K  # noqa: E402
from ballkit import Ball, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Two of these four tennis balls have gone flat. Find them by their bounce: put the two flat ones in the "
               "grey bin and the two good ones in the blue box.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("BALL_LOT", "A")
# the middle of the tray (its long side runs away from the robot), of the bin and of the box, and which balls are flat
LAYOUT = {"A": ((0.290, -0.260), (0.125, -0.100), (0.280, 0.150), (2, 3)),
          "B": ((0.280, -0.250), (0.115, -0.085), (0.290, 0.160), (1, 4))}[LOT]
TRAY_XY, BIN_XY, BOX_XY, FLAT = LAYOUT
TRAY_YAW = BIN_YAW = BOX_YAW = 0.0
GAP = 0.0004                       # the balls start this far above the tray's floor

BALLS = tuple(f"ball_{n}" for n in (1, 2, 3, 4))          # ball_n carries n pen dots and starts in place n, nearest first
DEAD = tuple(f"ball_{n}" for n in FLAT)
GOOD = tuple(b for b in BALLS if b not in DEAD)

# -- scoring -----------------------------------------------------------------------------------------------------------
CLEAR = 0.030            # "let go": both grippers at least this far from the ball
STILL = 0.02             # "at rest": m/s


def ball_start(k: int) -> np.ndarray:
    """Where ball ``k`` (0 to 3) starts (world)."""
    return K.tray_slot(TRAY_XY, TRAY_YAW, k) + np.array([0.0, 0.0, TOP + GAP])


def _objects() -> list:
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    objs = []
    for tag, centre, size in K.tray_boxes(TRAY_XY, TRAY_YAW):
        objs.append(Box(f"tray_{tag}", pos=tup(centre + [0.0, 0.0, TOP]), size=tup(size), color=K.WOOD, yaw=TRAY_YAW))
    for tag, centre, size in K.bin_boxes(BIN_XY, BIN_YAW):
        objs.append(Box(f"bin_{tag}", pos=tup(centre + [0.0, 0.0, TOP]), size=tup(size), color=K.BIN_COLOR, yaw=BIN_YAW))
    for tag, centre, size in K.bin_boxes(BOX_XY, BOX_YAW):
        objs.append(Box(f"box_{tag}", pos=tup(centre + [0.0, 0.0, TOP]), size=tup(size), color=K.BOX_COLOR, yaw=BOX_YAW))
    for k, name in enumerate(BALLS):
        bounce = K.BOUNCE_DEAD if name in DEAD else K.BOUNCE_GOOD
        objs.append(Ball(name, pos=tup(ball_start(k)), size=(K.BALL_D,) * 3, bounce=bounce, dots=k + 1))
    return objs


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, where={b: "tray" for b in BALLS}, free={b: True for b in BALLS},
                  near={b: 1.0 for b in BALLS}, speed={b: 0.0 for b in BALLS})


_clear()


def place_of(pos) -> str:
    """Where a ball is, from its middle (world): ``bin`` or ``box`` (over the inside of that one and below its rim),
    ``tray`` (over the inside of the tray, no higher than a ball resting on another), or ``elsewhere``."""
    z = float(pos[2]) - TOP
    if z <= K.BIN_FLOOR + K.BIN_RIM:
        if K.inside(pos, BIN_XY, BIN_YAW, K.BIN_INNER):
            return "bin"
        if K.inside(pos, BOX_XY, BOX_YAW, K.BIN_INNER):
            return "box"
        if K.inside(pos, TRAY_XY, TRAY_YAW, K.TRAY_INNER):
            return "tray"
    return "elsewhere"


def tally(where: dict, free: dict) -> tuple:
    """(balls in their right place, balls in the wrong one of the bin and the box), counting only balls that have
    been let go and are at rest. A flat ball's right place is the bin, a good ball's the box."""
    right = sum(1 for b in BALLS if free[b] and where[b] == ("bin" if b in DEAD else "box"))
    wrong = sum(1 for b in BALLS if free[b] and where[b] == ("box" if b in DEAD else "bin"))
    return right, wrong


def score_from(right: int, wrong: int) -> float:
    """Pure scoring rule, from counts of balls that have been let go and are at rest. ``right``: flat balls in the bin
    plus good balls in the box (0 to 4). ``wrong``: flat balls in the box plus good balls in the bin."""
    if wrong > 0:
        return 0.0
    return 0.25 * right


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    st = object_states(environment, list(BALLS))
    hands = np.array([p for side in K.arm_points(environment) for p in side])
    for b in BALLS:
        pos = st[b].pos
        s["where"][b] = place_of(pos)
        s["near"][b] = float(np.linalg.norm(hands - pos, axis=1).min()) - K.BALL_R
        s["speed"][b] = float(np.linalg.norm(st[b].lin_vel))
        s["free"][b] = bool(s["near"][b] >= CLEAR and s["speed"][b] < STILL)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial, from where each ball has been left (let go: both grippers at least 3 cm from
    it; at rest: under 2 cm/s). 0 if any ball is in the wrong one of the two (a flat ball in the box, or a good ball
    in the bin). Otherwise 0.25 for each ball in its right place: a flat ball in the bin, a good ball in the box. A
    ball still in the tray, on the table or in a gripper counts for nothing either way."""
    return score_from(*tally(_STATE["where"], _STATE["free"]))


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    return {"score": score(objects), "lot": LOT, "flat": list(DEAD), "where": dict(s["where"]), "free": dict(s["free"]),
            "near_mm": {b: round(1000 * v, 1) for b, v in s["near"].items()},
            "speed_mm_s": {b: round(1000 * v, 1) for b, v in s["speed"].items()}}


def demo(objects=None, seed=None):
    return {}
