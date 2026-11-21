"""On the station table: a full-size bench section of a dry-van trailer's right-hand rear door, standing upright and
facing the robot. The door is shut but not locked. Its cam-action lock rod (a 3/4 in pipe in two guides) carries the
handle on a hub and, at its lower end, the cam; the cam's keeper pin is on the sill. The handle hangs swung out toward
the robot, and the hasp's flip keeper is flipped up, as the last driver left them. To the robot's left are the seam
between the two doors and the neighbouring door's edge with its own rod, already locked.

The job: swing the handle in so that the cam takes hold of its keeper pin, lift the handle over the hasp's lip while
pushing it against the door seal's spring-back, let it down into the hasp's seat, and flip the keeper down over it.
Then let go.

A keeper that is down while the handle is not in its seat voids the trial: the door then looks locked and is not.

The fixture's place and the handle's starting angle change with ``DOOR_LOT`` (A or B).

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

import doorkit as K  # noqa: E402
from doorkit import Door, Handle, Keeper, Rod, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Lock this trailer door: swing the lock rod's handle in until it is flat against the door, drop it into "
               "its hasp, and flip the hasp's keeper down over it. Then let go.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("DOOR_LOT", "A")
ORIGIN = np.array({"A": (0.520, 0.190), "B": (0.535, 0.175)}[LOT])      # the door's face, at the lock rod's axis
THETA_START = float(os.environ.get("DOOR_THETA", {"A": 60.0, "B": 45.0}[LOT]))   # degrees the handle starts from locked
FLIP_START = float(os.environ.get("DOOR_FLIP", K.FLIP_OPEN))            # the keeper starts flipped up (0: down, for tests)

DOOR, ROD, HANDLE, KEEPER = "door", "rod", "handle", "keeper"

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_GAP = 0.006                  # in its seat: the bar's door-side face this close to the hasp's face, at the hasp
SEAT_RISE = 0.003                 # ... and its underside this close to the seat
KEEP_DOWN = 10.0                  # degrees from hanging: the keeper is down
CLEAR = 0.030                     # both grippers this far from the handle and the keeper
STILL = 0.05                      # rad/s: the rod and the handle at rest
KEEP_STILL = 0.3                  # rad/s: the keeper at rest


def door_pos() -> np.ndarray:
    return np.array([ORIGIN[0], ORIGIN[1], TOP])


def _objects() -> list:
    d = door_pos()
    rp, rq = K.rod_pose(d, THETA_START)
    hp, hq = K.handle_pose(d, THETA_START, 0.0)
    kp, kq = K.keeper_pose(d, FLIP_START)
    t = lambda v: tuple(float(x) for x in v)   # noqa: E731
    return [Door(DOOR, pos=t(d), size=(0.12, 0.67, 0.34)),
            Rod(ROD, pos=t(rp), quat=t(rq), size=(0.09, 0.09, 0.32), door=DOOR),
            Handle(HANDLE, pos=t(hp), quat=t(hq), size=(0.010, 0.356, 0.025), rod=ROD),
            Keeper(KEEPER, pos=t(kp), quat=t(kq), size=(0.020, 0.028, 0.053), door=DOOR)]


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, theta=THETA_START, lift=0.0, flip=FLIP_START, cam=False, seated=False, down=False,
                  released=False, still=False, gap=1.0, least_theta=THETA_START)


_clear()


def seated(theta_deg: float, lift_deg: float) -> bool:
    """The handle is in the hasp's seat: flat against the hasp's face (within 6 mm) and down (within 3 mm)."""
    return bool(K.seat_gap(theta_deg) <= SEAT_GAP and K.seat_rise(lift_deg) <= SEAT_RISE)


def score_from(cam: bool, seated: bool, down: bool, released: bool, still: bool) -> float:
    """Pure scoring rule. ``cam``: the keeper pin on the sill is between the cam's prongs. ``seated``: the handle is in
    the hasp's seat. ``down``: the hasp's keeper is flipped down. ``released``: both grippers are clear of the handle
    and the keeper. ``still``: nothing is moving."""
    if down and not seated:
        return 0.0                               # looks locked, is not
    if seated and down:
        return 1.0 if (released and still) else 0.75
    if seated:
        return 0.5
    return 0.25 if cam else 0.0


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    st = _STATE
    st["steps"] += 1
    obj = object_states(environment, [ROD, HANDLE, KEEPER])
    rod, handle, keeper = obj[ROD], obj[HANDLE], obj[KEEPER]
    st["theta"], st["lift"], st["flip"] = K.angles(rod.quat_xyzw, handle.quat_xyzw, keeper.quat_xyzw)
    st["least_theta"] = min(st["least_theta"], st["theta"])
    st["cam"] = bool(K.cam_engaged(st["theta"]))
    st["seated"] = seated(st["theta"], st["lift"])
    st["down"] = bool(st["flip"] <= KEEP_DOWN)
    hands = np.array([p for side in K.gripper_points(environment) for p in side])
    rot_h, rot_k = K._rotation(handle.quat_xyzw), K._rotation(keeper.quat_xyzw)
    tip = handle.pos + rot_h @ np.array([0.0, -(K.HANDLE_L - K.BAR_BACK), 0.0])
    end = keeper.pos + rot_k @ np.array([0.0, 0.0, -K.KEEP_L])
    st["gap"] = min(K.segment_distance(hands, handle.pos, tip) - K.BAR_H / 2,
                    K.segment_distance(hands, keeper.pos, end) - K.KEEP_W / 2)
    st["released"] = bool(st["gap"] >= CLEAR)
    st["still"] = bool(float(np.linalg.norm(rod.ang_vel)) < STILL and float(np.linalg.norm(handle.ang_vel)) < STILL
                       and float(np.linalg.norm(keeper.ang_vel)) < KEEP_STILL)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial, from where the parts are then. 0.25: the cam has hold of its keeper pin (the
    pin is between the cam's two prongs; the rod within 25 degrees of locked). 0.5: the handle is in the hasp's seat
    (its door-side face within 6 mm of the hasp's face and its underside within 3 mm of the seat, both at the hasp)
    and the keeper is not down. 0.75: the handle is in its seat and the keeper is down over it (within 10 degrees of
    hanging), but a gripper is within 3 cm of the handle or the keeper, or something is still moving. 1.0: handle in
    its seat, keeper down, both grippers at least 3 cm away, everything at rest. The trial scores 0 if the keeper is
    down while the handle is not in its seat, and 0 if the cam has no hold."""
    st = _STATE
    return score_from(st["cam"], st["seated"], st["down"], st["released"], st["still"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    st = _STATE
    return {"score": score(objects), "lot": LOT, "theta_deg": round(float(st["theta"]), 2),
            "lift_deg": round(float(st["lift"]), 2), "flip_deg": round(float(st["flip"]), 1),
            "seat_gap_mm": round(1000 * K.seat_gap(st["theta"]), 1), "seat_rise_mm": round(1000 * K.seat_rise(st["lift"]), 1),
            "cam": st["cam"], "seated": st["seated"], "keeper_down": st["down"], "released": st["released"],
            "still": st["still"], "gripper_gap_mm": round(1000 * st["gap"], 1),
            "least_theta_deg": round(float(st["least_theta"]), 2)}


def demo(objects=None, seed=None):
    return {}
