"""On the station table, which stands in for a hospital pharmacy's counter: an empty pneumatic tube carrier (the short
swivel-lid carrier for a 110 mm line) standing on its closed end with its lid swung open, and beside it a filled,
capped prescription vial. On the wall behind the counter hangs a bottom-loading tube station: the lower end of its
send tube, with an entry collar, is 30 cm above the counter. The station has already been given its destination and
is waiting for a carrier.

The job: put the vial in the carrier, swing the carrier's lid shut until its catch holds, and push the carrier up into
the station's send tube until the station takes it. Then let go.

The carrier is light and stands on a small end, so it has to be held while its lid is pushed round. A carrier that
goes into the tube with its lid not secured voids the trial: in a real system the lid can come off in the line.

The carrier's and the vial's places change with ``TUBE_LOT`` (A or B).

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

import tubekit as K  # noqa: E402
from tubekit import Carrier, Lid, Station, Vial, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Send this prescription vial to the ward by tube: put it in the carrier, swing the carrier's lid shut "
               "until it catches, then push the carrier up into the station's send tube until the station takes it, "
               "and let go.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("TUBE_LOT", "A")
CARRIER_AT = np.array({"A": (0.270, -0.080), "B": (0.285, -0.065)}[LOT])    # the carrier's axis
VIAL_AT = np.array({"A": (0.280, 0.120), "B": (0.310, 0.100)}[LOT])        # the vial's axis
MOUTH = np.array([0.410, -0.080, 0.300])        # the middle of the send tube's entry collar, at its lower edge
LID_START = float(os.environ.get("TUBE_LID", K.LID_MAX))      # degrees the lid starts open (0: shut, for a control run)
WALL_T = 0.020

STATION, CARRIER, LID, VIAL = "station", "carrier", "lid", "vial"

# -- scoring -----------------------------------------------------------------------------------------------------------
IN_TUBE = 0.020                   # the carrier is in the send tube when its top is this far above the collar's lower edge
ENTERED = 0.010                   # ... and has entered it at this height (for the lid rule)
CLEAR = 0.030                     # both grippers this far from the carrier
STILL = 0.020                     # m/s: the carrier at rest
FLOOR = 0.050                     # a part this far below the counter's top has fallen off it


def station_pos() -> np.ndarray:
    return np.array([MOUTH[0], MOUTH[1], TOP + MOUTH[2]])


def carrier_pos() -> np.ndarray:
    return np.array([CARRIER_AT[0], CARRIER_AT[1], TOP])


def _objects() -> list:
    c, s = carrier_pos(), station_pos()
    lid_pos, lid_quat = K.lid_pose(c, (0.0, 0.0, 0.0, 1.0), LID_START)
    wall_x = MOUTH[0] + K.CAB_BACK + WALL_T / 2
    wall_h = MOUTH[2] + K.CAB_UP + K.CAB_H + 0.06
    return [Box("wall", pos=(float(wall_x), float(MOUTH[1]), TOP + wall_h / 2), size=(WALL_T, 0.60, float(wall_h)),
                color=(0.90, 0.90, 0.87)),
            Station(STATION, pos=tuple(float(v) for v in s), size=(0.22, 0.22, 0.30)),
            Carrier(CARRIER, pos=tuple(float(v) for v in c), size=(0.104, 0.104, 0.216)),
            Lid(LID, pos=tuple(float(v) for v in lid_pos), quat=lid_quat, size=(0.09, 0.09, 0.014), carrier=CARRIER),
            Vial(VIAL, pos=(float(VIAL_AT[0]), float(VIAL_AT[1]), TOP), size=(0.037, 0.037, 0.069))]


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, broken=None, vial_in=False, lid_deg=LID_START, secured=False, up=-1.0, off=1.0, tilt=0.0,
                  in_tube=False, taken=False, released=False, still=False, highest=-1.0, gap=1.0)


_clear()


def score_from(vial_in: bool, secured: bool, in_tube: bool, taken: bool, released: bool, still: bool, broken) -> float:
    """Pure scoring rule. ``vial_in``: the whole vial is inside the carrier. ``secured``: the lid is swung shut and its
    catch holds. ``in_tube``: the carrier's top is in the station's send tube. ``taken``: the station has taken the
    carrier. ``released``: both grippers are clear of the carrier. ``still``: the carrier is at rest. ``broken``: the
    carrier entered the send tube with its lid not secured, or the vial or the carrier fell off the counter."""
    if broken or not vial_in:
        return 0.0
    if not secured:
        return 0.25
    if taken and released and still:
        return 1.0
    return 0.75 if (in_tube or taken) else 0.5


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    st = _STATE
    st["steps"] += 1
    obj = object_states(environment, [CARRIER, LID, VIAL])
    car, lid, vial = obj[CARRIER], obj[LID], obj[VIAL]
    s = station_pos() - np.array([0.0, 0.0, 0.0])
    st["vial_in"] = bool(K.vial_in_carrier(vial.pos, vial.quat_xyzw, car.pos, car.quat_xyzw))
    st["lid_deg"] = K.lid_angle(car.quat_xyzw, lid.quat_xyzw)
    st["secured"] = bool(st["lid_deg"] <= K.SHUT_AT)
    st["up"], st["off"], st["tilt"] = K.insertion(car.pos, car.quat_xyzw, s)
    st["highest"] = max(st["highest"], st["up"])
    st["taken"] = bool(K.held(CARRIER))
    st["in_tube"] = bool(st["up"] >= IN_TUBE and st["off"] <= K.COLLAR_R)
    if st["up"] >= ENTERED and st["off"] <= K.COLLAR_R and not st["secured"] and not st["broken"]:
        st["broken"] = f"the carrier entered the send tube with its lid {st['lid_deg']:.0f} deg open at step {st['steps']}"
    for name, part in ((CARRIER, car), (VIAL, vial)):
        if float(part.pos[2]) < TOP - FLOOR and not st["broken"]:
            st["broken"] = f"the {name} fell off the counter at step {st['steps']}"
    hands = np.array([p for side in K.gripper_points(environment) for p in side])
    st["gap"] = K.body_distance(hands, car.pos, car.quat_xyzw)
    st["released"] = bool(st["gap"] >= CLEAR)
    st["still"] = bool(float(np.linalg.norm(car.lin_vel)) < STILL)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial. 0 unless the whole vial is inside the carrier (both its ends within the bore,
    between the closed end and the mouth's rim). Then 0.25. 0.5 if also the lid is secured: swung shut to within
    3 degrees, where its catch holds it. 0.75 if also the carrier's top is in the station's send tube (2 cm or more
    above the entry collar's lower edge) but the station has not taken it, or a gripper is still within 3 cm of it, or
    it is still moving. 1.0: the station has taken the carrier (its top 7 cm up the send tube, on the tube's axis),
    both grippers are at least 3 cm from it, and it is at rest (under 2 cm/s). The whole trial scores 0 if the
    carrier's top ever enters the send tube (1 cm above the collar's lower edge) with the lid not secured, or if the
    vial or the carrier falls off the counter (5 cm below its top)."""
    st = _STATE
    return score_from(st["vial_in"], st["secured"], st["in_tube"], st["taken"], st["released"], st["still"], st["broken"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    st = _STATE
    return {"score": score(objects), "lot": LOT, "broken": st["broken"], "vial_in": st["vial_in"],
            "lid_deg": round(float(st["lid_deg"]), 1), "secured": st["secured"], "top_up_mm": round(1000 * st["up"], 1),
            "off_axis_mm": round(1000 * st["off"], 1), "tilt_deg": round(math.degrees(st["tilt"]), 1),
            "in_tube": st["in_tube"], "taken": st["taken"], "released": st["released"], "still": st["still"],
            "gripper_gap_mm": round(1000 * st["gap"], 1), "highest_mm": round(1000 * st["highest"], 1)}


def demo(objects=None, seed=None):
    return {}
