"""On the station table: a small folding camera drone, unfolded with its propellers on, stands on a landing pad.

The job is the compass calibration its maker's manual asks for before flying at a new site: pick the drone up, hold
it level and turn it one full turn about the vertical; then hold it vertically and turn it one full turn about the
vertical again; then put it back on its pad. The drone is never put down in between. One wrist turns about two thirds
of a circle, so each full turn needs the drone passed from one hand to the other in the air.

Where the pad is, and which way the drone faces on it, change with ``DRONE_LOT`` (A or B).

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

import dronekit as K  # noqa: E402
from dronekit import Drone, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Calibrate the drone's compass: pick it up, hold it level and turn it one full turn, then hold it "
               "vertically, nose down or nose up, and turn it one full turn. Do not put it down in between. Then stand "
               "it back on its pad.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("DRONE_LOT", "A")
# pad: the centre of the landing pad. yaw: which way the drone's nose points at the start, degrees round from straight
# away from the robot toward its left.
LAYOUT = {
    "A": {"pad": (0.260, -0.070), "yaw": -90.0},
    "B": {"pad": (0.235, -0.100), "yaw": 180.0},
}[LOT]
PAD_XY, YAW = LAYOUT["pad"], math.radians(LAYOUT["yaw"])
PAD_SIZE, PAD_T = 0.240, 0.004     # the pad: a square mat this wide and this thick
GAP = 0.0004                       # the drone starts this far above the pad

DRONE = "drone"
PAD = "landing_pad"

# -- scoring -----------------------------------------------------------------------------------------------------------
BINS = 36                # a full turn is every one of 36 ten-degree compass directions
LEVEL_TILT = 20.0        # "level": the drone's top within this many degrees of straight up
PLUMB_TILT = 20.0        # "upright on end": its nose-to-tail line within this many degrees of plumb
AIR = 0.020              # "held": its lowest point at least this far above the table
DROP_SPEED = 0.25        # m/s: coming within AIR of the table faster than this is a drop
STAND_TILT = 10.0        # standing: its top within this many degrees of straight up ...
ON_SURFACE = 0.003       # ... and its lowest point within this of the pad's top or of the table
CLEAR = 0.030            # both grippers this far from the drone for it to count as let go
STILL, STILL_TURN = 0.02, 0.5    # m/s and rad/s
HOLD = 20                # control steps (1 s) the full-score state has to last before the trial ends


def drone_start() -> tuple:
    """The drone's position (the middle of its fuselage) and quaternion at the start."""
    return np.array([PAD_XY[0], PAD_XY[1], TOP + PAD_T + K.STAND + GAP]), K._quat(K.standing_rot(YAW))


def _objects() -> list:
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    pos, quat = drone_start()
    return [Box(PAD, pos=(PAD_XY[0], PAD_XY[1], TOP + PAD_T / 2), size=(PAD_SIZE, PAD_SIZE, PAD_T), color=K.PAD_ORANGE),
            Drone(DRONE, pos=tup(pos), quat=quat, size=(K.LEN, K.SPAN, K.TALL), mass=K.MASS)]


OBJECTS = _objects()


# -- the turns, followed from the drone's measured motion -----------------------------------------------------------------
class Turns:
    """Follows one trial. A turn is counted over one unbroken hold: from when the drone's lowest point rises 2 cm above
    the table until it comes back within 2 cm of it. In a hold, the level turn is done when the nose has pointed in
    all 36 ten-degree compass directions while the drone's top was within 20 degrees of straight up; after that, the
    upright turn is done when its top has faced all 36 directions while its nose-to-tail line was within 20 degrees
    of plumb. Coming down ends the hold: a turn not finished by then starts again from nothing in the next hold."""

    def __init__(self) -> None:
        self.level: set = set()
        self.plumb: set = set()
        self.level_now = self.plumb_now = False    # done in the present hold
        self.level_done = self.both_done = False   # done in some hold (kept)
        self.held = False
        self.dropped = False

    @staticmethod
    def bin(direction_deg: float) -> int:
        return int((direction_deg % 360.0) // (360.0 / BINS)) % BINS

    def step(self, low: float, speed: float, att: dict) -> None:
        """``low``: the drone's lowest point above the table (m). ``speed``: its speed (m/s). ``att``: see
        ``dronekit.attitude``."""
        if low < AIR:
            if self.held and speed > DROP_SPEED:
                self.dropped = True
            self.held = False
            if not (self.level_now and self.plumb_now):
                self.level, self.plumb, self.level_now, self.plumb_now = set(), set(), False, False
            return
        self.held = True
        if not self.level_now:
            if att["up_tilt"] <= LEVEL_TILT:
                self.level.add(self.bin(att["nose"]))
                if len(self.level) == BINS:
                    self.level_now = self.level_done = True
        elif not self.plumb_now:
            if att["plumb_tilt"] <= PLUMB_TILT:
                self.plumb.add(self.bin(att["top"]))
                if len(self.plumb) == BINS:
                    self.plumb_now = self.both_done = True


def standing(low: float, up_tilt: float, speed: float, spin: float) -> bool:
    """The drone stands on its feet, at rest: its top within 10 degrees of straight up, its lowest point within 3 mm
    of the table or of the pad's top, under 2 cm/s and 0.5 rad/s."""
    on = low <= ON_SURFACE or abs(low - PAD_T) <= ON_SURFACE
    return bool(on and up_tilt <= STAND_TILT and speed < STILL and spin < STILL_TURN)


def on_pad(feet_xy) -> bool:
    """All four feet are over the pad."""
    half = PAD_SIZE / 2
    return bool(all(abs(f[0] - PAD_XY[0]) <= half and abs(f[1] - PAD_XY[1]) <= half for f in feet_xy))


def score_from(level_done: bool, both_done: bool, dropped: bool, stands: bool, pad: bool, clear: bool) -> float:
    """Pure scoring rule. ``level_done``: the level turn was finished in one hold. ``both_done``: the level turn and
    then the upright turn were finished in the same hold. ``dropped``: the drone came within 2 cm of the table faster
    than 0.25 m/s at any time. ``stands``, ``pad``, ``clear``: at the end the drone stands on its feet at rest, all
    four feet are over the pad, both grippers are at least 3 cm from it.
    0.25 for the level turn; 0.5 for both turns; 0.75 for both turns with the drone standing at the end and never
    dropped; 1.0 if it also stands on its pad and has been let go."""
    if not level_done:
        return 0.0
    if not both_done:
        return 0.25
    if dropped or not stands:
        return 0.5
    return 1.0 if (pad and clear) else 0.75


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, turns=Turns(), low=0.0, att={"up_tilt": 0.0, "plumb_tilt": 90.0, "nose": 0.0, "top": 0.0},
                  stands=True, pad=True, clear=True, near=0.0, score=0.0, held=0, level_at=None, both_at=None)


_clear()


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    d = object_states(environment, [DRONE])[DRONE]
    t: Turns = s["turns"]
    low = float(d.low[2] - TOP)
    att = K.attitude(d.quat_xyzw)
    speed, spin = float(np.linalg.norm(d.lin_vel)), float(np.linalg.norm(d.ang_vel))
    was = (t.level_done, t.both_done)
    t.step(low, speed, att)
    if t.level_done and not was[0]:
        s["level_at"] = s["steps"]
    if t.both_done and not was[1]:
        s["both_at"] = s["steps"]
    hands = np.array([p for side in K.gripper_points(environment) for p in side])
    near = K.drone_distance(hands, d.pos, d.quat_xyzw)
    feet = [K.place(d.pos, d.quat_xyzw, f)[:2] for f in K.FEET]
    s.update(low=low, att=att, near=near, stands=standing(low, att["up_tilt"], speed, spin), pad=on_pad(feet),
             clear=bool(near >= CLEAR))
    s["score"] = score_from(t.level_done, t.both_done, t.dropped, s["stands"], s["pad"], s["clear"])
    s["held"] = s["held"] + 1 if s["score"] >= 1.0 else 0


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """A turn is counted over one unbroken hold, from when the drone's lowest point rises 2 cm above the table until
    it comes back within 2 cm of it. The level turn is done when the nose has pointed in all 36 ten-degree compass
    directions while the drone's top was within 20 degrees of straight up; after it, in the same hold, the upright
    turn is done when its top has faced all 36 directions while its nose-to-tail line was within 20 degrees of
    plumb. 0.25 for the level turn; 0.5 for both; 0.75 if the drone then stands on its feet at rest at the end and
    was never dropped (never came within 2 cm of the table faster than 0.25 m/s); 1.0 if it also stands with all
    four feet on its pad and both grippers are at least 3 cm from it."""
    return float(_STATE["score"])


def success(objects) -> bool:
    return _STATE["held"] >= HOLD


def report(objects) -> dict:
    s = _STATE
    t: Turns = s["turns"]
    a = s["att"]
    return {"score": score(objects), "lot": LOT, "level_bins": len(t.level), "plumb_bins": len(t.plumb),
            "level_done": t.level_done, "both_done": t.both_done, "dropped": t.dropped, "held": t.held,
            "level_at": s["level_at"], "both_at": s["both_at"],
            "stands": s["stands"], "on_pad": s["pad"], "clear": s["clear"], "near_mm": round(1000 * float(s["near"]), 1),
            "low_mm": round(1000 * float(s["low"]), 1), "up_tilt": round(a["up_tilt"], 1), "plumb_tilt": round(a["plumb_tilt"], 1),
            "nose": round(a["nose"], 1), "top": round(a["top"], 1)}


def demo(objects=None, seed=None):
    return {}
