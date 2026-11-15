"""Lock a door's deadbolt by its inside thumb-turn while holding the door shut.

A section of an inswing exterior door (solid core, 44 mm thick, 53 cm from its hinge to its latch edge, 45 cm tall,
6.3 kg) hangs on two butt hinges in a fixed frame on the table, its inside face toward the robot. A deadbolt is fitted
60 mm from its latch edge: a thumb-turn on the inside rose turns a cam that throws a 22 mm steel bolt 25 mm out of the
door's edge into the strike in the frame's jamb. The door has no working latch, and the compressed weatherstrip on the
jamb's stop pushes it back off the stop: left alone it stands 12 mm open at its latch edge. Then the bolt meets the
jamb's face beside the strike after 3 mm and the thumb-turn stops. The bolt goes in only while the door is held shut,
within 2 mm of the stop, so one hand has to press the door shut while the other turns the thumb-turn a quarter turn.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechkit as mk  # noqa: E402
import numpy as np  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Lock the door with its deadbolt."

DOOR_WHITE = (0.90, 0.88, 0.83)
FRAME_WOOD = (0.62, 0.47, 0.32)
SATIN_NICKEL = (0.70, 0.71, 0.72)
BRASS = (0.78, 0.64, 0.30)
STEEL = (0.55, 0.56, 0.58)
RUBBER = (0.08, 0.08, 0.09)

# -- layout (world frame: x forward from the arms, y to the robot's left, metres; the door shown closed) -------------
XF = 0.400                  # the door's inside face when closed; its outside face is at XF + 0.044
DOOR_T = 0.044              # 1-3/4 in solid-core exterior door
HINGE_Y = 0.330             # the hinge axis, on the inside face at the door's hinge edge
LATCH_Y = -0.200            # the latch edge (its brass faceplate's outer face)
DOOR_Z = (TOP + 0.016, TOP + 0.466)   # 450 mm tall section, 4 mm above the sill
LATCH_R = HINGE_Y - LATCH_Y            # 0.530 m from the hinge axis to the latch edge
BZ = TOP + 0.200            # height of the bolt and thumb-turn axis
BACKSET = 0.060             # 2-3/8 in backset
TURN_Y = LATCH_Y + BACKSET  # the thumb-turn's axis
BOLT_X = XF + DOOR_T / 2    # the bolt runs in the middle of the door's thickness
BOLT_W = 0.022              # bolt cross-section 22 x 22 mm
BOLT_L = 0.085             # bar and tail
THROW = 0.025               # 1 in throw
JAMB_Y = -0.203             # the strike plate's face: 3 mm gap to the latch edge
HOLE = (0.026, 0.026)       # strike opening (x, z): 2 mm clearance round the bolt
HOLE_BACK_Y = -0.235        # the strike mortise's back, 32 mm deep
STOP_X = XF + DOOR_T + 0.002    # the jamb stop's face, 2 mm off the closed door
DOOR_KG = 6.3               # 0.53 x 0.45 x 0.044 m of solid core at about 600 kg/m^3
TURN_KG = 0.05
BOLT_KG = 0.06
STRIP_KG = 0.03

# Mechanism constants
STRIP_Y = (-0.199, -0.191)  # the weatherstrip bulb runs up the latch jamb's stop, under the door's latch edge
STRIP_R = HINGE_Y - sum(STRIP_Y) / 2     # 0.525 m from the hinge axis
STRIP_OUT = 0.012           # the weatherstrip stands 12 mm proud of the closed door's face when free
STRIP_RATE = 250.0          # N/m
STRIP_PRELOAD = 0.0035      # its spring rest lies 3.5 mm past its free stop: 0.9 N free, 3.9 N with the door shut
DOOR_START_DEG = -math.degrees(STRIP_OUT / STRIP_R)   # the door rests on the free weatherstrip: about 1.3 deg open
HINGE_FRICTION = 0.2        # N*m
TURN_FRICTION = 0.25        # N*m: the lock's friction, felt at the thumb-turn (it stays where it is left)
BOLT_DAMPING = 5.0          # N*s/m: the bolt's slide
TURN_DEG = 90.0             # a quarter turn throws the bolt fully


def _bx(x0, x1, y0, y1, z0, z1, origin, color=None):
    """A box from world extents (reference pose), in the frame of a part whose origin is ``origin``."""
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]) - np.asarray(origin)
    return mk.box((x1 - x0, y1 - y0, z1 - z0), tuple(c), color)


kit = mk.Kit()

# Frame (fixed): sill, hinge jamb, latch jamb with the strike mortise and strike plate, and the stops.
FRAME_O = (0.43, 0.0, TOP)
_hx0, _hx1 = 0.372, 0.480
_zt = TOP + 0.480
_hb = (BZ - HOLE[1] / 2, BZ + HOLE[1] / 2)
_hxa, _hxb = BOLT_X - HOLE[0] / 2, BOLT_X + HOLE[0] / 2
_fb = [
    _bx(_hx0, _hx1, -0.300, 0.370, TOP, TOP + 0.012, FRAME_O),                       # sill
    _bx(_hx0, _hx1, HINGE_Y + 0.004, 0.370, TOP + 0.012, _zt, FRAME_O),               # hinge jamb
    _bx(STOP_X, STOP_X + 0.016, 0.316, HINGE_Y + 0.004, TOP + 0.012, _zt, FRAME_O),  # hinge-side stop
    # latch jamb (behind the strike plate), built round the mortise
    _bx(_hx0, _hx1, -0.300, JAMB_Y - 0.0015, TOP + 0.012, _hb[0], FRAME_O),
    _bx(_hx0, _hx1, -0.300, JAMB_Y - 0.0015, _hb[1], _zt, FRAME_O),
    _bx(_hx0, _hxa, -0.300, JAMB_Y - 0.0015, _hb[0], _hb[1], FRAME_O),
    _bx(_hxb, _hx1, -0.300, JAMB_Y - 0.0015, _hb[0], _hb[1], FRAME_O),
    _bx(_hxa, _hxb, -0.300, HOLE_BACK_Y, _hb[0], _hb[1], FRAME_O),
    # latch-side stop, 2 mm off the closed door's outside face, overlapping its latch edge by 14 mm
    _bx(STOP_X, STOP_X + 0.016, JAMB_Y - 0.0015, LATCH_Y + 0.014, TOP + 0.012, _zt, FRAME_O),
    # wall stubs either side, so the frame reads as a doorway
    _bx(_hx0, _hx1, -0.330, -0.300, TOP, _zt, FRAME_O),
    _bx(_hx0, _hx1, 0.370, 0.400, TOP, _zt, FRAME_O),
]
_fb = [b if b.color else mk.box(b.size, b.at, FRAME_WOOD) for b in _fb]
_strike = [  # steel strike plate, flush on the jamb's face, round the opening
    _bx(XF, XF + DOOR_T, JAMB_Y - 0.0015, JAMB_Y, _hb[1], BZ + 0.040, FRAME_O, STEEL),
    _bx(XF, XF + DOOR_T, JAMB_Y - 0.0015, JAMB_Y, BZ - 0.040, _hb[0], FRAME_O, STEEL),
    _bx(XF, _hxa, JAMB_Y - 0.0015, JAMB_Y, _hb[0], _hb[1], FRAME_O, STEEL),
    _bx(_hxb, XF + DOOR_T, JAMB_Y - 0.0015, JAMB_Y, _hb[0], _hb[1], FRAME_O, STEEL),
]
kit.part("frame", FRAME_O, _fb + _strike, mass=None, color=FRAME_WOOD)

# Door (free, on its hinge): the slab, the latch-edge faceplate, the inside rose and two hinge knuckles.
DOOR_O = (XF, HINGE_Y, TOP)
_db = [
    _bx(XF, XF + DOOR_T, LATCH_Y + 0.001, HINGE_Y, *DOOR_Z, DOOR_O, DOOR_WHITE),
    _bx(XF + 0.006, XF + DOOR_T - 0.006, LATCH_Y, LATCH_Y + 0.001, BZ - 0.033, BZ + 0.033, DOOR_O, BRASS),
    mk.cylinder(0.031, 0.012, tuple(np.array([XF - 0.006, TURN_Y, BZ]) - DOOR_O), SATIN_NICKEL, axis="X"),  # rose
    _bx(XF - 0.007, XF + 0.007, HINGE_Y - 0.007, HINGE_Y + 0.003, TOP + 0.050, TOP + 0.130, DOOR_O, BRASS),
    _bx(XF - 0.007, XF + 0.007, HINGE_Y - 0.007, HINGE_Y + 0.003, TOP + 0.350, TOP + 0.430, DOOR_O, BRASS),
]
kit.part("door", DOOR_O, _db, mass=DOOR_KG, color=DOOR_WHITE)

# Thumb-turn (free, on its spindle through the rose): a wing 42 mm long, 12 mm thick, standing 18 mm out, on a hub; its
# spindle carries a crank inside the door whose pin, 17.7 mm off the axis, runs in a slot in the bolt's tail (the cam).
# A quarter turn swings the pin from 45 degrees on the hinge side of upright to 45 degrees on the latch side, which
# carries the bolt 2 x 17.7 x sin(45) = 25 mm out.
CRANK_R = 0.0125 * math.sqrt(2)     # 17.7 mm
PIN = 0.006                         # round pin, 6 mm across
SLOT = 0.007                        # the bolt's slot is 7 mm wide
TURN_O = (XF - 0.012, TURN_Y, BZ)


def _qx(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def _crank_boxes():
    o = np.asarray(TURN_O)
    d = np.array([0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])   # the crank's direction when upright
    ax = np.array([0.0, TURN_Y, BZ])
    arm_c = ax + d * CRANK_R / 2 + np.array([XF - 0.001, 0, 0])          # crank plate: x XF-4 .. XF+2 mm
    pin_c = ax + d * CRANK_R + np.array([XF + 0.004, 0, 0])              # pin: x XF-2 .. XF+10 mm
    return [
        mk.cylinder(0.004, 0.014, (0.007, 0.0, 0.0), STEEL, axis="X"),                                    # spindle
        mk.box((0.006, 0.006, CRANK_R + 0.006), tuple(arm_c - o), STEEL, quat=_qx(-math.pi / 4)),         # crank
        mk.cylinder(PIN / 2, 0.012, tuple(pin_c - o), STEEL, axis="X"),                                   # pin
    ]


_tb = [
    mk.cylinder(0.010, 0.010, (-0.005, 0.0, 0.0), SATIN_NICKEL, axis="X"),                                # hub
    _bx(XF - 0.040, XF - 0.022, TURN_Y - 0.006, TURN_Y + 0.006, BZ - 0.021, BZ + 0.021, TURN_O, SATIN_NICKEL),
    *_crank_boxes(),
]
kit.part("thumbturn", TURN_O, _tb, mass=TURN_KG, color=SATIN_NICKEL)
WING_X = (XF - 0.040, XF - 0.022)

# Bolt (free, in the door): a 22 x 22 mm steel bar whose end is flush with the faceplate when retracted, with a tail
# reaching past the spindle and the slot for the crank's pin on its inside face.
BOLT_O = (BOLT_X, LATCH_Y + BOLT_L / 2, BZ)
_slot_y = TURN_Y + 0.0125           # the pin's y with the thumb-turn upright: the slot's centre, retracted
_bb = [
    _bx(BOLT_X - BOLT_W / 2, BOLT_X + BOLT_W / 2, LATCH_Y, LATCH_Y + BOLT_L, BZ - BOLT_W / 2, BZ + BOLT_W / 2,
        BOLT_O, STEEL),
    _bx(XF + 0.0035, BOLT_X - BOLT_W / 2, _slot_y - SLOT / 2 - 0.004, _slot_y - SLOT / 2, BZ + 0.006, BZ + 0.025,
        BOLT_O, STEEL),
    _bx(XF + 0.0035, BOLT_X - BOLT_W / 2, _slot_y + SLOT / 2, _slot_y + SLOT / 2 + 0.004, BZ + 0.006, BZ + 0.025,
        BOLT_O, STEEL),
]
kit.part("bolt", BOLT_O, _bb, mass=BOLT_KG, color=STEEL)

# Weatherstrip (free, on a spring in the latch stop's kerf): reference = flush with the closed door's outside face.
STRIP_O = (XF + DOOR_T + 0.006, sum(STRIP_Y) / 2, TOP + 0.24)
kit.part("weatherstrip", STRIP_O, [_bx(XF + DOOR_T, XF + DOOR_T + 0.012, *STRIP_Y, TOP + 0.030, TOP + 0.450,
                                       STRIP_O, RUBBER)], mass=STRIP_KG, color=RUBBER)

kit.hinge("door_hinge", "frame", "door", anchor=(XF, HINGE_Y, TOP), axis=(0, 0, 1), lower_deg=-100.0, upper_deg=0.0,
          start_deg=DOOR_START_DEG, damping=0.3, friction=HINGE_FRICTION)
kit.hinge("turn", "door", "thumbturn", anchor=TURN_O, axis=(1, 0, 0), lower_deg=-2.0, upper_deg=TURN_DEG + 2.0,
          damping=0.002, friction=TURN_FRICTION)
kit.slider("bolt_slide", "door", "bolt", axis=(0, -1, 0), lower=0.0, upper=THROW, damping=BOLT_DAMPING)
kit.slider("strip_spring", "frame", "weatherstrip", axis=(-1, 0, 0), lower=-0.002, upper=STRIP_OUT,
           start=STRIP_OUT, stiffness=STRIP_RATE, rest=STRIP_OUT + STRIP_PRELOAD, damping=2.0)

OBJECTS = kit.objects()

# -- readouts (from body orientations and the fixed hinge axis, as a ruler and protractor would give them) -------------
BOLT_REF = np.asarray(BOLT_O) + kit.parts["bolt"].centre   # the bolt's geometric centre, retracted, door shut
HINGE_AXIS_PT = np.array([XF, HINGE_Y, 0.0])


def _rot(state) -> np.ndarray:
    return mk.quat_to_mat(state.quat_xyzw)


def door_angle(objects) -> float:
    """The door's swing (rad) from shut: negative is open (toward the robot)."""
    r = _rot(objects["door"])
    return float(math.atan2(r[1, 0], r[0, 0]))


def door_gap(objects) -> float:
    """How far (m) the door's latch edge stands off its closed position (0 shut, 12 mm resting on the weatherstrip)."""
    return max(0.0, -door_angle(objects)) * LATCH_R


def bolt_out(objects) -> float:
    """How far (m) the bolt stands out of the door's edge (0 retracted, 25 mm fully thrown)."""
    r = _rot(objects["door"])
    c = np.asarray(objects["bolt"].pos, float)
    local = r.T @ (c - HINGE_AXIS_PT) + HINGE_AXIS_PT
    return float(BOLT_REF[1] - local[1])


def turn_deg(objects) -> float:
    """The thumb-turn's angle (deg) on the door: 0 upright (unlocked), 90 turned toward the latch edge (locked)."""
    rel = _rot(objects["door"]).T @ _rot(objects["thumbturn"])
    return math.degrees(math.atan2(rel[2, 1], rel[1, 1]))


# -- scoring ---------------------------------------------------------------------------------------------------------
FLUSH = 0.002        # the door held within 2 mm of its stop
SEATED = 0.003       # within 3 mm: the bolt lines up with the strike (2 mm clearance, plus the bolt's own play)
ENTERED = 0.008      # bolt 8 mm out: 5 mm past the 3 mm gap, so in the strike
THROWN = 0.023       # bolt at least 23 of its 25 mm out
STILL = 0.0005       # neither the bolt nor the door's edge moved more than 0.5 mm since the last control step
HOLD = 20            # control steps (1 s at 20 Hz)
_STATE: dict = {}


def _reset_state() -> None:
    _STATE.pop("_this_step", None)
    _STATE.update(flush=False, entered=False, thrown=False, held=0, last=None)


_reset_state()
_orig_new_episode = mk.new_episode


def _new_episode() -> None:
    _orig_new_episode()
    _reset_state()


mk.new_episode = _new_episode   # the kit calls new_episode() at every reset


def _update(objects) -> tuple[float, bool]:
    g, b = door_gap(objects), bolt_out(objects)
    last, _STATE["last"] = _STATE["last"], (g, b)
    still = last is not None and abs(g - last[0]) < STILL and abs(b - last[1]) < STILL
    seated = g <= SEATED
    _STATE["flush"] |= bool(g <= FLUSH)
    _STATE["entered"] |= bool(seated and b >= ENTERED)
    locked = bool(seated and b >= THROWN)
    _STATE["thrown"] |= locked
    _STATE["held"] = _STATE["held"] + 1 if (locked and still) else 0
    if _STATE["held"] >= HOLD:   # success: locked and still for 20 control steps (1 s) in a row
        return 1.0, True
    if _STATE["thrown"]:
        return 0.75, False
    if _STATE["entered"]:
        return 0.5, False
    return (0.25 if _STATE["flush"] else 0.0), False


def success(objects) -> bool:
    """Locked: the bolt at least 23 mm into the strike with the door within 3 mm of its stop, still, for 1 s."""
    out = _update(objects)
    _STATE["_this_step"] = (objects, out)   # the runtime calls success() then score() on each step's states:
    return out[1]                            # score() reuses this, so the step counts once


def score(objects) -> float:
    """1.0 locked and still for 1 s; 0.75 bolt fully thrown into the strike at some point; 0.5 bolt into the strike
    (8 mm out with the door within 3 mm of its stop) at some point; 0.25 door held within 2 mm of its stop; else 0."""
    pending = _STATE.pop("_this_step", None)
    return (pending[1] if pending and pending[0] is objects else _update(objects))[0]
