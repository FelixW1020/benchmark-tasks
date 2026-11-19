"""Demo for the table fan task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 2400] [--no-record] [--mode full|offpin|nohold|noclips|still]

1. The right hand takes the blade by one of its three blades, a finger on each face, coming down its slope from
   the high edge. It carries the blade over the fan with the slot in its hub along the shaft's pin, lowers the hub
   onto the shaft until the slot is down over the pin, and lets go.
2. The left hand takes hold of the same blade. The right hand takes the blade cap from above, sets its bore over
   the shaft's threaded end and turns it anticlockwise in three bites until it is tight against the hub. Both hands
   let go.
3. The left hand takes the front grille by its rim, a finger above and a finger below, carries it over the fan
   tipped a little so that its bottom edge goes under the lip of the rear grille's support, and lets it down.
4. A fingertip presses each of the three clips shut: the left hand the one on the left, the right hand the other two.

Controls: ``--mode offpin`` puts the blade on a twelfth of a turn out, so that the hub rides on top of the pin, and
goes on as before. ``--mode nohold`` leaves the blade unheld while the cap is turned. ``--mode noclips`` stops after
the grille is on. ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``fankit.py``. The only feedback is
the hands' own measured positions (the robot's proprioception), used to correct for sag. The demo never reads a
part's state.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
FULL_OPEN = 0.0939                 # m between the pads at grip command 1.0


def frame(approach, closing) -> np.ndarray:
    """Tool rotation whose approach axis (z) and finger-closing axis (x) point along these world directions."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.asarray(closing, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def gap(width: float) -> float:
    """Grip command (1 open .. 0 closed) for a gap between the pads in metres."""
    return min(1.0, max(0.0, width / FULL_OPEN))


class Hand:
    """One arm: smooth moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = self.worst_deg = 0.0

    def _solve(self, pos, rot) -> None:
        from yam_arena import ik

        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, float(np.degrees(re)))
        self.cmd = np.r_[q, self.grip]

    def set_grip(self, grip: float, duration: float = 0.5):
        """Generator: open or close to ``grip`` in place."""
        from yam_arena import ik

        g0 = self.grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.grip = g0 + ik.smoothstep(k / n) * (grip - g0)
            self.cmd = np.r_[self.q, self.grip]
            yield

    def move_joints(self, q_goal, duration: float, grip: float | None = None):
        """Generator: swing to joint angles ``q_goal``, every joint moving together."""
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q, self.grip = q0 + s * (np.asarray(q_goal, float) - q0), g0 + s * (g1 - g0)
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
            self.cmd = np.r_[self.q, self.grip]
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*moves):
    """Generator: run several moves side by side until the longest is done."""
    moves = list(moves)
    while moves:
        for m in list(moves):
            try:
                next(m)
            except StopIteration:
                moves.remove(m)
        if moves:
            yield


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
UP = np.array([0.0, 0.0, 1.0])
DOWN = (0.0, 0.0, -1.0)
GAIN, BIAS_MAX = 0.25, 0.020             # how fast, and how far, a hand corrects for its own sag

# -- the blade ---------------------------------------------------------------------------------------------------------
HELD = 2                                 # which of the three blades the left hand takes
HOLD_R, HOLD_IN = 0.075, 0.030           # where on it: this far out from the shaft, this far up the chord from its middle
HOLD_SEG = 1                             # ... which is on the middle piece of the blade
BLADE_OPEN = 0.030                       # pads this far apart on the way onto the 3 mm blade
LEAD = 0.070                             # hands come in from this far back along their approach
LIFT = 0.060
OVER = 0.014                             # the hub's back face is carried this far above the shaft's end
PRESS = 0.0010                           # the hub is taken this far past its seat on the pin
BLADE_AWAY = 0.070                       # after letting go and backing off, the hand rises this far before it leaves
BLADE_CLEAR_X = 0.090                    # the blade's middle is brought back to this far beyond the shaft before it goes across
# -- the cap -----------------------------------------------------------------------------------------------------------
CAP_OPEN, CAP_HOLD = 0.060, 0.030        # pads astride the 3.8 cm cap, and commanded onto it
CAP_AT = 0.0165                          # the tool point's height above the cap's lower face
CAP_ON = 0.0045                          # the cap's lower face is set this far below the shaft's end: the thread catches
YAW0, BITE, BITES = 215.0, 180.0, 3      # the right hand's turn (degrees, anticlockwise from above): start, each bite (half a
#                                          turn brings the knob's flats back under the pads), how many
TURN_RATE = 70.0                         # degrees a second
# -- the left hand's stop ------------------------------------------------------------------------------------------------
STOP_AHEAD = 60.0                        # degrees round from the middle of the blade the right hand put on: between two blades
STOP_R, STOP_UP = 0.105, 0.005           # the tool point there: from the shaft, and above the blade's middle plane (m)
STOP_LEAN = 0.58                         # the hand leans back from the shaft (30 degrees): its fingers point down and in
PAD_TIP = 0.013                          # the fingertips' ends are this far past the tool point
# -- the grille --------------------------------------------------------------------------------------------------------
HOLD_AZ = 150.0                          # where round its rim the left hand takes it (degrees from its bottom edge)
RIM_OPEN = 0.034                         # pads this far apart on the way onto the 10 mm rim
SHORT = 0.007                            # the grille is brought in this far short of the support (its rim still outside the
#                                          blades' tips), then slid in under the support's lip
HOLD_UP, FAR_UP = 0.016, 0.001           # its rim above the rear grille's as it is slid in: at the hand, at its bottom edge
LET_GO = 0.016                           # the pads this far apart to let its 10 mm rim go: the lower finger has 4 mm under it
STEADY_R, STEADY_DOWN = 0.011, 0.001     # the right hand steadies it while the left hand's finger comes out from under its
#                                          rim: its fingertips on the band, this far outside the rim's inner edge, pressed this far
STEADY_LEAN = 0.58                       # ... its fingers pointing down and 30 degrees away from the robot (straight down is out of reach)
CARRY_Z = 0.265                          # the height above the table it is carried at, over the fan's base (which stands 21 cm high)
CARRY_FROM, CARRY_TO = 0.10, 0.80        # ... for this part of the way across
# -- the clips ---------------------------------------------------------------------------------------------------------
# A hand shuts a clip pointing down, its closed fingers side by side along the rim, the side of the fingertips against
# the open clip's outer end. Pushed straight in, the clip jams against the fingertip (it has to slide up the rubber as
# it turns). So the fingertips scoop it: in and up along the arc the clip's end follows, rising a little faster than it,
# until it goes over centre and shuts by itself; then they press its stem home.
TIP_HALF = (0.0077, 0.0137)              # half the fingertips' width: at their ends, and 13 to 23 mm up (measured from the robot's shapes)
PRESS_FROM = (0.034, 0.013)              # the tool point before the press: (out from the clip's hinge, above it); the
#                                          fingertips' ends are then level with the hinge
SCOOP_IN, SCOOP_RISE = 0.001, 1.3        # the fingertips' side is kept this far into the clip's end, and rises this many times as fast
SCOOP_AT = (71.2, 60.0, 50.0, 40.0, 30.0)    # the clip's end, degrees out from upright, at the scoop's way points (open: 71.2)
PRESS_HOME = (0.0117, 0.021)             # the tool point at the end: the fingertips' side on the shut clip's stem
CLIP_LEFT, CLIP_RIGHT = 0, (2, 1)        # the left hand presses the left clip; the right hand the right one, then the one
#                                          nearest the robot, which it reaches by going round the outside of the rim
ROUND_R, ROUND_UP, ROUND_LEAN = 0.200, 0.038, 0.36    # ... at this radius, the tool point this far above the hinges, the
#                                          fingers pointing down and 20 degrees in toward the fan
ROUND_STEP, ROUND_S = 15.0, 1.0          # ... in steps of this many degrees, each taking this long


def HOLD_RADIUS(K) -> float:
    """How far from the grille's middle the tool point holds its rim: the fingertips stop short of the band."""
    return K.RIM_IN + K.BAND_T + 0.001 + 0.013


class Sag:
    """Slowly offsets the hand's command by its measured error, so that its tool point ends up where it is wanted."""

    def __init__(self, hand):
        self.hand, self.bias, self.want = hand, np.zeros(3), None

    def to(self, pos, rot):
        if self.want is not None:
            self.bias = np.clip(self.bias + GAIN * (self.want - TOOL[self.hand.name]), -BIAS_MAX, BIAS_MAX)
        self.want = np.asarray(pos, float)
        self.hand._solve(self.want + self.bias, rot)


def glide(sag, target, seconds: float, rot, rot_from=None):
    """Generator: take the hand's tool point in a straight line to ``target``, correcting for sag on the way. With
    ``rot_from`` the hand also turns from that rotation to ``rot``."""
    from yam_arena import ik

    start = sag.want if sag.want is not None else sag.hand.pos.copy()
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        r = rot if rot_from is None else ik.slerp_matrix(rot_from, rot, s)
        sag.to(start + s * (np.asarray(target, float) - start), r)
        yield


def _seeds(q_now) -> list:
    from yam_arena.constants import READY

    out = [np.asarray(q_now, float).copy(), np.array(READY, float)]
    for q1 in (0.8, 1.3, -0.8, -1.3, 0.0):
        for q5 in (-1.0, 0.0, 1.0):
            s = np.array(READY, float)
            s[0], s[4] = q1, q5
            out.append(s)
    for q6 in (-2.0, -1.0, 1.0, 2.0):
        s = np.array(READY, float)
        s[5] = q6
        out.append(s)
    return out


def reach(hand, pos, rot, near=None) -> np.ndarray | None:
    """Joint angles that put the hand's tool point at ``pos`` with rotation ``rot``: of the solutions found from a
    spread of starting poses, the one nearest ``near`` (default: where the hand is). None if there is none."""
    from yam_arena import ik

    near = hand.q if near is None else np.asarray(near, float)
    best = None
    for seed in _seeds(near):
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        if pe > 0.0005 or re > math.radians(0.3):
            continue
        cost = float(np.abs(q - near).max())
        if best is None or cost < best[1]:
            best = (q, cost)
    return None if best is None else best[0]


def down(yaw_deg: float) -> np.ndarray:
    """A top-down hand's rotation, its closing axis ``yaw_deg`` degrees round from x."""
    a = math.radians(yaw_deg)
    return frame(DOWN, (math.cos(a), math.sin(a), 0.0))


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_turn(axis, deg: float) -> np.ndarray:
    """Rotation by ``deg`` degrees about ``axis``."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def plan(T, blade_off: float = 0.0) -> dict:
    """The demo's fixed targets, from the task file's layout (world, m). ``blade_off``: degrees by which the blade is
    put on turned away from the pin (a control run)."""
    K = T.K
    fan, _ = T.fan_pose()
    p: dict = {"fan": fan}

    # the blade: where the hand holds it, in the Blade's frame
    centre, rp, _size = K.blade_plate(HELD, HOLD_SEG)
    az = K.blade_azimuth(HELD)
    hold = K._rz(az) @ np.array([HOLD_R, 0.0, K.HUB_H / 2]) + rp @ np.array([0.0, HOLD_IN, 0.0])
    grip_rot = [frame(-rp[:, 1], s * rp[:, 2]) for s in (1.0, -1.0)]            # down the chord, a finger on each face
    b0, bq = T.blade_start()
    yaw0 = T.blade_yaw_on() - T.LAYOUT["turn"]
    yaw1 = T.blade_yaw_on() + blade_off
    over = fan + np.array([0.0, 0.0, K.SHAFT_TIP + OVER])
    seat = fan + np.array([0.0, 0.0, (K.HUB_SEAT if not blade_off else K.HUB_RIDING) - PRESS])
    aside = np.array([min(0.0, fan[0] + BLADE_CLEAR_X - b0[0]), 0.0, 0.0])
    p["blade"] = {"hold": hold, "grip_rot": grip_rot, "yaw0": yaw0, "yaw1": yaw1, "start": b0, "over": over, "seat": seat,
                  "aside": aside}

    # the left hand's stop: a closed fingertip between two blades, ahead of the held blade the way the cap is turned
    held = math.radians(yaw1) + K.blade_azimuth(HELD) + math.radians(STOP_AHEAD)
    sr = np.array([math.cos(held), math.sin(held), 0.0])
    st = np.array([-math.sin(held), math.cos(held), 0.0])
    lean_in = np.array(DOWN) - STOP_LEAN * sr
    p["stop"] = {"at": fan + STOP_R * sr + np.array([0.0, 0.0, K.HUB_SEAT + K.HUB_H / 2 + STOP_UP]),
                 "rots": [frame(lean_in, s * st) for s in (1.0, -1.0)]}

    # the cap
    c0, _ = T.cap_start()
    p["cap"] = {"start": c0 + np.array([0.0, 0.0, CAP_AT]),
                "above": fan + np.array([0.0, 0.0, K.SHAFT_TIP + 0.012 + CAP_AT]),
                "on": fan + np.array([0.0, 0.0, K.SHAFT_TIP - CAP_ON + CAP_AT]),
                "drop": K.PITCH * BITE / 360.0}

    # the grille: the hand's hold in the Grille's frame
    a = math.radians(HOLD_AZ)
    out = np.array([math.cos(a), math.sin(a), 0.0])
    g_hold = HOLD_RADIUS(K) * out + np.array([0.0, 0.0, K.GRILLE_T / 2])
    g_rot = [frame(-out, (0.0, 0.0, s)) for s in (1.0, -1.0)]
    g0, _ = T.grille_start()
    turn = K._rz(math.radians(T.LAYOUT["grille_turn"]))                           # how it lies on its towel
    far = np.array([K.RIM_OUT, 0.0, 0.0])                                        # its bottom edge, in its own frame
    span = far - HOLD_RADIUS(K) * out
    tip_axis = np.cross(span / np.linalg.norm(span), UP)                         # tipping about this lowers the bottom edge
    tip_deg = math.degrees(math.asin((HOLD_UP - FAR_UP) / np.linalg.norm(span)))
    tipped = axis_turn(tip_axis, -tip_deg)
    # the grille's origin when its rim at the hand is HOLD_UP above the rear grille's rim and it is tipped
    hand_at = fan + np.array([0.0, 0.0, K.GRILLE_SEAT + HOLD_UP]) + g_hold
    lean = np.array(DOWN, float) + STEADY_LEAN * np.array([1.0, 0.0, 0.0])
    lean = lean / np.linalg.norm(lean)
    across = np.cross(UP, out)
    steady = fan - (K.RIM_IN + STEADY_R) * out + np.array([0.0, 0.0, K.GRILLE_SEAT + K.BAND_Z]) - PAD_TIP * lean
    p["grille"] = {"hold": g_hold, "rot": g_rot, "start": g0, "tipped": tipped, "out": out, "turn": turn,
                   "hand_in": hand_at, "hand_short": hand_at - np.array([SHORT, 0.0, 0.0]),
                   "steady": steady, "steady_rot": [frame(lean, c) for c in (across, -across, out, -out)]}

    # the clips: where the tool point goes to shut each one
    clips = []
    end_r = math.hypot(K.CLIP_STEM[1][0], K.CLIP_STEM[1][1])                    # the clip's outer end from its hinge
    end_up0 = end_r * math.cos(math.radians(SCOOP_AT[0]))
    for k in range(3):
        a = math.radians(K.CLIP_AZ[k])
        r = np.array([math.cos(a), math.sin(a), 0.0])
        hinge = fan + K.CLIP_HINGE_R * r + np.array([0.0, 0.0, K.CLIP_HINGE_Z])
        at = lambda out, up, r=r, hinge=hinge: hinge + out * r + up * UP      # noqa: E731
        scoop = [at(end_r * math.sin(math.radians(d)) - SCOOP_IN + TIP_HALF[0],
                    PRESS_FROM[1] + SCOOP_RISE * (end_r * math.cos(math.radians(d)) - end_up0)) for d in SCOOP_AT]
        clips.append({"from": at(*PRESS_FROM), "scoop": scoop, "to": at(*PRESS_HOME), "out": at(PRESS_FROM[0], PRESS_HOME[1]),
                      "r": r, "t": np.cross(UP, r)})
    p["clips"] = clips
    return p


def _first(hand, poses, rots, near=None):
    """The first rotation of ``rots`` with which the hand reaches every one of ``poses`` (each a function of the
    rotation or a fixed point), and the joint angles for the first pose."""
    for rot in rots:
        qs, q = [], near
        for pose in poses:
            pos, r = pose(rot)
            q = reach(hand, pos, r, near=q)
            if q is None:
                break
            qs.append(q)
        else:
            return rot, qs
    return None, None


def script(T, left, right, mode: str = "full", idle_s: float = 120.0, upto: str = "clips"):
    """The demo as a generator: one yield per control step."""
    from yam_arena.constants import READY

    K = T.K
    if mode == "still":
        yield from wait(idle_s)
        return
    p = plan(T, 30.0 if mode == "offpin" else 0.0)
    ready = np.array(READY, float)
    yield from blade_and_cap(T, left, right, p, mode, upto)
    if upto in ("blade", "cap"):
        return
    yield from grille_and_clips(T, left, right, p, mode, upto)


def blade_and_cap(T, left, right, p, mode, upto):
    from yam_arena.constants import READY

    K = T.K
    ready = np.array(READY, float)

    # 1. The blade: the right hand takes it by one blade, carries it over the fan, lowers its hub onto the shaft and
    #    the pin, and lets go.
    b = p["blade"]
    turn0, turn1 = K._rz(math.radians(b["yaw0"])), K._rz(math.radians(b["yaw1"]))
    at0 = b["start"] + turn0 @ b["hold"]
    over, seat = b["over"] + turn1 @ b["hold"], b["seat"] + turn1 @ b["hold"]
    up0 = np.array([at0[0], at0[1], over[2]])

    def blade_poses(rot):
        r0, r1 = turn0 @ rot, turn1 @ rot
        return [lambda _r, v=(at0 - LEAD * r0[:, 2], r0): v, lambda _r, v=(at0, r0): v, lambda _r, v=(up0, r0): v,
                lambda _r, v=(up0 + b["aside"], r0): v, lambda _r, v=(over, r1): v, lambda _r, v=(seat, r1): v,
                lambda _r, v=(seat - LEAD * r1[:, 2] + BLADE_AWAY * UP, r1): v]

    rot, qs = None, None
    for cand in b["grip_rot"]:
        rot, qs = _first(right, blade_poses(cand), [cand])
        if rot is not None:
            break
    r0, r1 = turn0 @ rot, turn1 @ rot
    yield from right.move_joints(qs[0], 3.0, grip=gap(BLADE_OPEN))
    rsag = Sag(right)
    yield from glide(rsag, at0 - LEAD * r0[:, 2], 0.4, r0)
    yield from glide(rsag, at0, 1.6, r0)
    yield from glide(rsag, at0, 0.4, r0)
    yield from right.set_grip(0.0, 0.6)
    yield from wait(0.3)
    yield from glide(rsag, up0, 2.5, r0)                                    # straight up, to the height it is carried at
    yield from glide(rsag, up0 + b["aside"], 2.0, r0)                       # toward the robot, clear of the fan's base
    yield from glide(rsag, over, 3.5, r1, rot_from=r0)
    yield from glide(rsag, over, 0.6, r1)
    yield from glide(rsag, seat, 3.0, r1)
    yield from glide(rsag, seat, 0.6, r1)
    yield from right.set_grip(gap(BLADE_OPEN), 0.6)
    yield from glide(rsag, seat - LEAD * r1[:, 2], 1.5, r1)                 # back up the blade's slope, clear of it
    yield from glide(rsag, seat - LEAD * r1[:, 2] + BLADE_AWAY * UP, 1.2, r1)   # straight up, above all three blades
    yield from right.move_joints(ready, 2.5)                                # and up, before it goes for the cap
    if upto == "blade":
        yield from wait(2.0)
        return

    # 2. The left hand puts a closed fingertip down between two blades, ahead of the one the right hand put on (the
    #    shaft turns freely: without a stop the blade and the shaft would go round with the cap). Meanwhile the right
    #    hand fetches the cap from above.
    hold = mode not in ("nohold", "offpin")
    st = p["stop"]
    l_rot, l_qs = (None, None)
    if hold:
        l_rot, l_qs = _first(left, [lambda r, v=st["at"] + 0.045 * UP: (v, r), lambda r, v=st["at"]: (v, r)], st["rots"])
    lsag = Sag(left)

    def left_takes_hold():
        if l_rot is None:
            return
        yield from left.move_joints(l_qs[0], 3.5, grip=0.0)
        yield from glide(lsag, st["at"] + 0.045 * UP, 0.4, l_rot)
        yield from glide(lsag, st["at"], 1.5, l_rot)

    c = p["cap"]

    def right_fetches_cap():
        q_pick = reach(right, c["start"], down(-90.0))
        q_pick_up = reach(right, c["start"] + 0.08 * UP, down(-90.0), near=q_pick)
        yield from right.move_joints(q_pick_up, 3.0, grip=gap(CAP_OPEN))
        sag = Sag(right)
        yield from glide(sag, c["start"] + 0.08 * UP, 0.4, down(-90.0))
        yield from glide(sag, c["start"], 1.5, down(-90.0))
        yield from glide(sag, c["start"], 0.4, down(-90.0))
        yield from right.set_grip(gap(CAP_HOLD), 0.6)
        yield from wait(0.2)
        high = np.array([c["start"][0], c["start"][1], c["above"][2]])
        yield from glide(sag, high, 2.0, down(-90.0))                       # straight up, to the height of the shaft's end
        yield from glide(sag, c["above"], 4.0, down(YAW0), rot_from=down(-90.0))   # across to the fan at that height

    yield from together(left_takes_hold(), right_fetches_cap())

    #    The cap goes onto the shaft's end and is turned anticlockwise in bites until it is tight.
    rsag = Sag(right)
    yield from glide(rsag, c["above"], 0.6, down(YAW0))
    yield from glide(rsag, c["on"], 2.0, down(YAW0))
    yield from glide(rsag, c["on"], 0.5, down(YAW0))
    at = c["on"].copy()
    for bite in range(BITES):
        q0 = reach(right, at, down(YAW0), near=right.q)
        q1 = reach(right, at - c["drop"] * UP, down(YAW0 + BITE), near=q0)
        yield from right.move_joints(q1, BITE / TURN_RATE)                  # the wrist turns the cap; the hand follows it down
        yield from wait(0.2)
        at = at - c["drop"] * UP
        if bite < BITES - 1:
            yield from right.set_grip(gap(CAP_OPEN), 0.5)
            q_back = reach(right, at, down(YAW0), near=q0)
            yield from right.move_joints(q_back, 1.8)
            yield from right.set_grip(gap(CAP_HOLD), 0.5)
            yield from wait(0.1)
    yield from right.set_grip(gap(CAP_OPEN), 0.5)
    rsag = Sag(right)
    yield from glide(rsag, at + 0.020 * UP, 0.8, right.rot)
    if l_rot is not None:
        yield from glide(lsag, st["at"] + 0.045 * UP, 1.0, l_rot)
    yield from together(left.move_joints(ready, 3.0), right.move_joints(ready, 3.0))
    if upto == "cap":
        yield from wait(2.0)
        return



def grille_and_clips(T, left, right, p, mode, upto):
    from yam_arena.constants import READY

    K = T.K
    ready = np.array(READY, float)
    # 3. The front grille: by its rim, over the fan, its bottom edge under the support's lip, and down.
    g = p["grille"]

    def carry(rot):
        """The way across: (tool point, rotation) at the start, where the carrying height is reached, where it is left,
        and above the place short of the support."""
        from yam_arena import ik

        r_pad = g["turn"] @ rot
        at0 = g["start"] + g["turn"] @ g["hold"]
        end = g["hand_short"] + 0.03 * UP
        out = [(at0, r_pad)]
        for f in (CARRY_FROM, CARRY_TO):
            pos = at0 + f * (end - at0)
            out.append((np.array([pos[0], pos[1], T.TOP + CARRY_Z]), ik.slerp_matrix(r_pad, rot, f)))
        out.append((end, rot))
        return out

    def grille_poses(rot):
        r_tip, r_pad = g["tipped"] @ rot, g["turn"] @ rot
        at0 = g["start"] + g["turn"] @ g["hold"]
        return ([lambda _r, v=(at0 - LEAD * r_pad[:, 2], r_pad): v] + [lambda _r, v=v: v for v in carry(rot)]
                + [lambda _r, v=(g["hand_short"], r_tip): v, lambda _r, v=(g["hand_in"], r_tip): v])

    rot, qs = None, None
    for cand in g["rot"]:
        rot, qs = _first(left, grille_poses(cand), [cand])
        if rot is not None:
            break
    r_tip, r_pad = g["tipped"] @ rot, g["turn"] @ rot
    at0 = g["start"] + g["turn"] @ g["hold"]
    yield from left.move_joints(qs[0], 3.5, grip=gap(RIM_OPEN))
    lsag = Sag(left)
    yield from glide(lsag, at0 - LEAD * r_pad[:, 2], 0.4, r_pad)
    yield from glide(lsag, at0, 1.8, r_pad)
    yield from glide(lsag, at0, 0.4, r_pad)
    yield from left.set_grip(0.0, 0.7)
    yield from wait(0.3)
    way = carry(rot)
    yield from glide(lsag, way[1][0], 2.5, way[1][1], rot_from=way[0][1])       # up to the carrying height as it starts across
    yield from glide(lsag, way[2][0], 6.0, way[2][1], rot_from=way[1][1])       # across, over the fan's base
    yield from glide(lsag, way[3][0], 3.0, way[3][1], rot_from=way[2][1])       # down to just above the fan
    yield from glide(lsag, way[3][0], 0.8, rot)                                 # (the hand settles: 2 to 3 mm to spare)
    yield from glide(lsag, way[3][0], 1.0, r_tip, rot_from=rot)                 # tipped, its bottom edge low
    yield from glide(lsag, g["hand_short"], 2.0, r_tip)                         # down between the blades' tips and the lip
    yield from glide(lsag, g["hand_short"], 0.5, r_tip)
    yield from glide(lsag, g["hand_in"], 1.5, r_tip)
    yield from glide(lsag, g["hand_in"] - FAR_UP * UP, 0.6, r_tip)
    #    The left hand's lower finger is under the rim. The right hand's fingertips come down on the grille's far side
    #    and steady it while that finger is drawn out; the rim then drops onto the rear grille's.
    s_rot, s_qs = _first(right, [lambda r, v=g["steady"] + 0.05 * UP: (v, r), lambda r, v=g["steady"] - STEADY_DOWN * UP: (v, r)],
                         g["steady_rot"])
    yield from right.move_joints(s_qs[0], 2.5, grip=0.0)
    rsag = Sag(right)
    yield from glide(rsag, g["steady"] + 0.05 * UP, 0.4, s_rot)
    yield from glide(rsag, g["steady"] - STEADY_DOWN * UP, 1.2, s_rot)
    yield from left.set_grip(gap(LET_GO), 0.5)
    yield from wait(0.2)
    back = g["hand_in"] - FAR_UP * UP - 0.030 * r_tip[:, 2]
    yield from glide(lsag, back, 1.2, r_tip)
    yield from glide(lsag, back + 0.080 * UP, 1.2, r_tip)
    yield from glide(rsag, g["steady"] + 0.05 * UP, 0.8, s_rot)
    yield from together(left.move_joints(ready, 3.0), right.move_joints(ready, 2.5))
    if upto == "grille" or mode == "noclips":
        yield from wait(2.0)
        return
    yield from clips_shut(T, left, right, p)


def clips_shut(T, left, right, p):
    from yam_arena.constants import READY

    K = T.K
    ready = np.array(READY, float)
    # 4. The clips: fingertips press each one in until it shuts over the front grille's rim. The left hand, level,
    #    pushes the left one with its fingertips' ends; the right hand, pointing down, the other two with a finger's side.
    hands = {"left": left, "right": right}
    fan = p["fan"]

    def press_pose(hand, k):
        cl = p["clips"][k]
        rots = [frame(DOWN, sg * cl["t"]) for sg in (1.0, -1.0)]
        before = cl["from"] + 0.03 * cl["r"] + 0.025 * UP
        rot, qs = _first(hand, [lambda r, v=v: (v, r) for v in (before, cl["from"], *cl["scoop"], cl["to"], cl["out"])], rots)
        return before, rot, qs

    def press(hand, sag, k, before, rot):
        cl = p["clips"][k]
        yield from glide(sag, before, 0.4, rot)
        yield from glide(sag, cl["from"], 1.0, rot)
        yield from glide(sag, cl["scoop"][0], 0.9, rot)                     # in, onto the open clip's end
        for v in cl["scoop"][1:]:
            yield from glide(sag, v, 0.4, rot)                              # in and up with it, until it goes over
        yield from glide(sag, cl["to"], 0.6, rot)                           # and its stem pressed home
        yield from glide(sag, cl["to"], 0.3, rot)
        yield from glide(sag, cl["out"], 0.6, rot)
        yield from glide(sag, before, 0.6, rot)

    def round_rim(sag, az_from, az_to, sign):
        """The hand goes round the outside of the rim from one clip to another, its fingers pointing down and a
        little in toward the fan, above the open clips."""
        n = max(1, int(round(abs(az_to - az_from) / ROUND_STEP)))
        poses = []
        for i in range(n + 1):
            a = math.radians(az_from + (az_to - az_from) * i / n)
            r = np.array([math.cos(a), math.sin(a), 0.0])
            poses.append((fan + ROUND_R * r + np.array([0.0, 0.0, K.CLIP_HINGE_Z + ROUND_UP]),
                          frame(np.array(DOWN, float) - ROUND_LEAN * r, sign * np.cross(UP, r))))
        return poses

    def left_clip():
        k = CLIP_LEFT
        before, rot, qs = press_pose(left, k)
        yield from left.move_joints(qs[0], 3.0, grip=0.0)
        yield from press(left, Sag(left), k, before, rot)
        yield from left.move_joints(ready, 2.5)

    def right_clips():
        k, k2 = CLIP_RIGHT
        before, rot, qs = press_pose(right, k)
        yield from right.move_joints(qs[0], 3.0, grip=0.0)
        sag = Sag(right)
        yield from press(right, sag, k, before, rot)
        before2 = p["clips"][k2]["from"] + 0.03 * p["clips"][k2]["r"] + 0.025 * UP
        sign = 1.0 if float(rot[:, 0] @ p["clips"][k]["t"]) > 0 else -1.0
        way = round_rim(sag, K.CLIP_AZ[k], K.CLIP_AZ[k2], sign)
        rot2 = frame(DOWN, sign * p["clips"][k2]["t"])
        yield from glide(sag, way[0][0], 0.8, way[0][1], rot_from=rot)
        for (pa, ra), (pb, rb) in zip(way[:-1], way[1:]):
            yield from glide(sag, pb, ROUND_S, rb, rot_from=ra)
        yield from glide(sag, before2, 0.8, rot2, rot_from=way[-1][1])
        yield from press(right, sag, k2, before2, rot2)
        yield from glide(sag, way[-1][0], 0.8, way[-1][1], rot_from=rot2)
        for (pa, ra), (pb, rb) in zip(way[::-1][:-1], way[::-1][1:]):
            yield from glide(sag, pb, ROUND_S, rb, rot_from=ra)
        yield from glide(sag, before, 0.8, rot, rot_from=way[0][1])
        yield from right.move_joints(ready, 2.5)

    yield from together(left_clip(), right_clips())
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "full", every: int = 5, cam: tuple = (320, 176), upto: str = "clips") -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena import station
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam[0],
                               cam_height=cam[1], headless=not window)
    T = emb.task.module
    obs = emb.reset(Scene(id="fan-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ, upto=upto)
    frames: dict[str, list] = {}
    trace = []
    names = [T.FAN, T.ROTOR, T.BLADE, T.CAP, T.GRILLE, *T.CLIPS]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from fankit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))

    def hands() -> dict:
        from fankit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1)}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        return {n: mm(st[n].pos) for n in (T.BLADE, T.CAP, T.GRILLE)}

    snap_frames(obs)
    measure()
    start = {"task": T.report(states()), "places": places(states())}
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(left.cmd, right.cmd)))
        steps += 1
        measure()
        info = res.info
        snap_frames(res.observation)
        if steps % every == 0:
            st = states()
            trace.append({"step": steps, "score": info.get("score"), "task": T.report(st), "hands": hands(), "places": places(st)})
        if steps % 40 == 0 and trace:
            t = trace[-1]
            print(f"step {steps} score {t['score']} {t['task']} L {t['hands']['left']} R {t['hands']['right']} {t['places']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "upto": upto, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended,
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": {"task": T.report(states()), "places": places(states())}, "trace": trace}
    if record:
        import imageio

        for name, fs in frames.items():
            _write_mp4(fs, out / f"demo_{name}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "offpin", "nohold", "noclips", "still"))
    ap.add_argument("--upto", default="clips", choices=("blade", "cap", "grille", "clips"), help="stop after this part")
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to go on after the script ends")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")), hold_s=a.hold, upto=a.upto)
