"""Demo for the compass-calibration task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 3000] [--no-record] [--mode full|short|tilted|down|level_only|off_pad|still]

1. The right hand comes straight down over the drone, takes its fuselage by the sides and lifts it off the pad.
2. Level turn. The right wrist turns the drone about the vertical as far as it goes. The left hand comes in level
   behind the tail and takes the fuselage top and bottom; the right hand lets go, rises clear, turns back half a
   turn, comes down and takes the sides again; the left hand lets go and draws back; the right wrist turns the drone
   on, past a full turn in all.
3. The right hand turns the drone back to where the left can reach its tail; the left takes it, the right lets go,
   and the left hand swings from level to pointing straight down, which leaves the drone hanging nose down.
4. Upright turn. The same exchange with the hands' parts swapped: the left wrist turns the hanging drone, the right
   hand holds its middle while the left turns back half a turn, and the left wrist turns it on past a full turn.
5. The left hand swings back to level, the right hand takes the sides from above, the left lets go, and the right
   hand stands the drone on its pad and lets go.

Controls: ``--mode short`` stops the level turn at the exchange (about 200 degrees), then does everything else.
``--mode tilted`` makes the level turn with the drone held 30 degrees nose down. ``--mode down`` stands the drone on
its pad for the level turn's exchange instead of passing it in the air. ``--mode level_only`` puts the drone back on
its pad after the level turn. ``--mode off_pad`` is the full demo, but stands the drone on the bare table beside its
pad. ``--mode still`` keeps both arms still.

The script is open loop. It plans from the layout constants in the task file and the sizes in ``dronekit.py``, keeps
its own account of where the drone should be from what it has commanded, and never reads the drone's state. The only
feedback is each hand's own measured position and direction (the robot's proprioception), used to correct for the
arm's sag.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# -- inverse kinematics ---------------------------------------------------------------------------------------------------
# The same damped least squares as yam_arena.ik.solve, with the Jacobian written out instead of found by differences:
# the plan below tries several hundred poses, which the stock solver takes minutes over.
_CHAIN: list = []


def _chain() -> list:
    if not _CHAIN:
        from yam_arena import YAM_URDF
        from yam_arena.constants import ARM_JOINTS
        from yam_arena.urdf import parse_joints, rpy_matrix

        joints = parse_joints(YAM_URDF)
        by_child = {j.child: j for j in joints.values()}
        out, link = [], "gripper"
        while link != "base":
            j = by_child[link]
            out.append(j)
            link = j.parent
        index = {n: i for i, n in enumerate(ARM_JOINTS)}
        for j in out[::-1]:
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = rpy_matrix(j.rpy), j.xyz
            _CHAIN.append((t, np.asarray(j.axis, float) / np.linalg.norm(j.axis), index.get(j.name) if j.type != "fixed" else None))
    return _CHAIN


def _axis_turn(axis, a: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _fk(q, base) -> tuple:
    from yam_arena.constants import TCP_OFFSET

    T = np.eye(4)
    T[:3, 3] = base
    axes, origins = [None] * 6, [None] * 6
    for t, axis, i in _chain():
        T = T @ t
        if i is not None:
            axes[i], origins[i] = T[:3, :3] @ axis, T[:3, 3].copy()
            R = np.eye(4)
            R[:3, :3] = _axis_turn(axis, float(q[i]))
            T = T @ R
    return T[:3, :3] @ np.array(TCP_OFFSET) + T[:3, 3], T[:3, :3], axes, origins


def _rot_err(r_target, r) -> np.ndarray:
    m = r_target @ r.T
    c = max(-1.0, min(1.0, (np.trace(m) - 1.0) / 2.0))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    n = float(np.linalg.norm(v))
    if n < 1e-9:                              # half a turn: the axis is the eigenvector of the symmetric part
        _, vec = np.linalg.eigh((m + m.T) / 2.0)
        return vec[:, -1] * a
    return v / n * a


def solve_ik(pos, rot, seed, base, iters: int = 150, damping: float = 0.05, w: float = 0.3) -> tuple:
    """Joint angles reaching tool point ``pos`` with rotation ``rot``, from ``seed``: (q, position error m, rotation
    error rad)."""
    from yam_arena import ik

    q, target = np.array(seed, float), np.asarray(pos, float)
    for _ in range(iters):
        p, r, axes, origins = _fk(q, base)
        e = np.r_[target - p, w * _rot_err(rot, r)]
        if float(np.linalg.norm(e)) < 1e-5:
            break
        jm = np.zeros((6, 6))
        for i in range(6):
            jm[:3, i], jm[3:, i] = np.cross(axes[i], p - origins[i]), w * axes[i]
        q = np.clip(q + jm.T @ np.linalg.solve(jm @ jm.T + damping ** 2 * np.eye(6), e), ik.LOWER, ik.UPPER)
    p, r, _, _ = _fk(q, base)
    return q, float(np.linalg.norm(target - p)), float(np.linalg.norm(_rot_err(rot, r)))



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

        q, pe, re = solve_ik(pos, rot, self.q, self.base, iters=60)
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


def both(*gens):
    """Generator: run several moves side by side until all have finished."""
    live = list(gens)
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
        if live:
            yield


def best_ik(hand, pos, rot):
    """Joint angles for ``pos``/``rot`` from the best of a few seeds: the one that leaves the joints most room."""
    from yam_arena import ik

    seeds = [hand.q.copy()]
    for q6 in (-1.5, -0.7, 0.7, 1.5):
        s = hand.q.copy()
        s[5] = q6
        seeds.append(s)
    for q1, q5 in ((0.6, 0.8), (-0.6, -0.8), (1.0, 1.2), (-1.0, -1.2)):
        s = hand.q.copy()
        s[0], s[4] = q1, q5
        seeds.append(s)
    best = None
    for seed in seeds:
        q, pe, re = solve_ik(pos, rot, seed, hand.base, iters=400)
        room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
        cost = (pe > 5e-4 or re > 5e-3, -room)
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0]


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
TOOLROT: dict = {}                       # ... and which way the hand actually points
PHASE: dict = {"name": "start"}
SAGS: dict = {}                          # each hand's sag corrector, for the trace
HOLD: dict = {"left": None, "right": None}   # the grasp each hand has on the drone (None: not holding)
EST: dict = {}                           # where the script expects the drone to be (its own account)

UP = np.array([0.0, 0.0, 1.0])
# -- how the drone is held (in its own frame: x to the nose, y to its left, z through its top) -----------------------------
TAIL_IN = 0.020                          # the tail grasp's tool point is this far in from the tail's end
MID_Y_AT, MID_Z_AT = 0.008, 0.020        # the middle grasps' tool points, this far toward the nose from the middle
SIDE_OPEN, SIDE_GRIP = 0.090, 0.052      # pads apart on the way in, and the command when closed on the 58 mm width
FLAT_OPEN, FLAT_GRIP = 0.078, 0.034      # ... and on the 40 mm height
BACK = 0.085                             # an opened hand draws this far back from its grasp to be clear of the drone
WAITS = (0.160, 0.130, 0.100)            # ... and waits about this far back while the other hand turns the drone
NEAR_BASE = 0.120                        # a waiting hand keeps its tool point at least this far out from the arms' bases (x)
EASY = 0.20                              # rad of joint travel beyond which more makes a pose no better
Z_CAP = 0.190                            # a hand pointing straight down keeps its tool point below this
# -- where things happen (m above the table) ---------------------------------------------------------------------------
H_LEVEL = 0.135                          # the drone's middle during the level turn
LEVEL_XY = (0.270, -0.070)               # ... and where
H_ON_END = 0.155                         # the drone's middle during the upright turn
ON_END_XY = (0.210, 0.080)               # ... and where
DIP = 0.038                              # the right hand lowers the hanging drone this far for the left to turn back
TURN_RATE, BACK_RATE = 40.0, 70.0        # degrees a second: turning the drone, and turning an empty hand back
GAIN, BIAS_MAX = 0.25, 0.040             # how fast, and how far, a hand corrects for its own sag
TURN_GAIN, TURN_MAX = 0.15, math.radians(8.0)   # ... and for pointing off the way it is told to
ROOM = 0.06                              # rad of joint travel a plan keeps in hand
OVER = 20.0                              # degrees a full turn is overshot by


def _rotvec(rot) -> np.ndarray:
    """The rotation vector of a rotation matrix."""
    a = math.acos(float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    if a < 1e-9:
        return np.zeros(3)
    return np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) * (a / (2.0 * math.sin(a)))


def _turn(vec) -> np.ndarray:
    """The rotation matrix of a rotation vector."""
    a = float(np.linalg.norm(vec))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(vec, float) / a
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * (kx @ kx)


def _square(rot) -> np.ndarray:
    """The nearest rotation whose axes are the coordinate axes (the fixed turn between two ways of naming a frame)."""
    out = np.zeros((3, 3))
    m = np.abs(rot).copy()
    for _ in range(3):
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        out[i, j] = math.copysign(1.0, rot[i, j])
        m[i, :] = -1.0
        m[:, j] = -1.0
    return out


class Sag:
    """Slowly offsets a hand's command by its measured error, so that its tool point ends up where it is wanted; and,
    while ``live``, turns the command by the hand's measured error in direction, so that it points as wanted."""

    def __init__(self, hand):
        self.hand, self.bias, self.want = hand, np.zeros(3), None
        self.live, self.trim, self.want_rot, self.frame, self.off = True, np.zeros(3), None, None, 0.0

    def measured(self):
        """The hand's measured rotation, in the convention the commands use."""
        rot = TOOLROT.get(self.hand.name)
        if rot is None:
            return None
        if self.frame is None:                       # found once, while the arm rests at its ready pose
            self.frame = _square(rot.T @ self.hand.rot)
        return rot @ self.frame

    def to(self, pos, rot):
        if self.want is not None and self.hand.name in TOOL:
            self.bias = np.clip(self.bias + GAIN * (self.want - TOOL[self.hand.name]), -BIAS_MAX, BIAS_MAX)
        got = self.measured()
        if self.want_rot is not None and got is not None:
            err = _rotvec(self.want_rot @ got.T)
            self.off = math.degrees(float(np.linalg.norm(err)))
            if self.live:
                trim = self.trim + TURN_GAIN * err
                size = float(np.linalg.norm(trim))
                self.trim = trim if size <= TURN_MAX else trim * (TURN_MAX / size)
        self.want, self.want_rot = np.asarray(pos, float), rot
        self.hand._solve(self.want + self.bias, _turn(self.trim) @ rot)


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


def go(sag, pos, rot, seconds: float):
    """Generator: swing the arm to a pose it is not yet near, by its joints, then settle on it."""
    yield from sag.hand.move_joints(best_ik(sag.hand, pos, rot), seconds)
    sag.want = None
    yield from glide(sag, pos, 0.3, rot)



def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def drone_rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """The drone's axes in the world: its nose ``yaw`` round from straight away from the robot, pitched nose down by
    ``pitch`` (90: hanging nose down, its top facing ``yaw``)."""
    return _rz(math.radians(yaw_deg)) @ _ry(math.radians(pitch_deg))


def grasp(kind: str, s: int = 1, a: int = 1) -> tuple:
    """A grasp as (the tool's axes in the drone's frame, the tool point in the drone's frame, pads apart open, closed).
    ``side``: from above, pads on the fuselage's sides. ``tail``: from behind, pads on its top and bottom.
    ``mid_y``: from one side, pads on its top and bottom. ``mid_z``: from above or below, pads on its sides.
    ``s`` chooses which pad is on which face, ``a`` which side the hand comes from."""
    import dronekit as K

    if kind == "side":
        z, x, at, w = (0.0, 0.0, -1.0), (0.0, float(s), 0.0), (0.0, 0.0, 0.0), (SIDE_OPEN, SIDE_GRIP)
    elif kind == "tail":
        z, x, at, w = (1.0, 0.0, 0.0), (0.0, 0.0, float(s)), (-K.LEN / 2 + TAIL_IN, 0.0, 0.0), (FLAT_OPEN, FLAT_GRIP)
    elif kind == "mid_y":
        z, x, at, w = (0.0, float(a), 0.0), (0.0, 0.0, float(s)), (MID_Y_AT, 0.0, 0.0), (FLAT_OPEN, FLAT_GRIP)
    else:
        z, x, at, w = (0.0, 0.0, float(a)), (0.0, float(s), 0.0), (MID_Z_AT, 0.0, 0.0), (SIDE_OPEN, SIDE_GRIP)
    return frame(z, x), np.array(at), w[0], w[1]


class Est:
    """The script's own account of where the drone is: its middle (world) and its yaw and pitch in degrees."""

    def __init__(self, T):
        pos, _ = T.drone_start()
        self.pos, self.yaw, self.pitch = np.array(pos, float), math.degrees(T.YAW), 0.0
        self.top = T.TOP
        EST["est"] = self

    def rot(self) -> np.ndarray:
        return drone_rot(self.yaw, self.pitch)

    def tool_back(self, g, want: float = BACK) -> tuple:
        """Where a hand that has let go of grasp ``g`` and drawn back is: (tool point, rotation)."""
        pos, rot = self.tool(g)
        return pos - back_of(self, pos, rot, want) * rot[:, 2], rot

    def tool(self, g, pos=None, yaw=None, pitch=None) -> tuple:
        """The tool point and rotation of a hand with grasp ``g`` on the drone here (or at the pose given)."""
        r = drone_rot(self.yaw if yaw is None else yaw, self.pitch if pitch is None else pitch)
        return (self.pos if pos is None else np.asarray(pos, float)) + r @ g[1], r @ g[0]


def reachable(hand, pos, rot, seed=None, iters: int = 200):
    """Joint angles for a pose, from ``seed`` (default: where the hand is), or None if it cannot be reached with joint
    travel in hand."""
    from yam_arena import ik

    q, pe, re = solve_ik(pos, rot, hand.q if seed is None else seed, hand.base, iters=iters)
    if pe > 5e-4 or re > 5e-3 or float(np.minimum(q - ik.LOWER, ik.UPPER - q).min()) < ROOM:
        return None
    return q


def reach_any(hand, pos, rot):
    """Like ``reachable``, trying a few starting points; the answer with most joint travel in hand."""
    from yam_arena import ik

    best = None
    for seed in [hand.q.copy()] + [np.array(s, float) for s in ([0, 1, 1, -0.6, 0, 0], [0, 1, 1, -0.6, 0, 1.57], [0, 1, 1, -0.6, 0, -1.57],
                                                             [0.6, 1.2, 1.0, 0, 0.8, 0], [-0.6, 1.2, 1.0, 0, -0.8, 0],
                                                             [0.8, 1.5, 1.5, -0.3, 1.2, 1.0], [-0.8, 1.5, 1.5, -0.3, -1.2, -1.0])]:
        q = reachable(hand, pos, rot, seed)
        if q is not None:
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    return best


def yaw_span(hand, est, g, q0, step: float = 5.0, limit: float = 400.0) -> tuple:
    """How far the drone's yaw can go down and up from where it is with ``hand`` holding it by ``g``, the hand's joints
    moving continuously from ``q0``: (lowest yaw, highest yaw)."""
    ends = []
    for sgn in (-1.0, 1.0):
        q, a = np.array(q0, float), 0.0
        while abs(a) < limit:
            p, r = est.tool(g, yaw=est.yaw + a + sgn * step)
            q2 = reachable(hand, p, r, q, iters=60)
            if q2 is None or float(np.abs(q2 - q).max()) > 0.5:
                break
            q, a = q2, a + sgn * step
        ends.append(est.yaw + a)
    return ends[0], ends[1]


def carry(est, sag, g, seconds: float, pos=None, yaw=None, pitch=None, hold: float = 0.0):
    """Generator: the hand holding the drone by ``g`` takes it smoothly to a new place, yaw or pitch."""
    from yam_arena import ik

    sag.live = True
    p0, y0, b0 = est.pos.copy(), est.yaw, est.pitch
    p1 = p0 if pos is None else np.asarray(pos, float)
    y1, b1 = (y0 if yaw is None else yaw), (b0 if pitch is None else pitch)
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        est.pos, est.yaw, est.pitch = p0 + s * (p1 - p0), y0 + s * (y1 - y0), b0 + s * (b1 - b0)
        sag.to(*est.tool(g))
        yield
    for _ in range(round(hold * CONTROL_HZ)):
        sag.to(*est.tool(g))
        yield


def keep(est, sag, g, *gens):
    """Generator: run ``gens`` one after another while the hand holding the drone by ``g`` keeps it where the script
    expects it, correcting for the sag the drone's weight gives it."""
    sag.live = True
    for gen in gens:
        for _ in gen:
            sag.to(*est.tool(g))
            yield


def spin(sag, pos, rot0, degrees: float, seconds: float):
    """Generator: turn an empty hand in place about the vertical."""
    from yam_arena import ik

    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        sag.to(pos, _rz(math.radians(ik.smoothstep(k / n) * degrees)) @ rot0)
        yield


def back_of(est, pos, rot, want: float) -> float:
    """How far back from a grasp a hand can wait: ``want``, or less for a hand pointing straight down, which cannot
    hold its tool point more than ``Z_CAP`` above the table."""
    if want > 0.0 and rot[2, 2] < -0.9:
        return max(0.03, min(want, est.top + Z_CAP - float(pos[2])))
    return want


def take(est, sag, g, name: str, settle: float = 0.3):
    """Generator: an empty hand, waiting behind its grasp, comes in along its own pointing axis and closes."""
    pos, rot = est.tool(g)
    far = float(np.linalg.norm((sag.want if sag.want is not None else sag.hand.pos) - pos))
    yield from glide(sag, pos, max(1.0, far / 0.07), rot)
    yield from glide(sag, pos, settle, rot)
    sag.live = False                       # from here the drone, not the air, is what the hand turns against
    yield from sag.hand.set_grip(gap(g[3]), 0.7)
    HOLD[name] = g
    yield from wait(0.2)


def let_go(est, sag, g, name: str, back: float = BACK):
    """Generator: open, and draw back along the hand's own pointing axis."""
    yield from sag.hand.set_grip(gap(g[2]), 0.5)
    HOLD[name] = None
    sag.live = True
    pos, rot = est.tool(g)
    back = back_of(est, pos, rot, back)
    if back:
        yield from glide(sag, pos - back * rot[:, 2], max(0.8, back / 0.07), rot)


def come(est, sag, g, back: float, seconds: float = 2.6, yaw=None, pos=None, pitch=None):
    """Generator: take an empty hand in a straight line to wait ``back`` behind grasp ``g`` (on the drone where it is,
    or where it is going to be), turning and opening for it on the way."""
    p, rot = est.tool(g, pos, yaw, pitch)
    if sag.want is None:
        sag.want, sag.want_rot = sag.hand.pos.copy(), None
    yield from both(glide(sag, p - back_of(est, p, rot, back) * rot[:, 2], seconds, rot, rot_from=sag.hand.rot.copy()),
                    sag.hand.set_grip(gap(g[2]), 0.6))


def home(sag, seconds: float = 2.2, straight: bool = True):
    """Generator: take an empty hand back to where it is at the arm's ready pose, in a straight line or (``straight``
    False) by swinging its joints."""
    from yam_arena import ik
    from yam_arena.constants import READY

    if straight:
        pos, rot = ik.tcp_pose(np.array(READY, float), sag.hand.base)
        yield from both(glide(sag, pos, seconds, rot, rot_from=sag.hand.rot.copy()), sag.hand.set_grip(1.0, 0.6))
    else:
        yield from sag.hand.move_joints(np.array(READY, float), seconds, grip=1.0)
        sag.want = None


def pick_grasp(hand, est, options, pos=None, yaw=None, pitch=None):
    """The grasp among ``options`` the hand reaches best, on the drone where it is (or at the pose given): the one it
    can wait furthest back from, then the one that leaves its joints most travel. Returns (grasp, joint angles at the
    grasp, how far back it waits), or None."""
    from yam_arena import ik

    best = None
    for g in options:
        p, r = est.tool(g, pos, yaw, pitch)
        for want in WAITS:
            back = back_of(est, p, r, want)
            if (p - back * r[:, 2])[0] < NEAR_BASE:
                continue
            far = reach_any(hand, p - back * r[:, 2], r)
            if far is None:
                continue
            q, ok = far[0], True
            for k in range(1, 9):                       # the straight way in has to be within reach all along
                q = reachable(hand, p - back * (1.0 - k / 8.0) * r[:, 2], r, q, iters=80)
                if q is None:
                    ok = False
                    break
            if not ok:
                continue
            room = min(far[1], float(np.minimum(q - ik.LOWER, ik.UPPER - q).min()), EASY)
            if best is None or (room, back) > (best[3], best[2]):
                best = (g, q, back, room)
    return None if best is None else best[:3]


def plan_turn(turner, est, g, parker, park_options, q0) -> dict:
    """Plan one full turn of the drone's yaw. ``turner`` holds it by ``g`` (joints ``q0``); the hand ``parker`` will
    hold it by one of ``park_options`` while the turner turns back half a turn. Returns the yaws to go through:
    ``first`` (where the turn starts from, reached from the present yaw), ``swap`` (where the exchange is made) and
    ``last`` (where the turn ends), the parker's grasp, and the turner's grasp after the exchange."""
    lo, hi = yaw_span(turner, est, g, q0)
    g2 = (g[0] @ _rz(math.pi), g[1], g[2], g[3])          # the same faces, the pads changed over
    best = None
    for sense in (1.0, -1.0):
        # the exchange has to be within reach both before and after the turner's half turn back
        a, b = (lo + 180.0, hi) if sense > 0 else (lo, hi - 180.0)
        y = a
        while y <= b + 1e-6:
            got = pick_grasp(parker, est, park_options, yaw=y)
            if got is not None:
                first = y - sense * (180.0 + OVER)
                first = max(first, lo) if sense > 0 else min(first, hi)
                total = abs(y - first) + 180.0 - 5.0
                if total >= 360.0 + 10.0:
                    from yam_arena import ik
                    room = float(np.minimum(got[1] - ik.LOWER, ik.UPPER - got[1]).min())
                    if best is None or room > best["room"]:
                        best = {"first": first, "swap": y, "last": y + sense * (360.0 + OVER - abs(y - first)), "sense": sense,
                                "park": got[0], "after": g2, "room": room, "lo": lo, "hi": hi, "back": got[2]}
            y += 5.0
    if best is None:
        raise RuntimeError(f"no exchange found: the turner's yaw runs from {lo:.0f} to {hi:.0f}")
    return best


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    if mode == "still":
        yield from wait(idle_s)
        return
    ls, rs = Sag(left), Sag(right)
    SAGS.update(left=ls, right=rs)
    HOLD.update(left=None, right=None)
    est = Est(T)
    top = T.TOP

    # 1. The right hand takes the drone by its sides and lifts it off the pad.
    PHASE["name"] = "pick"
    left.grip = right.grip = 1.0
    g_side, _, back = pick_grasp(right, est, [grasp("side", 1), grasp("side", -1)])
    yield from come(est, rs, g_side, back)
    yield from take(est, rs, g_side, "right")
    lifted = np.array([LEVEL_XY[0], LEVEL_XY[1], top + H_LEVEL])
    yield from carry(est, rs, g_side, 1.0, pos=est.pos + 0.05 * UP)
    yield from carry(est, rs, g_side, 1.6, pos=lifted, pitch=30.0 if mode == "tilted" else None)

    # 2. The level turn.
    PHASE["name"] = "level turn"
    plan = plan_turn(right, est, g_side, left, [grasp("tail", 1), grasp("tail", -1)], right.q)
    PLAN["level"] = {k: round(v, 1) for k, v in plan.items() if isinstance(v, float)}
    g_tail = plan["park"]
    turn_s = lambda a, b: abs(a - b) / TURN_RATE + 0.4      # noqa: E731

    def right_turns():
        yield from carry(est, rs, g_side, turn_s(plan["first"], est.yaw), yaw=plan["first"])
        yield from carry(est, rs, g_side, turn_s(plan["swap"], plan["first"]), yaw=plan["swap"], hold=0.2)

    yield from both(right_turns(), come(est, ls, g_tail, plan["back"], 2.6, yaw=plan["swap"]))
    if mode == "down":
        # control: stand the drone on the table for the exchange instead of passing it from hand to hand
        PHASE["name"] = "set down"
        rest = np.array([est.pos[0], est.pos[1], top + T.PAD_T + K_STAND + 0.0005])
        yield from carry(est, rs, g_side, 2.0, pos=rest, hold=0.3)
        yield from let_go(est, rs, g_side, "right")
        yield from spin(rs, rs.want, rs.want_rot, -plan["sense"] * 180.0, 180.0 / BACK_RATE)
        g_side = plan["after"]
        yield from take(est, rs, g_side, "right")
        yield from carry(est, rs, g_side, 2.0, pos=lifted)
    else:
        PHASE["name"] = "level exchange"
        yield from keep(est, rs, g_side, take(est, ls, g_tail, "left"))
        g_old, g_side = g_side, plan["after"]
        yield from keep(est, ls, g_tail, let_go(est, rs, g_old, "right"),
                        spin(rs, *est.tool_back(g_old), -plan["sense"] * 180.0, 180.0 / BACK_RATE), take(est, rs, g_side, "right"))
        yield from keep(est, rs, g_side, let_go(est, ls, g_tail, "left", back=plan["back"]))
    PHASE["name"] = "level turn on"
    if mode != "short":
        yield from carry(est, rs, g_side, turn_s(plan["last"], est.yaw), yaw=plan["last"], hold=0.2)
        yield from carry(est, rs, g_side, turn_s(plan["swap"], est.yaw), yaw=plan["swap"])
    if mode == "tilted":
        yield from carry(est, rs, g_side, 1.0, pitch=0.0)
    pad = np.array([T.PAD_XY[0], T.PAD_XY[1], top + T.PAD_T + K_STAND])
    if mode == "level_only":
        # control: after the level turn, stand the drone back on its pad and stop
        PHASE["name"] = "put down"
        yield from keep(est, rs, g_side, home(ls))
        yield from carry(est, rs, g_side, 1.6, pos=pad + 0.0005 * UP, hold=0.4)
        yield from let_go(est, rs, g_side, "right")
        yield from home(rs)
        yield from wait(3.0)
        return

    # 3. The left hand takes the tail; the right lets go; the left swings down, and the drone hangs nose down.
    PHASE["name"] = "to upright"
    if mode == "down":
        yield from keep(est, rs, g_side, come(est, ls, g_tail, plan["back"], 2.6))
    yield from keep(est, rs, g_side, take(est, ls, g_tail, "left"))
    yield from keep(est, ls, g_tail, let_go(est, rs, g_side, "right"), home(rs))
    on_end = np.array([ON_END_XY[0], ON_END_XY[1], top + H_ON_END])
    yield from carry(est, ls, g_tail, 4.0, pos=on_end, pitch=90.0, hold=0.3)

    # 4. The upright turn.
    PHASE["name"] = "upright turn"
    mids = [grasp(k, s, a) for k in ("mid_y", "mid_z") for s in (1, -1) for a in (1, -1)]
    plan2 = plan_turn(left, est, g_tail, right, mids, left.q)
    PLAN["upright"] = {k: round(v, 1) for k, v in plan2.items() if isinstance(v, float)}
    g_mid = plan2["park"]
    yield from carry(est, ls, g_tail, turn_s(plan2["first"], est.yaw), yaw=plan2["first"])
    yield from carry(est, ls, g_tail, turn_s(plan2["swap"], plan2["first"]), yaw=plan2["swap"], hold=0.2)
    PHASE["name"] = "upright exchange"
    yield from keep(est, ls, g_tail, come(est, rs, g_mid, plan2["back"], 2.6), take(est, rs, g_mid, "right"))
    yield from keep(est, rs, g_mid, let_go(est, ls, g_tail, "left", back=0.0))
    spot = est.tool(g_tail)
    yield from carry(est, rs, g_mid, 1.4, pos=est.pos - DIP * UP)       # the left hand cannot rise: the drone goes down
    yield from keep(est, rs, g_mid, spin(ls, spot[0], spot[1], -plan2["sense"] * 180.0, 180.0 / BACK_RATE))
    yield from carry(est, rs, g_mid, 1.4, pos=est.pos + DIP * UP, hold=0.2)
    g_tail = plan2["after"]

    def left_closes():
        ls.live = False
        yield from left.set_grip(gap(g_tail[3]), 0.7)
        HOLD["left"] = g_tail
        yield from wait(0.2)

    yield from keep(est, rs, g_mid, left_closes())
    yield from keep(est, ls, g_tail, let_go(est, rs, g_mid, "right", back=plan2["back"]), home(rs))
    PHASE["name"] = "upright turn on"
    yield from carry(est, ls, g_tail, turn_s(plan2["last"], est.yaw), yaw=plan2["last"], hold=0.2)

    # 5. Back to level, into the right hand, and down on the pad.
    PHASE["name"] = "to level"
    back_yaw, g_put, back = level_out(left, right, est, g_tail, lifted)
    PLAN["out"] = {"yaw": round(back_yaw, 1)}
    yield from carry(est, ls, g_tail, turn_s(back_yaw, est.yaw), yaw=back_yaw)
    yield from carry(est, ls, g_tail, 4.0, pos=lifted, pitch=0.0, hold=0.3)
    PHASE["name"] = "put down"
    yield from keep(est, ls, g_tail, come(est, rs, g_put, back), take(est, rs, g_put, "right"))
    yield from keep(est, rs, g_put, let_go(est, ls, g_tail, "left", back=WAITS[-1]), home(ls, straight=False))
    square = round(est.yaw / 90.0) * 90.0                      # feet square to the pad
    lo, hi = yaw_span(right, est, g_put, right.q)
    if not lo <= square <= hi:
        square = est.yaw
    if mode == "off_pad":
        # control: stand the drone on the bare table beside its pad
        pad = np.array([T.PAD_XY[0] + OFF_PAD[0], T.PAD_XY[1] + OFF_PAD[1], top + K_STAND])
    yield from carry(est, rs, g_put, turn_s(square, est.yaw) + 0.6, pos=pad + 0.06 * UP, yaw=square)
    yield from carry(est, rs, g_put, 1.6, pos=pad + 0.0005 * UP, hold=0.4)
    yield from let_go(est, rs, g_put, "right")
    yield from home(rs)
    yield from wait(3.0)


PLAN: dict = {}
OFF_PAD = (0.0, -0.210)                  # the off_pad control stands the drone this far from the pad's centre
K_STAND = 0.0275                         # the drone's middle above what it stands on (dronekit.STAND)


def level_out(left, right, est, g_tail, lifted) -> tuple:
    """Where to turn the hanging drone to before the left hand swings it back to level: a yaw the left wrist can reach
    from which the swing up works and the right hand can then take the sides from above. Returns (yaw, the right
    hand's grasp)."""
    lo, hi = yaw_span(left, est, g_tail, left.q)
    best = None
    PLAN["out_span"] = (round(lo, 1), round(hi, 1), [])
    y = lo
    while y <= hi + 1e-6:
        # follow the left hand's joints from here to that yaw and through the swing up
        q, ok = left.q.copy(), True
        steps = [(est.pos, yy, 90.0) for yy in np.linspace(est.yaw, y, max(2, int(abs(y - est.yaw) / 5.0) + 1))]
        steps += [(est.pos + s * (lifted - est.pos), y, 90.0 * (1.0 - s)) for s in np.linspace(0.0, 1.0, 19)[1:]]
        room = 9.0
        for p, yy, b in steps:
            tp, tr = est.tool(g_tail, p, yy, b)
            q2 = reachable(left, tp, tr, q, iters=80)
            if q2 is None or float(np.abs(q2 - q).max()) > 0.5:
                ok = False
                PLAN["out_span"][2].append((round(y), round(b)))
                break
            from yam_arena import ik
            room = min(room, float(np.minimum(q2 - ik.LOWER, ik.UPPER - q2).min()))
            q = q2
        if ok:
            keep = (est.pos.copy(), est.yaw, est.pitch)
            est.pos, est.yaw, est.pitch = np.array(lifted, float), y, 0.0
            got = pick_grasp(right, est, [grasp("side", 1), grasp("side", -1)])
            est.pos, est.yaw, est.pitch = keep
            if got is not None and (best is None or room > best[3]):
                best = (y, got[0], got[2], room)
        y += 5.0
    if best is None:
        raise RuntimeError("no way back to level found")
    return best[:3]


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 3000, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "full", every: int = 5, cam: tuple = (320, 176)) -> dict:
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
    obs = emb.reset(Scene(id="drone-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def drone() -> list:
        from yam_arena.scene import object_states   # importable only once the simulator has started

        st = object_states(emb._env, [T.DRONE])[T.DRONE]
        att = T.K.attitude(st.quat_xyzw)
        return [round(1000 * float(v), 1) for v in st.pos - [0.0, 0.0, T.TOP]] + [round(att[k], 1) for k in ("up_tilt", "plumb_tilt", "nose", "top")]

    def expected() -> list:
        e = EST.get("est")
        if e is None:
            return []
        return [round(1000 * float(v), 1) for v in e.pos - [0.0, 0.0, T.TOP]] + [round(e.yaw % 360.0, 1), round(e.pitch, 1)]

    def measure() -> None:
        from dronekit import finger_frames, finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        TOOLROT["left"], TOOLROT["right"] = finger_frames(emb._env)

    def hands() -> dict:
        from dronekit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1),
                            "off_deg": round(SAGS[h.name].off, 2) if h.name in SAGS else None}
        return out_

    snap_frames(obs)
    measure()
    start = {"task": T.report(None), "drone": drone()}
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
            trace.append({"step": steps, "phase": PHASE["name"], "score": info.get("score"), "task": T.report(None),
                          "hands": hands(), "drone": drone(), "expected": expected(),
                          "hold": [n for n in ("left", "right") if HOLD[n] is not None]})
        if steps % 40 == 0 and trace:
            t = trace[-1]
            print(f"step {steps} {t['phase']} score {t['score']} {t['task']} drone {t['drone']} expected {t['expected']} hold {t['hold']} L {t['hands']['left']} R {t['hands']['right']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "plan": {k: v for k, v in PLAN.items() if k != "out_span"},
              "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended,
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": {"task": T.report(None), "drone": drone()}, "trace": trace}
    if record:
        import imageio

        for name, fs in frames.items():
            _write_mp4(fs, out / f"demo_{name}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item()))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "short", "tilted", "down", "level_only", "off_pad", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
