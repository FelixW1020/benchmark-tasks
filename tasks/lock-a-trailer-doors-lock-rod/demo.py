"""Demo for the trailer door lock-rod task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 900] [--no-record]
                                   [--mode full|cam|nolift|seat|held|false|still|look]

1. The left hand, level and turned to face the handle's flat side, closes on the handle 20 cm out from the pipe, one
   finger above the bar and one below.
2. It swings the handle in on its arc, turning with it. The cam's prongs go either side of the keeper pin on the sill.
3. Eight degrees short of locked it lifts the handle 4.5 degrees on its pin, carries it over the hasp's lip against the
   door seal's push-back until the bar is flat on the hasp's face, and lets it down into the seat. It lets go and
   draws back; the handle springs out against the lip and stays.
4. The right hand, fingers closed, pushes the flip keeper's tab over its top; the keeper falls and hangs over the
   handle.

Controls: ``--mode cam`` swings the handle to 10 degrees and lets go (0.25); ``--mode nolift`` swings it all the way
without lifting, so it stops against the hasp's lip (0.25); ``--mode seat`` seats the handle and leaves the keeper up
(0.5); ``--mode held`` does everything but leaves the right hand where it pushed the keeper over, about 2 cm from it (0.75); ``--mode false`` flips the keeper down
first and then swings the handle to 10 degrees (trial 0); ``--mode still`` keeps both arms still. ``--mode look`` is
not a trial: it raises the right arm so that its wrist camera sees the hasp, for the setup picture.

Every target comes from the layout constants in the task file and the sizes in ``doorkit.py``. The only feedback is
each hand's own measured position and attitude (the robot's proprioception), used to correct for the arm's sag. The
demo never reads a part's state.
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
JOINT_K = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])   # N m/rad: the arm joints' position gains (yam_arena constants)


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
        self.push = np.zeros(3)             # N, world: the force this hand is to hold on what it touches
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = self.worst_deg = 0.0

    def lean(self) -> np.ndarray:
        """The joint angles to command so that the joints' springs hold ``push`` at the tool point: the pose's own
        angles plus (Jacobian transpose times the force) over each joint's stiffness."""
        if not np.any(self.push):
            return self.q
        from yam_arena import ik

        p0 = np.asarray(ik.tcp_pose(self.q, self.base)[0], float)
        jac = np.zeros((3, 6))
        for i in range(6):
            dq = self.q.copy()
            dq[i] += 1e-4
            jac[:, i] = (np.asarray(ik.tcp_pose(dq, self.base)[0], float) - p0) / 1e-4
        return np.clip(self.q + (jac.T @ self.push) / JOINT_K, ik.LOWER, ik.UPPER)

    def _solve(self, pos, rot) -> None:
        from yam_arena import ik

        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, float(np.degrees(re)))
        self.cmd = np.r_[self.lean(), self.grip]

    def move(self, pos, duration: float, rot=None, grip: float | None = None):
        """Generator: go in a straight line to ``pos`` (world, tool point) and ``rot``; ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        r1 = r0 if rot is None else rot
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, r1, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.5):
        """Generator: open or close to ``grip`` in place."""
        from yam_arena import ik

        g0 = self.grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.grip = g0 + ik.smoothstep(k / n) * (grip - g0)
            self.cmd = np.r_[self.lean(), self.grip]
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
            self.cmd = np.r_[self.lean(), self.grip]
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def chain(*gens):
    for g in gens:
        yield from g


def together(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def best_ik(hand, pos, rot):
    """Joint angles for ``pos``/``rot`` from the best of a few seeds (the hand's own, and its wrist turned either way)."""
    from yam_arena import ik

    best = None
    for q6 in (None, -1.0, 1.0):
        seed = hand.q.copy()
        if q6 is not None:
            seed[5] = q6
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        cost = pe + 0.05 * re
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0]


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
GAIN, BIAS_MAX = 0.25, 0.100             # how fast, and how far, a hand corrects for its own sag

ROT: dict = {}                           # each hand's measured attitude (updated every control step)
RGAIN, TURN_MAX = 0.20, math.radians(10.0)   # how fast, and how far, a hand corrects its own droop in attitude


def rotvec(m) -> np.ndarray:
    """The turn (axis times angle, rad) that a rotation matrix makes."""
    m = np.asarray(m, float)
    c = max(-1.0, min(1.0, (m[0, 0] + m[1, 1] + m[2, 2] - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n < 1e-9 else v / n * ang


def turned(w) -> np.ndarray:
    """The rotation matrix of a turn ``w`` (axis times angle, rad)."""
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = np.asarray(w, float) / ang
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * kx + (1.0 - math.cos(ang)) * (kx @ kx)


class Sag:
    """Slowly offsets a hand's command by its measured error, in place and in attitude, so that its tool point and
    its fingers end up where they are wanted under load. ``cap`` limits the offset along each world axis (m): a hand
    that grips a part held by something else must not wind up against it."""

    def __init__(self, hand, cap=(BIAS_MAX, BIAS_MAX, BIAS_MAX)):
        self.hand, self.bias, self.want = hand, np.zeros(3), None
        self.turn, self.rot = np.zeros(3), None
        self.cap = np.asarray(cap, float)

    def to(self, pos, rot):
        name = self.hand.name
        if self.want is not None:
            self.bias = np.clip(self.bias + GAIN * (self.want - TOOL[name]), -self.cap, self.cap)
            have = ROT.get(name)
            if have is not None and self.rot is not None:
                err = rotvec(self.rot @ np.asarray(have, float).T)          # from where the hand is to where it is wanted
                if float(np.linalg.norm(err)) < math.radians(20.0):
                    self.turn = self.turn + RGAIN * err
                    size = float(np.linalg.norm(self.turn))
                    if size > TURN_MAX:
                        self.turn *= TURN_MAX / size
        self.want, self.rot = np.asarray(pos, float), np.asarray(rot, float)
        self.hand._solve(self.want + self.bias, turned(self.turn) @ self.rot)

    def ease(self, seconds: float):
        """Generator: let the offsets go, in place (before a hand lets go of something that is now held elsewhere)."""
        n = max(1, round(seconds * CONTROL_HZ))
        b0, t0 = self.bias.copy(), self.turn.copy()
        for k in range(1, n + 1):
            self.bias, self.turn = b0 * (1.0 - k / n), t0 * (1.0 - k / n)
            self.hand._solve(self.want + self.bias, turned(self.turn) @ self.rot)
            yield


def glide(sag, target, seconds: float, rot):
    """Generator: take the hand's tool point in a straight line to ``target``, correcting for sag on the way."""
    from yam_arena import ik

    start = sag.want if sag.want is not None else sag.hand.pos.copy()
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        sag.to(start + ik.smoothstep(k / n) * (np.asarray(target, float) - start), rot)
        yield



LEVER = 0.200                            # the left hand takes the handle this far out from the pipe's axis
TCP_BACK = 0.003                         # the tool point is this far short of the bar's middle (toward the robot's side)
BAR_GAP = 0.008                          # the pads are sent to this gap on the 25 mm bar (a firm pinch, about 40 N a pad)
LEVER_R = 0.352                          # the right hand's closed fingertips push the bar this far out, beyond the hasp
SHARE_R = 0.80                           # the share of the seal's torque that the right hand's push carries
PUSH_FROM = 22.0                         # degrees from locked at which the right hand joins, before the seal is met
HANDLE_UP = 5.4                          # N: the lifted handle's weight on the left hand
REST = 0.8                               # degrees from locked at which the seated handle rests against the hasp's lip
OVER = 0.1                               # degrees from locked at which the lifted handle is let down: between the hasp's face and its lip
CAP_HELD = (0.060, 0.012, 0.030)         # m: how far the gripping hand may offset its command along x, y, z while it holds the handle
CAP_PUSH = (0.020, 0.012, 0.030)         # ... and the pushing hand
TIP_PAST = 0.013                         # the closed fingertips' end, ahead of the tool point (yam_arena constants)
R_CLOSE = __import__("os").environ.get("DOOR_RCLOSE", "along")   # the pushing hand's finger axis: along the bar, or up
UP_SIGN = float(__import__("os").environ.get("DOOR_UP", "1"))   # which finger is above the bar (+1: the tool's x up)
LIFT = 4.5                               # degrees the handle is lifted to clear the hasp's lip
LIFT_AT = 8.0                            # ... from this many degrees short of locked
PAST = 0.6                               # the handle is sent this many degrees past locked (the hasp's face stops it)
R_FLIP = frame((math.cos(math.radians(30.0)), 0.0, -math.sin(math.radians(30.0))), (0.0, -1.0, 0.0))   # the right hand
TIP_AHEAD = 0.011                        # the fingertips' end, forward of the tool point (level distance, hand 30 degrees down)
FINGER_HALF = 0.008                      # the closed fingers' side face, out from the tool point
TAB_UP = 0.036                           # the flipped-up keeper's tab above its pin
TAB_OUT = 0.010                          # the fingertip's end this far in front of the keeper's plate
FLIP_FROM, FLIP_TO = -0.056, 0.024       # the right hand's tool point, along y from the keeper's pin: start and end of the push


def hold(T, theta: float, lift: float, back: float = 0.0):
    """The left hand's tool point and attitude for a handle at ``theta``/``lift`` degrees (``back`` m drawn back)."""
    p, into, up, _ = T.K.grip_pose(T.door_pos(), theta, lift, LEVER)
    return p - (TCP_BACK + back) * into, frame(into, UP_SIGN * up)


def sweep(sag, T, a0, a1, seconds: float):
    """Generator: take the held handle from (theta, lift) ``a0`` to ``a1``."""
    from yam_arena import ik

    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        sag.to(*hold(T, a0[0] + s * (a1[0] - a0[0]), a0[1] + s * (a1[1] - a0[1])))
        yield


def stay(sag, T, a, seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        sag.to(*hold(T, *a))
        yield


def push_at(T, theta: float, lift: float, back: float = 0.0):
    """The right hand's tool point and attitude with its closed fingertips on the bar's outer face, out near the tip."""
    p, into, up, along = T.K.grip_pose(T.door_pos(), theta, lift, LEVER_R)
    return p - (T.K.BAR_T / 2 + TIP_PAST + back) * into, frame(into, along if R_CLOSE == "along" else up)


def seal_load(T, theta: float, lift: float, scale: float = 1.0):
    """The forces (world, N) the left and the right hand hold on the handle against the door's seal at ``theta``:
    the fixture's stated spring (and the rod's drag) shared between them, and the lifted handle's weight on the left."""
    torque = scale * (T.K.seal_torque(theta) + T.K.ROD_DRAG)
    _, into, _, _ = T.K.grip_pose(T.door_pos(), theta, lift, LEVER)
    fl = (1.0 - SHARE_R) * torque / LEVER * into
    if lift > 0.3:
        fl = fl + np.array([0.0, 0.0, HANDLE_UP])
    return fl, SHARE_R * torque / LEVER_R * into


def sweep2(ls, rs, T, a0, a1, seconds: float, k0: float = 1.0, k1: float = 1.0):
    """Generator: both hands take the handle from (theta, lift) ``a0`` to ``a1``, each holding its share of the
    seal's push-back (scaled from ``k0`` to ``k1``)."""
    from yam_arena import ik

    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        th, li = a0[0] + s * (a1[0] - a0[0]), a0[1] + s * (a1[1] - a0[1])
        ls.hand.push, rs.hand.push = seal_load(T, th, li, k0 + s * (k1 - k0))
        ls.to(*hold(T, th, li))
        rs.to(*push_at(T, th, li))
        yield


def join_push(T, right, ls):
    """The right hand, fingers closed, comes up behind the handle's tip while the left hand holds the handle still."""
    a = (PUSH_FROM, 0.0)
    pre, rot = push_at(T, PUSH_FROM, 0.0, back=0.060)
    yield from together(stay(ls, T, a, 2.5), right.move_joints(best_ik(right, pre, rot), 2.5, grip=0.0))
    rs = Sag(right)
    yield from together(stay(ls, T, a, 0.3), glide(rs, pre, 0.3, rot))
    yield from together(stay(ls, T, a, 1.0), glide(rs, push_at(T, PUSH_FROM, 0.0, back=0.002)[0], 1.0, rot))
    rs.cap = np.asarray(CAP_PUSH, float)
    rs.bias = np.clip(rs.bias, -rs.cap, rs.cap)
    return rs


def leave_push(T, right, rs, theta: float):
    """The right hand draws straight back from the bar."""
    right.push = np.zeros(3)
    pos, rot = push_at(T, theta, 0.0, back=0.070)
    yield from glide(rs, pos, 1.0, rot)


def seat_handle2(T, left, right, ls, lift: bool = True):
    """Both hands: swing the handle in against the seal, lift it over the hasp's lip, carry it in between the lip and
    the hasp's face, let it down into its seat and ease off until the lip holds it."""
    yield from sweep(ls, T, (T.THETA_START, 0.0), (PUSH_FROM, 0.0), 3.0)
    ls.cap = np.asarray(CAP_HELD, float)
    ls.bias = np.clip(ls.bias, -ls.cap, ls.cap)
    rs = yield from join_push(T, right, ls)
    up = LIFT if lift else 0.0
    yield from sweep2(ls, rs, T, (PUSH_FROM, 0.0), (LIFT_AT, 0.0), 2.5)
    yield from sweep2(ls, rs, T, (LIFT_AT, 0.0), (LIFT_AT, up), 1.2)
    yield from sweep2(ls, rs, T, (LIFT_AT, up), (OVER, up), 2.0)
    yield from sweep2(ls, rs, T, (OVER, up), (OVER, up), 0.6)
    yield from sweep2(ls, rs, T, (OVER, up), (OVER, -0.5), 1.2)
    yield from sweep2(ls, rs, T, (OVER, -0.5), (OVER, -0.5), 0.6)
    yield from sweep2(ls, rs, T, (OVER, -0.5), (REST, -0.5), 1.5, 1.0, 0.0)      # ease off: the lip takes the seal's push
    left.push = np.zeros(3)
    yield from together(chain(stay(ls, T, (REST, -0.5), 0.4), ls.ease(0.6)), leave_push(T, right, rs, REST))


def take_handle(T, left):
    """The left hand closes on the handle where it hangs."""
    t0 = T.THETA_START
    pre, rot = hold(T, t0, 0.0, back=0.060)
    yield from left.move_joints(best_ik(left, pre, rot), 2.5, grip=gap(0.060))     # (a straight line from rest runs a joint to its stop)
    s = Sag(left)
    yield from glide(s, pre, 0.3, rot)
    yield from glide(s, hold(T, t0, 0.0)[0], 1.5, rot)
    yield from left.set_grip(gap(BAR_GAP), 0.8)
    yield from wait(0.3)
    return s


def let_go(T, left, s, theta: float):
    """The left hand opens and draws straight back from the handle."""
    yield from s.ease(0.4)
    yield from left.set_grip(gap(0.060), 0.6)
    yield from wait(0.3)
    pos, rot = hold(T, theta, 0.0, back=0.060)
    yield from glide(s, pos, 1.2, rot)


def seat_handle(T, left, s, lift: bool = True):
    """Swing the handle in, lift it over the hasp's lip, press it flat on the hasp's face and let it down."""
    t0 = T.THETA_START
    yield from sweep(s, T, (t0, 0.0), (LIFT_AT, 0.0), 4.0)
    up = LIFT if lift else 0.0
    yield from sweep(s, T, (LIFT_AT, 0.0), (LIFT_AT, up), 1.2)
    yield from sweep(s, T, (LIFT_AT, up), (-PAST, up), 1.5)
    yield from stay(s, T, (-PAST, up), 0.5)
    yield from sweep(s, T, (-PAST, up), (-PAST, -0.5), 1.2)
    yield from stay(s, T, (-PAST, -0.5), 0.5)


def flip_keeper(T, right, stay_by: bool = False):
    """The right hand, fingers closed, pushes the keeper's tab over its top; the keeper falls and hangs."""
    pin = T.door_pos() + T.K.KEEP_AT
    x = pin[0] - T.K.KEEP_T / 2 - TAB_OUT - TIP_AHEAD
    start = np.array([x, pin[1] + FLIP_FROM, pin[2] + TAB_UP])
    end = np.array([x, pin[1] + FLIP_TO, pin[2] + TAB_UP])
    yield from right.move(start + (-0.060, 0.0, 0.030), 2.5, rot=R_FLIP, grip=0.0)
    s = Sag(right)
    yield from glide(s, start + (-0.060, 0.0, 0.030), 0.3, R_FLIP)
    yield from glide(s, start, 1.0, R_FLIP)
    yield from glide(s, end, 1.5, R_FLIP)
    yield from wait(0.5)
    if stay_by:                                                  # control: the hand stays where it pushed the keeper over,
        yield from wait(5.0)                                     # about 2 cm from where the keeper now hangs
        return
    yield from glide(s, end + (-0.080, 0.0, 0.020), 1.0, R_FLIP)


LOOK_FROM = (0.200, -0.240, 0.300)       # where the right hand goes for the setup picture (x, y, height)


def look(T, right, seconds: float = 3.0):
    """Generator: the right hand moves aside and points at the hasp, so that its wrist camera sees it."""
    pos = np.array([LOOK_FROM[0], LOOK_FROM[1], T.TOP + LOOK_FROM[2]])
    aim = T.door_pos() + T.K.KEEP_AT + (0.0, 0.10, -0.06) - pos
    yield from right.move(pos, seconds, rot=frame(aim, (aim[1], -aim[0], 0.0)), grip=1.0)


HOME: dict = {}                          # each hand's tool point and attitude at the start


def go_home(hand, seconds: float = 2.5):
    """Generator: straight back to where the hand started, then exactly to its starting joint angles."""
    from yam_arena.constants import READY

    pos, rot = HOME[hand.name]
    yield from hand.move(pos, seconds, rot=rot, grip=1.0)
    yield from hand.move_joints(np.array(READY, float), 0.5, grip=1.0)


def script(T, left, right, mode: str = "full", idle_s: float = 45.0):
    """The demo as a generator: one yield per control step."""
    if mode == "still":
        yield from wait(idle_s)
        return
    HOME.update({h.name: (h.pos.copy(), h.rot.copy()) for h in (left, right)})
    if mode == "look":
        yield from look(T, right)
        yield from wait(3.0)
        return
    if mode == "false":
        # control: the keeper is flipped down first; the handle is then swung in and let go short of its seat
        yield from flip_keeper(T, right)
        yield from go_home(right)
    ls = yield from take_handle(T, left)
    if mode in ("cam", "false"):
        # control: the handle is swung in until the cam has its pin, short of the seal, and let go there
        yield from sweep(ls, T, (T.THETA_START, 0.0), (20.0, 0.0), 4.0)
        yield from stay(ls, T, (20.0, 0.0), 0.5)
        yield from let_go(T, left, ls, 20.0)
        yield from go_home(left)
        yield from wait(3.0)
        return
    yield from seat_handle2(T, left, right, ls, lift=mode != "nolift")
    yield from let_go(T, left, ls, REST)
    if mode in ("seat", "nolift"):
        yield from together(go_home(left), go_home(right))
        yield from wait(3.0)
        return
    yield from together(go_home(left), flip_keeper(T, right, stay_by=mode == "held"))
    if mode == "held":                                           # control: the right hand stays at the keeper
        return
    yield from go_home(right)
    yield from wait(3.0)

def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 900, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "full", every: int = 5, cam: tuple = (320, 176), top_cam: str | None = None) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena import station
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam[0],
                               cam_height=cam[1], headless=not window, **({'top_cam': top_cam} if top_cam else {}))
    T = emb.task.module
    obs = emb.reset(Scene(id="door-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.ROD, T.HANDLE, T.KEEPER]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from doorkit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        from doorkit import _np, _rotation

        for h in ("left", "right"):
            robot = emb._env.unwrapped.scene[f"{h}_robot"]
            ROT[h] = _rotation(_np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")][3:7])

    def hands() -> dict:
        from doorkit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1),
                            "droop_deg": round(math.degrees(float(np.linalg.norm(rotvec(h.rot @ ROT[h.name].T)))), 1),
                            "joint_err_deg": [round(math.degrees(float(q[jn.index(f"joint{i + 1}")]) - float(h.q[i])), 1) for i in range(6)]}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        return {n: mm(st[n].pos) for n in names}

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
            trace.append({"step": steps, "score": info.get("score"), "task": T.report(st), "hands": hands(),
                          "places": places(st)})
        if steps % 40 == 0 and trace:
            t = trace[-1]
            print(f"step {steps} score {t['score']} {t['task']} {t['hands']} {t['places']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "seal_n_m": float(T.K.SEAL_MAX),
              "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
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
    (out / "result.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item()))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "cam", "nolift", "seat", "held", "false", "still", "look"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    ap.add_argument("--top-cam", default=None, choices=("calibrated", "demo"), help="the station's overhead camera setting")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")), top_cam=a.top_cam)
