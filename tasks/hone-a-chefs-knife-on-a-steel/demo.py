"""Demo for the knife-honing task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 2800] [--no-record] [--mode full|...]

1. The right hand comes in line with the lying knife's handle, from the right, one finger over it and one under,
   closes hard on its two flat sides 4.5 cm from its end, lifts the knife and parks it, blade forward, to the right of
   where the steel will stand.
2. The left hand takes the lying steel's handle from the robot's side, behind the guard, one finger over and one under,
   lifts it, rolls its wrist so that the steel hangs tip down, stands the tip on the pad and holds it there.
3. Eight strokes, the first on the steel's right. For each, the right hand rolls its wrist so that the blade's face
   is 15 degrees off the steel, brings the edge against the steel near the heel, 20 cm up the rod, aiming it 4 mm
   into the rod's place, and draws the knife down and back until the contact has run to the tip, 7 cm up the rod. It
   then takes the knife off, draws it back clear, crosses to the steel's other side and goes back up for the next
   stroke on the other face.
4. The right hand parks; the left hand lays the steel back and lets go; the right hand lays the knife back and lets go.

Controls (``--mode``): ``still``; ``probe`` (one stroke, then hold); ``two`` (two good strokes, then everything laid
down); ``inhand`` (eight good strokes, then both tools kept in hand). These stop with the tools in hand:
``oneface`` (four strokes on one face); ``steep`` (two strokes at 40 degrees); ``middle`` (two strokes over the
middle of the edge only); ``heavy`` (two strokes with the edge aimed 3 cm into the steel's place); ``guard`` (the
blade is taken up past the steel's guard).

Every target comes from the layout constants in the task file and the sizes in ``honekit.py``. The only feedback is
each hand's own measured place and tilt (the robot's proprioception), used to correct for the arm's sag; a hand that
is leaning on something stops correcting in that direction. The demo never reads a part's state.
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
MODES = ("full", "still", "probe", "pair", "oneface", "steep", "middle", "heavy", "inhand", "guard", "two", "hover", "reverse")


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


def pose(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


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
        from yam_arena import ik

        g0 = self.grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.grip = g0 + ik.smoothstep(k / n) * (grip - g0)
            self.cmd = np.r_[self.q, self.grip]
            yield

    def move_joints(self, q_goal, duration: float, grip: float | None = None):
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


def best_ik(hand, pos, rot, seeds=()):
    """Joint angles for ``pos``/``rot`` from the best of a few seeds."""
    from yam_arena import ik

    best = None
    for seed in [hand.q.copy()] + [np.array(s, float) for s in seeds] + [np.r_[hand.q[:5], v] for v in (-1.0, 1.0)]:
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
        cost = pe + 0.05 * re - 0.002 * min(room, 0.3)
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0]


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)

PHI = math.radians(25.0)                 # the knife points this far left of straight ahead while it is honed
ALPHA = math.radians(15.0)               # the blade's face is held this far off the steel
GRIP_BACK = 0.095                        # the right hand takes the knife's handle this far behind the heel (4.5 cm from its end)
KNIFE_GAP = 0.0065                       # pad gap commanded on the 20 mm handle: about 34 N on each pad (the real gripper gives about 40)
STEEL_GAP = 0.016                        # pad gap commanded on the steel's 26 mm handle
S0, S1 = 0.018, 0.186                    # a stroke's contact runs between these places on the edge (m from the heel)
Z0, Z1 = 0.200, 0.070                    # ... and between these heights up the rod from its tip
PRESS = 0.0040                           # the edge is aimed this far inside the rod's surface at the heel ...
PRESS_TIP = 0.0080                       # ... and this far at the tip, where the edge curves away and the contact is lightest
OFF = 0.025                              # off the steel: the edge this far outside the rod's surface
BACK = 0.050                             # drawn back this far past the stroke's end before crossing sides
STROKE_S = 2.6                           # seconds for one stroke
HOLD_FROM = math.radians(20.0)           # the hand on the upright steel points this far forward of straight right
PLANT = 0.0020                           # the steel's tip is aimed this far below the pad's top
TURN_GAIN, TURN_MAX = 0.15, math.radians(10.0)   # the same for its tilt
DROOP = math.radians(0.0)                # allowance for the knife settling tip-down in the pinch (none: the hand cannot pitch up that far)
GAIN, BIAS_MAX = 0.25, 0.040             # how fast, and how far, a hand corrects for its own sag
# joint angles near each hand's working poses, as starting points for the solver
SEED_L_PICK = (1.07, 2.39, 1.02, 1.37, 1.07, -1.57)
SEED_L_HOLD = (0.23, 1.49, 1.41, 0.08, 1.46, 0.0)
SEED_L_LIFT = (1.07, 1.78, 1.32, 0.46, 1.07, -1.57)
SEED_R_PICK = (0.46, 2.61, 1.54, 1.07, -1.11, -1.57)


def _rotvec(rot) -> np.ndarray:
    """The rotation vector of a rotation matrix (small angles)."""
    c = max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0))
    a = math.acos(c)
    v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-9 else v / n * a


def _rot(vec) -> np.ndarray:
    a = float(np.linalg.norm(vec))
    if a < 1e-9:
        return np.eye(3)
    k = np.asarray(vec, float) / a
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * (kx @ kx)


class Sag:
    """Slowly offsets the hand's command by its measured error, in place and in tilt, so that its tool point ends up
    where it is wanted and its hand as it is wanted (a loaded wrist gives a few degrees)."""

    def __init__(self, hand):
        self.hand, self.bias, self.want = hand, np.zeros(3), None
        self.turn, self.want_rot = np.zeros(3), None

    def _learn(self, free=None):
        if self.want is None:
            return
        err = self.want - TOOL[self.hand.name]
        if free is not None:
            err = err - (err @ free) * np.asarray(free, float)
        self.bias = np.clip(self.bias + GAIN * err, -BIAS_MAX, BIAS_MAX)
        if self.want_rot is not None and free is None:
            off = _rotvec(self.want_rot @ TOOL[self.hand.name + "_rot"].T)
            self.turn = np.clip(self.turn + TURN_GAIN * off, -TURN_MAX, TURN_MAX)

    def _send(self, pos, rot):
        self.want, self.want_rot = np.asarray(pos, float), np.asarray(rot, float)
        self.hand._solve(self.want + self.bias, _rot(self.turn) @ self.want_rot)
        self.hand.pos, self.hand.rot = self.want, self.want_rot      # what it is meant to be doing

    def to(self, pos, rot, free=None):
        """Send the hand to ``pos``. ``free``: a direction in which the hand is pressing on something, along which
        its error is left alone (correcting it would push harder and harder)."""
        self._learn(free)
        self._send(pos, rot)

    def hold(self, pos, rot):
        """Keep the hand at ``pos`` with the correction it already has (it is leaning on something)."""
        self._send(pos, rot)

    def sync(self):
        """After a joint-space move: start again from where the hand was sent, with no correction."""
        self.bias, self.want = np.zeros(3), self.hand.pos.copy()
        self.turn, self.want_rot = np.zeros(3), None


def glide(sag, target, seconds: float, rot=None):
    """Generator: take the hand's tool point in a straight line to ``target`` (turning to ``rot`` on the way),
    correcting for sag."""
    from yam_arena import ik

    start = sag.want if sag.want is not None else sag.hand.pos.copy()
    r0 = sag.hand.rot.copy()
    r1 = r0 if rot is None else rot
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        sag.to(start + s * (np.asarray(target, float) - start), ik.slerp_matrix(r0, r1, s))
        yield


def plan(T) -> dict:
    """The demo's fixed poses, from the task's layout: tool points and rotations (world)."""
    K = T.K
    up = np.array([0.0, 0.0, 1.0])
    # the right hand on the lying knife: in line with it, at the end of its handle, one finger over it and one under
    kp, kq = T.knife_start()
    rk = K._rotation(kq)
    r_rot = frame(rk[:, 0], rk[:, 2])
    r_pick = kp + rk @ np.array([-GRIP_BACK, K.BH - K.HANDLE_H / 2 - 0.001, 0.0])
    knife_in_right = np.linalg.inv(pose(r_pick, r_rot)) @ pose(kp, rk)
    c, s = math.cos(DROOP), math.sin(DROOP)                      # ... as it sits once it has settled in the pinch
    knife_held = pose(np.zeros(3), np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])) @ knife_in_right
    # the left hand on the lying steel: from the robot's side of its handle, one finger over it and one under
    sp, sq = T.steel_start()
    rs = K._rotation(sq)
    l_rot = frame(-rs[:, 1], rs[:, 2])
    l_pick = sp + rs @ np.array([T.HOLD_AT, 0.0, 0.0])
    steel_in_left = np.linalg.inv(pose(l_pick, l_rot)) @ pose(sp, rs)
    # the steel upright on the pad, the left hand holding it from the left
    ap = np.array([math.sin(HOLD_FROM), -math.cos(HOLD_FROM), 0.0])
    hold_rot = frame(ap, -np.array([math.cos(HOLD_FROM), math.sin(HOLD_FROM), 0.0]))
    rs_up = hold_rot @ steel_in_left[:3, :3]
    tip = np.array([T.PAD_XY[0], T.PAD_XY[1], T.TOP + T.PAD_TOP - PLANT])
    l_hold = pose(tip, rs_up) @ np.linalg.inv(steel_in_left)
    d = np.array([math.cos(PHI), math.sin(PHI), 0.0])
    side = np.cross(up, d)
    axis0 = np.array([T.PAD_XY[0], T.PAD_XY[1], T.TOP + T.PAD_TOP])

    def knife_at(face: int, s: float, z: float, out: float, alpha: float = ALPHA) -> np.ndarray:
        """The knife's pose with the point of its edge ``s`` from the heel ``out`` outside the rod's surface (inside
        it if negative), ``z`` up the rod, on the steel's left (face +1) or right (-1), its face ``alpha`` off it."""
        r = face * side
        e = -math.sin(alpha) * r - math.cos(alpha) * up                 # from the spine to the edge
        rot = np.column_stack([d, -e, np.cross(d, -e)])
        loc = np.array([s, float(K.edge_y(min(s, K.BL))), 0.0])
        c = axis0 + z * up + (K.ROD_R + out) * r
        return pose(c - rot @ loc, rot)

    def right_for(face: int, s: float, z: float, out: float, alpha: float = ALPHA) -> tuple:
        g = knife_at(face, s, z, out, alpha) @ np.linalg.inv(knife_held)
        return g[:3, 3], g[:3, :3]

    park_p, park_r = right_for(-1, S0 + 0.05, Z0, 0.15)                 # parked: to the right of where the steel stands
    return {"r_rot": r_rot, "r_pick": r_pick, "l_rot": l_rot, "l_pick": l_pick, "l_hold": l_hold, "ap": ap,
            "knife_in_right": knife_in_right, "knife_held": knife_held, "steel_in_left": steel_in_left, "right_for": right_for, "d": d,
            "park": (park_p, park_r), "side": side}


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    from yam_arena.constants import READY

    if mode == "still":
        yield from wait(idle_s)
        return
    P = plan(T)
    ls, rs = Sag(left), Sag(right)
    up = np.array([0.0, 0.0, 1.0])
    rf = P["right_for"]

    # 1. The right hand takes the knife by its handle, in line with it, and parks it.
    r_back = P["r_pick"] - 0.075 * P["r_rot"][:, 2]
    yield from right.move_joints(best_ik(right, r_back + 0.05 * up, P["r_rot"], [SEED_R_PICK]), 3.0, grip=gap(0.050))
    yield from glide(rs, r_back + 0.05 * up, 0.5, P["r_rot"])
    yield from glide(rs, r_back, 1.2)
    yield from glide(rs, P["r_pick"], 1.5)
    yield from glide(rs, P["r_pick"], 0.4)
    yield from right.set_grip(gap(KNIFE_GAP), 0.8)
    yield from wait(0.3)
    yield from glide(rs, P["r_pick"] + 0.10 * up, 2.0)
    yield from glide(rs, P["park"][0], 5.5, P["park"][1])
    yield from glide(rs, P["park"][0], 0.4)

    # 2. The left hand takes the steel by its handle, lifts it, hangs it tip down, and stands it on the pad.
    l_back = P["l_pick"] - 0.055 * P["l_rot"][:, 2]
    l_over = l_back + 0.13 * up                                  # above and behind the handle: the way in and out
    yield from left.move_joints(best_ik(left, l_over, P["l_rot"], [SEED_L_LIFT]), 3.5, grip=gap(0.060))
    yield from glide(ls, l_over, 0.5, P["l_rot"])
    yield from glide(ls, l_back, 1.8)
    yield from glide(ls, P["l_pick"], 1.5)
    yield from glide(ls, P["l_pick"], 0.4)
    yield from left.set_grip(gap(STEEL_GAP), 0.8)
    yield from wait(0.3)
    hold_p, hold_r = P["l_hold"][:3, 3], P["l_hold"][:3, :3]
    lift = np.array([P["l_pick"][0], P["l_pick"][1], hold_p[2] - 0.06])
    yield from glide(ls, lift, 3.0)
    q_over = best_ik(left, hold_p + 0.04 * up, hold_r, [SEED_L_HOLD])
    yield from left.move_joints(q_over, 4.5)                     # the wrist rolls a quarter turn on the way: tip down
    ls.sync()
    yield from glide(ls, hold_p + 0.04 * up, 0.6, hold_r)
    n = round(1.5 * CONTROL_HZ)
    for k in range(1, n + 1):                                    # down onto the pad, with the correction it has
        ls.hold(hold_p + 0.04 * (1 - k / n) * up, hold_r)
        yield
    for _ in range(round(0.5 * CONTROL_HZ)):
        ls.hold(hold_p, hold_r)
        yield

    # 3. The strokes.
    n_strokes = {"probe": 1, "pair": 2, "two": 2, "oneface": 4, "steep": 2, "middle": 2, "heavy": 2, "hover": 2, "reverse": 2}.get(mode, 2 * T.EACH)
    alpha = math.radians(40.0) if mode == "steep" else ALPHA
    s0, s1 = (0.060, 0.140) if mode == "middle" else (S0, S1)
    z0, z1 = Z0, Z1
    if mode == "reverse":                                        # tip to heel, and up the steel
        s0, s1, z0, z1 = S1, S0, Z1, Z0
    press = 0.030 if mode == "heavy" else (-0.0030 if mode == "hover" else PRESS)   # hover: the edge 3 mm clear of the rod
    press_tip = PRESS_TIP if press == PRESS else press
    back = 0.090 if mode == "middle" else BACK                    # drawn back until the tip is clear of the steel before crossing sides

    def hold_left():
        ls.hold(hold_p, hold_r)

    def go(target, seconds, rot, free=None):
        """The right hand glides while the left holds the steel in place."""
        from yam_arena import ik

        start, r0 = rs.want.copy(), right.rot.copy()
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            rs.to(start + s * (np.asarray(target, float) - start), ik.slerp_matrix(r0, rot, s), free)
            hold_left()
            yield

    if mode == "guard":
        p, r = rf(-1, S0, Z0, OFF)
        yield from go(p, 3.0, r)
        p, r = rf(-1, 0.10, T.K.ROD_L + 0.03, 0.02)
        yield from go(p, 3.0, r)
        yield from go(p, 1.0, r)
        p, r = rf(-1, S0, Z0, OFF)
        yield from go(p, 2.0, r)
        n_strokes = 0
    face = -1
    for i in range(n_strokes):
        p, r = rf(face, s0, z0, OFF, alpha)
        yield from go(p, 2.5 if i == 0 else 0.3, r)
        across = face * P["side"]                                # the way the edge leans on the steel
        yield from go(p, 0.4, r)                                 # settle, off the steel
        p, r = rf(face, s0, z0, -press, alpha)
        yield from go(p, 0.6, r, across)
        n = max(1, round(STROKE_S * CONTROL_HZ))
        for k in range(1, n + 1):
            f = k / n
            p, r = rf(face, s0 + f * (s1 - s0), z0 + f * (z1 - z0), -(press + f * (press_tip - press)), alpha)
            rs.to(p, r, across)
            hold_left()
            yield
        p, r = rf(face, s1, z1, OFF, alpha)
        yield from go(p, 0.5, r, across)
        s_out = max(s0, s1) + back                               # drawn back until the tip is clear of the steel
        p, r = rf(face, s_out, min(z0, z1), OFF, alpha)
        yield from go(p, 0.6 if back == BACK else 1.5, r)
        if mode == "probe" or i == n_strokes - 1:
            break
        if mode != "oneface":
            face = -face
            p, r = rf(face, s_out, min(z0, z1), OFF, alpha)
            yield from go(p, 1.2, r)
        p, r = rf(face, s0, z0, OFF, alpha)                      # back up for the next stroke
        yield from go(p, 2.0, r)
    if mode == "probe":
        for _ in range(int(3.0 * CONTROL_HZ)):
            hold_left()
            yield
        return
    if face == 1 and n_strokes:                                  # back to the right of the steel before parking
        p, r = rf(-1, max(s0, s1) + back, min(z0, z1), OFF, alpha)
        yield from go(p, 1.2, r)
    yield from go(P["park"][0], 2.5, P["park"][1])
    if mode in ("inhand", "pair", "oneface", "steep", "middle", "heavy", "guard", "hover", "reverse"):
        for _ in range(int(3.0 * CONTROL_HZ)):
            rs.to(*P["park"])
            hold_left()
            yield
        return

    # 4. The steel goes back, then the knife.
    yield from glide(ls, hold_p + 0.04 * up, 1.2, hold_r)
    yield from left.move_joints(best_ik(left, lift, P["l_rot"], [SEED_L_LIFT]), 4.5)
    ls.sync()
    yield from glide(ls, lift, 0.6, P["l_rot"])
    yield from glide(ls, P["l_pick"] + 0.012 * up, 3.0)
    yield from glide(ls, P["l_pick"] + 0.001 * up, 1.0)
    yield from glide(ls, P["l_pick"] + 0.001 * up, 0.4)
    yield from left.set_grip(gap(0.060), 0.8)
    yield from wait(0.3)
    yield from glide(ls, l_back, 1.5)
    yield from glide(ls, l_over, 1.5)
    yield from left.move_joints(READY, 3.5)

    yield from glide(rs, P["r_pick"] + 0.10 * up, 3.5, P["r_rot"])
    yield from glide(rs, P["r_pick"] + 0.012 * up, 1.5)
    yield from glide(rs, P["r_pick"] + 0.001 * up, 1.0)
    yield from glide(rs, P["r_pick"] + 0.001 * up, 0.4)
    yield from right.set_grip(gap(0.050), 0.8)
    yield from wait(0.3)
    yield from glide(rs, r_back, 1.5)
    yield from glide(rs, r_back + 0.05 * up, 0.8)
    yield from right.move_joints(READY, 3.0)
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 2800, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "full", every: int = 2, cam: tuple = (320, 176)) -> dict:
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
    K = T.K
    obs = emb.reset(Scene(id="hone-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.KNIFE, T.STEEL, T.BOARD]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        (pl, rl), (pr, rr) = K.finger_points(emb._env)
        TOOL["left"], TOOL["right"] = np.asarray(pl, float), np.asarray(pr, float)
        TOOL["left_rot"], TOOL["right_rot"] = rl, rr

    def hands() -> dict:
        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = K._np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1)}
        forces = K.grip_forces(emb._env)
        for name in out_:
            out_[name]["grip_n"] = forces.get(name)
        return out_

    def places(st) -> dict:
        """For the record only (the demo does not use it): where the knife and the steel are, and how the knife sits
        in the right hand and the steel in the left."""
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float)]   # noqa: E731
        k, s = st[T.KNIFE], st[T.STEEL]
        rk, rst = K._rotation(k.quat_xyzw), K._rotation(s.quat_xyzw)
        c = T.touch((k.pos, k.quat_xyzw), (s.pos, s.quat_xyzw))
        rel_k = TOOL["right_rot"].T @ rk
        rel_s = TOOL["left_rot"].T @ rst
        return {"knife": mm(np.asarray(k.pos) - [0.0, 0.0, T.TOP]), "steel_tip": mm(np.asarray(s.pos) - [0.0, 0.0, T.TOP]),
                "steel_lean_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(rst[2, 0]))))), 1),
                "gap_mm": None if not math.isfinite(c["gap"]) else round(1000 * c["gap"], 2),
                "s_mm": None if c["s"] is None else round(1000 * c["s"]), "t_mm": None if c["t"] is None else round(1000 * c["t"]),
                "angle": None if c["angle"] is None else round(c["angle"], 1), "face": c["face"],
                "knife_in_hand": [round(float(v), 3) for v in (rel_k[0, 0], rel_k[1, 0], rel_k[2, 0], rel_k[0, 2])],
                "steel_in_hand": [round(float(v), 3) for v in (rel_s[0, 0], rel_s[1, 0], rel_s[2, 0])],
                "steel_roll": [round(float(rst[2, 1]), 3), round(float(rst[2, 2]), 3)],
                "steel_vel": [round(float(v), 3) for v in list(s.lin_vel) + list(s.ang_vel)],
                "knife_roll": [round(float(rk[2, 1]), 3), round(float(rk[2, 2]), 3), round(float(rk[2, 0]), 3)],
                "board_mm": round(1000 * (float(st[T.BOARD].pos[2]) - float(T.board_start()[2])), 3),
                "board_tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(K._rotation(st[T.BOARD].quat_xyzw)[2, 2]))))), 2)}

    snap_frames(obs)
    measure()
    start = {"task": T.report(states()), "places": places(states())}
    steps, ended, success, info = 0, None, False, {}
    grip = {h: {"peak_n": 0.0, "peak_in_strokes_n": 0.0} for h in ("left", "right")}   # per finger, as the simulator applies it
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
        forces, stroking = K.grip_forces(emb._env), T._STATE["now"] is not None
        for h, ff in forces.items():
            if ff:
                grip[h]["peak_n"] = max(grip[h]["peak_n"], *ff)
                if stroking:
                    grip[h]["peak_in_strokes_n"] = max(grip[h]["peak_in_strokes_n"], *ff)
        if steps % every == 0:
            st = states()
            rep = T.report(st)
            rep.pop("strokes", None)
            trace.append({"step": steps, "score": info.get("score"), "task": rep, "hands": hands(), "places": places(st)})
        if steps % 100 == 0 and trace:
            t = trace[-1]
            print(f"step {steps} score {t['score']} {t['task']} {t['places']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended, "grip_per_finger": grip,
              "contact_view": {"made": K._CONTACT["view"] is not None, "physics_steps_read": K._CONTACT["calls"],
                               "physics_dt": K._CONTACT["dt"], "error": K._CONTACT["error"]},
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": {"task": T.report(states()), "places": places(states())}, "trace": trace}
    if record:
        import imageio

        for name, fs in frames.items():
            _write_mp4(fs, out / f"demo_{name}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
        for name, fs in frames.items():
            imageio.imwrite(out / f"setup_{name}.png", fs[0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item()))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2800)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=MODES)
    ap.add_argument("--trace-every", type=int, default=2, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
