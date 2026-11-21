"""Demo for the pneumatic tube task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1300] [--no-record]
                                   [--mode full|vial|lid|open|empty|loose|held|still|look]

1. The left hand, leaning 30 degrees forward with its fingers side by side, takes the vial by its body, carries it
   over the carrier's mouth (a little to the side away from the open lid) and lets it go: it drops in.
2. The right hand, coming from the right and leaning 15 degrees down, closes round the carrier's body between its
   two rings. The carrier stays standing on the counter.
3. The left hand, fingers closed, swings the lid shut: its fingertip stays on the lid's rim and follows it round the
   pin, pushing first toward the robot's right and then away from the robot, until the catch pulls the lid shut.
4. The right hand lifts the carrier, takes it under the station's send tube, and pushes it straight up until the
   station takes it; then it opens, draws back and goes home. Meanwhile the left hand moves aside and points at the
   send tube's mouth, so that its wrist camera sees the carrier go in.

Controls: ``--mode vial`` stops after the vial is in (0.25); ``--mode lid`` stops with the lid shut and the carrier
on the counter (0.5); ``--mode held`` loads the carrier and keeps hold of it (0.75); ``--mode open`` swings the lid
only to 12 degrees from shut, where its catch does not take it, and loads the carrier like that (trial 0); ``--mode empty``
shuts the lid and loads the carrier without the vial (0); ``--mode loose`` pushes the lid without the other hand on
the carrier; ``--mode still`` keeps both arms still. ``--mode look`` is not a trial: it raises the left arm so that its wrist
camera sees the station and the counter, for the setup picture.

Every target comes from the layout constants in the task file and the sizes in ``tubekit.py``. The only feedback is
each hand's own measured position (the robot's proprioception), used to correct for the arm's sag. The demo never
reads a part's state.
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


def lean(deg: float) -> np.ndarray:
    """A hand pointing away from the robot and ``deg`` degrees down, its fingers side by side (closing across y)."""
    a = math.radians(deg)
    return frame((math.cos(a), 0.0, -math.sin(a)), (0.0, -1.0, 0.0))


R_VIAL = lean(30.0)                      # the left hand, for the vial and for pushing the lid
YAW = math.radians(70.0)                 # the right hand comes at the carrier from the right, turned this far from straight ahead
FROM = np.array([math.cos(YAW), math.sin(YAW), 0.0])     # ... along this level direction
R_BODY = frame((math.cos(math.radians(15.0)) * FROM[0], math.cos(math.radians(15.0)) * FROM[1], -math.sin(math.radians(15.0))),
               (FROM[1], -FROM[0], 0.0))                 # leaning 15 degrees down, fingers side by side
AXIS_AHEAD = 0.012                       # the carrier's axis is this far ahead of the right hand's tool point when it is held
GRIP_UP = 0.075                          # the right hand holds the carrier this far above its closed end
BODY_GAP = 0.076                         # the pads are sent to this gap round the 86 mm body (5 mm of squeeze a side)
VIAL_UP = 0.034                          # the left hand holds the vial this far above its underside
VIAL_GAP = 0.024                         # ... with the pads sent to this gap round its 32 mm body
DROP_OFF = (0.0092, -0.0077)             # the vial is let go this far from the carrier's axis, away from the open lid
DROP_CLEAR = 0.012                       # ... with its underside this far above the mouth's rim
FINGER_HALF = 0.008                      # the closed fingers' side face, out from the tool point
PUSH_UP = 0.232                          # the pushing hand's tool point above the carrier's closed end
PUSH_PAST = 4.0                          # the lid is pushed on to this many degrees past shut (its stop holds it)
TIP_AHEAD = 0.011                        # the fingertips' end, forward of the tool point (level distance, hand 30 degrees down)
UNDER = 0.020                            # the carrier's top is brought this far below the collar's lower edge
PUSH_IN = 0.010                          # the carrier is sent this far past where the station takes it
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
    its fingers end up where they are wanted under load."""

    def __init__(self, hand):
        self.hand, self.bias, self.want = hand, np.zeros(3), None
        self.turn, self.rot = np.zeros(3), None

    def to(self, pos, rot):
        name = self.hand.name
        if self.want is not None:
            self.bias = np.clip(self.bias + GAIN * (self.want - TOOL[name]), -BIAS_MAX, BIAS_MAX)
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


def glide(sag, target, seconds: float, rot):
    """Generator: take the hand's tool point in a straight line to ``target``, correcting for sag on the way."""
    from yam_arena import ik

    start = sag.want if sag.want is not None else sag.hand.pos.copy()
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        sag.to(start + ik.smoothstep(k / n) * (np.asarray(target, float) - start), rot)
        yield


def at(T, xy, up: float) -> np.ndarray:
    return np.array([float(xy[0]), float(xy[1]), T.TOP + up])


def put_vial_in(T, left):
    """The left hand takes the vial and lets it go over the carrier's mouth."""
    K = T.K
    v, c = T.VIAL_AT, T.CARRIER_AT
    above = at(T, v, VIAL_UP + 0.090) + (-0.050, 0.0, 0.0)
    yield from left.move(above, 2.5, rot=R_VIAL, grip=gap(0.070))
    s = Sag(left)
    yield from glide(s, above, 0.3, R_VIAL)
    yield from glide(s, at(T, v, VIAL_UP + 0.012) + (-0.012, 0.0, 0.0), 1.5, R_VIAL)
    yield from glide(s, at(T, v, VIAL_UP), 0.8, R_VIAL)
    yield from left.set_grip(gap(VIAL_GAP), 0.8)
    yield from wait(0.3)
    top = K.MOUTH_Z + DROP_CLEAR + VIAL_UP
    yield from glide(s, at(T, v, top + 0.020), 2.0, R_VIAL)
    over = at(T, (c[0] + DROP_OFF[0], c[1] + DROP_OFF[1]), top)
    yield from glide(s, over + (0.0, 0.0, 0.020), 2.5, R_VIAL)
    yield from glide(s, over, 0.8, R_VIAL)
    yield from wait(0.5)
    yield from left.set_grip(gap(0.060), 0.6)
    yield from wait(0.8)
    yield from glide(s, over + (-0.040, 0.100, 0.040), 1.2, R_VIAL)
    return s


def hold_body(T, right):
    """The right hand closes round the carrier's body where it stands."""
    c = T.CARRIER_AT
    first = at(T, c, GRIP_UP + 0.035) - 0.120 * FROM
    yield from right.move(first, 2.5, rot=R_BODY, grip=1.0)
    s = Sag(right)
    yield from glide(s, first, 0.3, R_BODY)
    yield from glide(s, at(T, c, GRIP_UP) - 0.110 * FROM, 0.8, R_BODY)
    yield from glide(s, at(T, c, GRIP_UP) - AXIS_AHEAD * FROM, 1.8, R_BODY)
    yield from right.set_grip(gap(BODY_GAP), 1.0)
    yield from wait(0.4)
    return s


def lid_corner(T, deg: float) -> np.ndarray:
    """Where the pushing fingertip's leading corner has to be (x, y on the counter) for the lid to stand ``deg`` open:
    on the lid's rim, at the point that faces back along the way the lid swings shut."""
    K = T.K
    a = math.radians(deg)
    pin = T.CARRIER_AT + np.array(K.PIN[:2])
    return pin + (-K.PIN_R * math.sin(a) - K.LID_R * math.cos(a), -K.PIN_R * math.cos(a) + K.LID_R * math.sin(a))


def push_at(T, deg: float, back: float = 0.0) -> np.ndarray:
    """The pushing hand's tool point for that: its fingertip's far right corner on the lid's rim (``back`` m short of
    it, along the way the lid swings)."""
    a = math.radians(deg)
    xy = lid_corner(T, deg) - (TIP_AHEAD, -FINGER_HALF) - back * np.array([math.cos(a), -math.sin(a)])
    return at(T, xy, PUSH_UP)


def shut_lid(T, left, s=None, to: float | None = None):
    """The left hand, fingers closed, swings the lid shut: its fingertip follows the lid's rim round the pin, pushing
    first toward the robot's right and then away from the robot, until the catch takes the lid."""
    from yam_arena import ik

    start = push_at(T, T.LID_START, back=0.020)
    if s is None:
        yield from left.move(start + (-0.030, 0.040, 0.050), 2.5, rot=R_VIAL, grip=0.0)
        s = Sag(left)
    else:
        yield from left.set_grip(0.0, 0.6)
    yield from glide(s, start + (-0.030, 0.040, 0.050), 1.5, R_VIAL)
    yield from glide(s, start, 1.2, R_VIAL)
    yield from glide(s, push_at(T, T.LID_START), 0.6, R_VIAL)
    to = -PUSH_PAST if to is None else to                       # (a control stops short, with the lid not caught)
    n = round(4.0 * CONTROL_HZ)
    for k in range(1, n + 1):                                   # round the pin, to a little past shut
        s.to(push_at(T, T.LID_START + (to - T.LID_START) * ik.smoothstep(k / n)), R_VIAL)
        yield
    end = push_at(T, to)
    for _ in range(round(1.5 * CONTROL_HZ)):                    # keep pushing until the hand is where it was sent:
        s.to(end, R_VIAL)                                       # the lid goes over its catch's bump
        yield
    yield from glide(s, end + (-0.010, 0.050, 0.030), 1.2, R_VIAL)
    return s


def load(T, right, s, let_go: bool = True):
    """The right hand lifts the carrier, takes it under the send tube and pushes it up until the station takes it."""
    K = T.K
    c, m = T.CARRIER_AT, T.MOUTH
    back = AXIS_AHEAD * FROM
    yield from glide(s, at(T, c, GRIP_UP + 0.015) - back, 0.8, R_BODY)
    under = at(T, m[:2], m[2] - UNDER - K.LENGTH + GRIP_UP) - back
    yield from glide(s, np.array([c[0], c[1], under[2]]) - back, 1.2, R_BODY)
    yield from glide(s, under, 2.5, R_BODY)
    yield from wait(0.6)
    up = at(T, m[:2], m[2] + K.TAKE + PUSH_IN - K.LENGTH + GRIP_UP) - back
    yield from glide(s, up, 3.0, R_BODY)
    yield from wait(0.8)
    if not let_go:                                               # control: the carrier is left in the hand
        return
    yield from right.set_grip(1.0, 0.8)
    yield from wait(0.3)
    yield from glide(s, up - 0.110 * FROM + (0.0, 0.0, -0.020), 1.5, R_BODY)


WATCH_FROM = (0.180, 0.140, 0.300)       # where the left hand waits while the carrier is loaded (x, y, height)
WATCH_AT = (0.0, 0.0, 0.0)               # ... pointing at this spot, from the middle of the send tube's lower edge


def watch(T, left, seconds: float = 2.5):
    """Generator: the left hand moves off to the side and points at the send tube's mouth, so that its wrist camera
    sees the carrier go in."""
    pos = np.array([WATCH_FROM[0], WATCH_FROM[1], T.TOP + WATCH_FROM[2]])
    aim = T.station_pos() + WATCH_AT - pos
    yield from left.move(pos, seconds, rot=frame(aim, (aim[1], -aim[0], 0.0)), grip=1.0)


HOME: dict = {}                          # each hand's tool point and attitude at the start


def go_home(hand, seconds: float = 2.5):
    """Generator: straight back to where the hand started, then exactly to its starting joint angles."""
    from yam_arena.constants import READY

    pos, rot = HOME[hand.name]
    yield from hand.move(pos, seconds, rot=rot, grip=1.0)
    yield from hand.move_joints(np.array(READY, float), 0.5, grip=1.0)


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    if mode == "still":
        yield from wait(idle_s)
        return
    HOME.update({h.name: (h.pos.copy(), h.rot.copy()) for h in (left, right)})
    if mode == "look":
        # not a trial: the left arm moves aside so that its wrist camera looks at the station's send tube and the carrier
        yield from watch(T, left, 3.0)
        yield from wait(3.0)
        return
    ls = None
    if mode != "empty":
        ls = yield from put_vial_in(T, left)
    if mode == "vial":
        yield from go_home(left)
        yield from wait(2.0)
        return
    if mode == "loose":
        # control: the lid is pushed without the other hand on the carrier
        ls = yield from shut_lid(T, left, ls)
        yield from go_home(left)
        yield from wait(2.0)
        return
    rs = yield from hold_body(T, right)
    ls = yield from shut_lid(T, left, ls, to=12.0 if mode == "open" else None)
    if mode == "lid":
        yield from right.set_grip(1.0, 0.8)
        yield from glide(rs, rs.want - 0.110 * FROM + (0.0, 0.0, 0.020), 1.2, R_BODY)
        yield from together(go_home(left), go_home(right))
        yield from wait(2.0)
        return
    yield from together(load(T, right, rs, let_go=mode != "held"), watch(T, left))
    if mode == "held":
        yield from wait(3.0)
        return
    yield from together(go_home(right), go_home(left))
    yield from wait(3.0)

def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1300, record: bool = True, window: bool = False,
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
    obs = emb.reset(Scene(id="tube-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.CARRIER, T.LID, T.VIAL]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from tubekit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        from tubekit import _np, _rotation

        for h in ("left", "right"):
            robot = emb._env.unwrapped.scene[f"{h}_robot"]
            ROT[h] = _rotation(_np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")][3:7])

    def hands() -> dict:
        from tubekit import _np

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
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
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
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "vial", "lid", "open", "empty", "loose", "held", "still", "look"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    ap.add_argument("--top-cam", default=None, choices=("calibrated", "demo"), help="the station's overhead camera setting")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")), top_cam=a.top_cam)
