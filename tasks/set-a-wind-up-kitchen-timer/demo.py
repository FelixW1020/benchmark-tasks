"""Demo for the wind-up kitchen timer task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1800] [--no-record] [--mode full|direct|onehand|still]

1. The right hand comes down over the timer, takes its upper part between its fingers and lifts it. It swings the
   timer onto its side, 14 cm above the table, the timer's underside toward the left hand.
2. The left hand comes in along the timer's axis and takes the lower part. The two hands now face each other with
   one part each.
3. Both wrists roll, in opposite senses, until the lower part has gone 345 degrees clockwise (seen from above the
   timer) under the upper: 57.5 minutes, past the 55-minute mark. Then both roll back until it reads 10.
4. The left hand lets go and draws back. The right hand stands the timer on the table again and lets go.

Controls: ``--mode direct`` does the same but turns the timer straight to 10 minutes. ``--mode onehand`` leaves the
timer standing and turns its upper part with the right hand alone (two bites of 172.5 degrees each way). ``--mode still``
keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``timerkit.py``. The only feedback
is the hands' own measured positions (the robot's proprioception), used to correct for sag. The demo never reads a
part's state: how far the timer has been turned is known only from how far the wrists have rolled.
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
ROLLED: dict = {}                        # how far each wrist has actually rolled (the robot's own reading, rad)

DOWN = (0.0, 0.0, -1.0)
T_IN = 0.013                             # the right hand's tool point, below the timer's upper face
B_IN = 0.010                             # the left hand's, in from the timer's underside
OPEN_GAP = 0.085                         # pads this far apart on the way in, astride the 6 cm timer
HOLD_GAP = 0.0565                        # ... and commanded this far apart on it: 1.75 mm of squeeze a side
LIFT = 0.070                             # straight up off the table before the timer is swung onto its side
HOLD_AT = (0.280, 0.000, 0.140)          # the middle of the timer while it is set, above the table
LEAD_IN = 0.070                          # the left hand starts this far back along the timer's axis
WIND_TO = 57.5                           # minutes: past the 55-minute mark, short of the stop
SET_TO = 10.3                            # minutes: a little over 10, as the timer runs while it is put down
HALF = 0.5                               # the share of each turn that the right wrist does
TURN_RATE = 60.0                         # degrees a second of the upper part round the lower
PUT_UP = 0.0010                          # the timer's underside is brought this close to the table before it is let go
GAIN, BIAS_MAX = 0.25, 0.020             # how fast, and how far, a hand corrects for its own sag


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
    for q1 in (0.8, 1.3, -0.8, -1.3):
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


def level(approach, roll: float) -> np.ndarray:
    """A level hand's rotation: pointing along ``approach`` (horizontal), its closing axis ``roll`` rad round from
    straight up."""
    a = np.asarray(approach, float)
    up = np.array([0.0, 0.0, 1.0])
    return frame(a, math.cos(roll) * up + math.sin(roll) * np.cross(a, up))


def down(yaw: float) -> np.ndarray:
    """A top-down hand's rotation, its closing axis ``yaw`` rad round from x."""
    return frame(DOWN, (math.cos(yaw), math.sin(yaw), 0.0))


def centred(hand, pos, rot_of, room: float, guess: float = 0.0):
    """A hold whose wrist can roll ``room`` rad onward (the way ``rot_of``'s angle grows) with as much to spare at
    each end of the last joint's range as possible. ``rot_of(angle)`` gives the hand's rotation for an angle about
    its approach axis. Returns (joint angles, angle, the last joint's sense)."""
    from yam_arena import ik

    q0 = reach(hand, pos, rot_of(guess))
    q1 = reach(hand, pos, rot_of(guess + 0.2), near=q0) if q0 is not None else None
    if q0 is None or q1 is None:
        return None
    sense = 1.0 if q1[5] > q0[5] else -1.0               # which way the last joint goes as the angle grows
    mid = 0.5 * (ik.LOWER[5] + ik.UPPER[5])
    angle = guess + sense * ((mid - sense * room / 2) - q0[5])
    q = reach(hand, pos, rot_of(angle), near=q0)
    return None if q is None else (q, angle, sense)


def plan(T, wind_to: float = WIND_TO) -> dict:
    """The demo's fixed targets, from the task file's layout (world, m)."""
    K = T.K
    (sx, sy), _deg = T.LAYOUT
    axis = np.array([0.0, 1.0, 0.0])                                 # the way the timer's underside faces while it is set
    middle = np.array([HOLD_AT[0], HOLD_AT[1], T.TOP + HOLD_AT[2]])
    wind = math.radians(K.DEG_PER_MIN * wind_to)
    back = math.radians(K.DEG_PER_MIN * (wind_to - SET_TO)) if wind_to > SET_TO else 0.0
    return {"stand": np.array([sx, sy, T.TOP + K.HEIGHT - T_IN]), "axis": axis,
            "right_at": middle - (K.HEIGHT / 2 - T_IN) * axis, "left_at": middle + (K.HEIGHT / 2 - B_IN) * axis,
            "wind": wind, "back": back}


def roll_both(left, right, d_left: float, d_right: float, seconds: float):
    """Generator: roll both wrists (their last joints only), by ``d_left`` and ``d_right`` rad."""
    ql, qr = left.q.copy(), right.q.copy()
    ql[5] += d_left
    qr[5] += d_right
    yield from together(left.move_joints(ql, seconds), right.move_joints(qr, seconds))


def script(T, left, right, mode: str = "full", idle_s: float = 90.0, bites: int = 2):
    """The demo as a generator: one yield per control step."""
    K = T.K
    if mode == "still":
        yield from wait(idle_s)
        return
    p = plan(T, WIND_TO if mode != "direct" else SET_TO)
    up = np.array([0.0, 0.0, 1.0])
    turn = p["wind"] if mode != "direct" else math.radians(K.DEG_PER_MIN * SET_TO)

    if mode == "onehand":                # control: the timer stays where it stands; the right hand turns its upper part
        bite, bite_back = p["wind"] / 2, p["back"] / 2
        got = centred(right, p["stand"], down, bite)                 # the angle grows anticlockwise, seen from above,
        q_at, angle, sense = got                                     # which is the way the upper part goes to wind
        rot = down(angle)
        yield from right.move_joints(reach(right, p["stand"] + LIFT * up, rot, near=q_at), 2.5, grip=gap(OPEN_GAP))
        sag = Sag(right)
        yield from glide(sag, p["stand"] + LIFT * up, 0.4, rot)
        yield from glide(sag, p["stand"], 1.5, rot)
        yield from glide(sag, p["stand"], 0.4, rot)

        def roll(by: float, seconds: float):
            q = right.q.copy()
            q[5] += sense * by
            yield from right.move_joints(q, seconds)

        for _ in range(bites):           # wind: two bites
            yield from right.set_grip(gap(HOLD_GAP), 0.5)
            yield from roll(bite, math.degrees(bite) / TURN_RATE * 2)
            yield from wait(0.2)
            yield from right.set_grip(gap(OPEN_GAP), 0.5)
            yield from roll(-bite, 2.0)
        for _ in range(bites):           # and back: two bites
            yield from roll(bite_back, 2.0)
            yield from right.set_grip(gap(HOLD_GAP), 0.5)
            yield from roll(-bite_back, math.degrees(bite_back) / TURN_RATE * 2)
            yield from wait(0.2)
            yield from right.set_grip(gap(OPEN_GAP), 0.5)
        sag = Sag(right)
        yield from glide(sag, right.pos + 0.100 * up, 1.2, right.rot)
        yield from wait(3.0)
        return

    # 1. Over the timer, down astride its upper part, take it, lift it, and swing it onto its side.
    # To wind, each wrist rolls anticlockwise about its own approach axis: the angles below grow that way.
    right_hold = centred(right, p["right_at"], lambda a: level(p["axis"], -a), HALF * max(p["wind"], turn), guess=0.0)
    q_level, r_roll, r_sense = right_hold
    r_rot = level(p["axis"], -r_roll)
    best = None
    for deg in (-90, -60, -30, 0, 30, 60, -120, -150):                # the pick-up whose swing to that hold is the shortest
        rot = down(math.radians(deg))
        q_dn = reach(right, p["stand"], rot)
        q_up = reach(right, p["stand"] + LIFT * up, rot, near=q_dn) if q_dn is not None else None
        if q_up is None:
            continue
        cost = float(np.abs(q_level - q_up).max())
        if best is None or cost < best[0]:
            best = (cost, rot, q_up)
    _, take, q_up = best
    yield from right.move_joints(q_up, 2.5, grip=gap(OPEN_GAP))
    sag = Sag(right)
    yield from glide(sag, p["stand"] + LIFT * up, 0.4, take)
    yield from glide(sag, p["stand"], 1.5, take)
    yield from glide(sag, p["stand"], 0.4, take)
    yield from right.set_grip(gap(HOLD_GAP), 0.6)
    yield from wait(0.3)
    yield from glide(sag, p["stand"] + LIFT * up, 1.2, take)
    yield from right.move_joints(q_level, 4.0)
    sag = Sag(right)
    yield from glide(sag, p["right_at"], 0.5, r_rot)
    yield from glide(sag, p["right_at"], 0.8, r_rot)

    # 2. The left hand comes in along the timer's axis and takes the lower part.
    left_hold = centred(left, p["left_at"], lambda a: level(-p["axis"], -a), (1 - HALF) * max(p["wind"], turn), guess=0.0)
    q_left, l_roll, l_sense = left_hold
    l_rot = level(-p["axis"], -l_roll)
    before = p["left_at"] + LEAD_IN * p["axis"]
    yield from left.move_joints(reach(left, before, l_rot, near=q_left), 3.0, grip=gap(OPEN_GAP))
    lsag = Sag(left)
    yield from glide(lsag, before, 0.4, l_rot)
    yield from glide(lsag, p["left_at"], 1.6, l_rot)
    yield from glide(lsag, p["left_at"], 0.4, l_rot)
    yield from left.set_grip(gap(HOLD_GAP), 0.6)
    yield from wait(0.3)

    # 3. Wind: both wrists roll anticlockwise about their own approach axes, which turns the lower part clockwise
    #    (seen from above the timer) under the upper. Then back to the time wanted.
    seconds = math.degrees(turn) / TURN_RATE
    yield from roll_both(left, right, l_sense * (1 - HALF) * turn, r_sense * HALF * turn, seconds)
    yield from wait(0.4)
    if mode != "direct":
        seconds = math.degrees(p["back"]) / TURN_RATE
        yield from roll_both(left, right, -l_sense * (1 - HALF) * p["back"], -r_sense * HALF * p["back"], seconds)
        yield from wait(0.4)

    # 4. The left hand lets go and draws back; the right stands the timer on the table again and lets go.
    yield from left.set_grip(gap(OPEN_GAP), 0.6)
    yield from wait(0.2)
    lsag = Sag(left)
    yield from glide(lsag, left.pos + LEAD_IN * p["axis"], 1.2, left.rot)
    from yam_arena.constants import READY

    put = p["stand"] + np.array([0.0, 0.0, PUT_UP])
    yield from together(left.move_joints(np.array(READY, float), 3.0), right.move_joints(q_up, 4.0))
    sag = Sag(right)
    yield from glide(sag, p["stand"] + LIFT * up, 0.4, take)
    yield from glide(sag, put, 1.6, take)
    yield from glide(sag, put, 0.5, take)
    yield from right.set_grip(gap(OPEN_GAP), 0.6)
    yield from wait(0.3)
    yield from glide(sag, put + 0.090 * up, 1.2, take)
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "full", every: int = 5, cam: tuple = (320, 176), bites: int = 2) -> dict:
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
    obs = emb.reset(Scene(id="timer-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ, bites=bites)
    frames: dict[str, list] = {}
    trace = []
    names = [T.TIMER, T.DIAL]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from timerkit import finger_points, wrist_rolls

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        ROLLED["left"], ROLLED["right"] = wrist_rolls(emb._env)

    def hands() -> dict:
        from timerkit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1),
                            "roll_deg": round(math.degrees(ROLLED[h.name]), 1), "roll_cmd_deg": round(math.degrees(float(h.q[5])), 1)}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        return {n: mm(st[n].pos) for n in st}

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
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "direct", "onehand", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    ap.add_argument("--bites", type=int, default=2, help="onehand only: bites each way (3 runs the dial into its stops)")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to go on after the script ends (to watch the timer run)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")), bites=a.bites, hold_s=a.hold)
