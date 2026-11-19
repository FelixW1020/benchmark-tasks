"""Demo for the rebar-cap task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1500] [--no-record] [--mode full|perch|mushroom|still]

For each of the two bare bars, the right hand
1. comes down over a rated cap that lies on its side, leaning 25 degrees forward, and takes its neck between its
   fingers 12 mm from the mouth (the 4 in top is wider than the gripper opens, and a hand above it cannot reach
   the neck);
2. lifts it and leans forward to 85 degrees, which turns the cap mouth-down under a level hand, the square top
   above the fingers;
3. carries it over the bar, lowers it onto the bar's end and pushes it down until the bar's end is against the steel
   insert;
4. lets go, draws straight back from under the top, and lifts away.
The left arm is not used.

Controls: ``--mode perch`` sets the first cap on its bar without pushing and lets go. ``--mode mushroom`` puts the
mushroom cap that lies on its side onto the first bar in the same way. In both the second bar gets a rated cap, pushed
home. ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``rebarkit.py``. The only feedback is
the hand's own measured position (the robot's proprioception), used to correct for the arm's sag. The demo never
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
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
        cost = (pe > 5e-4 or re > 5e-3, -room)
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0]


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
TOOLROT: dict = {}                       # ... and which way the hand actually points

UP = np.array([0.0, 0.0, 1.0])
LEAN_HOLD = 85.0                         # the hand leans this far from straight down once the cap hangs mouth-down
GRIP_UP = 0.012                          # the tool point this far up the neck from the mouth
NECK_OPEN, NECK_GRIP = 0.062, 0.027      # pads apart on the way in, and the command when closed on the 36 mm neck
STEM_OPEN, STEM_GRIP = 0.050, 0.019      # the same for a mushroom cap's 25 mm stem
CLOSE = -1.0                             # which way round the hand is (+1: its closing axis along +y)
OVER = 0.015                             # the cap's mouth clears the bar's end by this much on the way over
PUSH = 0.010                             # the hand's command goes this far below where it would carry no load: it pushes
PERCH = 0.017                            # control: the cap is let go with the bar's end this far up its bore
BACK = 0.095                             # the hand draws back this far from under the cap's top before it lifts
BACK_SMALL = 0.045                       # ... and this far from under a mushroom cap's head
CARRY_Z = 0.200                          # the hand turns a cap mouth-down, and crosses the table, with its tool point this high
TURN_Z = 0.160                           # ... and turns back upright for the next cap this high
STAGE_X = 0.215                          # the cap is brought here, in front of its bar, before it goes over it
WAIT_XY, WAIT_Z = (0.170, -0.270), 0.240  # where the hand goes first, on its own side, before it reaches for a cap
PARK = (0.240, -0.290, 0.260)            # ... and where it ends
GAIN, BIAS_MAX = 0.25, 0.025             # how fast, and how far, the hand corrects for its own sag
LEAN_GAIN, LEAN_MAX = 0.10, math.radians(15.0)   # ... and for the droop of its wrist under a cap's weight


class Sag:
    """Slowly offsets the hand's command by its measured error, so that its tool point ends up where it is wanted;
    and, while ``level`` is on, leans the command back by however far the loaded wrist is measured to droop."""

    def __init__(self, hand):
        self.hand, self.bias, self.want, self.frozen = hand, np.zeros(3), None, False
        self.level, self.droop, self.want_rot = False, 0.0, None

    @staticmethod
    def lean(rot) -> float:
        """How far a hand leans from straight down toward +x (rad)."""
        return math.atan2(rot[0, 2], -rot[2, 2])

    def to(self, pos, rot):
        if self.want is not None:
            err = self.want - TOOL[self.hand.name]
            if self.frozen:                                # while the hand presses down on the bar: sideways only
                err = err * np.array([1.0, 1.0, 0.0])
            self.bias = np.clip(self.bias + GAIN * err, -BIAS_MAX, BIAS_MAX)
            if self.level and self.want_rot is not None:
                d = self.lean(TOOLROT[self.hand.name]) - self.lean(self.want_rot)
                self.droop = float(np.clip(self.droop + LEAN_GAIN * d, -LEAN_MAX, LEAN_MAX))
        self.want, self.want_rot = np.asarray(pos, float), rot
        c, s_ = math.cos(self.droop), math.sin(self.droop)
        back = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])       # a turn about y that takes lean off
        self.hand._solve(self.want + self.bias, back @ rot)


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


def lean_frame(lean_deg: float) -> np.ndarray:
    """The hand leaning ``lean_deg`` from straight down toward +x, its fingers closing across y."""
    a = math.radians(lean_deg)
    return frame((math.sin(a), 0.0, -math.cos(a)), (0.0, CLOSE, 0.0))


def job(T, cap: str, bar: int, mushroom: bool = False) -> dict:
    """The fixed targets for putting one lying cap on one bar, from the task file's layout (world, m)."""
    K = T.K
    pos, quat = T.mushroom_start(cap) if mushroom else T.cap_start(cap)
    dims = (K.M_STEM_R, K.M_HEAD_R, K.M_STEM_L) if mushroom else (K.NECK_R, K.PLATE / 2, K.NECK_L)
    tilt = math.degrees(K.lying(0.0, *dims)[2])
    take = K.place(pos, quat, (0.0, 0.0, GRIP_UP))                 # the grip point on the lying cap
    lean0 = tilt - (90.0 - LEAN_HOLD)                              # so that at LEAN_HOLD the cap hangs plumb
    tip = T.bar_tip(bar)
    seat = tip[2] - (K.M_STEM_L if mushroom else K.INSERT_Z) + GRIP_UP
    return {"take": take, "rot0": lean_frame(lean0), "rot1": lean_frame(LEAN_HOLD), "lean0": lean0,
            "bar": tip, "over_z": tip[2] + OVER + GRIP_UP, "seat_z": seat,
            "open": gap(STEM_OPEN if mushroom else NECK_OPEN), "grip": gap(STEM_GRIP if mushroom else NECK_GRIP)}


def plan(T, mode: str = "full") -> list:
    """The jobs in order: (cap, bar, how). ``how``: "home" pushed to the stop, "perch", or "mushroom"."""
    first = {"full": "home", "perch": "perch", "mushroom": "mushroom"}[mode]
    bars = T.BARE
    return [("mushroom_c" if first == "mushroom" else "cap_a", bars[0], first), ("cap_b", bars[1], "home")]


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    if mode == "still":
        yield from wait(idle_s)
        return
    sag = Sag(right)
    first = True
    for cap, bar, how in plan(T, mode):
        j = job(T, cap, bar, mushroom=how == "mushroom")
        r0, r1 = j["rot0"], j["rot1"]
        ap = r0[:, 2]
        # 1. Down the hand's own line onto the neck, and close.
        back = j["take"] - 0.085 * ap
        if first:
            # back and across first, clear of whatever stands on the nearest bar, then over to the cap
            high = np.array([WAIT_XY[0], WAIT_XY[1], T.TOP + WAIT_Z])
            yield from right.move_joints(best_ik(right, high, r0), 1.8, grip=j["open"])
            yield from glide(sag, high, 0.3, r0)
            yield from glide(sag, back, 1.8, r0)
            first = False
        else:
            yield from right.set_grip(j["open"], 0.3)
            above = np.array([back[0], back[1], T.TOP + TURN_Z])
            yield from glide(sag, above, 2.4, r0, rot_from=r1)          # turn back upright well above everything
            yield from glide(sag, back, 1.4, r0)
        yield from glide(sag, back, 0.4, r0)
        yield from glide(sag, j["take"], 1.6, r0)
        yield from glide(sag, j["take"], 0.4, r0)
        yield from right.set_grip(j["grip"], 0.8)
        yield from wait(0.3)
        sag.level = True                                    # from here the wrist carries the cap: watch its droop
        # 2. Lift, and lean forward until the cap hangs mouth-down, in front of its bar.
        yield from glide(sag, j["take"] + 0.050 * UP, 1.0, r0)
        turned = np.array([j["take"][0] + 0.030, j["take"][1], T.TOP + CARRY_Z])
        yield from glide(sag, turned, 3.4, r1, rot_from=r0)             # straight up, turning as it goes
        stage = np.array([STAGE_X, j["bar"][1], j["over_z"]])
        yield from glide(sag, stage, 2.0, r1)
        yield from glide(sag, stage, 0.4, r1)
        # 3. Over the bar, down onto it, and home.
        over = np.array([j["bar"][0], j["bar"][1], j["over_z"]])
        yield from glide(sag, over, 2.0, r1)
        yield from glide(sag, over, 1.0, r1)
        home = np.array([over[0], over[1], j["seat_z"]])
        if how == "perch":
            low = home + (T.K.INSERT_Z - PERCH) * UP
            yield from glide(sag, low, 1.6, r1)
            yield from glide(sag, low, 0.4, r1)
        elif how == "mushroom":
            low = home + 0.002 * UP
            yield from glide(sag, low, 2.4, r1)
            yield from glide(sag, low, 0.4, r1)
        else:
            touch = home + (T.K.INSERT_Z - 0.016) * UP          # the bar's end is about to meet the fins
            yield from glide(sag, touch, 1.4, r1)
            sag.frozen = True                                   # from here the bar pushes back: no more correction in height
            carry = float(sag.bias[2])                          # the hand's command stands this far above its tool point
            #                                                     while it carries the cap; on the bar it carries nothing
            yield from glide(sag, home - (carry + PUSH) * UP, 3.2, r1)
            yield from glide(sag, home - (carry + PUSH) * UP, 0.6, r1)
            yield from glide(sag, home - carry * UP, 0.6, r1)   # ease off: neither pushing nor lifting
        # 4. Let go, draw straight back from under the top, lift away.
        yield from right.set_grip(j["open"], 0.6)
        yield from wait(0.3)
        sag.frozen, sag.level, sag.droop = False, False, 0.0
        sag.bias[2] = 0.0                                       # the hand is empty again
        low = TOOL["right"].copy()
        sag.want = low.copy()
        out = low - (BACK_SMALL if how == "mushroom" else BACK) * np.array([1.0, 0.0, 0.0])
        yield from glide(sag, out, 1.6, r1)
        yield from glide(sag, np.array([out[0], out[1], T.TOP + CARRY_Z]), 1.0, r1)
    yield from glide(sag, np.array([PARK[0], PARK[1], T.TOP + PARK[2]]), 2.0, r1)
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1500, record: bool = True, window: bool = False,
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
    obs = emb.reset(Scene(id="rebar-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [*T.CAPS, *T.MUSHROOMS]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from rebarkit import finger_frames, finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        TOOLROT["left"], TOOLROT["right"] = finger_frames(emb._env)

    def hands() -> dict:
        from rebarkit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1),
                            "lean_deg": round(math.degrees(Sag.lean(TOOLROT[h.name])), 1)}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        up = lambda q: round(math.degrees(math.acos(float(np.clip(T.K._rotation(q)[2, 2], -1.0, 1.0)))), 1)   # noqa: E731
        return {n: mm(st[n].pos) + [up(st[n].quat_xyzw)] for n in names}

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
            print(f"step {steps} score {t['score']} {t['task']} R {t['hands']['right']} {t['places']}", flush=True)
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
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "perch", "mushroom", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
