"""Demo for the lampshade task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1200] [--no-record] [--mode full|drop|alone|aside|low|still]

1. The left hand comes in low from the left and takes the lamp's stem between its fingers, 6.5 cm above the table,
   under where the shade will hang.
2. The right hand takes the shade by its top from outside, a fingertip on each of two opposite panels just under the
   top edge, the hand leaning back toward the robot so that the arm can reach high enough to clear the bulb.
3. It lifts the shade until its bottom is 2 cm above the bulb's tip, carries it over the lamp, lowers it down the
   lamp's axis until the fitter's ring is on the bulb's tip, and presses 3 mm further.
4. The right hand lets go, lifts away and moves aside; the left hand lets go of the stem and draws back.

Controls: ``--mode drop`` carries the shade over the lamp the same way and lets go of it 3 cm above its seat, without
pressing it home. ``--mode alone`` is the full motion without the left hand. ``--mode aside`` is the same, the shade
brought down 1.5 cm to one side of the lamp's axis. ``--mode low`` is the same as ``alone``, the shade carried across
with its bottom 2 cm below the bulb's tip instead of above it. ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``shadekit.py``. The only feedback is
each hand's own measured position (the robot's proprioception): to correct for the arms' sag, and, once the left hand
has closed on the stem, to tell where the stem is. The demo never reads a part's state.
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


def together(*gens):
    """Generator: run several generators side by side, one step of each per control step, until all are done."""
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
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
        cost = (pe > 5e-4 or re > 5e-3, -room)
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0]


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)

UP = np.array([0.0, 0.0, 1.0])
LEAN = 66.0                              # the right hand leans this far from straight down (at 66 degrees the lower edge of each
                                         # fingertip is level, and lies along the panel it presses on)
RISE = 0.002                             # its tool point this far above the shade's top (the fingertips' lower edges are 12.5 mm
                                         # below the tool point, so they meet the panels 10.5 mm under the top edge)
BACK = 0.0014                            # ... and this far behind the shade's axis, which puts the two edges level with the axis
SPAN_OPEN, SPAN_GRIP = 0.092, 0.080      # pads apart on the way down, and closed on the shade (83 mm across there)
OVER_TIP = 0.020                         # the shade's bottom clears the bulb's tip by this much on the way over
PRESS = 0.003                            # the right hand is sent this far below the seated height
CLOSE = -1.0                              # which way round the right hand is (+1: its closing axis along the shade's +y)
STEM_AT = 0.065                          # the left hand takes the stem this far above the table
STEM_LEAN = 75.0                         # ... leaning this far from straight down
STEM_OPEN, STEM_GRIP = 0.040, 0.016      # pads apart on the way in, and closed on the 2 cm stem
DROP_UP = 0.030                          # control: the shade is let go this far above its seat
ASIDE = 0.015                            # control: the shade comes down this far to one side of the lamp's axis
TOO_LOW = 0.020                          # control: the shade's bottom is carried across this far below the bulb's tip
GAIN, BIAS_MAX = 0.25, 0.025             # how fast, and how far, a hand corrects for its own sag


class Sag:
    """Slowly offsets the hand's command by its measured error, so that its tool point ends up where it is wanted."""

    def __init__(self, hand):
        self.hand, self.bias, self.want, self.frozen = hand, np.zeros(3), None, False

    def to(self, pos, rot):
        if self.want is not None and not self.frozen:      # frozen while the hand presses on something
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


def shade_hold(K) -> tuple:
    """The right hand's hold on the shade, in the shade's frame: tool rotation and tool point. A fingertip on the
    outside of panel 0 and one on the outside of panel 3 (the two panels that face the shade's +y and -y), the fingers
    pointing along the shade's x and down."""
    th = math.radians(LEAN)
    return frame((math.sin(th), 0.0, -math.cos(th)), (0.0, CLOSE, 0.0)), np.array([-BACK, 0.0, K.HS + RISE])


def span_at(K, below_top: float) -> float:
    """The shade's width across panels 0 and 3, outside, this far below its top."""
    return K.TOP_AF + 2 * below_top * math.tan(K.SLOPE) + K.WALL / math.cos(K.SLOPE)


def plan(T) -> dict:
    """The demo's fixed targets, from the task file's layout (world, m)."""
    K = T.K
    rot_s, at_s = shade_hold(K)
    turn = K._rz(T.SHADE_YAW)
    sp = T.shade_start()[0]
    lp = T.lamp_start()[0]
    hold_rot, offset = turn @ rot_s, turn @ at_s                 # the shade keeps its heading all the way
    lean = math.radians(STEM_LEAN)
    stem_ap = np.array([0.0, -math.sin(lean), -math.cos(lean)])  # the left hand points to the right and a little down
    return {"hold_rot": hold_rot, "offset": offset, "take": sp + offset,
            "carry_z": T.TOP + K.TIP_Z + OVER_TIP + offset[2], "seat_z": T.TOP + K.TOP_SEATED - K.HS + offset[2],
            "lamp": lp, "stem_ap": stem_ap, "stem_rot": frame(stem_ap, (-1.0, 0.0, 0.0)),
            "stem": np.array([lp[0], lp[1], T.TOP + STEM_AT])}


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    if mode == "still":
        yield from wait(idle_s)
        return
    p = plan(T)
    rot, offset = p["hold_rot"], p["offset"]
    lsag, rsag = Sag(left), Sag(right)
    use_left = mode in ("full", "drop")

    def take_stem():
        back = p["stem"] - 0.080 * p["stem_ap"]
        yield from left.move_joints(best_ik(left, back, p["stem_rot"]), 2.0, grip=gap(STEM_OPEN))
        yield from glide(lsag, back, 0.4, p["stem_rot"])
        yield from glide(lsag, p["stem"], 1.6, p["stem_rot"])
        yield from glide(lsag, p["stem"], 0.4, p["stem_rot"])
        yield from left.set_grip(gap(STEM_GRIP), 0.6)
        yield from wait(0.3)

    def take_shade():
        above = p["take"] + 0.070 * UP
        yield from right.move_joints(best_ik(right, above, rot), 2.2, grip=gap(SPAN_OPEN))
        yield from glide(rsag, above, 0.4, rot)
        yield from glide(rsag, p["take"], 1.6, rot)
        yield from glide(rsag, p["take"], 0.4, rot)
        yield from right.set_grip(gap(SPAN_GRIP), 0.8)
        yield from wait(0.3)

    # 1 and 2. The left hand takes the stem while the right hand takes the shade.
    yield from together(*([take_stem()] if use_left else []), take_shade())

    # 3. Up, over the lamp, down its axis. Once the left hand has closed on the stem, the stem is between its pads:
    #    the left hand's own measured position tells where the lamp's axis is across them.
    axis = p["lamp"][:2].copy()
    if use_left:
        axis[0] = TOOL["left"][0]
    if mode == "aside":
        axis[0] += ASIDE
    carry_z = p["carry_z"] - (OVER_TIP + TOO_LOW if mode == "low" else 0.0)
    over = np.array([axis[0] + offset[0], axis[1] + offset[1], carry_z])
    yield from glide(rsag, (p["take"][0], p["take"][1], carry_z), 3.0, rot)
    yield from glide(rsag, over, 3.0, rot)
    yield from glide(rsag, over, 0.6, rot)
    if mode == "drop":
        low = np.array([over[0], over[1], p["seat_z"] + DROP_UP])
        yield from glide(rsag, low, 2.5, rot)
        yield from glide(rsag, low, 0.4, rot)
    else:
        home = np.array([over[0], over[1], p["seat_z"]])
        yield from glide(rsag, home + 0.012 * UP, 3.0, rot)
        rsag.frozen = True                               # from here the bulb pushes back: no more sag correction
        yield from glide(rsag, home - PRESS * UP, 1.2, rot)
        yield from glide(rsag, home - PRESS * UP, 0.5, rot)
        low = home - PRESS * UP

    # 4. The right hand lets go and lifts away; the left hand lets go and draws back.
    yield from right.set_grip(gap(SPAN_OPEN), 0.6)
    yield from wait(0.3)
    rsag.frozen = False
    yield from glide(rsag, low + 0.070 * UP, 1.5, rot)
    aside = glide(rsag, low + 0.070 * UP + np.array([-0.060, -0.140, 0.0]), 2.0, rot)   # out of the camera's view of the lamp
    if use_left:
        yield from left.set_grip(gap(STEM_OPEN), 0.6)
        yield from wait(0.2)
        yield from together(glide(lsag, p["stem"] - 0.090 * p["stem_ap"], 1.5, p["stem_rot"]), aside)
    else:
        yield from aside
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, window: bool = False,
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
    obs = emb.reset(Scene(id="shade-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.LAMP, T.SHADE, *T.JAWS]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from shadekit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))

    def hands() -> dict:
        from shadekit import _np

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
        return {n: mm(st[n].pos) for n in (T.LAMP, T.SHADE)}

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
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "drop", "alone", "aside", "low", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
