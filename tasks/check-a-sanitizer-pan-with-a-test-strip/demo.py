"""Demo for the sanitizer-pan task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1800] [--no-record]
                   [--mode full|testonly|always|blind|double|still]

1. The right hand takes the test strip out of its rest by its handle end, fingers on its two edges, and carries it
   over the pan's rim. Above the liquid it stops and looks at the strip's dry pad through its own wrist camera.
2. It lowers the strip until the pad is under the surface, holds it there for a second and a half, and lifts it back
   to the same place. It looks again: the pad's colour now, against its colour before, is the reading.
3. If the pad has not turned green (no sanitizer in the pan yet), the left hand takes one tablet off the board and
   drops it in the pan. If it has (the pan is already made up), the left hand does nothing.
4. The right hand drops the used strip in the waste cup.

Controls: ``--mode testonly`` tests and bins the strip but never adds a tablet; ``--mode always`` tests, then adds
one tablet whatever the strip showed; ``--mode blind`` adds a tablet without testing; ``--mode double`` tests, then
adds both tablets; ``--mode still`` keeps both arms still. In ``always`` and ``double`` the tablets go in before the
strip is binned.

Every target comes from the layout constants in the task file and the sizes in ``sanikit.py``. The feedback used is
what the robot has: each hand's own measured position (to correct an arm's sag before a placement) and the right
wrist camera's picture (to read the pad). The demo never reads the strip's, the tablets' or the solution's state.
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
SEEN: dict = {}                          # the latest camera pictures (updated every control step)
NOTES: dict = {}                         # what the demo read and decided, for result.json
AIM_MAX = 0.015                          # the most a placement is aimed off to make up for an arm's sag

# -- the right hand: the strip ------------------------------------------------------------------------------------------
RIGHT_DOWN = frame((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))   # straight down, fingers closing along x on the strip's two
                                                        # edges; the wrist camera is then on the pad's side (+y)
PINCH = 0.049                            # the strip is taken this far up from its pad end (1.5 cm from its top)
STRIP_OPEN, STRIP_SHUT = 0.030, 0.0050   # finger gap round the 6.4 mm strip / commanded on it
CARRY = 0.165                            # tool height at which the hanging pad end clears the pan's rim by 2 cm
LIFT = 0.060                             # the strip is first lifted this far, straight up out of its rest
PAD_DEEP = 0.012                         # the strip's end goes this far under the surface
DIP_HOLD = 1.6                           # s
# reading the pad from the wrist camera: the strip hangs in the middle of the picture
BAND = 0.14                              # look this far either side of the picture's centre line (fraction of width)
GREENER = 14                             # a pad pixel has turned if green now exceeds red and blue by this much ...
CHANGED = 30                             # ... and the pixel differs from the first look by this much (sum over RGB)
MIN_PIXELS = 4                           # at 320 x 176 the pad covers about 110 pixels

# -- the left hand: a tablet ------------------------------------------------------------------------------------------
LEFT_DOWN = frame((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))
TIP = 0.0135                             # the pads' tips are 13 mm past the tool point: the tool point stops this far up
TABLET_OPEN, TABLET_SHUT = 0.032, 0.0105  # finger gap round the 13 mm tablet / commanded on it
DROP_XY, DROP_Z = (0.300, 0.020), 0.125  # over the liquid, the fingertips 1 cm above the rim


def aim(hand, target):
    """Correct ``target`` once for the hand's measured sag, by at most AIM_MAX."""
    target = np.asarray(target, float)
    off = target - TOOL[hand.name]
    n = float(np.linalg.norm(off))
    if n > AIM_MAX:
        off = off * AIM_MAX / n
    return target + off


def settle(hand, target, rounds: int = 2):
    for _ in range(rounds):
        yield from chain(hand.move(aim(hand, target), 0.5), wait(0.3))


def down_strip(T) -> np.ndarray:
    """From the tool point along the held strip toward its pad end."""
    return -np.asarray(T.UP_STRIP, float)


def over_liquid(T, z: float) -> np.ndarray:
    """The tool point that puts the held strip's pad end over the dip point, with the tool at height ``z``."""
    end = np.array([0.290, -0.105])                                  # 4.5 cm in from the pan's near wall
    off = PINCH * down_strip(T)
    return np.array([end[0] - off[0], end[1] - off[1], T.TOP + z])


def pad_turned(before, after) -> dict:
    """Has the pad turned green between two pictures from the same place? Pure image arithmetic."""
    b, a = np.asarray(before).astype(int), np.asarray(after).astype(int)
    w = a.shape[1]
    lo, hi = int(w * (0.5 - BAND)), int(w * (0.5 + BAND))
    r, g, bl = a[..., 0], a[..., 1], a[..., 2]
    green = (g - r >= GREENER) & (g - bl >= GREENER)
    moved = np.abs(a - b).sum(axis=2) >= CHANGED
    mask = np.zeros(green.shape, bool)
    mask[:, lo:hi] = True
    hit = green & moved & mask
    n = int(hit.sum())
    out = {"pixels": n, "turned": n >= MIN_PIXELS}
    if n:
        ys, xs = np.nonzero(hit)
        out.update(mean_rgb=[int(v) for v in a[hit].mean(axis=0)], before_rgb=[int(v) for v in b[hit].mean(axis=0)],
                   at=[int(xs.mean()), int(ys.mean())])
    return out


def test(T, right):
    """Take the strip, dip its pad, and read it: NOTES["made_up"] says whether the pan already holds sanitizer."""
    at = T.strip_start() + PINCH * T.UP_STRIP
    yield from right.move_joints(best_ik(right, at + [0.0, 0.0, 0.06], RIGHT_DOWN), 2.5, grip=gap(STRIP_OPEN))
    yield from chain(right.move(at, 1.5), wait(0.2))
    yield from settle(right, at, 1)
    yield from chain(right.set_grip(gap(STRIP_SHUT), 0.6), wait(0.3))
    yield from right.move(right.pos + [0.0, 0.0, LIFT], 2.0)             # straight up, out of the rest
    look = over_liquid(T, CARRY)
    yield from right.move([right.pos[0], right.pos[1], look[2]], 1.0)
    yield from right.move(look, 2.5)
    yield from chain(settle(right, look, 1), wait(0.5))
    NOTES["before"] = np.array(SEEN["right_cam"])
    dip = over_liquid(T, T.SURFACE - PAD_DEEP - PINCH * down_strip(T)[2])        # the pad end PAD_DEEP under
    yield from right.move(dip, 1.6)
    yield from wait(DIP_HOLD)
    yield from right.move(look, 1.6)
    yield from chain(settle(right, look, 1), wait(0.5))
    NOTES["after"] = np.array(SEEN["right_cam"])
    NOTES["look"] = pad_turned(NOTES["before"], NOTES["after"])
    NOTES["made_up"] = bool(NOTES["look"]["turned"])


def bin_strip(T, right):
    """Drop the used strip in the waste cup."""
    mid = (PINCH - T.K.STRIP_L / 2) * down_strip(T)                               # the strip's middle, from the tool point
    over = np.array([T.CUP_XY[0] - mid[0], T.CUP_XY[1] - mid[1], T.TOP + CARRY])
    yield from right.move(over, 2.5)
    low = over.copy()
    low[2] = T.TOP + T.CUP_H + 0.012 - PINCH * down_strip(T)[2]                   # the pad end 1.2 cm above the rim
    yield from right.move(low, 1.2)
    yield from chain(right.set_grip(1.0, 0.6), wait(0.5))
    yield from right.move(low + [-0.10, 0.02, 0.03], 2.0)


def dose(T, left, name: str):
    """Take tablet ``name`` off the board and drop it in the pan."""
    x, y = T.TABLET_XY[name]
    at = np.array([x, y, T.TOP + T.BOARD[2] + TIP])                               # the pads' tips just on the board
    yield from left.move_joints(best_ik(left, at + [0.0, 0.0, 0.06], LEFT_DOWN), 2.5, grip=gap(TABLET_OPEN))
    yield from chain(left.move(at, 1.5), wait(0.2))
    yield from settle(left, at, 1)
    yield from chain(left.set_grip(gap(TABLET_SHUT), 0.6), wait(0.3))
    yield from left.move([x, y, T.TOP + DROP_Z], 1.5)
    yield from left.move([DROP_XY[0], DROP_XY[1], T.TOP + DROP_Z], 2.5)
    yield from chain(left.set_grip(1.0, 0.5), wait(0.5))
    yield from left.move([x, y + 0.04, T.TOP + DROP_Z], 2.5)


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    yield from wait(0.3)
    if mode == "still":
        yield from wait(idle_s)
        return
    names = list(T.TABLETS)
    if mode == "blind":
        yield from dose(T, left, names[0])
        yield from wait(2.0)
        return
    yield from test(T, right)
    if mode == "testonly":
        add = []
    elif mode == "double":
        add = names
    elif mode == "always":
        add = names[:1]
    else:
        add = [] if NOTES["made_up"] else names[:1]
    NOTES["tablets_added"] = len(add)
    if mode in ("double", "always"):
        # the tablets go in before the strip is binned: a trial ends as soon as it reaches full score, which a binned
        # strip (and, in a pan with no sanitizer, the first tablet) would do before the unwanted tablet got there
        yield from chain(*(dose(T, left, n) for n in add))
        yield from bin_strip(T, right)
    else:
        yield from together(bin_strip(T, right), chain(*(dose(T, left, n) for n in add)))
    yield from wait(2.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, window: bool = False,
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
    obs = emb.reset(Scene(id="sanitizer-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []

    def snap(o) -> None:
        for name, img in o.images.items():
            SEEN[name] = np.asarray(img)
            if record:
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, [T.STRIP, T.CUP, *T.TABLETS])

    def measure() -> None:
        from sanikit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))

    def hands() -> dict:
        from sanikit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, names = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[names.index(j)])) for j in ("joint7", "joint8")), 1)}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        return {"strip_end": mm(st[T.STRIP].pos), "cup": mm(st[T.CUP].pos), **{n: mm(st[n].pos) for n in T.TABLETS}}

    snap(obs)
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
        snap(res.observation)
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
    looked = {k: v for k, v in NOTES.items() if k not in ("before", "after")}
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended, "read": looked,
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": {"task": T.report(states()), "places": places(states())}, "trace": trace}
    import imageio

    for name in ("before", "after"):
        if name in NOTES:
            imageio.imwrite(out / f"look_{name}.png", NOTES[name])
    if record:
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
    ap.add_argument("--mode", default="full", choices=("full", "testonly", "always", "blind", "double", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
