"""Demo for the transfer-forceps task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1800] [--no-record] [--mode full|one|over|hand|still]

Only the right arm works; the left stays where it starts, clear of both sterile areas.

1. The right hand, leaning 72 degrees forward so that its fingers are nearly level, takes the forceps by the part of
   their handle that sticks out over the field's edge, and squeezes them until the tips are about 6 mm apart.
2. It lifts the handle while the tips stay low: straightened to a 28 degree lean, the hand holds the forceps pointing
   44 degrees down, their tips 6.5 cm ahead of its fingertips. From here on the hand works from in front of the sterile areas
   and only the forceps reach over them.
3. For each cup: the tips open to 10 mm and come down astride the cup's wall at its side, one tip inside the cup and
   one outside, 5.5 mm below the rim; the hand squeezes; the cup is lifted (it hangs tipped from its rim), carried along
   the front of the pack and the field, let go just above the field inside its border, where it rights itself.
4. The forceps go back across the field's near edge, working end on the field, handle out over the edge, and the hand
   lets go and draws back.

Controls: ``--mode one`` moves one cup only; ``--mode over`` sets the cups 5 cm further into the field, so that the
hand passes over it; ``--mode hand`` moves a cup with the gripper itself; ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``xferkit.py``. The only feedback is
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
PAD_STIFF = 5000.0                 # N/m: each finger's position gain (yam_arena.constants.GRIPPER)


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

L_PICK = math.radians(72.0)              # the hand's lean (from straight down, forward) when it takes or lays the forceps
L_WORK = math.radians(28.0)              # ... and while it works with them: the forceps then point 44 degrees down
PINCH = 0.030                            # the pads hold the forceps this far from the tail, at the blades' widest point
TIPS_HOLD = 0.005                        # m between the tips while the forceps are carried empty
TIPS_OPEN = 0.010                        # ... and when they come down astride a cup's wall
SQUEEZE = 0.001                          # pad gap commanded to grip a cup (about 16 N on each pad, under 3 N at the tips)
DEPTH = 0.0055                           # the tips' lower edges go this far below the cup's rim
PAST = 0.003                             # ... and their ends this far past the middle of the cup
LIFT = 0.045                             # a held cup is carried this far above where it stood
HANG = 0.014                             # a cup held by its rim hangs tipped about 22 degrees: its lowest edge is this far
                                         # below where its base would be if it hung straight, and its base's middle
SWING = (0.004, 0.018)                   # ... this far (x, y) toward the tips. It rights itself when it is set down.
DROP = 0.002                             # a cup is let go with its lowest edge this far above the field
NOSE = math.radians(0.7)                 # the forceps are laid down this much nose down: tips first, then the handle
SLOTS = ((0.392, 0.035), (0.392, -0.045))    # where the cups go on the field (their middles)
GAIN, BIAS_MAX = 0.25, 0.020             # how fast, and how far, the hand corrects for its own sag


def approach(lean: float) -> np.ndarray:
    return np.array([math.sin(lean), 0.0, -math.cos(lean)])


def rot_for(lean: float) -> np.ndarray:
    return frame(approach(lean), (0.0, -1.0, 0.0))


def axis(lean: float) -> np.ndarray:
    """Which way the held forceps point when the hand leans ``lean``: they were taken lying level at ``L_PICK``."""
    t = L_PICK - lean
    return np.array([math.cos(t), 0.0, -math.sin(t)])


def pads_for(K, tips: float) -> float:
    """Pad gap to command for the tips to stand ``tips`` m apart: the blades' width there, less what the fingers'
    position control gives under the spring's push."""
    width = 2 * (K.F_BOW + K.F_TH) + tips * PINCH / K.F_LEN
    push = K.F_CLOSE_N * K.F_GRIP / PINCH * (1.0 - tips / K.F_OPEN)
    return width - 2 * push / PAD_STIFF


class Sag:
    """Slowly offsets the hand's command by its measured error, so that its tool point ends up where it is wanted."""

    def __init__(self, hand):
        self.hand, self.bias, self.want = hand, np.zeros(3), None

    def to(self, pos, rot):
        if self.want is not None:
            self.bias = np.clip(self.bias + GAIN * (self.want - TOOL[self.hand.name]), -BIAS_MAX, BIAS_MAX)
        self.want = np.asarray(pos, float)
        self.hand._solve(self.want + self.bias, rot)


def glide(sag, target, seconds: float, rot):
    """Generator: take the hand's tool point in a straight line to ``target``, correcting for sag on the way."""
    from yam_arena import ik

    start = sag.want if sag.want is not None else sag.hand.pos.copy()
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        sag.to(start + ik.smoothstep(k / n) * (np.asarray(target, float) - start), rot)
        yield


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    K = T.K
    if mode == "still":
        yield from wait(idle_s)
        return
    lever = K.F_LEN - PINCH
    field_z, pack_z = T.TOP + T.FIELD_TOP, T.TOP + T.PACK_TOP
    sag = Sag(right)

    def tool(tip, lean: float) -> np.ndarray:                       # the hand's tool point for the forceps' tips at ``tip``
        return np.asarray(tip, float) - lever * axis(lean)

    tip_now = np.zeros(3)

    def tips_to(tip, seconds: float, lean: float = L_WORK):
        yield from glide(sag, tool(tip, lean), seconds, rot_for(lean))
        tip_now[:] = tip

    def pivot(tip0, tip1, lean0: float, lean1: float, seconds: float):
        from yam_arena import ik

        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            lean = lean0 + s * (lean1 - lean0)
            sag.to(tool(np.asarray(tip0, float) + s * (np.asarray(tip1, float) - tip0), lean), rot_for(lean))
            yield
        tip_now[:] = tip1

    if mode == "hand":
        # control: the gripper itself moves a cup
        down = frame((0.0, 0.0, -1.0), (0.0, -1.0, 0.0))
        c = T.cup_start(0)
        over = np.array([c[0], c[1], pack_z + K.C_H + 0.06])
        yield from right.move_joints(best_ik(right, over, down), 2.5, grip=gap(0.085))
        yield from glide(sag, over, 0.5, down)
        yield from glide(sag, (c[0], c[1], pack_z + K.C_H - 0.012), 1.5, down)
        yield from right.set_grip(gap(0.040), 0.8)
        yield from glide(sag, over, 1.2, down)
        yield from glide(sag, (SLOTS[0][0], SLOTS[0][1], over[2]), 2.5, down)
        yield from glide(sag, (SLOTS[0][0], SLOTS[0][1], field_z + K.C_H - 0.010), 1.5, down)
        yield from right.set_grip(gap(0.085), 0.8)
        yield from glide(sag, (SLOTS[0][0] - 0.10, SLOTS[0][1], over[2] + 0.03), 2.0, down)
        yield from wait(3.0)
        return

    # 1. Take the forceps by the handle that sticks out over the field's edge.
    tail = T.tail_start()
    rest_tip = tail + (K.F_LEN, 0.0, 0.0)
    grasp = tool(rest_tip, L_PICK)
    back = grasp - 0.07 * approach(L_PICK)
    yield from right.move(back, 2.5, rot=rot_for(L_PICK), grip=gap(0.030))     # straight there: never forward of where it starts
    yield from glide(sag, back, 0.5, rot_for(L_PICK))
    yield from glide(sag, grasp, 1.5, rot_for(L_PICK))
    yield from glide(sag, grasp, 0.4, rot_for(L_PICK))
    yield from right.set_grip(gap(pads_for(K, TIPS_HOLD)), 0.8)
    yield from wait(0.3)

    # 2. Lift the handle, the tips staying low, and draw the forceps back so that the hand stays in front of the field.
    work_tip = np.array([T.NEAR + T.BORDER + 0.035, tail[1], field_z + 0.030])
    yield from pivot(rest_tip, work_tip, L_PICK, L_WORK, 3.0)

    # 3. The cups, one at a time.
    deeper = 0.05 if mode == "over" else 0.0
    side = K._wall_r(K.C_H) - K.C_WALL / 2 - 0.0002                    # the middle of what the tips close on, from the cup's axis
    hold_z = K.C_H - DEPTH + 0.0011                                     # the tips' middle above the base of a cup they hold
    travel = pack_z + hold_z + LIFT                                     # the tips' height while they travel
    for i in ((0,) if mode == "one" else (0, 1)):
        c = T.cup_start(i)
        grab = np.array([c[0] + PAST, c[1] + side, pack_z + hold_z])
        before = grab - 0.035 * axis(L_WORK)
        yield from tips_to((tip_now[0], tip_now[1], travel), 1.0)
        yield from tips_to((before[0], grab[1], travel), 2.5)
        yield from right.set_grip(gap(pads_for(K, TIPS_OPEN)), 0.5)
        yield from tips_to(before, 1.2)
        yield from tips_to(before, 0.4)
        yield from tips_to(grab, 1.2)
        yield from tips_to(grab, 0.4)
        yield from right.set_grip(gap(SQUEEZE), 0.8)
        yield from wait(0.4)
        yield from tips_to(grab + (0.0, 0.0, LIFT), 1.5)
        put = np.array([SLOTS[i][0] + deeper + PAST + SWING[0], SLOTS[i][1] + side - SWING[1], field_z + DROP + HANG + hold_z])
        yield from tips_to((put[0], put[1], travel), 3.0)
        yield from tips_to(put, 1.5)
        yield from tips_to(put, 0.4)
        yield from right.set_grip(gap(pads_for(K, TIPS_OPEN)), 0.6)
        yield from wait(0.5)
        yield from tips_to(put - 0.045 * axis(L_WORK), 1.2)
        yield from right.set_grip(gap(pads_for(K, TIPS_HOLD)), 0.4)

    # 4. Lay the forceps back across the field's near edge and let go.
    yield from tips_to((work_tip[0], tip_now[1], travel), 1.0)
    yield from tips_to((work_tip[0], tail[1], travel), 2.5)
    yield from tips_to(work_tip, 1.0)
    lay = L_PICK - NOSE
    down_tip = rest_tip + (0.0, 0.0, 0.0005)
    yield from pivot(work_tip, down_tip + (0.0, 0.0, 0.004), L_WORK, lay, 3.0)
    yield from tips_to(down_tip, 0.8, lean=lay)
    yield from tips_to(down_tip, 0.4, lean=lay)
    yield from right.set_grip(gap(0.030), 1.2)
    yield from wait(0.3)
    yield from glide(sag, tool(down_tip, lay) - 0.08 * approach(lay), 1.5, rot_for(lay))
    yield from glide(sag, tool(down_tip, lay) - 0.10 * approach(lay) + (0.0, 0.0, 0.05), 1.0, rot_for(lay))
    yield from wait(3.0)


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
    obs = emb.reset(Scene(id="xfer-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.FORCEPS, T.BLADE, *T.CUPS]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from xferkit import finger_points

        (pl, _), (pr, rr) = finger_points(emb._env)
        TOOL["left"], TOOL["right"] = np.asarray(pl, float), np.asarray(pr, float)
        TOOL["right_rot"] = rr

    def hands() -> dict:
        from xferkit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1)}
        lean = math.degrees(math.atan2(float(TOOL["right_rot"][0, 2]), float(-TOOL["right_rot"][2, 2])))
        out_["right"]["lean_deg"] = round(lean, 1)
        return out_

    def places(st) -> dict:
        from xferkit import F_LEN, cup_tilt, place

        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.TOP]]   # noqa: E731
        d = {n: mm(st[n].pos) for n in names}
        d["tip_a"] = mm(place(st[T.FORCEPS].pos, st[T.FORCEPS].quat_xyzw, (F_LEN, 0.0, 0.0)))
        d["tilt_deg"] = [round(math.degrees(cup_tilt(st[n].quat_xyzw)), 1) for n in T.CUPS]
        return d

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
            print(f"step {steps} score {t['score']} {t['task']} {t['hands']['right']} {t['places']}", flush=True)
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
    ap.add_argument("--mode", default="full", choices=("full", "one", "over", "hand", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
