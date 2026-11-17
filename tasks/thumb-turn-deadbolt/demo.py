"""Scripted full-score demo for the thumb-turn deadbolt task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record] [--turn-s 3]

The left arm, its fingers closed and pointing at the door, presses the door shut against its weatherstrip about
halfway across and keeps pressing. The right arm comes in level, its fingers open either side of the thumb-turn's
upright wing, pinches it and turns the hand a quarter turn toward the latch edge, which throws the bolt into the strike.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder. With
DEMO_DEBUG=1 it also writes trace.csv: the gap, the thumb-turn's angle, the bolt's travel and the score after every
control step.
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


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)


def reachable_yaw(arm, pos, yaw):
    """The grip yaw, or the same parallel grip turned 180 degrees, whichever the arm reaches more exactly."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in (yaw, yaw + math.pi, yaw - math.pi):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def _rot_x(angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def level_grip(roll: float = 0.0) -> np.ndarray:
    """The TCP rotation of a level grip pointing forward (+x), fingers closing along world y, rolled ``roll`` (rad)
    about the pointing axis."""
    from yam_arena.demo import grip_rotation

    pitch = np.array([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]])   # turns the straight-down grip to point along +x
    return _rot_x(roll) @ pitch @ grip_rotation(0.0)


# The demo runs open loop: it reads no simulator state while it runs (DEMO_DEBUG only prints it). Every target comes
# from the published layout in task.py; the layout is fixed (object_jitter=0).
TCP_ABOVE_TIP = 0.013   # the pads' tips are 13 mm from the TCP along the hand's pointing axis
PRESS_Y = 0.050         # the left hand presses the door 280 mm from its hinge axis, at the bolt's height
PRESS_PAST = 0.014      # its fingertips are sent 14 mm past the shut door's face: the compliant arm presses it shut
TIP_GAP = 0.0115        # the right fingertips stop 11.5 mm in front of the rose, 1.5 mm short of the thumb-turn's round
#                         hub, so the pads close on the flat wing (16.5 of its 18 mm depth), not on the hub
R_OPEN = 46 / 94        # the right fingers open 46 mm round the 12 mm wing
ROLL = math.radians(100)    # the hand is turned 100 degrees: the thumb-turn stops at 90 and the compliant wrist lags


def script(task, ctx, turn_s: float = 3.0):
    """The demo as a generator: one yield per control step. ``turn_s`` is how long the quarter turn takes."""
    L, R = ctx.left, ctx.right
    T = task
    rot0 = level_grip(0.0)
    press = np.array([T.XF + PRESS_PAST - TCP_ABOVE_TIP, PRESS_Y, T.BZ])
    pinch = np.array([T.XF - 0.012 - TIP_GAP - TCP_ABOVE_TIP, T.TURN_Y, T.BZ])
    back = np.array([-0.07, 0, 0])

    # 1. Left arm: fingers closed, come in level in front of the door and press it shut, and keep pressing.
    yield from L.move(press + back, rot0, 2.5, grip=0.0)
    yield from L.move(press, rot0, 2.0)
    yield from wait(0.5)

    # 2. Right arm: fingers open either side of the upright wing, in until the fingertips are just short of its hub,
    #    and pinch the wing.
    yield from R.move(pinch + back, rot0, 2.5, grip=R_OPEN)
    yield from R.move(pinch, rot0, 1.5)
    yield from R.set_grip(0.0, 0.8)
    yield from wait(0.3)

    # 3. Turn the hand a quarter turn about the thumb-turn's axis, its top toward the latch edge: the cam throws the
    #    bolt into the strike. Hold it there.
    yield from R.move(pinch, level_grip(ROLL), turn_s)
    yield from wait(3.0)


def _tilt(state):
    """The door's (roll, pitch) in rad: it should only turn about its vertical hinge axis."""
    from mechkit import quat_to_mat

    r = quat_to_mat(state.quat_xyzw)
    return np.array([math.atan2(r[2, 1], r[2, 2]), -math.asin(max(-1.0, min(1.0, r[2, 0])))])


def _bolt_off(task, st, part="bolt"):
    """A part's centre in the door's frame, minus where it is with the door shut and the bolt retracted (m): for the
    bolt x, y (thrown is -y), z; for the door itself, how far it has drifted off its hinge axis."""
    from mechkit import quat_to_mat

    r = quat_to_mat(st["door"].quat_xyzw)
    ref = task.BOLT_REF if part == "bolt" else np.asarray(task.DOOR_O) + task.kit.parts["door"].centre
    return r.T @ (np.asarray(st[part].pos) - task.HINGE_AXIS_PT) + task.HINGE_AXIS_PT - ref


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0, turn_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx, turn_s)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    trace = []   # DEMO_DEBUG only: (step, gap mm, thumb-turn deg, bolt mm, score) after every control step
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if os.environ.get("DEMO_DEBUG"):
            r = task.readings(ctx.states())
            trace.append((steps, r["gap_mm"], r["turn_deg"], r["bolt_mm"], info.get("score")))
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 20 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                from yam_arena import ik, station

                st14 = np.asarray(next(iter(res.observation.state.values())), float).ravel()
                for k, arm in enumerate(("left", "right")):
                    p_, _ = ik.tcp_pose(st14[7 * k:7 * k + 6], station.BASES[arm])
                    msg += f" | {arm[0]} tcp {np.round(1000 * p_, 1)} grip {st14[7 * k + 6]:.2f}"
                msg += (f" | door gap {1000 * task.door_gap(st):.2f}mm bolt {1000 * task.bolt_out(st):.2f}mm"
                        f" turn {task.turn_deg(st):.1f}deg strip {1000 * task.kit.q(st, 'strip_spring'):.1f}mm"
                        f" | door yaw {math.degrees(task.door_angle(st)):.3f}"
                        f" tilt {np.round(np.degrees(_tilt(st['door'])), 3)}"
                        f" bolt off {np.round(1000 * _bolt_off(task, st), 2)}"
                        f" door drift {np.round(1000 * _bolt_off(task, st, 'door'), 2)} j6 r {st14[12]:.2f}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "final": task.readings(ctx.states()),   # read once, after the last step: gap, thumb-turn angle, bolt
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if trace:
        (out / "trace.csv").write_text("step,gap_mm,turn_deg,bolt_mm,score\n" + "".join(
            f"{k},{g},{t},{bo},{sc}\n" for k, g, t, bo, sc in trace))
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--turn-s", type=float, default=3.0,
                    help="seconds for the quarter turn (3 in the demo; 30 with DEMO_DEBUG=1 for a slow pass that "
                         "shows the bolt's travel against the thumb-turn's angle in trace.csv)")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, turn_s=a.turn_s)
