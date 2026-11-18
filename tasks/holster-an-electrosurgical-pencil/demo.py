"""Demo for the electrosurgical pencil task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 900] [--no-record] [--mode full|pinch|tray|still]

1. The right hand comes down over the pencil's handle behind its two keys (10.5 cm back from the nose) and pinches it
   there, pads on the keyed face and the face opposite, where there is no key to press.
2. It lifts the pencil 5 cm, turns it until its electrode points into the holster's mouth, and lowers it in line
   with the holster. The cord trails behind over the drape.
3. It slides the pencil into the holster, electrode first, until its fingers are just short of the mouth's lip, lets
   go, and draws back.

Controls: ``--mode pinch`` takes the pencil the plain way, by the front of its handle: the pads close on the COAG key.
``--mode tray`` takes it behind the keys and lays it in the instrument tray. ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``esukit.py``. The only feedback is
the hand's own measured position (the robot's proprioception), which it uses to correct for its sag. The demo never
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

DOWN = (0.0, 0.0, -1.0)
TCP_UP = 0.014                           # tool point above the drape when the pads' tips are 1 mm above it
OPEN_GAP = 0.040                         # pads this far apart on the way down, astride the 1.6 cm handle
PINCH_AT = 0.049                         # control: the plain pinch closes this far behind the nose, over the COAG key
CARRY_UP = 0.050                         # tool point above the drape on the way to the holster
TRAY_CARRY_UP = 0.100                    # ... and on the way to the tray, over its rim
LEAD_IN = 0.020                          # the electrode's tip starts this far outside the holster's mouth
STOP_SHORT = 0.004                       # the fingers stop this far from the mouth's lip
TRAY_AT = (-0.055, 0.080)                # control: where the grasped point of the handle goes, in the tray's frame
GAIN, BIAS_MAX = 0.25, 0.020             # how fast, and how far, the hand corrects for its own sag


def grip_gap(K) -> float:
    return K.TALL - 0.002                # 1 mm of squeeze a side on the 1.6 cm handle


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


def plan(T) -> dict:
    """The demo's fixed targets, from the task file's layout (world, m)."""
    K = T.K
    (gx, gy), deg, _keys = T.LAYOUT
    yaw = math.radians(deg)
    back = np.array([math.cos(yaw), math.sin(yaw), 0.0])            # the direction the pencil's cord end points
    grasp = np.array([gx, gy, T.FLOOR + TCP_UP])
    hp, _ = T.holster_pose()
    inward = np.array([math.cos(T.HOLSTER_YAW), math.sin(T.HOLSTER_YAW), 0.0])
    # in line with the holster: the pencil's axis at the middle of the holster's height
    line_up = hp[2] + K.H_IN_H / 2 + (TCP_UP - K.WIDE / 2)
    mouth = np.array([hp[0], hp[1], line_up])
    before = mouth - (LEAD_IN + K.EXPOSED + T.GRASP) * inward       # the tool point with the tip LEAD_IN outside the mouth
    home = mouth - (K.H_FLARE + K.PAD_HALF + STOP_SHORT) * inward   # ... and with the fingers just short of the lip
    tp, _ = T.tray_pose()
    tray = np.array([tp[0] + TRAY_AT[0], tp[1] + TRAY_AT[1], tp[2] + TCP_UP])
    return {"back": back, "grasp": grasp, "pinch": grasp - (T.GRASP - PINCH_AT) * back,
            "take": frame(DOWN, (math.sin(yaw), -math.cos(yaw), 0.0)),          # pads on the keyed face and the one opposite
            "give": frame(DOWN, (-inward[1], inward[0], 0.0)),                  # the same hold, the electrode pointing into the holster
            "before": before, "home": home, "tray": tray, "nose_in": T.GRASP - (K.H_FLARE + K.PAD_HALF + STOP_SHORT)}


def script(T, left, right, mode: str = "full", idle_s: float = 45.0):
    """The demo as a generator: one yield per control step."""
    K = T.K
    if mode == "still":
        yield from wait(idle_s)
        return
    p = plan(T)
    up = np.array([0.0, 0.0, 1.0])
    at = p["pinch"] if mode == "pinch" else p["grasp"]

    # 1. Over the handle, down astride it, and take it.
    above = at + 0.080 * up
    yield from right.move_joints(best_ik(right, above, p["take"]), 2.0, grip=gap(OPEN_GAP))
    sag = Sag(right)
    yield from glide(sag, above, 0.4, p["take"])
    yield from glide(sag, at + 0.020 * up, 1.2, p["take"])
    yield from glide(sag, at, 1.0, p["take"])
    yield from glide(sag, at, 0.4, p["take"])
    yield from right.set_grip(gap(grip_gap(K)), 0.6)
    yield from wait(0.3)

    if mode == "pinch":                 # control: lift it, hold it, put it back
        yield from glide(sag, at + 0.030 * up, 0.8, p["take"])
        yield from wait(1.0)
        yield from glide(sag, at, 0.8, p["take"])
        yield from right.set_grip(gap(OPEN_GAP), 0.6)
        yield from glide(sag, at + 0.100 * up, 1.2, p["take"])
        yield from wait(3.0)
        return

    if mode == "tray":                  # control: lay it in the instrument tray
        high = T.FLOOR + TRAY_CARRY_UP
        yield from glide(sag, (at[0], at[1], high), 1.2, p["take"])
        yield from glide(sag, (p["tray"][0], p["tray"][1], high), 3.5, p["give"], rot_from=p["take"])
        yield from glide(sag, p["tray"], 1.5, p["give"])
        yield from glide(sag, p["tray"], 0.4, p["give"])
        yield from right.set_grip(gap(OPEN_GAP), 0.6)
        yield from wait(0.3)
        yield from glide(sag, p["tray"] + 0.080 * up, 1.2, p["give"])
        yield from wait(3.0)
        return

    # 2. Lift, turn the electrode toward the holster's mouth, and come down in line with the holster.
    carry = T.FLOOR + CARRY_UP
    yield from glide(sag, (at[0], at[1], carry), 0.8, p["take"])
    yield from glide(sag, (p["before"][0], p["before"][1], carry), 3.0, p["give"], rot_from=p["take"])
    yield from glide(sag, p["before"], 1.0, p["give"])
    yield from glide(sag, p["before"], 0.4, p["give"])

    # 3. In, electrode first, until the fingers are just short of the mouth; let go; draw back.
    yield from glide(sag, p["home"], 4.0, p["give"])
    yield from glide(sag, p["home"], 0.3, p["give"])
    yield from right.set_grip(gap(OPEN_GAP), 0.6)
    yield from wait(0.3)
    yield from glide(sag, p["home"] + 0.100 * up, 1.2, p["give"])
    yield from wait(3.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 900, record: bool = True, window: bool = False,
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
    obs = emb.reset(Scene(id="esu-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.PENCIL, *T.KEY_NAMES.values()]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states(which=names):
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, list(which))

    def measure() -> None:
        from esukit import finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))

    def hands() -> dict:
        from esukit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.FLOOR]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.FLOOR]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1)}
        return out_

    def places(st) -> dict:
        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float) - [0.0, 0.0, T.FLOOR]]   # noqa: E731
        return {n: mm(st[n].pos) for n in st}

    def cord() -> dict:
        st = states(T.LINKS)
        z = [float(s.pos[2]) - T.FLOOR for s in st.values()]
        v = [float(np.linalg.norm(s.lin_vel)) for s in st.values()]
        gaps = [float(np.linalg.norm(np.asarray(st[b].pos) - T.K.place(st[a].pos, st[a].quat_xyzw, (T.K.PITCH, 0.0, 0.0))))
                for a, b in zip(T.LINKS, T.LINKS[1:])]
        return {"z_mm": [round(1000 * min(z), 1), round(1000 * max(z), 1)], "v_max": round(max(v), 3),
                "v_at": int(np.argmax(v)), "gap_mm": round(1000 * max(gaps), 2), "gap_at": int(np.argmax(gaps)),
                "v_mm_s": [round(1000 * x) for x in v]}

    snap_frames(obs)
    measure()
    start = {"task": T.report(states()), "places": places(states()), "cord": cord()}
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
                          "places": places(st), "cord": cord()})
        if steps % 40 == 0 and trace:
            t = trace[-1]
            print(f"step {steps} score {t['score']} {t['task']} {t['hands']['right']} {t['places']} {t['cord']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended,
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": {"task": T.report(states()), "places": places(states()), "cord": cord()}, "trace": trace}
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
    ap.add_argument("--mode", default="full", choices=("full", "pinch", "tray", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
