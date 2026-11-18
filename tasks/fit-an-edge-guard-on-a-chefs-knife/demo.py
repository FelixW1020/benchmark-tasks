"""Demo for the knife edge-guard task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1500] [--no-record] [--mode full|half|shallow|blade|still]

1. The right hand takes the guard, which leans against the board's front edge, by its folded edge, one finger on
   each face, lifts it, and leans forward until the guard lies flat in its fingers, open side forward, in the plane
   the blade is about to be brought into, turned 1.5 degrees so that its heel end leads.
2. The left hand comes straight down over the knife's handle, 4 cm behind the heel, closes on the handle's back
   and belly, lifts the knife 6 cm off the board and tips it 5 degrees (edge up), blade level along its length:
   the edge is now 1.5 cm from the guard's open side.
3. It pushes the guard onto the edge, heel end first, straightening it on the way, until the edge is at the bottom
   of the slot (half a millimetre short of it), slides it 4.5 mm along the blade up to the handle, lets go and draws back.
4. The left hand lays the knife, guard on, back on the board and lets go.

Controls: ``--mode half`` stops with the guard part of the way on; ``--mode shallow`` stops with the edge 7 mm short
of the slot's bottom; ``--mode blade`` takes the knife by its blade, a finger on the edge; ``--mode still`` keeps
both arms still.

Every target comes from the layout constants in the task file and the sizes in ``guardkit.py``. The only feedback is
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


def lean_rot(lean: float, sign: float = 1.0) -> np.ndarray:
    """A hand leaning ``lean`` rad forward from straight down, its fingers closing in the x-z plane (``sign`` picks
    which finger is in front)."""
    return frame((math.sin(lean), 0.0, -math.cos(lean)), (sign * math.cos(lean), 0.0, sign * math.sin(lean)))


def gap(width: float) -> float:
    """Grip command (1 open .. 0 closed) for a gap between the pads in metres."""
    return min(1.0, max(0.0, width / FULL_OPEN))


def pose(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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

PAD_TIP = 0.013                          # the pads' ends are this far past the tool point
Y_GRIP = 0.040                           # the left hand takes the handle this far behind the heel
PICK_ABOVE = 0.0135                      # ... with its tool point this far above the board (its pads' ends on the board)
HANDLE_GAP = 0.022                       # pad gap commanded on the 30 mm handle (about 40 N on each pad)
Z_K = 0.100                              # the knife's handle is carried with its middle this far above the table
TILT = math.radians(5.0)                 # ... and the knife tipped this much, edge up
LEAN_FLAT = math.pi / 2 - TILT           # the right hand's lean when the guard lies flat in it, in the blade's plane
S_GRIP = 0.070                           # the right hand takes the guard this far from its heel end
U_TCP = -0.0045                          # ... its tool point this far from the folded edge (outside it: the pads' ends
                                         #     then stand just clear of the board's top)
GUARD_GAP = 0.002                        # pad gap commanded on the 5.9 mm guard (about 20 N on each pad)
S_CLEAR = 0.005                          # the guard goes on with its heel end this far in front of the knife's heel,
S_END = 0.0005                           # ... and is then slid along the blade until it is this far from it
SEAT_GAP = 0.0005                        # the slot's bottom goes this far below the edge
LEAD = 0.015                             # before it goes on, the guard's open side is this far short of the edge
YAW = math.radians(-1.5)                 # ... and the guard is turned this much, its heel end leading
PITCH = math.radians(0.9)                # the guard's far end is held this much low: the knife's tip hangs half a
                                         #     degree in the left hand's grip, and the guard sits askew in the right's
HALF_ON = 0.012                          # --mode half: the guard's open side stops this far past the heel's edge
SHALLOW = 0.007                          # --mode shallow: the slot's bottom stops this far below the edge
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
    board_z = T.TOP + T.BOARD_TOP
    kp, kq = T.knife_start()
    gp, gq = T.guard_start()
    rk, rg = K._rotation(kq), K._rotation(gq)
    # the left hand on the handle
    l_rot = lean_rot(0.0, -1.0)
    handle = kp + rk @ np.array([-Y_GRIP, K.BH - K.HANDLE_H / 2, 0.0])
    l_pick = np.array([handle[0], handle[1], board_z + PICK_ABOVE])
    knife_in_left = np.linalg.inv(pose(l_pick, l_rot)) @ pose(kp, rk)
    l_carry_rot = lean_rot(-TILT, -1.0)
    l_carry = np.array([l_pick[0], l_pick[1], T.TOP + Z_K + (l_pick[2] - handle[2])])
    knife_carried = pose(l_carry, l_carry_rot) @ knife_in_left
    # the right hand on the guard's folded edge
    r_rot = lean_rot(-T.GUARD_LEAN, 1.0)
    r_pick = gp + rg @ np.array([S_GRIP, U_TCP, 0.0])
    guard_in_right = np.linalg.inv(pose(r_pick, r_rot)) @ pose(gp, rg)

    droop = pose((S_GRIP, 0.0, 0.0), rot_y(PITCH)) @ pose((-S_GRIP, 0.0, 0.0), np.eye(3))   # about the right hand's hold

    def right_for(depth: float, yaw: float, along: float = S_CLEAR) -> tuple:
        """The right hand's tool pose for the guard on the carried knife: its folded edge ``depth`` below the
        knife's edge line, turned ``yaw`` about its heel end, which is ``along`` in front of the knife's heel."""
        g = knife_carried @ pose((along, -depth, 0.0), rot_z(yaw)) @ droop @ np.linalg.inv(guard_in_right)
        return g[:3, 3], g[:3, :3]

    return {"l_rot": l_rot, "l_pick": l_pick, "l_carry": l_carry, "l_carry_rot": l_carry_rot, "r_rot": r_rot,
            "r_pick": r_pick, "right_for": right_for, "knife_carried": knife_carried, "handle": handle,
            "knife_in_left": knife_in_left, "guard_in_right": guard_in_right}


def script(T, left, right, mode: str = "full", idle_s: float = 60.0):
    """The demo as a generator: one yield per control step."""
    from yam_arena import ik
    from yam_arena.constants import READY

    K = T.K
    if mode == "still":
        yield from wait(idle_s)
        return
    P = plan(T)
    ls, rs = Sag(left), Sag(right)
    up = np.array([0.0, 0.0, 1.0])
    board_z = T.TOP + T.BOARD_TOP

    if mode == "blade":
        # control: the left hand takes the knife by its blade, one finger on the edge and one on the spine
        kp, kq = T.knife_start()
        mid = kp + K._rotation(kq) @ np.array([0.060, K.BH / 2, 0.0])
        grab = np.array([mid[0], mid[1], board_z + PICK_ABOVE])
        yield from left.move_joints(best_ik(left, grab + 0.07 * up, P["l_rot"]), 2.5, grip=gap(0.075))
        yield from glide(ls, grab + 0.07 * up, 0.5, P["l_rot"])
        yield from glide(ls, grab, 1.5)
        yield from left.set_grip(gap(0.045), 0.8)
        yield from glide(ls, grab + 0.03 * up, 1.0)
        yield from glide(ls, grab + 0.001 * up, 1.0)
        yield from left.set_grip(gap(0.075), 0.8)
        yield from glide(ls, grab + 0.08 * up, 1.2)
        yield from left.move_joints(READY, 2.5)
        yield from wait(2.0)
        return

    # 1. The right hand takes the guard by its folded edge, lifts it, and leans forward until the guard lies flat in
    #    its fingers, where the blade's plane will be, 1.5 cm short of where the edge will be.
    back = P["r_pick"] - 0.06 * P["r_rot"][:, 2]
    yield from right.move_joints(best_ik(right, back, P["r_rot"]), 2.5, grip=gap(0.030))
    yield from glide(rs, back, 0.5, P["r_rot"])
    yield from glide(rs, P["r_pick"], 1.5)
    yield from glide(rs, P["r_pick"], 0.4)
    yield from right.set_grip(gap(GUARD_GAP), 0.8)
    yield from wait(0.3)
    yield from glide(rs, P["r_pick"] + 0.07 * up, 1.5)
    pre_p, pre_r = P["right_for"](K.GW + LEAD, YAW)
    yield from glide(rs, pre_p, 4.0, pre_r)
    yield from glide(rs, pre_p, 0.6)

    # 2. The left hand takes the knife by its handle, lifts it and tips it, edge up, into that plane.
    over = P["l_pick"] + 0.07 * up
    yield from left.move_joints(best_ik(left, over, P["l_rot"]), 2.5, grip=gap(0.060))
    yield from glide(ls, over, 0.5, P["l_rot"])
    yield from glide(ls, P["l_pick"], 1.5)
    yield from glide(ls, P["l_pick"], 0.4)
    yield from left.set_grip(gap(HANDLE_GAP), 0.8)
    yield from wait(0.3)
    yield from glide(ls, P["l_carry"], 2.5, P["l_carry_rot"])
    yield from glide(ls, P["l_carry"], 0.6)
    yield from glide(rs, pre_p, 0.4)

    # 3. Onto the edge, heel end first, straightening on the way; let go; draw back.
    full = K.G_FOLD + SEAT_GAP
    stop = {"half": K.GW - HALF_ON, "shallow": K.G_FOLD + SHALLOW}.get(mode, full)
    n = max(1, round(4.0 * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        depth = K.GW + LEAD + s * (stop - K.GW - LEAD)
        yaw = YAW * max(0.0, 1.0 - s / 0.6)
        rs.to(*P["right_for"](depth, yaw))
        ls.to(P["l_carry"], P["l_carry_rot"])                        # the left hand holds its place against the push
        yield
    in_p, in_r = P["right_for"](stop, 0.0)
    yield from glide(rs, in_p, 0.4, in_r)
    end_p, end_r = P["right_for"](stop, 0.0, S_END)                  # along the blade, up to the handle
    yield from glide(rs, end_p, 1.0, end_r)
    yield from glide(rs, end_p, 0.5, end_r)
    yield from right.set_grip(gap(0.030), 0.8)
    yield from wait(0.3)
    away = end_p - 0.065 * end_r[:, 2]
    yield from glide(rs, away, 1.5)
    yield from glide(rs, away + 0.05 * up, 1.0)
    yield from right.move_joints(READY, 3.0)

    # 4. The left hand lays the knife back on the board and lets go.
    yield from glide(ls, P["l_pick"] + 0.02 * up, 2.5, P["l_rot"])
    yield from glide(ls, P["l_pick"] + 0.001 * up, 1.0)
    yield from glide(ls, P["l_pick"] + 0.001 * up, 0.4)
    yield from left.set_grip(gap(0.060), 0.8)
    yield from wait(0.3)
    yield from glide(ls, P["l_pick"] + 0.08 * up, 1.5)
    yield from left.move_joints(READY, 2.5)
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
    obs = emb.reset(Scene(id="guard-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ)
    frames: dict[str, list] = {}
    trace = []
    names = [T.KNIFE, T.GUARD]

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, names)

    def measure() -> None:
        from guardkit import finger_points

        (pl, rl), (pr, rr) = finger_points(emb._env)
        TOOL["left"], TOOL["right"] = np.asarray(pl, float), np.asarray(pr, float)
        TOOL["left_rot"], TOOL["right_rot"] = rl, rr

    def hands() -> dict:
        from guardkit import _np

        out_ = {}
        for h in (left, right):
            robot = emb._env.unwrapped.scene[f"{h.name}_robot"]
            q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            rot = TOOL[f"{h.name}_rot"]
            out_[h.name] = {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                            "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                            "gap_mm": round(1000 * sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")), 1),
                            "lean_deg": round(math.degrees(math.atan2(float(rot[0, 2]), float(-rot[2, 2]))), 1)}
        return out_

    def places(st) -> dict:
        """For the record only (the demo does not use it): where the knife and the guard are, and the guard's place
        on the knife."""
        from guardkit import _rotation, to_local

        mm = lambda v: [round(1000 * float(x), 1) for x in np.asarray(v, float)]   # noqa: E731
        k, g = st[T.KNIFE], st[T.GUARD]
        rk, rg = _rotation(k.quat_xyzw), _rotation(g.quat_xyzw)
        rel = rk.T @ rg
        return {"knife": mm(np.asarray(k.pos) - [0.0, 0.0, T.TOP]), "guard": mm(np.asarray(g.pos) - [0.0, 0.0, T.TOP]),
                "knife_tip_drop_mm": round(-1000 * T.K.BL * float(rk[2, 0]), 1),
                "knife_edge_up_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(-rk[2, 1]))))), 1),
                "guard_on_knife_mm": mm(to_local([np.asarray(g.pos, float)], k.pos, k.quat_xyzw)[0]),
                "guard_yaw_deg": round(math.degrees(math.atan2(float(rel[1, 0]), float(rel[0, 0]))), 2),
                "guard_roll_deg": round(math.degrees(math.atan2(float(rel[2, 1]), float(rel[2, 2]))), 2),
                "guard_pitch_deg": round(math.degrees(-math.asin(max(-1.0, min(1.0, float(rel[2, 0]))))), 2),
                "guard_off_plane_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(rel[2, 2]))))), 2)}

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
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--window", action="store_true", help="show the Isaac Sim window on the instance's desktop")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "half", "shallow", "blade", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
