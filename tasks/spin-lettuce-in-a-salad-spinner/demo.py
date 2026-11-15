"""Demo for the salad-spinner task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1200] [--no-record] [--mode spin|nohold|slow|lift|still]

1. The left hand comes straight down on the bowl's side handle and holds it between its fingers.
2. The right hand, leaning out from its arm's base, comes down on the crank's knob and takes it between its fingers.
3. It turns the crank clockwise thirteen times at about one and a quarter turns a second, leaning lightly on the knob so
   that the lid stays down on the bowl. The knob carries the hand round the crank's real circle, so each control step
   the hand works that circle out from where it has just been and aims a little further round it. The knob turns in
   its fingers; the basket spins up past 5 turns a second and stays there. If its fingers close on nothing (the knob
   has slipped out), it opens them and takes the knob again where it last held it.
4. It lets go of the knob and rises clear; the left hand lets go of the handle.

Controls: ``--mode nohold`` does the same cranking with the left hand kept away, so nothing holds the bowl;
``--mode slow`` cranks four turns at about 0.4 turns a second (the basket never gets up to drying speed);
``--mode lift`` picks the lid up off the bowl by the knob, puts it back and then cranks (the lid came off);
``--mode still`` keeps both arms still (a scene check).

Every target comes from the layout constants in the task file and the spinner's geometry in ``spinnerkit.py``. The
only feedback is each hand's own measured position and finger opening (the robot's proprioception): to aim each
first grasp, to steer the cranking, and to notice a slipped knob. The demo never reads the spinner's state to act.
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
TIP = 0.013                        # the pads' tips reach this far past the tool point



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

    def follow(self, path, duration: float, rot=None):
        """Generator: follow ``path(s)`` (world tool point for s from 0 to 1) at a smooth pace."""
        from yam_arena import ik

        r = self.rot if rot is None else rot
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self._solve(path(ik.smoothstep(k / n)), r)
            yield

    def joints_for(self, pos, rot, seed=None) -> np.ndarray:
        from yam_arena import ik

        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), np.array(self.q if seed is None else seed, float),
                             base_pos=self.base, iters=400)
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, float(np.degrees(re)))
        return q

    def set_grip(self, grip: float, duration: float = 0.5):
        yield from self.move(self.pos, duration, grip=grip)

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
    for q6 in (None, -1.57, 1.57):
        seed = hand.q.copy()
        if q6 is not None:
            seed[5] = q6
        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=hand.base, iters=400)
        if best is None or pe < best[1]:
            best = (q, pe)
    return best[0]



TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
WIDTH: dict = {}                         # each hand's measured finger opening, m (updated every control step)
DOWN_X = frame((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))     # straight down, fingers closing along x
DOWN_Y = frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))     # straight down, fingers closing along y
HANDLE_OPEN = 0.035                      # finger gap round the 5 mm handle
HANDLE_SHUT = 0.002                      # ... and commanded on it: firm, without twisting the bowl round in the fingers
KNOB_OPEN, GRIP = 0.045, 0.012           # finger gap round the 22 mm knob / commanded on it
HOLDING = 0.015                          # m: the fingers measured wider than this have the knob between them
KNOB_UP = 0.019                          # the tool point this far above the crank arm's top, on the knob's axis: the
                                         # knob's top is then between the fingers' wide part, and the leaning
                                         # fingertips, 12 mm lower, clear the arm by 7 mm
LEAN = math.radians(25.0)                # the cranking hand leans this far out from its arm's base
LEAD = 0.8                               # rad: the hand aims this far round ahead of where it is
SLOW_LEAD = 0.2                          # rad: the same for the slow control (about 0.4 turns a second)
RAMP_S = 2.5                             # s over which the lead eases in
TURNS, SLOW_TURNS = 13.0, 4.0            # crank turns given
PRESS = 0.006                            # m: while cranking the hand is sent this far below where it took the knob. It
                                         # cannot slide down the knob, so it leans on it and keeps the lid down
FIT_STEPS = 20                           # the hand's last second of positions, for the circle it is going round
FIT_GAIN = 0.3                           # how fast the circle's centre and radius follow each new fit
ARC_MIN = math.radians(50.0)             # a fit is used only if those positions span this much of the circle
R_OK = (0.040, 0.070)                    # ... and its radius is believable for a 5.5 cm crank
LIFT = 0.030                             # m: how far the lift control picks the lid up by the knob
AIM_MAX = 0.012                          # the most the first grasp of each hand is aimed off for its lag


def aim(hand, target):
    """Correct ``target`` for the hand's measured lag across the table, once, by at most AIM_MAX (never lower: the
    fingertips must stay clear of what is below)."""
    target = np.asarray(target, float)
    off = target - TOOL[hand.name]
    off[2] = 0.0
    if np.linalg.norm(off) > AIM_MAX:
        off = off * AIM_MAX / np.linalg.norm(off)
    return target + off


def handle_point(T) -> np.ndarray:
    """Where the left hand takes the handle: pad tips 4 mm above the handle's lower edge, 1.5 cm in from its end."""
    import spinnerkit as K

    c = T.spinner_start() + [0.0, K.BOWL_R + K.TAB[1] - 0.017, K.TAB_Z]
    return c + [0.0, 0.0, -K.TAB[2] / 2 + 0.004 + TIP]


def hold(T, left):
    at = handle_point(T)
    yield from left.move_joints(best_ik(left, at + [0.0, 0.0, 0.06], DOWN_X), 2.0, grip=gap(HANDLE_OPEN))
    yield from chain(left.move(at, 1.0), wait(0.3))
    yield from chain(left.move(aim(left, at), 0.4), wait(0.2))
    yield from chain(left.set_grip(gap(HANDLE_SHUT), 0.5), wait(0.3))


def release_hold(left):
    yield from left.set_grip(gap(HANDLE_OPEN), 0.4)
    yield from left.move(left.pos + [0.0, 0.0, 0.06], 0.8)


def fit_circle(points):
    """Centre and radius of the circle through ``points`` (x, y) by least squares, or None if they do not span one."""
    pts = np.asarray(points, float)
    if len(pts) < 6:
        return None
    a = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    sol, *_ = np.linalg.lstsq(a, -(pts ** 2).sum(axis=1), rcond=None)
    centre = -sol[:2] / 2
    r2 = float(centre @ centre - sol[2])
    if r2 <= 0:
        return None
    ang = np.unwrap(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    if abs(ang[-1] - ang[0]) < ARC_MIN:
        return None
    return centre, math.sqrt(r2)


def crank(T, right, turns: float = TURNS, lead_max: float = LEAD, lift: bool = False):
    """Take the knob and turn the crank clockwise by following the hand's own measured position. The knob carries the
    hand round the crank's real circle, wherever the bowl has ended up, so the hand works that circle out from where
    it has just been (a circle fitted to its last second of measured positions) and every control step aims
    ``lead_max`` further round it. So it pulls the knob round without pushing it sideways, however fast it goes, and a
    slip in the fingers cannot add up. It stops once it has gone ``turns`` times round. ``lift`` (a control): first
    pick the lid up by the knob and put it back."""
    import spinnerkit as K

    s0, q0 = T.spinner_start(), (0.0, 0.0, 0.0, 1.0)
    at = K.knob_point(s0, q0, 0.0, up=KNOB_UP)
    # The hand leans out from its arm's base toward the spinner, fingers closing across the lean: its two finger faces
    # stay upright either side of the knob, and every joint keeps room all the way round the crank's circle (held
    # straight down, the far side of the circle is past where this arm can keep its hand upright).
    out = np.array([s0[0] - right.base[0], s0[1] - right.base[1], 0.0])
    out /= np.linalg.norm(out)
    hand = frame(-math.cos(LEAN) * np.array([0.0, 0.0, 1.0]) + math.sin(LEAN) * out, (out[1], -out[0], 0.0))
    yield from right.move_joints(best_ik(right, at + [0.0, 0.0, 0.06], hand), 2.5, grip=gap(KNOB_OPEN))
    yield from chain(right.move(at, 1.0), wait(0.3))
    yield from chain(right.move(aim(right, at), 0.4), wait(0.2))
    yield from chain(right.set_grip(gap(GRIP), 0.5), wait(0.3))
    if lift:
        home = right.pos.copy()
        yield from chain(right.move(home + [0.0, 0.0, LIFT], 1.0), wait(0.5), right.move(home, 1.0), wait(0.3))
    height = right.pos[2]
    centre, radius = np.array(s0[:2], float), K.KNOB_AT            # where the crank's circle is, to begin with
    path = [TOOL["right"][:2].copy()]

    def angle():
        d = TOOL["right"][:2] - centre
        return math.atan2(d[1], d[0])

    last = angle()
    gone = 0.0
    lead = 0.0
    press = 0.0
    held_at = TOOL["right"].copy()                                           # where the knob last was in the fingers
    while gone < turns:
        if WIDTH["right"] < HOLDING:                                         # the knob has slipped out: take it again
            yield from right.set_grip(gap(KNOB_OPEN), 0.3)
            yield from right.move(right.pos + [0.0, 0.0, 0.02], 0.4)
            yield from right.move(np.array([held_at[0], held_at[1], height + 0.02]), 0.6)
            yield from chain(right.move(np.array([held_at[0], held_at[1], height]), 0.5), wait(0.2))
            yield from chain(right.set_grip(gap(GRIP), 0.5), wait(0.3))
            last = angle()
            lead = 0.0
            path = [TOOL["right"][:2].copy()]
            continue
        now = angle()
        gone += K.FORWARD * math.remainder(now - last, 2 * math.pi) / (2 * math.pi)
        path = (path + [TOOL["right"][:2].copy()])[-FIT_STEPS:]
        fit = fit_circle(path)
        if fit is not None and R_OK[0] <= fit[1] <= R_OK[1]:                 # the circle the knob is carrying the hand round
            centre = centre + FIT_GAIN * (fit[0] - centre)
            radius = radius + FIT_GAIN * (fit[1] - radius)
            now = angle()
        last = now
        held_at = TOOL["right"].copy()
        lead = min(lead_max, lead + lead_max / (RAMP_S * CONTROL_HZ))        # ease in
        a = now + K.FORWARD * lead
        # lean on the knob: a hand left at the height it took the knob creeps up as it speeds up, and lifts the lid by it
        press = min(PRESS, press + PRESS / (1.0 * CONTROL_HZ))
        target = np.array([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), height - press])
        right._solve(target, right.rot)
        yield
    yield from chain(right.move(right.pos, 0.3), wait(0.2))
    yield from right.set_grip(gap(KNOB_OPEN), 0.4)
    yield from right.move(right.pos + [0.0, 0.0, 0.06], 0.8)


def script(T, left, right, mode: str = "spin"):
    yield from wait(0.3)
    if mode == "still":
        yield from wait(8.0)
        return
    if mode != "nohold":
        yield from hold(T, left)
    if mode == "slow":
        yield from crank(T, right, turns=SLOW_TURNS, lead_max=SLOW_LEAD)
    else:
        yield from crank(T, right, lift=mode == "lift")
    if mode != "nohold":
        yield from release_hold(left)
    yield from wait(2.0)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    with imageio.get_writer(path, fps=CONTROL_HZ, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, window: bool = False,
        hold_s: float = 2.5, mode: str = "spin", every: int = 5) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    import spinnerkit as K
    from yam_arena import station
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=320, cam_height=176,
                               headless=not window)
    T = emb.task.module
    obs = emb.reset(Scene(id="spinner-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    gen = script(T, left, right, mode)
    frames: dict[str, list] = {}
    trace = []

    def snap(o) -> None:
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, [T.NAME, T.LID])


    def hands() -> dict:
        from spinnerkit import _np, finger_points

        out = {}
        for n, p in zip(("left", "right"), finger_points(emb._env)):
            robot = emb._env.unwrapped.scene[f"{n}_robot"]
            q, names = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            out[n] = {"tool_cm": [round(100 * float(v), 2) for v in p], "q6": round(float(q[names.index("joint6")]), 3),
                      "gap_mm": round(1000 * sum(abs(float(q[names.index(j)])) for j in ("joint7", "joint8")), 1)}
        return out

    snap(obs)
    start = T.report(states())
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(left.cmd, right.cmd)))
        steps += 1
        from spinnerkit import _np, finger_points
        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        for hn in ("left", "right"):
            robot = emb._env.unwrapped.scene[f"{hn}_robot"]
            jq, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
            WIDTH[hn] = sum(abs(float(jq[jn.index(j)])) for j in ("joint7", "joint8"))
        info = res.info
        snap(res.observation)
        if steps % every == 0:
            r = T.report(states())
            lid = states()[T.LID]
            knob = K.knob_point(lid.pos, lid.quat_xyzw, lid.joints["crank"], up=KNOB_UP)
            trace.append({"step": steps, "score": info.get("score"), "task": r, "hands": hands(), "want_q6": round(float(right.q[5]), 3),
                          "lid_cm": [round(100 * float(v), 2) for v in lid.pos], "lid_yaw_deg": round(math.degrees(T.yaw(lid.quat_xyzw)), 1),
                          "crank_deg": round(math.degrees(lid.joints["crank"]), 1), "knob_cm": [round(100 * float(v), 2) for v in knob],
                          "want_cm": {h.name: [round(100 * float(v), 2) for v in h.pos] for h in (left, right)}})
        if steps % (10 if mode.endswith("only") else 40) == 0 and trace:
            t = trace[-1]
            print(f"step {steps} score {t['score']} {t['task']} {t['hands']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended,
              "worst_ik": {h.name: {"mm": round(h.worst_mm, 2), "deg": round(h.worst_deg, 2)} for h in (left, right)},
              "start": start, "end": T.report(states()), "trace": trace}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
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
    ap.add_argument("--mode", default="spin", choices=("spin", "nohold", "slow", "lift", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every)
