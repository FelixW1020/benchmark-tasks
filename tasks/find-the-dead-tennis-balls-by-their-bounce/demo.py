"""Demo for the dead-tennis-ball task, run in the yam_arena embodiment.

    python demo.py task.py out_dir [--window] [--max-steps 1800] [--no-record] [--mode full|guess|half|still]

For each ball in turn, nearest first:

1. The right hand takes the ball out of the tray, carries it to a clear spot on the table to the tray's left and holds
   it there with its bottom 35 cm above the table. The hand leans back toward the robot, which is what lets the arm reach that high.
2. The left hand waits round the line the ball will fall along, level, its fingers open and its pads at the height
   the middle of a good ball comes back to (22.7 cm), so that the falling ball passes between them.
3. The right hand lets go. The ball falls through the left hand's fingers, bounces on the table, and the left hand
   closes at the moment a good ball is at the top of its bounce.
4. The left hand's own fingers say what happened: stopped 6.6 cm apart, they hold a ball, and it is a good one; closed
   to 5 cm, the ball did not come back that high, and it is flat.
5. A good ball is carried to the blue box by the left hand, which already holds it. A flat one is left to settle
   where it fell; the left hand draws back, and the right hand picks the ball up and drops it in the grey bin.

Once two good balls have been found the rest are flat and go straight to the bin. Once two flat ones have been found
the rest are good; they still go to the box by the same drop and catch, which is also how a ball gets from the right
hand to the left.

Controls: ``--mode guess`` bins the two nearest balls without testing any. ``--mode half`` tests and sorts the first
two balls and stops. ``--mode still`` keeps both arms still.

Every target comes from the layout constants in the task file and the sizes in ``ballkit.py``. The only feedback is
the robot's own: each hand's measured position (to correct for the arms' sag) and the left hand's measured finger
opening (to tell whether it caught the ball). The demo never reads a ball's state.
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
G = 9.81
FINGER_SPEED = 0.146               # m/s: the fastest a finger moves (yam_arena.constants.GRIPPER)


def frame(approach, closing) -> np.ndarray:
    """Tool rotation whose approach axis (z) and finger-closing axis (x) point along these world directions."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.asarray(closing, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def leaning(heading_deg: float, lean_deg: float) -> np.ndarray:
    """A hand leaning ``lean_deg`` from straight down, its fingers pointing toward ``heading_deg`` (0: away from the
    robot, 90: to its left) and down, its closing axis level."""
    h, a = math.radians(heading_deg), math.radians(lean_deg)
    return frame((math.sin(a) * math.cos(h), math.sin(a) * math.sin(h), -math.cos(a)), (-math.sin(h), math.cos(h), 0.0))


HALF_TURN = np.diag([-1.0, -1.0, 1.0])       # the same hold with the hand turned half a turn about its approach axis


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

    def snap_grip(self, grip: float) -> None:
        """Send the fingers to ``grip`` at once (they then move as fast as they can)."""
        self.grip = grip
        self.cmd = np.r_[self.q, self.grip]

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
    """Joint angles for ``pos``/``rot`` from the best of a few seeds: the one that leaves the joints most room.
    Returns (q, room in rad, reached)."""
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
            best = (q, cost, room)
    return best[0], best[2], not best[1][0]


def way_round(hand, pos, rot) -> np.ndarray:
    """``rot`` or the same hold turned half a turn about the approach axis (a parallel gripper holds the same either
    way): whichever the arm reaches with more room at its joints."""
    a, b = best_ik(hand, pos, rot), best_ik(hand, pos, rot @ HALF_TURN)
    return rot if (not a[2], -a[1]) <= (not b[2], -b[1]) else rot @ HALF_TURN


TOOL: dict = {}                          # where each hand's tool point actually is (updated every control step)
GAPS: dict = {}                          # how far apart each hand's pads actually are (updated every control step)

UP = np.array([0.0, 0.0, 1.0])
DROP = 0.350                             # the ball's bottom is this far above the table when it is let go
SPOT = (-0.010, 0.180)                   # the drop spot, from the middle of the tray: on the bare table to its left
R_LEAN, R_HEAD = 66.0, 0.0               # how the right hand holds the ball for the drop: leaning back toward the robot
RING_HEAD = -90.0                        # the left hand waits round the fall line pointing to the robot's right, level
RING_OPEN = 0.088                        # pads apart while the ball falls through (the ball is 66 mm across)
CATCH = 0.050                            # the left hand's pads are sent this far apart to catch
HELD = 0.058                             # more than this between the pads afterwards: they hold a ball
TAKE_OPEN, TAKE_GRIP = 0.090, 0.060      # pads apart to take a ball, and closed on it
LIFT = 0.090                             # a ball is lifted this far out of the tray before it is carried
LET_GO_LAG = 0.020                       # s from the command to open until the ball is falling (an estimate)
OVER_BIN = 0.035                         # a flat ball is let go with its bottom this far above the bin's rim
BOX_LEAN = 30.0                          # the left hand leans this far from straight down to reach into the box
OVER_FLOOR = 0.045                       # a good ball is let go with its bottom this far above the box's floor
PARK = 0.150                             # ... and the left hand waits this far above that, over the box
APART = 0.037                            # two balls go into the bin or the box this far either side of its middle
GAIN, BIAS_MAX = 0.25, 0.025             # how fast, and how far, a hand corrects for its own sag


class Sag:
    """Slowly offsets the hand's command by its measured error, so that its tool point ends up where it is wanted."""

    def __init__(self, hand):
        self.hand, self.bias, self.want, self.frozen = hand, np.zeros(3), None, False

    def to(self, pos, rot):
        if self.want is not None and not self.frozen and self.hand.name in TOOL:
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


def swing(hand, sag, pos, rot, seconds: float, grip: float | None = None):
    """Generator: swing the arm joint by joint to ``pos``/``rot`` (for long moves through free space), then let the
    sag correction take over from there."""
    q, _, _ = best_ik(hand, pos, rot)
    yield from hand.move_joints(q, seconds, grip=grip)
    sag.want = None
    sag.to(pos, rot)
    yield


def apex_time(drop: float, bounce: float) -> float:
    """Seconds from the moment a ball starts to fall, its bottom ``drop`` above the table, to the top of its first bounce."""
    return (1.0 + bounce) * math.sqrt(2.0 * drop / G)


def plan(T, drop: float = DROP) -> dict:
    """The demo's fixed targets, from the task file's layout (world, m)."""
    K = T.K
    spot = np.array([T.TRAY_XY[0] + SPOT[0], T.TRAY_XY[1] + SPOT[1], T.TOP])
    slots = [T.ball_start(k) for k in range(4)]
    ring = spot + rebound_mid(K, drop) * UP
    rim = T.TOP + K.BIN_FLOOR + K.BIN_RIM
    over_bin = [np.array([T.BIN_XY[0], T.BIN_XY[1] + side * APART, rim + OVER_BIN + K.BALL_R]) for side in (1.0, -1.0)]
    in_box = [np.array([T.BOX_XY[0], T.BOX_XY[1] + side * APART, T.TOP + K.BIN_FLOOR + OVER_FLOOR + K.BALL_R]) for side in (-1.0, 1.0)]
    # close so that the pads meet the ball as a good one reaches the top of its bounce
    close_for = (RING_OPEN - K.BALL_D) / 2 / FINGER_SPEED + 0.020
    lead = max(1, round((LET_GO_LAG + apex_time(drop, K.BOUNCE_GOOD) - close_for) * CONTROL_HZ))
    return {"spot": spot, "slots": slots, "hold": spot + (drop + K.BALL_R) * UP, "ring": ring, "lead": lead,
            "over_bin": over_bin, "in_box": in_box, "park": in_box[0] + PARK * UP,
            "settle": 2.6 * apex_time(drop, K.BOUNCE_DEAD) + 0.6}


def rebound_mid(K, drop: float) -> float:
    """Height above the table of the middle of a good ball at the top of its first bounce."""
    return K.rebound(drop, K.BOUNCE_GOOD) + K.BALL_R


def nearer(rot, ref) -> np.ndarray:
    """``rot`` or the same hold turned half a turn about the approach axis: whichever is the smaller turn from ``ref``."""
    return rot if np.trace(rot.T @ ref) >= np.trace((rot @ HALF_TURN).T @ ref) else rot @ HALF_TURN


def script(T, left, right, mode: str = "full", idle_s: float = 60.0, notes: dict | None = None):
    """The demo as a generator: one yield per control step. ``notes`` collects what the demo found out."""
    notes = notes if notes is not None else {}
    notes.update(tested=[], verdict={}, binned=[], boxed=[], caught_gap_mm={}, phase="start")

    def mark(what: str) -> None:
        notes["phase"] = what

    if mode == "still":
        yield from wait(idle_s)
        return
    K = T.K
    p = plan(T)
    lsag, rsag = Sag(left), Sag(right)
    take_rot = way_round(right, p["slots"][1], frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))     # straight down, closing across the tray
    hold_rot = nearer(leaning(R_HEAD, R_LEAN), take_rot)
    bin_rot = nearer(leaning(0.0, 30.0), take_rot)
    ring_rot = way_round(left, p["ring"], leaning(RING_HEAD, 90.0))
    box_rot = nearer(leaning(RING_HEAD, BOX_LEAN), ring_rot)
    flat, good = [], []

    def take(k: int):
        """Right hand: take ball k out of the tray and lift it."""
        mark(f"take {T.BALLS[k]}")
        at = p["slots"][k]
        yield from swing(right, rsag, at + LIFT * UP, take_rot, 1.6, grip=gap(TAKE_OPEN))
        yield from glide(rsag, at + LIFT * UP, 0.3, take_rot)
        yield from glide(rsag, at, 1.2, take_rot)
        yield from glide(rsag, at, 0.3, take_rot)
        yield from right.set_grip(gap(TAKE_GRIP), 0.4)
        yield from wait(0.2)
        yield from glide(rsag, at + LIFT * UP, 0.8, take_rot)

    def to_hold():
        """Right hand: carry the ball to the drop spot and hold it up, leaning back."""
        mark("carry to the drop spot")
        start = rsag.want.copy()
        yield from glide(rsag, (p["hold"][0], p["hold"][1], start[2] + 0.060), 1.4, take_rot)
        yield from glide(rsag, p["hold"], 2.0, hold_rot, rot_from=take_rot)
        yield from glide(rsag, p["hold"], 0.6, hold_rot)

    def ring_in():
        """Left hand: from over the box to round the fall line, level, fingers open."""
        yield from wait(1.4)                          # the right hand goes first: the two would cross on the way
        yield from glide(lsag, p["ring"], 2.0, ring_rot, rot_from=box_rot)
        yield from glide(lsag, p["ring"], 0.6, ring_rot)

    def to_park(seconds: float = 1.8):
        yield from glide(lsag, p["park"], seconds, box_rot, rot_from=ring_rot)

    def to_box(n: int):
        """Left hand: carry the ball it caught to the box, lower it in and let go; then wait over the box."""
        mark("to the box")
        at = p["in_box"][n]
        yield from glide(lsag, at + PARK * UP, 2.2, box_rot, rot_from=ring_rot)
        yield from glide(lsag, at, 1.2, box_rot)
        yield from glide(lsag, at, 0.3, box_rot)
        yield from left.set_grip(gap(RING_OPEN), 0.4)
        yield from wait(0.3)
        yield from glide(lsag, at + PARK * UP, 0.9, box_rot)
        yield from glide(lsag, p["park"], 0.5, box_rot)

    def from_table():
        """Right hand: pick up the ball that settled at the drop spot."""
        mark("pick the ball up off the table")
        at = p["spot"] + K.BALL_R * UP
        yield from right.set_grip(gap(TAKE_OPEN), 0.3)
        yield from glide(rsag, at + 0.120 * UP, 2.0, take_rot, rot_from=hold_rot)
        yield from glide(rsag, at, 1.2, take_rot)
        yield from glide(rsag, at, 0.3, take_rot)
        yield from right.set_grip(gap(TAKE_GRIP), 0.4)
        yield from wait(0.2)
        yield from glide(rsag, at + 0.150 * UP, 1.0, take_rot)

    def to_bin(n: int):
        """Right hand: carry the ball over the bin and let it go."""
        mark("to the bin")
        over = p["over_bin"][n]
        start = rsag.want.copy()
        yield from glide(rsag, (start[0], start[1], max(start[2], over[2])), 0.5, take_rot)
        yield from glide(rsag, over, 2.0, bin_rot, rot_from=take_rot)
        yield from glide(rsag, over, 0.3, bin_rot)
        yield from right.set_grip(gap(TAKE_OPEN), 0.3)
        yield from wait(0.5)
        yield from glide(rsag, over + 0.060 * UP, 0.6, bin_rot)

    def left_to_park():
        yield from swing(left, lsag, p["park"], box_rot, 2.2, grip=gap(RING_OPEN))

    if mode == "guess":                              # control: the two nearest balls go to the bin untested, the others to the box
        yield from left_to_park()
        for n, k in enumerate((0, 1)):
            yield from take(k)
            yield from to_bin(n)
            notes["binned"].append(T.BALLS[k])
        mark("hands away")
        yield from wait(3.0)
        return

    first = True
    for k in range(2 if mode == "half" else 4):
        name = T.BALLS[k]
        if len(good) == 2:                           # both good balls found: the rest are flat and go straight to the bin
            yield from take(k)
            yield from to_bin(len(flat))
            flat.append(k)
            notes["binned"].append(name)
            notes["verdict"][name] = "flat (the other two are good)"
            continue
        # (once both flat balls are found the rest are good; they still travel to the box by the same drop and catch)
        if first:
            yield from together(left_to_park(), take(k))
            first = False
        else:
            yield from take(k)
        yield from together(to_hold(), ring_in())
        # let go, and close the left hand when a good ball would be at the top of its bounce
        mark(f"drop {name}")
        right.snap_grip(gap(TAKE_OPEN))
        yield from wait(p["lead"] / CONTROL_HZ)
        left.snap_grip(gap(CATCH))
        yield from wait(0.5)
        caught = GAPS.get("left", 0.0) > HELD
        notes["tested"].append(name)
        notes["caught_gap_mm"][name] = round(1000 * GAPS.get("left", 0.0), 1)
        if caught:
            notes["verdict"][name] = "good"
            yield from to_box(len(good))
            good.append(k)
            notes["boxed"].append(name)
        else:
            notes["verdict"][name] = "flat"
            mark("wait for the ball to settle")
            yield from together(to_park(), wait(p["settle"]))
            yield from left.set_grip(gap(RING_OPEN), 0.3)
            yield from from_table()
            yield from to_bin(len(flat))
            flat.append(k)
            notes["binned"].append(name)

    # both hands clear of everything
    mark("hands away")
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
    obs = emb.reset(Scene(id="ball-demo", instruction=T.INSTRUCTION))
    left, right = Hand("left", station.BASES["left"]), Hand("right", station.BASES["right"])
    notes: dict = {}
    gen = script(T, left, right, mode, idle_s=max_steps / CONTROL_HZ, notes=notes)
    frames: dict[str, list] = {}
    trace = []

    def snap_frames(o) -> None:
        if record:
            for name, img in o.images.items():
                frames.setdefault(name, []).append(np.asarray(img))

    def states():
        from yam_arena.scene import object_states   # importable only once the simulator has started

        return object_states(emb._env, list(T.BALLS))

    def measure() -> None:
        from ballkit import finger_gaps, finger_points

        TOOL["left"], TOOL["right"] = (np.asarray(p, float) for p in finger_points(emb._env))
        GAPS["left"], GAPS["right"] = finger_gaps(emb._env)

    def hands() -> dict:
        return {h.name: {"tool_mm": [round(1000 * float(v), 1) for v in TOOL[h.name] - [0.0, 0.0, T.TOP]],
                         "want_mm": [round(1000 * float(v), 1) for v in h.pos - [0.0, 0.0, T.TOP]],
                         "gap_mm": round(1000 * GAPS[h.name], 1)} for h in (left, right)}

    def places(st) -> dict:
        return {n: [round(1000 * float(x), 1) for x in np.asarray(st[n].pos, float) - [0.0, 0.0, T.TOP]] for n in T.BALLS}

    snap_frames(obs)
    measure()
    start = {"task": T.report(states()), "places": places(states())}
    steps, ended, success, info = 0, None, False, {}
    high = {n: 0.0 for n in T.BALLS}
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
            print(f"step {steps} score {t['score']} where {t['task']['where']} L {t['hands']['left']} R {t['hands']['right']} "
                  f"{t['places']} notes {notes}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"lot": T.LOT, "mode": mode, "task_score": 1.0 if success else float(info.get("score", 0.0) or 0.0),
              "success": success, "steps": steps, "script_steps": ended, "notes": notes,
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
    ap.add_argument("--mode", default="full", choices=("full", "guess", "half", "still"))
    ap.add_argument("--trace-every", type=int, default=5, help="steps between entries of the trace in result.json")
    ap.add_argument("--cam", default="320x176", help="recorded frame size, e.g. 640x352")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.task).resolve().parent))
    run(a.task, a.out, a.max_steps, record=not a.no_record, window=a.window, mode=a.mode, every=a.trace_every,
        cam=tuple(int(v) for v in a.cam.split("x")))
