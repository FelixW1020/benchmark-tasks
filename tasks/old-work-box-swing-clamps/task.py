"""Install an old-work electrical box in a drywall cut-out. A wall section stands across the far side of the table,
facing the arms: 1/2 in drywall on 2x4 studs (16 in on centre) between a bottom and a top plate, with a 1-gang
cut-out (2-1/4 x 3-5/8 in, the box's template) in the stud bay, its long side vertical. A 14/2 NM-B cable comes down
inside the stud bay from the top plate, out through the cut-out and into the back of a blue 1-gang old-work box (PVC,
4-1/8 x 2-1/4 x 2-3/4 in, 57 g), which hangs on it in front of the cut-out, its two swing-clamp screws backed out and
its two clamp arms folded inside. On the right: a #2 Phillips stubby screwdriver standing tip down in a wooden block.

Task: put the box in the cut-out with its ears flat on the wall, and drive both clamp screws with the screwdriver
until both clamps grip the drywall from behind; then put the screwdriver down and take both hands off.

Turning a clamp screw clockwise first swings its arm out through the slot in the box's end wall (the thread's
friction carries it round to its stop), then draws it toward the ears until it presses on the back of the drywall
and the screw stalls. All of it is rigid bodies, joints and contact (boxparts.py).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E device=cpu --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import boxparts as bp  # noqa: E402
from boxparts import MM, on_reset, on_step, rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

bp.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("The cable is already in the blue electrical box. Install the box in the hole in the wall: push it in "
               "until its ears are flat on the wall, with the cable going back into the wall behind it, then drive both "
               "of its clamp screws with the screwdriver until the box is held tight. Put the screwdriver down and take "
               "both hands away.")

# -- layout (world: x forward from the arm bases, y to the left, z up; TOP is the table's top) ---------------------------
XW = 0.34                       # the drywall's face
HOLE_Z = 0.19                   # the cut-out's centre above the table
HOLE = np.array([XW, 0.0, TOP + HOLE_Z])
if bp.CABLE:
    # The cable is already fed into the box, and the box hangs on it in front of the cut-out, square to the wall,
    # its ears' back faces START_OUT (78 mm) in front of the wall (its back 11 mm short of the wall's face).
    BOX_START = tuple(float(v) for v in HOLE - np.array([bp.START_OUT, 0.0, 0.0]))
    BOX_START_XYZW = (0.0, 0.0, 0.0, 1.0)
    BOX_ROOT, BOX_ROOT_XYZW = bp.cable_spawn(BOX_START)            # where the articulation is spawned (the cable unloaded)
    CABLE_ANCHOR = bp.cable_anchor_point(BOX_START)                # where the cable leaves the top plate
else:   # the bare box on the table (the clamp bench only)
    BOX_START = (0.20, 0.17, TOP + bp.DEPTH + 0.3 * MM)            # on its back, its long side along x, open face up
    BOX_START_XYZW = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))   # box x (its depth) pointing down
    BOX_ROOT, BOX_ROOT_XYZW = BOX_START, BOX_START_XYZW
BLOCK_XY = (0.17, -0.17)
DRIVER_START = (BLOCK_XY[0], BLOCK_XY[1], TOP + bp.BLOCK[2] - bp.SHAFT_TOP + 0.3 * MM)   # tip down, handle on the block
DRIVER_START_XYZW = (0.0, 0.0, 0.0, 1.0)                     # upright: tip (its origin) down, handle up

OBJECTS = [
    bp.part("wall", "wall", HOLE, fixed=True, hole_z=HOLE_Z),
    bp.part("block", "block", (BLOCK_XY[0], BLOCK_XY[1], TOP), fixed=True),
    bp.part("box", "box", BOX_ROOT, BOX_ROOT_XYZW),
    bp.part("driver", "driver", DRIVER_START, DRIVER_START_XYZW),
]

# -- scoring numbers ------------------------------------------------------------------------------------------------------
FLUSH_GAP = 0.5 * MM            # flush: each ear's back face within 0.5 mm of the wall at both its corners
EAR_POINTS = [(0.0, sy * 26.0 * MM, sz * 50.0 * MM) for sz in (1, -1) for sy in (1, -1)]  # box frame: ears' corners
IN_HOLE = 2.0 * MM              # the box's centre within this of the cut-out's centre (across and up)
SWUNG_DEG = 80.0                # a clamp arm swung out at least this far (90 at its stop)
SNUG_N = 20.0                   # clamped: the arm presses the drywall's back with at least this (see README)
UNDRIVEN_TURNS = 0.25           # a clamp counts only if its screw turned (net) no more than this without the screwdriver in it
CLEAR = 0.03                    # hands off: every point of both grippers at least 3 cm from the box and the screwdriver
DOWN_Z = 0.05                   # put down: the screwdriver's lowest point within 5 cm of the table top ...
STILL = 1.0 * MM                # ... and it and the box moved no more than 1 mm over the last 1 s
HOLD_STEPS = 20                 # 1 s at 20 Hz
ENGAGE_R = 2.5 * MM             # the driver is in a screw: its tip within 2.5 mm of the screw's axis (in its cross) ...
ENGAGE_DEG = 25.0               # ... its shaft within 25 degrees of the axis, its tip in the recess
CABLE_OUT = 15.0 * MM           # the cable counts as still fed in if no more than this much of it has been drawn back out
CREDIT = {"flush": 0.25, "one": 0.5, "both": 0.75}


# -- geometry (pure; used in simulation and by the offline tests) ------------------------------------------------------

def ear_gaps(box_pos, box_r) -> np.ndarray:
    """Gaps (m) between the wall's face and the back faces of the ears at their four corners."""
    pts = np.asarray(box_pos, float) + np.array(EAR_POINTS) @ np.asarray(box_r, float).T
    return XW - pts[:, 0]


def is_flush(box_pos, box_r) -> bool:
    g = ear_gaps(box_pos, box_r)
    c = np.asarray(box_pos, float) - HOLE
    return bool(np.all((g >= -FLUSH_GAP) & (g <= FLUSH_GAP)) and abs(c[1]) <= IN_HOLE and abs(c[2]) <= IN_HOLE)


def driver_points(d_pos, d_r) -> tuple[np.ndarray, np.ndarray]:
    d_r = np.asarray(d_r, float)
    tip = np.asarray(d_pos, float)
    return tip, tip + bp.HANDLE_TOP * d_r[:, 2]


def engaged_screw(box_pos, box_r, d_pos, d_r) -> str | None:
    """Which screw ("t" or "b") the screwdriver's tip is in, if any."""
    box_r, d_r = np.asarray(box_r, float), np.asarray(d_r, float)
    p = box_r.T @ (np.asarray(d_pos, float) - np.asarray(box_pos, float))
    a = box_r.T @ d_r[:, 2]
    if math.degrees(math.acos(max(-1.0, min(1.0, -float(a[0]))))) > ENGAGE_DEG:
        return None
    x_face = -bp.EAR_T - bp.HEAD_H
    if not (x_face - 0.5 * MM <= p[0] <= x_face + bp.RECESS_D + 0.3 * MM):
        return None
    for t, z in (("t", bp.SCREW_Z), ("b", -bp.SCREW_Z)):
        if math.hypot(p[1], p[2] - z) <= ENGAGE_R:
            return t
    return None


def _seg_dist(points, a, b) -> np.ndarray:
    d = b - a
    t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


def clearance(points, box_pos, box_r, d_pos, d_r) -> tuple[float, float]:
    """The least distance (m) from the grippers' points to the box (its outline) and to the screwdriver."""
    if points is None or len(points) == 0:
        return float("inf"), float("inf")
    pts = np.asarray(points, float)
    lo, hi = bp.bounds("box")
    local = (pts - np.asarray(box_pos, float)) @ np.asarray(box_r, float)
    d_box = np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1)
    a, b = driver_points(d_pos, d_r)
    d_drv = _seg_dist(pts, a, b) - bp.HANDLE_AF / 2 / math.cos(math.pi / 8)
    return float(d_box.min()), float(d_drv.min())


def driver_low(d_pos, d_r) -> float:
    """Height (m) of the screwdriver's lowest point above the table: its tip, or the edge of its handle."""
    a = np.asarray(d_r, float)[:, 2]
    tip = np.asarray(d_pos, float)
    lat = bp.HANDLE_AF / 2 / math.cos(math.pi / 8) * math.sqrt(max(0.0, 1.0 - a[2] ** 2))
    return float(min(tip[2], tip[2] + bp.SHAFT_TOP * a[2] - lat, tip[2] + bp.HANDLE_TOP * a[2] - lat) - TOP)


# -- the score, step by step ----------------------------------------------------------------------------------------------

_EP: dict = {}
_DEBUG = bool(os.environ.get("OWB_DEBUG"))


@on_reset
def _reset_state() -> None:
    _EP.clear()
    _EP.update(ever_flush=False, turns={"t": None, "b": None}, driven={"t": 0.0, "b": 0.0},
               undriven={"t": 0.0, "b": 0.0}, recent=[], held=0, score=0.0, steps=0, info={})


_reset_state()


def observe(box_pos, box_r, joints, d_pos, d_r, gripper_points) -> float:
    """One control step: account the screws' turns (with or without the screwdriver in them), then work out the
    score. ``joints``: the box's joint positions by name. Returns the score."""
    box_pos, box_r = np.asarray(box_pos, float), np.asarray(box_r, float)
    d_pos, d_r = np.asarray(d_pos, float), np.asarray(d_r, float)
    mech = bp.mechanism(joints)
    eng = engaged_screw(box_pos, box_r, d_pos, d_r)
    for t in ("t", "b"):
        last, now = _EP["turns"][t], mech[t]["turns"]
        if last is not None:   # net turns (signed: a screw shaking back and forth adds nothing)
            (_EP["driven"] if eng == t else _EP["undriven"])[t] += now - last
        _EP["turns"][t] = now
    flush = is_flush(box_pos, box_r)
    _EP["ever_flush"] |= flush
    fed = bp.cable_fed(joints)
    cable_in = fed >= bp.FED - CABLE_OUT       # a box clamped with its cable drawn out has to come out again
    snug = {t: bool(mech[t]["swing"] >= SWUNG_DEG and mech[t]["force"] >= SNUG_N
                    and _EP["undriven"][t] <= UNDRIVEN_TURNS) for t in ("t", "b")}
    tip, _ = driver_points(d_pos, d_r)
    snap = np.vstack([box_pos, tip])
    recent = _EP["recent"]
    recent.append(snap)
    del recent[:-(HOLD_STEPS + 1)]
    still = len(recent) == HOLD_STEPS + 1 and all(float(np.abs(r - snap).max()) <= STILL for r in recent)
    c_box, c_drv = clearance(gripper_points, box_pos, box_r, d_pos, d_r)
    down = driver_low(d_pos, d_r) <= DOWN_Z and eng is None
    n_snug = sum(snug.values())
    done = bool(flush and cable_in and n_snug == 2 and down and still and min(c_box, c_drv) >= CLEAR)
    _EP["held"] = _EP["held"] + 1 if done else 0
    if _EP["held"] >= HOLD_STEPS:
        score = 1.0
    elif flush and cable_in and n_snug == 2:
        score = CREDIT["both"]
    elif flush and cable_in and n_snug == 1:
        score = CREDIT["one"]
    elif _EP["ever_flush"]:
        score = CREDIT["flush"]
    else:
        score = 0.0
    _EP.update(score=score, steps=_EP["steps"] + 1,
               info={"flush": flush, "gaps_mm": np.round(1000 * ear_gaps(box_pos, box_r), 2).tolist(), "snug": snug,
                     "mech": mech, "engaged": eng, "down": down, "still": still, "clear_box": c_box,
                     "cable_fed_mm": round(1000 * fed, 1), "cable_in": cable_in,
                     "clear_driver": c_drv, "driven": dict(_EP["driven"]), "undriven": dict(_EP["undriven"])})
    return score


def success(objects) -> bool:
    """True only on steps where score() is 1.0."""
    return _EP["score"] >= 1.0


def score(objects) -> float:
    """1.0 once, for 1 s (20 steps in a row): the box flush in the cut-out (each ear's back face within 0.5 mm of the
    wall at both its corners), both clamp arms swung out and pressing the back of the drywall with 20 N or more,
    each screw turned with the screwdriver, the screwdriver put down (its lowest point within 5 cm of the table, not
    in a screw), both grippers 3 cm or more from the box and the screwdriver, and the box and screwdriver still.
    0.75: flush and both clamps tight. 0.5: flush and one clamp tight. 0.25: the box was flush in the cut-out at
    some point. Else 0. The three higher scores also need the cable still fed into the box: no more than 15 mm of
    it drawn back out through the entry."""
    return float(_EP["score"])


# -- in the simulator ------------------------------------------------------------------------------------------------------

_HULLS: dict = {}


def _gripper_points(base, origin) -> np.ndarray:
    if not _HULLS:
        from yam_arena.analysis import link_hulls

        hulls = link_hulls()
        for link in ("gripper", "tip_left", "tip_right"):
            _HULLS[link] = np.vstack([h.vertices for h in hulls[link]])
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        poses = bp.as_numpy(robot.data.body_link_pose_w)[0]
        for j, n in enumerate(robot.body_names):
            link = n.rsplit("/", 1)[-1]
            if link in _HULLS:
                out.append((_HULLS[link] @ rotation(poses[j][3:7]).T) + poses[j][:3] - origin)
    return np.vstack(out) if out else np.zeros((0, 3))


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = bp.as_numpy(base.scene.env_origins[0])
    b = bp.part_state(env, "box")
    d = bp.part_state(env, "driver")
    observe(b.pos, rotation(b.quat_xyzw), b.joints, d.pos, rotation(d.quat_xyzw), _gripper_points(base, origin))
    if _DEBUG and _EP["steps"] % 10 == 0:
        inf = _EP["info"]
        m = inf["mech"]
        print(f"[owb] step {_EP['steps']} score {_EP['score']} flush {inf['flush']} gaps {inf['gaps_mm']} "
              f"box-hole {np.round(1000 * (b.pos - HOLE), 1).tolist()} "
              f"tip-box {np.round(1000 * (rotation(b.quat_xyzw).T @ (d.pos - b.pos)), 1).tolist()} "
              f"eng {inf['engaged']} down {inf['down']} still {inf['still']} fed {inf['cable_fed_mm']} clear {inf['clear_box']:.3f}/"
              f"{inf['clear_driver']:.3f} "
              + " ".join(f"{t}: turns {m[t]['turns']:.2f} swing {m[t]['swing']:.0f} x {1000 * m[t]['arm_x']:.1f} "
                         f"F {m[t]['force']:.1f}" for t in ("t", "b"))
              + f" driven {inf['driven']} undriven {inf['undriven']}", file=sys.stderr, flush=True)
