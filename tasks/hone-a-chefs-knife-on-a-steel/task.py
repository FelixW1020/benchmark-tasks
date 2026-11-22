"""On the station table (standing in for a kitchen counter): a polyethylene cutting board standing on a bench scale
whose display is a separate indicator.
An 8 in chef's knife lies on the board with its handle over the board's right edge, cutting edge away from the robot.
A 10 in honing steel lies on the board, on its tip and its guard, with its handle over the board's left edge. A green non-slip pad is on the
board.

The job is the one a cook does before starting to cut: hone the knife. One hand stands the steel upright with its
tip on the pad and holds it there by its handle, behind the guard. The other hand takes the knife by its handle and
draws the edge down the steel, from the heel to the tip, the blade's face 15 to 20 degrees off the steel, lightly:
four strokes on each face, a stroke on one side of the steel and then one on the other. Then both go back on the
board, the knife's edge turned away from the robot.

A stroke is counted only while the edge is really pressed on the steel (the simulator's contact force between the
blade and the rod; on a real rig a lamp wired across the two), running from the heel to the tip and down the steel in
that order. The scale under the board is how "lightly" is measured: its reading may rise by no more than 30 g during
a stroke.

The board's, the knife's and the steel's places change with ``HONE_LOT`` (A or B).

Run it:    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:      python demo.py task.py out_dir
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import honekit as K  # noqa: E402
from honekit import Board, Knife, Scale, Steel, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Hone the chef's knife on the honing steel: stand the steel on its tip on the green pad and draw the "
               "knife's edge down it from heel to tip at 15 to 20 degrees with light pressure, four strokes on each "
               "side of the blade, alternating sides. Then lay the knife and the steel back on the board, the "
               "knife's edge away from you.")

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("HONE_LOT", "A")
BOARD_X0 = {"A": 0.205, "B": 0.212}[LOT]         # x of the board's near edge
BOARD_Y0 = {"A": 0.146, "B": 0.152}[LOT]         # y of its right edge
BOARD_RECT = (BOARD_X0, BOARD_X0 + K.BOARD[0], BOARD_Y0, BOARD_Y0 + K.BOARD[1])
BOARD_MID = (BOARD_X0 + K.BOARD[0] / 2, BOARD_Y0 + K.BOARD[1] / 2)
BOARD_TOP = K.SCALE[2] + K.SCALE_GAP + K.BOARD[2]          # the board's top above the table, nothing on it
SCALE_MID = (BOARD_MID[0], BOARD_MID[1] - 0.065)           # the scale stands under the board, 6.5 cm right of its middle
PAD_XY = {"A": (0.370, 0.220), "B": (0.382, 0.228)}[LOT]   # the middle of the non-slip pad
PAD_TOP = BOARD_TOP + K.PAD[2]
KNIFE_EDGE_X = {"A": 0.478, "B": 0.486}[LOT]     # the lying knife's edge line (its spine is 5.3 cm nearer the robot)
KNIFE_HEEL_Y = BOARD_Y0 - 0.002                  # its heel: the handle starts just outside the board's right edge
STEEL_TURN = math.radians({"A": -90.0, "B": -87.0}[LOT])   # the lying steel points this way, handle to tip (from +x toward +y)
HOLD_AT = K.ROD_L + K.GUARD_T + 0.050            # where its handle is to be taken: 5 cm behind the guard, from the tip
STEEL_X = {"A": 0.360, "B": 0.368}[LOT]          # x of the lying steel (it lies along the board, across the robot's view)

KNIFE, STEEL, BOARD, SCALE = "knife", "steel", "board", "scale"

# -- scoring -----------------------------------------------------------------------------------------------------------
LOADED_N = 0.02          # the edge is on the steel when the simulator reports at least this contact force (N) between them
TOUCH = 0.0020           # ... and the edge's nearest point is within this of the rod's surface (where the contact is read off)
BACKTRACK = 0.015        # a stroke's contact may go back toward the heel, or back up the steel, by no more than this
NEAR_END = 0.030         # a stroke starts within this of the heel and ends within this of the tip
ANGLE = (10.0, 25.0)     # degrees between the blade's face and the steel, all through a stroke
STEEP = 30.0             # an edge on the steel at more than this is driven into it
TRAVEL = 0.050           # a stroke runs at least this far down the steel
LIGHT_G = 30.0           # the scale's reading may rise this much during a stroke (about 1 N of push on the rod)
HEAVY_G = 90.0           # ... and a rise of this much voids the trial
UPRIGHT = 15.0           # the steel is upright within this many degrees
HAND_NEAR = 0.020        # the blade may not come this near the gripper that holds the steel
EACH = 4                 # strokes on each face
CLEAR = 0.030            # both arms this far from the knife and the steel for them to be let go
STILL, TURN = 0.02, 0.5  # at rest: under this many m/s and rad/s
LIE = 0.006              # lying on the board: lowest point within this of the board's top
BREAK = 1                # a stroke ends at the first control step without loaded contact

_EDGE = K.edge_local(0.001)                      # the edge every millimetre, in the knife's frame
_EDGE_S = _EDGE[:, 0]


def knife_start() -> tuple:
    """The lying knife: flat on the board, blade along +y, edge away from the robot, handle over the right edge."""
    pos = np.array([KNIFE_EDGE_X, KNIFE_HEEL_Y, TOP + BOARD_TOP + K.T_SPINE / 2 + 0.0008])
    rot = np.column_stack([(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    return pos, K._quat(rot)


STEEL_TILT = math.asin((K.GUARD_B - K.ROD_R) / (K.ROD_L + K.GUARD_T / 2 - K.ROD_R))   # it lies on its guard and its tip
GUARD_IN = 0.006                                 # the lying steel's guard is this far in from the board's left edge


def steel_axes() -> np.ndarray:
    """The lying steel's axes in the world (columns): its x from the tip back and a little up to the handle, its z
    up, its guard's wider axis level."""
    u = np.array([math.cos(STEEL_TURN), math.sin(STEEL_TURN), 0.0])       # level, from the handle to the tip
    up = np.array([0.0, 0.0, 1.0])
    x = -math.cos(STEEL_TILT) * u + math.sin(STEEL_TILT) * up
    y = np.cross(up, -u)
    return np.column_stack([x, y, np.cross(x, y)])


def steel_start() -> tuple:
    """The lying steel: its tip and the flat of its guard on the board, the guard just inside the left edge, the
    handle over the edge."""
    rot = steel_axes()
    guard_mid = np.array([STEEL_X, BOARD_RECT[3] - GUARD_IN - K.GUARD_T / 2, TOP + BOARD_TOP + K.GUARD_B + 0.0008])
    tip = guard_mid - rot @ np.array([K.ROD_L + K.GUARD_T / 2, 0.0, 0.0])
    return tip, K._quat(rot)


def board_start() -> np.ndarray:
    return np.array([BOARD_MID[0], BOARD_MID[1], TOP + BOARD_TOP - K.BOARD[2] / 2])


def _objects() -> list:
    kp, kq = knife_start()
    sp, sq = steel_start()
    bp = board_start()
    scale = Scale(SCALE, pos=(SCALE_MID[0], SCALE_MID[1], TOP + K.SCALE[2] / 2), size=K.SCALE)
    board = Board(BOARD, pos=tuple(float(v) for v in bp), size=K.BOARD,
                  pad=(PAD_XY[0] - BOARD_MID[0], PAD_XY[1] - BOARD_MID[1]))
    knife = Knife(KNIFE, pos=tuple(float(v) for v in kp), size=(K.BL + K.HL, K.BH, K.HANDLE_T), quat=kq)
    steel = Steel(STEEL, pos=tuple(float(v) for v in sp), size=(K.ROD_L + K.GUARD_T + K.S_HANDLE, 2 * K.GUARD_A, 2 * K.GUARD_B),
                  quat=sq)
    return [scale, board, knife, steel]


OBJECTS = _objects()


# -- where things are (pure) ---------------------------------------------------------------------------------------------

def _apart(a, b) -> float:
    """The least distance between two sets of points."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if not len(a) or not len(b):
        return math.inf
    best = math.inf
    for i in range(0, len(a), 256):
        d = np.linalg.norm(a[i:i + 256, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def over_board(points, slack: float = 0.0) -> bool:
    """Are all these points over the board (or within ``slack`` of its outline)?"""
    p = np.atleast_2d(points)
    r = BOARD_RECT
    return bool(np.all((p[:, 0] >= r[0] - slack) & (p[:, 0] <= r[1] + slack) & (p[:, 1] >= r[2] - slack) & (p[:, 1] <= r[3] + slack)))


def touch(knife, steel) -> dict:
    """Where the knife's edge is against the steel's rod. ``knife``, ``steel``: (pos, quat).
    Returns the gap between the edge and the rod's surface (m; the least over the edge, along the rod only), and
    there: ``s`` (m along the edge from the heel), ``t`` (m up the rod from its tip), ``angle`` (degrees between the
    blade's face and the rod's surface), ``face`` (+1 or -1: which face of the blade is toward the rod), ``lead``
    (the edge points toward the rod's tip, so a stroke down the rod is edge first)."""
    kp, kq = knife
    sp, sq = steel
    rk, rs = K._rotation(kq), K._rotation(sq)
    a = rs[:, 0]                                             # the rod's axis, tip to handle
    e = K.place(kp, kq, _EDGE) - np.asarray(sp, float)
    t = e @ a
    rho = e - np.outer(t, a)
    d = np.linalg.norm(rho, axis=1)
    on = (t >= 0.0) & (t <= K.ROD_L)
    if not on.any():
        return {"gap": math.inf, "s": None, "t": None, "angle": None, "face": 0, "lead": False}
    i = int(np.argmin(np.where(on, d, math.inf)))
    r = rho[i] / max(d[i], 1e-9)
    n = rk[:, 2]                                             # the blade's normal
    c = float(n @ r)
    return {"gap": float(d[i] - K.ROD_R), "s": float(_EDGE_S[i]), "t": float(t[i]),
            "angle": math.degrees(math.acos(min(1.0, abs(c)))), "face": 1 if c < 0 else -1,
            "lead": bool((-rk[:, 1]) @ a < 0.0)}


def new_state() -> dict:
    return {"steps": 0, "void": "", "strokes": [], "counted": 0, "last_face": 0, "now": None, "off": 0,
            "idle_g": 0.0, "grams": 0.0, "peak_g": 0.0, "steep": 0, "knife_down": False, "steel_down": False,
            "edge_away": False, "let_go": False, "still": False, "nearest_hand_mm": None, "force_n": 0.0}


def _close(s: dict) -> None:
    """End the stroke in progress and judge it."""
    k = s["now"]
    s["now"], s["off"] = None, 0
    if k is None:
        return
    why = []
    if k["s0"] > NEAR_END:
        why.append(f"started {1000 * k['s0']:.0f} mm from the heel")
    if k["s1"] < K.BL - NEAR_END:
        why.append(f"ended {1000 * (K.BL - k['s1']):.0f} mm from the tip")
    if k["back_s"] > BACKTRACK:
        why.append(f"contact went {1000 * k['back_s']:.0f} mm back toward the heel")
    if k["back_t"] > BACKTRACK:
        why.append(f"contact went {1000 * k['back_t']:.0f} mm back up the steel")
    if k["a_min"] < ANGLE[0] or k["a_max"] > ANGLE[1]:
        why.append(f"angle {k['a_min']:.0f} to {k['a_max']:.0f} deg")
    if k["faces"] != 1:
        why.append("changed face")
    if k["t0"] - k["t1"] < TRAVEL:
        why.append(f"ran {1000 * (k['t0'] - k['t1']):.0f} mm down the steel")
    if not k["lead"]:
        why.append("edge trailing")
    if not k["upright"]:
        why.append("steel not upright on the pad")
    if k["rise"] > LIGHT_G:
        why.append(f"scale rose {k['rise']:.0f} g")
    rec = {"face": k["face"], "from_heel_mm": round(1000 * k["s0"]), "to_tip_mm": round(1000 * (K.BL - k["s1"])),
           "angle_deg": [round(k["a_min"], 1), round(k["a_max"], 1)], "down_steel_mm": round(1000 * (k["t0"] - k["t1"])),
           "rise_g": round(k["rise"], 1), "force_n": [round(k["f_min"], 2), round(k["f_max"], 2)],
           "back_mm": [round(1000 * k["back_s"], 1), round(1000 * k["back_t"], 1)], "steps": k["n"], "valid": not why,
           "why": "; ".join(why)}
    if not why and k["face"] != s["last_face"]:
        s["counted"] += 1
        s["last_face"] = k["face"]
        rec["counted"] = True
    else:
        rec["counted"] = False
    s["strokes"].append(rec)


def advance(s: dict, knife, steel, grams: float, arms, speeds, force: float = 0.0) -> dict:
    """One control step of the record. ``knife``, ``steel``: (pos, quat). ``grams``: the scale's reading.
    ``arms``: [left, right], each (tool point, points on that arm's gripper and fingers). ``speeds``: {name: (m/s,
    rad/s)} for the knife and the steel. ``force``: the contact force between the blade and the rod over this control
    step (N), as the simulator reports it; without it nothing is on the steel."""
    s["steps"] += 1
    s["grams"] = round(float(grams), 1)
    s["force_n"] = round(float(force), 3)
    c = touch(knife, steel)
    sp, sq = steel
    rs = K._rotation(sq)
    tip = np.asarray(sp, float)
    upright = (math.degrees(math.acos(max(-1.0, min(1.0, float(rs[2, 0]))))) <= UPRIGHT
               and abs(tip[0] - PAD_XY[0]) <= K.PAD[0] / 2 and abs(tip[1] - PAD_XY[1]) <= K.PAD[1] / 2
               and tip[2] - TOP - PAD_TOP <= 0.004)
    on = float(force) >= LOADED_N and c["gap"] <= TOUCH
    if on:
        k = s["now"]
        if k is None:
            # s0, t0: where the contact began; s1, t1: where it is now (in time order, not extremes)
            k = s["now"] = {"face": c["face"], "s0": c["s"], "s1": c["s"], "t0": c["t"], "t1": c["t"], "a_min": c["angle"],
                            "a_max": c["angle"], "faces": 1, "lead": c["lead"], "upright": upright, "ref": s["idle_g"],
                            "rise": 0.0, "n": 0, "s_hi": c["s"], "t_lo": c["t"], "back_s": 0.0, "back_t": 0.0,
                            "f_min": float(force), "f_max": float(force)}
        k["n"] += 1
        k["s1"], k["t1"] = c["s"], c["t"]
        k["back_s"] = max(k["back_s"], k["s_hi"] - c["s"])       # how far the contact has gone back toward the heel
        k["back_t"] = max(k["back_t"], c["t"] - k["t_lo"])       # ... and back up the steel
        k["s_hi"], k["t_lo"] = max(k["s_hi"], c["s"]), min(k["t_lo"], c["t"])
        k["f_min"], k["f_max"] = min(k["f_min"], float(force)), max(k["f_max"], float(force))
        k["a_min"], k["a_max"] = min(k["a_min"], c["angle"]), max(k["a_max"], c["angle"])
        k["faces"] = k["faces"] if c["face"] == k["face"] else 2
        k["lead"] = k["lead"] and c["lead"]
        k["upright"] = k["upright"] and upright
        k["rise"] = max(k["rise"], grams - k["ref"])
        s["off"] = 0
        s["peak_g"] = max(s["peak_g"], round(k["rise"], 1))
        s["steep"] = s["steep"] + 1 if c["angle"] > STEEP else 0
        if s["steep"] >= 3 and not s["void"]:
            s["void"] = f"step {s['steps']}: the edge was on the steel at {c['angle']:.0f} degrees"
        if k["rise"] >= HEAVY_G and not s["void"]:
            s["void"] = f"step {s['steps']}: the scale rose {k['rise']:.0f} g during a stroke"
    else:
        s["steep"] = 0
        if s["now"] is not None:
            s["off"] += 1
            if s["off"] >= BREAK:
                _close(s)
        else:
            s["idle_g"] = float(grams)
    # while the steel stands on the pad, the blade stays away from the hand that holds it, and below the guard
    blade = K.blade_points(*knife)
    grip_at = tip + rs @ np.array([HOLD_AT, 0.0, 0.0])
    for tool, pts in arms:
        if upright and np.linalg.norm(np.asarray(tool, float) - grip_at) <= 0.06:
            near = _apart(pts, blade)
            if s["nearest_hand_mm"] is None or 1000 * near < s["nearest_hand_mm"]:
                s["nearest_hand_mm"] = round(1000 * near, 1)
            if near < HAND_NEAR and not s["void"]:
                s["void"] = f"step {s['steps']}: the blade came within {1000 * near:.0f} mm of the hand on the steel"
            rel = blade - tip
            up = rel @ rs[:, 0]
            out = np.linalg.norm(rel - np.outer(up, rs[:, 0]), axis=1)
            if bool(np.any((up > K.ROD_L) & (out < 0.05))) and not s["void"]:
                s["void"] = f"step {s['steps']}: the blade passed the steel's guard"
    # how things have been left
    rk = K._rotation(knife[1])
    kp = K.knife_points(*knife)
    low = float(kp[:, 2].min()) - TOP - BOARD_TOP
    s["knife_down"] = bool(over_board(blade, 0.01) and abs(low) <= LIE + K.HANDLE_T / 2 and abs(float(rk[2, 2])) >= 0.97)
    s["edge_away"] = bool(float(-rk[0, 1]) >= -0.26)
    rod = K.place(sp, sq, np.array([[0.0, 0.0, 0.0], [K.ROD_L, 0.0, 0.0]]))
    s["steel_down"] = bool(over_board(rod[:1]) and abs(float(rs[2, 0])) <= 0.17
                           and abs(float(rod[0, 2]) - TOP - BOARD_TOP - K.ROD_R) <= LIE + 0.004)
    both = np.vstack([kp, K.steel_points(sp, sq)])
    s["let_go"] = bool(all(_apart(pts, both) >= CLEAR for _, pts in arms))
    s["still"] = bool(all(speeds[n][0] < STILL and speeds[n][1] < TURN for n in (KNIFE, STEEL)))
    return s


def score_from(void: bool, counted: int, knife_down: bool, steel_down: bool, edge_away: bool, let_go: bool, still: bool) -> float:
    """Pure scoring rule. ``void``: the blade came within 2 cm of the gripper holding the steel or passed the steel's
    guard, the edge was on the steel at more than 30 degrees, or the scale rose 90 g during a stroke. ``counted``:
    valid strokes, each on the other face from the one counted before. The rest describe the end of the trial."""
    if void:
        return 0.0
    if counted >= 2 * EACH:
        if knife_down and steel_down:
            return 1.0 if (edge_away and let_go and still) else 0.75
        return 0.5
    return 0.25 if counted >= 2 else 0.0


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = new_state()


def _clear() -> None:
    _STATE.clear()
    _STATE.update(new_state())


@on_reset
def _reset(environment) -> None:
    _clear()


def _hands(environment) -> list:
    """For each arm: its tool point and the points on its last link, gripper and fingers (world)."""
    base = environment.unwrapped
    origin = K._origin(environment)
    local = K.link_points()
    out = []
    for (tool, _), arm in zip(K.finger_points(environment), ("left_robot", "right_robot")):
        robot = base.scene[arm]
        poses = K._np(robot.data.body_link_pose_w)[0]
        names = list(robot.body_names)
        pts = []
        for link in ("link5", "gripper", "tip_left", "tip_right"):
            if link in names and len(local[link]):
                pose = poses[names.index(link)]
                pts.append(local[link] @ K._rotation(pose[3:7]).T + pose[:3] - origin)
        out.append((tool, np.vstack(pts)))
    return out


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    st = object_states(environment, [KNIFE, STEEL, BOARD])
    poses = {n: (st[n].pos, st[n].quat_xyzw) for n in (KNIFE, STEEL)}
    speeds = {n: (float(np.linalg.norm(st[n].lin_vel)), float(np.linalg.norm(st[n].ang_vel))) for n in (KNIFE, STEEL)}
    grams = K.scale_grams(float(st[BOARD].pos[2]), float(board_start()[2]))
    force, share, _ = K.edge_force()
    _STATE["contact_share"] = round(share, 2)
    advance(_STATE, poses[KNIFE], poses[STEEL], grams, _hands(environment), speeds, force)


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored from a record kept through the trial and from the end state.
    The edge is on the steel while the simulator reports a contact force of 0.02 N or more between the blade and the
    rod (on a real rig: while a lamp wired across the knife and the steel is lit). A stroke is one unbroken spell of
    that; it ends at the first control step without it. It is valid if the contact begins within 3 cm of the heel
    and ends within 3 cm of the tip, moving toward the tip and down the steel in time order (never more than 1.5 cm
    back toward the heel or back up the steel), the blade's face stays 10 to 25 degrees off the steel, on one face,
    the contact ends at least 5 cm lower on the steel than it began with the edge leading, the steel stands within
    15 degrees of upright with its tip on the pad, and the scale's reading never rises more than 30 g above what it
    read before the stroke. A valid stroke is counted if it is on the other face from the last one counted.
    0 if the blade came within 2 cm of the gripper holding the steel or passed the guard, if the edge was on the
    steel at more than 30 degrees for 0.15 s, or if the scale rose 90 g during a stroke.
    Otherwise 0.25 for two counted strokes; 0.5 for eight; 0.75 if, as well, the knife lies flat on the board and the
    steel lies on the board; 1.0 if, as well, the knife's edge is not turned toward the robot, both arms are 3 cm
    from both, and both are at rest (under 2 cm/s and 0.5 rad/s)."""
    s = _STATE
    return score_from(bool(s["void"]), s["counted"], s["knife_down"], s["steel_down"], s["edge_away"], s["let_go"], s["still"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    return {"score": score(objects), "lot": LOT, "void": s["void"], "counted_strokes": s["counted"],
            "strokes": list(s["strokes"]), "in_stroke": s["now"] is not None, "scale_g": s["grams"],
            "edge_force_n": s["force_n"], "contact_share": s.get("contact_share"),
            "peak_rise_g": s["peak_g"], "nearest_to_steel_hand_mm": s["nearest_hand_mm"], "knife_down": s["knife_down"],
            "steel_down": s["steel_down"], "edge_away": s["edge_away"], "let_go": s["let_go"], "at_rest": s["still"]}


def demo(objects=None, seed=None):
    return {}
