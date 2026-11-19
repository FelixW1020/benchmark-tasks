"""On the station table: a 12 inch table fan, unplugged, lying on its back on a folded towel with its motor shaft
pointing up and its rear grille on. The three parts that come off for washing lie dry beside it: the blade on a
folded tea towel, the blade cap on the table, the front grille on a dish rack.

The job is to put the fan back together, in the order the maker's instructions give:

1. the blade onto the motor shaft, turned so that the slot in the back of its hub drops over the shaft's cross pin;
2. the blade cap onto the shaft's threaded end and tight against the hub. The thread is left-handed (it tightens
   anticlockwise), and the shaft of a switched-off motor turns freely, so the blade has to be held while the cap is
   turned;
3. the front grille onto the rear grille, its bottom edge under the lip of the rear grille's support, and the rear
   grille's three clips pressed shut over its rim.

Where the fan and the parts lie, and the way the shaft's pin points, change with ``FAN_LOT`` (A or B).

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

import fankit as K  # noqa: E402
from fankit import Blade, Cap, Clip, Fan, Grille, Rotor, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the fan back together."

# -- layout (m; x away from the robot, y left, heights above the table) ---------------------------------------------------
LOT = os.environ.get("FAN_LOT", "A")
# fan: under the motor shaft; pin: the way the shaft's cross pin points (degrees from x); blade, cap, grille: their
# middles; turn: how far the blade lies turned (degrees) from the way it has to go on; grille_turn: how far the front
# grille lies turned (degrees) from the way it goes on
LAYOUT = {"A": {"fan": (0.240, -0.100), "pin": 0.0, "blade": (0.440, -0.420), "turn": 0.0, "cap": (0.220, -0.430),
                "grille": (0.550, 0.200), "grille_turn": 30.0},
          "B": {"fan": (0.250, -0.090), "pin": 15.0, "blade": (0.450, -0.410), "turn": 15.0, "cap": (0.190, -0.440),
                "grille": (0.560, 0.205), "grille_turn": 25.0}}[LOT]
FAN_TOWEL = (0.460, 0.300, 0.025)        # the towel the fan lies on (x, y, thickness)
BLADE_TOWEL = (0.200, 0.200, 0.020)      # the tea towel under the blade
RACK = (0.300, 0.300, 0.140)             # the dish rack the front grille lies on (a block of its size)
GRILLE_HOLD_AZ = 150.0                   # where round its rim the grille is taken (degrees from its bottom edge)
RACK_YAW = GRILLE_HOLD_AZ + LAYOUT["grille_turn"] - 180.0     # degrees: one of the rack's sides lies under that part of the rim
BLADE_DOWN = 0.0085                      # the blade's trailing edges hang this far below its hub's back face
REST = 0.0005                            # parts are set down this far above what they lie on
TOWEL = (0.86, 0.82, 0.72)

FAN, ROTOR, BLADE, CAP, GRILLE = "fan", "rotor", "blade", "cap", "grille"
CLIPS = ("clip_a", "clip_b", "clip_c")
THREAD, HINGES = f"{CAP}_thread", tuple(f"{c}_hinge" for c in CLIPS)
LOOSE = (BLADE, CAP, GRILLE)

# -- scoring -----------------------------------------------------------------------------------------------------------
ON_AXIS, BORE_TILT, PIN_HOME = 0.0015, 3.0, 0.0010     # the hub on the shaft: m off its axis, degrees, m above the pin's seat
CAP_GAP = 0.0008              # the cap's face this near the hub's
GRILLE_OFF, GRILLE_UP, GRILLE_TURN = 0.003, 0.002, 6.0   # the front grille on the rear one: m, m, degrees
CLEAR = 0.030                 # both grippers this far from every part at the end
CONTROL_HZ = 20
REST_S, REST_MOVE, REST_TURN = 0.5, 0.002, 3.0   # at rest: in this long no part has moved this far (m) or turned this much (degrees)


def fan_pose() -> tuple:
    x, y = LAYOUT["fan"]
    return np.array([x, y, TOP + FAN_TOWEL[2]]), (0.0, 0.0, 0.0, 1.0)


def blade_yaw_on() -> float:
    """The way the blade has to be turned (degrees about the shaft) for its slot to lie along the pin, with the blade
    the demo holds on the fan's left."""
    return 180.0 + LAYOUT["pin"]


def blade_start() -> tuple:
    x, y = LAYOUT["blade"]
    return (np.array([x, y, TOP + BLADE_TOWEL[2] + BLADE_DOWN + REST]), K.turn_z(math.radians(blade_yaw_on() - LAYOUT["turn"])))


def cap_start() -> tuple:
    x, y = LAYOUT["cap"]
    return np.array([x, y, TOP + REST]), (0.0, 0.0, 0.0, 1.0)


def grille_start() -> tuple:
    x, y = LAYOUT["grille"]
    return np.array([x, y, TOP + RACK[2] + REST]), K.turn_z(math.radians(LAYOUT["grille_turn"]))


def _objects() -> list:
    tup = lambda v: tuple(float(x) for x in v)   # noqa: E731
    fp, fq = fan_pose()
    bp, bq = blade_start()
    cp, cq = cap_start()
    gp, gq = grille_start()
    out = [Box("fan_towel", pos=(fp[0] + 0.130, fp[1], TOP + FAN_TOWEL[2] / 2), size=FAN_TOWEL, color=TOWEL),
           Box("blade_towel", pos=(bp[0], bp[1], TOP + BLADE_TOWEL[2] / 2), size=BLADE_TOWEL, color=TOWEL),
           Box("rack", pos=(gp[0], gp[1], TOP + RACK[2] / 2), size=RACK, color=(0.62, 0.64, 0.66), yaw=math.radians(RACK_YAW)),
           Fan(FAN, pos=tup(fp), quat=fq, size=(0.60, 0.35, 0.21)),
           Rotor(ROTOR, pos=tup(fp), quat=K.turn_z(math.radians(LAYOUT["pin"])), size=(0.016, 0.016, 0.05), fan=FAN)]
    for k, name in enumerate(CLIPS):
        out.append(Clip(name, pos=tup(K.place(fp, fq, K.clip_hinge(k))), quat=K._quat(K._rotation(fq) @ K.clip_rot(k, K.CLIP_OPEN)),
                        size=(0.013, 0.024, 0.017), fan=FAN, index=k))
    out += [Blade(BLADE, pos=tup(bp), quat=bq, size=(0.30, 0.30, 0.04)),
            Cap(CAP, pos=tup(cp), quat=cq, size=(0.038, 0.038, 0.018), rotor=ROTOR, fan=FAN),
            Grille(GRILLE, pos=tup(gp), quat=gq, size=(0.35, 0.35, 0.058))]
    return out


OBJECTS = _objects()


# -- what the trial looks like now ------------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, blade={}, cap={}, grille={}, clips=[K.CLIP_OPEN] * 3, thread={}, blade_seated=False,
                  cap_on=False, cap_tight=False, grille_down=False, shut=0, released=True, still=True, near=1.0,
                  first={}, seen=[])


_clear()


def blade_seated(fit: dict) -> bool:
    """The hub is on the shaft with the pin at the bottom of its slot."""
    return bool(fit["off"] <= ON_AXIS and fit["tilt"] <= BORE_TILT and abs(fit["up"]) <= PIN_HOME)


def cap_tight(fit: dict, on: bool, target: float) -> bool:
    """The cap is on the thread, turned down until the thread draws it against the hub, and its face is on the
    hub's."""
    return bool(on and target <= K.CAP_SEAT - K.SNUG + 1e-6 and fit["up"] <= CAP_GAP and fit["tilt"] <= BORE_TILT)


def grille_down(fit: dict) -> bool:
    """The front grille lies on the rear grille's rim, centred, its bottom edge under the support's lip."""
    return bool(fit["off"] <= GRILLE_OFF and fit["up"] <= GRILLE_UP and fit["low"] >= -GRILLE_UP and fit["turn"] <= GRILLE_TURN
                and fit["under"])


def at_rest(seen: list) -> bool:
    """Whether every part has been at rest for ``REST_S``. ``seen`` holds one entry per control step, newest last; each
    entry is a list of (position, quaternion) for the loose parts and the clips. At rest: over the last ``REST_S``
    no part has been more than ``REST_MOVE`` from where it is now, or turned more than ``REST_TURN`` degrees from
    how it lies now. (The simulator's reported speeds of small sprung parts chatter while the parts stay put, so
    rest is judged from places, as it would be from a video.)"""
    n = int(round(REST_S * CONTROL_HZ))
    if len(seen) < n + 1:
        return False
    now = seen[-1]
    for old in seen[-(n + 1):-1]:
        for (p0, q0), (p1, q1) in zip(old, now):
            if float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float))) > REST_MOVE:
                return False
            dot = min(1.0, abs(float(np.dot(np.asarray(q0, float), np.asarray(q1, float)))))
            if math.degrees(2.0 * math.acos(dot)) > REST_TURN:
                return False
    return True


def score_from(blade_ok: bool, cap_ok: bool, grille_ok: bool, shut: int, released: bool, still: bool) -> float:
    """Pure scoring rule. ``blade_ok``: the hub is on the shaft with the pin in its slot. ``cap_ok``: the cap is
    tight against the hub. ``grille_ok``: the front grille is down on the rear one with its bottom edge in the
    support. ``shut``: how many of the three clips are shut over it. ``released``, ``still``: both grippers are clear
    of every part and nothing moves."""
    if not blade_ok:
        return 0.0
    if not cap_ok:
        return 0.25
    if not grille_ok:
        return 0.5
    if shut >= 3 and released and still:
        return 1.0
    return 0.75


def part_points(st) -> np.ndarray:
    """Points on the three loose parts and the clips (world), for the distance to the grippers."""
    pts = []
    b = st[BLADE]
    for piece in K.blade_boxes():
        centre, rot, size = piece
        for u in (-0.4, 0.0, 0.4):
            for v in (-0.4, 0.0, 0.4):
                pts.append(K.place(b.pos, b.quat_xyzw, centre + rot @ np.array([u * size[0], v * size[1], 0.0])))
    pts.append(K.place(b.pos, b.quat_xyzw, (0.0, 0.0, K.HUB_H)))
    c = st[CAP]
    for a in range(0, 360, 60):
        pts.append(K.place(c.pos, c.quat_xyzw, (K.CAP_R * math.cos(math.radians(a)), K.CAP_R * math.sin(math.radians(a)), K.CAP_H / 2)))
    pts.append(K.place(c.pos, c.quat_xyzw, (0.0, 0.0, K.CAP_H)))
    g = st[GRILLE]
    for a in range(0, 360, 6):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        pts.append(K.place(g.pos, g.quat_xyzw, (K.RIM_OUT * ca, K.RIM_OUT * sa, K.GRILLE_T / 2)))
        if a % 30 == 0:
            for r, z in K.DOME[1:]:
                pts.append(K.place(g.pos, g.quat_xyzw, (r * ca, r * sa, z)))
    pts.append(K.place(g.pos, g.quat_xyzw, (0.0, 0.0, K.BADGE_Z[1])))
    for name in CLIPS:
        s = st[name]
        pts.append(K.place(s.pos, s.quat_xyzw, (0.0, 0.0, 0.012)))
    return np.array(pts)


@on_reset
def _reset(environment) -> None:
    _clear()


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    st = object_states(environment, [FAN, ROTOR, BLADE, CAP, GRILLE, *CLIPS])
    fan, rotor, blade, cap, grille = (st[n] for n in (FAN, ROTOR, BLADE, CAP, GRILLE))
    s["blade"] = K.blade_fit(blade.pos, blade.quat_xyzw, fan.pos, fan.quat_xyzw, rotor.quat_xyzw)
    s["pin"] = K._yaw_about(fan.quat_xyzw, rotor.quat_xyzw)
    s["spin"] = float(np.degrees(np.asarray(rotor.ang_vel, float) @ K._rotation(fan.quat_xyzw)[:, 2]))
    s["cap"] = K.cap_fit(cap.pos, cap.quat_xyzw, fan.pos, fan.quat_xyzw)
    s["grille"] = K.grille_fit(grille.pos, grille.quat_xyzw, fan.pos, fan.quat_xyzw)
    s["thread"] = K.mechanism(THREAD)
    s["clips"] = [K.clip_deg(k, fan.quat_xyzw, st[name].quat_xyzw) for k, name in enumerate(CLIPS)]
    was = (s["blade_seated"], s["cap_tight"], s["grille_down"], s["shut"])
    s["blade_seated"] = blade_seated(s["blade"])
    s["cap_on"] = bool(s["thread"].get("on"))
    s["cap_tight"] = cap_tight(s["cap"], s["cap_on"], float(s["thread"].get("target", 1.0)))
    s["grille_down"] = grille_down(s["grille"])
    s["shut"] = int(sum(1 for d in s["clips"] if d <= K.CLIP_SHUT)) if s["grille_down"] else 0
    for key, old, new in zip(("blade_seated", "cap_tight", "grille_down", "shut"), was, (s["blade_seated"], s["cap_tight"], s["grille_down"], s["shut"])):
        if new and not old and key not in s["first"]:
            s["first"][key] = s["steps"]
    hands = np.array([p for side in K.gripper_points(environment) for p in side])
    parts = part_points(st)
    s["near"] = float(np.min(np.linalg.norm(hands[:, None, :] - parts[None, :, :], axis=2)))
    s["released"] = bool(s["near"] >= CLEAR)
    moving = [st[n] for n in (*LOOSE, *CLIPS)]
    s["seen"].append([(np.array(m.pos, float), np.array(m.quat_xyzw, float)) for m in moving])
    del s["seen"][:-(int(round(REST_S * CONTROL_HZ)) + 1)]
    s["still"] = at_rest(s["seen"])


# -- scoring --------------------------------------------------------------------------------------------------------
def score(objects) -> float:
    """Scored at the end of the trial, highest row that holds. 0.25: the blade is on the shaft with the pin at the
    bottom of the slot in its hub (the hub within 1.5 mm of the shaft's axis, its back face within 1 mm of its seat;
    riding on top of the pin it stands 7 mm high). 0.5: also the blade cap is on the thread and tight: turned down
    until its face is within 0.8 mm of the hub's and the thread draws it against the hub. 0.75: also the front grille
    lies on the rear grille's rim (centred within 3 mm, no point of its rim more than 2 mm up) with its bottom edge
    under the lip of the rear grille's support. 1.0: also all three clips are shut over the front grille's rim
    (within 6 degrees of fully in), both grippers are at least 3 cm from every part, and everything is at rest (in the
    last half second no part or clip has moved more than 2 mm or turned more than 3 degrees). 0 otherwise."""
    s = _STATE
    return score_from(s["blade_seated"], s["cap_tight"], s["grille_down"], s["shut"], s["released"], s["still"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    r3 = lambda d: {k: (round(1000 * v, 2) if k in ("off", "up", "low") else (round(v, 1) if isinstance(v, float) else v)) for k, v in d.items()}   # noqa: E731
    th = s["thread"]
    return {"score": score(objects), "lot": LOT, "pin_deg": round(s.get("pin", 0.0), 1), "spin_deg_s": round(s.get("spin", 0.0), 1), "blade_seated": s["blade_seated"], "cap_on": s["cap_on"], "cap_tight": s["cap_tight"],
            "grille_down": s["grille_down"], "shut": s["shut"], "released": s["released"], "still": s["still"],
            "blade": r3(s["blade"]), "cap": r3(s["cap"]), "grille": r3(s["grille"]), "clips_deg": [round(d, 1) for d in s["clips"]],
            "thread": {"on": th.get("on"), "turns": round(float(th.get("turns", 0.0)), 3), "caught": th.get("caught"),
                       "came_off": th.get("came_off"), "to_go_mm": round(1000 * (float(th.get("target", K.SHAFT_TIP)) - (K.CAP_SEAT - K.SNUG)), 2)},
            "near_mm": round(1000 * s["near"], 1), "first": dict(s["first"])}


def demo(objects=None, seed=None):
    return {}
