"""fankit: a 12 inch table fan lying on its back, and the three parts that come off it for washing.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from fankit import Blade, Cap, Clip, Fan, Grille, Rotor, on_reset, on_step

The parts, each with its own frame:

- :class:`Fan`: the motor housing with the rear grille on it, the pillar and the base, lying on their back. It does
  not move. ``pos`` is the point under the motor shaft at the level the housing lies on; z runs up the shaft (toward
  the fan's front), x toward the fan's bottom (the pillar side), where the rear grille's support is.
- :class:`Rotor`: the motor shaft with its cross pin. It turns freely in the housing against a little drag. Same
  origin as the fan; its x runs along the pin.
- :class:`Blade`: the three-blade propeller. ``pos`` is the middle of the hub's back face; z runs through the bore
  toward the front; x runs along the slot in the hub's back that takes the pin.
- :class:`Cap`: the blade cap (spinner). ``pos`` is the middle of the face that meets the hub; z runs to its top.
- :class:`Grille`: the front grille. ``pos`` is the middle of the plane its rim sits on; z runs to the badge; x
  toward its bottom edge, the one that goes into the rear grille's support.
- :class:`Clip`: one of the rear grille's three clips. ``pos`` is its hinge; z runs up its stem when it is shut,
  x away from the fan's axis.

Every fit is made by colliding shapes:

- the hub's bore is a ring of boxes round the shaft, and the slot across its back is a gap between boxes: the hub
  comes down onto the pin only when the slot is turned to lie along it, and otherwise rides on top of the pin;
- the pin in the slot is what turns the blade with the shaft;
- the cap's bore goes over the shaft's threaded end. The thread itself cannot be cut in this simulator: once the
  cap's bore is over the shaft's end, a joint keeps the cap on the shaft's axis, free to turn against thread
  friction, and a spring draws it along the shaft by the thread's pitch for every turn of the cap on the shaft,
  anticlockwise seen from the front (a left-hand thread). Turned the other way it backs off and comes free;
- the grille's rim rests on the rear grille's rim; its bottom edge goes under the lip of the rear grille's support;
- each clip is a hooked lever on a hinge with an over-centre spring: it stands open until it is pressed in past half
  way, and then shuts over the front grille's rim. A rim that is not down on the rear grille stops it.

All sizes are plain numbers at the top of this file, so the same numbers serve the scene, the scoring and the
offline checks. Isaac Sim is reached only when the scene is built.

The structure follows the kits used by tasks already in this repository (Box subclasses so the loader accepts the
parts, custom spawners, hooks installed through ``sys.meta_path``, passive joints authored afresh at every reset).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Blade", "Cap", "Clip", "Fan", "Grille", "KitState", "Rotor", "blade_fit", "cap_fit", "clip_deg",
           "clip_hinge", "env", "finger_points", "grille_fit", "gripper_points", "install", "on_reset", "on_step",
           "place", "thread", "tilt_between", "to_local", "turn_z"]

# -- the fan as it lies (m, in the fan's frame; estimates unless marked) -------------------------------------------------
HOUSING_R, HOUSING_H = 0.055, 0.095        # the motor housing, lying on its back
BOSS_R, BOSS_Z = 0.016, 0.108              # the front bearing boss and the rear grille's locknut round it
RIM_OUT, RIM_IN = 0.1745, 0.1605           # both grilles' rims: 34.9 cm across (the fan's listed width), 14 mm wide
RIM_Z, RIM_T = 0.112, 0.010                # the top of the rear grille's rim, which the front grille sits on
DISH = ((0.030, 0.085), (RIM_IN, 0.102))     # the rear grille's wires, from the locknut out to the rim (radius, height)
N_WIRES, N_SPOKES = 36, 12                 # wires drawn, and how many of them collide
SHAFT_R, SHAFT_Z0, SHAFT_TIP = 0.004, 0.109, 0.150          # an 8 mm shaft (it begins clear of the locknut's shape)
THREAD_R, THREADED = 0.003, 0.008          # its threaded end: 6 mm across, 8 mm long
TAPER = (0.0038, 0.0035, 0.0032)           # the shoulder between them, in three 1 mm steps
PIN_Z, PIN_D, PIN_L = 0.125, 0.003, 0.016                   # the cross pin: its middle's height, thickness, length
THREAD_LEN = 0.0065                        # how much of the shaft's threaded end stands out of a seated hub
SUPPORT_HALF = 0.020                       # the rear grille's support: half its width along the rim
SUPPORT_WALL = (RIM_OUT + 0.0015, RIM_OUT + 0.0045)         # its outer wall (radii)
SUPPORT_LIP = (RIM_Z + RIM_T + 0.0040, RIM_Z + RIM_T + 0.0065)   # its lip (heights): 4 mm over a seated rim
SUPPORT_REACH = RIM_OUT - 0.005            # the lip reaches in to this radius
PILLAR = ((0.050, -0.022, 0.012), (0.250, 0.022, 0.056))    # the pillar, lying behind the housing (a box: lo, hi)
BASE_X, BASE_R, BASE_T = 0.262, 0.105, 0.024                # the base, standing on its edge behind the pillar
ROTOR_DRAG = 0.004                         # N m: what it takes to turn the switched-off motor
ROTOR_DAMP = 0.005                         # N m s/rad: its drag below that
ROTOR_SPIN_DAMP = 20.0                     # 1/s: the shaft's own damping (it stops when let go)
ROTOR_MASS, ROTOR_INERTIA = 0.150, 4.0e-5  # kg, kg m2 about the shaft

# -- the blade ---------------------------------------------------------------------------------------------------------
BLADE_R = 0.150                            # a 12 inch (30 cm) blade (the maker's size)
HUB_R, HUB_H = 0.022, 0.024
BORE_R = 0.0043                            # 0.3 mm of room round the shaft
HUB_WALL = 0.0090                          # the hub's shell begins this far from the axis (past the pin's ends)
SLOT_W, SLOT_D = 0.0046, 0.0070            # the slot across the hub's back: 4.6 mm for a 3 mm pin, 7 mm deep
BLADE_T = 0.003
BLADE_SEGS = ((0.022, 0.060, 0.044, 30.0), (0.060, 0.105, 0.100, 22.0), (0.105, 0.134, 0.085, 15.0),
              (0.134, 0.149, 0.034, 15.0))     # the last piece rounds the tip: no corner is outside BLADE_R
#             from, to (radius), chord, pitch (degrees; the leading edge, the way the angle grows round the hub, is the high one)
BLADE_MASS = 0.070

# -- the blade cap -----------------------------------------------------------------------------------------------------
CAP_R, CAP_H = 0.019, 0.018
CAP_BORE_R, CAP_BORE_D = 0.0034, 0.010
CAP_MOUTH_R, CAP_MOUTH_D = 0.0048, 0.002  # the bore's mouth is countersunk
CAP_MASS = 0.010
PITCH = 0.004                              # m a turn (coarser than a real fan's thread, to keep a trial short)
CATCH_IN = 0.0025                          # the thread catches once the shaft's end is this far into the cap's bore
CATCH_OFF, CATCH_TILT = 0.0015, 10.0       # ... with the cap this near the shaft's axis (m) and this upright (degrees)
THREAD_K, THREAD_PULL = 8000.0, 6.0        # the thread's draw along the shaft: N/m, at most N
THREAD_DAMP = 20.0                         # N s/m
THREAD_DRAG = 0.020                        # N m s/rad: thread friction; it carries the free shaft round with the cap
SNUG = 0.0006                              # the spring's target stops this far past the hub's face: the cap is tight

# -- the front grille --------------------------------------------------------------------------------------------------
GRILLE_T = 0.010                           # its rim's thickness
BAND_T, BAND_Z = 0.003, 0.030              # the band that stands on the rim's inner edge, up to where the ribs begin
DOME = ((RIM_IN + 0.0015, 0.030), (0.125, 0.058), (0.033, 0.070))     # its ribs, from the rim in to the badge (radius, height)
N_RIBS, N_RIB_SOLID = 36, 12
BADGE_R, BADGE_Z = 0.035, (0.064, 0.076)
GRILLE_MASS = 0.110

# -- the clips ---------------------------------------------------------------------------------------------------------
CLIP_AZ = (90.0, 180.0, 270.0)             # where they sit round the rear grille's rim (degrees from the fan's x)
CLIP_HINGE_R, CLIP_HINGE_Z = RIM_OUT + 0.0005, RIM_Z - 0.003
CLIP_HALF = 0.012                          # half a clip's width along the rim
CLIP_STEM = ((0.0010, 0.0), (0.0040, 0.0170))      # its stem: (out, up) corners in the clip's frame, shut
CLIP_HOOK = ((-0.0090, 0.0140), (0.0040, 0.0170))  # its hook
CLIP_OPEN, CLIP_MID = 55.0, 27.0           # degrees: where it stands open, and the over-centre point
CLIP_SHUT = 6.0                            # a clip within this of fully in counts as shut
CLIP_K, CLIP_PRE = 0.0004, 8.0             # its spring: N m a degree, wound 8 degrees past each end
CLIP_MASS = 0.003

PAD_PATCH = 0.004                          # radius of the patch over which a rubber gripper pad meets a part (an estimate)
FRICTION = (0.6, 0.5)
PALM_BACK, FINGER_THICK = 0.065, 0.012

WHITE, GREY, STEEL = (0.90, 0.90, 0.88), (0.55, 0.56, 0.58), (0.72, 0.73, 0.75)
BLUE, DARK, RED = (0.55, 0.74, 0.90), (0.16, 0.17, 0.19), (0.80, 0.10, 0.10)

# heights that follow from the sizes
HUB_SEAT = PIN_Z + PIN_D / 2 - SLOT_D      # the hub's back face when the pin is at the bottom of its slot
HUB_RIDING = PIN_Z + PIN_D / 2             # ... and when the hub rides on top of the pin
CAP_SEAT = HUB_SEAT + HUB_H                # the cap's lower face when it is down on a seated hub
GRILLE_SEAT = RIM_Z                        # the front grille's rim plane when it is down on the rear grille


# -- geometry (pure) -------------------------------------------------------------------------------------------------

def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat(rot) -> tuple:
    m = np.asarray(rot, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return tuple(float(v) for v in q)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def turn_z(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn of ``angle`` rad about the vertical."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def place(pos, quat, local) -> np.ndarray:
    """A point given in a part's own frame, in the world."""
    return np.asarray(pos, float) + _rotation(quat) @ np.asarray(local, float)


def to_local(point, pos, quat) -> np.ndarray:
    return _rotation(quat).T @ (np.asarray(point, float) - np.asarray(pos, float))


def tilt_between(quat_a, quat_b) -> float:
    """Degrees between two parts' z axes."""
    c = float(_rotation(quat_a)[:, 2] @ _rotation(quat_b)[:, 2])
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _yaw_about(quat_a, quat_b) -> float:
    """How far part b is turned from part a about a's z, anticlockwise seen from a's +z (degrees, -180..180)."""
    rel = _rotation(quat_a).T @ _rotation(quat_b)
    return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def _ring(r_in: float, r_out: float, z0: float, z1: float, n: int, start: float = 0.0) -> list:
    """``n`` boxes standing in a ring about z, their inner faces touching the circle of radius ``r_in``:
    (centre, rotation, size) each. Sizes are (radial, along the ring, up)."""
    out = []
    width = 2 * r_out * math.tan(math.pi / n)
    for k in range(n):
        a = start + 2 * math.pi * k / n
        rot = _rz(a)
        out.append((rot @ np.array([(r_in + r_out) / 2, 0.0, (z0 + z1) / 2]), rot, (r_out - r_in, width, z1 - z0)))
    return out


def _aligned(lo, hi) -> tuple:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return ((lo + hi) / 2, np.eye(3), tuple(hi - lo))


def _span(p0, p1, thick: float, wide: float, azimuth: float) -> tuple:
    """A bar from (radius, height) ``p0`` to ``p1`` in the plane ``azimuth`` rad round from x: ``thick`` across that
    plane's normal ... sizes (along, across, deep)."""
    (r0, z0), (r1, z1) = p0, p1
    length = math.hypot(r1 - r0, z1 - z0)
    slope = math.atan2(z1 - z0, r1 - r0)
    rot = _rz(azimuth) @ _ry(-slope)
    return (_rz(azimuth) @ np.array([(r0 + r1) / 2, 0.0, (z0 + z1) / 2]), rot, (length, wide, thick))


def fan_boxes() -> list:
    """The fan's box colliders in the Fan's frame: the rear grille's rim, the wires that collide, the support's wall
    and lip, the pillar."""
    out = _ring(RIM_IN, RIM_OUT, RIM_Z - RIM_T, RIM_Z, 24)
    for k in range(N_SPOKES):
        out.append(_span(DISH[0], DISH[1], 0.004, 0.004, 2 * math.pi * (k + 0.5) / N_SPOKES))
    out.append(_aligned((SUPPORT_WALL[0], -SUPPORT_HALF, RIM_Z - RIM_T), (SUPPORT_WALL[1], SUPPORT_HALF, SUPPORT_LIP[1])))
    out.append(_aligned((SUPPORT_REACH, -SUPPORT_HALF, SUPPORT_LIP[0]), (SUPPORT_WALL[1], SUPPORT_HALF, SUPPORT_LIP[1])))
    out.append(_aligned(*PILLAR))
    return out


def fan_wires() -> list:
    """The rear grille's wires that are only drawn."""
    return [_span(DISH[0], DISH[1], 0.0022, 0.0022, 2 * math.pi * k / N_WIRES) for k in range(N_WIRES)]


def rotor_boxes() -> list:
    """The cross pin, in the Rotor's frame (the shaft itself is a cylinder)."""
    return [_aligned((-PIN_L / 2, -PIN_D / 2, PIN_Z - PIN_D / 2), (PIN_L / 2, PIN_D / 2, PIN_Z + PIN_D / 2))]


def hub_boxes() -> list:
    """The hub in the Blade's frame: the bore's wall above the slot, the hub's shell, and the back boss either side
    of the slot."""
    out = _ring(BORE_R, HUB_WALL, SLOT_D, HUB_H, 8) + _ring(HUB_WALL, HUB_R, 0.0, HUB_H, 8)
    half = SLOT_W / 2
    for s in (1.0, -1.0):
        lo_y, hi_y = (BORE_R, HUB_WALL) if s > 0 else (-HUB_WALL, -BORE_R)
        out.append(_aligned((-HUB_WALL, lo_y, 0.0), (HUB_WALL, hi_y, SLOT_D)))              # behind the shaft
        lo_y, hi_y = (half, BORE_R) if s > 0 else (-BORE_R, -half)
        out.append(_aligned((BORE_R, lo_y, 0.0), (HUB_WALL, hi_y, SLOT_D)))                 # beside it
        out.append(_aligned((-HUB_WALL, lo_y, 0.0), (-BORE_R, hi_y, SLOT_D)))
    return out


def blade_azimuth(k: int) -> float:
    """Where blade ``k`` (0, 1, 2) leaves the hub, rad round from the Blade's x."""
    return math.radians(30.0 + 120.0 * k)


def blade_plate(k: int, seg: int) -> tuple:
    """One piece of blade ``k``: (centre, rotation, size). The plate's x runs out along the blade, its y along the
    chord toward the leading edge, its z is the plate's normal (toward the fan's front)."""
    r0, r1, chord, pitch = BLADE_SEGS[seg]
    rot = _rz(blade_azimuth(k)) @ _rx(math.radians(pitch))
    return (_rz(blade_azimuth(k)) @ np.array([(r0 + r1) / 2, 0.0, HUB_H / 2]), rot, (r1 - r0, chord, BLADE_T))


def blade_boxes() -> list:
    return [blade_plate(k, s) for k in range(3) for s in range(len(BLADE_SEGS))]


def cap_boxes() -> list:
    """The blade cap in the Cap's frame, an eight-sided knob 38 mm across its flats: the wall round its bore, and
    its solid top."""
    return (_ring(CAP_MOUTH_R, CAP_R, 0.0, CAP_MOUTH_D, 8) + _ring(CAP_BORE_R, CAP_R, CAP_MOUTH_D, CAP_BORE_D, 8)
            + _ring(0.0, CAP_R, CAP_BORE_D, CAP_H, 8))


def grille_boxes() -> list:
    """The front grille's box colliders in the Grille's frame: its rim and every third rib."""
    out = _ring(RIM_IN, RIM_OUT, 0.0, GRILLE_T, 24) + _ring(RIM_IN, RIM_IN + BAND_T, GRILLE_T, BAND_Z, 24)
    for k in range(N_RIB_SOLID):
        a = 2 * math.pi * (k + 0.5) / N_RIB_SOLID
        out.append(_span(DOME[0], DOME[1], 0.006, 0.004, a))
        out.append(_span(DOME[1], DOME[2], 0.006, 0.004, a))
    return out


def grille_ribs() -> list:
    out = []
    for k in range(N_RIBS):
        a = 2 * math.pi * k / N_RIBS
        out.append(_span(DOME[0], DOME[1], 0.004, 0.0025, a))
        out.append(_span(DOME[1], DOME[2], 0.004, 0.0025, a))
    return out


def clip_boxes() -> list:
    """A clip in its own frame, shut: the stem and the hook."""
    (sx0, sz0), (sx1, sz1) = CLIP_STEM
    (hx0, hz0), (hx1, hz1) = CLIP_HOOK
    return [_aligned((sx0, -CLIP_HALF, sz0), (sx1, CLIP_HALF, sz1)), _aligned((hx0, -CLIP_HALF, hz0), (hx1, CLIP_HALF, hz1))]


def clip_hinge(k: int) -> np.ndarray:
    """Clip ``k``'s hinge in the Fan's frame."""
    a = math.radians(CLIP_AZ[k])
    return np.array([CLIP_HINGE_R * math.cos(a), CLIP_HINGE_R * math.sin(a), CLIP_HINGE_Z])


def clip_rot(k: int, deg: float) -> np.ndarray:
    """Clip ``k``'s rotation in the Fan's frame when it stands ``deg`` degrees out from shut."""
    return _rz(math.radians(CLIP_AZ[k])) @ _ry(math.radians(deg))


def clip_deg(k: int, fan_quat, clip_quat) -> float:
    """How far clip ``k`` stands out from shut (degrees)."""
    rel = clip_rot(k, 0.0).T @ _rotation(fan_quat).T @ _rotation(clip_quat)
    return math.degrees(math.atan2(rel[0, 2], rel[2, 2]))


def clip_spring_target(deg: float) -> float:
    """Where the clip's over-centre spring pulls it (degrees): out past open when it stands beyond half way, in past
    shut when it has been pressed in past half way."""
    return CLIP_OPEN + CLIP_PRE if deg > CLIP_MID else -CLIP_PRE


def blade_fit(blade_pos, blade_quat, fan_pos, fan_quat, rotor_quat) -> dict:
    """How the blade sits on the shaft. ``off``: the hub's middle from the shaft's axis (m); ``tilt``: degrees between
    bore and shaft; ``up``: the hub's back face above where it sits with the pin at the bottom of the slot (m);
    ``slot``: degrees between the slot and the pin (0..90)."""
    p = to_local(place(blade_pos, blade_quat, (0.0, 0.0, HUB_H / 2)), fan_pos, fan_quat)
    back = to_local(blade_pos, fan_pos, fan_quat)
    slot = abs(_yaw_about(rotor_quat, blade_quat)) % 180.0
    return {"off": float(math.hypot(p[0], p[1])), "tilt": tilt_between(blade_quat, fan_quat),
            "up": float(back[2] - HUB_SEAT), "slot": float(min(slot, 180.0 - slot))}


def cap_fit(cap_pos, cap_quat, fan_pos, fan_quat) -> dict:
    """How the cap sits on the shaft's end. ``off``: its bore's mouth from the shaft's axis (m); ``tilt``: degrees;
    ``up``: its lower face above where it meets a seated hub (m)."""
    p = to_local(cap_pos, fan_pos, fan_quat)
    return {"off": float(math.hypot(p[0], p[1])), "tilt": tilt_between(cap_quat, fan_quat), "up": float(p[2] - CAP_SEAT)}


def grille_fit(grille_pos, grille_quat, fan_pos, fan_quat) -> dict:
    """How the front grille sits on the rear one. ``off``: its middle from the fan's axis (m); ``tilt``: degrees;
    ``up``: the highest point of its rim plane above the rear rim (m); ``low``: the lowest; ``turn``: degrees its
    bottom edge is turned from the support; ``under``: its bottom edge is under the support's lip."""
    rot = _rotation(grille_quat)
    heights = []
    for k in range(24):
        a = 2 * math.pi * k / 24
        q = to_local(np.asarray(grille_pos, float) + rot @ np.array([RIM_OUT * math.cos(a), RIM_OUT * math.sin(a), 0.0]), fan_pos, fan_quat)
        heights.append(q[2] - GRILLE_SEAT)
    c = to_local(grille_pos, fan_pos, fan_quat)
    edge = to_local(np.asarray(grille_pos, float) + rot @ np.array([RIM_OUT, 0.0, GRILLE_T]), fan_pos, fan_quat)
    turn = abs(_yaw_about(fan_quat, grille_quat))
    under = bool(abs(edge[1]) <= SUPPORT_HALF and edge[2] <= SUPPORT_LIP[0] + 0.0005 and edge[0] >= SUPPORT_REACH + 0.002)
    return {"off": float(math.hypot(c[0], c[1])), "tilt": tilt_between(grille_quat, fan_quat), "up": float(max(heights)),
            "low": float(min(heights)), "turn": float(turn), "under": under}


def thread(cap_pos, cap_quat, fan_pos, fan_quat) -> dict:
    """Where the cap is on the shaft: ``s`` its lower face's height in the fan's frame (m), ``off`` and ``tilt`` as
    in :func:`cap_fit`, ``catches`` whether the shaft's end is far enough into its bore, near enough to its axis and
    upright enough for the thread to catch."""
    p = to_local(cap_pos, fan_pos, fan_quat)
    off, tilt = float(math.hypot(p[0], p[1])), tilt_between(cap_quat, fan_quat)
    catches = bool(p[2] <= SHAFT_TIP - CATCH_IN and p[2] >= CAP_SEAT - 0.003 and off <= CATCH_OFF and tilt <= CATCH_TILT)
    return {"s": float(p[2]), "off": off, "tilt": tilt, "catches": catches}


def thread_target(s0: float, turns: float) -> float:
    """Where the thread draws the cap's lower face (height in the fan's frame, m) after ``turns`` turns of the cap on
    the shaft, anticlockwise seen from the front, from where it caught at ``s0``. It stops ``SNUG`` past the face of
    a seated hub: the cap is then tight."""
    return max(s0 - PITCH * turns, CAP_SEAT - SNUG)


# -- the parts -----------------------------------------------------------------------------------------------------------

class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Fan(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 1.5


@dataclass(frozen=True)
class Rotor(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = ROTOR_MASS
    fan: str = "fan"


@dataclass(frozen=True)
class Blade(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = BLADE_MASS


@dataclass(frozen=True)
class Cap(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = CAP_MASS
    rotor: str = "rotor"
    fan: str = "fan"


@dataclass(frozen=True)
class Grille(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = GRILLE_MASS


@dataclass(frozen=True)
class Clip(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = CLIP_MASS
    fan: str = "fan"
    index: int = 0


@dataclass(frozen=True)
class KitState(ObjectState):
    """A kit object as the simulator has it; ``pos`` is its own origin (see each class)."""

    joints: dict = field(default_factory=dict)


def _material(sim_utils, sf, df, combine="average"):
    try:
        return sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, friction_combine_mode=combine)
    except TypeError:
        return sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df)


# -- authoring helpers -----------------------------------------------------------------------------------------------

class _Author:
    def __init__(self, prim_path, translation, orientation):
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        self.sim_utils, self.Gf, self.UsdGeom = sim_utils, Gf, UsdGeom
        self.stage = get_current_stage()
        self.path = prim_path
        double = UsdGeom.XformOp.PrecisionDouble
        root = UsdGeom.Xform.Define(self.stage, prim_path)
        root.AddTranslateOp(precision=double).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
        x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
        root.AddOrientOp(precision=double).Set(Gf.Quatd(w, x, y, z))
        self.root = root.GetPrim()
        self.looks, self.mats, self.n = {}, {}, 0

    def look(self, color, rough=0.6, metal=0.0) -> str:
        from yam_arena.station import linear_rgb

        key = (tuple(color), rough, metal)
        if key not in self.looks:
            path = f"{self.path}/Looks/v{len(self.looks)}"
            vis = self.sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=rough, metallic=metal)
            vis.func(path, vis)
            self.looks[key] = path
        return self.looks[key]

    def mat(self, key, sf, df, combine="average") -> str:
        if key not in self.mats:
            path = f"{self.path}/Looks/phys_{key}"
            pm = _material(self.sim_utils, sf, df, combine)
            pm.func(path, pm)
            self.mats[key] = path
        return self.mats[key]

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, fixed=False, inertia=None):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        if fixed:
            rigid.CreateKinematicEnabledAttr(True)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(mass))
        if inertia:
            api.CreateCenterOfMassAttr(self.Gf.Vec3f(*(float(v) for v in inertia[0])))
            api.CreateDiagonalInertiaAttr(self.Gf.Vec3f(*(float(v) for v in inertia[1])))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        rb.CreateSleepThresholdAttr(0.0)
        rb.CreateStabilizationThresholdAttr(0.0)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def _collide(self, prim, mat, offset, patch):
        from isaaclab.sim.utils import bind_physics_material
        from pxr import PhysxSchema, UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(offset)
        cp.CreateRestOffsetAttr(0.0)
        if patch:
            cp.CreateTorsionalPatchRadiusAttr(float(patch))
            cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
        bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)

    def bar(self, piece, look, mat=None, offset=0.0006, patch=0.0):
        """A box given as (centre, rotation, size): drawn if ``look`` is given, a box collider if ``mat`` is (thin
        plates cannot be cooked as convex hulls for the GPU pipeline, and then collide with nothing)."""
        from isaaclab.sim.utils import bind_visual_material

        centre, rot, size = piece
        self.n += 1
        cube = self.UsdGeom.Cube.Define(self.stage, f"{self.path}/b{self.n}")
        cube.CreateSizeAttr(1.0)
        x, y, z, w = _quat(rot)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in centre)))
        cube.AddOrientOp().Set(self.Gf.Quatf(float(w), float(x), float(y), float(z)))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*(float(v) for v in size)))
        prim = cube.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        else:
            self.UsdGeom.Imageable(prim).MakeInvisible()
        if mat:
            self._collide(prim, mat, offset, patch)
        return prim

    def cylinder(self, z0, z1, radius, look, mat=None, offset=0.0006, patch=0.0, centre=(0.0, 0.0), axis="Z"):
        """A round piece about the part's z (or lying along ``axis``) between two heights."""
        from isaaclab.sim.utils import bind_visual_material

        self.n += 1
        cyl = self.UsdGeom.Cylinder.Define(self.stage, f"{self.path}/c{self.n}")
        cyl.CreateAxisAttr(axis)
        cyl.CreateHeightAttr(float(z1 - z0))
        cyl.CreateRadiusAttr(float(radius))
        if axis == "Z":
            cyl.AddTranslateOp().Set(self.Gf.Vec3d(float(centre[0]), float(centre[1]), float(z0 + z1) / 2))
        else:                                           # along x: z0, z1 are x positions; centre is (y, z)
            cyl.AddTranslateOp().Set(self.Gf.Vec3d(float(z0 + z1) / 2, float(centre[0]), float(centre[1])))
        prim = cyl.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        else:
            self.UsdGeom.Imageable(prim).MakeInvisible()
        if mat:
            self._collide(prim, mat, offset, patch)
        return prim


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _start(cfg, prim_path, translation, orientation, fixed=False, inertia=None, iterations=32, ang_damp=0.1):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, fixed=fixed, inertia=inertia, ang_damp=ang_damp)
    return spec, a, a.mat("part", *FRICTION)


def _spawn_fan(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True)
    white, grey = a.look(WHITE, rough=0.5), a.look(GREY, rough=0.5)
    a.cylinder(0.0, HOUSING_H, HOUSING_R, white, m)
    a.cylinder(HOUSING_H, BOSS_Z, BOSS_R + 0.012, grey, m)                    # the locknut over the rear grille's hub
    boxes = fan_boxes()
    for piece in boxes[:24]:
        a.bar(piece, white, m)
    for piece in boxes[24:24 + N_SPOKES]:
        a.bar(piece, None, m)
    for piece in boxes[24 + N_SPOKES:24 + N_SPOKES + 2]:
        a.bar(piece, grey, m)
    a.bar(boxes[-1], white, m)
    wire = a.look(STEEL, rough=0.4, metal=0.6)
    for piece in fan_wires():
        a.bar(piece, wire)
    a.cylinder(BASE_X, BASE_X + BASE_T, BASE_R, white, m, centre=(0.0, BASE_R), axis="X")   # the base, on its edge
    return a.root


def _spawn_rotor(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation,
                        inertia=((0.0, 0.0, 0.06), (ROTOR_INERTIA, ROTOR_INERTIA, ROTOR_INERTIA)), ang_damp=ROTOR_SPIN_DAMP)
    steel = a.look(STEEL, rough=0.3, metal=0.9)
    top = SHAFT_TIP - THREADED - 0.001 * len(TAPER)
    a.cylinder(SHAFT_Z0, top, SHAFT_R, steel, m)
    for i, r in enumerate(TAPER):
        a.cylinder(top + 0.001 * i, top + 0.001 * (i + 1), r, steel, m)
    a.cylinder(SHAFT_TIP - THREADED, SHAFT_TIP, THREAD_R, a.look(DARK, rough=0.5, metal=0.6), m)
    for piece in rotor_boxes():
        a.bar(piece, steel, m)
    return a.root


def _spawn_blade(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation)
    blue = a.look(BLUE, rough=0.45)
    for piece in hub_boxes():
        a.bar(piece, blue, m)
    for piece in blade_boxes():
        a.bar(piece, blue, m, patch=PAD_PATCH)
    return a.root


def _spawn_cap(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation)
    grip = a.mat("grip", 1.0, 0.9)
    white = a.look(WHITE, rough=0.5)
    for piece in cap_boxes():
        a.bar(piece, white, grip, patch=PAD_PATCH)
    return a.root


def _spawn_grille(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation)
    white, grey = a.look(WHITE, rough=0.5), a.look(GREY, rough=0.5)
    boxes = grille_boxes()
    for piece in boxes[:24]:
        a.bar(piece, white, m, patch=PAD_PATCH)
    for piece in boxes[24:48]:
        a.bar(piece, white, m)
    for piece in boxes[48:]:
        a.bar(piece, None, m)
    for piece in grille_ribs():
        a.bar(piece, white)
    a.cylinder(BADGE_Z[0], BADGE_Z[1], BADGE_R, grey, m)
    a.bar(_aligned((RIM_IN - 0.004, -0.010, GRILLE_T), (RIM_OUT, 0.010, GRILLE_T + 0.0015)), a.look(RED))   # marks its bottom edge
    return a.root


def _spawn_clip(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=16)
    grey = a.look(DARK, rough=0.5)
    for piece in clip_boxes():
        a.bar(piece, grey, m)
    return a.root


# -- joints --------------------------------------------------------------------------------------------------------------

def _joints() -> list:
    """(name, kind, first body, second body, spec) for every joint the specs call for."""
    out = []
    for n, s in _SPECS.items():
        if isinstance(s, Rotor) and all(k in _PATHS for k in (n, s.fan)):
            out.append((f"{n}_bearing", "bearing", s.fan, n, s))
        if isinstance(s, Clip) and all(k in _PATHS for k in (n, s.fan)):
            out.append((f"{n}_hinge", "clip", s.fan, n, s))
        if isinstance(s, Cap) and all(k in _PATHS for k in (n, s.rotor)):
            out.append((f"{n}_thread", "thread", s.rotor, n, s))
    return out


def _joint_path(j) -> str:
    return f"{_PATHS[j[2]]}/{j[0]}"


_X_TO_Z = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)        # (w, x, y, z): turns a joint's x onto the body's z


def _author(j) -> None:
    """Author one joint: disabled first, its frames set, enabled last (the thread stays off until it catches)."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdPhysics

    name, kind, first, second, spec = j
    stage = get_current_stage()
    path = _joint_path(j)
    if kind == "bearing":
        jt = UsdPhysics.RevoluteJoint.Define(stage, path)
        jt.CreateJointEnabledAttr().Set(False)
        jt.CreateAxisAttr().Set("Z")
        anchor0, rot0, rot1 = (0.0, 0.0, 0.0), Gf.Quatf(1.0), Gf.Quatf(1.0)
    elif kind == "clip":
        jt = UsdPhysics.RevoluteJoint.Define(stage, path)
        jt.CreateJointEnabledAttr().Set(False)
        jt.CreateAxisAttr().Set("Y")
        x, y, z, w = _quat(clip_rot(spec.index, 0.0))
        anchor0, rot0, rot1 = clip_hinge(spec.index), Gf.Quatf(w, x, y, z), Gf.Quatf(1.0)
        jt.CreateLowerLimitAttr().Set(-1.0)
        jt.CreateUpperLimitAttr().Set(float(CLIP_OPEN + 3.0))
    else:
        jt = UsdPhysics.Joint.Define(stage, path)               # a plain joint: limits lock or bound each axis
        jt.CreateJointEnabledAttr().Set(False)
        anchor0, rot0, rot1 = (0.0, 0.0, 0.0), Gf.Quatf(*_X_TO_Z), Gf.Quatf(*_X_TO_Z)
    jt.CreateBody0Rel().SetTargets([_PATHS[first]])
    jt.CreateBody1Rel().SetTargets([_PATHS[second]])
    jt.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in anchor0)))
    jt.CreateLocalRot0Attr().Set(rot0)
    jt.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    jt.CreateLocalRot1Attr().Set(rot1)
    jt.CreateExcludeFromArticulationAttr().Set(True)
    jt.CreateCollisionEnabledAttr().Set(False)
    prim = jt.GetPrim()
    if kind == "bearing":
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")       # the motor's drag: holds the shaft still up to ROTOR_DRAG
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(float(ROTOR_DAMP) * math.pi / 180.0)   # per degree a second; stiffer chatters
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(float(ROTOR_DRAG))
    elif kind == "clip":
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")       # the over-centre spring (USD angular drives take degrees)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(CLIP_K))
        drive.CreateDampingAttr().Set(float(CLIP_K) * 0.02)
        drive.CreateTargetPositionAttr().Set(float(CLIP_OPEN + CLIP_PRE))
        drive.CreateMaxForceAttr().Set(1.0)
    else:
        for dof in ("transY", "transZ", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(prim, dof)          # low > high: locked
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        limit = UsdPhysics.LimitAPI.Apply(prim, "transX")
        limit.CreateLowAttr(float(CAP_SEAT - 0.004))
        limit.CreateHighAttr(float(SHAFT_TIP + 0.002))
        turn = UsdPhysics.DriveAPI.Apply(prim, "rotX")            # thread friction (per degree a second)
        turn.CreateTypeAttr().Set("force")
        turn.CreateStiffnessAttr().Set(0.0)
        turn.CreateDampingAttr().Set(float(THREAD_DRAG) * math.pi / 180.0)
        turn.CreateTargetVelocityAttr().Set(0.0)
        turn.CreateMaxForceAttr().Set(1.0e6)
        slide = UsdPhysics.DriveAPI.Apply(prim, "transX")         # the thread's draw along the shaft
        slide.CreateTypeAttr().Set("force")
        slide.CreateStiffnessAttr().Set(float(THREAD_K))
        slide.CreateDampingAttr().Set(float(THREAD_DAMP))
        slide.CreateTargetPositionAttr().Set(float(SHAFT_TIP))
        slide.CreateMaxForceAttr().Set(float(THREAD_PULL))
    PhysxSchema.PhysxJointAPI.Apply(prim)
    if kind != "thread":
        jt.CreateJointEnabledAttr().Set(True)       # last write, after the frames


def _rigid_object(spec, spawn):
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class KitSpawnerCfg(RigidObjectSpawnerCfg):
        func: object = clone(spawn)
        kit_name: str = ""

    _SPECS[spec.name] = spec
    _PATHS.pop(spec.name, None)
    quat = tuple(float(v) for v in getattr(spec, "quat", (0.0, 0.0, 0.0, 1.0)))
    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=KitSpawnerCfg(kit_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


_SPAWNERS = {Fan: _spawn_fan, Rotor: _spawn_rotor, Blade: _spawn_blade, Cap: _spawn_cap, Grille: _spawn_grille,
             Clip: _spawn_clip}


def _local_bounds(spec) -> tuple:
    if isinstance(spec, Fan):
        return np.array([-RIM_OUT, -RIM_OUT, 0.0]), np.array([BASE_X + BASE_T, RIM_OUT, 2 * BASE_R])
    if isinstance(spec, Rotor):
        return np.array([-PIN_L / 2, -PIN_L / 2, SHAFT_Z0]), np.array([PIN_L / 2, PIN_L / 2, SHAFT_TIP])
    if isinstance(spec, Blade):
        return np.array([-BLADE_R, -BLADE_R, -0.010]), np.array([BLADE_R, BLADE_R, HUB_H + 0.010])
    if isinstance(spec, Cap):
        return np.array([-CAP_R, -CAP_R, 0.0]), np.array([CAP_R, CAP_R, CAP_H])
    if isinstance(spec, Grille):
        return np.array([-RIM_OUT, -RIM_OUT, 0.0]), np.array([RIM_OUT, RIM_OUT, BADGE_Z[1]])
    return np.array([-0.009, -CLIP_HALF, 0.0]), np.array([0.004, CLIP_HALF, 0.017])


# -- run time ----------------------------------------------------------------------------------------------------------

def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    if hasattr(x, "numpy") and not hasattr(x, "detach"):
        return np.asarray(x.numpy(), dtype=np.float64)
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _origin(environment) -> np.ndarray:
    return _np(environment.unwrapped.scene.env_origins[0])


def _pose(environment, name: str) -> tuple:
    pose = _np(environment.unwrapped.scene[name].data.root_link_pose_w)[0]
    return pose[:3] - _origin(environment), pose[3:7]


def _disarm() -> None:
    """Switch every joint off (before a reset puts the parts back)."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for j in _joints():
        prim = stage.GetPrimAtPath(_joint_path(j))
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)


_MECH: dict = {}                       # each joint's state, followed step by step


def _arm(environment) -> None:
    """Join the parts as the reset left them: the shaft in its bearing, the clips on their hinges standing open, the
    thread off."""
    _MECH.clear()
    for j in _joints():
        _author(j)
        name, kind = j[0], j[1]
        if kind == "clip":
            _MECH[name] = {"deg": CLIP_OPEN, "target": CLIP_OPEN + CLIP_PRE}
        elif kind == "thread":
            _MECH[name] = {"on": False, "s0": 0.0, "turns": 0.0, "phi": None, "target": SHAFT_TIP, "armed": True,
                           "caught": 0, "came_off": 0}
    _follow(environment)


def _attr(j, api: str, dof: str, which: str):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    prim = get_current_stage().GetPrimAtPath(_joint_path(j))
    if api == "enabled":
        return UsdPhysics.Joint(prim).GetJointEnabledAttr()
    drive = UsdPhysics.DriveAPI(prim, dof)
    return drive.GetTargetPositionAttr() if which == "target" else drive.GetDampingAttr()


def _follow(environment) -> None:
    """Follow each clip and the thread through this step, and set their springs for the next."""
    for j in _joints():
        name, kind, first, second, spec = j
        m = _MECH.get(name)
        if m is None:
            continue
        try:
            if kind == "clip":
                _, fq = _pose(environment, first)
                _, cq = _pose(environment, second)
                m["deg"] = clip_deg(spec.index, fq, cq)
                target = clip_spring_target(m["deg"])
                if target != m["target"]:
                    m["target"] = target
                    _attr(j, "drive", "angular", "target").Set(float(target))
            elif kind == "thread":
                fp, fq = _pose(environment, spec.fan)
                _, rq = _pose(environment, first)
                cp, cq = _pose(environment, second)
                g = thread(cp, cq, fp, fq)
                phi = _yaw_about(rq, cq)                       # the cap's turn on the shaft, anticlockwise from the front
                if m["on"]:
                    m["turns"] += ((phi - m["phi"] + 180.0) % 360.0 - 180.0) / 360.0
                    m["phi"] = phi
                    free = m["s0"] - PITCH * m["turns"]
                    if free >= SHAFT_TIP - CATCH_IN + 0.0003:          # unscrewed off the end of the thread
                        _attr(j, "enabled", "", "").Set(False)
                        m.update(on=False, armed=False, came_off=m["came_off"] + 1)
                    else:
                        target = thread_target(m["s0"], m["turns"])
                        if abs(target - m["target"]) > 1e-6:
                            m["target"] = target
                            _attr(j, "drive", "transX", "target").Set(float(target))
                elif not m["armed"]:                           # a cap just unscrewed has to leave the shaft first
                    m["armed"] = bool(g["s"] > SHAFT_TIP + 0.002 or g["off"] > 0.010)
                elif g["catches"]:
                    _attr(j, "drive", "transX", "target").Set(float(g["s"]))
                    _attr(j, "enabled", "", "").Set(True)
                    m.update(on=True, s0=g["s"], turns=0.0, phi=phi, target=g["s"], caught=m["caught"] + 1)
        except Exception as err:                     # noqa: BLE001
            _warn(f"{name}: {err!r}")


def mechanism(name: str) -> dict:
    """A joint's state as followed since the last reset: a clip's ``deg``; the thread's ``on``, ``turns`` ..."""
    return dict(_MECH.get(name, {}))


def _kit_state(environment, spec) -> KitState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    lo, hi = _local_bounds(spec)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = corners @ _rotation(quat).T + pos
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0))


def _warn(text: str) -> None:
    if text not in _WARNED:
        _WARNED.append(text)
        print(f"[fankit] {text}", file=sys.stderr, flush=True)


_WARNED: list = []


def finger_points(environment) -> list:
    """The grasp point of each gripper, left then right (world, relative to the env origin)."""
    from yam_arena.constants import TCP_OFFSET

    base = environment.unwrapped
    origin = _origin(environment)
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        pts.append(pose[:3] + _rotation(pose[3:7]) @ np.array(TCP_OFFSET, float) - origin)
    return pts


def gripper_points(environment) -> list:
    """Points on each gripper that could touch something held between or beside its fingers, left gripper then right
    (world, relative to the env origin): along both fingers' inner and outer faces, from their tips back to the palm,
    and the palm's middle. The fingers' opening is read from their joints."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _origin(environment)
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        rot = _rotation(pose[3:7])
        tcp = pose[:3] + rot @ np.array(TCP_OFFSET, float) - origin
        q, names = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        pts = [tcp - PALM_BACK * rot[:, 2]]
        for side, joint in ((-1.0, "joint7"), (1.0, "joint8")):
            half = abs(float(q[names.index(joint)]))
            for depth in (PAD_TIP_PAST_TCP, 0.0, -PALM_BACK / 2, -PALM_BACK):
                for face in (0.0, FINGER_THICK):
                    pts.append(tcp + side * (half + face) * rot[:, 0] + depth * rot[:, 2])
        out.append(pts)
    return out


# -- hooking yam_arena -------------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    _STEP.append(fn)
    return fn


def on_reset(fn):
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_fankit_patched", False):
        return
    stock_object, stock_states = scene.task_object, scene.object_states

    def task_object(spec):
        for kind, spawn in _SPAWNERS.items():
            if isinstance(spec, kind):
                return _rigid_object(spec, spawn)
        return stock_object(spec)

    def object_states(environment, names):
        mine = {n for n in names if isinstance(_SPECS.get(n), tuple(_SPAWNERS))}
        plain = [n for n in names if n not in mine]
        out = stock_states(environment, plain) if plain else {}
        for n in mine:
            out[n] = _kit_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._fankit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_fankit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        _follow(self._env)
        for fn in _STEP:
            fn(self._env)
        return stock_step(self, *a, **k)

    def reset(self, *a, **k):
        # The joints have to be whole before the reset's settling steps and first picture: switch them off, let the
        # environment put the parts back, then author them afresh from where the parts are.
        base, saved, had = None, None, False
        try:
            self._ensure_sim(None)
            base = self._env.unwrapped
            env_reset = base.reset
            saved, had = vars(base).get("reset"), "reset" in vars(base)

            def reset_then_arm(*aa, **kk):
                _disarm()
                result = env_reset(*aa, **kk)
                _arm(self._env)
                return result

            base.reset = reset_then_arm
        except Exception as err:                     # noqa: BLE001
            print(f"[fankit] the joints were not re-made around the reset: {err!r}", file=sys.stderr, flush=True)
            base = None
        try:
            obs = stock_reset(self, *a, **k)
        finally:
            if base is not None:
                if had:
                    base.reset = saved
                else:
                    del base.reset
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._fankit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        stock_exec = loader.exec_module

        def exec_module(module):
            stock_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena (now for modules already imported, otherwise when they are). Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
