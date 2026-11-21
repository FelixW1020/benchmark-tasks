"""doorkit: a full-size bench section of a dry-van trailer's rear door with one cam-action lock rod, for yam_arena
task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from doorkit import Door, Rod, Handle, Keeper, on_reset, on_step

Frames. The fixture's own frame has its origin on the table under the lock rod's axis, in the plane of the door's
outer face: x runs into the door (away from the robot), y to the robot's left, z up. The rod's axis is ``ROD_OFF``
in front of the door's face (at -x).

- :class:`Door`: everything that is fixed: the door panel section, the sill under it, the neighbouring door's edge
  with its own (locked) rod, the cam's keeper on the sill, the pipe guides and the hasp. ``pos`` is the fixture's origin.
- :class:`Rod`: the lock rod (a pipe) with its cam at the lower end and the handle's hub, one rigid part turning in
  the fixture about the pipe's axis. ``pos`` is the point of the axis on the table. Locked, its axes are the
  fixture's; unlocking turns it about -z (``theta`` degrees, + is unlocked).
- :class:`Handle`: the handle, pinned to the rod's hub so that it can be lifted (``lift`` degrees, + is up). ``pos``
  is the pin. Locked and down, its axes are the fixture's and it lies along -y.
- :class:`Keeper`: the hasp's flip keeper, a plate on a pin that stands out of the hasp above the handle's seat.
  ``pos`` is the pin. Down (``flip`` 0) it hangs over the seated handle; flipped up and over (``FLIP_OPEN``) it leans
  on its stop.

What holds what is contact: the handle sits in the hasp's seat behind a lip, the keeper hangs over it, the cam's two
prongs go either side of the keeper pin on the sill. One thing is a force law set by the kit rather than a part: the
door seal's push-back, a torsion spring on the rod over its last ``SEAL_FROM`` degrees (see :func:`seal_torque`).

All shapes are convex pieces built from point sets in NumPy when the task file is read (``*_pieces()``), so the same
numbers serve the scene, the scoring and the offline checks. Isaac Sim is reached only when the scene is built.

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

__all__ = ["Door", "Handle", "Keeper", "KitState", "Rod", "angles", "cam_engaged", "door_pieces", "env",
           "finger_points", "grip_pose", "gripper_points", "handle_pieces", "handle_pose", "install", "keeper_pieces",
           "keeper_pose", "on_reset", "on_step", "place", "rod_pieces", "rod_pose", "seal_torque", "to_local", "turn_z",
           "write_pose"]

IN = 0.0254
# -- the lock rod (m) ---------------------------------------------------------------------------------------------------
PIPE_R = 1.050 * IN / 2       # 3/4 in Schedule 40 pipe is 1.050 in outside (the pipe size the kit's listing names)
ROD_OFF = 0.035               # the pipe's axis in front of the door's face (an estimate)
ROD_Z = (0.018, 0.335)        # the cut length of pipe on the bench fixture, above the table
ROD_MASS = 0.55               # pipe, cam and hub (an estimate: 0.32 m of pipe at 1.7 kg/m)
# -- the handle ----------------------------------------------------------------------------------------------------------
Z_H = 0.160                   # the handle's middle above the table on this fixture
HUB_R = 0.030                 # the handle's pin, out from the pipe's axis (an estimate)
HANDLE_L = 14.0 * IN          # the handle's length (the kit's listing: "Handle length: 14\"")
BAR_BACK = 0.012              # ... of which this much is behind the pin
BAR_T = 0.008                 # the bar's thickness, across the door's face (an estimate)
BAR_H = 0.025                 # the bar's height (an estimate)
TIP_R = HUB_R + HANDLE_L - BAR_BACK      # the handle's tip, out from the pipe's axis
HANDLE_MASS = 0.55            # kg (an estimate: the bar's volume in steel)
LIFT_MAX = 30.0               # degrees the handle lifts on its pin (an estimate)
# -- the hasp ------------------------------------------------------------------------------------------------------------
HASP_W = 2.0 * IN             # the kit's listing: "Hasp dimensions: 5-1/8\" tall x 2\" wide"
HASP_TALL = 5.125 * IN
HASP_AT = 0.300               # the hasp's middle, out from the pipe's axis along the locked handle (an estimate)
HASP_Z = (Z_H - 0.050, Z_H - 0.050 + HASP_TALL)
BASE_X = -(ROD_OFF - BAR_T / 2)          # the hasp's face that the locked handle lies against
SLOT = 0.012                  # the seat's width, from that face to the lip (an estimate): the bar has 4 mm of play
LIP_T = 0.005                 # the lip's thickness (an estimate)
LIP_H = 0.012                 # the lip's height above the seat (an estimate)
SEAT_Z = Z_H - BAR_H / 2 - 0.0005   # the seat's top: half a millimetre under the bar when it is down on its pin's stop
SEAT_T = 0.006
# -- the hasp's flip keeper ----------------------------------------------------------------------------------------------
KEEP_UP = 0.060               # the keeper's pin above the handle's middle (an estimate)
KEEP_L = 0.045                # the keeper's length below its pin when it hangs (an estimate)
KEEP_STUB = 0.008             # ... and above it
KEEP_W = 0.028                # its width (an estimate)
KEEP_T = 0.006                # its thickness (an estimate)
TAB = 0.014                   # the finger tab at its free end stands this far out toward the robot (an estimate)
KEEP_MASS = 0.040             # kg (an estimate)
FLIP_OPEN = 205.0             # degrees: flipped up and over, the keeper leans on its stop 25 degrees past upright
# -- the cam and its keeper pin on the sill ---------------------------------------------------------------------------------
CAM_Z = (0.025, 0.035)        # the cam plate above the table
PRONG_IN = (0.020, 0.027)     # the cam's inner prong, radii from the pipe's axis (estimates)
PRONG_OUT = (0.035, 0.042)    # ... and its outer prong
PRONG_ARC = (-45.0, -5.0)     # the prongs run over these angles ahead of the handle (degrees, the way it locks)
PLATE_ARC = (-5.0, 40.0)      # the cam's plate behind them
PIN_AT = (0.031, -12.0)       # the keeper pin: radius from the pipe's axis, and angle from the locked handle's line
PIN_R = 0.003
PIN_Z = (0.012, 0.048)
CAM_IN = 25.0                 # the cam has hold of the pin when the rod is within this many degrees of locked
# -- the door's seal, as a spring on the rod ------------------------------------------------------------------------------
SEAL_FROM = 15.0              # degrees from locked at which the cam starts to draw the door onto its seal (an estimate)
SEAL_MAX = 0.40               # N m pushing the rod back at locked (an estimate; a loaded trailer door takes far more)
ROD_DRAG = 0.03               # N m of friction in the pipe guides (an estimate)
LIFT_DRAG = 0.01              # N m in the handle's pin (an estimate)
KEEP_DRAG = 0.0005            # N m in the keeper's pin (an estimate)
KEEP_DAMP = 3.0               # 1/s: the keeper's swing dies away in its pin (an estimate)
# -- the fixture -----------------------------------------------------------------------------------------------------------
PANEL_T = 0.018               # plywood
PANEL_Y = (-0.420, 0.070)     # the door section, from its cut edge to the seam between the two doors
PANEL_Z = (0.060, 0.340)
SILL_Z = 0.060                # the sill under the doors
SILL_X = 0.060
NEXT_Y = (0.076, 0.250)       # the neighbouring door's edge
NEXT_ROD_Y = 0.140            # ... and its own lock rod, locked, with its handle lying away along +y
GUIDE_Z = ((0.075, 0.113), (0.262, 0.300))     # the two pipe guides (the kit's "retainers", 1-1/2 in wide)

STEEL = (0.62, 0.64, 0.66)
ZINC = (0.74, 0.76, 0.78)
PLY = (0.86, 0.84, 0.78)
SILL = (0.33, 0.34, 0.36)
RED = (0.72, 0.10, 0.08)
FRICTION = (0.5, 0.4)         # steel on steel and on the grippers' rubber (estimates)
PAD_PATCH = 0.004
PALM_BACK = 0.065
FINGER_THICK = 0.012
N_SEG = 16


# -- shapes (pure): every piece is a set of points whose convex hull is the piece -----------------------------------------

def _ring(r: float, z: float, n: int = N_SEG, centre=(0.0, 0.0)) -> np.ndarray:
    t = 2 * math.pi * np.arange(n) / n
    return np.column_stack([centre[0] + r * np.cos(t), centre[1] + r * np.sin(t), np.full(n, z)])


def _solid(r: float, z0: float, z1: float, centre=(0.0, 0.0)) -> np.ndarray:
    return np.vstack([_ring(r, z0, centre=centre), _ring(r, z1, centre=centre)])


def _slab(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def _along(deg: float) -> np.ndarray:
    """The level direction ``deg`` degrees round from the locked handle's line (-y), the way the handle unlocks."""
    a = math.radians(deg)
    return np.array([-math.sin(a), -math.cos(a), 0.0])


def _sector(r0: float, r1: float, a0: float, a1: float, z0: float, z1: float, step: float = 10.0) -> list:
    """A flat ring sector about the pipe's axis as convex pieces (angles as in :func:`_along`)."""
    out = []
    n = max(1, int(math.ceil((a1 - a0) / step)))
    for k in range(n):
        b0, b1 = a0 + (a1 - a0) * k / n, a0 + (a1 - a0) * (k + 1) / n
        pts = [r * _along(b) + (0.0, 0.0, z) for b in (b0, b1) for r in (r0, r1) for z in (z0, z1)]
        out.append(np.array(pts))
    return out


def door_pieces() -> list:
    """The fixture as (points, colour, solid) pieces in its own frame."""
    out = [(_slab((0.0, PANEL_Y[0], PANEL_Z[0]), (PANEL_T, PANEL_Y[1], PANEL_Z[1])), PLY, True),          # the door
           (_slab((0.0, NEXT_Y[0], PANEL_Z[0]), (PANEL_T, NEXT_Y[1], PANEL_Z[1])), PLY, True),            # its neighbour
           (_slab((0.0, PANEL_Y[0], 0.0), (SILL_X, NEXT_Y[1], SILL_Z)), SILL, True),                      # the sill
           (_solid(PIPE_R, ROD_Z[0], ROD_Z[1], centre=(-ROD_OFF, NEXT_ROD_Y)), ZINC, True),                 # the next rod
           (_slab((-ROD_OFF - BAR_T / 2, NEXT_ROD_Y + 0.016, Z_H - BAR_H / 2),
                  (-ROD_OFF + BAR_T / 2, NEXT_Y[1], Z_H + BAR_H / 2)), ZINC, True)]                        # ... its handle
    for z0, z1 in GUIDE_Z:                                   # pipe guides: drawn, not solid (the rod turns inside them)
        for y in (0.0, NEXT_ROD_Y):
            out.append((_slab((-ROD_OFF - PIPE_R - 0.004, y - 0.020, z0), (0.0, y + 0.020, z1)), STEEL, False))
            out.append((_slab((-0.004, y - 2.0 * IN, z0), (0.0, y + 2.0 * IN, z1)), STEEL, False))
    # the hasp: its body, the seat the handle drops into, the lip in front of the seat, and the keeper's pin
    y0, y1 = -HASP_AT - HASP_W / 2, -HASP_AT + HASP_W / 2
    out += [(_slab((BASE_X, y0, HASP_Z[0]), (0.0, y1, HASP_Z[1])), ZINC, True),
            (_slab((BASE_X - SLOT - LIP_T, y0, SEAT_Z - SEAT_T), (BASE_X, y1, SEAT_Z)), ZINC, True),
            (_slab((BASE_X - SLOT - LIP_T, y0, SEAT_Z), (BASE_X - SLOT, y1, SEAT_Z + LIP_H)), ZINC, True),
            (_slab((-ROD_OFF - KEEP_T / 2 - 0.004, -HASP_AT - 0.004, Z_H + KEEP_UP - 0.004),
                   (BASE_X, -HASP_AT + 0.004, Z_H + KEEP_UP + 0.004)), STEEL, False)]
    # the cam's keeper on the sill: a pin between two plates
    pin = -ROD_OFF * np.array([1.0, 0.0, 0.0]) + PIN_AT[0] * _along(PIN_AT[1])
    out.append((_solid(PIN_R, PIN_Z[0], PIN_Z[1], centre=(pin[0], pin[1])), STEEL, True))
    for z0, z1 in ((PIN_Z[0] - 0.004, PIN_Z[0]), (PIN_Z[1], PIN_Z[1] + 0.004)):
        out.append((_slab((pin[0] - 0.008, pin[1] - 0.010, z0), (0.0, pin[1] + 0.010, z1)), STEEL, True))
    return out


def rod_pieces() -> list:
    """The rod in its own frame (origin: the point of its axis on the table): pipe, cam, hub."""
    out = [(_solid(PIPE_R, ROD_Z[0], ROD_Z[1]), ZINC)]
    out += [(p, STEEL) for p in _sector(PIPE_R - 0.002, PRONG_OUT[1], *PLATE_ARC, *CAM_Z)]
    out += [(p, STEEL) for p in _sector(*PRONG_IN, *PRONG_ARC, *CAM_Z)]
    out += [(p, STEEL) for p in _sector(*PRONG_OUT, *PRONG_ARC, *CAM_Z)]
    for z0, z1 in ((Z_H + BAR_H / 2 + 0.001, Z_H + BAR_H / 2 + 0.007), (Z_H - BAR_H / 2 - 0.007, Z_H - BAR_H / 2 - 0.001)):
        out.append((_slab((-0.009, -HUB_R - 0.012, z0), (0.009, 0.0, z1)), STEEL))                # the hub's two ears
    return out


def handle_pieces() -> list:
    """The handle in its own frame (origin: its pin; it lies along -y)."""
    return [(_slab((-BAR_T / 2, -(HANDLE_L - BAR_BACK), -BAR_H / 2), (BAR_T / 2, BAR_BACK, BAR_H / 2)), RED)]


def keeper_pieces() -> list:
    """The flip keeper in its own frame (origin: its pin; it hangs along -z), with its finger tab toward the robot."""
    return [(_slab((-KEEP_T / 2, -KEEP_W / 2, -KEEP_L), (KEEP_T / 2, KEEP_W / 2, KEEP_STUB)), STEEL),
            (_slab((-KEEP_T / 2 - TAB, -KEEP_W / 2, -KEEP_L), (-KEEP_T / 2, KEEP_W / 2, -KEEP_L + 0.006)), STEEL)]


def seal_torque(theta_deg: float) -> float:
    """The door seal's push-back on the rod (N m, the way that unlocks it) at ``theta_deg`` from locked."""
    t = float(theta_deg)
    if t >= SEAL_FROM:
        return 0.0
    return SEAL_MAX * (1.0 - max(t, -3.0) / SEAL_FROM)


def cam_engaged(theta_deg: float) -> bool:
    """The keeper pin on the sill is between the cam's two prongs."""
    return float(theta_deg) <= CAM_IN
def turn_z(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn of ``angle`` rad about z."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def hull_faces(points: np.ndarray, eps: float = 1e-7) -> list:
    """The faces of the convex hull of ``points`` (4 to about 40 of them): each a list of point indices, wound
    counter-clockwise seen from outside. Brute force over triples; coplanar triangles are merged into one polygon."""
    p = np.asarray(points, float)
    n = len(p)
    centre = p.mean(0)
    planes: dict = {}
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            e1 = p[j] - p[i]
            for k in range(j + 1, n):
                nrm = np.cross(e1, p[k] - p[i])
                size = float(np.linalg.norm(nrm))
                if size < 1e-12:
                    continue
                nrm = nrm / size
                d = (p - p[i]) @ nrm
                if d.max() > eps and d.min() < -eps:
                    continue
                if d.max() > eps:
                    nrm = -nrm
                key = tuple(np.round(np.r_[nrm, nrm @ p[i]], 5))
                planes.setdefault(key, (nrm, set()))[1].update((i, j, k))
    faces = []
    for nrm, idx in planes.values():
        idx = sorted(idx)
        c = p[idx].mean(0)
        u = p[idx[0]] - c
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        ang = [math.atan2((p[m] - c) @ v, (p[m] - c) @ u) for m in idx]
        order = [m for _, m in sorted(zip(ang, idx))]
        if nrm @ (c - centre) < 0:
            order = order[::-1]
        faces.append(order)
    return faces



class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Door(_Part):
    """Everything fixed: door section, sill, neighbouring door edge, pipe guides, hasp, the cam's keeper pin."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 20.0


@dataclass(frozen=True)
class Rod(_Part):
    """The lock rod with its cam and hub. ``door`` names the :class:`Door` it turns in."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = ROD_MASS
    door: str = "door"


@dataclass(frozen=True)
class Handle(_Part):
    """The handle. ``rod`` names the :class:`Rod` whose hub it is pinned to."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = HANDLE_MASS
    rod: str = "rod"


@dataclass(frozen=True)
class Keeper(_Part):
    """The hasp's flip keeper. ``door`` names the :class:`Door` whose hasp it is pinned to."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = KEEP_MASS
    door: str = "door"


@dataclass(frozen=True)
class KitState(ObjectState):
    """A kit object as the simulator has it; ``pos`` is its own origin (see each class)."""

    joints: dict = field(default_factory=dict)


# -- geometry (pure) -------------------------------------------------------------------------------------------------

def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def place(pos, quat, local) -> np.ndarray:
    """A point (or points) given in a part's own frame, in the world."""
    return np.asarray(pos, float) + np.asarray(local, float) @ _rotation(quat).T


def to_local(point, pos, quat) -> np.ndarray:
    """A world point (or points) in a part's own frame."""
    return (np.asarray(point, float) - np.asarray(pos, float)) @ _rotation(quat)




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


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


ROD_AT = np.array([-ROD_OFF, 0.0, 0.0])                       # in the fixture's frame
PIN_AT_ROD = np.array([0.0, -HUB_R, Z_H])                     # the handle's pin in the rod's frame
KEEP_AT = np.array([-ROD_OFF, -HASP_AT, Z_H + KEEP_UP])       # the keeper's pin in the fixture's frame


def rod_pose(door_pos, theta_deg: float = 0.0) -> tuple:
    """The rod's position and quaternion when it stands ``theta_deg`` from locked."""
    return np.asarray(door_pos, float) + ROD_AT, turn_z(-math.radians(theta_deg))


def handle_pose(door_pos, theta_deg: float = 0.0, lift_deg: float = 0.0) -> tuple:
    """The handle's position (its pin) and quaternion."""
    rz = _rz(-math.radians(theta_deg))
    return np.asarray(door_pos, float) + ROD_AT + rz @ PIN_AT_ROD, _quat(rz @ _rx(-math.radians(lift_deg)))


def keeper_pose(door_pos, flip_deg: float = 0.0) -> tuple:
    """The keeper's position (its pin) and quaternion when it is flipped ``flip_deg`` up from hanging."""
    return np.asarray(door_pos, float) + KEEP_AT, _quat(_rx(math.radians(flip_deg)))


def angles(rod_quat, handle_quat, keeper_quat) -> tuple:
    """(theta, lift, flip) in degrees from the three parts' quaternions (the fixture square to the world)."""
    r, h, k = _rotation(rod_quat), _rotation(handle_quat), _rotation(keeper_quat)
    theta = -math.degrees(math.atan2(r[1, 0], r[0, 0]))
    rel = r.T @ h
    lift = -math.degrees(math.atan2(rel[2, 1], rel[1, 1]))
    flip = math.degrees(math.atan2(k[2, 1], k[1, 1]))
    if flip < -90.0:
        flip += 360.0
    return theta, lift, flip


def grip_pose(door_pos, theta_deg: float, lift_deg: float, lever: float) -> tuple:
    """A point on the handle's centre line ``lever`` out from the pipe's axis (measured along the handle), with the
    direction into the door's face there (level), the handle's own up, and the direction along the handle to its tip."""
    pos, quat = handle_pose(door_pos, theta_deg, lift_deg)
    rot = _rotation(quat)
    return pos + rot @ np.array([0.0, -(lever - HUB_R), 0.0]), rot[:, 0], rot[:, 2], -rot[:, 1]


def seat_gap(theta_deg: float) -> float:
    """How far the bar's door-side face is from the hasp's face, at the hasp's middle (m)."""
    return HASP_AT * math.sin(math.radians(theta_deg))


def seat_rise(lift_deg: float) -> float:
    """How far the bar's underside is above the seat, at the hasp's middle (m)."""
    return (HASP_AT - HUB_R) * math.sin(math.radians(lift_deg)) + (Z_H - BAR_H / 2 - SEAT_Z)


def segment_distance(points, a, b) -> float:
    """The smallest distance from any of ``points`` to the segment a-b."""
    p, a, b = np.atleast_2d(np.asarray(points, float)), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t[:, None] * ab), axis=1).min())

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
        self.looks, self.mats = {}, {}

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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, fixed=False):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        if fixed:
            rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def hull(self, path, points, look, mat=None, offset=0.0008, patch=0.0):
        """A convex piece: drawn as its hull's faces, and a convex-hull collider if ``mat`` is given."""
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from pxr import PhysxSchema, UsdPhysics, Vt

        pts = np.asarray(points, float)
        faces = hull_faces(pts)
        mesh = self.UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([self.Gf.Vec3f(*(float(v) for v in p)) for p in pts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in faces for i in f]))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        prim = mesh.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(offset)
            cp.CreateRestOffsetAttr(0.0)
            if patch:                                # soft foam meets a pad over a patch, which resists twisting
                cp.CreateTorsionalPatchRadiusAttr(float(patch))
                cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def pieces(self, stem, pieces, mat, rough=0.6, patch=0.0):
        for n, piece in enumerate(pieces):
            pts, color = piece[0], piece[1]
            solid = len(piece) < 3 or piece[2]
            self.hull(f"{stem}_{n}", pts, self.look(color, rough=rough), mat if solid else None, patch=patch)



# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _start(cfg, prim_path, translation, orientation, iterations=None, fixed=False, friction=FRICTION):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, fixed=fixed)
    return spec, a, a.mat("part", *friction)


def _spawn_door(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True)
    a.pieces(f"{prim_path}/p", door_pieces(), m, rough=0.5)
    return a.root


def _spawn_rod(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", rod_pieces(), m, rough=0.35)
    return a.root


def _spawn_handle(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32, friction=(0.8, 0.7))
    a.pieces(f"{prim_path}/p", handle_pieces(), m, rough=0.4, patch=PAD_PATCH)
    return a.root


def _spawn_keeper(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.body(prim_path, float(spec.mass), ang_damp=KEEP_DAMP, iterations=32)      # (again, for the keeper's own damping)
    a.pieces(f"{prim_path}/p", keeper_pieces(), m, rough=0.35)
    return a.root


# -- joints ------------------------------------------------------------------------------------------------------------

def _joints() -> dict:
    """name -> (body0, body1, the joint's place in body0, axis, lower, upper, drag, bodies collide, zero): the pin's
    angle is 0 when body1 is turned ``zero`` degrees about the axis from body0's own attitude."""
    out = {}
    for n, s in _SPECS.items():
        if n not in _PATHS:
            continue
        if isinstance(s, Rod) and s.door in _PATHS:
            out[f"turn_{n}"] = (s.door, n, ROD_AT, "Z", -110.0, 3.0, ROD_DRAG, True, 0.0)
        elif isinstance(s, Handle) and s.rod in _PATHS:
            out[f"lift_{n}"] = (s.rod, n, PIN_AT_ROD, "X", -LIFT_MAX, 0.0, LIFT_DRAG, False, 0.0)
        elif isinstance(s, Keeper) and s.door in _PATHS:
            # the keeper's travel is more than half a turn: its pin's zero is put at mid-travel, so that the angle
            # the solver measures never wraps
            out[f"flip_{n}"] = (s.door, n, KEEP_AT, "X", -FLIP_OPEN / 2, FLIP_OPEN / 2, KEEP_DRAG, True, FLIP_OPEN / 2)
    return out


def _joint_path(name: str, body: str) -> str:
    return f"{_PATHS[body]}/{name}_joint"


def _author(name: str, spec: tuple) -> None:
    """Author one pin: disabled first, its frames set, enabled last."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    body0, body1, at, axis, lower, upper, drag, collide, zero = spec
    half = math.radians(zero) / 2
    turn = Gf.Quatf(math.cos(half), *(math.sin(half) * float(axis == k) for k in "XYZ"))
    stage = get_current_stage()
    joint = UsdPhysics.RevoluteJoint.Define(stage, _joint_path(name, body0))
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateBody0Rel().SetTargets([_PATHS[body0]])
    joint.CreateBody1Rel().SetTargets([_PATHS[body1]])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in at)))
    joint.CreateLocalRot0Attr().Set(turn)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(bool(collide))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(float(lower))
    joint.CreateUpperLimitAttr().Set(float(upper))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")     # dry friction: a stiff damper that tops out
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(1.0)
    drive.CreateMaxForceAttr().Set(float(drag))
    drive.CreateTargetVelocityAttr().Set(0.0)
    joint.CreateJointEnabledAttr().Set(True)       # last write, after the frames


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



_SPAWNERS = {Door: _spawn_door, Rod: _spawn_rod, Handle: _spawn_handle, Keeper: _spawn_keeper}
_BOUNDS: dict = {}


def _points_of(spec) -> np.ndarray:
    if isinstance(spec, Door):
        return np.vstack([p[0] for p in door_pieces()])
    if isinstance(spec, Rod):
        return np.vstack([p[0] for p in rod_pieces()])
    if isinstance(spec, Handle):
        return np.vstack([p[0] for p in handle_pieces()])
    return np.vstack([p[0] for p in keeper_pieces()])


# -- run time ----------------------------------------------------------------------------------------------------------

def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    if hasattr(x, "numpy") and not hasattr(x, "detach"):
        return np.asarray(x.numpy(), dtype=np.float64)
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _origin(environment) -> np.ndarray:
    return _np(environment.unwrapped.scene.env_origins[0])


def write_pose(environment, name: str, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
    """Put a loose object at ``pos`` (relative to the env origin), ``quat_xyzw``, at rest."""
    import torch

    asset = environment.unwrapped.scene[name]
    dev = _t(asset.data.root_link_pose_w).device
    pose = torch.tensor([[*(np.asarray(pos, float) + _origin(environment)), *quat_xyzw]], dtype=torch.float32, device=dev)
    vel = torch.zeros((1, 6), dtype=torch.float32, device=dev)
    for stem, kw in (("write_root_pose_to_sim", {"root_pose": pose}), ("write_root_velocity_to_sim", {"root_velocity": vel})):
        fn = getattr(asset, f"{stem}_index", None)
        if fn is not None:
            fn(env_ids=[0], **kw)
        else:
            getattr(asset, stem)(*kw.values())


def _pose(environment, name: str):
    pose = _np(environment.unwrapped.scene[name].data.root_link_pose_w)[0]
    return pose[:3], pose[3:7]


def _disarm() -> None:
    """Switch every joint off (before a reset puts the parts back)."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for name, spec in _joints().items():
        prim = stage.GetPrimAtPath(_joint_path(name, spec[0]))
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)


def _arm(environment) -> None:
    """Join the parts as the reset left them: the rod to the fixture, the handle to the rod's hub, the keeper to the hasp."""
    for name, spec in _joints().items():
        _author(name, spec)


def _rel(environment, name: str) -> tuple:
    pos, quat = _pose(environment, name)
    return pos - _origin(environment), quat


def _wrench(environment, name: str, torque_world, lift: float = 0.0) -> None:
    """Hold a torque (N m) and an upward force (N) on a body until the next call."""
    import torch

    asset = environment.unwrapped.scene[name]
    t = torch.tensor([[[float(v) for v in torque_world]]], dtype=torch.float32, device=asset.device)
    f = torch.tensor([[[0.0, 0.0, float(lift)]]], dtype=torch.float32, device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=t, is_global=True)



def _rule(environment) -> None:
    """The kit's one force law, applied for the coming control step: the door seal's push-back on each rod."""
    for rod in (n for n, s in _SPECS.items() if isinstance(s, Rod) and n in _PATHS):
        try:
            _, rq = _rel(environment, rod)
        except Exception:                            # noqa: BLE001
            continue
        r = _rotation(rq)
        theta = -math.degrees(math.atan2(r[1, 0], r[0, 0]))
        _wrench(environment, rod, (0.0, 0.0, -seal_torque(theta)))


def _local_bounds(spec) -> tuple:
    if spec.name not in _BOUNDS:
        pts = _points_of(spec)
        _BOUNDS[spec.name] = (pts.min(0), pts.max(0))
    return _BOUNDS[spec.name]


def _kit_state(environment, spec) -> KitState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    lo, hi = _local_bounds(spec)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = corners @ _rotation(quat).T + pos
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0))


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
    if getattr(scene, "_doorkit_patched", False):
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
    scene._doorkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_doorkit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        _rule(self._env)
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
                _rule(self._env)
                return result

            base.reset = reset_then_arm
        except Exception as err:                     # noqa: BLE001
            print(f"[doorkit] joints not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._doorkit_patched = True


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
