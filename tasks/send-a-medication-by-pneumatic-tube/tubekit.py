"""tubekit: a hospital pneumatic tube carrier with a swivel lid, a prescription vial, and the send tube of a
bottom-loading tube station, for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from tubekit import Carrier, Lid, Station, Vial, on_reset, on_step

Frames (every part starts with its axes along the world's unless a ``quat`` is given: x away from the robot, y left,
z up).

- :class:`Carrier`: the carrier's body without its lid, a loose rigid part. ``pos`` is the middle of the underside of
  its closed end; its z runs along its axis to the mouth.
- :class:`Lid`: the swivel lid, a loose rigid part joined to its ``carrier`` by a pin beside the mouth's rim, parallel
  to the carrier's axis. ``pos`` is that pin, level with the mouth's rim. Shut, its axes are the carrier's; it opens by
  turning about its -z. A torque law on the pin is its catch: a bump to push past, then it is held shut.
- :class:`Vial`: a prescription vial with its cap on, loose. ``pos`` is the middle of its underside; z is its axis.
- :class:`Station`: the part of a bottom-loading station that a person loads, fixed: the send tube's lower end with
  its entry collar, and the cabinet above it. ``pos`` is the middle of the collar's lower edge, the lowest point of
  the station; the send tube runs up its +z.

The station takes a carrier that is pushed far enough up its send tube (:func:`take_depth`): from then on the kit
holds the carrier there, as the station's own catch does. That is a rule of this kit, not contact physics.

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

__all__ = ["Carrier", "KitState", "Lid", "Station", "Vial", "carrier_pieces", "env", "finger_points",
           "gripper_points", "install", "lid_angle", "lid_pieces", "lid_pose", "lid_torque", "on_reset", "on_step",
           "place", "station_pieces", "to_local", "turn_z", "vial_in_carrier", "vial_pieces", "write_pose"]

# -- the carrier (m): the short swivel-lid carrier for a 110 mm line ------------------------------------------------------
LENGTH = 0.228                # overall (a maker's listing: "approx ext. 104mmD x 228mmL")
RING_D = 0.104                # across the two sealing rings (the same listing)
BORE = 0.080                  # inside diameter (a hospital design guide's table: 110 Short, internal diameter 80)
WALL = 0.003                  # wall thickness (an estimate); the body between the rings is then 86 mm across
BODY_R = BORE / 2 + WALL
RING_R = 0.0505               # the rings' radius as they collide: their hook-and-loop pile gives about 1.5 mm
#                               (an estimate), so the rigid ring here is 101 mm across, not 104
RING_W = 0.014                # each ring's width along the carrier (an estimate)
RING_AT = (0.032, 0.186)      # the rings' middles above the closed end (estimates)
BASE_T = 0.008                # the closed end's thickness (an estimate)
LID_T = 0.012                 # the lid's thickness (an estimate)
MOUTH_Z = LENGTH - LID_T      # the mouth's rim above the closed end
LID_R = 0.044                 # the lid's radius (an estimate)
PIN_R = 0.047                 # the pin's distance from the carrier's axis, on a lug outside the wall (an estimate)
PIN = (0.0, PIN_R, MOUTH_Z)   # ... on the carrier's +y side; the lid swings open toward -x
LID_MAX = 80.0                # degrees the lid swings open to its stop (an estimate)
CARRIER_MASS = 0.26           # kg without the lid (an estimate from the shell's volume; no listing gives a weight)
LID_MASS = 0.03               # (an estimate)
# The lid's catch, as a torque on the pin (N m in the opening sense; - shuts). It is a snap: the lid has to be pushed
# past a bump just before it is shut, and is then held.
T_SHUT = 0.060                # holds the shut lid: 0.7 N at its far edge, 91 mm from the pin, opens it (an estimate)
T_BUMP = 0.060                # the bump to push past on the way shut, taken to be as high as the hold (an estimate)
BUMP_AT, BUMP_FROM = 5.0, 10.0   # degrees: the bump is highest at BUMP_AT and begins at BUMP_FROM (estimates)
LID_DRAG = 0.012              # N m of friction in the pin (an estimate): the lid stays where it is left
SHUT_AT = 3.0                 # the lid is secured within this many degrees of shut (the catch holds it there)

# -- the vial: a 13 dram amber prescription vial with a child-resistant cap ------------------------------------------------
VIAL_R = 0.016                # a distributor's listing: "Width: 32 mm"
VIAL_BODY = 0.065             # "Height (without cap): 65 mm"
VIAL_H = 0.069                # "Height (with cap): 69 mm"
CAP_R = 0.0185                # the cap stands out round the vial (an estimate)
CAP_H = 0.016                 # the cap's skirt (an estimate)
VIAL_MASS = 0.030             # kg with thirty tablets (an estimate)

# -- the station's send tube ---------------------------------------------------------------------------------------------
TUBE_OD = 0.110               # the line's nominal size
TUBE_WALL = 0.0023            # (an estimate: a maker's table gives 2.3 mm for its 124 mm tube)
TUBE_R = TUBE_OD / 2 - TUBE_WALL
COLLAR_H = 0.020              # the entry collar below the tube: it widens downward (an estimate)
COLLAR_R = 0.066              # ... to this radius at its lower edge (an estimate)
TUBE_UP = 0.300               # how far the tube is modelled above the collar
TAKE = 0.070                  # the station takes a carrier whose top is this far above the collar's lower edge (an
#                               estimate): its upper ring is then 1.5 cm up the tube
TAKE_OFF = 0.008              # ... and no further than this from the tube's axis
CAB_UP = 0.040                # the cabinet's underside above the collar's lower edge (an estimate)
CAB_H = 0.260                 # the cabinet's height (an estimate)
CAB_FRONT = 0.075             # its front face, toward the robot from the tube's axis (an estimate)
CAB_BACK = 0.140              # its back, on the wall (an estimate)
CAB_HALF = 0.110              # half its width (an estimate)

POLY = (0.62, 0.78, 0.90)     # clear blue polycarbonate
RING = (0.10, 0.10, 0.11)
LID_BLUE = (0.10, 0.30, 0.62)
AMBER = (0.78, 0.42, 0.08)
CAP_WHITE = (0.95, 0.95, 0.93)
CAB_GREY = (0.82, 0.83, 0.84)
TUBE_GREY = (0.55, 0.56, 0.58)
GREEN = (0.10, 0.70, 0.25)
FRICTION = (0.7, 0.6)         # polycarbonate and polypropylene against the grippers' rubber and each other (estimates)
PAD_PATCH = 0.004             # radius of the patch over which a rubber pad meets the carrier's body (an estimate)
PALM_BACK = 0.065
FINGER_THICK = 0.012
N_SEG = 16


# -- shapes (pure): every piece is a set of points whose convex hull is the piece -----------------------------------------

def _ring(r: float, z: float, n: int = N_SEG, centre=(0.0, 0.0)) -> np.ndarray:
    t = 2 * math.pi * np.arange(n) / n
    return np.column_stack([centre[0] + r * np.cos(t), centre[1] + r * np.sin(t), np.full(n, z)])


def _solid(r: float, z0: float, z1: float, centre=(0.0, 0.0), r1: float | None = None) -> np.ndarray:
    """A solid round piece from ``z0`` to ``z1`` (radius ``r`` below, ``r1`` above)."""
    return np.vstack([_ring(r, z0, centre=centre), _ring(r if r1 is None else r1, z1, centre=centre)])


def _shell(r_in: float, r_out: float, z0: float, z1: float, r_in1: float | None = None, r_out1: float | None = None) -> list:
    """A hollow round wall as ``N_SEG`` convex pieces (inner and outer radius at ``z0``; at ``z1`` if they differ)."""
    out = []
    t = 2 * math.pi * np.arange(N_SEG + 1) / N_SEG
    for a0, a1 in zip(t, t[1:]):
        pts = []
        for a in (a0, a1):
            c, s = math.cos(a), math.sin(a)
            for z, ri, ro in ((z0, r_in, r_out), (z1, r_in if r_in1 is None else r_in1, r_out if r_out1 is None else r_out1)):
                pts += [[ri * c, ri * s, z], [ro * c, ro * s, z]]
        out.append(np.array(pts))
    return out


def _slab(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def carrier_pieces() -> list:
    """The carrier's body as (points, colour) pieces in its own frame: closed end, wall, two rings."""
    out = [(_solid(BODY_R, 0.0, BASE_T), POLY)]
    out += [(p, POLY) for p in _shell(BORE / 2, BODY_R, BASE_T, MOUTH_Z)]
    for z in RING_AT:
        out += [(p, RING) for p in _shell(BODY_R - 0.0005, RING_R, z - RING_W / 2, z + RING_W / 2)]
    out.append((_slab((-0.007, BODY_R - 0.001, MOUTH_Z - 0.012), (0.007, PIN_R + 0.006, MOUTH_Z)), POLY))   # the pin's lug
    return out


def lid_pieces() -> list:
    """The lid in its own frame (origin on the pin, level with the mouth's rim; shut, its middle is at -y)."""
    return [(_solid(LID_R, 0.0, LID_T, centre=(0.0, -PIN_R)), LID_BLUE),
            (_solid(0.006, 0.0, LID_T + 0.002), RING)]


def vial_pieces() -> list:
    return [(_solid(VIAL_R, 0.0, VIAL_H - CAP_H), AMBER), (_solid(CAP_R, VIAL_H - CAP_H, VIAL_H), CAP_WHITE)]


def station_pieces() -> list:
    """The station as (points, colour, solid) pieces in its own frame (origin: the middle of the collar's lower edge)."""
    out = [(p, TUBE_GREY, True) for p in _shell(COLLAR_R, COLLAR_R + 0.004, 0.0, COLLAR_H, r_in1=TUBE_R, r_out1=TUBE_OD / 2)]
    out += [(p, TUBE_GREY, True) for p in _shell(TUBE_R, TUBE_OD / 2, COLLAR_H, TUBE_UP)]
    lo, hi, t = CAB_UP, CAB_UP + CAB_H, 0.004
    r = TUBE_OD / 2 + 0.001
    out += [(_slab((-CAB_FRONT, -CAB_HALF, lo), (-CAB_FRONT + t, CAB_HALF, hi)), CAB_GREY, True),      # front
            (_slab((-CAB_FRONT, -CAB_HALF, lo), (CAB_BACK, -CAB_HALF + t, hi)), CAB_GREY, True),      # sides
            (_slab((-CAB_FRONT, CAB_HALF - t, lo), (CAB_BACK, CAB_HALF, hi)), CAB_GREY, True),
            (_slab((-CAB_FRONT, -CAB_HALF, hi - t), (CAB_BACK, CAB_HALF, hi)), CAB_GREY, True),       # top
            (_slab((-CAB_FRONT, -CAB_HALF, lo), (-r, CAB_HALF, lo + t)), CAB_GREY, True),             # underside, round the tube
            (_slab((r, -CAB_HALF, lo), (CAB_BACK, CAB_HALF, lo + t)), CAB_GREY, True),
            (_slab((-r, -CAB_HALF, lo), (r, -r, lo + t)), CAB_GREY, True),
            (_slab((-r, r, lo), (r, CAB_HALF, lo + t)), CAB_GREY, True),
            (_slab((-CAB_FRONT - 0.002, -0.045, lo + 0.150), (-CAB_FRONT, 0.045, lo + 0.215)), (0.08, 0.10, 0.12), False),   # display
            (_slab((-CAB_FRONT - 0.003, 0.055, lo + 0.170), (-CAB_FRONT, 0.075, lo + 0.190)), GREEN, False)]               # ready light
    return out


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
class Station(_Part):
    """The station's send tube and cabinet, fixed. ``pos``: the middle of the entry collar's lower edge."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 8.0


@dataclass(frozen=True)
class Carrier(_Part):
    """The carrier's body. ``pos``: the middle of the underside of its closed end."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = CARRIER_MASS


@dataclass(frozen=True)
class Lid(_Part):
    """The swivel lid. ``carrier`` names the :class:`Carrier` it is pinned to. ``pos``: the pin."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = LID_MASS
    carrier: str = "carrier"


@dataclass(frozen=True)
class Vial(_Part):
    """The prescription vial. ``pos``: the middle of its underside."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = VIAL_MASS


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


def lid_pose(carrier_pos, carrier_quat, angle_deg: float = 0.0) -> tuple:
    """Where the lid is when it stands ``angle_deg`` open: its position and quaternion in the world."""
    rot = _rotation(carrier_quat) @ _rz(-math.radians(angle_deg))
    return place(carrier_pos, carrier_quat, PIN), _quat(rot)


def lid_angle(carrier_quat, lid_quat) -> float:
    """How far the lid stands open (degrees; 0 shut, + open)."""
    rel = _rotation(carrier_quat).T @ _rotation(lid_quat)
    return -math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def lid_torque(angle_deg: float) -> float:
    """The catch: the torque on the pin at a lid angle (N m in the opening sense, about the carrier's -z; - shuts)."""
    a = float(angle_deg)
    if a <= SHUT_AT:
        return -T_SHUT
    if a <= BUMP_AT:
        return -T_SHUT + (T_SHUT + T_BUMP) * (a - SHUT_AT) / (BUMP_AT - SHUT_AT)
    if a <= BUMP_FROM:
        return T_BUMP * (BUMP_FROM - a) / (BUMP_FROM - BUMP_AT)
    return 0.0


def lid_middle(angle_deg: float) -> np.ndarray:
    """The middle of the lid's top, in the carrier's frame, when it stands ``angle_deg`` open."""
    a = math.radians(angle_deg)
    return np.array([-PIN_R * math.sin(a), PIN_R - PIN_R * math.cos(a), LENGTH])


def vial_in_carrier(vial_pos, vial_quat, carrier_pos, carrier_quat) -> bool:
    """The whole vial is inside the carrier: both its ends within the bore and between the closed end and the rim."""
    for end in (0.0, VIAL_H):
        p = to_local(place(vial_pos, vial_quat, (0.0, 0.0, end)), carrier_pos, carrier_quat)
        if math.hypot(p[0], p[1]) > BORE / 2 or not (BASE_T - 0.003 <= p[2] <= MOUTH_Z + 0.001):
            return False
    return True


def carrier_top(carrier_pos, carrier_quat) -> np.ndarray:
    """The middle of the carrier's top (the lid's outer face when shut), in the world."""
    return place(carrier_pos, carrier_quat, (0.0, 0.0, LENGTH))


def insertion(carrier_pos, carrier_quat, station_pos) -> tuple:
    """How far the carrier's top is above the collar's lower edge (m, negative below it), how far it is from the send
    tube's axis, and the carrier's tilt from the tube's axis (rad)."""
    top = carrier_top(carrier_pos, carrier_quat) - np.asarray(station_pos, float)
    axis = _rotation(carrier_quat)[:, 2]
    return float(top[2]), float(math.hypot(top[0], top[1])), float(math.acos(max(-1.0, min(1.0, axis[2]))))


def taken(carrier_pos, carrier_quat, station_pos) -> bool:
    """The station's rule: it takes a carrier whose top is ``TAKE`` up the send tube and on its axis."""
    up, off, tilt = insertion(carrier_pos, carrier_quat, station_pos)
    return up >= TAKE and off <= TAKE_OFF and tilt <= math.radians(6.0)


def held_pose(station_pos) -> tuple:
    """Where the station holds a carrier it has taken: the carrier's position and quaternion."""
    s = np.asarray(station_pos, float)
    return s + np.array([0.0, 0.0, TAKE + 0.005 - LENGTH]), (0.0, 0.0, 0.0, 1.0)


def body_distance(points, carrier_pos, carrier_quat) -> float:
    """The smallest distance (m, at least 0) from any of ``points`` (world) to the carrier, taken as a cylinder of
    the rings' radius over its whole length."""
    q = to_local(np.atleast_2d(np.asarray(points, float)), carrier_pos, carrier_quat)
    dr = np.maximum(0.0, np.hypot(q[:, 0], q[:, 1]) - RING_D / 2)
    dz = np.maximum(0.0, np.maximum(-q[:, 2], q[:, 2] - LENGTH))
    return float(np.hypot(dr, dz).min())


def segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))



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


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned
_HELD: dict = {}                       # carriers the station has taken: name -> (station name, lift in N)


def _start(cfg, prim_path, translation, orientation, iterations=None, fixed=False, friction=FRICTION):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, fixed=fixed)
    return spec, a, a.mat("part", *friction)


def _spawn_station(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True, friction=(0.3, 0.25))
    a.pieces(f"{prim_path}/p", station_pieces(), m, rough=0.5)
    return a.root


def _spawn_carrier(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", carrier_pieces(), m, rough=0.25, patch=PAD_PATCH)
    return a.root


def _spawn_lid(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", lid_pieces(), m, rough=0.4)
    return a.root


def _spawn_vial(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", vial_pieces(), m, rough=0.3, patch=PAD_PATCH)
    return a.root


# -- joints ------------------------------------------------------------------------------------------------------------

def _joints() -> dict:
    """name -> (the carrier's name, the lid's name)."""
    return {f"pin_{n}": (s.carrier, n) for n, s in _SPECS.items()
            if isinstance(s, Lid) and n in _PATHS and s.carrier in _PATHS}


def _joint_path(name: str, body: str) -> str:
    return f"{_PATHS[body]}/{name}_joint"


def _author(name: str, spec: tuple) -> None:
    """Author one lid's pin: disabled first, its frames set, enabled last."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    carrier, lid = spec
    stage = get_current_stage()
    joint = UsdPhysics.RevoluteJoint.Define(stage, _joint_path(name, carrier))
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateBody0Rel().SetTargets([_PATHS[carrier]])
    joint.CreateBody1Rel().SetTargets([_PATHS[lid]])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in PIN)))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(-float(LID_MAX))      # opening turns the lid about the carrier's -z
    joint.CreateUpperLimitAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")     # dry friction: a stiff damper that tops out
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(1.0)
    drive.CreateMaxForceAttr().Set(float(LID_DRAG))
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



_SPAWNERS = {Station: _spawn_station, Carrier: _spawn_carrier, Lid: _spawn_lid, Vial: _spawn_vial}
_BOUNDS: dict = {}


def _points_of(spec) -> np.ndarray:
    if isinstance(spec, Station):
        return np.vstack([p[0] for p in station_pieces()])
    if isinstance(spec, Carrier):
        return np.vstack([p[0] for p in carrier_pieces()])
    if isinstance(spec, Lid):
        return np.vstack([p[0] for p in lid_pieces()])
    return np.vstack([p[0] for p in vial_pieces()])


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
    """Join the parts as the reset left them: each lid to its carrier's pin."""
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


def held(name: str = "carrier") -> bool:
    """The station has taken this carrier."""
    return name in _HELD


def _rule(environment) -> None:
    """The kit's two rules, applied for the coming control step: each lid's catch (a torque law on its pin, with its
    reaction on the carrier), and the station's hold on a carrier it has taken."""
    stations = [n for n, s in _SPECS.items() if isinstance(s, Station) and n in _PATHS]
    vials = [n for n, s in _SPECS.items() if isinstance(s, Vial) and n in _PATHS]
    for lid, spec in ((n, s) for n, s in _SPECS.items() if isinstance(s, Lid) and n in _PATHS and s.carrier in _PATHS):
        car = spec.carrier
        try:
            (cp, cq), (_, lq) = _rel(environment, car), _rel(environment, lid)
        except Exception:                            # noqa: BLE001
            continue
        if car not in _HELD:
            for st in stations:
                sp, _ = _rel(environment, st)
                if taken(cp, cq, sp):
                    _HELD[car] = st
                    break
        angle = lid_angle(cq, lq)
        lift_c = lift_l = 0.0
        if car in _HELD:
            sp, _ = _rel(environment, _HELD[car])
            inside = sum(1 for v in vials if vial_in_carrier(*_rel(environment, v), cp, cq))
            cp, cq = held_pose(sp)
            write_pose(environment, car, cp, cq)
            write_pose(environment, lid, *lid_pose(cp, cq, angle))
            lift_c, lift_l = (CARRIER_MASS + inside * VIAL_MASS) * 9.81, LID_MASS * 9.81
        tau = lid_torque(angle)
        axis = -_rotation(cq)[:, 2]                  # opening turns the lid about the carrier's -z
        _wrench(environment, lid, tau * axis, lift=lift_l)
        _wrench(environment, car, -tau * axis, lift=lift_c)


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
    if getattr(scene, "_tubekit_patched", False):
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
    scene._tubekit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_tubekit_patched", False):
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
                _HELD.clear()
                result = env_reset(*aa, **kk)
                _arm(self._env)
                _rule(self._env)
                return result

            base.reset = reset_then_arm
        except Exception as err:                     # noqa: BLE001
            print(f"[tubekit] joints not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._tubekit_patched = True


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
