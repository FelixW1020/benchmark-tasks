"""shadekit: a small candlestick accent lamp with a candelabra torpedo bulb, and its clip-on shade (a six-sided shade
whose fitter has two sprung wire loops that slide down over the bulb), for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from shadekit import Jaw, Lamp, Shade, on_reset, on_step

Frames (m):

- :class:`Lamp`: base, stem, candle cup and bulb, one loose rigid part. ``pos`` is the middle of the underside of its
  base; its z runs up its axis.
- :class:`Shade`: the six panels, the top wire ring's three spokes and the fitter's hub ring, one loose rigid part.
  ``pos`` is the middle of its bottom opening; its z runs up its axis; panel 0 faces its +y, and the panels follow
  every 60 degrees counter-clockwise seen from above. The spokes run in from the tops of panels 1, 3 and 5.
- :class:`Jaw`: one of the fitter's two wire loops, a loose part hinged to the shade's hub ring along its top bar and
  sprung inward. ``pos`` is the middle of its top bar; its x points away from the shade's axis, its y along the hinge,
  its z up when the loop hangs straight down.

Every shape is built from numbers in NumPy when the task file is read, so the same numbers serve the scene, the
scoring and the offline checks. Isaac Sim is reached only when the scene is built.

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

__all__ = ["Jaw", "KitState", "Lamp", "Shade", "arm_points", "bulb_radius", "env", "finger_points", "gripper_points",
           "install", "jaw_angle", "jaw_rest", "lamp_pieces", "on_reset", "on_step", "panel_frame", "place",
           "seat", "shade_boxes", "shade_wires", "to_local", "turn_z", "write_pose"]

IN = 0.0254

# -- the shade (Shade frame) --------------------------------------------------------------------------------------------
TOP_AF = 3.0 * IN              # across the flats at the top (a 3 x 5 x 4 1/4 inch clip-on shade)
BOT_AF = 5.0 * IN              # across the flats at the bottom
HS = 4.25 * IN                 # height
SIDES = 6
WALL = 0.002                   # panel thickness: fabric on a styrene backing with its edge trim (an estimate)
SLOPE = math.atan((BOT_AF - TOP_AF) / 2 / HS)      # a panel's lean from the vertical (13.2 degrees)
STRIPS = 6                     # each panel is built from this many bars, bottom to top
SHADE_MASS = 0.046             # kg without the two loops: 50 g in all (an estimate)
RECESS = 0.013                 # the hub ring sits this far below the top (a 1/2 inch drop)
HUB_Z = HS - RECESS            # the hub ring's height in the shade
RING_R = 0.0065                # the hub ring: radius to the middle of its wire
WIRE = 0.0010                  # radius of the ring's and the spokes' wire (2 mm wire, an estimate)
SPOKES = (1, 3, 5)             # the panels whose tops carry a spoke

# -- the fitter's two loops (Jaw frame) -----------------------------------------------------------------------------------
JAW_L = 0.032                  # a loop's length below its hinge (an estimate from photographs of candelabra clip fitters)
JAW_HALF = 0.008               # half its width
JAW_WIRE = 0.0008              # radius of its wire (1.6 mm spring wire, an estimate)
JAW_MASS = 0.002               # kg each: heavier than the wire itself so that the contact solver holds it
JAW_REST = math.radians(3.0)   # at rest a loop hangs this far outward of straight down, against its stop
JAW_OPEN = math.radians(40.0)  # ... and can be sprung this much further out
JAW_STIFF = 0.20               # N m per radian of spring (an estimate)
JAW_PRELOAD = 0.012            # N m holding a loop against its stop at rest (an estimate)
JAW_DAMP = 0.0006              # N m s per radian
JAW_AZ = (0.0, math.pi)        # where the two loops hang on the hub ring (from the shade's x)

# -- the lamp (Lamp frame) -----------------------------------------------------------------------------------------------
BASE_R = 3.25 * IN / 2         # the round base is 3 1/4 inches wide
BASE_H = 0.012                 # (estimates from here down to the cup: the listing gives only the lamp's height and base)
FOOT_R, FOOT_H = 0.022, 0.026  # the turned foot above the base
STEM_R = 0.010                 # the candlestick's stem
PAN_Z, PAN_R, PAN_H = 0.150, 0.021, 0.004     # the drip pan under the candle cup
CUP_R = 0.0135                 # the candle cup that holds the socket
SOCKET_RIM = 0.1955            # top of the cup, where the bulb's glass begins
LAMP_MASS = 0.80               # kg (an estimate; the listing gives no weight)
LAMP_COM = 0.050               # height of the centre of mass above the table (an estimate: most of the weight is the base)
# the bulb: a B10 torpedo, 1.26 inches (32 mm) across and 3.72 inches (94.5 mm) high with its screw base (a
# manufacturer's data sheet). Radius at each distance below the tip; the glass is taken to stand 77 mm above the cup,
# the rest being the base inside it. The outline between the tip and the widest point is an estimate from the sheet's
# drawing.
BULB_H = 0.077
BULB_PROFILE = ((0.000, 0.0000), (0.004, 0.0045), (0.012, 0.0083), (0.026, 0.0130), (0.050, 0.0160), (0.077, 0.0085))
BULB_SIDES = 12
TIP_Z = SOCKET_RIM + BULB_H    # the bulb's tip above the table

BRASS, GLASS, SILK, TRIM, WIREC = (0.80, 0.62, 0.22), (0.96, 0.95, 0.90), (0.93, 0.90, 0.80), (0.80, 0.74, 0.58), (0.75, 0.75, 0.72)
FRICTION = (0.5, 0.4)          # the shade and the lamp's brass
SLICK = (0.3, 0.25)            # glass against spring wire (estimates)
PALM_BACK = 0.065
FINGER_THICK = 0.012


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


def turn_z(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn of ``angle`` rad about the vertical."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def place(pos, quat, local) -> np.ndarray:
    """A point given in a part's own frame, in the world."""
    return np.asarray(pos, float) + _rotation(quat) @ np.asarray(local, float)


def to_local(point, pos, quat) -> np.ndarray:
    return _rotation(quat).T @ (np.asarray(point, float) - np.asarray(pos, float))


def panel_frame(k: int) -> tuple:
    """Panel ``k`` of the shade, in the Shade's frame: the middle of its top edge (on its mid-surface), and unit
    vectors along that edge (``t``, counter-clockwise seen from above), down the panel (``d``) and out of it (``n``)."""
    az = math.pi / 2 + k * 2 * math.pi / SIDES
    out = np.array([math.cos(az), math.sin(az), 0.0])
    t = np.array([-math.sin(az), math.cos(az), 0.0])
    d = -math.cos(SLOPE) * np.array([0.0, 0.0, 1.0]) + math.sin(SLOPE) * out
    n = math.cos(SLOPE) * out + math.sin(SLOPE) * np.array([0.0, 0.0, 1.0])
    top = out * (TOP_AF / 2) + np.array([0.0, 0.0, HS])
    return top, t, d, n


def panel_width(down: float) -> float:
    """A panel's width this far down it from its top edge."""
    slant = HS / math.cos(SLOPE)
    w0, w1 = TOP_AF / math.sqrt(3.0), BOT_AF / math.sqrt(3.0)
    return w0 + (w1 - w0) * down / slant


def shade_boxes() -> list:
    """The shade's panels as bars: (centre, rotation, size) in the Shade's frame; a bar's x runs along the panel's
    width, its y up the panel, its z out of it."""
    slant = HS / math.cos(SLOPE)
    step = slant / STRIPS
    out = []
    for k in range(SIDES):
        top, t, d, n = panel_frame(k)
        rot = np.column_stack([t, -d, n])
        for i in range(STRIPS):
            mid = (i + 0.5) * step
            out.append((top + d * mid, rot, np.array([panel_width(mid), step, WALL])))
    return out


def shade_wires() -> list:
    """The fitter's fixed wires in the Shade's frame: (start, end, radius). Three spokes from the tops of panels 1, 3
    and 5 down to the hub ring, and the hub ring itself as six short bars."""
    out = []
    hub = np.array([0.0, 0.0, HUB_Z])
    for k in SPOKES:
        top, t, d, n = panel_frame(k)
        az = math.pi / 2 + k * 2 * math.pi / SIDES
        inner = hub + RING_R * np.array([math.cos(az), math.sin(az), 0.0])
        out.append((top - n * (WALL / 2 + WIRE) + d * 0.002, inner, WIRE))
    ring = [hub + RING_R / math.cos(math.pi / 6) * np.array([math.cos(a), math.sin(a), 0.0])
            for a in np.arange(7) * math.pi / 3 + math.pi / 6]
    out += [(a, b, WIRE) for a, b in zip(ring, ring[1:])]
    return out


def jaw_wires() -> list:
    """One loop in its own frame, hanging straight down: (start, end, radius). A top bar along the hinge, two legs and
    a bottom bar."""
    h, low = JAW_HALF, -JAW_L
    return [((0.0, -h, 0.0), (0.0, h, 0.0), JAW_WIRE), ((0.0, -h, 0.0), (0.0, -h, low), JAW_WIRE),
            ((0.0, h, 0.0), (0.0, h, low), JAW_WIRE), ((0.0, -h, low), (0.0, h, low), JAW_WIRE)]


def jaw_rest(which: int) -> tuple:
    """Where loop ``which`` (0 or 1) hangs at rest in the Shade's frame: its hinge point and its rotation."""
    az = JAW_AZ[which]
    hinge = np.array([RING_R * math.cos(az), RING_R * math.sin(az), HUB_Z])
    return hinge, _rz(az) @ _ry(-JAW_REST)


def jaw_angle(which: int, shade_quat, jaw_quat) -> float:
    """How far a loop is sprung outward from its rest (rad; 0 at rest)."""
    rest = _rotation(shade_quat) @ jaw_rest(which)[1]
    rel = rest.T @ _rotation(jaw_quat)                 # a turn about the loop's y
    return float(-math.atan2(rel[0, 2], rel[0, 0]))


def bulb_radius(below_tip: float) -> float:
    """The bulb's radius this far below its tip (m), from its outline."""
    s = np.array([p[0] for p in BULB_PROFILE])
    r = np.array([p[1] for p in BULB_PROFILE])
    return float(np.interp(below_tip, s, r))


def _prism(z0: float, z1: float, r0: float, r1: float | None = None, sides: int = 12) -> np.ndarray:
    a = 2 * math.pi * np.arange(sides) / sides
    ring = np.column_stack([np.cos(a), np.sin(a)])
    r1 = r0 if r1 is None else r1
    return np.vstack([np.column_stack([r0 * ring, np.full(sides, z0)]), np.column_stack([r1 * ring, np.full(sides, z1)])])


def bulb_points() -> np.ndarray:
    """The bulb's glass in the Lamp's frame: rings of points down its outline, and its tip."""
    a = 2 * math.pi * np.arange(BULB_SIDES) / BULB_SIDES
    pts = [[0.0, 0.0, TIP_Z]]
    for s, r in BULB_PROFILE[1:]:
        pts += [[r * math.cos(v), r * math.sin(v), TIP_Z - s] for v in a]
    return np.array(pts)


def lamp_pieces() -> list:
    """The lamp in its own frame: (points, colour) convex pieces. Base, foot, stem, drip pan, candle cup, bulb."""
    return [(_prism(0.0, BASE_H, BASE_R, BASE_R - 0.003), BRASS),
            (_prism(BASE_H, BASE_H + FOOT_H, FOOT_R, STEM_R + 0.002), BRASS),
            (_prism(BASE_H + FOOT_H, PAN_Z, STEM_R, sides=8), BRASS),
            (_prism(PAN_Z, PAN_Z + PAN_H, PAN_R, sides=12), BRASS),
            (_prism(PAN_Z + PAN_H, SOCKET_RIM, CUP_R, sides=8), BRASS),
            (bulb_points(), GLASS)]


def lamp_radius(z: float) -> float:
    """The lamp's radius at height ``z`` in its own frame (0 above the bulb's tip), for the offline checks."""
    if z < 0.0 or z > TIP_Z:
        return 0.0
    if z <= BASE_H:
        return BASE_R
    if z <= BASE_H + FOOT_H:
        return FOOT_R
    if z < PAN_Z:
        return STEM_R
    if z <= PAN_Z + PAN_H:
        return PAN_R
    if z <= SOCKET_RIM:
        return CUP_R
    return bulb_radius(TIP_Z - z)


def seat_depth() -> float:
    """How far below the bulb's tip the hub ring's middle comes to rest: where the glass is as wide as the ring's
    opening."""
    s = np.array([p[0] for p in BULB_PROFILE[:4]])
    r = np.array([p[1] for p in BULB_PROFILE[:4]])
    return float(np.interp(RING_R - WIRE, r, s))


SEAT = seat_depth()                       # about 5.5 mm
TOP_SEATED = TIP_Z - SEAT + RECESS        # the shade's top above the table when it is seated on the lamp (11 inches)


def seat(shade_pos, shade_quat, lamp_pos, lamp_quat) -> dict:
    """Where the shade is on the lamp. ``up``: the hub ring's height above its seated place on the bulb (m; 0 seated,
    positive higher). ``off``: the hub ring's distance from the lamp's axis. ``tilt``: the angle between the shade's
    axis and the lamp's (deg; 180 for a shade put on upside down). ``over``: whether the bulb's tip is inside the shade
    (between its bottom and its hub ring, through which it stands a few millimetres when the shade is home, and
    within its walls), the shade the right way up."""
    hub = to_local(place(shade_pos, shade_quat, (0.0, 0.0, HUB_Z)), lamp_pos, lamp_quat)
    za, zb = _rotation(shade_quat)[:, 2], _rotation(lamp_quat)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(za @ zb, -1.0, 1.0))))
    tip = to_local(place(lamp_pos, lamp_quat, (0.0, 0.0, TIP_Z)), shade_pos, shade_quat)
    wide = (TOP_AF + (BOT_AF - TOP_AF) * (HS - tip[2]) / HS) / 2 if 0.0 <= tip[2] <= HS else 0.0
    over = bool(tilt < 90.0 and 0.0 <= tip[2] <= HUB_Z + SEAT + 0.003 and math.hypot(tip[0], tip[1]) <= wide)
    return {"up": float(hub[2] - (TIP_Z - SEAT)), "off": float(math.hypot(hub[0], hub[1])), "tilt": tilt, "over": over}


def segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def shade_distance(points, shade_pos, shade_quat) -> float:
    """The least distance from any of ``points`` (world) to the shade's panels."""
    p = (np.asarray(points, float).reshape(-1, 3) - np.asarray(shade_pos, float)) @ _rotation(shade_quat)
    best = np.inf
    for centre, rot, size in shade_boxes():
        q = np.abs((p - centre) @ rot) - size / 2
        d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        best = min(best, float(d.min()))
    return best


def lamp_distance(points, lamp_pos, lamp_quat) -> float:
    """The least distance from any of ``points`` (world) to the lamp, taken as round about its axis (a little
    generous: each point is compared with the lamp's radius at its own height)."""
    p = (np.asarray(points, float).reshape(-1, 3) - np.asarray(lamp_pos, float)) @ _rotation(lamp_quat)
    best = np.inf
    for q in p:
        z = float(np.clip(q[2], 0.0, TIP_Z))
        best = min(best, math.hypot(math.hypot(q[0], q[1]) - lamp_radius(z) if math.hypot(q[0], q[1]) > lamp_radius(z) else 0.0,
                                    q[2] - z))
    return float(best)


# -- the parts -----------------------------------------------------------------------------------------------------------

class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Lamp(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = LAMP_MASS


@dataclass(frozen=True)
class Shade(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = SHADE_MASS


@dataclass(frozen=True)
class Jaw(_Part):
    """Loop ``which`` (0 or 1) of the fitter of ``shade``."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = JAW_MASS
    shade: str = "shade"
    which: int = 0


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

def hull_faces(points: np.ndarray, eps: float = 1e-7) -> list:
    """The faces of the convex hull of ``points``: each a list of point indices, wound counter-clockwise seen from
    outside. Brute force over triples; coplanar triangles are merged into one polygon."""
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
                if size < 1e-14:
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

    def look(self, color, rough=0.6, metal=0.0, opacity=1.0) -> str:
        from yam_arena.station import linear_rgb

        key = (tuple(color), rough, metal, opacity)
        if key not in self.looks:
            path = f"{self.path}/Looks/v{len(self.looks)}"
            kw = {"opacity": opacity} if opacity < 1.0 else {}
            vis = self.sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=rough, metallic=metal, **kw)
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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, com=None):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))
        if com is not None:
            mass_api.CreateCenterOfMassAttr(self.Gf.Vec3f(*(float(v) for v in com)))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def _collide(self, prim, mat, offset):
        from isaaclab.sim.utils import bind_physics_material
        from pxr import PhysxSchema, UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(offset)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)

    def hull(self, path, points, look, mat=None, offset=0.0008):
        """A convex piece: drawn as its hull's faces, and a convex-hull collider if ``mat`` is given."""
        from isaaclab.sim.utils import bind_visual_material
        from pxr import UsdPhysics, Vt

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
            self._collide(prim, mat, offset)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        return prim

    def bar(self, path, centre, rot, size, look, mat=None, offset=0.0008):
        """A box of ``size`` at ``centre``, its edges along the columns of ``rot``: a box collider (thin plates cannot
        be cooked as convex hulls for the GPU pipeline, and then collide with nothing)."""
        from isaaclab.sim.utils import bind_visual_material

        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        x, y, z, w = _quat(rot)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in centre)))
        cube.AddOrientOp().Set(self.Gf.Quatf(float(w), float(x), float(y), float(z)))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*(float(v) for v in size)))
        prim = cube.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            self._collide(prim, mat, offset)
        return prim

    def wire(self, path, p0, p1, radius, look, mat=None, offset=0.0006):
        """A round bar from ``p0`` to ``p1`` with rounded ends."""
        from isaaclab.sim.utils import bind_visual_material

        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        x = d / length
        y = np.cross((0.0, 0.0, 1.0) if abs(x[2]) < 0.9 else (1.0, 0.0, 0.0), x)
        y /= np.linalg.norm(y)
        qx, qy, qz, qw = _quat(np.column_stack([x, y, np.cross(x, y)]))
        cap = self.UsdGeom.Capsule.Define(self.stage, path)
        cap.CreateAxisAttr("X")
        cap.CreateHeightAttr(length)
        cap.CreateRadiusAttr(float(radius))
        cap.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in (p0 + p1) / 2)))
        cap.AddOrientOp().Set(self.Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        prim = cap.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            self._collide(prim, mat, offset)
        return prim


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _start(cfg, prim_path, translation, orientation, iterations=None, lin_damp=0.05, ang_damp=0.1, com=None):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, lin_damp=lin_damp, ang_damp=ang_damp, com=com)
    return spec, a


def _spawn_lamp(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, com=(0.0, 0.0, LAMP_COM))
    brass, glass = a.mat("brass", *FRICTION), a.mat("glass", *SLICK)
    pieces = lamp_pieces()
    for n, (pts, color) in enumerate(pieces[:-1]):
        a.hull(f"{prim_path}/p_{n}", pts, a.look(color, rough=0.3, metal=0.9), brass)
    a.hull(f"{prim_path}/bulb", pieces[-1][0], a.look(GLASS, rough=0.2), glass)
    return a.root


def _spawn_shade(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, ang_damp=0.3)
    m, w = a.mat("silk", *FRICTION), a.mat("wire", *SLICK)
    look = a.look(SILK, rough=0.8)
    for n, (centre, rot, size) in enumerate(shade_boxes()):
        a.bar(f"{prim_path}/s_{n}", centre, rot, size, look, m)
    wl = a.look(WIREC, rough=0.4, metal=0.8)
    for n, (p0, p1, r) in enumerate(shade_wires()):
        a.wire(f"{prim_path}/w_{n}", p0, p1, r, wl, w)
    return a.root


def _spawn_jaw(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, lin_damp=0.1, ang_damp=0.5)
    w, wl = a.mat("wire", *SLICK), a.look(WIREC, rough=0.4, metal=0.8)
    for n, (p0, p1, r) in enumerate(jaw_wires()):
        a.wire(f"{prim_path}/w_{n}", p0, p1, r, wl, w)
    return a.root


# -- joints ------------------------------------------------------------------------------------------------------------

def _joints() -> list:
    """Every joint the parts in the scene need: (name, shade, loop, which)."""
    return [(f"{n}_hinge", s.shade, n, s.which) for n, s in _SPECS.items()
            if isinstance(s, Jaw) and n in _PATHS and s.shade in _PATHS]


def _joint_path(joint) -> str:
    return f"{_PATHS[joint[1]]}/{joint[0]}"


def _author(joint) -> None:
    """Author one loop's hinge: disabled first, its frames set, enabled last."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    name, shade, jaw, which = joint
    hinge, rot = jaw_rest(which)
    qx, qy, qz, qw = _quat(rot)
    stage = get_current_stage()
    j = UsdPhysics.RevoluteJoint.Define(stage, _joint_path(joint))
    j.CreateJointEnabledAttr().Set(False)
    j.CreateBody0Rel().SetTargets([_PATHS[shade]])
    j.CreateBody1Rel().SetTargets([_PATHS[jaw]])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in hinge)))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    # The loop turns about its top bar (its y). A positive turn swings its lower end inward, so the stop at rest is
    # the upper limit (0) and springing it outward is negative. The spring pulls toward a target past the stop:
    # JAW_PRELOAD holds it there, and every radian outward adds JAW_STIFF. USD angular drives are per degree.
    j.CreateAxisAttr().Set("Y")
    j.CreateLowerLimitAttr().Set(-math.degrees(JAW_OPEN))
    j.CreateUpperLimitAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(math.radians(JAW_STIFF)))
    drive.CreateDampingAttr().Set(float(math.radians(JAW_DAMP)))
    drive.CreateTargetPositionAttr().Set(float(math.degrees(JAW_PRELOAD / JAW_STIFF)))
    j.CreateJointEnabledAttr().Set(True)       # last write, after the frames


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


_SPAWNERS = {Lamp: _spawn_lamp, Shade: _spawn_shade, Jaw: _spawn_jaw}
_BOUNDS: dict = {}


def _points_of(spec) -> np.ndarray:
    if isinstance(spec, Lamp):
        return np.vstack([p for p, _ in lamp_pieces()])
    if isinstance(spec, Shade):
        half = BOT_AF / math.sqrt(3.0)
        return np.array([[sx * half, sy * half, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, HS)])
    return np.array([p for w in jaw_wires() for p in w[:2]], float)


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


def _disarm() -> None:
    """Switch every joint off (before a reset puts the parts back)."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for joint in _joints():
        prim = stage.GetPrimAtPath(_joint_path(joint))
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)


def _arm(environment) -> None:
    """Join the parts as the reset left them: each loop to the shade's hub ring."""
    for joint in _joints():
        _author(joint)


def _local_bounds(spec) -> tuple:
    key = type(spec)
    if key not in _BOUNDS:
        pts = _points_of(spec)
        _BOUNDS[key] = (pts.min(0), pts.max(0))
    return _BOUNDS[key]


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


def arm_points(environment) -> list:
    """``gripper_points`` with the middle of each wrist link added (the two links above the gripper), for judging
    whether an arm is clear of a part."""
    base = environment.unwrapped
    origin = _origin(environment)
    out = gripper_points(environment)
    for pts, arm in zip(out, ("left_robot", "right_robot")):
        robot = base.scene[arm]
        poses, names = _np(robot.data.body_link_pose_w)[0], list(robot.body_names)
        for link in ("link4", "link5", "gripper"):
            if link in names:
                pts.append(poses[names.index(link)][:3] - origin)
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
    if getattr(scene, "_shadekit_patched", False):
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
    scene._shadekit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_shadekit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
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
            print(f"[shadekit] joints not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._shadekit_patched = True


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
