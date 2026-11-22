"""honekit: an 8 in chef's knife, a 10 in honing steel, and a cutting board standing on a kitchen scale, for yam_arena
task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from honekit import Board, Knife, Scale, Steel, on_reset, on_step

Frames (every part starts with its axes along the world's unless a ``quat`` is given: x away from the robot, y left,
z up):

- :class:`Knife`: ``pos`` is the heel, on the edge (the corner where the cutting edge starts, below the handle). Its x
  runs along the blade to the tip, its y from the edge up to the spine, its z through the blade's thickness. The
  handle runs back from the heel along -x.
- :class:`Steel`: ``pos`` is the tip of the rod. Its x runs along the rod to the guard and on through the handle; the
  guard's long axis and the handle's wider faces lie along y.
- :class:`Board`: ``pos`` is the middle of the board. It has no weight of its own and hangs on a spring (the scale's
  load cell): the scale reads the spring's travel. A non-slip pad is part of it.
- :class:`Scale`: the scale's body under the board, fixed. ``pos`` is its middle.

The blade's colliders are boxes (thin plates cannot be cooked as convex hulls for the GPU pipeline); the blade is
drawn as a smooth wedge. The steel's rod is a capsule. The same numbers serve the scene, the scoring and the offline
checks. Isaac Sim is reached only when the scene is built.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box, ObjectState

# -- the knife (m): a stamped 20 cm (8 in) chef's knife with a moulded handle ------------------------------------------
BL = 0.200                    # blade length, heel to tip (maker: 7.9 in)
HL = 0.140                    # handle length (overall 13.4 in less the blade)
BH = 0.053                    # blade height at the heel, edge to spine (an estimate, from the maker's width of 2.2 in)
T_SPINE = 0.0023              # blade thickness at the spine, at the heel (an estimate) ...
T_TIP = 0.0012                # ... and near the tip (an estimate)
STRAIGHT = 0.090              # the edge is straight this far from the heel (an estimate, from pictures) ...
TIP_RISE = 0.032              # ... then curves up to the tip, which stands this far above the heel's edge line
SPINE_FLAT = 0.120            # the spine is straight this far from the heel, then drops to the tip (estimates)
HANDLE_T = 0.020              # handle thickness (maker: 0.8 in for the knife lying flat)
HANDLE_H = 0.030              # handle height, back to belly (an estimate)
K_MASS = 0.181                # maker: 6.4 oz
K_COM = (0.010, 0.031, 0.0)   # centre of mass: 1 cm in front of the heel (an estimate, from the blade's and the handle's volumes)
K_INERTIA = (2.0e-5, 1.7e-3, 1.7e-3)   # kg m2 about the knife's x, y, z through the centre of mass (estimates)
EDGE_FROM = 0.004             # the edge is scored from this far in front of the heel to the tip

# -- the steel: a 10 in round honing steel with a finger guard and a moulded handle --------------------------------------
ROD_L = 0.259                 # the rod, tip to guard (maker: "Length of blade: 10.2 in")
ROD_R = 0.0055                # its radius (an estimate: the maker gives no diameter)
GUARD_T = 0.008               # the guard's thickness (an estimate)
GUARD_A, GUARD_B = 0.0265, 0.0205      # the guard's half width and half height (maker: 2.1 in wide, 1.6 in high)
S_HANDLE = 0.125              # the handle's length (an estimate; overall 16.1 in less rod, guard and hanging loop)
S_HANDLE_W, S_HANDLE_T = 0.030, 0.026  # the handle's width (y) and thickness (z) (estimates)
S_LOOP = 0.017                # the hanging loop past the handle's end (drawn, no collider)
S_MASS = 0.275                # maker: 9.7 oz
S_COM = (0.186, 0.0, 0.0)     # 7.3 cm from the guard toward the tip (computed: a steel rod and a polyamide handle)
S_INERTIA = (1.2e-5, 3.1e-3, 3.1e-3)   # estimates, from the same split

# -- the board and the scale ---------------------------------------------------------------------------------------------
BOARD = (0.305, 0.457, 0.0127) # the cutting board: depth (x), width (y), thickness (12 x 18 x 1/2 in, polyethylene)
PAD = (0.080, 0.080, 0.003)   # the non-slip pad the steel's tip stands on
SCALE = (0.300, 0.300, 0.050) # the bench scale's base and pan: depth, width, height (maker: 300 x 300 x 50 mm); its display is a separate indicator
SCALE_GAP = 0.003             # the board's underside is this far above the scale's body when nothing is on it
SPRING = 10000.0              # N/m: the scale's load cell as a spring (an estimate: 1 mm under 1 kg)
SPRING_DAMP = 400.0           # N s/m
BOARD_MASS = 1.45             # kg (the board's inertia on its spring, from its 3.24 lb shipping weight; it is tared out, so it has no weight)

STEEL, HANDLE_BLACK, ROD_GREY = (0.72, 0.73, 0.75), (0.06, 0.06, 0.07), (0.55, 0.56, 0.58)
WALNUT, PAD_GREEN, SCALE_WHITE = (0.42, 0.29, 0.19), (0.20, 0.42, 0.30), (0.62, 0.64, 0.66)   # brown board, green pad, grey scale
FRICTION = {"handle": (0.9, 0.8), "blade": (0.3, 0.25), "rod": (0.3, 0.25), "board": (0.5, 0.45), "pad": (1.1, 1.0)}   # estimates
PAD_PATCH = 0.006             # radius of the patch over which a rubber gripper pad meets a handle (an estimate)
ARM_LINKS = ("link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right")
FINE_LINKS = ("gripper", "tip_left", "tip_right")     # sampled every 4 mm; the others every 12 mm


# -- the blade's outline (pure) ----------------------------------------------------------------------------------------

def edge_y(s) -> np.ndarray:
    """How far the cutting edge stands above the heel's edge line, ``s`` m from the heel."""
    s = np.asarray(s, float)
    return TIP_RISE * np.clip((s - STRAIGHT) / (BL - STRAIGHT), 0.0, 1.0) ** 2


def spine_y(s) -> np.ndarray:
    """The spine's height above the heel's edge line, ``s`` m from the heel."""
    s = np.asarray(s, float)
    return BH - (BH - TIP_RISE) * np.clip((s - SPINE_FLAT) / (BL - SPINE_FLAT), 0.0, 1.0) ** 1.5


def spine_t(s) -> np.ndarray:
    """The blade's thickness at the spine, ``s`` m from the heel."""
    return T_SPINE + (T_TIP - T_SPINE) * np.clip(np.asarray(s, float) / BL, 0.0, 1.0)


_SEGMENTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.135, 0.15, 0.1625, 0.175, 0.1875, 0.20)
_STRIPS = (0.0, 0.10, 0.25, 0.45, 0.70, 1.0)          # across the blade's height, edge to spine
_T_MIN = 0.0010                                        # the thinnest collider


def knife_boxes() -> list:
    """The blade's colliders: (lo, hi) boxes in the knife's frame. Twelve segments along the blade, five strips from
    the edge to the spine, each strip as thick as the blade's wedge is in its middle."""
    out = []
    for s0, s1 in zip(_SEGMENTS, _SEGMENTS[1:]):
        sm = (s0 + s1) / 2
        e, p, t = float(edge_y(sm)), float(spine_y(sm)), float(spine_t(sm))
        for f0, f1 in zip(_STRIPS, _STRIPS[1:]):
            th = max(_T_MIN, t * (f0 + f1) / 2)
            out.append((np.array([s0, e + f0 * (p - e), -th / 2]), np.array([s1, e + f1 * (p - e), th / 2])))
    return out


def blade_pieces() -> list:
    """The blade as it is drawn: a smooth wedge, (points, colour) pieces in the knife's frame."""
    out = []
    for s0, s1 in zip(_SEGMENTS, _SEGMENTS[1:]):
        pts = []
        for s in (s0, s1):
            e, p, t = float(edge_y(s)), float(spine_y(s)), float(spine_t(s))
            p = max(p, e + 0.0008)
            pts += [[s, e, -0.00008], [s, e, 0.00008], [s, p, -t / 2], [s, p, t / 2]]
        out.append((np.array(pts), STEEL))
    return out


_CHAMFER = 0.004
# the handle's sections: distance from the heel (negative: behind it), half thickness, belly and back heights
_HANDLE = ((0.0, 0.0080, 0.027, BH), (-0.012, HANDLE_T / 2, BH - HANDLE_H, BH), (-0.060, HANDLE_T / 2, BH - HANDLE_H, BH),
           (-0.115, HANDLE_T / 2, BH - HANDLE_H - 0.002, BH - 0.002), (-HL, 0.0085, BH - HANDLE_H - 0.001, BH - 0.005))


def _section(x: float, hz: float, y0: float, y1: float) -> np.ndarray:
    c = min(_CHAMFER, 0.45 * hz, 0.45 * (y1 - y0) / 2)
    yz = [(y1 - c, hz), (y1, hz - c), (y1, -hz + c), (y1 - c, -hz), (y0 + c, -hz), (y0, -hz + c), (y0, hz - c), (y0 + c, hz)]
    return np.array([[x, y, z] for y, z in yz])


def handle_pieces() -> list:
    """The handle: (points, colour) convex pieces in the knife's frame. Between 12 and 60 mm behind the heel its back
    and belly are flat and parallel, ``HANDLE_H`` apart."""
    secs = [_section(*s) for s in _HANDLE]
    return [(np.vstack(pair), HANDLE_BLACK) for pair in zip(secs, secs[1:])]



def _oct(x: float, hy: float, hz: float, c: float) -> np.ndarray:
    """An eight-sided section at ``x``: a rectangle 2 hy by 2 hz with its corners cut by ``c``."""
    yz = [(hy - c, hz), (hy, hz - c), (hy, -hz + c), (hy - c, -hz), (-hy + c, -hz), (-hy, -hz + c), (-hy, hz - c), (-hy + c, hz)]
    return np.array([[x, y, z] for y, z in yz])


def steel_pieces() -> list:
    """The steel's guard and handle: (points, colour) convex pieces in the steel's frame. The handle's two wide faces
    (across z) are flat and parallel, ``S_HANDLE_T`` apart, from 1.5 cm behind the guard to 1.5 cm from its end."""
    g0, g1 = ROD_L, ROD_L + GUARD_T
    h1 = g1 + S_HANDLE
    guard = np.vstack([_oct(g0, GUARD_A, GUARD_B, 0.012), _oct(g1, GUARD_A, GUARD_B, 0.012)])
    secs = [_oct(g1, 0.011, 0.010, 0.004), _oct(g1 + 0.015, S_HANDLE_W / 2, S_HANDLE_T / 2, 0.006),
            _oct(h1 - 0.015, S_HANDLE_W / 2, S_HANDLE_T / 2, 0.006), _oct(h1, 0.011, 0.010, 0.004)]
    out = [(guard, HANDLE_BLACK)]
    out += [(np.vstack(pair), HANDLE_BLACK) for pair in zip(secs, secs[1:])]
    return out


def _box_points(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


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


class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Knife(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = K_MASS


@dataclass(frozen=True)
class Steel(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = S_MASS


@dataclass(frozen=True)
class Board(_Part):
    """The cutting board on the scale's spring. ``pad``: the middle of the non-slip pad, from the board's middle (x, y)."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = BOARD_MASS
    pad: tuple = (0.0, 0.0)


@dataclass(frozen=True)
class Scale(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 1.0


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


def place(pos, quat, local) -> np.ndarray:
    """Points given in a part's own frame, in the world."""
    return np.asarray(pos, float) + np.asarray(local, float) @ _rotation(quat).T


def to_local(points, pos, quat) -> np.ndarray:
    return (np.asarray(points, float) - np.asarray(pos, float)) @ _rotation(quat)


def edge_local(step: float = 0.004) -> np.ndarray:
    """Points along the cutting edge, from ``EDGE_FROM`` in front of the heel to the tip, in the knife's frame."""
    s = np.r_[np.arange(EDGE_FROM, BL, step), BL]
    return np.column_stack([s, edge_y(s), np.zeros(len(s))])


_EDGE = edge_local()
STRAIGHT_EDGE = _EDGE[:, 0] <= STRAIGHT + 1e-9          # the edge points on the straight part


def edge_points(pos, quat) -> np.ndarray:
    """The cutting edge's points in the world."""
    return place(pos, quat, _EDGE)


def _knife_samples() -> np.ndarray:
    s = np.linspace(0.0, BL, 21)
    pts = [[x, float(edge_y(x)), 0.0] for x in s] + [[x, float(spine_y(x)), 0.0] for x in s]
    pts += [[x, float(edge_y(x) + spine_y(x)) / 2, 0.0] for x in s]
    for x, hz, y0, y1 in _HANDLE:
        pts += [[x, y, z] for y in (y0, y1) for z in (-hz, hz)]
    return np.array(pts)


_KNIFE = _knife_samples()


def knife_points(pos, quat) -> np.ndarray:
    """World points over the knife: along its edge, spine and middle, and at the corners of its handle."""
    return place(pos, quat, _KNIFE)


def _steel_samples() -> np.ndarray:
    xs = np.linspace(0.0, ROD_L, 14)
    pts = [[x, 0.0, 0.0] for x in xs]
    pts += [p for piece in steel_pieces() for p in piece[0].tolist()]
    return np.array(pts)


_STEEL = _steel_samples()


def steel_points(pos, quat) -> np.ndarray:
    """World points over the steel: along its rod's axis and at the corners of its guard and handle."""
    return place(pos, quat, _STEEL)


def blade_points(pos, quat) -> np.ndarray:
    """World points over the blade only (edge, spine and middle)."""
    return place(pos, quat, _KNIFE[:63])


def scale_grams(board_z: float, rest_z: float) -> float:
    """What the scale reads (g, tared with the bare board) when the board's middle is at ``board_z`` and was at
    ``rest_z`` with nothing on it."""
    return float(SPRING * (rest_z - board_z) / 9.81 * 1000.0)


# -- the arms' shapes (pure): points on each link's collision mesh, in the link's own frame ------------------------------

def _stl_points(path: Path) -> np.ndarray:
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(tri["v"].reshape(-1, 3).astype(float).round(4), axis=0)


def _spread(points: np.ndarray, cell: float) -> np.ndarray:
    """One point per ``cell``-sized box, keeping the extremes; long triangles' edges are filled in first, so that a
    flat face is covered and not just its corners."""
    keep = {}
    for p in points:
        keep.setdefault(tuple(np.floor(p / cell).astype(int)), p)
    extremes = [points[points[:, k].argmin()] for k in range(3)] + [points[points[:, k].argmax()] for k in range(3)]
    return np.vstack([np.array(list(keep.values())), np.array(extremes)])


def _stl_surface(path: Path, cell: float) -> np.ndarray:
    """Points over the mesh's triangles, about ``cell`` apart."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    out = []
    for v in tri["v"].astype(float):
        a, bb, c = v
        k = int(max(np.linalg.norm(bb - a), np.linalg.norm(c - a), np.linalg.norm(c - bb)) / cell) + 1
        for i in range(k + 1):
            for j in range(k + 1 - i):
                out.append(a + (bb - a) * i / k + (c - a) * j / k)
    return np.array(out)


def _link_points() -> dict:
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = Path(str(YAM_URDF)).parent
    text = Path(str(YAM_URDF)).read_text()
    out = {}
    for name in ARM_LINKS:
        cell = 0.004 if name in FINE_LINKS else 0.012
        m = re.search(r'<link name="%s">(.*?)</link>' % name, text, re.S)
        pts = []
        for c in re.findall(r"<collision.*?</collision>", m.group(1) if m else "", re.S):
            o = re.search(r'origin xyz="([^"]+)" rpy="([^"]+)"', c)
            f = re.search(r'filename="([^"]+)"', c)
            if not (o and f):
                continue
            xyz = np.array([float(v) for v in o.group(1).split()])
            rot = rpy_matrix(np.array([float(v) for v in o.group(2).split()]))
            pts.append(_stl_surface(root / f.group(1), cell) @ rot.T + xyz)
        out[name] = _spread(np.vstack(pts), cell) if pts else np.zeros((0, 3))
    return out


_LINKS: dict = {}


def link_points() -> dict:
    """Points on each arm link's collision mesh, in the link's frame (read once from the robot's URDF): about 4 mm
    apart on the gripper and the fingertips, 12 mm on the other links."""
    if not _LINKS:
        _LINKS.update(_link_points())
    return _LINKS


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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, com=None, inertia=None, report=False):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if report:                                   # the simulator reports this body's contacts (every one, however light)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(mass))
        if com is not None:
            api.CreateCenterOfMassAttr(self.Gf.Vec3f(*(float(v) for v in com)))
        if inertia is not None:
            api.CreateDiagonalInertiaAttr(self.Gf.Vec3f(*(float(v) for v in inertia)))
            api.CreatePrincipalAxesAttr(self.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def hull(self, path, points, look, mat=None, offset=0.0005, patch=0.0):
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
            if patch:                                # a rubber pad meets the handle over a patch, which resists twisting
                cp.CreateTorsionalPatchRadiusAttr(float(patch))
                cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def box(self, path, lo, hi, look, mat=None, offset=0.0005, patch=0.0):
        """A box with its edges along the part's axes: drawn if ``look`` is given, a box collider if ``mat`` is."""
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from pxr import PhysxSchema, UsdPhysics

        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in (lo + hi) / 2)))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*(float(v) for v in hi - lo)))
        prim = cube.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        else:
            self.UsdGeom.Imageable(prim).MakeInvisible()       # a collider only: the smooth shape is drawn instead
        if mat:
            UsdPhysics.CollisionAPI.Apply(prim)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(offset)
            cp.CreateRestOffsetAttr(0.0)
            if patch:
                cp.CreateTorsionalPatchRadiusAttr(float(patch))
                cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim


    def capsule(self, path, x0, x1, radius, look, mat, offset=0.0005):
        """A capsule along the part's x from ``x0`` to ``x1`` (its round ends included): drawn, and a collider."""
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from pxr import PhysxSchema, UsdPhysics

        cap = self.UsdGeom.Capsule.Define(self.stage, path)
        cap.CreateAxisAttr("X")
        cap.CreateHeightAttr(float(x1 - x0 - 2 * radius))
        cap.CreateRadiusAttr(float(radius))
        cap.AddTranslateOp().Set(self.Gf.Vec3d(float((x0 + x1) / 2), 0.0, 0.0))
        prim = cap.GetPrim()
        bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(offset)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _spawn_knife(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32, com=K_COM, inertia=K_INERTIA, report=True)
    grip, steel = a.mat("handle", *FRICTION["handle"]), a.mat("blade", *FRICTION["blade"])
    for n, (pts, color) in enumerate(handle_pieces()):
        a.hull(f"{prim_path}/h_{n}", pts, a.look(color, rough=0.75), grip, patch=PAD_PATCH)
    for n, (pts, color) in enumerate(blade_pieces()):
        a.hull(f"{prim_path}/v_{n}", pts, a.look(color, rough=0.3, metal=0.9))
    for n, (lo, hi) in enumerate(knife_boxes()):
        a.box(f"{prim_path}/b_{n}", lo, hi, None, steel)
    return a.root


def _spawn_steel(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32, com=S_COM, inertia=S_INERTIA, ang_damp=1.0, report=True)    # it does not roll away
    grip, rod = a.mat("handle", *FRICTION["handle"]), a.mat("rod", *FRICTION["rod"])
    for n, (pts, color) in enumerate(steel_pieces()):
        a.hull(f"{prim_path}/h_{n}", pts, a.look(color, rough=0.75), grip, patch=PAD_PATCH)
    a.capsule(f"{prim_path}/rod", 0.0, ROD_L + 0.003, ROD_R, a.look(ROD_GREY, rough=0.45, metal=0.9), rod)
    return a.root


def _spawn_board(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    from pxr import PhysxSchema

    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    prim = a.body(prim_path, float(spec.mass), iterations=32, lin_damp=0.0, ang_damp=0.0)
    PhysxSchema.PhysxRigidBodyAPI(prim).CreateDisableGravityAttr(True)       # tared: the spring carries only what is put on it
    hx, hy, hz = (v / 2 for v in BOARD)
    a.box(f"{prim_path}/board", (-hx, -hy, -hz), (hx, hy, hz), a.look(WALNUT, rough=0.6), a.mat("board", *FRICTION["board"]))
    px, py = (float(v) for v in spec.pad)
    a.box(f"{prim_path}/pad", (px - PAD[0] / 2, py - PAD[1] / 2, hz), (px + PAD[0] / 2, py + PAD[1] / 2, hz + PAD[2]),
          a.look(PAD_GREEN, rough=0.9), a.mat("pad", *FRICTION["pad"], combine="max"))
    return a.root


def _spawn_scale(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    from pxr import UsdPhysics

    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    prim = a.body(prim_path, float(spec.mass))
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    hx, hy, hz = (v / 2 for v in SCALE)
    a.box(f"{prim_path}/body", (-hx, -hy, -hz), (hx, hy, hz), a.look(SCALE_WHITE, rough=0.4), a.mat("board", *FRICTION["board"]))
    return a.root


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


_SPAWNERS = {Knife: _spawn_knife, Steel: _spawn_steel, Board: _spawn_board, Scale: _spawn_scale}
_SAMPLES = {Knife: _KNIFE, Steel: _STEEL, Board: _box_points([-v / 2 for v in BOARD], [v / 2 for v in BOARD]),
            Scale: _box_points([-v / 2 for v in SCALE], [v / 2 for v in SCALE])}




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


def _spring_path(name: str) -> str:
    return f"{_PATHS[name]}/scale_spring"


def _boards() -> list:
    return [n for n, s in _SPECS.items() if isinstance(s, Board) and n in _PATHS]


def _disarm() -> None:
    """Switch the scale's spring off (before a reset puts the board back)."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for name in _boards():
        prim = stage.GetPrimAtPath(_spring_path(name))
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)


def _arm(environment) -> None:
    """Hang each board on its spring, from where the reset left it: a vertical slide to the world with a spring."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    stage = get_current_stage()
    origin = _origin(environment)
    for name in _boards():
        spec = _SPECS[name]
        write_pose(environment, name, spec.pos, spec.quat)      # level and at rest, where the task file puts it
        pos = np.asarray(spec.pos, float) + origin
        x, y, z, w = (float(v) for v in spec.quat)
        jt = UsdPhysics.PrismaticJoint.Define(stage, _spring_path(name))
        jt.CreateJointEnabledAttr().Set(False)
        jt.CreateBody1Rel().SetTargets([_PATHS[name]])
        jt.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in pos)))
        jt.CreateLocalRot0Attr().Set(Gf.Quatf(w, x, y, z))
        jt.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        jt.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        jt.CreateExcludeFromArticulationAttr().Set(True)
        jt.CreateAxisAttr().Set("Z")
        jt.CreateLowerLimitAttr().Set(-float(SCALE_GAP) + 0.0003)
        jt.CreateUpperLimitAttr().Set(0.002)
        drive = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(SPRING))
        drive.CreateDampingAttr().Set(float(SPRING_DAMP))
        drive.CreateTargetPositionAttr().Set(0.0)
        jt.CreateJointEnabledAttr().Set(True)       # last write, after the frames


# -- the contact force between the knife and the steel -----------------------------------------------------------------
_CONTACT: dict = {"view": None, "sub": None, "sum": 0.0, "steps": 0, "hits": 0, "dt": 1.0 / 240.0, "error": "", "calls": 0}


def _pair_paths() -> tuple:
    knife = next((n for n, s in _SPECS.items() if isinstance(s, Knife)), None)
    steel = next((n for n, s in _SPECS.items() if isinstance(s, Steel)), None)
    return _PATHS.get(knife), _PATHS.get(steel)


def _pair_force() -> float:
    """The contact force (N) between the knife's body and the steel's body over the last physics step, as PhysX
    reports it (the same contact view Isaac Lab's contact sensor reads)."""
    view = _CONTACT["view"]
    if view is None:
        return 0.0
    m = _np(view.get_contact_force_matrix(dt=_CONTACT["dt"])).reshape(-1, 3)
    return float(np.linalg.norm(m.sum(axis=0)))


def _on_physics_step(dt) -> None:
    """After every physics step: add that step's knife-on-steel contact force to the running count."""
    try:
        _CONTACT["calls"] += 1
        f = _pair_force()
        _CONTACT["sum"] += f
        _CONTACT["steps"] += 1
        _CONTACT["hits"] += 1 if f > 1e-4 else 0
    except Exception as err:                         # noqa: BLE001
        _CONTACT["error"] = repr(err)


def _watch_contacts(environment) -> None:
    """Make the contact view for the knife against the steel, and read it after every physics step (once)."""
    if _CONTACT["view"] is not None:
        return
    try:
        import omni.physx
        from isaaclab_physx.physics import PhysxManager

        base = environment.unwrapped
        dt = getattr(base, "physics_dt", None)
        if dt is None:
            dt = float(base.sim.get_physics_dt())
        _CONTACT["dt"] = float(dt)
        kpath, spath = _pair_paths()
        sim_view = PhysxManager.get_physics_sim_view()
        _CONTACT["view"] = sim_view.create_rigid_contact_view([kpath], filter_patterns=[[spath]], max_contact_data_count=64)
        _CONTACT["sub"] = omni.physx.get_physx_interface().subscribe_physics_on_step_events(_on_physics_step, pre_step=False, order=50)
    except Exception as err:                         # noqa: BLE001
        _CONTACT["error"] = repr(err)
        print(f"[honekit] no contact view: {err!r}", file=sys.stderr, flush=True)


def edge_force() -> tuple:
    """The contact force between the knife and the steel since this was last asked (one control step): its mean over
    the physics steps (N), the share of those steps with contact, and how many physics steps there were. If the
    per-step reading is not running, the force over the last physics step alone."""
    n = _CONTACT["steps"]
    if n:
        mean, share = _CONTACT["sum"] / n, _CONTACT["hits"] / n
    else:
        try:
            mean = _pair_force()
        except Exception as err:                     # noqa: BLE001
            _CONTACT["error"], mean = repr(err), 0.0
        share = 1.0 if mean > 1e-4 else 0.0
    _CONTACT["sum"], _CONTACT["steps"], _CONTACT["hits"] = 0.0, 0, 0
    return mean, share, n


def grip_forces(environment) -> dict:
    """The force each finger's motor is applying now (N), per arm: {"left": [f, f], "right": [f, f]}."""
    out = {}
    for arm in ("left", "right"):
        robot = environment.unwrapped.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        tau = None
        for attr in ("applied_torque", "computed_torque"):
            if hasattr(robot.data, attr):
                tau = _np(getattr(robot.data, attr))[0]
                break
        out[arm] = [round(abs(float(tau[names.index(j)])), 2) for j in ("joint7", "joint8")] if tau is not None else None
    return out


def _kit_state(environment, spec) -> KitState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pts = place(pos, quat, _SAMPLES[type(spec)])
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0))


def finger_points(environment) -> list:
    """The grasp point of each gripper, left then right (world, relative to the env origin), and its rotation."""
    from yam_arena.constants import TCP_OFFSET

    base = environment.unwrapped
    origin = _origin(environment)
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        rot = _rotation(pose[3:7])
        out.append((pose[:3] + rot @ np.array(TCP_OFFSET, float) - origin, rot))
    return out


def arm_points(environment) -> np.ndarray:
    """Points on both arms' collision shapes from the upper arm to the fingertips (world, relative to the env
    origin): every link's mesh, placed where the simulator has the link."""
    base = environment.unwrapped
    origin = _origin(environment)
    local = link_points()
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        poses = _np(robot.data.body_link_pose_w)[0]
        names = list(robot.body_names)
        for link in ARM_LINKS:
            if link in names and len(local[link]):
                pose = poses[names.index(link)]
                out.append(local[link] @ _rotation(pose[3:7]).T + pose[:3] - origin)
    return np.vstack(out)


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
    if getattr(scene, "_honekit_patched", False):
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
    scene._honekit_patched = True



def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_honekit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return stock_step(self, *a, **k)

    def reset(self, *a, **k):
        # The spring has to be whole before the reset's settling steps: switch it off, let the environment put the
        # board back, then author it afresh from where the board is.
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
            print(f"[honekit] spring not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
        _watch_contacts(self._env)
        edge_force()                                 # start the count afresh
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._honekit_patched = True


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
