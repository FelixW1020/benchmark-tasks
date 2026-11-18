"""guardkit: a 20 cm chef's knife and its plastic edge guard, for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from guardkit import Guard, Knife, on_reset, on_step

Frames (every part starts with its axes along the world's unless a ``quat`` is given: x away from the robot, y left,
z up):

- :class:`Knife`: ``pos`` is the heel, on the edge (the corner where the cutting edge starts, below the handle). Its x
  runs along the blade to the tip, its y from the edge up to the spine, its z through the blade's thickness. The
  handle runs back from the heel along -x.
- :class:`Guard`: ``pos`` is the corner of the guard's folded edge at its heel end, in the middle of its thickness.
  Its x runs along the guard, its y from the fold to the open side (the way the blade's spine points when the guard
  is on), its z through its thickness. The slot between its two walls is open along the far side (y) and at both
  ends.

The blade's and the guard's colliders are boxes (thin plates cannot be cooked as convex hulls for the GPU pipeline);
the blade is drawn as a smooth wedge. The same numbers serve the scene, the scoring and the offline checks. Isaac
Sim is reached only when the scene is built.

The structure follows the kits used by tasks already in this repository (Box subclasses so the loader accepts the
parts, custom spawners, hooks installed through ``sys.meta_path``).
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

__all__ = ["Guard", "Knife", "KitState", "arm_points", "edge_local", "edge_points", "env", "finger_points",
           "guard_boxes", "guard_points", "handle_pieces", "in_slot", "install", "knife_boxes", "knife_points",
           "on_reset", "on_step", "place", "to_local", "turn_z", "write_pose"]

# -- the knife (m): a stamped 20 cm (8 in) chef's knife with a moulded handle ------------------------------------------
BL = 0.200                    # blade length, heel to tip (maker: 7.9 in)
HL = 0.140                    # handle length (overall 13.4 in less the blade; a retailer lists 140 mm)
BH = 0.053                    # blade height at the heel, edge to spine (a retailer lists 2.1 in)
T_SPINE = 0.0023              # blade thickness at the spine, at the heel (a retailer lists 2.3 mm) ...
T_TIP = 0.0012                # ... and near the tip (an estimate)
STRAIGHT = 0.090              # the edge is straight this far from the heel (an estimate, from pictures) ...
TIP_RISE = 0.032              # ... then curves up to the tip, which stands this far above the heel's edge line
SPINE_FLAT = 0.120            # the spine is straight this far from the heel, then drops to the tip (estimates)
HANDLE_T = 0.020              # handle thickness (maker: 0.8 in for the knife lying flat)
HANDLE_H = 0.030              # handle height, back to belly (an estimate)
K_MASS = 0.181                # maker: 6.4 oz
K_COM = (-0.020, 0.036, 0.0)  # centre of mass: 2 cm behind the heel, in the front of the handle (an estimate)
K_INERTIA = (2.0e-5, 1.7e-3, 1.7e-3)   # kg m2 about the knife's x, y, z through the centre of mass (estimates)
EDGE_FROM = 0.004             # the edge is scored from this far in front of the heel to the tip

# -- the guard: a folded PVC edge guard sold for 8 in chef's knives -----------------------------------------------------
GL = 0.216                    # length (sold as 8.5 in)
GW = 0.0492                   # width, fold to open side (sold as 1 15/16 in)
G_WALL = 0.0012               # each wall's thickness (an estimate)
G_SLOT = 0.0035               # between the walls (an estimate: a real guard's walls lie closer and give)
G_FOLD = 0.0030               # the fold: the slot's bottom is this far in from the folded edge (an estimate)
G_T = G_SLOT + 2 * G_WALL     # the guard's thickness
G_MASS = 0.030                # an estimate (22 cm3 of PVC)
MOUTH = 0.005                 # a point of the edge counts as covered once it is this far inside the slot's open side

STEEL, HANDLE_BLACK, PVC = (0.72, 0.73, 0.75), (0.06, 0.06, 0.07), (0.16, 0.17, 0.20)
FRICTION = {"handle": (0.9, 0.8), "blade": (0.3, 0.25), "guard": (0.45, 0.4)}    # estimates
PAD_PATCH = 0.004             # radius of the patch over which a rubber gripper pad meets the handle (an estimate) ...
GUARD_PATCH = 0.006           # ... and the guard's flat face (an estimate)
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
_T_MIN = 0.0004                                        # the thinnest collider


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


def guard_boxes() -> list:
    """The guard's colliders: (lo, hi) boxes in the guard's frame: two walls and the fold that joins them."""
    h = G_SLOT / 2
    return [(np.array([0.0, 0.0, h]), np.array([GL, GW, h + G_WALL])),
            (np.array([0.0, 0.0, -h - G_WALL]), np.array([GL, GW, -h])),
            (np.array([0.0, 0.0, -h]), np.array([GL, G_FOLD, h]))]


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
class Guard(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = G_MASS


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


def _guard_samples() -> np.ndarray:
    xs, ys = np.linspace(0.0, GL, 10), np.linspace(0.0, GW, 4)
    return np.array([[x, y, z] for x in xs for y in ys for z in (-G_T / 2, G_T / 2)])


_GUARD = _guard_samples()


def guard_points(pos, quat) -> np.ndarray:
    """World points over the guard's two faces."""
    return place(pos, quat, _GUARD)


def in_slot(points, pos, quat, heel: float = 0.0, slack: float = 0.0003) -> tuple:
    """For each world point: is it inside the guard's slot (between the walls, between the slot's bottom and a line
    ``MOUTH`` in from the open side, within the guard's length, ``heel`` allowed beyond its heel end)? Also returns
    the points in the guard's frame."""
    loc = to_local(np.atleast_2d(points), pos, quat)
    inside = ((loc[:, 0] >= -heel) & (loc[:, 0] <= GL) & (loc[:, 1] >= G_FOLD - 0.0005) & (loc[:, 1] <= GW - MOUTH)
              & (np.abs(loc[:, 2]) <= G_SLOT / 2 + slack))
    return inside, loc


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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, com=None, inertia=None):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
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


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _spawn_knife(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32, com=K_COM, inertia=K_INERTIA)
    grip, steel = a.mat("handle", *FRICTION["handle"]), a.mat("blade", *FRICTION["blade"])
    for n, (pts, color) in enumerate(handle_pieces()):
        a.hull(f"{prim_path}/h_{n}", pts, a.look(color, rough=0.75), grip, patch=PAD_PATCH)
    for n, (pts, color) in enumerate(blade_pieces()):
        a.hull(f"{prim_path}/v_{n}", pts, a.look(color, rough=0.3, metal=0.9))
    for n, (lo, hi) in enumerate(knife_boxes()):
        a.box(f"{prim_path}/b_{n}", lo, hi, None, steel)
    return a.root


def _spawn_guard(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32, lin_damp=0.1, ang_damp=0.3)
    pvc = a.mat("guard", *FRICTION["guard"])
    look = a.look(PVC, rough=0.5, opacity=0.55)
    for n, (lo, hi) in enumerate(guard_boxes()):
        a.box(f"{prim_path}/g_{n}", lo, hi, look, pvc, patch=GUARD_PATCH)
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


_SPAWNERS = {Knife: _spawn_knife, Guard: _spawn_guard}
_SAMPLES = {Knife: _KNIFE, Guard: _GUARD}


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
    if getattr(scene, "_guardkit_patched", False):
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
    scene._guardkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_guardkit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return stock_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = stock_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._guardkit_patched = True


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
