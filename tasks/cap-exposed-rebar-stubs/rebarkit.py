"""rebarkit: impalement-rated rebar safety caps (a 4 in square steel-reinforced top on a round neck, with four sprung
fins and a steel insert inside), plain plastic mushroom caps, and a row of reinforcing-bar stubs standing in a board,
for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rebarkit import Cap, Fin, Footing, Mushroom, on_reset, on_step

Frames (m):

- :class:`Cap`: the rated cap's square top and round neck with the steel insert at the top of its bore, one loose rigid
  part. ``pos`` is the middle of its mouth (the open end of the neck); its z runs up the neck's axis to the top; the
  square top's sides are along its x and y.
- :class:`Fin`: one of a cap's four fins, a loose part that slides along a radius of the cap's bore and is sprung
  toward the axis. ``pos`` is on the cap's axis, FIN_Z up from the mouth, when the fin is at rest; its x points away
  from the axis, its z up the neck.
- :class:`Mushroom`: a plain plastic mushroom cap, one loose rigid part. ``pos`` is the middle of its mouth; its z runs
  up its stem to its head.
- :class:`Footing`: the board with the bar stubs standing in it, fixed. ``pos`` is the middle of the board's
  underside; the bars stand along its y at the offsets in ``bars``; its z is up.

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

__all__ = ["Cap", "Fin", "Footing", "KitState", "Mushroom", "arm_points", "bar_in_cap", "bar_in_mushroom", "cap_boxes",
           "cap_distance", "env", "fin_rest", "fin_travel", "fin_wires", "finger_frames", "finger_points", "gripper_points", "install",
           "lying", "mushroom_distance", "on_reset", "on_step", "place", "to_local", "turn_z", "write_pose"]

IN = 0.0254

# -- the rated cap (Cap frame) --------------------------------------------------------------------------------------------
PLATE = 4.0 * IN               # the square top: 4 x 4 in (16 sq in, the least the rules for these covers allow)
CAP_H = 2.75 * IN              # overall height (a listing gives 4 x 4 x 2.75 in)
PLATE_T = 0.013                # thickness of the square top (an estimate from the maker's photograph)
NECK_L = CAP_H - PLATE_T       # the round neck below it
NECK_R = 0.018                 # the neck's outside radius: 36 mm across its flats (an estimate from the photograph)
NECK_WALL = 0.003              # its wall (an estimate)
BORE_R = NECK_R - NECK_WALL    # the bore's radius: 15 mm, wide enough for a #8 bar
SIDES = 12                     # the neck is built from this many flat staves
INSERT_Z = 0.052               # the steel insert's underside above the mouth: how far a bar goes in (an estimate)
INSERT_R = 0.013
CAP_MASS = 0.227               # kg: 0.5 lb, the weight one maker lists for its 4 x 4 in cap for #3 to #8 bar
CAP_COM = 0.055                # height of the centre of mass above the mouth (an estimate: the steel is in the top)
NECK_PATCH = 0.009             # radius of the patch over which a rubber gripper pad meets the neck (an estimate)

# -- its four fins (Fin frame) ----------------------------------------------------------------------------------------------
FINS = 4
FIN_AZ = tuple(math.radians(a) for a in (45.0, 135.0, 225.0, 315.0))   # where they stand round the bore (from the cap's x)
FIN_Z = 0.032                  # the Fin frame's origin above the mouth: where a fin's lead-in meets its straight edge
FIN_IN = 0.004                 # at rest the fins leave 8 mm between them, less than a #3 bar's 9.5 mm
FIN_EDGE = 0.014               # length of a fin's straight edge, up the bore from FIN_Z
FIN_ROD = 0.0025               # a fin's bearing edge is rounded to this radius
LEAD_LOW = (0.0105, 0.004)     # the lower end of a fin's lead-in: radius of its inner edge, and height above the mouth
FIN_TRAVEL = 0.0055            # a fin can be pushed this far outward
FIN_PRELOAD = 0.70             # N holding a fin against its stop at rest (an estimate)
FIN_STIFF = 100.0              # N/m of spring (an estimate): 0.94 N on a #4 bar, which pushes each fin out 2.35 mm
FIN_DAMP = 2.0                 # N s/m
FIN_MASS = 0.002               # kg each: heavier than a plastic fin so that the contact solver holds it
LEAD_SLOPE = math.atan((LEAD_LOW[0] - FIN_IN) / (FIN_Z - LEAD_LOW[1]))   # the lead-in's angle from the axis (13 degrees)

# -- the mushroom cap (Mushroom frame) ------------------------------------------------------------------------------------
M_STEM_L = 0.045               # the stem below the head (every size of the mushroom cap is an estimate, see README)
M_STEM_R = 0.0125              # the stem's outside radius
M_WALL = 0.0025
M_HEAD_R = 0.0285              # the head: 57 mm (2 1/4 in) across
M_HEAD_T = 0.013
M_RIB_IN = 0.0068              # four fixed ribs inside the stem leave 13.6 mm between them
M_RIB_Z = (0.006, 0.040)
M_MASS = 0.009                 # kg: 0.02 lb, a maker's figure for its mushroom cap
M_PATCH = 0.004

# -- the bars and their board (Footing frame) -------------------------------------------------------------------------------
BAR_R = 0.00635                # a #4 bar: 1/2 in nominal diameter
BAR_SIDES = 12
BOARD = (0.100, 0.600, 0.019)  # a strip of 3/4 in plywood
BAR_UP = 0.141                 # the bars stand this far above the board

ORANGE, BLACK, STEEL, RUST, PLY = (0.95, 0.38, 0.06), (0.12, 0.12, 0.13), (0.62, 0.63, 0.64), (0.36, 0.24, 0.18), (0.72, 0.58, 0.40)
LIME = (0.70, 0.86, 0.12)
FRICTION = (0.5, 0.45)         # the caps, the bars and the board (estimates)
SLICK = (0.3, 0.25)            # a fin's rounded plastic edge (an estimate)
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


def segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _obb(points, centre, rot, size) -> np.ndarray:
    """Signed distance from each point to a box of ``size`` at ``centre`` with its edges along the columns of ``rot``."""
    q = np.abs((np.asarray(points, float).reshape(-1, 3) - centre) @ rot) - np.asarray(size, float) / 2
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def _staves(radius: float, wall: float, z0: float, z1: float, sides: int) -> list:
    """A tube as flat staves: (centre, rotation, size); ``radius`` is the outside distance from the axis to a flat.
    Stave 0 faces +x; a stave's x runs out of the tube, its y round it, its z along it."""
    out = []
    wide = 2 * radius * math.tan(math.pi / sides)
    for k in range(sides):
        az = 2 * math.pi * k / sides
        rot = _rz(az)
        out.append((rot @ np.array([radius - wall / 2, 0.0, (z0 + z1) / 2]), rot, np.array([wall, wide, z1 - z0])))
    return out


def cap_boxes() -> list:
    """The rated cap as boxes in the Cap's frame: (centre, rotation, size, what). The square top, the neck's staves
    and the steel insert that closes the top of the bore."""
    eye = np.eye(3)
    out = [(np.array([0.0, 0.0, NECK_L + PLATE_T / 2]), eye, np.array([PLATE, PLATE, PLATE_T]), "top")]
    out += [(c, r, s, "neck") for c, r, s in _staves(NECK_R, NECK_WALL, 0.0, NECK_L, SIDES)]
    out.append((np.array([0.0, 0.0, (INSERT_Z + NECK_L) / 2]), eye, np.array([2 * INSERT_R, 2 * INSERT_R, NECK_L - INSERT_Z]), "insert"))
    return out


def fin_wires() -> list:
    """One fin in its own frame, as round bars: (start, end, radius). A straight edge that bears on the bar, and
    below it a lead-in that slopes outward down toward the mouth, over which a bar's end pushes the fin aside."""
    r = FIN_ROD
    top = (FIN_IN + r, 0.0, FIN_EDGE)
    knee = (FIN_IN + r, 0.0, 0.0)
    low = (LEAD_LOW[0] + r, 0.0, LEAD_LOW[1] - FIN_Z)
    return [(knee, top, r), (low, knee, r)]


def fin_rest(which: int) -> tuple:
    """Where fin ``which`` (0 to 3) is at rest in the Cap's frame: its origin and its rotation."""
    return np.array([0.0, 0.0, FIN_Z]), _rz(FIN_AZ[which])


def fin_travel(which: int, cap_pos, cap_quat, fin_pos) -> float:
    """How far a fin has been pushed outward from its rest (m)."""
    origin, rot = fin_rest(which)
    return float((to_local(fin_pos, cap_pos, cap_quat) - origin) @ rot[:, 0])


def mushroom_boxes() -> list:
    """The mushroom cap's stem and ribs as boxes in its own frame: (centre, rotation, size, what)."""
    out = [(c, r, s, "stem") for c, r, s in _staves(M_STEM_R, M_WALL, 0.0, M_STEM_L, SIDES)]
    inner = M_STEM_R - M_WALL
    for k in range(4):
        rot = _rz(math.pi / 4 + k * math.pi / 2)
        out.append((rot @ np.array([(M_RIB_IN + inner) / 2, 0.0, sum(M_RIB_Z) / 2]), rot,
                    np.array([inner - M_RIB_IN, 0.002, M_RIB_Z[1] - M_RIB_Z[0]]), "rib"))
    return out


def _prism(z0: float, z1: float, r0: float, r1: float | None = None, sides: int = 12) -> np.ndarray:
    a = 2 * math.pi * np.arange(sides) / sides
    ring = np.column_stack([np.cos(a), np.sin(a)])
    r1 = r0 if r1 is None else r1
    return np.vstack([np.column_stack([r0 * ring, np.full(sides, z0)]), np.column_stack([r1 * ring, np.full(sides, z1)])])


def mushroom_head() -> list:
    """The mushroom cap's head in its own frame: convex pieces (points)."""
    return [_prism(M_STEM_L, M_STEM_L + 0.005, M_HEAD_R, M_HEAD_R, 16),
            _prism(M_STEM_L + 0.005, M_STEM_L + M_HEAD_T, M_HEAD_R, 0.6 * M_HEAD_R, 16)]


def bar_points(y: float, top: float) -> np.ndarray:
    """A bar stub in the Footing's frame: a 12-sided prism from the board's underside to ``top``."""
    p = _prism(0.0, top, BAR_R / math.cos(math.pi / BAR_SIDES), sides=BAR_SIDES)
    return p + np.array([0.0, y, 0.0])


def lying(heading: float, radius: float, wide: float, length: float) -> tuple:
    """A cap lying on the table on the edge of its head and the rim of its mouth, its mouth pointing along
    ``heading`` (rad from x) and down. ``radius``: the neck's outside radius; ``wide``: half the head's width;
    ``length``: from the mouth to the underside of the head. Returns the height of the middle of its mouth above the
    table, its quaternion, and its tilt (rad)."""
    tilt = math.atan((wide - radius) / length)
    # Cap z (mouth to head) points up at ``tilt`` and away from ``heading``; cap x lies in the vertical plane, down.
    turn = _rz(heading + math.pi)
    rot = turn @ _ry(math.pi / 2 - tilt)
    return radius * math.cos(tilt), _quat(rot), tilt


def bar_in_cap(cap_pos, cap_quat, tip) -> dict:
    """Where a bar's end is in a rated cap. ``depth``: how far up the bore from the mouth (m; INSERT_Z when the end is
    against the insert). ``off``: its distance from the cap's axis. ``tilt``: the cap's axis from the vertical (deg;
    180 for a cap put on upside down). ``inside``: the end is in the bore."""
    t = to_local(tip, cap_pos, cap_quat)
    off = float(math.hypot(t[0], t[1]))
    tilt = math.degrees(math.acos(float(np.clip(_rotation(cap_quat)[2, 2], -1.0, 1.0))))
    return {"depth": float(t[2]), "off": off, "tilt": tilt, "inside": bool(0.0 <= t[2] <= INSERT_Z + 0.003 and off <= BORE_R)}


def bar_in_mushroom(pos, quat, tip) -> dict:
    """The same for a mushroom cap: its bore runs from its mouth up to the underside of its head."""
    t = to_local(tip, pos, quat)
    off = float(math.hypot(t[0], t[1]))
    tilt = math.degrees(math.acos(float(np.clip(_rotation(quat)[2, 2], -1.0, 1.0))))
    return {"depth": float(t[2]), "off": off, "tilt": tilt,
            "inside": bool(0.0 <= t[2] <= M_STEM_L + 0.003 and off <= M_STEM_R - M_WALL)}


def cap_distance(points, cap_pos, cap_quat) -> float:
    """The least distance from any of ``points`` (world) to a rated cap's top and neck."""
    p = (np.asarray(points, float).reshape(-1, 3) - np.asarray(cap_pos, float)) @ _rotation(cap_quat)
    return float(min(_obb(p, c, r, s).min() for c, r, s, _ in cap_boxes()))


def mushroom_distance(points, pos, quat) -> float:
    """The least distance from any of ``points`` (world) to a mushroom cap (its stem, and its head taken as a disc)."""
    p = (np.asarray(points, float).reshape(-1, 3) - np.asarray(pos, float)) @ _rotation(quat)
    best = min(float(_obb(p, c, r, s).min()) for c, r, s, _ in mushroom_boxes())
    rad = np.maximum(np.hypot(p[:, 0], p[:, 1]) - M_HEAD_R, 0.0)
    up = np.maximum(np.maximum(M_STEM_L - p[:, 2], p[:, 2] - M_STEM_L - M_HEAD_T), 0.0)
    return float(min(best, np.hypot(rad, up).min()))


# -- the parts -----------------------------------------------------------------------------------------------------------

class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Cap(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = CAP_MASS - FINS * FIN_MASS


@dataclass(frozen=True)
class Fin(_Part):
    """Fin ``which`` (0 to 3) of ``cap``."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = FIN_MASS
    cap: str = "cap"
    which: int = 0


@dataclass(frozen=True)
class Mushroom(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = M_MASS
    color: tuple = LIME


@dataclass(frozen=True)
class Footing(_Part):
    """The board and its bars; ``bars`` are the bars' places along the board's y, ``top`` their height above its underside."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 5.0
    bars: tuple = (0.0,)
    top: float = BOARD[2] + BAR_UP


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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, com=None, fixed=False):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        if fixed:
            rigid.CreateKinematicEnabledAttr(True)
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

    def _collide(self, prim, mat, offset, patch=0.0):
        from isaaclab.sim.utils import bind_physics_material
        from pxr import PhysxSchema, UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(offset)
        cp.CreateRestOffsetAttr(0.0)
        if patch:                                    # a rubber pad meets the part over a patch, which resists twisting
            cp.CreateTorsionalPatchRadiusAttr(float(patch))
            cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
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

    def bar(self, path, centre, rot, size, look, mat=None, offset=0.0008, patch=0.0):
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
            self._collide(prim, mat, offset, patch)
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


def _start(cfg, prim_path, translation, orientation, iterations=None, lin_damp=0.05, ang_damp=0.1, com=None, fixed=False):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, lin_damp=lin_damp, ang_damp=ang_damp, com=com, fixed=fixed)
    return spec, a


def _spawn_cap(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, ang_damp=0.3, com=(0.0, 0.0, CAP_COM))
    m = a.mat("part", *FRICTION)
    looks = {"top": a.look(ORANGE, rough=0.5), "neck": a.look(BLACK, rough=0.6), "insert": a.look(STEEL, rough=0.35, metal=0.9)}
    for n, (centre, rot, size, what) in enumerate(cap_boxes()):
        a.bar(f"{prim_path}/b_{n}", centre, rot, size, looks[what], m, patch=NECK_PATCH if what == "neck" else 0.0)
    return a.root


def _spawn_fin(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, lin_damp=0.2, ang_damp=0.5)
    m, look = a.mat("fin", *SLICK), a.look(BLACK, rough=0.6)
    for n, (p0, p1, r) in enumerate(fin_wires()):
        a.wire(f"{prim_path}/w_{n}", p0, p1, r, look, m)
    return a.root


def _spawn_mushroom(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, iterations=32, ang_damp=0.3)
    m, look = a.mat("part", *FRICTION), a.look(spec.color, rough=0.5)
    for n, (centre, rot, size, what) in enumerate(mushroom_boxes()):
        a.bar(f"{prim_path}/b_{n}", centre, rot, size, look, m, patch=M_PATCH if what == "stem" else 0.0)
    for n, pts in enumerate(mushroom_head()):
        a.hull(f"{prim_path}/h_{n}", pts, look, m)
    return a.root


def _spawn_footing(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, fixed=True)
    m = a.mat("part", *FRICTION)
    a.bar(f"{prim_path}/board", (0.0, 0.0, BOARD[2] / 2), np.eye(3), BOARD, a.look(PLY, rough=0.8), m)
    look = a.look(RUST, rough=0.7, metal=0.4)
    for n, y in enumerate(spec.bars):
        a.hull(f"{prim_path}/bar_{n}", bar_points(float(y), float(spec.top)), look, m)
    return a.root


# -- joints ------------------------------------------------------------------------------------------------------------

def _joints() -> list:
    """Every joint the parts in the scene need: (name, cap, fin, which)."""
    return [(f"{n}_slide", s.cap, n, s.which) for n, s in _SPECS.items()
            if isinstance(s, Fin) and n in _PATHS and s.cap in _PATHS]


def _joint_path(joint) -> str:
    return f"{_PATHS[joint[1]]}/{joint[0]}"


def _author(joint) -> None:
    """Author one fin's slide: disabled first, its frames set, enabled last."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    name, cap, fin, which = joint
    origin, rot = fin_rest(which)
    qx, qy, qz, qw = _quat(rot)
    stage = get_current_stage()
    j = UsdPhysics.PrismaticJoint.Define(stage, _joint_path(joint))
    j.CreateJointEnabledAttr().Set(False)
    j.CreateBody0Rel().SetTargets([_PATHS[cap]])
    j.CreateBody1Rel().SetTargets([_PATHS[fin]])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in origin)))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    # The fin slides along its own x, a radius of the bore: 0 at rest against its inner stop, FIN_TRAVEL pushed out.
    # A spring pushes it toward a target inside the stop: FIN_PRELOAD holds it there, and every metre outward adds
    # FIN_STIFF.
    j.CreateAxisAttr().Set("X")
    j.CreateLowerLimitAttr().Set(0.0)
    j.CreateUpperLimitAttr().Set(float(FIN_TRAVEL))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(FIN_STIFF))
    drive.CreateDampingAttr().Set(float(FIN_DAMP))
    drive.CreateTargetPositionAttr().Set(float(-FIN_PRELOAD / FIN_STIFF))
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


_SPAWNERS = {Cap: _spawn_cap, Fin: _spawn_fin, Mushroom: _spawn_mushroom, Footing: _spawn_footing}
_BOUNDS: dict = {}


def _points_of(spec) -> np.ndarray:
    if isinstance(spec, Cap):
        h = PLATE / 2
        return np.array([[sx * h, sy * h, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, CAP_H)])
    if isinstance(spec, Fin):
        return np.array([[FIN_IN, -FIN_ROD, LEAD_LOW[1] - FIN_Z - FIN_ROD], [LEAD_LOW[0] + 2 * FIN_ROD, FIN_ROD, FIN_EDGE + FIN_ROD]])
    if isinstance(spec, Mushroom):
        r = M_HEAD_R
        return np.array([[sx * r, sy * r, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, M_STEM_L + M_HEAD_T)])
    ys = [float(y) for y in spec.bars]
    return np.array([[-BOARD[0] / 2, min(-BOARD[1] / 2, min(ys) - BAR_R), 0.0], [BOARD[0] / 2, max(BOARD[1] / 2, max(ys) + BAR_R), float(spec.top)]])


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
    """Join the parts as the reset left them: each fin to its cap."""
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


def finger_frames(environment) -> list:
    """Each gripper's own rotation as the robot has it, left then right: columns are its closing axis, the axis
    across its fingers, and the axis its fingers point along."""
    base = environment.unwrapped
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        out.append(_rotation(pose[3:7]))
    return out


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
    if getattr(scene, "_rebarkit_patched", False):
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
    scene._rebarkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_rebarkit_patched", False):
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
            print(f"[rebarkit] joints not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._rebarkit_patched = True


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
