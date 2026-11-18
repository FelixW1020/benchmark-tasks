"""timerkit: a two-part wind-up (clockwork) kitchen timer for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from timerkit import Base, Top, on_reset, on_step

The timer is a cylinder in two parts that turn on each other about its axis:

- :class:`Base`: the lower part, the one the maker's drawing shows being turned. It carries the minute scale.
  ``pos`` is the middle of its underside; its z runs up the timer's axis, its x points at the scale's zero mark.
- :class:`Top`: the upper part, which carries the index mark. ``pos`` is the middle of its lower face (at the seam);
  its z runs up the axis, its x points at the index mark.

Between them are the three things a hand feels in a clockwork timer:

- a hinge about the axis with a stop at zero and a stop one turn on. The simulator's hinge stops cannot be more than
  half a turn apart, so the stops are carried by two more hinges in series through a small idler wheel
  (:class:`Train`): the base to the idler and the idler to the top, each with stops half the range apart;
- the going train behind its slipping clutch: left alone, the timer runs back toward zero at the clock's rate
  (6 degrees a minute); a hand has to overcome the clutch to turn it either way;
- the mainspring, wound by turning the lower part clockwise (seen from above) against the upper, which pulls it
  back toward zero the harder the further it is wound.

The reading is the angle the lower part has been turned clockwise (seen from above) against the upper part from
zero, 6 degrees to the minute. That is the same as the upper part turned anticlockwise against the lower.

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

__all__ = ["Base", "KitState", "Top", "Train", "axis_distance", "dial", "env", "finger_points", "gripper_points", "index_quads",
           "install", "minutes", "on_reset", "on_step", "place", "relative_turn", "scale_quads", "spring_torque",
           "tilt", "to_local", "turn_z", "unwrap"]

# -- the timer ---------------------------------------------------------------------------------------------------------
RADIUS = 0.030                # 6 cm across (the maker's size)
HEIGHT = 0.060                # 6 cm tall (the maker's size)
SPLIT = 0.026                 # the lower part's height (an estimate, from the drawing in the maker's instructions)
SEAM = 0.0006                 # the gap between the two parts (an estimate)
TOP_H = HEIGHT - SPLIT - SEAM
BASE_MASS, TOP_MASS = 0.061, 0.045     # kg (estimates: the lower part holds the clockwork)
TRAIN_MASS = 0.004            # kg: the idler wheel between the two halves of the hinge (counted with the lower part)
TRAIN_INERTIA = 1.0e-6        # kg m2 about each of its axes

DEG_PER_MIN = 6.0             # a 60-minute scale round one turn
STOP_DEG = 357.0              # the far stop: 59.5 minutes (an estimate; such dials stop about one turn from zero)
RUN_DEG_S = DEG_PER_MIN / 60  # the clockwork runs the top back at 6 degrees a minute
CLUTCH = 0.035                # N m: the going train's slipping clutch (an estimate; no maker publishes it)
SPRING_0, SPRING_1 = 0.004, 0.020      # N m: the mainspring's pull at zero and at one full turn (estimates)
GOVERNOR = 3.0                # N m s/deg: how stiffly the escapement holds the run-down rate below the clutch's limit

# -- the marks (drawn only) ----------------------------------------------------------------------------------------------
TICK_TOP = SPLIT - 0.0010     # the scale sits just under the seam
TICK_MIN, TICK_FIVE = 0.0020, 0.0038
DIGIT_H, DIGIT_W, DIGIT_BAR, DIGIT_GAP = 0.0046, 0.0026, 0.0006, 0.0008
DIGIT_TOP = TICK_TOP - TICK_FIVE - 0.0012
INDEX_H, INDEX_W = 0.0050, 0.0018
PROUD = 0.00025               # the marks stand this far off the metal, so that they are drawn cleanly

STEEL, INK, RED, DARK = (0.75, 0.76, 0.78), (0.07, 0.07, 0.08), (0.80, 0.08, 0.08), (0.20, 0.21, 0.23)
FRICTION = (0.5, 0.4)
PAD_PATCH = 0.004             # radius of the patch over which a rubber gripper pad meets the steel (an estimate)
PALM_BACK = 0.065
FINGER_THICK = 0.012


# -- geometry (pure) -------------------------------------------------------------------------------------------------

def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def turn_z(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn of ``angle`` rad about the vertical."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def place(pos, quat, local) -> np.ndarray:
    """A point given in a part's own frame, in the world."""
    return np.asarray(pos, float) + _rotation(quat) @ np.asarray(local, float)


def to_local(point, pos, quat) -> np.ndarray:
    return _rotation(quat).T @ (np.asarray(point, float) - np.asarray(pos, float))


def relative_turn(base_quat, top_quat) -> float:
    """How far the base is turned clockwise (seen from above the timer) against the top, in degrees, between -180
    and 180: the same as the top turned anticlockwise against the base. One turn is lost each time it passes 180:
    see :func:`unwrap`."""
    rel = _rotation(base_quat).T @ _rotation(top_quat)
    return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def unwrap(turn: float, before: float) -> float:
    """``turn`` (degrees, known only up to whole turns) as the value nearest ``before``."""
    return turn + 360.0 * round((before - turn) / 360.0)


def minutes(deg: float) -> float:
    """The timer's reading for a dial turned ``deg`` degrees from zero."""
    return deg / DEG_PER_MIN


def spring_torque(deg: float) -> float:
    """The mainspring's pull toward zero (N m) with the dial ``deg`` degrees from zero."""
    return SPRING_0 + (SPRING_1 - SPRING_0) * min(max(deg, 0.0), 360.0) / 360.0


def tilt(quat) -> float:
    """How far a part's axis leans from the vertical (degrees; 180 is upside down)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_rotation(quat)[2, 2])))))


def axis_distance(point, base_pos, base_quat) -> float:
    """How far ``point`` is from the timer's surface (m; negative inside), the timer taken as one cylinder standing
    on the base's origin."""
    p = to_local(point, base_pos, base_quat)
    radial, along = math.hypot(p[0], p[1]) - RADIUS, max(-p[2], p[2] - HEIGHT)
    if radial <= 0.0 and along <= 0.0:
        return max(radial, along)
    return math.hypot(max(radial, 0.0), max(along, 0.0))


def _quad(azimuth: float, half_width: float, z0: float, z1: float) -> list:
    """A mark on the cylinder's side: four corners, ``azimuth`` rad round from the part's x, between two heights."""
    r = RADIUS + PROUD
    a0, a1 = azimuth - half_width / RADIUS, azimuth + half_width / RADIUS
    return [(r * math.cos(a0), r * math.sin(a0), z0), (r * math.cos(a1), r * math.sin(a1), z0),
            (r * math.cos(a1), r * math.sin(a1), z1), (r * math.cos(a0), r * math.sin(a0), z1)]


# seven-segment numerals: which bars each digit lights (a top, b upper right, c lower right, d bottom, e lower left,
# f upper left, g middle)
_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
             "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def _digit(azimuth: float, z_top: float, digit: str) -> list:
    """The bars of one numeral whose middle is ``azimuth`` round the cylinder and whose top is at ``z_top``. Seen
    from outside, the numeral's right is the way the azimuth grows."""
    w, h, t = DIGIT_W / 2, DIGIT_H, DIGIT_BAR
    mid = z_top - h / 2

    def bar(x0, x1, z0, z1):
        return _quad(azimuth + (x0 + x1) / 2 / RADIUS, (x1 - x0) / 2, z0, z1)

    bars = {"a": bar(-w, w, z_top - t, z_top), "g": bar(-w, w, mid - t / 2, mid + t / 2), "d": bar(-w, w, z_top - h, z_top - h + t),
            "f": bar(-w, -w + t, mid, z_top), "b": bar(w - t, w, mid, z_top),
            "e": bar(-w, -w + t, z_top - h, mid), "c": bar(w - t, w, z_top - h, mid)}
    return [bars[s] for s in _SEGMENTS[digit]]


def mark_azimuth(minute: float) -> float:
    """Where the scale's mark for ``minute`` is round the base (rad from the base's x). As the base is wound
    clockwise (seen from above) under the top, the top's index, which starts over zero, comes to stand over marks
    further anticlockwise round the base: seen from the side, the numerals rise from left to right."""
    return math.radians(DEG_PER_MIN * minute)


def scale_quads() -> list:
    """The base's scale, as flat four-cornered marks in the Base's frame: a tick for every minute, a longer one and
    a numeral for every fifth."""
    out = []
    for m in range(60):
        five = m % 5 == 0
        out.append(_quad(mark_azimuth(m), 0.00045 if five else 0.00025, TICK_TOP - (TICK_FIVE if five else TICK_MIN), TICK_TOP))
        if five:
            text = str(m)
            width = len(text) * DIGIT_W + (len(text) - 1) * DIGIT_GAP
            for i, ch in enumerate(text):
                x = -width / 2 + DIGIT_W / 2 + i * (DIGIT_W + DIGIT_GAP)
                out += _digit(mark_azimuth(m) + x / RADIUS, DIGIT_TOP, ch)
    return out


def index_quads() -> list:
    """The top's index mark, in the Top's frame: a bar on its side just above the seam, and a line across its upper
    face from the middle to the rim over the bar, so that the top's turn shows from above."""
    side = _quad(0.0, INDEX_W / 2, 0.0006, 0.0006 + INDEX_H)
    z = TOP_H + PROUD
    face = [(0.004, -INDEX_W / 2, z), (RADIUS - 0.002, -INDEX_W / 2, z), (RADIUS - 0.002, INDEX_W / 2, z), (0.004, INDEX_W / 2, z)]
    return [side, face]


# -- the parts -----------------------------------------------------------------------------------------------------------

class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Base(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = BASE_MASS
    fixed: bool = False               # for bench checks of the mechanism only


@dataclass(frozen=True)
class Train(_Part):
    """The idler wheel inside the timer whose lower part is ``base``: nothing to see or touch. ``pos`` is the middle
    of the seam."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = TRAIN_MASS
    base: str = "timer"


@dataclass(frozen=True)
class Top(_Part):
    """The upper part of the timer whose lower part is ``base`` and whose idler wheel is ``train``."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = TOP_MASS
    base: str = "timer"
    train: str = "timer_train"


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

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, fixed=False, inertia=None):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        if fixed:
            rigid.CreateKinematicEnabledAttr(True)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(mass))
        if inertia:                                  # a body with no shape of its own
            api.CreateCenterOfMassAttr(self.Gf.Vec3f(0.0, 0.0, 0.0))
            api.CreateDiagonalInertiaAttr(self.Gf.Vec3f(float(inertia), float(inertia), float(inertia)))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        rb.CreateSleepThresholdAttr(0.0)             # the clockwork turns far too slowly to keep a body awake
        rb.CreateStabilizationThresholdAttr(0.0)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def cylinder(self, path, z0, z1, radius, look, mat=None, offset=0.0008, patch=0.0):
        """A round piece about the part's z between two heights, with a collider if ``mat`` is given."""
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from pxr import PhysxSchema, UsdPhysics

        cyl = self.UsdGeom.Cylinder.Define(self.stage, path)
        cyl.CreateAxisAttr("Z")
        cyl.CreateHeightAttr(float(z1 - z0))
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateExtentAttr([(-radius, -radius, -(z1 - z0) / 2), (radius, radius, (z1 - z0) / 2)])
        cyl.AddTranslateOp().Set(self.Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
        prim = cyl.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            UsdPhysics.CollisionAPI.Apply(prim)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(offset)
            cp.CreateRestOffsetAttr(0.0)
            if patch:                                # a rubber pad meets the part over a patch, which resists twisting
                cp.CreateTorsionalPatchRadiusAttr(float(patch))
                cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def quads(self, path, quads, look):
        """Flat four-cornered marks as one mesh, drawn only."""
        from isaaclab.sim.utils import bind_visual_material
        from pxr import Vt

        pts = [p for q in quads for p in q]
        mesh = self.UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([self.Gf.Vec3f(*(float(v) for v in p)) for p in pts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(pts)))))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        bind_visual_material(str(mesh.GetPrim().GetPath()), look, stage=self.stage)
        return mesh.GetPrim()


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _start(cfg, prim_path, translation, orientation, fixed=False, inertia=None):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32, fixed=fixed, inertia=inertia)
    return spec, a, a.mat("part", *FRICTION)


def _spawn_base(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=bool(spec_fixed(cfg)))
    a.cylinder(f"{prim_path}/body", 0.0, SPLIT, RADIUS, a.look(STEEL, rough=0.35, metal=0.85), m, patch=PAD_PATCH)
    a.cylinder(f"{prim_path}/neck", SPLIT, SPLIT + SEAM, RADIUS - 0.0015, a.look(DARK, rough=0.6))     # the seam's shadow
    a.quads(f"{prim_path}/scale", scale_quads(), a.look(INK, rough=0.8))
    return a.root


def spec_fixed(cfg) -> bool:
    return bool(getattr(_SPECS[cfg.kit_name], "fixed", False))


def _spawn_train(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, inertia=TRAIN_INERTIA)
    return a.root


def _spawn_top(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation)
    a.cylinder(f"{prim_path}/body", 0.0, TOP_H, RADIUS, a.look(STEEL, rough=0.35, metal=0.85), m, patch=PAD_PATCH)
    a.quads(f"{prim_path}/index", index_quads(), a.look(RED, rough=0.7))
    return a.root


# -- the hinge -----------------------------------------------------------------------------------------------------------

def _hinges() -> list:
    """Every timer's joints: (name, kind, first body, second body, anchor in the first, top). "hinge" joins the base
    to the top and carries the going train; the two "stop" joints, base to idler and idler to top, carry the stops."""
    out = []
    for n, s in _SPECS.items():
        if isinstance(s, Top) and all(k in _PATHS for k in (n, s.base, s.train)):
            out.append((f"{n}_hinge", "hinge", s.base, n, (0.0, 0.0, SPLIT + SEAM), n))
            out.append((f"{n}_stop_a", "stop", s.base, s.train, (0.0, 0.0, SPLIT + SEAM / 2), n))
            out.append((f"{n}_stop_b", "stop", s.train, n, (0.0, 0.0, SEAM / 2), n))
    return out


def _hinge_path(hinge) -> str:
    return f"{_PATHS[hinge[2]]}/{hinge[0]}"


def _author(hinge) -> None:
    """Author one joint: disabled first, its frames set, enabled last. Its angle is the second body's turn about
    the axis from the first, anticlockwise seen from above; the reading is that angle of the top from the base."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdPhysics

    name, kind, first, second, anchor, _top = hinge
    stage = get_current_stage()
    j = UsdPhysics.RevoluteJoint.Define(stage, _hinge_path(hinge))
    j.CreateJointEnabledAttr().Set(False)
    j.CreateBody0Rel().SetTargets([_PATHS[first]])
    j.CreateBody1Rel().SetTargets([_PATHS[second]])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in anchor)))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    j.CreateAxisAttr().Set("Z")
    if kind == "stop":
        # The simulator's hinge stops cannot be more than half a turn apart, so the stops are carried by two
        # hinges in series through the idler, each with half the range.
        j.CreateLowerLimitAttr().Set(0.0)                       # the zero stop
        j.CreateUpperLimitAttr().Set(float(STOP_DEG) / 2)       # this half's share of the way to the far stop
    else:
        # The going train: it runs the timer back toward zero at the clock's rate, and gives way (the clutch
        # slips) to any torque over CLUTCH.
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(float(GOVERNOR))
        drive.CreateTargetVelocityAttr().Set(-float(RUN_DEG_S))
        drive.CreateMaxForceAttr().Set(float(CLUTCH))
    PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
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


_SPAWNERS = {Base: _spawn_base, Train: _spawn_train, Top: _spawn_top}


def _local_bounds(spec) -> tuple:
    if isinstance(spec, Train):
        return np.array([-0.005, -0.005, -SEAM / 2]), np.array([0.005, 0.005, SEAM / 2])
    h = SPLIT if isinstance(spec, Base) else TOP_H
    return np.array([-RADIUS, -RADIUS, 0.0]), np.array([RADIUS, RADIUS, h])


# -- run time ----------------------------------------------------------------------------------------------------------

def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    if hasattr(x, "numpy") and not hasattr(x, "detach"):
        return np.asarray(x.numpy(), dtype=np.float64)
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _origin(environment) -> np.ndarray:
    return _np(environment.unwrapped.scene.env_origins[0])


def _disarm() -> None:
    """Switch every hinge off (before a reset puts the parts back)."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for hinge in _hinges():
        prim = stage.GetPrimAtPath(_hinge_path(hinge))
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)


def _arm(environment) -> None:
    """Join the parts as the reset left them, with the dial at zero and the mainspring's pull as it is there."""
    for hinge in _hinges():
        _author(hinge)
        _DIAL[hinge[5]] = 0.0
    _wind(environment)


def _kit_state(environment, spec) -> KitState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    lo, hi = _local_bounds(spec)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = corners @ _rotation(quat).T + pos
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0))


_DIAL: dict = {}                       # each timer's turn from zero (degrees), by its top's name, followed step by step


def dial(top: str) -> float:
    """How far the timer whose upper part is ``top`` has been turned from zero (degrees), as followed since the last
    reset."""
    return float(_DIAL.get(top, 0.0))


def _torque(environment, name: str, torque) -> None:
    """Hold ``torque`` (N m, world) on a part at every physics step until changed."""
    import torch

    asset = environment.unwrapped.scene[name]
    t = torch.tensor([[[float(v) for v in torque]]], dtype=torch.float32, device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=torch.zeros_like(t), torques=t, body_ids=[0], is_global=True)


def _wind(environment) -> None:
    """Follow each dial through this step, and set the mainspring's pull for the next: on the base toward zero and
    the opposite on the top."""
    scene = environment.unwrapped.scene
    for _, kind, base, top, _anchor, _top in _hinges():
        if kind != "hinge":
            continue
        qb = _np(scene[base].data.root_link_pose_w)[0][3:7]
        qt = _np(scene[top].data.root_link_pose_w)[0][3:7]
        _DIAL[top] = unwrap(relative_turn(qb, qt), _DIAL.get(top, 0.0))
        pull = spring_torque(_DIAL[top]) * _rotation(qb)[:, 2]         # about the axis, anticlockwise seen from above
        try:
            _torque(environment, base, pull)                           # the base back anticlockwise, toward zero
            _torque(environment, top, -pull)
        except Exception as err:                     # noqa: BLE001
            _warn(f"the mainspring's pull could not be applied: {err!r}")


def _warn(text: str) -> None:
    if text not in _WARNED:
        _WARNED.append(text)
        print(f"[timerkit] {text}", file=sys.stderr, flush=True)


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


def wrist_rolls(environment) -> list:
    """Each arm's last joint angle (rad), left then right: the robot's own reading of how far its wrist has rolled."""
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = environment.unwrapped.scene[arm]
        out.append(float(_np(robot.data.joint_pos)[0][list(robot.joint_names).index("joint6")]))
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
    if getattr(scene, "_timerkit_patched", False):
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
    scene._timerkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_timerkit_patched", False):
        return
    stock_step, stock_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        _wind(self._env)
        for fn in _STEP:
            fn(self._env)
        return stock_step(self, *a, **k)

    def reset(self, *a, **k):
        # The hinge has to be whole before the reset's settling steps and first picture: switch it off, let the
        # environment put the parts back, then author it afresh from where the parts are.
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
            print(f"[timerkit] the hinge was not re-made around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._timerkit_patched = True


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
