"""ballkit: tennis balls that bounce as the rules of tennis say they should, or as a ball that has lost its pressure
does, for yam_arena task files; with the boards of a shallow wooden tray and of two small open boxes.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from ballkit import Ball, on_reset, on_step

Frames (m):

- :class:`Ball`: one loose rigid sphere. ``pos`` is its centre. ``bounce`` is its coefficient of restitution on a
  rigid surface (the speed it leaves at, over the speed it arrives at); ``dots`` is the number of pen dots that tell
  it apart from the others (drawn at both of its poles).
- :func:`tray_boxes` and :func:`bin_boxes` give the boards of an open tray and of an open bin or box as plain fixed boxes
  (centre, size), for ``yam_arena.task_file.Box``.

Every shape is built from numbers in NumPy when the task file is read, so the same numbers serve the scene, the
scoring and the offline checks. Isaac Sim is reached only when the scene is built.

The structure follows the kits used by tasks already in this repository (Box subclasses so the loader accepts the
parts, custom spawners, hooks installed through ``sys.meta_path``).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Ball", "KitState", "arm_points", "bin_boxes", "env", "finger_gaps", "finger_points", "gripper_points",
           "install", "on_reset", "on_step", "rebound", "seam_points", "tray_boxes", "write_pose"]

# -- the ball (International Tennis Federation, rules of tennis, appendix I, and its ball test methods) -----------------
BALL_D = 0.0660                # diameter: the ITF's high-specification test ball (6.60 cm; the rule allows 6.54 to 6.86)
BALL_R = BALL_D / 2
BALL_MASS = 0.0576             # kg: the same test ball (57.6 g; the rule allows 56.0 to 59.4)
DROP_ITF = 2.54                # the rebound test drops the ball 254 cm, measured from its bottom ...
REBOUND_ITF = (1.35, 1.47)     # ... and a ball in play (type 2) comes back 135 to 147 cm
REBOUND_GOOD = 1.41            # the high-specification test ball: 141 cm
BOUNCE_GOOD = math.sqrt(REBOUND_GOOD / DROP_ITF)    # its coefficient of restitution on a rigid block: 0.745
PUNCTURED = 0.80               # a pressurized ball keeps this share of its coefficient of restitution once its
                               # pressure is gone (Haake, Carre and Goodwill 2003: "decreased by 20% when punctured")
BOUNCE_DEAD = PUNCTURED * BOUNCE_GOOD               # 0.596: it would come back 90 cm in the ITF test
FELT = (0.6, 0.5)              # friction of the felt, static and dynamic (estimates)
ROLL_DAMP = 4.0                # 1/s: stands in for the felt's rolling resistance (an estimate)
OPTIC = (0.80, 0.86, 0.16)     # optic yellow
SEAM = (0.93, 0.93, 0.88)
INK = (0.05, 0.05, 0.06)
DOT_R = 0.0035                 # a pen dot
SEAM_R = 0.0016                # half the seam's width


def rebound(drop: float, bounce: float) -> float:
    """How high the bottom of a ball comes back (m) when it is dropped with its bottom ``drop`` above a rigid surface."""
    return bounce * bounce * drop


def seam_points(n: int = 64, radius: float = BALL_R) -> np.ndarray:
    """Points along the seam of a tennis ball (the one closed curve that divides it into two dumbbell-shaped panels)."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    a, b = 0.75, 0.25
    p = np.column_stack([a * np.cos(t) + b * np.cos(3 * t), a * np.sin(t) - b * np.sin(3 * t), 2 * math.sqrt(a * b) * np.sin(2 * t)])
    return radius * p / np.linalg.norm(p, axis=1, keepdims=True)


def dot_points(dots: int, radius: float = BALL_R) -> np.ndarray:
    """Where a ball's pen dots are: ``dots`` of them in a row at each pole."""
    out = []
    for pole in (1.0, -1.0):
        for k in range(dots):
            ang = (k - (dots - 1) / 2) * 0.26
            out.append(radius * np.array([math.sin(ang), 0.0, pole * math.cos(ang)]))
    return np.array(out)


# -- the tray and the bin (pure) ---------------------------------------------------------------------------------------
def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _open_box(centre_xy, yaw: float, inner: tuple, floor: float, wall: float, rim: float) -> list:
    """The five boards of an open box standing on z = 0: (tag, centre, size), sizes along the box's own axes. ``inner``
    is its inside length and width, ``rim`` the height of its walls above its floor."""
    lx, ly = inner
    rot = _rz(yaw)
    at = lambda x, y, z: np.array([centre_xy[0], centre_xy[1], 0.0]) + rot @ np.array([x, y, z])   # noqa: E731
    h = floor + rim
    return [("floor", at(0.0, 0.0, floor / 2), (lx + 2 * wall, ly + 2 * wall, floor)),
            ("wall_a", at(0.0, (ly + wall) / 2, h / 2), (lx + 2 * wall, wall, h)),
            ("wall_b", at(0.0, -(ly + wall) / 2, h / 2), (lx + 2 * wall, wall, h)),
            ("wall_c", at((lx + wall) / 2, 0.0, h / 2), (wall, ly, h)),
            ("wall_d", at(-(lx + wall) / 2, 0.0, h / 2), (wall, ly, h))]


TRAY_INNER = (0.304, 0.076)    # inside length and width: four balls in a row, 7.6 cm apart
TRAY_FLOOR = 0.006
TRAY_WALL = 0.008
TRAY_RIM = 0.015               # the rim stands this far above the tray's floor: a ball's middle is well above it
TRAY_PITCH = 0.076
WOOD = (0.72, 0.56, 0.36)

BIN_INNER = (0.150, 0.150)     # the bin and the box are the same size: two balls lie side by side in either
BIN_FLOOR = 0.004
BIN_WALL = 0.004
BIN_RIM = 0.090
BIN_COLOR = (0.30, 0.31, 0.33)   # grey
BOX_COLOR = (0.13, 0.33, 0.62)   # blue


def tray_boxes(centre_xy, yaw: float = 0.0) -> list:
    return _open_box(centre_xy, yaw, TRAY_INNER, TRAY_FLOOR, TRAY_WALL, TRAY_RIM)


def bin_boxes(centre_xy, yaw: float = 0.0) -> list:
    return _open_box(centre_xy, yaw, BIN_INNER, BIN_FLOOR, BIN_WALL, BIN_RIM)


def tray_slot(centre_xy, yaw: float, k: int) -> np.ndarray:
    """The middle of ball ``k`` (0 to 3) as it lies in the tray, above the table (z = 0)."""
    along = (k - 1.5) * TRAY_PITCH
    return np.array([centre_xy[0], centre_xy[1], 0.0]) + _rz(yaw) @ np.array([along, 0.0, TRAY_FLOOR + BALL_R])


def inside(point, centre_xy, yaw: float, inner: tuple, margin: float = 0.0) -> bool:
    """Is ``point`` over the inside of an open box (its floor, less ``margin`` all round)?"""
    d = _rz(-yaw) @ (np.array([point[0], point[1], 0.0]) - np.array([centre_xy[0], centre_xy[1], 0.0]))
    return bool(abs(d[0]) <= inner[0] / 2 - margin and abs(d[1]) <= inner[1] / 2 - margin)


# -- the parts -----------------------------------------------------------------------------------------------------------
class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Ball(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = BALL_MASS
    bounce: float = BOUNCE_GOOD
    dots: int = 1


@dataclass(frozen=True)
class KitState(ObjectState):
    """A kit object as the simulator has it; ``pos`` is its own origin (a ball's centre)."""


def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- spawning ----------------------------------------------------------------------------------------------------------
_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _spawn_ball(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    stage = get_current_stage()
    double = UsdGeom.XformOp.PrecisionDouble
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=double).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=double).Set(Gf.Quatd(w, x, y, z))
    prim = root.GetPrim()

    # one rigid body: no air drag worth modelling at these speeds; angular damping stands in for rolling resistance
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(spec.mass))
    shell = 2.0 / 3.0 * float(spec.mass) * BALL_R ** 2          # a hollow ball: a thin shell's moment of inertia
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(shell, shell, shell))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateLinearDampingAttr(0.0)
    rb.CreateAngularDampingAttr(ROLL_DAMP)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)

    def look(name, color, rough):
        path = f"{prim_path}/Looks/{name}"
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=rough)
        vis.func(path, vis)
        return path

    felt, seam, ink = look("felt", OPTIC, 0.95), look("seam", SEAM, 0.8), look("ink", INK, 0.6)
    mat_path = f"{prim_path}/Looks/rubber"
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=FELT[0], dynamic_friction=FELT[1], restitution=float(spec.bounce),
                                         friction_combine_mode="average", restitution_combine_mode="max")
    mat.func(mat_path, mat)

    ball = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    ball.CreateRadiusAttr(float(BALL_R))
    bp = ball.GetPrim()
    bind_visual_material(str(bp.GetPath()), felt, stage=stage)
    UsdPhysics.CollisionAPI.Apply(bp)
    cp = PhysxSchema.PhysxCollisionAPI.Apply(bp)
    cp.CreateContactOffsetAttr(0.002)
    cp.CreateRestOffsetAttr(0.0)
    bind_physics_material(str(bp.GetPath()), mat_path, stage=stage)

    # what a camera sees: the seam and the pen dots (no colliders)
    pts = seam_points()
    for n, (p0, p1) in enumerate(zip(pts, np.roll(pts, -1, axis=0))):
        d = p1 - p0
        length = float(np.linalg.norm(d))
        ex = d / length
        ey = np.cross(p0 / np.linalg.norm(p0), ex)
        ey /= np.linalg.norm(ey)
        qx, qy, qz, qw = _quat(np.column_stack([ex, ey, np.cross(ex, ey)]))
        cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/seam_{n}")
        cap.CreateAxisAttr("X")
        cap.CreateHeightAttr(length)
        cap.CreateRadiusAttr(float(SEAM_R))
        cap.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in (p0 + p1) / 2)))
        cap.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        bind_visual_material(str(cap.GetPrim().GetPath()), seam, stage=stage)
    for n, p in enumerate(dot_points(int(spec.dots))):
        dot = UsdGeom.Sphere.Define(stage, f"{prim_path}/dot_{n}")
        dot.CreateRadiusAttr(float(DOT_R))
        dot.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in p * (1.0 - 0.6 * DOT_R / BALL_R))))
        bind_visual_material(str(dot.GetPrim().GetPath()), ink, stage=stage)
    return prim


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


_SPAWNERS = {Ball: _spawn_ball}


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
    """Put a loose object at ``pos`` (relative to the env origin), ``quat_xyzw``, at rest. Used by tests of the physics
    only, never by the task or its demo."""
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
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - BALL_R, high=pos + BALL_R)


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


def finger_gaps(environment) -> list:
    """The distance between each gripper's two pads as its finger joints have it (m), left then right."""
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = environment.unwrapped.scene[arm]
        q, names = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        out.append(sum(abs(float(q[names.index(j)])) for j in ("joint7", "joint8")))
    return out


PALM_BACK = 0.065
FINGER_THICK = 0.012


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
    whether an arm is clear of a ball."""
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
    if getattr(scene, "_ballkit_patched", False):
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
    scene._ballkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_ballkit_patched", False):
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
    cls._ballkit_patched = True


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
