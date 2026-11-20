"""Parts for the compass-calibration task: a small folding camera drone, unfolded, as one free rigid body.

What a task file can list in ``OBJECTS`` (a Box subclass so the loader accepts it):

``Drone``   The drone with its four arms out and its propellers on: a fuselage, a camera block under the nose, four
            arms each with a motor, a two-blade propeller and a landing leg. One rigid body of 249 g. Its ``pos`` is
            the centre of the fuselage's outline; its own x runs to the nose, its y to its left, its z up through
            its top.

The landing pad is a plain fixed ``Box``.

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

# -- the drone (m) -------------------------------------------------------------------------------------------------------
# Sourced (the maker's user manual for a sub-250 g folding camera drone): folded 138 x 81 x 58 mm; unfolded with
# propellers 245 x 289 x 56 mm; takeoff weight under 249 g. Everything else here is an estimate from photographs.
LEN = 0.138                    # nose to tail, camera included (the folded length)
WID = 0.058                    # the fuselage's width (an estimate; folded, with the arms alongside, it is 81 mm)
HGT = 0.040                    # the fuselage's height (an estimate)
TALL = 0.056                   # feet to propellers (the unfolded height)
SPAN = 0.289                   # across the propellers with their blades pointing out sideways (the unfolded width)
MASS = 0.249                   # kg
CAM_L, CAM_W, CAM_H = 0.014, 0.028, 0.022          # the camera block at the nose (an estimate)
FUS_L = LEN - CAM_L                                # the fuselage proper
FUS_X = -CAM_L / 2                                 # ... and where its middle is
ARM_X = 0.0625                 # the arms leave the fuselage this far ahead of and behind its middle (an estimate)
ARM_W, ARM_T = 0.013, 0.010    # an arm's width and thickness (an estimate)
MOTOR_X, MOTOR_Y = 0.0675, 0.0895                  # the motors' centres (an estimate)
MOTOR = 0.020                  # a motor's width (an estimate)
MOTOR_H = 0.016
PROP_L, PROP_W, PROP_T = 0.110, 0.012, 0.003       # a propeller, tip to tip (an estimate)
LEG = 0.008                    # a landing leg's width (an estimate)
REAR_LEG_Y = 0.050             # the rear legs stand this far out from the middle line (an estimate)
TOP_Z = HGT / 2                # the fuselage's top, above its middle
FOOT_Z = -(TALL - (TOP_Z + 0.0085))                # the feet, below the fuselage's middle: -0.0275
STAND = -FOOT_Z                # the fuselage's middle stands this far above what the drone stands on

GREY = (0.78, 0.79, 0.80)
DARK = (0.16, 0.17, 0.19)
ARM_GREY = (0.62, 0.63, 0.65)
PAD_ORANGE = (0.90, 0.47, 0.13)
PLASTIC = (0.45, 0.40)         # the drone's plastic on the table and the pad (an estimate)
RUBBER = (0.90, 0.80)          # the fuselage against a gripper's rubber pad (an estimate)
PAD_PATCH = 0.008              # radius of the patch over which a rubber pad meets the fuselage (an estimate)
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


def place(pos, quat, local) -> np.ndarray:
    """A point given in the drone's own frame, in the world."""
    return np.asarray(pos, float) + _rotation(quat) @ np.asarray(local, float)


def boards() -> list:
    """The drone as boxes in its own frame: (tag, centre, size, colour, grip). ``grip`` marks the fuselage, which is
    what a gripper's pads hold."""
    out = [("fuselage", (FUS_X, 0.0, 0.0), (FUS_L, WID, HGT), GREY, True),
           ("camera", (LEN / 2 - CAM_L / 2, 0.0, -0.008), (CAM_L, CAM_W, CAM_H), DARK, False)]
    arm_len = MOTOR_Y - MOTOR / 2 - WID / 2
    for sx, fa in ((1, "front"), (-1, "rear")):
        for sy, lr in ((1, "left"), (-1, "right")):
            tag = f"{fa}_{lr}"
            out.append((f"arm_{tag}", (sx * ARM_X, sy * (WID / 2 + arm_len / 2), TOP_Z - ARM_T / 2), (ARM_W, arm_len, ARM_T), ARM_GREY, False))
            out.append((f"motor_{tag}", (sx * MOTOR_X, sy * MOTOR_Y, TOP_Z - 0.004), (MOTOR, MOTOR, MOTOR_H), DARK, False))
            out.append((f"prop_{tag}", (sx * MOTOR_X, sy * MOTOR_Y, TOP_Z + 0.0065), (PROP_W, PROP_L, PROP_T), DARK, False))
            if sx > 0:        # a front leg stands under its motor
                top = TOP_Z - 0.004 - MOTOR_H / 2
                out.append((f"leg_{tag}", (MOTOR_X, sy * MOTOR_Y, (top + FOOT_Z) / 2), (LEG, LEG, top - FOOT_Z), ARM_GREY, False))
            else:             # a rear leg stands under its arm
                top = TOP_Z - ARM_T
                out.append((f"leg_{tag}", (-ARM_X, sy * REAR_LEG_Y, (top + FOOT_Z) / 2), (LEG, LEG, top - FOOT_Z), ARM_GREY, False))
    return out


def corners() -> np.ndarray:
    """Every corner of every box of the drone, in its own frame."""
    pts = []
    for _, c, s, _, _ in boards():
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append([c[0] + sx * s[0] / 2, c[1] + sy * s[1] / 2, c[2] + sz * s[2] / 2])
    return np.array(pts)


_CORNERS = corners()
FEET = np.array([[MOTOR_X, MOTOR_Y, FOOT_Z], [MOTOR_X, -MOTOR_Y, FOOT_Z], [-ARM_X, REAR_LEG_Y, FOOT_Z], [-ARM_X, -REAR_LEG_Y, FOOT_Z]])


def lowest(pos, quat) -> float:
    """The height (world z) of the drone's lowest point."""
    return float((_CORNERS @ _rotation(quat)[2] + float(pos[2])).min())


def standing_rot(yaw: float = 0.0) -> np.ndarray:
    """The drone on its feet, its nose pointing ``yaw`` round from straight away from the robot (toward its left)."""
    return _rz(yaw)


def attitude(quat) -> dict:
    """How the drone is held. ``up_tilt``: degrees its top is off straight up. ``plumb_tilt``: degrees its
    nose-to-tail line is off plumb (either end up). ``nose``: compass direction of its nose, degrees, seen from above.
    ``top``: compass direction its top faces. Directions are measured from straight away from the robot, toward its
    left."""
    r = _rotation(quat)
    x, z = r[:, 0], r[:, 2]
    return {"up_tilt": math.degrees(math.acos(float(np.clip(z[2], -1.0, 1.0)))),
            "plumb_tilt": math.degrees(math.acos(float(np.clip(abs(x[2]), 0.0, 1.0)))),
            "nose": math.degrees(math.atan2(x[1], x[0])) % 360.0,
            "top": math.degrees(math.atan2(z[1], z[0])) % 360.0}


def drone_distance(points, pos, quat) -> float:
    """The least distance from any of ``points`` (world) to any part of the drone."""
    rot = _rotation(quat)
    local = (np.asarray(points, float).reshape(-1, 3) - np.asarray(pos, float)) @ rot
    best = 9.0
    for _, c, s, _, _ in boards():
        q = np.abs(local - np.asarray(c)) - np.asarray(s) / 2
        d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        best = min(best, float(d.min()))
    return best


# -- the part ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Drone(Box):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


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

    def mat(self, key, sf, df, bounce=0.0) -> str:
        if key not in self.mats:
            path = f"{self.path}/Looks/phys_{key}"
            try:
                pm = self.sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=bounce,
                                                         friction_combine_mode="average", restitution_combine_mode="average")
            except TypeError:
                pm = self.sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=bounce)
            pm.func(path, pm)
            self.mats[key] = path
        return self.mats[key]

    def body(self, path, mass, lin_damp=0.05, ang_damp=0.1, iterations=None, depen=0.3):
        from pxr import PhysxSchema, UsdPhysics

        prim = self.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(float(depen))
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        rb.CreateSleepThresholdAttr(0.0)
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

    def bar(self, path, centre, size, look, mat=None, offset=0.0008, patch=0.0):
        """A box of ``size`` at ``centre``: a box collider (thin plates cannot be cooked as convex hulls for the GPU
        pipeline, and then collide with nothing)."""
        from isaaclab.sim.utils import bind_visual_material

        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in centre)))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*(float(v) for v in size)))
        prim = cube.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            self._collide(prim, mat, offset, patch)
        return prim


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _spawn_drone(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=32)
    plastic, grip = a.mat("plastic", *PLASTIC), a.mat("grip", *RUBBER)
    for tag, centre, size, color, held in boards():
        a.bar(f"{prim_path}/{tag}", centre, size, a.look(color, rough=0.45), grip if held else plastic,
              patch=PAD_PATCH if held else 0.0)
    # marks that show which way the drone faces: a dark stripe across the top behind the nose, a light on the tail
    a.bar(f"{prim_path}/mark_nose", (FUS_X + FUS_L / 2 - 0.012, 0.0, TOP_Z + 0.0003), (0.010, WID * 0.8, 0.0006), a.look(DARK))
    a.bar(f"{prim_path}/mark_tail", (-LEN / 2 - 0.0003, 0.0, 0.004), (0.0006, 0.016, 0.006), a.look((0.20, 0.75, 0.30)))
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


_SPAWNERS = {Drone: _spawn_drone}


# -- run time ----------------------------------------------------------------------------------------------------------

def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    if hasattr(x, "numpy") and not hasattr(x, "detach"):
        return np.asarray(x.numpy(), dtype=np.float64)
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _origin(environment) -> np.ndarray:
    return _np(environment.unwrapped.scene.env_origins[0])


def _kit_state(environment, spec) -> ObjectState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pts = _CORNERS @ _rotation(quat).T + pos
    return ObjectState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0))


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
    (world, relative to the env origin): the pad tips, the tool point and the palm, at the fingers' current opening."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _origin(environment)
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        names = list(robot.body_names)
        pose = _np(robot.data.body_link_pose_w)[0][names.index("gripper")]
        rot = _rotation(pose[3:7])
        tcp = pose[:3] + rot @ np.array(TCP_OFFSET, float) - origin
        q, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        half = sum(abs(float(q[jn.index(j)])) for j in ("joint7", "joint8")) / 2 + FINGER_THICK
        pts = [tcp, tcp - rot[:, 2] * PALM_BACK]
        for s in (-1.0, 1.0):
            for along in (PAD_TIP_PAST_TCP, 0.0, -0.03, -PALM_BACK):
                pts.append(tcp + rot[:, 0] * s * half + rot[:, 2] * along)
        out.append(np.array(pts))
    return out


# -- hooks ---------------------------------------------------------------------------------------------------------------

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
    if getattr(scene, "_dronekit_patched", False):
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
    scene._dronekit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_dronekit_patched", False):
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
    cls._dronekit_patched = True


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
