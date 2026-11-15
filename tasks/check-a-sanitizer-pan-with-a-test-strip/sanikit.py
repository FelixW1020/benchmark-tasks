"""sanikit: a pan of sanitizer, a test strip, a waste cup, a colour chart and fixed blocks, for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from sanikit import Chart, Cup, Fixed, Pan, Strip, on_reset, on_step

- :class:`Fixed` is a fixed box with its own friction, look and orientation (the strip's rest, the tablet board).
- :class:`Pan` is a steel food pan that does not move (``pos`` is the middle of its base on the table, ``size`` its
  inside length, inside width and height). The liquid in it is drawn, not simulated: a clear block up to ``depth``.
  Nothing floats or is slowed by it; the task file works out what is under its surface.
- :class:`Strip` is a loose plastic test strip. ``pos`` is its pad end, its length runs along its own z, its width
  along its own x, and the reagent pad is on its +y face. :func:`recolor` changes the pad's colour.
- :class:`Cup` is a loose, open plastic cup (``pos`` the middle of its base, ``size`` its inside diameter twice and
  its height).
- :class:`Chart` is a printed colour chart lying flat (it touches nothing): a card with one swatch per reading and
  the reading in digits under each swatch.

``on_step`` and ``on_reset`` register callbacks ``fn(env)``.

The structure follows the kits used by tasks already in this repository (Box subclasses so the loader accepts the
parts, custom spawners, hooks installed through ``sys.meta_path``). Pure Python at import time; Isaac Sim is reached
only when the scene is built.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Chart", "Cup", "Fixed", "KitState", "Pan", "Strip", "env", "finger_points", "gripper_points", "install",
           "on_reset", "on_step", "recolor", "segment_distance", "turn_x", "write_pose"]

# -- a plastic sanitizer test strip (m) -----------------------------------------------------------------------------
STRIP_W, STRIP_T, STRIP_L = 0.0064, 0.00035, 0.064    # 1/4 in wide, 2.5 in long
PAD_L = 0.0064                                        # the square reagent pad ...
PAD_FROM_END = 0.0010                                 # ... starts this far from the strip's end
STRIP_MASS = 0.0005
STRIP_FRICTION = (0.6, 0.5)

PALM_BACK = 0.065                                     # a gripper's palm, this far back from its grasp point
FINGER_THICK = 0.012                                  # a finger's outer face, this far out from its pad

# seven-segment digits for the chart's printed readings: which of the segments a..g are inked
SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
            "8": "abcdefg", "9": "abcdfg"}


@dataclass(frozen=True)
class Fixed(Box):
    """A fixed task-file ``Box`` with its own friction, look and orientation (``quat``, xyzw)."""

    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    roughness: float = 0.6
    metallic: float = 0.0
    quat: tuple = (0.0, 0.0, 0.0, 1.0)


class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Pan(_Part):
    """A steel pan that stays where it is. ``size``: inside length (x), inside width (y), height. ``depth``: how deep
    the liquid stands; ``tint`` its colour."""

    wall: float = 0.002
    flange: float = 0.012
    depth: float = 0.05
    tint: tuple = (0.97, 0.99, 1.0)


@dataclass(frozen=True)
class Strip(_Part):
    """A loose test strip. ``pos`` is its pad end; ``quat`` (xyzw) turns its own z (pad end to handle end), x (width)
    and y (the pad's face) into the world. ``pad`` is the pad's colour when dry."""

    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = STRIP_MASS
    pad: tuple = (0.96, 0.90, 0.55)


@dataclass(frozen=True)
class Cup(_Part):
    """A loose open cup. ``size``: inside diameter, inside diameter, height."""

    mass: float | None = 0.02
    wall: float = 0.002
    friction: tuple = (0.6, 0.5)


@dataclass(frozen=True)
class Chart(_Part):
    """A colour chart lying flat, read from the arms' side (readings run left to right as y falls). ``size``: the
    card (x, y, thickness). ``swatches``: one ``(colour, digits)`` per reading."""

    swatches: tuple = ()
    swatch: float = 0.020


@dataclass(frozen=True)
class KitState(ObjectState):
    """A kit object as the simulator has it. For a :class:`Strip`, ``pos`` is its pad end."""

    joints: dict = field(default_factory=dict)


# -- geometry (pure) -------------------------------------------------------------------------------------------------

def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def turn_x(angle: float) -> tuple:
    """The quaternion (xyzw) of a turn by ``angle`` about x."""
    return (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))


def segment_distance(p, a, b) -> float:
    """The distance from point ``p`` to the straight piece from ``a`` to ``b``."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def strip_points(pos, quat) -> dict:
    """Points of a strip at this pose: its ``foot`` (pad end), ``top`` (handle end), ``mid`` and the pad's four
    ``corners``."""
    pos, rot = np.asarray(pos, float), _rotation(quat)
    corners = [pos + rot @ np.array([sx * STRIP_W / 2, 0.0, z]) for sx in (-1.0, 1.0)
               for z in (PAD_FROM_END, PAD_FROM_END + PAD_L)]
    return {"foot": pos, "top": pos + rot @ np.array([0.0, 0.0, STRIP_L]),
            "mid": pos + rot @ np.array([0.0, 0.0, STRIP_L / 2]), "corners": corners}


def digit_boxes(text: str, height: float, width: float, stroke: float, gap: float) -> list:
    """Seven-segment strokes for ``text``: ``(u, v, du, dv)`` boxes, u along the reading direction and v up the
    digits, the whole text centred on (0, 0)."""
    total = len(text) * width + (len(text) - 1) * gap
    out = []
    for k, ch in enumerate(text):
        u0 = -total / 2 + k * (width + gap) + width / 2
        h2 = height / 2
        seg = {"a": (u0, h2 - stroke / 2, width, stroke), "g": (u0, 0.0, width, stroke),
               "d": (u0, -h2 + stroke / 2, width, stroke),
               "f": (u0 - width / 2 + stroke / 2, h2 / 2, stroke, h2), "b": (u0 + width / 2 - stroke / 2, h2 / 2, stroke, h2),
               "e": (u0 - width / 2 + stroke / 2, -h2 / 2, stroke, h2), "c": (u0 + width / 2 - stroke / 2, -h2 / 2, stroke, h2)}
        out += [seg[s] for s in SEGMENTS[ch]]
    return out


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

    def look(self, color, rough=0.5, metal=0.0, name: str | None = None) -> str:
        from yam_arena.station import linear_rgb

        key = name or (tuple(color), rough, metal)
        if key not in self.looks:
            path = f"{self.path}/Looks/{name or f'v{len(self.looks)}'}"
            vis = self.sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=rough, metallic=metal)
            vis.func(path, vis)
            self.looks[key] = path
        return self.looks[key]

    def glass(self, color, ior=1.33) -> str:
        """A see-through look (the real-time renderer ignores a plain surface's opacity)."""
        key = ("glass", tuple(color), ior)
        if key not in self.looks:
            path = f"{self.path}/Looks/g{len(self.looks)}"
            try:
                vis = self.sim_utils.GlassMdlCfg(glass_color=tuple(float(c) for c in color), frosting_roughness=0.0,
                                                 thin_walled=False, glass_ior=ior)
                vis.func(path, vis)
            except Exception as err:                                          # noqa: BLE001
                print(f"[sanikit] glass look unavailable ({err!r}); drawn as a plain surface", file=sys.stderr, flush=True)
                from yam_arena.station import linear_rgb

                vis = self.sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.05, opacity=0.3)
                vis.func(path, vis)
            self.looks[key] = path
        return self.looks[key]

    def mat(self, key, sf, df) -> str:
        if key not in self.mats:
            path = f"{self.path}/Looks/phys_{key}"
            pm = _material(self.sim_utils, sf, df)
            pm.func(path, pm)
            self.mats[key] = path
        return self.mats[key]

    def body(self, path, pos, mass, lin_damp=0.0, ang_damp=0.0, inertia=None, kinematic=False, iterations=None):
        from pxr import PhysxSchema, UsdPhysics

        xf = self.UsdGeom.Xform.Define(self.stage, path)
        if path != self.path and pos is not None:
            xf.AddTranslateOp(precision=self.UsdGeom.XformOp.PrecisionDouble).Set(self.Gf.Vec3d(*pos))
            xf.AddOrientOp(precision=self.UsdGeom.XformOp.PrecisionDouble).Set(self.Gf.Quatd(1, 0, 0, 0))
        prim = xf.GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        if kinematic:
            rigid.CreateKinematicEnabledAttr(True)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        if inertia is not None:
            m.CreateDiagonalInertiaAttr(self.Gf.Vec3f(*(float(v) for v in inertia)))
            m.CreateCenterOfMassAttr(self.Gf.Vec3f(0.0, 0.0, 0.0))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.3)
        rb.CreateLinearDampingAttr(lin_damp)
        rb.CreateAngularDampingAttr(ang_damp)
        if iterations:
            rb.CreateSolverPositionIterationCountAttr(int(iterations))
            rb.CreateSolverVelocityIterationCountAttr(4)
        return prim

    def collide(self, prim, mat, offset=0.0008):
        from isaaclab.sim.utils import bind_physics_material
        from pxr import PhysxSchema, UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(offset)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)

    def paint(self, prim, look):
        from isaaclab.sim.utils import bind_visual_material

        bind_visual_material(str(prim.GetPath()), look, stage=self.stage)

    def _finish(self, prim, look, mat, offset=0.0008):
        if look:
            self.paint(prim, look)
        else:
            self.UsdGeom.Imageable(prim).MakeInvisible()
        if mat:
            self.collide(prim, mat, offset)
        return prim

    def box(self, path, centre, size, look, mat=None, yaw=0.0, offset=0.0008):
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*(float(v) for v in centre)))
        if yaw:
            cube.AddOrientOp().Set(self.Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*(float(v) for v in size)))
        return self._finish(cube.GetPrim(), look, mat, offset)

    def cyl(self, path, r, z0, z1, look, mat=None):
        c = self.UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(z1 - z0))
        c.AddTranslateOp().Set(self.Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
        return self._finish(c.GetPrim(), look, mat)


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned
STEEL = (0.76, 0.77, 0.79)


def _spawn_pan(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Floor, four walls and a rolled rim, all one body that is moved by nothing; the liquid is a clear block."""
    spec: Pan = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, None, 1.0, kinematic=True)
    steel = a.look(STEEL, rough=0.45, metal=0.5)                         # brushed steel: bright under the room's lights
    m = a.mat("steel", 0.4, 0.3)
    ix, iy, h = (float(v) for v in spec.size)
    w, f = float(spec.wall), float(spec.flange)
    a.box(f"{prim_path}/floor", (0.0, 0.0, w / 2), (ix + 2 * w, iy + 2 * w, w), steel, m)
    for name, sx, sy in (("wall_far", 1, 0), ("wall_near", -1, 0), ("wall_left", 0, 1), ("wall_right", 0, -1)):
        if sx:
            a.box(f"{prim_path}/{name}", (sx * (ix + w) / 2, 0.0, h / 2), (w, iy + 2 * w, h), steel, m)
            a.box(f"{prim_path}/rim_{name}", (sx * (ix / 2 + w + f / 2), 0.0, h - w / 2), (f, iy + 2 * w + 2 * f, w), steel, m)
        else:
            a.box(f"{prim_path}/{name}", (0.0, sy * (iy + w) / 2, h / 2), (ix + 2 * w, w, h), steel, m)
            a.box(f"{prim_path}/rim_{name}", (0.0, sy * (iy / 2 + w + f / 2), h - w / 2), (ix + 2 * w, f, w), steel, m)
    if spec.depth > 0:
        a.box(f"{prim_path}/liquid", (0.0, 0.0, w + spec.depth / 2), (ix - 0.0004, iy - 0.0004, spec.depth),
              a.glass(spec.tint))
    return a.root


def _spawn_strip(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """A thin plate with its pad end at the origin, its length along z, its width along x, the pad on its +y face."""
    spec: Strip = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, None, float(spec.mass), lin_damp=0.05, ang_damp=0.1, iterations=32)
    m = a.mat("strip", *STRIP_FRICTION)
    white = a.look((0.95, 0.95, 0.93), rough=0.45)
    a.box(f"{prim_path}/plate", (0.0, 0.0, STRIP_L / 2), (STRIP_W, STRIP_T, STRIP_L), white, m, offset=0.0015)
    a.box(f"{prim_path}/pad", (0.0, STRIP_T / 2 + 0.0002, PAD_FROM_END + PAD_L / 2), (STRIP_W, 0.0004, PAD_L),
          a.look(spec.pad, rough=0.8, name="pad"))
    a.box(f"{prim_path}/grip_mark", (0.0, STRIP_T / 2 + 0.0001, STRIP_L - 0.006), (STRIP_W, 0.0002, 0.010),
          a.look((0.20, 0.42, 0.72), rough=0.5))
    return a.root


def _spawn_cup(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """A round open cup: a floor disc and a ring of flat wall pieces, one loose body with its base at the origin."""
    spec: Cup = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, None, float(spec.mass), lin_damp=0.05, ang_damp=0.1)
    m = a.mat("cup", *spec.friction)
    look = a.look(spec.color, rough=0.5)
    r_in, h, w = float(spec.size[0]) / 2, float(spec.size[2]), float(spec.wall)
    a.cyl(f"{prim_path}/floor", r_in + w, 0.0, w, look, m)
    n = 20
    piece = 2 * (r_in + w) * math.tan(math.pi / n) + 0.0004
    for k in range(n):
        t = 2 * math.pi * k / n
        a.box(f"{prim_path}/wall_{k}", ((r_in + w / 2) * math.cos(t), (r_in + w / 2) * math.sin(t), h / 2), (w, piece, h),
              look, m, yaw=t)
    return a.root


def _spawn_chart(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """A card with printed swatches and digits (nothing here collides). On the card, readings run along -y and the
    digits' tops point along +x, so it reads the right way up from the arms' side."""
    spec: Chart = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, None, 0.01, inertia=(1.0e-6, 1.0e-6, 1.0e-6), kinematic=True)
    cx, cy, ct = (float(v) for v in spec.size)
    a.box(f"{prim_path}/card", (0.0, 0.0, ct / 2), (cx, cy, ct), a.look(spec.color, rough=0.7))
    ink = a.look((0.10, 0.10, 0.11), rough=0.6)
    n, s = len(spec.swatches), float(spec.swatch)
    pitch = (cy - 0.008) / max(n, 1)
    for k, (color, digits) in enumerate(spec.swatches):
        u = (k - (n - 1) / 2) * pitch                                         # along the reading direction (-y)
        a.box(f"{prim_path}/swatch_{k}", (0.009, -u, ct + 0.0001), (s, s, 0.0002), a.look(color, rough=0.8))
        for j, (du, dv, wu, wv) in enumerate(digit_boxes(str(digits), 0.008, 0.0045, 0.0011, 0.0016)):
            a.box(f"{prim_path}/ink_{k}_{j}", (-0.012 + dv, -(u + du), ct + 0.0001), (wv, wu, 0.0002), ink)
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
    quat = tuple(float(v) for v in getattr(spec, "quat", (0.0, 0.0, 0.0, 1.0)))
    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=KitSpawnerCfg(kit_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


def _fixed_object(scene, spec: Fixed):
    sim_utils = scene.sim_utils
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=spec.roughness,
                                       metallic=spec.metallic)
    cfg = sim_utils.CuboidCfg(size=tuple(spec.size),
                              physics_material=_material(sim_utils, spec.static_friction, spec.dynamic_friction),
                              visual_material=look, collision_props=scene.PhysxCollisionPropertiesCfg())
    return scene.Object(name=spec.name, object_type=scene.ObjectType.BASE, spawner_cfg=cfg,
                        initial_pose=scene.Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                                rotation_xyzw=tuple(float(v) for v in spec.quat)))


_SPAWNERS = {Pan: _spawn_pan, Strip: _spawn_strip, Cup: _spawn_cup, Chart: _spawn_chart}


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


_SHADERS: dict = {}
_WARNED: set = set()


def recolor(name: str, look: str, srgb) -> None:
    """Change the colour of kit object ``name``'s look ``look`` while the simulation runs (a strip's ``"pad"``).
    Cosmetic: outside the simulator it does nothing, and a failure is reported once and never stops a trial."""
    if _ENV["env"] is None or name not in _PATHS:
        return
    key = (name, look)
    try:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, Usd, UsdShade
        from yam_arena.station import linear_rgb

        if key not in _SHADERS:
            root = get_current_stage().GetPrimAtPath(f"{_PATHS[name]}/Looks/{look}")
            _SHADERS[key] = [UsdShade.Shader(p) for p in Usd.PrimRange(root) if p.IsA(UsdShade.Shader)] if root else []
        colour = Gf.Vec3f(*(float(c) for c in linear_rgb(srgb)))
        done = False
        for shader in _SHADERS[key]:
            for port_name in ("diffuseColor", "diffuse_color_constant"):
                port = shader.GetInput(port_name)
                if port:
                    port.Set(colour)
                    done = True
        if not done and key not in _WARNED:
            _WARNED.add(key)
            print(f"[sanikit] no material found to recolour {name}/{look}", file=sys.stderr, flush=True)
    except Exception as err:                                                  # noqa: BLE001
        if key not in _WARNED:
            _WARNED.add(key)
            print(f"[sanikit] could not recolour {name}/{look}: {err!r}", file=sys.stderr, flush=True)


def _box_bounds(pos, rot, lo, hi):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = corners @ rot.T + pos
    return pts.min(0), pts.max(0)


def _kit_state(environment, spec) -> KitState:
    asset = environment.unwrapped.scene[spec.name]
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - _origin(environment), pose[3:7]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    rot = _rotation(quat)
    if isinstance(spec, Strip):
        low, high = _box_bounds(pos, rot, (-STRIP_W / 2, -STRIP_T / 2, 0.0), (STRIP_W / 2, STRIP_T / 2, STRIP_L))
    else:
        sx, sy, sz = (float(v) for v in spec.size)
        low, high = _box_bounds(pos, rot, (-sx / 2, -sy / 2, 0.0), (sx / 2, sy / 2, sz))
    return KitState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=low, high=high)


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
        # joint7 carries the finger on the gripper's -x side, joint8 the one on its +x side; each pad is |q| from the middle
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
    if getattr(scene, "_sanikit_patched", False):
        return
    stock_object, stock_states = scene.task_object, scene.object_states

    def task_object(spec):
        for kind, spawn in _SPAWNERS.items():
            if isinstance(spec, kind):
                return _rigid_object(spec, spawn)
        if isinstance(spec, Fixed):
            return _fixed_object(scene, spec)
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
    scene._sanikit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_sanikit_patched", False):
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
        if _RESET:                                   # show the scene as the reset callbacks left it
            try:
                base = self._env.unwrapped
                base.sim.step(render=False)          # one physics step (1/240 s) carries the written state to the bodies
                base.scene.update(base.physics_dt)
                for _ in range(3):
                    base.sim.render()
                obs = self._observation()
            except Exception as err:                 # noqa: BLE001
                print(f"[sanikit] first frame not refreshed: {err!r}", file=sys.stderr, flush=True)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._sanikit_patched = True


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
