"""The task's parts: a wall section (1/2 in drywall on 2x4 studs, with a 1-gang cut-out), a 1-gang old-work box with
two swing clamps, a #2 Phillips stubby screwdriver and the wooden block it stands in. Geometry is our own, drawn
from the box's published size (4-1/8 x 2-1/4 x 2-3/4 in), its cut-out template (2-1/4 x 3-5/8 in) and standard
lumber and drywall sizes; no manufacturer CAD is used.

yam_arena task files support single primitives and library assets only. This module is the task's own minimal
mechanism, kept in the task folder (the same approach as the end-bracket and lever-set tasks):

- :class:`Part` is a :class:`yam_arena.task_file.Box` (so the stock loader accepts it) that names which part it is.
- :func:`install` wraps ``yam_arena.scene.task_object`` and ``object_states`` when that module is imported, and the
  embodiment's ``reset`` and step (to run :func:`on_reset` and :func:`on_step` functions). Every Part is spawned from
  a USD file written here; nothing in yam_arena is modified.

The swing clamp (one at each end of the box; all of it rigid bodies, joints and contact):

- The clamp screw turns in the box (revolute joint ``screw_*``, against the friction of its head on the box).
- The clamp's threaded hub rides on the screw: a revolute joint ``nut_*`` (the hub turning on the screw, against the
  thread's friction) and a prismatic joint ``slide_*`` along the screw, coupled by a PhysX mimic joint: the thread.
  Turning the hub one turn on the screw moves it LEAD along the screw.
- The clamp arm (``wing_*``) sits on the hub on a stiff spring along the screw (prismatic, the plastic arm's bending
  compliance), so the force it presses the drywall with is the spring's force.
- Nothing is scripted: turned clockwise, the thread's friction carries the arm round with the screw until it swings
  out through the slot in the box's end wall and hits the slot's edge (a contact stop); from then on the screw turns
  inside the stopped hub and the thread draws the arm toward the box's ears until it presses on the back of the
  drywall (contact), and the screw stalls. Turned back, the arm backs off and swings in again.

Frames. Box: origin on the plane of the ears' back faces (the wall's face when the box is home) at the middle of the
cut-out; x into the wall (the box's depth), y across (its 2-1/4 in width), z up its length. Screws on the box's
centre line at z = +-SCREW_Z, axes along x, heads on the front. Driver: origin at the tip, shaft along +z.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from yam_arena import task_file

MM = 1e-3
IN = 25.4 * MM

# -- the wall section ---------------------------------------------------------------------------------------------
DRYWALL_T = 0.5 * IN               # 1/2 in gypsum wallboard
HOLE_W, HOLE_H = 2.25 * IN, 3.625 * IN   # the box's cut-out template: 2-1/4 x 3-5/8 in (57.2 x 92.1 mm)
STUD_W, STUD_D = 1.5 * IN, 3.5 * IN      # a 2x4 is 1-1/2 x 3-1/2 in
STUD_SPACING = 16 * IN                    # 16 in on centre
WALL_W, WALL_H = 0.48, 0.40               # the wall section: drywall and plates 480 mm wide, 400 mm tall

# -- the box (1-gang old-work box, e.g. a Carlon B114R: 4-1/8 L x 2-1/4 W x 2-3/4 D in, 14 cu in, PVC, 56.7 g) --------
EAR_T = 2.5 * MM                   # the front rim and ears: x -EAR_T .. 0
RIM_HW, RIM_HH = 28.5 * MM, 52.4 * MM     # front rim and ears: 57 x 104.8 mm (4-1/8 in long overall)
OPEN_HW, OPEN_HH = 24.5 * MM, 33.0 * MM   # the opening in the front rim (the ears' plates carry the screws)
DEPTH = 2.75 * IN - EAR_T          # the body behind the ears' back faces (to the back's outer face): 67.35 mm
BODY_FRONT = (27.0 * MM, 44.5 * MM)       # outer half width and half length of the body at the ears ...
BODY_BACK = (25.0 * MM, 42.5 * MM)        # ... and at the back (2 mm of moulding draft each side)
WALL = 2.5 * MM                    # PVC wall thickness
SCREW_Z = 37.0 * MM                # the clamp screws: on the centre line, this far above and below the middle
HEAD_R, HEAD_H = 3.7 * MM, 2.3 * MM       # pan head (#8: 7.4 mm, 2.3 mm tall) on the ear plate's front face
RECESS_D = 1.8 * MM                # #2 Phillips recess depth
SLOT_HW = 0.8 * MM                 # recess slot half width (the tip's blades are 0.6 to 1.0 mm thick)
SLOT_END = 2.95 * MM               # each slot of the cross ends this far from the axis (the tip's blades reach 2.8)
SLOT_X = (4.0 * MM, 31.0 * MM)     # the slot in each end wall the clamp arm swings out through (along the depth) ...
SLOT_Y = (-4.5 * MM, 22.5 * MM)    # ... and across (top end; the bottom one mirrored). Its -y edge is the arm's stop
ARM_LEN, ARM_HW, ARM_T = 19.0 * MM, 3.5 * MM, 3.0 * MM   # clamp arm: 19 mm from the screw's axis, 7 wide, 3 thick
ARM_X0 = 24.0 * MM                 # the arm's front face, as delivered (backed off): 24 mm behind the ears
ARM_X_MIN = 6.0 * MM               # the hub's travel ends here (the arm's front face; only reached with no wall)
LEAD = 4.0 * MM                    # SIMULATION thread lead per turn (see README: a real #6-32/#8-32 screw is ~0.8 mm)
T_HEAD = 0.02                      # N m: friction of the screw head on the ear plate (estimate)
T_THREAD = 0.01                    # N m: thread friction carrying the arm round with the screw (estimate)
SCREW_DAMPING = 0.002              # N m s/rad: the thread's grease (keeps a screw spun by a light torque from racing)
K_ARM = 40.0e3                     # N/m: the arm's bending stiffness along the screw (PVC cantilever 19 x 7 x 3 mm)
ARM_FLEX = 3.0 * MM                # the arm's spring travel (120 N at full travel)
MASS = {"body": 0.0511, "screw": 0.0012, "nut": 0.0002, "slide": 0.0002, "wing": 0.0012}
SCREW_ARMATURE = 2.0e-6            # kg m^2: the screw and hub (light; a heavier armature makes the tip bounce off)
LINK_ARMATURE = 1.0e-3

# -- the screwdriver: a #2 Phillips stubby (38 mm blade, 6 mm shaft, 30 mm octagonal handle 50 mm long, 92 mm overall)
TIP_LEN = 4.0 * MM
SHAFT_R, SHAFT_TOP = 3.0 * MM, 42.0 * MM
HANDLE_AF, HANDLE_TOP = 30.0 * MM, 92.0 * MM   # handle 30 mm across flats, from SHAFT_TOP to HANDLE_TOP
DRIVER_MASS = 0.050

# -- the driver's block: a 50 x 50 x 44 mm hardwood block with a 10 mm square hole through it --------------------------
BLOCK = (50.0 * MM, 50.0 * MM, 44.0 * MM)
BLOCK_HOLE = 10.0 * MM

MATERIALS = {  # static, dynamic friction, combine mode
    "gypsum": (0.5, 0.45, "average"),
    "wood": (0.5, 0.45, "average"),
    "pvc": (0.4, 0.35, "average"),
    "steel": (0.25, 0.2, "average"),        # the driver's tip in the screw's recess
    "handle": (0.9, 0.8, "average"),        # the driver's two-component handle
}
LOOKS = {  # sRGB, roughness, metallic
    "drywall": ((0.90, 0.89, 0.86), 0.9, 0.0),
    "paper": ((0.80, 0.78, 0.72), 0.9, 0.0),
    "pine": ((0.84, 0.70, 0.50), 0.7, 0.0),
    "blue": ((0.12, 0.33, 0.78), 0.45, 0.0),
    "clamp": ((0.92, 0.92, 0.90), 0.5, 0.0),
    "zinc": ((0.78, 0.78, 0.76), 0.3, 0.9),
    "steel": ((0.62, 0.64, 0.66), 0.3, 0.9),
    "red": ((0.80, 0.10, 0.08), 0.5, 0.0),
    "yellow": ((0.95, 0.78, 0.10), 0.5, 0.0),
    "beech": ((0.80, 0.66, 0.48), 0.7, 0.0),
    "dark": ((0.05, 0.05, 0.05), 0.9, 0.0),
}
CONTACT_OFFSET = 0.5 * MM


# -- shapes -------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """One piece of a part in its body frame (all bodies of a part share the part's frame at joint zero).

    kind "box": ``center``, ``size``. kind "cyl": ``center``, ``radius``, ``length`` along ``axis``. kind "loft": a
    convex 2-D profile ``prof0`` at ``s0`` and ``prof1`` (same number of points) at ``s1`` along ``axis``; the profile
    coordinates are the other two axes in order (x: (y, z); y: (z, x); z: (x, y)). ``material`` None: looks only;
    ``look`` None: a hidden collider."""

    body: str
    kind: str
    look: str | None
    material: str | None
    center: tuple = (0.0, 0.0, 0.0)
    size: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0
    axis: str = "Z"
    prof0: tuple = ()
    prof1: tuple = ()
    s0: float = 0.0
    s1: float = 0.0

    def lift(self, p, s) -> tuple:
        a = "XYZ".index(self.axis)
        out = [0.0, 0.0, 0.0]
        out[a] = s
        out[(a + 1) % 3], out[(a + 2) % 3] = p[0], p[1]
        return tuple(out)

    def points(self) -> np.ndarray:
        if self.kind == "loft":
            return np.array([self.lift(p, self.s0) for p in self.prof0] + [self.lift(p, self.s1) for p in self.prof1])
        if self.kind == "box":
            c, h = np.array(self.center), np.array(self.size) / 2
            return np.array([c - h, c + h])
        h = np.full(3, self.radius)
        h["XYZ".index(self.axis)] = self.length / 2
        return np.array([np.array(self.center) - h, np.array(self.center) + h])


def box(body, look, material, lo, hi) -> Shape:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Shape(body, "box", look, material, center=tuple((lo + hi) / 2), size=tuple(hi - lo))


def cyl(body, look, material, center, radius, length, axis) -> Shape:
    return Shape(body, "cyl", look, material, center=tuple(center), radius=radius, length=length, axis=axis)


def loft(body, look, material, axis, s0, prof0, s1, prof1=None) -> Shape:
    return Shape(body, "loft", look, material, axis=axis, s0=s0, s1=s1, prof0=tuple(map(tuple, prof0)),
                 prof1=tuple(map(tuple, prof1 if prof1 is not None else prof0)))


def rect(a0, a1, b0, b1):
    return ((a0, b0), (a1, b0), (a1, b1), (a0, b1))


def octagon(r_flat):
    r = r_flat / math.cos(math.pi / 8)
    return tuple((r * math.cos(math.pi / 8 + k * math.pi / 4), r * math.sin(math.pi / 8 + k * math.pi / 4))
                 for k in range(8))


# -- the wall section (one fixed part; world-aligned frame: origin at the cut-out's centre on the drywall's face) --------

def wall_shapes_at(hole_z: float) -> list[Shape]:
    """The wall section with the cut-out's centre ``hole_z`` above the section's bottom (its frame's origin is the
    cut-out's centre on the drywall's face; the section stands on z = -hole_z)."""
    t, hw, hh, b = DRYWALL_T, HOLE_W / 2, HOLE_H / 2, -hole_z
    W, H = WALL_W / 2, WALL_H
    s = [  # 1/2 in drywall, x 0 .. t, around the cut-out
        box("body", "drywall", "gypsum", (0, -W, b), (t, W, -hh)),
        box("body", "drywall", "gypsum", (0, -W, hh), (t, W, b + H)),
        box("body", "drywall", "gypsum", (0, -W, -hh), (t, -hw, hh)),
        box("body", "drywall", "gypsum", (0, hw, -hh), (t, W, hh)),
    ]
    sd = STUD_D
    s += [box("body", "pine", "wood", (t, -W, b), (t + sd, W, b + STUD_W)),          # bottom plate (2x4 flat)
          box("body", "pine", "wood", (t, -W, b + H - STUD_W), (t + sd, W, b + H))]  # top plate
    for y in (-STUD_SPACING / 2, STUD_SPACING / 2):                                  # two studs, 16 in on centre
        s.append(box("body", "pine", "wood", (t, y - STUD_W / 2, b + STUD_W), (t + sd, y + STUD_W / 2, b + H - STUD_W)))
    # the cut-out's cut faces (paper-brown gypsum core), looks only
    e = 0.2 * MM
    s += [box("body", "paper", None, (0.0005, -hw, -hh - e), (t - 0.0005, hw, -hh)),
          box("body", "paper", None, (0.0005, -hw, hh), (t - 0.0005, hw, hh + e)),
          box("body", "paper", None, (0.0005, -hw - e, -hh), (t - 0.0005, -hw, hh)),
          box("body", "paper", None, (0.0005, hw, -hh), (t - 0.0005, hw + e, hh))]
    return s


# -- the box -------------------------------------------------------------------------------------------------------------

def _body_half(x: float) -> tuple[float, float]:
    """The body's outer half width and half length at depth x (linear draft from the ears to the back)."""
    f = min(max(x / DEPTH, 0.0), 1.0)
    return (BODY_FRONT[0] + f * (BODY_BACK[0] - BODY_FRONT[0]), BODY_FRONT[1] + f * (BODY_BACK[1] - BODY_FRONT[1]))


def _end_wall(sign: int, x0: float, x1: float, y0: float | None = None, y1: float | None = None) -> Shape:
    """A piece of the top (sign +1) or bottom (-1) end wall from depth x0 to x1, across y0..y1 (default: full)."""
    def prof(x):
        hw, hh = _body_half(x)
        a, b = (-hw if y0 is None else y0), (hw if y1 is None else y1)
        if sign < 0:
            a, b = -b, -a
        return rect(a, b, sign * (hh - WALL), sign * hh) if sign > 0 else rect(a, b, sign * hh, sign * (hh - WALL))
    return loft("body", "blue", "pvc", "X", x0, prof(x0), x1, prof(x1))


def box_shapes() -> list[Shape]:
    s = []
    # front rim and ears (x -EAR_T .. 0): a frame round the opening
    s += [box("body", "blue", "pvc", (-EAR_T, -RIM_HW, OPEN_HH), (0, RIM_HW, RIM_HH)),
          box("body", "blue", "pvc", (-EAR_T, -RIM_HW, -RIM_HH), (0, RIM_HW, -OPEN_HH)),
          box("body", "blue", "pvc", (-EAR_T, -RIM_HW, -OPEN_HH), (0, -OPEN_HW, OPEN_HH)),
          box("body", "blue", "pvc", (-EAR_T, OPEN_HW, -OPEN_HH), (0, RIM_HW, OPEN_HH))]
    xb = DEPTH - WALL
    for sy in (-1, 1):  # side walls
        def side(x):
            hw, hh = _body_half(x)
            a, b = sorted((sy * hw, sy * (hw - WALL)))
            return rect(a, b, -hh, hh)
        s.append(loft("body", "blue", "pvc", "X", 0.0, side(0.0), xb, side(xb)))
    for sign in (1, -1):  # end walls, each with the slot its clamp arm swings out through
        s.append(_end_wall(sign, 0.0, SLOT_X[0]))
        s.append(_end_wall(sign, SLOT_X[1], xb))
        hw0 = _body_half(SLOT_X[0])[0]
        s.append(_end_wall(sign, SLOT_X[0], SLOT_X[1], -hw0, SLOT_Y[0]))   # the arm's stop is this piece's edge
        s.append(_end_wall(sign, SLOT_X[0], SLOT_X[1], SLOT_Y[1], hw0))
    hw, hh = BODY_BACK
    s.append(box("body", "blue", "pvc", (xb, -hw, -hh), (DEPTH, hw, hh)))  # back
    for sign, tag in ((1, "t"), (-1, "b")):
        s += _clamp_shapes(sign, tag)
    return s


def _rot_yz(p, sign):
    """A (y, z) point of the top clamp turned to the bottom one (180 degrees about the box's x axis) if sign < 0."""
    return (p[0], p[1]) if sign > 0 else (-p[0], -p[1])


def _clamp_shapes(sign: int, tag: str) -> list[Shape]:
    zc = sign * SCREW_Z
    s = []
    # the screw: a pan head on the ear plate's front face; its #2 Phillips recess: four lobes round a cross, on a floor
    x_face, x_floor = -EAR_T - HEAD_H, -EAR_T - HEAD_H + RECESS_D
    # Boxes, not meshes: PhysX did not collide the tip with lofted lobes this small (a bench showed no contact).
    hw, r_end = SLOT_HW, SLOT_END
    for qy, qz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):   # four lobes between the slots of the cross
        a, b = sorted((qy * hw, qy * r_end)); c, d = sorted((zc + qz * hw, zc + qz * r_end))
        s.append(box(f"screw_{tag}", "zinc", "steel", (x_face, a, c), (-EAR_T, b, d)))
    for sy, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):      # each slot closed at its end
        if sy:
            a, b = sorted((sy * r_end, sy * (r_end + 0.7 * MM)))
            s.append(box(f"screw_{tag}", "zinc", "steel", (x_face, a, zc - hw), (-EAR_T, b, zc + hw)))
        else:
            c, d = sorted((zc + sz * r_end, zc + sz * (r_end + 0.7 * MM)))
            s.append(box(f"screw_{tag}", "zinc", "steel", (x_face, -hw, c), (-EAR_T, hw, d)))
    s.append(cyl(f"screw_{tag}", "zinc", "steel", (( x_floor - EAR_T) / 2, 0.0, zc), HEAD_R * 0.98, -EAR_T - x_floor,
                 "X"))
    s.append(cyl(f"screw_{tag}", "zinc", None, ((ARM_X0 + 16 * MM) / 2, 0.0, zc), 1.75 * MM, ARM_X0 + 16 * MM, "X"))
    # the clamp arm, as delivered: folded inside the box along +y (top; the bottom one along -y), front face at ARM_X0
    arm = [_rot_yz(p, sign) for p in rect(-ARM_HW, ARM_LEN, SCREW_Z - ARM_HW, SCREW_Z + ARM_HW)]
    s.append(loft(f"wing_{tag}", "clamp", "pvc", "X", ARM_X0, arm, ARM_X0 + ARM_T, arm))
    return s


def clamp_joints(sign: int, tag: str) -> list[tuple]:
    """(name, type, parent, child, anchor) for one clamp; every axis along the box's x."""
    a = (0.0, 0.0, sign * SCREW_Z)
    return [(f"screw_{tag}", "rev", "body", f"screw_{tag}", a),
            (f"nut_{tag}", "rev", f"screw_{tag}", f"nut_{tag}", a),
            (f"slide_{tag}", "pri", f"nut_{tag}", f"slide_{tag}", a),
            (f"wing_{tag}", "pri", f"slide_{tag}", f"wing_{tag}", a)]


BOX_MASS = sum(MASS[b.split("_")[0]] for b in ["body"] + [f"{k}_{t}" for t in ("t", "b")
                                                             for k in ("screw", "nut", "slide", "wing")])
BOX_BODIES = ["body"] + [f"{k}_{t}" for t in ("t", "b") for k in ("screw", "nut", "slide", "wing")]
BOX_JOINTS = clamp_joints(1, "t") + clamp_joints(-1, "b")


# -- the screwdriver and its block -------------------------------------------------------------------------------------

def driver_shapes() -> list[Shape]:
    s = []
    for blade in (0, 1):  # the #2 Phillips tip: two tapered blades crossing, along x and along y
        def prof(half_len, half_t):
            p = rect(-half_len, half_len, -half_t, half_t)
            return p if blade == 0 else [(-b, a) for a, b in p]
        s.append(loft("driver", "steel", "steel", "Z", 0.0, prof(1.0 * MM, 0.3 * MM), 1.8 * MM, prof(2.8 * MM, 0.5 * MM)))
        s.append(loft("driver", "steel", "steel", "Z", 1.8 * MM, prof(2.8 * MM, 0.5 * MM), TIP_LEN,
                      prof(2.8 * MM, 0.5 * MM)))
    s.append(cyl("driver", "steel", "steel", (0, 0, (TIP_LEN - 0.5 * MM + SHAFT_TOP) / 2), SHAFT_R,
                 SHAFT_TOP - TIP_LEN + 0.5 * MM, "Z"))
    oc = octagon(HANDLE_AF / 2)
    s.append(loft("driver", "red", "handle", "Z", SHAFT_TOP, oc, SHAFT_TOP + 8 * MM, oc))
    s.append(loft("driver", "yellow", "handle", "Z", SHAFT_TOP + 8 * MM, oc, HANDLE_TOP - 8 * MM, oc))
    s.append(loft("driver", "red", "handle", "Z", HANDLE_TOP - 8 * MM, oc, HANDLE_TOP, octagon(HANDLE_AF / 2 - 2 * MM)))
    return s


def block_shapes() -> list[Shape]:
    """The screwdriver's block: origin on the table under the hole's centre."""
    hx, hy, h = BLOCK[0] / 2, BLOCK[1] / 2, BLOCK[2]
    r = BLOCK_HOLE / 2
    return [box("body", "beech", "wood", (-hx, -hy, 0), (-r, hy, h)), box("body", "beech", "wood", (r, -hy, 0), (hx, hy, h)),
            box("body", "beech", "wood", (-r, -hy, 0), (r, -r, h)), box("body", "beech", "wood", (-r, r, 0), (r, hy, h)),
            box("body", "dark", None, (-r, -r, h - 0.2 * MM), (r, r, h - 0.1 * MM))]


# -- the registry --------------------------------------------------------------------------------------------------------

_WALL_HOLE_Z = [0.2]


def shapes_of(kind: str) -> list[Shape]:
    return {"wall": lambda: wall_shapes_at(_WALL_HOLE_Z[0]), "box": box_shapes, "driver": driver_shapes,
            "block": block_shapes}[kind]()


def bodies_of(kind: str) -> list[str]:
    return BOX_BODIES if kind == "box" else (["driver"] if kind == "driver" else ["body"])


def bounds(kind: str) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([sh.points() for sh in shapes_of(kind)])
    return pts.min(0), pts.max(0)


@dataclass(frozen=True)
class Part(task_file.Box):
    """One of this module's parts as a task-file object: ``pos`` is the part's origin, ``quat`` its orientation
    (XYZW); ``mass`` None: fixed."""

    kind: str = "box"
    quat: tuple = (0.0, 0.0, 0.0, 1.0)

    def _check(self) -> None:
        super()._check()
        if self.kind not in ("wall", "box", "driver", "block"):
            raise ValueError(f"{self.name}: unknown part kind {self.kind!r}")


def part(name: str, kind: str, pos, quat=(0.0, 0.0, 0.0, 1.0), fixed: bool = False, hole_z: float | None = None):
    if hole_z is not None:
        _WALL_HOLE_Z[0] = float(hole_z)
    lo, hi = bounds(kind)
    mass = None if fixed else (BOX_MASS if kind == "box" else DRIVER_MASS)
    color = {"wall": LOOKS["drywall"][0], "box": LOOKS["blue"][0], "driver": LOOKS["red"][0],
             "block": LOOKS["beech"][0]}[kind]
    return Part(name=name, pos=tuple(float(v) for v in pos), size=tuple(float(v) for v in hi - lo), mass=mass,
                color=color, kind=kind, quat=tuple(float(v) for v in quat))


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mechanism(q: dict) -> dict:
    """The clamps' state from the box's joint positions (rad, m): for each clamp ("t" top, "b" bottom) the screw's
    turns (clockwise seen from the front positive), the arm's swing (degrees, 0 folded, 90 out), its front face's
    depth behind the ears (m), and the force it presses with (N)."""
    out = {}
    for t in ("t", "b"):
        th, ph = q[f"screw_{t}"], q[f"nut_{t}"]
        c = max(0.0, q[f"wing_{t}"])
        out[t] = {"turns": th / (2 * math.pi), "swing": math.degrees(th + ph), "arm_x": ARM_X0 + q[f"slide_{t}"] + c,
                  "force": K_ARM * c}
    return out


# -- USD (after the simulator has started) -----------------------------------------------------------------------------

_USD_DIR: Path | None = None


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _apply(prim, schema: str, attrs: dict) -> None:
    from pxr import Sdf
    prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float,
             str: Sdf.ValueTypeNames.Token}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _ccw(prof):
    a = 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(prof, list(prof[1:]) + [prof[0]]))
    return list(prof) if a >= 0 else list(prof)[::-1]


def _loft_mesh(sh: Shape):
    p0, p1 = _ccw(sh.prof0), _ccw(sh.prof1)
    n = len(p0)
    pts = [sh.lift(p, sh.s0) for p in p0] + [sh.lift(p, sh.s1) for p in p1]
    counts, idx = [], []
    for k in range(1, n - 1):        # convex caps: fans
        counts += [3, 3]
        idx += [0, k + 1, k, n, n + k, n + k + 1]
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    return pts, counts, idx


def write_usd(kind: str, path: Path, fixed: bool) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Part").GetPrim()
    stage.SetDefaultPrim(root)
    looks, mats = {}, {}
    for name, (rgb, rough, metal) in LOOKS.items():
        mat = UsdShade.Material.Define(stage, f"/Part/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/Part/Looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(_srgb_to_linear, rgb)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        looks[name] = mat
    for name, (st, dy, combine) in MATERIALS.items():
        mat = UsdShade.Material.Define(stage, f"/Part/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(st)
        api.CreateDynamicFrictionAttr(dy)
        api.CreateRestitutionAttr(0.0)
        _apply(mat.GetPrim(), "PhysxMaterialAPI", {"physxMaterial:frictionCombineMode": combine})
        mats[name] = mat
    jointed = kind == "box"
    body_paths = {}
    for b in bodies_of(kind):
        bp = f"/Part/{b}" if jointed else "/Part"
        prim = UsdGeom.Xform.Define(stage, bp).GetPrim() if jointed else root
        body_paths[b] = bp
        if fixed:
            continue
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(MASS[b.split("_")[0]] if jointed else DRIVER_MASS)
        if jointed and b.split("_")[0] in ("nut", "slide"):     # no colliders: give them an inertia
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9, 1e-9, 1e-9))
            m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        _apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
            "physxRigidBody:enableCCD": False,
        })
    for i, sh in enumerate(shapes_of(kind)):
        parent = body_paths[sh.body]
        if sh.kind == "loft":
            geom = UsdGeom.Mesh.Define(stage, f"{parent}/loft_{i}")
            pts, counts, idx = _loft_mesh(sh)
            geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            geom.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            geom.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            geom.CreateDoubleSidedAttr(True)
        elif sh.kind == "box":
            geom = UsdGeom.Cube.Define(stage, f"{parent}/box_{i}")
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*sh.center))
            geom.AddScaleOp().Set(Gf.Vec3f(*sh.size))
        else:
            geom = UsdGeom.Cylinder.Define(stage, f"{parent}/cyl_{i}")
            geom.CreateRadiusAttr(sh.radius)
            geom.CreateHeightAttr(sh.length)
            geom.CreateAxisAttr(sh.axis)
            geom.AddTranslateOp().Set(Gf.Vec3d(*sh.center))
        prim = geom.GetPrim()
        if sh.look is None:
            UsdGeom.Imageable(prim).MakeInvisible()
        else:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[sh.look])
        if sh.material is None:
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        if sh.kind == "loft":
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI(prim).Bind(mats[sh.material], UsdShade.Tokens.strongerThanDescendants, "physics")
        _apply(prim, "PhysxCollisionAPI", {"physxCollision:contactOffset": CONTACT_OFFSET,
                                           "physxCollision:restOffset": 0.0})
    if jointed:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        _apply(root, "PhysxArticulationAPI", {
            "physxArticulation:enabledSelfCollisions": True,   # the clamp arm stops on the box's slot edge
            "physxArticulation:solverPositionIterationCount": 64,
            "physxArticulation:solverVelocityIterationCount": 4,
            "physxArticulation:sleepThreshold": 0.0,     # never asleep: a light touch of the tip must reach it
            "physxArticulation:stabilizationThreshold": 0.0,
        })
        ident = Gf.Quatf(1, 0, 0, 0)
        arm_travel = ARM_X0 - ARM_X_MIN
        for name, jt, parent, child, anchor in BOX_JOINTS:
            jp = f"/Part/joints/{name}"
            joint = (UsdPhysics.RevoluteJoint if jt == "rev" else UsdPhysics.PrismaticJoint).Define(stage, jp)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_paths[parent])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body_paths[child])])
            joint.CreateAxisAttr("X")
            if name.startswith("slide"):
                joint.CreateLowerLimitAttr(-arm_travel)
                joint.CreateUpperLimitAttr(0.0)
            elif name.startswith("wing"):
                joint.CreateLowerLimitAttr(0.0)
                joint.CreateUpperLimitAttr(ARM_FLEX)
            elif name.startswith("nut"):    # the thread's range (a mimic joint needs finite limits)
                joint.CreateLowerLimitAttr(-360.0 * arm_travel / LEAD - 2.0)
                joint.CreateUpperLimitAttr(2.0)
            else:                           # the screw: from 2 turns out to 6 turns in
                joint.CreateLowerLimitAttr(-720.0)
                joint.CreateUpperLimitAttr(2160.0)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalRot0Attr(ident)
            joint.CreateLocalRot1Attr(ident)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if jt == "rev" else "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(K_ARM if name.startswith("wing") else 0.0)
            drive.CreateDampingAttr(20.0 if name.startswith("wing") else 0.0)
            drive.CreateTargetPositionAttr(0.0)
            _apply(joint.GetPrim(), "PhysxJointAPI",
                   {"physxJoint:armature": SCREW_ARMATURE if name.startswith("screw") else LINK_ARMATURE})
        for t in ("t", "b"):
            # the thread: hub angle on the screw (deg) + gearing * travel along it (m) = 0, i.e. the hub turned
            # -360 degrees on the screw (the screw turned one turn clockwise in a held hub) moves it LEAD toward
            # the ears (-x)
            nut = stage.GetPrimAtPath(f"/Part/joints/nut_{t}")
            _apply(nut, "PhysxMimicJointAPI:rotX", {"physxMimicJoint:rotX:gearing": float(-360.0 / LEAD),
                                                    "physxMimicJoint:rotX:offset": 0.0})
            nut.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets(
                [Sdf.Path(f"/Part/joints/slide_{t}")])
    stage.GetRootLayer().Save()
    return path


def usd_file(kind: str, fixed: bool) -> str:
    global _USD_DIR
    if _USD_DIR is None:
        _USD_DIR = Path(tempfile.mkdtemp(prefix="owb_parts_"))
    path = _USD_DIR / f"{kind}{'_fixed' if fixed else ''}.usda"
    if not path.exists():
        write_usd(kind, path, fixed)
    return str(path)


_SPECS: dict[str, Part] = {}


def arena_object(spec: Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    _SPECS[spec.name] = spec

    class _Box(Object):
        """The box: the screws' and threads' friction and the arms' springs, written to PhysX by implicit actuators."""

        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.actuators = {
                "screws": ImplicitActuatorCfg(joint_names_expr=["screw_.*"], stiffness=0.0,
                                              damping=SCREW_DAMPING, armature=SCREW_ARMATURE, friction=T_HEAD,
                                              dynamic_friction=T_HEAD),
                "nuts": ImplicitActuatorCfg(joint_names_expr=["nut_.*"], stiffness=0.0, damping=0.0,
                                            armature=LINK_ARMATURE * 1e-3, friction=T_THREAD,
                                            dynamic_friction=T_THREAD),
                "slides": ImplicitActuatorCfg(joint_names_expr=["slide_.*"], stiffness=0.0, damping=0.0,
                                              armature=LINK_ARMATURE),
                "wings": ImplicitActuatorCfg(joint_names_expr=["wing_.*"], stiffness=K_ARM, damping=20.0,
                                             armature=LINK_ARMATURE),
            }
            cfg.init_state.joint_pos = {".*": 0.0}
            return cfg

    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=tuple(spec.quat))
    fixed = spec.mass is None
    cfg = sim_utils.UsdFileCfg(usd_path=usd_file(spec.kind, fixed))
    if spec.kind == "box":
        cfg.articulation_props = sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                                                                         solver_position_iteration_count=64,
                                                                         solver_velocity_iteration_count=4,
                                                                         sleep_threshold=0.0, stabilization_threshold=0.0)
        return _Box(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)
    return Object(name=spec.name, object_type=ObjectType.BASE if fixed else ObjectType.RIGID, spawner_cfg=cfg,
                  initial_pose=pose)


def as_numpy(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class PartState(task_file.ObjectState):
    """A part's ``ObjectState`` (``pos``/``quat_xyzw``: its origin's pose, world, m) plus, for the box, its joint
    positions (``joints``: name -> rad or m)."""

    joints: dict | None = None


def part_state(env, name: str) -> PartState:
    spec = _SPECS[name]
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    lo, hi = bounds(spec.kind)
    if spec.mass is None:
        pos, quat = np.asarray(spec.pos, float), np.asarray(spec.quat, float)
        lin = ang = np.zeros(3)
        joints = None
    else:
        obj = base.scene[name]
        data = obj.data
        pose = as_numpy(data.root_link_pose_w if hasattr(data, "root_link_pose_w") else data.root_pose_w)[0]
        pos, quat = pose[:3] - origin, pose[3:7]
        vel = as_numpy(data.root_com_vel_w)[0] if hasattr(data, "root_com_vel_w") else np.zeros(6)
        lin, ang = vel[:3], vel[3:6]
        joints = None
        if spec.kind == "box":
            q = as_numpy(data.joint_pos)[0]
            joints = {n: float(v) for n, v in zip(obj.joint_names, q)}
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)]) @ rotation(quat).T + pos
    return PartState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=lin, ang_vel=ang, low=corners.min(0),
                     high=corners.max(0), joints=joints)


# -- hooks: yam_arena.scene (spawning, states) and the embodiment (reset, step) -----------------------------------------

def _patch_scene(scene) -> None:
    orig_obj, orig_states = getattr(scene, "task_object", None), getattr(scene, "object_states", None)
    for fn, params in ((orig_obj, ["spec"]), (orig_states, ["env", "names"])):
        if not callable(fn) or list(inspect.signature(fn).parameters)[: len(params)] != params:
            raise RuntimeError("boxparts: yam_arena.scene no longer has task_object(spec) and object_states(env, names)")
    if getattr(orig_obj, "_owb", False):
        return

    def task_object(spec):
        return arena_object(spec) if isinstance(spec, Part) else orig_obj(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(env, plain) if plain else {}
        out.update({n: part_state(env, n) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    task_object._owb = True
    scene.task_object, scene.object_states = task_object, object_states


_RESET: list = []
_STEP: list = []


def on_reset(fn):
    """Register ``fn()`` to run after every episode reset."""
    _RESET.append(fn)
    return fn


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def _patch_embodiment(module) -> None:
    cls = getattr(module, "YamArenaEmbodiment", None)
    if cls is None or getattr(cls, "_owb", False):
        return
    base_reset, base_task_step = cls.reset, cls._task_step

    def reset(self, *args, **kwargs):
        obs = base_reset(self, *args, **kwargs)
        for fn in _RESET:
            fn()
        return obs

    def _task_step(self, *args, **kwargs):
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *args, **kwargs)

    cls.reset, cls._task_step = reset, _task_step
    cls._owb = True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _TARGETS[fullname](module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena.scene and the embodiment (now if already imported, otherwise when they are). Idempotent."""
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
