"""veinkit: an adult venipuncture training arm, a single-use tube holder with a multi-sample needle, and the cradle
the holder lies in, for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from veinkit import Cradle, Holder, Stick, TrainArm, on_reset, on_step

Frames (every part starts with its axes along the world's: x away from the robot, y left, z up).

- :class:`TrainArm`: the training arm, fixed. ``pos`` is the point on the table under the middle of the elbow crease.
  The arm lies palm up along y: the shoulder end toward +y, the hand toward -y. The vein runs along y on top (x 0).
- :class:`Holder`: the tube holder with its needle, a loose rigid part. ``pos`` is the middle of the barrel, on its
  axis. The needle points along its +y; the needle's bevel faces its +z (the side with the white mark on the hub).
- :class:`Cradle`: a fixed block with a V groove in which the holder lies level. ``pos`` is the point on the table
  under the middle of the barrel.

The needle is drawn but has no collider: a 0.8 mm shaft cannot be simulated as a body here, and the skin does not
resist it. What the needle does is worked out from where its tip and shaft are, by :class:`Stick` (pure Python): where
it crosses the skin, at what angle and with which side up, how deep the tip is, whether the vein was held, whether the
tip is in the vein or has gone through it.

All shapes are convex pieces built from point sets in NumPy when the task file is read (``*_pieces()``), so the same
numbers serve the scene, the scoring and the offline checks. Isaac Sim is reached only when the scene is built.

The structure follows the kits used by tasks already in this repository (Box subclasses so the loader accepts the
parts, custom spawners, hooks installed through ``sys.meta_path``).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Cradle", "Holder", "KitState", "Stick", "TrainArm", "arm_distance", "arm_pieces", "cradle_pieces", "env",
           "finger_points", "gripper_points", "holder_pieces", "install", "needle", "on_reset", "on_step", "place",
           "skin_top", "to_local", "write_pose"]

# -- the training arm (m), in its own frame: origin on the table under the middle of the elbow crease, y toward the
# -- shoulder, z up. Every size here is an estimate of an adult arm (the listing gives only the case's size). -------------
ELBOW_R = 0.042                  # the arm is round at the elbow, 8.4 cm across; it lies on the table
ELBOW_Y = (-0.060, 0.060)        # ... and is modelled as a level cylinder this far either side of the crease
SKIN_H = 2 * ELBOW_R             # the skin's height over the vein at the crease
UPPER = ((0.060, 0.042), (0.200, 0.048), (0.330, 0.050))            # upper arm sections: y, radius
FORE = ((-0.060, 0.042), (-0.150, 0.036), (-0.250, 0.029))          # forearm sections: y, radius (the wrist last)
PALM = ((-0.042, 0.042), (-0.345, -0.250), (0.004, 0.030))          # x, y, z extents
FINGERS = ((-0.038, 0.038), (-0.430, -0.345), (0.004, 0.022))
THUMB = ((0.036, -0.275), (0.068, -0.335), 0.010)                   # a rod from the palm's edge outward, and its radius
N_ARM = 24                       # sides of the arm's sections
SKIN = (0.87, 0.68, 0.56)
VEIN_BLUE = (0.25, 0.42, 0.70)
VEIN_R = 0.002                   # the vein's lumen: 4 mm across (an estimate)
VEIN_DEPTH = 0.0035              # the middle of the vein under the skin (an estimate)
VEIN_Y = (-0.200, 0.200)         # the vein runs this far along the arm
VEIN_ROLL = 0.005                # an unheld vein moves this far aside when the needle reaches it (an estimate)

# -- the site and the anchoring window, in the arm's frame --------------------------------------------------------------
SITE_X = 0.008                   # the needle has to cross the skin within this of the vein's line ...
SITE_Y = 0.012                   # ... and within this of the crease
ANCHOR_Y = (-0.140, -0.080)      # the anchoring fingertip: this far below (toward the hand from) the crease ...
ANCHOR_X = 0.020                 # ... and within this of the vein's line
ANCHOR_TOUCH = 0.0025            # on the skin: within this of it

# -- the holder and needle (m), in the holder's frame: origin at the middle of the barrel, y toward the needle's tip ------
BARREL_R = 0.0095                # 19 mm across (an estimate of a single-use holder for 13 and 16 mm tubes)
BARREL_Y = (-0.024, 0.024)       # 48 mm long (an estimate)
FLANGE = ((-0.017, 0.017), (-0.024, -0.022), (-0.012, 0.012))       # the finger flange at the open end (an estimate)
HUB = ((0.024, 0.005), (0.034, 0.002))                              # the needle's hub: y, radius at each end (estimates)
NEEDLE_L = 0.0254                # a 1 in needle
NEEDLE_R = 0.0004                # 21 gauge: 0.8 mm across
HUB_Y = HUB[1][0]                # where the needle leaves the hub
TIP_Y = HUB_Y + NEEDLE_L
HOLDER_MASS = 0.005              # holder and needle (an estimate)
HOLDER_BODY = 0.020              # the simulated body; a steady upward force carries the difference
HOLDER_FRICTION = (0.6, 0.5)     # polypropylene against the grippers' rubber (estimates)
CLEAR_PP = (0.86, 0.90, 0.93)
GREEN = (0.20, 0.62, 0.30)       # 21 gauge hubs are green
STEEL = (0.75, 0.76, 0.78)
WHITE = (0.96, 0.96, 0.96)

# -- the cradle (m), in its own frame: origin on the table under the middle of the barrel -------------------------------
AXIS_H = 0.030                   # the barrel's axis above the table when the holder lies in the cradle
V_HALF = 0.0085                  # half the V block's width
V_Y = (-0.015, 0.015)
V_TOP = AXIS_H - 0.006
V_APEX = AXIS_H - BARREL_R * math.sqrt(2.0)
SHELF = ((-0.019, 0.019), (-0.028, -0.018), (0.0, AXIS_H + FLANGE[2][0] - 0.0003))   # under the flange: keeps it from rolling
CRADLE_GREY = (0.55, 0.57, 0.60)
FRICTION = (0.8, 0.7)            # vinyl skin and the cradle against the grippers' rubber, and against each other

# -- the needle's rule (m, degrees) -------------------------------------------------------------------------------------
ANGLE = (10.0, 30.0)             # the needle's angle to the skin where it goes in
BEVEL = 30.0                     # the bevel faces away from the skin within this
IN_SKIN = 0.001                  # the needle is in the skin once its tip is this far under it
SWEEP = 0.002                    # the shaft may not move sideways in the skin by more than this
PALM_BACK = 0.065
FINGER_THICK = 0.012


# -- shapes (pure): every piece is a set of points whose convex hull is the piece -----------------------------------------

def _ring(y: float, r: float, zc: float | None = None, n: int = N_ARM, x: float = 0.0) -> np.ndarray:
    """A ring of ``n`` points of radius ``r`` round the y axis at ``y``; its middle ``zc`` up (default: lying on z 0)."""
    t = 2 * math.pi * (np.arange(n) / n) + math.pi / 2          # one point exactly on top
    zc = r if zc is None else zc
    return np.column_stack([x + r * np.cos(t), np.full(n, y), zc + r * np.sin(t)])


def _slab(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def _rod(p0, p1, r0: float, r1: float | None = None, n: int = 8) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(d, (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    t = 2 * math.pi * np.arange(n) / n
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    return np.vstack([p0 + r0 * ring, p1 + (r0 if r1 is None else r1) * ring])


def fore_top(y: float) -> float:
    """The height of the forearm's skin on the vein's line at ``y`` (between the elbow and the wrist)."""
    ys, rs = zip(*FORE)
    return 2.0 * float(np.interp(-y, [-v for v in ys], rs))


def arm_pieces() -> list:
    """The training arm: (points, colour, solid) pieces in its own frame. The vein is drawn, not solid."""
    out = [(np.vstack([_ring(ELBOW_Y[0], ELBOW_R), _ring(ELBOW_Y[1], ELBOW_R)]), SKIN, True)]
    for (ya, ra), (yb, rb) in zip(UPPER, UPPER[1:]):
        out.append((np.vstack([_ring(ya, ra), _ring(yb, rb)]), SKIN, True))
    out.append((np.vstack([_ring(UPPER[-1][0], UPPER[-1][1]), _ring(UPPER[-1][0] + 0.020, 0.036, zc=UPPER[-1][1])]), SKIN, True))
    for (ya, ra), (yb, rb) in zip(FORE, FORE[1:]):
        out.append((np.vstack([_ring(ya, ra), _ring(yb, rb)]), SKIN, True))
    out.append((_slab(*zip(*PALM)), SKIN, True))
    out.append((_slab(*zip(*FINGERS)), SKIN, True))
    (x0, y0), (x1, y1), r = THUMB
    out.append((_rod((x0, y0, 0.018), (x1, y1, 0.014), r), SKIN, True))
    # the vein: a thin blue ribbon on the skin, in level runs
    w, t = 0.002, 0.0006
    runs = [(ELBOW_Y[0], ELBOW_Y[1], SKIN_H, SKIN_H)]
    runs += [(ya, yb, 2 * ra, 2 * rb) for (ya, ra), (yb, rb) in zip(UPPER, UPPER[1:]) if ya < VEIN_Y[1]]
    runs += [(yb, ya, 2 * rb, 2 * ra) for (ya, ra), (yb, rb) in zip(FORE, FORE[1:]) if ya > VEIN_Y[0]]
    for ya, yb, za, zb in runs:
        ya, yb = max(ya, VEIN_Y[0]), min(yb, VEIN_Y[1])
        pts = [[sx * w, y, z - 0.0002 + dz] for (y, z) in ((ya, za), (yb, zb)) for sx in (-1, 1) for dz in (0.0, t)]
        out.append((np.array(pts), VEIN_BLUE, False))
    return out


def holder_pieces() -> list:
    """The tube holder and needle: (points, colour, solid) pieces in the holder's frame. The needle is drawn only."""
    ring = lambda y, r, n=12: np.column_stack([r * np.cos(2 * math.pi * np.arange(n) / n), np.full(n, y),  # noqa: E731
                                                r * np.sin(2 * math.pi * np.arange(n) / n)])
    out = [(np.vstack([ring(BARREL_Y[0], BARREL_R), ring(BARREL_Y[1], BARREL_R)]), CLEAR_PP, True),
           (_slab(*zip(*FLANGE)), CLEAR_PP, True),
           (np.vstack([ring(HUB[0][0], HUB[0][1], 8), ring(HUB[1][0], HUB[1][1], 8)]), GREEN, True),
           (np.vstack([ring(HUB_Y, 0.0006, 6), ring(TIP_Y - 0.002, 0.0006, 6)]), STEEL, False),
           # the bevel: the tip is cut so that its point is on the -z side and its open face looks up (+z)
           (np.vstack([ring(TIP_Y - 0.002, 0.0006, 6), [[0.0, TIP_Y, -0.0006]]]), STEEL, False),
           (_slab((-0.0015, HUB[0][0], HUB[0][1] - 0.0003), (0.0015, HUB[0][0] + 0.007, HUB[0][1] + 0.0012)), WHITE, False)]
    return out


def cradle_pieces() -> list:
    """The cradle: two wedges that make a V groove under the barrel, and a shelf under the holder's flange."""
    out = []
    for sx in (-1.0, 1.0):
        prof = [(sx * V_HALF, 0.0), (0.0, 0.0), (0.0, V_APEX), (sx * (V_TOP - V_APEX), V_TOP), (sx * V_HALF, V_TOP)]
        out.append((np.array([[x, y, z] for y in V_Y for x, z in prof]), CRADLE_GREY, True))
    out.append((_slab(*zip(*SHELF)), CRADLE_GREY, True))
    return out


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
class TrainArm(_Part):
    """The training arm, fixed. ``pos``: on the table under the middle of the elbow crease."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 2.0


@dataclass(frozen=True)
class Cradle(_Part):
    """The holder's cradle, fixed. ``pos``: on the table under the middle of the barrel."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 0.2


@dataclass(frozen=True)
class Holder(_Part):
    """The tube holder with its needle, loose. ``pos``: the middle of the barrel. ``mass`` is the real part's; the
    simulated body weighs ``HOLDER_BODY`` and a steady upward force carries the difference."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = HOLDER_MASS


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


def turn_x(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn about x by ``angle`` rad."""
    return (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))


def needle(holder_pos, holder_quat) -> dict:
    """The needle in the world from the holder's pose: ``tip``, ``base`` (where it leaves the hub), ``dir`` (unit,
    base to tip) and ``up`` (unit, the way the bevel faces)."""
    rot = _rotation(holder_quat)
    pos = np.asarray(holder_pos, float)
    return {"tip": pos + rot[:, 1] * TIP_Y, "base": pos + rot[:, 1] * HUB_Y, "dir": rot[:, 1].copy(), "up": rot[:, 2].copy()}


def skin_top(x: float) -> float:
    """The height of the skin over the elbow, ``x`` from the vein's line (the arm's frame)."""
    return ELBOW_R + math.sqrt(max(ELBOW_R ** 2 - x * x, 0.0))


_HULLS: dict = {}


def _arm_planes() -> list:
    """Each solid convex piece of the arm as outward planes (normals, offsets), for point-to-arm distances."""
    if "arm" not in _HULLS:
        planes = []
        for pts, _, solid in arm_pieces():
            if not solid:
                continue
            ns, ds = [], []
            for f in hull_faces(pts):
                a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
                n = np.cross(b - a, c - a)
                n /= np.linalg.norm(n)
                ns.append(n)
                ds.append(float(n @ a))
            planes.append((np.array(ns), np.array(ds)))
        _HULLS["arm"] = planes
    return _HULLS["arm"]


def arm_distance(points) -> np.ndarray:
    """For each of ``points`` (the arm's frame): its distance (m, at least 0) outside the arm's convex pieces (the
    largest plane distance of the nearest piece, which is exact off a face and a little small off an edge)."""
    q = np.atleast_2d(np.asarray(points, float))
    d = np.full(len(q), 9.0)
    for ns, ds in _arm_planes():
        d = np.minimum(d, np.maximum(0.0, (q @ ns.T - ds).max(axis=1)))
    return d


def anchored(points) -> bool:
    """Whether any of ``points`` (a gripper's finger points, the arm's frame) is on the forearm's skin inside the
    anchoring window: 8 to 14 cm below the crease, within 2 cm of the vein's line, within 2.5 mm of the skin."""
    q = np.atleast_2d(np.asarray(points, float))
    if not len(q):
        return False
    inside = (q[:, 1] >= ANCHOR_Y[0]) & (q[:, 1] <= ANCHOR_Y[1]) & (np.abs(q[:, 0]) <= ANCHOR_X) & (q[:, 2] > 0.02)
    if not inside.any():
        return False
    return bool((arm_distance(q[inside]) <= ANCHOR_TOUCH).any())


class Stick:
    """What the needle does, from where it is (pure; everything in the arm's frame). Call :meth:`step` once per
    control step with the needle's tip, direction and bevel-up vector, and whether the vein is held just then.

    - The needle is **in the skin** once its tip is 1 mm under the skin over the elbow. The crossing is **good** if
      it is at the site (within 8 mm of the vein's line and 12 mm of the crease), 10 to 30 degrees to the skin, with
      the bevel facing away from the skin within 30 degrees. Any other crossing is a bad stick.
    - The **vein is reached** when the tip is as deep as the vein's near wall. If the vein is not held at that moment
      it moves 5 mm aside, away from the needle, and stays there while the needle is in. That is a rule, not physics.
    - The tip is **in the vein** while it is inside the vein's 4 mm lumen. It has gone **through** once it is deeper
      than the vein's far wall under the lumen.
    - The shaft may not be **swept** sideways in the skin: where it crosses the skin may not move more than 2 mm.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.state = "out"                    # "out" or "in"
        self.sticks = 0                       # how many times the needle has gone into the skin
        self.good = False                     # the present stick crossed the skin at the site, angle and bevel right
        self.entry = None                     # where it crossed the skin
        self.entry_angle = self.entry_bevel = None
        self.why = None                       # why the present stick is bad
        self.reached = self.held = False      # the tip has reached the vein; the vein was held when it did
        self.kept = True                      # ... and has been held ever since
        self.vein_x = 0.0                     # where the vein is now (it may have rolled aside)
        self.in_vein = False
        self.through = False
        self.swept = False
        self.depth = 0.0                      # the tip under the skin (m, negative outside)
        self.angle = self.bevel = 0.0         # the needle's angle to the skin and the bevel's error, now (degrees)
        self.off_site = False                 # the tip is at the skin away from the site
        self.over = False                     # the tip is over the elbow's skin

    @staticmethod
    def attitude(n, up, x: float) -> tuple:
        """(angle to the skin, bevel error) in degrees for a needle along ``n`` with its bevel facing ``up``, over the
        skin ``x`` from the vein's line."""
        s = max(-1.0, min(1.0, x / ELBOW_R))
        normal = np.array([s, 0.0, math.sqrt(1.0 - s * s)])
        n, up = np.asarray(n, float), np.asarray(up, float)
        angle = math.degrees(math.asin(max(-1.0, min(1.0, float(-(n @ normal))))))
        a, b = up - (up @ n) * n, normal - (normal @ n) * n
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        bevel = 180.0 if la < 1e-9 or lb < 1e-9 else math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (la * lb)))))
        return angle, bevel

    def step(self, tip, n, up, held_now: bool) -> None:
        tip, n = np.asarray(tip, float), np.asarray(n, float)
        x, y, z = (float(v) for v in tip)
        self.over = ELBOW_Y[0] <= y <= ELBOW_Y[1] and abs(x) < ELBOW_R
        self.depth = (ELBOW_R - math.hypot(x, z - ELBOW_R)) if self.over else -1.0
        self.angle, self.bevel = self.attitude(n, up, x)
        at_site = abs(x) <= SITE_X and abs(y) <= SITE_Y
        if self.state == "out":
            self.off_site = bool(self.over and self.depth > -0.0005 and not at_site)
            if self.over and self.depth >= IN_SKIN:
                self.state, self.sticks = "in", self.sticks + 1
                s = (self.depth / max(math.sin(math.radians(max(self.angle, 1.0))), 0.05))
                self.entry = tip - n * min(s, NEEDLE_L)
                self.entry_angle, self.entry_bevel = self.angle, self.bevel
                self.why = None
                if not (abs(float(self.entry[0])) <= SITE_X and abs(float(self.entry[1])) <= SITE_Y):
                    self.why = "away from the site"
                elif not (ANGLE[0] <= self.angle <= ANGLE[1]):
                    self.why = f"at {self.angle:.0f} degrees"
                elif self.bevel > BEVEL:
                    self.why = f"bevel {self.bevel:.0f} degrees off"
                self.good = self.why is None
                self.reached = self.held = self.in_vein = False
                self.kept, self.vein_x = True, 0.0
            return
        # in the skin
        if not self.over or self.depth < 0.0:
            self.state, self.in_vein = "out", False          # drawn out again
            return
        to_entry = self.entry - tip
        side = to_entry - (to_entry @ n) * n                 # from the shaft's line to where it crossed the skin
        if float(np.linalg.norm(side)) > SWEEP:
            self.swept = True
        near = SKIN_H - VEIN_DEPTH + VEIN_R                  # the height of the vein's near wall
        if not self.reached and z <= near and abs(x) <= VEIN_R + VEIN_ROLL:
            self.reached, self.held, self.kept = True, bool(held_now), bool(held_now)
            if not held_now:
                self.vein_x = VEIN_ROLL if x <= 0.0 else -VEIN_ROLL
        if self.reached and self.held and not held_now:
            self.kept = False
        zc = SKIN_H - VEIN_DEPTH
        self.in_vein = bool(self.reached and VEIN_Y[0] <= y <= VEIN_Y[1] and math.hypot(x - self.vein_x, z - zc) < VEIN_R)
        if self.reached and abs(x - self.vein_x) < VEIN_R and z < zc - VEIN_R:
            self.through = True

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


def _start(cfg, prim_path, translation, orientation, mass, iterations=None, fixed=False):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(mass), iterations=iterations, fixed=fixed)
    return spec, a


def _spawn_arm(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, 2.0, fixed=True)
    a.pieces(f"{prim_path}/p", arm_pieces(), a.mat("skin", *FRICTION), rough=0.7)
    return a.root


def _spawn_cradle(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, 0.2, fixed=True)
    a.pieces(f"{prim_path}/p", cradle_pieces(), a.mat("cradle", 0.4, 0.3), rough=0.6)
    return a.root


def _spawn_holder(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a = _start(cfg, prim_path, translation, orientation, HOLDER_BODY, iterations=32)
    a.pieces(f"{prim_path}/p", holder_pieces(), a.mat("pp", *HOLDER_FRICTION), rough=0.3, patch=0.004)
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



_SPAWNERS = {TrainArm: _spawn_arm, Cradle: _spawn_cradle, Holder: _spawn_holder}
_PIECES = {TrainArm: arm_pieces, Cradle: cradle_pieces, Holder: holder_pieces}
_BOUNDS: dict = {}


def _carry(environment) -> None:
    """Hold on the holder the steady upward force that carries what its simulated body weighs beyond the real part."""
    import torch

    for n, spec in _SPECS.items():
        if not isinstance(spec, Holder) or n not in _PATHS:
            continue
        extra = (HOLDER_BODY - float(spec.mass)) * 9.81
        try:
            asset = environment.unwrapped.scene[n]
            ft = torch.tensor([[[0.0, 0.0, float(extra)]]], dtype=torch.float32, device=asset.device)
            asset.permanent_wrench_composer.set_forces_and_torques_index(forces=ft, torques=torch.zeros_like(ft), is_global=True)
        except Exception as err:                     # noqa: BLE001
            print(f"[veinkit] weight not carried for {n}: {err!r}", file=sys.stderr, flush=True)


def _local_bounds(spec) -> tuple:
    if spec.name not in _BOUNDS:
        pts = np.vstack([p[0] for p in _PIECES[type(spec)]()])
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
    if getattr(scene, "_veinkit_patched", False):
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
    scene._veinkit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_veinkit_patched", False):
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
        _carry(self._env)
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._veinkit_patched = True


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
