"""esukit: a hand-switched electrosurgical pencil with two sprung keys and its cord, a pencil holster, and an
instrument tray on a Mayo stand, for yam_arena task files.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from esukit import Clip, Holster, Key, Link, Pencil, Tray, on_reset, on_step

Frames (m):

- :class:`Pencil`: the handle with its blade electrode, one loose rigid part. ``pos`` is the handle's nose, on its
  axis, where the electrode comes out. Its x runs back along the handle to the cord end, its z points out of the face
  that carries the two keys, its y across. The electrode's tip is at x = -EXPOSED.
- :class:`Key`: the CUT or COAG key, a loose part held to the pencil by a sprung slide along the pencil's z.
- :class:`Link`: one link of the cord (a round bar PITCH long). ``pos`` is its end nearer the pencil; its x runs along it.
  Ball joints join the pencil's cord end to the first link, each link to the next, and the last to the clip.
- :class:`Clip`: the cord clip that fixes the cord to the drape, fixed. ``pos`` is the point where the cord is held.
- :class:`Holster`: the pencil holster, fixed. ``pos`` is the middle of its mouth, on its inside floor; its x runs into
  the holster, its z up.
- :class:`Tray`: an instrument tray on a draped Mayo stand, with three instruments in it, fixed. ``pos`` is the middle
  of the tray's floor (its top surface); its x runs along its short side.

All shapes are convex pieces built from point sets in NumPy when the task file is read, so the same numbers serve the
scene, the scoring and the offline checks. Isaac Sim is reached only when the scene is built.

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

__all__ = ["Clip", "Holster", "Key", "KitState", "Link", "Pencil", "Tray", "cord_path", "env", "finger_points",
           "gripper_points", "holster_pieces", "in_holster", "install", "is_box", "key_rest", "key_travel", "on_reset", "on_step",
           "on_tray", "pencil_pieces", "place", "side_quat", "to_local", "tray_pieces", "turn_z", "write_pose"]

# -- the pencil (Pencil frame) -------------------------------------------------------------------------------------------
HANDLE = 0.160                # handle length (data sheets of hand-switched pencils give 160 mm)
EXPOSED = 0.045               # electrode beyond the nose (an estimate: a 70 mm electrode, about 25 mm of it in the handle)
ROD = 0.00118                 # the electrode's radius (2.36 mm diameter, from a pencil's data sheet)
SLEEVE = 0.0019               # ... and of the insulating sleeve over its first 2 cm (an estimate)
# the handle's sections: x from the nose, half width (y), half height (z). Estimates from product photographs.
SECTIONS = ((0.000, 0.0045, 0.0050), (0.012, 0.0058, 0.0068), (0.024, 0.0065, 0.0080), (0.110, 0.0065, 0.0080),
            (0.140, 0.0058, 0.0066), (0.160, 0.0045, 0.0050))
WIDE, TALL = 0.013, 0.016     # the handle's middle: across, and from its underside to the face with the keys
CHAMFER = 0.003
KEYS = {"cut": 0.031, "coag": 0.049}   # each key's middle, back from the nose (estimates)
KEY_L, KEY_W, KEY_T = 0.010, 0.007, 0.003   # a key: along the handle, across, and its thickness
KEY_PROUD = 0.0015            # a key stands this far out of the handle
KEY_TRAVEL = 0.0010           # ... and can be pushed in this far
KEY_ON = 0.0006               # the switch closes once a key is pushed in this far
KEY_PRELOAD = 1.5             # N: the force that starts a key moving (an estimate)
KEY_FORCE_ON = 2.5            # N: the force at which the switch closes (an estimate; not a measured device)
KEY_SPRING = (KEY_FORCE_ON - KEY_PRELOAD) / KEY_ON     # N/m
KEY_MASS = 0.002              # kg each: heavier than a real key cap so that the contact solver holds it; the handle is
                              # lighter by the same amount, so the pencil's weight is unchanged
PENCIL_MASS = 0.028           # handle, electrode and both keys (an estimate)
KEYS_BACK = max(KEYS.values()) + KEY_L / 2    # the back edge of the rear key, from the nose (0.054)
PAD_HALF = 0.014              # half the width of a gripper pad along the handle (13.7 mm on the robot's fingertip mesh)

# -- the cord ----------------------------------------------------------------------------------------------------------
PITCH = 0.025                 # one link
CORD_R = 0.00175              # 3.5 mm cord (an estimate)
LINK_MASS = 0.0008            # kg a link: 32 g a metre (an estimate for a three-wire PVC cord)
CORD_BEND = 60.0              # degrees one link can bend from the next

# -- the holster (Holster frame) -----------------------------------------------------------------------------------------
H_IN_W, H_IN_H, H_DEEP = 0.030, 0.020, 0.175    # inside width, inside height, depth (estimates)
H_WALL = 0.0015
H_FLARE = 0.008               # the mouth's lip: this long, opening out to ...
H_LIP_W, H_LIP_H = 0.040, 0.026

# -- the tray on its Mayo stand (Tray frame) -------------------------------------------------------------------------------
TRAY_IN = (0.248, 0.346)      # the tray's short and long sides (a 13 5/8 x 9 3/4 x 5/8 inch instrument tray)
TRAY_RIM = 0.016
TRAY_WALL = 0.001
STAND_H = 0.040               # the draped stand top under the tray, above the patient drape (an estimate; see README)
STAND_PAD = 0.008             # the stand top shows this far round the tray

ABS, YELLOW, BLUE_KEY, STEEL = (0.93, 0.93, 0.90), (0.95, 0.80, 0.10), (0.10, 0.25, 0.75), (0.74, 0.75, 0.77)
CORD, ORANGE, DRAPE = (0.88, 0.88, 0.86), (0.95, 0.62, 0.12), (0.20, 0.47, 0.60)
FRICTION = (0.5, 0.4)
PAD_PATCH = 0.003             # radius of the patch over which a rubber gripper pad meets the handle (an estimate)
PALM_BACK = 0.065
FINGER_THICK = 0.012
N_RING = 8


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


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def turn_z(angle: float) -> tuple:
    """Quaternion (xyzw) of a turn of ``angle`` rad about the vertical."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def side_quat(yaw: float, keys: float = 1.0) -> tuple:
    """Quaternion of a pencil lying on its side with its cord end pointing along ``yaw`` (rad from the world's x):
    ``keys`` +1 turns the keys to the left of that direction, -1 to the right."""
    return _quat(_rz(yaw) @ _rx(-keys * math.pi / 2))


def place(pos, quat, local) -> np.ndarray:
    """A point given in a part's own frame, in the world."""
    return np.asarray(pos, float) + _rotation(quat) @ np.asarray(local, float)


def to_local(point, pos, quat) -> np.ndarray:
    return _rotation(quat).T @ (np.asarray(point, float) - np.asarray(pos, float))


def key_rest(which: str) -> np.ndarray:
    """Where a key's middle is in the pencil's frame when nothing presses it."""
    return np.array([KEYS[which], 0.0, TALL / 2 + KEY_PROUD - KEY_T / 2])


def key_travel(which: str, pencil_pos, pencil_quat, key_pos) -> float:
    """How far a key is pushed in (m; 0 at rest)."""
    return float(key_rest(which)[2] - to_local(key_pos, pencil_pos, pencil_quat)[2])


def in_holster(pencil_pos, pencil_quat, holster_pos, holster_quat) -> dict:
    """Where the pencil is in the holster's frame: ``nose`` and ``tip`` (x into the holster from its mouth, y off its
    midline, z above its floor), ``keys_in`` how far the back edge of the rear key is inside the mouth (m),
    ``electrode_in`` whether the whole electrode is inside, ``inside`` whether nose, tip and the rear key are between
    the holster's walls, floor and roof."""
    def loc(x):
        return to_local(place(pencil_pos, pencil_quat, (x, 0.0, 0.0)), holster_pos, holster_quat)
    nose, tip, back = loc(0.0), loc(-EXPOSED), loc(KEYS_BACK)

    def walls(p):
        return bool(abs(p[1]) <= H_IN_W / 2 and -H_WALL <= p[2] <= H_IN_H)
    electrode = bool(0.0 <= nose[0] < tip[0] <= H_DEEP and walls(nose) and walls(tip))     # electrode first
    return {"nose": nose, "tip": tip, "keys_in": float(back[0]), "electrode_in": electrode,
            "inside": bool(electrode and walls(back))}


def on_tray(pencil_pos, pencil_quat, tray_pos, tray_quat) -> dict:
    """Whether the pencil lies in the tray: ``ends`` both ends of its handle over the tray's floor, ``up`` the height
    of the handle's middle above that floor (m)."""
    pts = [to_local(place(pencil_pos, pencil_quat, (x, 0.0, 0.0)), tray_pos, tray_quat) for x in (0.0, HANDLE / 2, HANDLE)]
    over = all(abs(p[0]) <= TRAY_IN[0] / 2 and abs(p[1]) <= TRAY_IN[1] / 2 for p in pts)
    return {"ends": bool(over), "up": float(pts[1][2])}


def segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def cord_path(start, heading, end, n: int, pitch: float = PITCH) -> np.ndarray:
    """``n + 1`` points on the drape, each ``pitch`` from the last: a cord of ``n`` links that leaves ``start`` along
    ``heading`` and ends at ``end`` (2-vectors). Its slack lies in one smooth bow, as a limp cord's does: the links'
    directions are those with the least bending between neighbours that still bring the cord's end to ``end``."""
    p0, p1 = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
    chord = p1 - p0
    span = float(np.linalg.norm(chord))
    if span > n * pitch:
        raise ValueError("the cord cannot reach its clip")
    lead = math.atan2(float(heading[1]), float(heading[0]))
    along = math.atan2(chord[1], chord[0])
    off = (lead - along + math.pi) % (2 * math.pi) - math.pi
    side = 1.0 if off >= 0 else -1.0                      # the bow goes to the side the cord leaves on
    th = along + side * 1.2 * np.cos(math.pi * np.arange(n) / (n - 1))
    th[0] = lead

    def tip(angles) -> np.ndarray:
        return p0 + pitch * np.array([np.cos(angles).sum(), np.sin(angles).sum()])

    rows = np.zeros((n - 1, n))                           # bending between neighbours
    rows[np.arange(n - 1), np.arange(n - 1)], rows[np.arange(n - 1), np.arange(1, n)] = -1.0, 1.0
    for weight in (30.0, 300.0, 3000.0, 30000.0):         # the end is pulled to its place ever harder
        damp = 1e-3
        for _ in range(400):
            res = np.r_[rows @ th, weight * (tip(th) - p1)]
            jac = np.vstack([rows, weight * pitch * np.vstack([-np.sin(th), np.cos(th)])])[:, 1:]   # the first link is fixed
            step = np.linalg.solve(jac.T @ jac + damp * np.eye(n - 1), -jac.T @ res)
            trial = th.copy()
            trial[1:] += step
            new = np.r_[rows @ trial, weight * (tip(trial) - p1)]
            if new @ new < res @ res:
                th, damp = trial, max(damp / 3, 1e-9)
                if np.abs(step).max() < 1e-11:
                    break
            else:
                damp = min(damp * 5, 1e9)
                if damp >= 1e9:
                    break
    pts = p0 + pitch * np.vstack([np.zeros(2), np.cumsum(np.column_stack([np.cos(th), np.sin(th)]), axis=0)])
    return pts


# -- shapes (pure): every piece is a set of points whose convex hull is the piece -----------------------------------------

def _section(x: float, hw: float, hh: float) -> np.ndarray:
    """A handle section in the y-z plane: a rectangle with its corners cut."""
    c = min(CHAMFER, 0.45 * hw, 0.45 * hh)
    yz = [(hw, hh - c), (hw - c, hh), (-hw + c, hh), (-hw, hh - c), (-hw, -hh + c), (-hw + c, -hh), (hw - c, -hh), (hw, -hh + c)]
    return np.array([[x, y, z] for y, z in yz])


def _box(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def is_box(points) -> bool:
    """Whether a piece is a box with its edges along the axes (eight corners)."""
    p = np.asarray(points, float)
    if len(p) != 8:
        return False
    lo, hi = p.min(0), p.max(0)
    corners = {tuple(np.round(c, 9)) for c in _box(lo, hi)}
    return corners == {tuple(np.round(c, 9)) for c in p} and len(corners) == 8


def _rod(p0, p1, r0: float, r1: float | None = None) -> np.ndarray:
    """A round bar from ``p0`` to ``p1``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(d, (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    t = 2 * math.pi * np.arange(N_RING) / N_RING
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    return np.vstack([p0 + r0 * ring, p1 + (r0 if r1 is None else r1) * ring])


def pencil_pieces() -> list:
    """The handle and its electrode, without the keys: (points, colour) pieces in the Pencil's frame."""
    secs = [_section(*s) for s in SECTIONS]
    out = [(np.vstack(pair), ABS) for pair in zip(secs, secs[1:])]
    out.append((_rod((0.0, 0.0, 0.0), (-0.020, 0.0, 0.0), SLEEVE), (0.25, 0.27, 0.30)))        # insulating sleeve
    out.append((_rod((-0.020, 0.0, 0.0), (-0.026, 0.0, 0.0), ROD), STEEL))
    blade = [[x, y, z] for x in (-0.026, -EXPOSED) for y in (-ROD, ROD) for z in (-0.0003, 0.0003)]
    out.append((np.array(blade + [[-0.026, 0.0, -ROD], [-0.026, 0.0, ROD]]), STEEL))            # the flat blade
    return out


def key_pieces(which: str) -> list:
    """A key, in its own frame (origin at its middle)."""
    return [(_box((-KEY_L / 2, -KEY_W / 2, -KEY_T / 2), (KEY_L / 2, KEY_W / 2, KEY_T / 2)), YELLOW if which == "cut" else BLUE_KEY)]


def holster_pieces() -> list:
    """The holster: a flat tube closed at its far end, with a lip round its mouth, in the Holster's frame."""
    w, h, d, t = H_IN_W / 2, H_IN_H, H_DEEP, H_WALL
    out = [(_box((-H_FLARE, -w - t, -t), (d + t, w + t, 0.0)), ORANGE)]                            # floor, out to the lip
    out.append((_box((0.0, -w - t, h), (d + t, w + t, h + t)), ORANGE))                           # roof
    for s in (-1.0, 1.0):
        out.append((_box((0.0, min(s * w, s * (w + t)), 0.0), (d + t, max(s * w, s * (w + t)), h)), ORANGE))   # side
        lip = [[0.0, s * y, z] for y in (w, w + t) for z in (0.0, h)]
        lip += [[-H_FLARE, s * y, z] for y in (H_LIP_W / 2 - t, H_LIP_W / 2) for z in (0.0, H_LIP_H)]
        out.append((np.array(lip), ORANGE))                                                       # side lip
    top = [[0.0, y, z] for y in (-w - t, w + t) for z in (h, h + t)]
    top += [[-H_FLARE, y, z] for y in (-H_LIP_W / 2, H_LIP_W / 2) for z in (H_LIP_H, H_LIP_H + t)]
    out.append((np.array(top), ORANGE))                                                           # roof lip
    out.append((_box((d, -w - t, 0.0), (d + t, w + t, h)), ORANGE))                               # end wall
    out.append((_box((0.020, -w - t, -t - 0.0004), (d - 0.020, w + t, -t)), (0.95, 0.95, 0.92)))  # adhesive strip
    return out


def _ring(centre, r: float, bar: float) -> list:
    """A finger ring lying flat: eight short bars."""
    c = np.asarray(centre, float)
    t = 2 * math.pi * np.arange(9) / 8
    pts = c + np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(9)])
    return [_rod(a, b, bar) for a, b in zip(pts, pts[1:])]


def _ring_instrument(at, length: float, yaw: float) -> list:
    """A closed ring-handled instrument lying flat (rings at its origin end, jaws at the other), turned ``yaw``."""
    z = 0.0035
    bars = _ring((0.011, 0.014, z), 0.0115, 0.0022) + _ring((0.011, -0.014, z), 0.0115, 0.0022)
    hinge = length * 0.62
    for s in (-1.0, 1.0):
        bars.append(_rod((0.022, s * 0.010, z), (hinge, s * 0.0025, z), 0.0025, 0.0030))
        bars.append(_rod((hinge, s * 0.0025, z), (length, s * 0.0012, z), 0.0030, 0.0014))
    bars.append(_rod((hinge - 0.004, 0.0, z), (hinge + 0.004, 0.0, z), 0.0042))                 # the box lock
    r = _rz(yaw)
    return [(b @ r.T + np.asarray(at, float), STEEL) for b in bars]


def tray_pieces() -> list:
    """The draped stand top, the tray and three instruments in it, in the Tray's frame."""
    a, b = TRAY_IN[0] / 2, TRAY_IN[1] / 2
    t, rim, pad = TRAY_WALL, TRAY_RIM, STAND_PAD
    out = [(_box((-a - pad, -b - pad, -t - STAND_H), (a + pad, b + pad, -t)), DRAPE)]            # the stand's draped top
    out.append((_box((-a - t, -b - t, -t), (a + t, b + t, 0.0)), STEEL))                         # tray floor
    for s in (-1.0, 1.0):
        out.append((_box((min(s * a, s * (a + t)), -b - t, 0.0), (max(s * a, s * (a + t)), b + t, rim)), STEEL))
        out.append((_box((-a - t, min(s * b, s * (b + t)), 0.0), (a + t, max(s * b, s * (b + t)), rim)), STEEL))
    quarter = math.pi / 2                                                                        # the instruments lie along the tray
    out += _ring_instrument((0.030, -0.135, 0.0), 0.140, quarter + math.radians(4.0))             # a haemostat
    out += _ring_instrument((0.072, -0.145, 0.0), 0.160, quarter - math.radians(3.0))             # a needle holder
    for s in (-1.0, 1.0):                                                                        # dressing forceps
        blade = [[x, s * (0.0045 - 0.004 * x / 0.14) + s * e, z] for x in (0.0, 0.14) for e in (0.0, 0.0012)
                 for z in (0.0006, 0.0090 - 0.006 * x / 0.14)]
        out.append((np.array(blade) @ _rz(quarter + math.radians(2.0)).T + (0.105, -0.125, 0.0), STEEL))
    return out


def clip_pieces() -> list:
    """The cord clip stuck to the drape, and the cord running on from it to the table's far edge (drawn only)."""
    out = [(_box((-0.010, -0.009, -CORD_R - 0.001), (0.010, 0.009, CORD_R + 0.0035)), (0.25, 0.27, 0.30))]
    return out


def link_points() -> np.ndarray:
    return _rod((0.0, 0.0, 0.0), (PITCH, 0.0, 0.0), CORD_R)


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


# -- the parts -----------------------------------------------------------------------------------------------------------

class _Part(Box):
    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")


@dataclass(frozen=True)
class Pencil(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = PENCIL_MASS - 2 * KEY_MASS


@dataclass(frozen=True)
class Key(_Part):
    """A key of ``pencil``; ``which`` is "cut" or "coag"."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = KEY_MASS
    pencil: str = "pencil"
    which: str = "cut"


@dataclass(frozen=True)
class Link(_Part):
    """A cord link, joined at its start to ``prev`` (the pencil for the first link, else the link before)."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = LINK_MASS
    prev: str = "pencil"
    first: bool = False


@dataclass(frozen=True)
class Clip(_Part):
    """The cord clip, fixed; it holds the end of link ``last``."""
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 0.05
    last: str = "link"


@dataclass(frozen=True)
class Holster(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 0.05


@dataclass(frozen=True)
class Tray(_Part):
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    mass: float | None = 1.0


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
            if patch:                                # a rubber pad meets the part over a patch, which resists twisting
                cp.CreateTorsionalPatchRadiusAttr(float(patch))
                cp.CreateMinTorsionalPatchRadiusAttr(float(patch))
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def box(self, path, lo, hi, look, mat=None, offset=0.0008):
        """A box with its edges along the part's axes: a box collider (thin plates cannot be cooked as convex hulls
        for the GPU pipeline, and then collide with nothing)."""
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
        if mat:
            UsdPhysics.CollisionAPI.Apply(prim)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(offset)
            cp.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def capsule(self, path, length, radius, look, mat=None, offset=0.0008):
        """A round bar along x from the origin, with rounded ends."""
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from pxr import PhysxSchema, UsdPhysics

        cap = self.UsdGeom.Capsule.Define(self.stage, path)
        cap.CreateAxisAttr("X")
        cap.CreateHeightAttr(float(length))
        cap.CreateRadiusAttr(float(radius))
        cap.AddTranslateOp().Set(self.Gf.Vec3d(float(length) / 2, 0.0, 0.0))
        prim = cap.GetPrim()
        if look:
            bind_visual_material(str(prim.GetPath()), look, stage=self.stage)
        if mat:
            UsdPhysics.CollisionAPI.Apply(prim)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(offset)
            cp.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(prim.GetPath()), mat, stage=self.stage)
        return prim

    def pieces(self, stem, pieces, mat, rough=0.6, metal_of=(), patch=0.0, opacity=1.0):
        for n, (pts, color) in enumerate(pieces):
            metal = 0.9 if tuple(color) in metal_of else 0.0
            look = self.look(color, rough=0.35 if metal else rough, metal=metal, opacity=opacity)
            if is_box(pts) and not patch:
                self.box(f"{stem}_{n}", pts.min(0), pts.max(0), look, mat)
            else:
                self.hull(f"{stem}_{n}", pts, look, mat, patch=patch)


# -- spawning ----------------------------------------------------------------------------------------------------------

_SPECS: dict = {}
_PATHS: dict = {}                      # where each kit object's prim was spawned


def _start(cfg, prim_path, translation, orientation, iterations=None, fixed=False, lin_damp=0.05, ang_damp=0.1):
    spec = _SPECS[cfg.kit_name]
    _PATHS[spec.name] = prim_path
    a = _Author(prim_path, translation, orientation)
    a.body(prim_path, float(spec.mass), iterations=iterations, fixed=fixed, lin_damp=lin_damp, ang_damp=ang_damp)
    return spec, a, a.mat("part", *FRICTION)


def _spawn_pencil(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", pencil_pieces(), m, rough=0.5, metal_of=(STEEL,), patch=PAD_PATCH)
    return a.root


def _spawn_key(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32)
    a.pieces(f"{prim_path}/p", key_pieces(spec.which), m, rough=0.5)
    return a.root


def _spawn_link(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, iterations=32, lin_damp=0.4, ang_damp=3.0)
    a.capsule(f"{prim_path}/p_0", PITCH, CORD_R, a.look(CORD, rough=0.7), m)     # round, so that it lies quietly
    return a.root


def _spawn_clip(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True)
    a.pieces(f"{prim_path}/p", clip_pieces(), m)
    run = getattr(spec, "size", (0.2, 0.0, 0.0))[0]                    # the cord beyond the clip, drawn only
    a.hull(f"{prim_path}/tail", _rod((0.0, 0.0, 0.0), (run, 0.0, 0.0), CORD_R), a.look(CORD, rough=0.7))
    return a.root


def _spawn_holster(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True)
    pieces = holster_pieces()
    a.pieces(f"{prim_path}/p", pieces[:-1], m, rough=0.5)
    a.pieces(f"{prim_path}/s", pieces[-1:], m, rough=0.8)
    return a.root


def _spawn_tray(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    spec, a, m = _start(cfg, prim_path, translation, orientation, fixed=True)
    a.pieces(f"{prim_path}/p", tray_pieces(), m, rough=0.7, metal_of=(STEEL,))
    return a.root


# -- joints ------------------------------------------------------------------------------------------------------------

def _joints() -> list:
    """Every joint the parts in the scene need: (name, kind, first body, second body, anchor in the first body,
    anchor in the second body). Kinds: "slide" a key's sprung slide, "ball" a ball joint with a bend limit, "pivot"
    a ball joint without one."""
    out = []
    for n, s in _SPECS.items():
        if n not in _PATHS:
            continue
        if isinstance(s, Key) and s.pencil in _PATHS:
            out.append((f"{n}_slide", "slide", s.pencil, n, tuple(key_rest(s.which)), (0.0, 0.0, 0.0)))
        elif isinstance(s, Link) and s.prev in _PATHS:
            out.append((f"{n}_ball", "ball", s.prev, n, (HANDLE if s.first else PITCH, 0.0, 0.0), (0.0, 0.0, 0.0)))
        elif isinstance(s, Clip) and s.last in _PATHS:
            out.append((f"{n}_ball", "pivot", s.last, n, (PITCH, 0.0, 0.0), (0.0, 0.0, 0.0)))
    return out


def _joint_path(joint) -> str:
    return f"{_PATHS[joint[2]]}/{joint[0]}"


def _author(joint) -> None:
    """Author one joint: disabled first, its frames set, enabled last."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    name, kind, body0, body1, local0, local1 = joint
    stage = get_current_stage()
    cls = UsdPhysics.PrismaticJoint if kind == "slide" else UsdPhysics.SphericalJoint
    j = cls.Define(stage, _joint_path(joint))
    j.CreateJointEnabledAttr().Set(False)
    j.CreateBody0Rel().SetTargets([_PATHS[body0]])
    j.CreateBody1Rel().SetTargets([_PATHS[body1]])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in local0)))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*(float(v) for v in local1)))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    if kind == "slide":
        # The key slides along the pencil's z between its rest (0) and KEY_TRAVEL pushed in. A spring pushes it out
        # against the rest stop: KEY_PRELOAD there, KEY_FORCE_ON where the switch closes.
        j.CreateAxisAttr().Set("Z")
        j.CreateLowerLimitAttr().Set(-KEY_TRAVEL)
        j.CreateUpperLimitAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(KEY_SPRING))
        drive.CreateDampingAttr().Set(3.0)
        drive.CreateTargetPositionAttr().Set(float(KEY_PRELOAD / KEY_SPRING))
    elif kind == "ball":                       # one link can bend this far from the next; the clip lets the cord pivot freely
        j.CreateAxisAttr().Set("X")
        j.CreateConeAngle0LimitAttr().Set(CORD_BEND)
        j.CreateConeAngle1LimitAttr().Set(CORD_BEND)
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


_SPAWNERS = {Pencil: _spawn_pencil, Key: _spawn_key, Link: _spawn_link, Clip: _spawn_clip, Holster: _spawn_holster,
             Tray: _spawn_tray}
_BOUNDS: dict = {}


def _points_of(spec) -> np.ndarray:
    if isinstance(spec, Pencil):
        return np.vstack([p for p, _ in pencil_pieces()])
    if isinstance(spec, Key):
        return key_pieces(spec.which)[0][0]
    if isinstance(spec, Link):
        return link_points()
    if isinstance(spec, Clip):
        return np.vstack([p for p, _ in clip_pieces()])
    if isinstance(spec, Holster):
        return np.vstack([p for p, _ in holster_pieces()])
    return np.vstack([p for p, _ in tray_pieces()])


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
    """Join the parts as the reset left them: the keys to the pencil, the cord's links in a chain to the clip."""
    for joint in _joints():
        _author(joint)


def _local_bounds(spec) -> tuple:
    key = (type(spec), getattr(spec, "which", ""))
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
    if getattr(scene, "_esukit_patched", False):
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
    scene._esukit_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_esukit_patched", False):
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
            print(f"[esukit] joints not re-armed around the reset: {err!r}", file=sys.stderr, flush=True)
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
    cls._esukit_patched = True


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
