"""On the station table (a kitchen worktop): a half-size, 4 in deep steel food pan holding one US gallon of water for
sanitizing, which may or may not already have its sanitizer tablet dissolved in it (one tablet per gallon makes
200 ppm). To its right a fresh sanitizer test strip leans in a small rest, pad end down, and an open waste cup
stands beyond it. To its left two sanitizer tablets lie on a small board. A printed colour chart (0, 100, 200,
400 ppm) lies in front of the pan.

The job is the check a kitchen makes before it uses a pan of sanitizer: dip the strip's pad in the solution for a
second or two, read its colour, and only then decide. A pan that reads 0 gets one tablet; a pan that already reads 200
gets none (the tablets are labelled one to the gallon, and a second would make the pan twice its labelled strength).
The used strip goes in the waste cup.

Which pan it is changes from trial to trial (``SANI_LOT=A``: no sanitizer yet; ``SANI_LOT=B``: already made up). The
two pans look the same: a dissolved tablet does not colour the water, so nothing in the scene shows which it is
until a strip has been dipped.

Run it:    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:      python demo.py task.py out_dir
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import sanikit as K  # noqa: E402
from sanikit import Chart, Cup, Fixed, Pan, Strip, on_reset, on_step  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Test the pan of sanitizer with the test strip: dip the strip's pad in the solution for one to two seconds "
               "and compare its colour with the chart. If it reads below 200 ppm, put one sanitizer tablet in the pan; "
               "if it reads 200 ppm, add none. Then drop the used strip in the waste cup.")

# -- the solution ----------------------------------------------------------------------------------------------------
LOT = os.environ.get("SANI_LOT", "A")
START_PPM = {"A": 0, "B": 200}[LOT]      # A: plain water, no tablet yet; B: already made up
TABLET_PPM = 200                         # one tablet in one gallon
TARGET_PPM = 200
GALLON = 0.003785411784                  # m3
LEVELS = (0, 100, 200, 400)              # the readings printed on the chart
# what the pad (and the chart's swatch) looks like at each reading; a dry pad looks like a pad that read 0
PAD_COLORS = {0: (0.96, 0.90, 0.55), 100: (0.80, 0.84, 0.42), 200: (0.42, 0.66, 0.34), 400: (0.10, 0.36, 0.36)}

# -- layout (m; x forward from the arms, y left, heights above the worktop) ---------------------------------------------
PAN_XY = np.array([0.365, -0.065])       # the middle of the pan
PAN_IN = (0.240, 0.300)                  # inside, front to back (x) and across (y)
PAN_H, PAN_WALL = 0.100, 0.002
DEPTH = GALLON / (PAN_IN[0] * PAN_IN[1])  # one gallon stands 5.3 cm deep
SURFACE = PAN_WALL + DEPTH               # the liquid's surface, above the worktop

SLANT = math.radians(60.0)               # the strip leans in its rest this steeply, pad end down and toward the pan
REST_FOOT = np.array([0.210, -0.290, 0.012])                     # where the strip's pad end rests
UP_STRIP = np.array([0.0, -math.cos(SLANT), math.sin(SLANT)])    # from its pad end along it to its handle end
STRIP_QUAT = K.turn_x(math.pi / 2 - SLANT)

CUP_XY = np.array([0.360, -0.330])
CUP_IN, CUP_H = 0.100, 0.080             # inside diameter, height

BOARD_XY = np.array([0.260, 0.220])
BOARD = (0.150, 0.100, 0.006)
TABLET_R, TABLET_H, TABLET_MASS = 0.0065, 0.005, 0.0015
TABLET_XY = {"tablet_a": (0.230, 0.220), "tablet_b": (0.290, 0.220)}

CHART_XY = np.array([0.165, -0.060])
CHART = (0.050, 0.112, 0.0005)

STRIP, CUP, PAN = "strip", "waste_cup", "pan"
TABLETS = tuple(TABLET_XY)

# -- scoring -----------------------------------------------------------------------------------------------------------
DIP_S = 1.0                              # the pad shows a reading once all of it has been under for this long (the
                                         # strips' directions: dip for 1 to 2 seconds, compare within 10 seconds)
PAD_UNDER = 0.001                        # "under": every corner of the pad at least this far below the surface
HANDLE_UP = 0.020                        # a dip, not a drop: the strip's handle end stays this far above the surface
STILL = 0.02                             # m/s: the strip, the cup and the tablets are at rest below this
CLEAR = 0.030                            # both grippers this far from the strip, and out of the pan, at the end

STEEL_GREY, BOARD_WHITE, REST_GREY = (0.76, 0.77, 0.79), (0.93, 0.93, 0.90), (0.36, 0.38, 0.42)
TABLET_BLUE = (0.25, 0.47, 0.80)
CLEAR_WATER = (0.97, 0.99, 1.0)           # plain water and made-up sanitizer look the same


def strip_start() -> np.ndarray:
    """The strip's pad end at the start, in its rest."""
    return REST_FOOT + [0.0, 0.0, TOP] + 0.0003 * UP_STRIP


def reading(ppm: float) -> int:
    """The chart level a pad shows for a solution of this strength."""
    return min(LEVELS, key=lambda level: (abs(level - ppm), level)) if ppm < LEVELS[-1] else LEVELS[-1]


def in_pan(p, margin: float = 0.0) -> bool:
    """Over the inside of the pan (seen from above)."""
    return bool(abs(p[0] - PAN_XY[0]) < PAN_IN[0] / 2 - margin and abs(p[1] - PAN_XY[1]) < PAN_IN[1] / 2 - margin)


def under(p, by: float = 0.0) -> bool:
    """In the liquid: inside the pan and at least ``by`` below its surface."""
    return in_pan(p) and float(p[2]) - TOP < SURFACE - by


def pad_wet(pos, quat) -> bool:
    """All of the strip's pad is under the surface."""
    return all(under(c, PAD_UNDER) for c in K.strip_points(pos, quat)["corners"])


def handle_up(pos, quat) -> bool:
    """The strip's handle end is well above the surface (the strip is being held in, not lying in the pan)."""
    return float(K.strip_points(pos, quat)["top"][2]) - TOP >= SURFACE + HANDLE_UP


def in_cup(mid, cup_pos, cup_quat) -> bool:
    """The strip's middle is inside the upright waste cup."""
    axis = K._rotation(cup_quat)[:, 2]
    v = np.asarray(mid, float) - np.asarray(cup_pos, float)
    h = float(v @ axis)
    return bool(axis[2] > math.cos(math.radians(30.0)) and 0.0 < h < CUP_H and np.linalg.norm(v - h * axis) < CUP_IN / 2)


def _rest_part(name: str, local, size, color=REST_GREY) -> Fixed:
    """A piece of the strip's rest, given in the resting strip's own frame (x its width, y out of its pad face, z
    along it from its pad end)."""
    at = REST_FOOT + [0.0, 0.0, TOP] + K._rotation(STRIP_QUAT) @ np.asarray(local, float)
    return Fixed(name, pos=tuple(float(v) for v in at), size=tuple(size), color=color, quat=STRIP_QUAT,
                 static_friction=0.5, dynamic_friction=0.4)


def _objects() -> list:
    pan = Pan(PAN, pos=(float(PAN_XY[0]), float(PAN_XY[1]), TOP), size=(PAN_IN[0], PAN_IN[1], PAN_H), wall=PAN_WALL,
              depth=DEPTH, tint=CLEAR_WATER, color=STEEL_GREY)
    back = -K.STRIP_T / 2 - 0.003                                     # the middle of the 6 mm plate the strip lies on
    half_w = K.STRIP_W / 2 + 0.0007                                   # the groove: the strip's width and 1.4 mm
    rest = [_rest_part("rest_ramp", (0.0, back, 0.013), (0.016, 0.006, 0.026)),        # under the strip's lower part
            _rest_part("rest_rib", (0.0, back, 0.034), (0.004, 0.006, 0.016)),         # a rib narrower than the strip
            _rest_part("rest_wall_a", (half_w + 0.0015, 0.0005, 0.012), (0.003, 0.008, 0.024)),
            _rest_part("rest_wall_b", (-half_w - 0.0015, 0.0005, 0.012), (0.003, 0.008, 0.024)),
            _rest_part("rest_stop", (0.0, -0.0015, -0.002), (0.022, 0.014, 0.004)),
            _rest_part("rest_brace", (0.0, back - 0.003 - 0.012, 0.018), (0.022, 0.024, 0.016)),
            Fixed("rest_post", pos=(REST_FOOT[0], REST_FOOT[1] - 0.030, TOP + 0.009), size=(0.022, 0.020, 0.018),
                  color=REST_GREY),
            Fixed("rest_foot", pos=(REST_FOOT[0], REST_FOOT[1] - 0.018, TOP + 0.002), size=(0.050, 0.075, 0.004),
                  color=REST_GREY)]
    strip = Strip(STRIP, pos=tuple(float(v) for v in strip_start()), quat=STRIP_QUAT,
                  size=(K.STRIP_W, K.STRIP_T, K.STRIP_L), pad=PAD_COLORS[0])
    cup = Cup(CUP, pos=(float(CUP_XY[0]), float(CUP_XY[1]), TOP + 0.0005), size=(CUP_IN, CUP_IN, CUP_H),
              color=(0.90, 0.91, 0.92))
    board = Fixed("board", pos=(float(BOARD_XY[0]), float(BOARD_XY[1]), TOP + BOARD[2] / 2), size=BOARD, color=BOARD_WHITE,
                  static_friction=0.6, dynamic_friction=0.5)
    tablets = [Cylinder(n, pos=(x, y, TOP + BOARD[2] + TABLET_H / 2 + 0.0005), color=TABLET_BLUE, mass=TABLET_MASS,
                        radius=TABLET_R, height=TABLET_H) for n, (x, y) in TABLET_XY.items()]
    chart = Chart("chart", pos=(float(CHART_XY[0]), float(CHART_XY[1]), TOP + 0.0002), size=CHART, color=(0.97, 0.97, 0.95),
                  swatches=tuple((PAD_COLORS[level], str(level)) for level in LEVELS))
    return [pan, *rest, strip, cup, board, *tablets, chart]


OBJECTS = _objects()


# -- what happened during the trial ---------------------------------------------------------------------------------
_STATE: dict = {}


def _clear() -> None:
    _STATE.update(steps=0, pad_s=0.0, dip_s=0.0, tested=False, read=None, shown=None, blind=False, tablets=0,
                  ppm=START_PPM, in_cup=False, settled=False, wet_hand=False, strip=None)


_clear()


@on_reset
def _reset(environment) -> None:
    _clear()
    K.recolor(STRIP, "pad", PAD_COLORS[0])


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states   # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    dt = float(environment.unwrapped.step_dt)
    st = object_states(environment, [STRIP, CUP, *TABLETS])
    strip, cup = st[STRIP], st[CUP]
    pts = K.strip_points(strip.pos, strip.quat_xyzw)
    s["strip"] = pts["mid"]
    hands = [p for side in K.gripper_points(environment) for p in side]

    # tablets in the liquid; one put in before the pan has been tested is a blind dose
    s["tablets"] = sum(1 for n in TABLETS if under(st[n].pos))
    s["ppm"] = START_PPM + TABLET_PPM * s["tablets"]
    if s["tablets"] and not s["tested"]:
        s["blind"] = True

    # the pad: it takes up its colour once all of it has been under for DIP_S; that is a test only if the strip was
    # dipped by its handle with no part of a gripper in the liquid
    wet = pad_wet(strip.pos, strip.quat_xyzw)
    s["wet_hand"] = any(under(p) for p in hands)
    s["pad_s"] = s["pad_s"] + dt if wet else 0.0
    s["dip_s"] = s["dip_s"] + dt if (wet and handle_up(strip.pos, strip.quat_xyzw) and not s["wet_hand"]) else 0.0
    if s["pad_s"] >= DIP_S - 1e-9 and s["shown"] != reading(s["ppm"]):
        s["shown"] = reading(s["ppm"])
        K.recolor(STRIP, "pad", PAD_COLORS[s["shown"]])
    if s["dip_s"] >= DIP_S - 1e-9 and not s["tested"]:
        s["tested"], s["read"] = True, reading(s["ppm"])

    # the end state: the used strip in the waste cup, everything at rest, both grippers clear
    s["in_cup"] = in_cup(pts["mid"], cup.pos, cup.quat_xyzw)
    still = all(float(np.linalg.norm(st[n].lin_vel)) < STILL for n in (STRIP, CUP, *TABLETS))
    clear = all(K.segment_distance(p, pts["foot"], pts["top"]) >= CLEAR and not (in_pan(p) and p[2] - TOP < PAN_H)
                for p in hands)
    s["settled"] = bool(still and clear)


# -- scoring --------------------------------------------------------------------------------------------------------
def score_from(tested: bool, blind: bool, ppm: float, strip_in_cup: bool, settled: bool) -> float:
    """Pure scoring rule. ``tested``: the strip's pad was held fully under the liquid for DIP_S at some time, the
    strip dipped by its handle and no part of a gripper in the liquid. ``blind``: a tablet went into the liquid before
    that. ``ppm``: the strength the pan will have with the tablets now in it. ``strip_in_cup``: the strip is in the
    waste cup now. ``settled``: the strip, the cup and the tablets are at rest and both grippers are clear."""
    if not tested or blind or ppm > TARGET_PPM:
        return 0.0
    if ppm < TARGET_PPM:
        return 0.25
    if not strip_in_cup:
        return 0.5
    return 1.0 if settled else 0.75


def score(objects) -> float:
    """0 if the pan was never tested, if a tablet went in before it was tested, or if it ends above 200 ppm (a tablet
    in a pan that was already made up, or two tablets). Otherwise 0.25: tested (the pad fully under the liquid for 1 s,
    the strip held by its handle end at least 2 cm above the surface, no part of a gripper in the liquid). 0.5: and
    the pan ends at 200 ppm (its starting strength plus 200 ppm for each tablet in the liquid). 0.75: and the used
    strip is inside the upright waste cup. 1.0: and the strip, the cup and the tablets are at rest (under 2 cm/s) with
    both grippers at least 3 cm from the strip and out of the pan."""
    s = _STATE
    return score_from(s["tested"], s["blind"], s["ppm"], s["in_cup"], s["settled"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def report(objects) -> dict:
    s = _STATE
    mid = s["strip"]
    return {"score": score(objects), "lot": LOT, "tested": s["tested"], "read_ppm": s["read"], "pad_shows": s["shown"],
            "dip_s": round(s["dip_s"], 2), "pad_s": round(s["pad_s"], 2), "wet_hand": s["wet_hand"], "blind": s["blind"],
            "tablets_in": s["tablets"], "ppm": s["ppm"], "strip_in_cup": s["in_cup"], "settled": s["settled"],
            "strip_mid_mm": None if mid is None else [round(1000 * float(v), 1) for v in mid - [0.0, 0.0, TOP]]}


def demo(objects=None, seed=None):
    return {}
