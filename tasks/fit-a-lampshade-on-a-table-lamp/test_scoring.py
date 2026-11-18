"""Offline checks of the lampshade task: the shapes, the layouts, the demo's plan and the scoring rule. No simulator.

    python -m pytest -q test_scoring.py
"""
from __future__ import annotations

import importlib
import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import shadekit as K  # noqa: E402

IN = 0.0254


def load(lot: str):
    os.environ["SHADE_LOT"] = lot
    from yam_arena import task_file

    for name in ("task", "demo"):
        sys.modules.pop(name, None)
    return task_file.load(str(HERE / "task.py")).module


@pytest.fixture(params=["A", "B"])
def T(request):
    return load(request.param)


IDENT = (0.0, 0.0, 0.0, 1.0)


def seated_pose(lamp_xy=(0.0, 0.0), up=0.0, yaw=0.0):
    """Lamp at the origin of the table and the shade this far above its seat."""
    lamp = np.array([lamp_xy[0], lamp_xy[1], 0.0])
    return np.array([lamp_xy[0], lamp_xy[1], K.TOP_SEATED - K.HS + up]), K.turn_z(yaw), lamp, IDENT


# -- the shade -----------------------------------------------------------------------------------------------------------

def test_shade_is_the_listed_size():
    assert K.TOP_AF == pytest.approx(3 * IN) and K.BOT_AF == pytest.approx(5 * IN) and K.HS == pytest.approx(4.25 * IN)
    assert math.degrees(K.SLOPE) == pytest.approx(13.24, abs=0.05)


def test_panels_close_the_hexagon():
    tops = [K.panel_frame(k) for k in range(6)]
    for (top, t, d, n), (top2, t2, d2, n2) in zip(tops, tops[1:] + tops[:1]):
        corner = top + t * K.panel_width(0.0) / 2            # the end of one panel's top edge ...
        corner2 = top2 - t2 * K.panel_width(0.0) / 2         # ... is the start of the next one's
        assert np.linalg.norm(corner - corner2) < 1e-9
        assert abs(t @ d) < 1e-12 and abs(t @ n) < 1e-12 and abs(d @ n) < 1e-12
    slant = K.HS / math.cos(K.SLOPE)
    assert K.panel_width(slant) == pytest.approx(K.BOT_AF / math.sqrt(3))


def test_panel_bars_cover_each_panel_from_top_to_bottom():
    boxes = K.shade_boxes()
    assert len(boxes) == 6 * K.STRIPS
    zs = np.array([c[2] for c, _, _ in boxes[:K.STRIPS]])
    assert zs.max() < K.HS and zs.min() > 0.0 and np.all(np.diff(zs) < 0)
    low = boxes[K.STRIPS - 1]
    assert (low[0] - low[1][:, 1] * low[2][1] / 2)[2] == pytest.approx(0.0, abs=1e-9)   # the last bar ends on the bottom plane
    for _, rot, _ in boxes:
        assert np.linalg.det(rot) == pytest.approx(1.0)                         # a proper rotation, as the scene needs


def test_spokes_run_down_to_the_ring():
    for a, b, r in K.shade_wires()[:3]:
        assert np.asarray(b)[2] == pytest.approx(K.HUB_Z) and np.asarray(a)[2] > K.HUB_Z


def test_shade_distance_reads_inside_outside_and_wall():
    top, t, d, n = K.panel_frame(0)
    on_wall = top + d * 0.03
    assert K.shade_distance([on_wall], (0, 0, 0), IDENT) < 0
    assert K.shade_distance([on_wall + n * 0.011], (0, 0, 0), IDENT) == pytest.approx(0.010, abs=1e-6)
    assert K.shade_distance([(0.0, 0.0, 0.05)], (0, 0, 0), IDENT) > 0.04       # on the axis, inside


# -- the lamp and its bulb -----------------------------------------------------------------------------------------------

def test_lamp_is_eleven_inches_with_its_shade_on():
    assert K.TOP_SEATED == pytest.approx(11 * IN, abs=0.0005)
    assert K.TOP_SEATED - K.HS == pytest.approx(6.75 * IN, abs=0.0005)          # the shade's bottom above the table
    assert 2 * K.BASE_R == pytest.approx(3.25 * IN)


def test_bulb_outline_is_a_b10_and_convex():
    s = np.array([p[0] for p in K.BULB_PROFILE])
    r = np.array([p[1] for p in K.BULB_PROFILE])
    assert 2 * r.max() == pytest.approx(1.26 * IN, abs=0.0005)                  # the data sheet's diameter
    assert K.BULB_H + 0.0175 == pytest.approx(3.72 * IN, abs=0.001)             # ... and height, with its screw base in the cup
    slopes = np.diff(r) / np.diff(s)
    assert np.all(np.diff(slopes) < 0)                                          # convex all the way down
    assert len(K.bulb_points()) <= 64                                           # one convex piece for the simulator


def test_ring_stops_on_the_bulbs_tip():
    assert 0.003 < K.SEAT < 0.008
    assert K.bulb_radius(K.SEAT) == pytest.approx(K.RING_R - K.WIRE, abs=1e-6)
    assert K.bulb_radius(0.050) > K.RING_R                                      # the ring cannot pass the bulb


def test_loops_have_to_be_sprung_by_the_bulb():
    """At the shade's seat, the loops' lower ends hang where the bulb is wider than the gap between them at rest."""
    low = K.SEAT + K.JAW_L * math.cos(K.JAW_REST)
    rest_r = K.RING_R + K.JAW_L * math.sin(K.JAW_REST)
    need = K.bulb_radius(low) + K.JAW_WIRE
    assert need > rest_r + 0.004
    opened = math.asin((need - K.RING_R) / K.JAW_L) - K.JAW_REST
    assert 0.1 < opened < K.JAW_OPEN                                            # within the hinge's travel
    grip = (K.JAW_PRELOAD + K.JAW_STIFF * opened) / K.JAW_L                     # each loop's force on the glass, N
    assert 0.3 < grip < 2.0


def test_jaw_angle_reads_the_spring_travel():
    for which in (0, 1):
        hinge, rot = K.jaw_rest(which)
        assert K.jaw_angle(which, IDENT, K._quat(rot)) == pytest.approx(0.0, abs=1e-9)
        out = rot @ K._ry(-0.2)                                                 # 0.2 rad further out
        assert K.jaw_angle(which, IDENT, K._quat(out)) == pytest.approx(0.2, abs=1e-9)
        low = np.asarray(hinge) + out @ np.array([0.0, 0.0, -K.JAW_L])
        rest_low = np.asarray(hinge) + rot @ np.array([0.0, 0.0, -K.JAW_L])
        assert math.hypot(low[0], low[1]) > math.hypot(rest_low[0], rest_low[1])   # outward is away from the axis


def test_lamp_goes_over_under_a_small_push_at_the_bulb():
    """What steadying the lamp is for: the sideways push at the bulb's widest point that starts the lamp tipping."""
    push = K.LAMP_MASS * 9.81 * K.BASE_R / (K.TIP_Z - 0.050)
    assert 1.0 < push < 2.5


# -- where the shade is on the lamp ----------------------------------------------------------------------------------------

def test_seat_reads_zero_when_home():
    s = K.seat(*seated_pose())
    assert s["up"] == pytest.approx(0.0, abs=1e-9) and s["off"] == pytest.approx(0.0, abs=1e-9)
    assert s["tilt"] == pytest.approx(0.0, abs=1e-6) and s["over"]


def test_seat_reads_height_offset_and_turn():
    s = K.seat(*seated_pose(up=0.020, yaw=1.0))
    assert s["up"] == pytest.approx(0.020) and s["tilt"] == pytest.approx(0.0, abs=1e-6) and s["over"]
    sp, sq, lp, lq = seated_pose()
    s = K.seat(sp + [0.006, 0.0, 0.0], sq, lp, lq)
    assert s["off"] == pytest.approx(0.006)


def test_shade_beside_the_lamp_or_upside_down_is_not_over_it():
    sp, sq, lp, lq = seated_pose()
    assert not K.seat(sp + [0.20, 0.0, -0.17], sq, lp, lq)["over"]
    flip = (1.0, 0.0, 0.0, 0.0)                                                 # half a turn about x: small end down
    for drop in (0.0, 0.01, 0.03):
        hub_at = np.array([0.0, 0.0, K.TIP_Z - drop])                           # its ring this far down over the tip
        s = K.seat(hub_at + [0.0, 0.0, K.HUB_Z], flip, lp, lq)
        assert s["tilt"] == pytest.approx(180.0) and not s["over"]


def test_tilt_is_measured_against_the_lamp(T):
    sp, sq, lp, lq = seated_pose()
    lean = K._quat(K._ry(math.radians(8.0)))
    hub = K.place(sp, sq, (0, 0, K.HUB_Z))
    crooked = hub - K._rotation(lean) @ np.array([0, 0, K.HUB_Z])               # turned about its ring
    f = T.fitted(crooked, lean, lp, lq)
    assert f["tilt"] == pytest.approx(8.0, abs=1e-6) and f["seated"]


# -- the scoring rule ------------------------------------------------------------------------------------------------------

def test_full_score_needs_everything(T):
    full = dict(fell=False, over=True, on=True, seated=True, tilt=2.0, lamp_ok=True, released=True, still=True)
    assert T.score_from(**full) == 1.0
    assert T.score_from(**{**full, "fell": True}) == 0.0
    assert T.score_from(**{**full, "tilt": 8.0}) == 0.75
    assert T.score_from(**{**full, "lamp_ok": False}) == 0.75
    assert T.score_from(**{**full, "tilt": 20.0}) == 0.5
    assert T.score_from(**{**full, "seated": False}) == 0.5
    assert T.score_from(**{**full, "released": False}) == 0.25
    assert T.score_from(**{**full, "still": False}) == 0.25
    assert T.score_from(**{**full, "seated": False, "on": False}) == 0.25
    assert T.score_from(**{**full, "seated": False, "on": False, "over": False}) == 0.0


def test_every_outcome_has_a_score(T):
    seen = set()
    for bits in range(128):
        fell, over, on, seated, lamp_ok, released, still = (bool(bits >> i & 1) for i in range(7))
        for tilt in (0.0, 8.0, 20.0):
            v = T.score_from(fell, over, on, seated, tilt, lamp_ok, released, still)
            seen.add(v)
            if fell:
                assert v == 0.0
            if not released or not still:
                assert v <= 0.25
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_thresholds(T):
    lp, lq = T.lamp_start()
    sp = np.array([lp[0], lp[1], lp[2] + K.TOP_SEATED - K.HS])
    f = lambda **kw: T.fitted(sp + kw.get("d", np.zeros(3)), IDENT, lp, lq)   # noqa: E731
    assert f()["seated"] and f()["on"] and f()["over"]
    assert f(d=np.array([0, 0, 0.0039]))["seated"] and not f(d=np.array([0, 0, 0.0041]))["seated"]
    assert f(d=np.array([0.0039, 0, 0]))["seated"] and not f(d=np.array([0.0041, 0, 0]))["seated"]
    assert f(d=np.array([0, 0, 0.044]))["on"] and not f(d=np.array([0, 0, 0.046]))["on"]
    assert f(d=np.array([0.009, 0, 0.02]))["on"] and not f(d=np.array([0.011, 0, 0.02]))["on"]
    assert f(d=np.array([0, 0, 0.05]))["over"] and not f(d=np.array([0, 0, K.HUB_Z + 0.02]))["over"]


def test_lamp_facts(T):
    lp, lq = T.lamp_start()
    assert T.lamp_pose_facts(lp, lq) == {"lean": pytest.approx(0.0, abs=1e-6), "moved": pytest.approx(0.0)}
    lean = K._quat(K._ry(math.radians(25.0)))
    assert T.lamp_pose_facts(lp + [0.04, 0.0, 0.0], lean)["lean"] == pytest.approx(25.0)
    assert T.lamp_pose_facts(lp + [0.04, 0.0, 0.0], lq)["moved"] == pytest.approx(0.04)
    assert T.UPRIGHT < T.FELL and T.LEVEL < T.CROOKED < T.ON_TILT


# -- the layouts and the demo's plan ---------------------------------------------------------------------------------------

def test_layout(T):
    names = [o.name for o in T.OBJECTS]
    assert names == [T.LAMP, T.SHADE, *T.JAWS] and all(n.isidentifier() for n in names)
    lamp, shade = np.array(T.LAMP_XY), np.array(T.SHADE_XY)
    apart = np.linalg.norm(lamp - shade)
    assert apart > K.BASE_R + K.BOT_AF / math.sqrt(3) + 0.05                    # the shade stands clear of the lamp
    sp, sq = T.shade_start()
    for which in (0, 1):                                                        # each loop starts on its hinge, at rest
        jp, jq = T.jaw_start(which)
        hinge, rot = K.jaw_rest(which)
        assert np.allclose(jp, K.place(sp, sq, hinge)) and K.jaw_angle(which, sq, jq) == pytest.approx(0.0, abs=1e-9)
    assert T.fitted(sp, sq, *T.lamp_start())["over"] is False


def test_layouts_differ():
    a, b = load("A"), load("B")
    assert a.LAMP_XY != b.LAMP_XY and a.SHADE_XY != b.SHADE_XY and a.SHADE_YAW != b.SHADE_YAW


def test_demo_plan(T):
    import demo as D

    importlib.reload(D)
    p = D.plan(T)
    rot_s, at_s = D.shade_hold(K)
    assert abs(rot_s[1, 0]) == pytest.approx(1.0)                               # the pads close across the shade, panel 0 to panel 3
    assert math.degrees(math.acos(-rot_s[2, 2])) == pytest.approx(D.LEAN)       # the hand leans this far from straight down
    assert rot_s[0, 2] > 0 and abs(rot_s[1, 2]) < 1e-12                         # its fingers point along the shade's x and down
    below = 0.0125 - D.RISE                                                     # the fingertips' lower edges, below the shade's top
    assert 0.005 < below < 0.020
    width = D.span_at(K, below)                                                 # the shade there: wider than the closed hand,
    assert D.SPAN_GRIP < width < D.SPAN_OPEN - 0.006 and D.SPAN_OPEN <= D.FULL_OPEN   # narrower than the open one
    assert width == pytest.approx(0.0832, abs=0.0005)
    assert abs(at_s[1]) < 1e-12 and at_s[2] == pytest.approx(K.HS + D.RISE)
    bottom = p["carry_z"] - p["offset"][2] - T.TOP
    assert bottom == pytest.approx(K.TIP_Z + D.OVER_TIP)                        # the shade clears the bulb's tip on the way over
    assert p["seat_z"] - p["offset"][2] - T.TOP == pytest.approx(K.TOP_SEATED - K.HS)
    assert K.BASE_H + K.FOOT_H < D.STEM_AT - 0.014 and D.STEM_AT + 0.014 < K.PAN_Z   # the left hand's fingers fit on the stem
    assert D.STEM_GRIP < 2 * K.STEM_R < D.STEM_OPEN
    assert D.STEM_AT + 0.07 < K.TOP_SEATED - K.HS                               # ... and stay under the shade


def test_task_json_fits_the_boards_limits():
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    j = json.loads(path.read_text())
    assert len(j["title"]) <= 120 and len(j["fidelity"]) <= 500 and len(j.get("context", "")) <= 1000 and len(j["scores"]) <= 300
    assert j["category"] == "Societal relatability"
    assert j["instruction"] == load("A").INSTRUCTION
    for words in ("4 mm", "5 degrees", "12 degrees", "3 cm", "20 degrees", "4.5 cm"):
        assert words in j["scoring"], words
