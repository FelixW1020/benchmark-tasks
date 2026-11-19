"""Offline checks of the fan task: the parts' shapes and fits, the layouts, the demo's plan and the scoring rule.

    python -m pytest -q test_scoring.py

No simulator is needed: ``fankit`` and ``task`` reach Isaac Sim only when a scene is built.
"""
from __future__ import annotations

import importlib
import math
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fankit as K  # noqa: E402


def load(lot: str):
    os.environ["FAN_LOT"] = lot
    import task

    return importlib.reload(task)


def inside(point, piece, slack: float = 0.0) -> bool:
    centre, rot, size = piece
    q = np.asarray(rot).T @ (np.asarray(point, float) - np.asarray(centre, float))
    return bool(np.all(np.abs(q) <= np.asarray(size) / 2 + slack))


def hits(points, pieces) -> int:
    return sum(1 for p in points for piece in pieces if inside(p, piece))


def circle(radius: float, z: float, n: int = 72) -> list:
    return [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def pin_points(yaw_deg: float, z: float) -> list:
    """Points along the cross pin's top edge when the pin lies ``yaw_deg`` from the hub's slot and its top is ``z``
    above the hub's back face."""
    a = math.radians(yaw_deg)
    return [(t * math.cos(a), t * math.sin(a), z) for t in np.linspace(-K.PIN_L / 2, K.PIN_L / 2, 33)]


# -- sizes ---------------------------------------------------------------------------------------------------------------

def test_heights_follow_from_the_sizes():
    assert K.HUB_SEAT == pytest.approx(K.PIN_Z + K.PIN_D / 2 - K.SLOT_D)
    assert K.HUB_RIDING - K.HUB_SEAT == pytest.approx(K.SLOT_D)
    assert K.CAP_SEAT == pytest.approx(K.HUB_SEAT + K.HUB_H)
    assert K.SHAFT_TIP - K.CAP_SEAT == pytest.approx(K.THREAD_LEN)          # 6.5 mm of thread stands out of a seated hub
    assert K.SHAFT_TIP - K.THREADED < K.CAP_SEAT                              # the thread begins inside the hub


def test_a_hub_riding_on_the_pin_hides_the_thread():
    assert K.HUB_RIDING + K.HUB_H > K.SHAFT_TIP                               # the cap then has nothing to catch


def test_the_blade_turns_inside_the_grilles():
    assert K.BLADE_R < K.RIM_IN - 0.005
    mid = K.HUB_SEAT + K.HUB_H / 2
    for r0, r1, chord, pitch in K.BLADE_SEGS:
        half = chord / 2 * math.sin(math.radians(pitch)) + K.BLADE_T
        for r in (r0, r1):
            if r <= K.DISH[0][0]:
                continue
            wire = K.DISH[0][1] + (r - K.DISH[0][0]) / (K.DISH[1][0] - K.DISH[0][0]) * (K.DISH[1][1] - K.DISH[0][1])
            assert mid - half > wire + 0.002                                  # clear of the rear grille's wires
            (ra, za), (rb, zb), (rc, zc) = K.DOME
            rib = za + (ra - r) / (ra - rb) * (zb - za) if r >= rb else zb + (rb - r) / (rb - rc) * (zc - zb)
            assert mid + half < K.GRILLE_SEAT + rib - 0.003                   # and of the front grille's ribs
    assert K.CAP_SEAT + K.CAP_H < K.GRILLE_SEAT + K.BADGE_Z[0] - 0.003        # the cap is clear of the badge


def test_the_fan_is_the_listed_width():
    assert 2 * K.RIM_OUT == pytest.approx(13.75 * 0.0254, abs=0.001)
    assert 2 * K.BLADE_R == pytest.approx(12 * 0.0254, abs=0.006)


# -- the hub on the shaft and the pin ----------------------------------------------------------------------------------

def test_the_bore_is_open_and_fits_the_shaft():
    boxes = K.hub_boxes()
    for z in np.linspace(0.0005, K.HUB_H - 0.0005, 12):
        assert hits(circle(K.SHAFT_R, z), boxes) == 0                         # the 8 mm shaft passes
        assert hits(circle(K.BORE_R + 0.0006, z, 144), boxes) > 0 or z < K.SLOT_D      # and has a wall round it
    assert K.BORE_R - K.SHAFT_R == pytest.approx(0.0003)


def test_the_slot_takes_the_pin_only_when_it_lies_along_it():
    boxes = K.hub_boxes()
    for z in (0.0005, K.SLOT_D - 0.0005):
        for dy in (-K.PIN_D / 2, K.PIN_D / 2):
            pts = [(x, y + dy * math.cos(0.0), zz) for x, y, zz in pin_points(0.0, z)]
            assert hits(pts, boxes) == 0                                      # along the slot: free to its bottom
    assert hits(pin_points(0.0, K.SLOT_D + 0.0005), boxes) > 0                # the slot's roof stops it there
    for yaw in (20.0, 45.0, 60.0, 90.0):
        assert hits(pin_points(yaw, 0.0005), boxes) > 0                       # turned away: the pin meets the boss
    room = math.degrees(math.atan((K.SLOT_W - K.PIN_D) / K.PIN_L))
    assert 4.0 < room < 8.0                                                   # a few degrees of play, no more


def test_blade_fit_tells_seated_from_riding():
    fan_pos, fan_q = np.array([0.24, -0.10, 0.80]), (0.0, 0.0, 0.0, 1.0)
    rotor_q = K.turn_z(math.radians(20.0))
    seated = K.blade_fit(fan_pos + [0, 0, K.HUB_SEAT], K.turn_z(math.radians(200.0)), fan_pos, fan_q, rotor_q)
    assert seated["off"] < 1e-9 and seated["up"] == pytest.approx(0.0) and seated["slot"] == pytest.approx(0.0, abs=1e-6)
    riding = K.blade_fit(fan_pos + [0, 0, K.HUB_RIDING], K.turn_z(math.radians(80.0)), fan_pos, fan_q, rotor_q)
    assert riding["up"] == pytest.approx(K.SLOT_D) and riding["slot"] == pytest.approx(60.0)


def test_each_blade_is_four_plates_out_to_its_tip():
    plates = K.blade_boxes()
    assert len(plates) == 12
    tips = [np.hypot(*(c[:2])) + s[0] / 2 for c, _r, s in plates]
    assert max(tips) == pytest.approx(K.BLADE_R, abs=0.0015)
    corners = [math.hypot(np.hypot(*(c[:2])) + s[0] / 2, s[1] / 2) for c, _r, s in plates]
    assert max(corners) <= K.BLADE_R                                          # no corner of a plate is outside the blade's circle
    for k in range(3):
        _c, rot, _s = K.blade_plate(k, 1)
        assert rot[2, 1] > 0                                                  # the leading edge is the high one
        assert np.linalg.det(rot) == pytest.approx(1.0)


# -- the cap and its thread -----------------------------------------------------------------------------------------------

def test_the_cap_goes_over_the_threaded_end():
    boxes = K.cap_boxes()
    for z in np.linspace(0.0003, K.CAP_BORE_D - 0.0003, 8):
        assert hits(circle(K.THREAD_R, z), boxes) == 0
    assert hits([(0.0, 0.0, K.CAP_BORE_D + 0.001)], boxes) > 0                # the bore is blind
    assert K.CAP_BORE_D > K.THREAD_LEN                                        # the shaft's end does not bottom in it
    assert K.CAP_MOUTH_R > K.CAP_BORE_R > K.THREAD_R


def test_the_thread_catches_only_on_the_shaft():
    fan_pos, q = np.zeros(3), (0.0, 0.0, 0.0, 1.0)
    on = K.thread([0, 0, K.SHAFT_TIP - 0.004], q, fan_pos, q)
    assert on["catches"]
    assert not K.thread([0, 0, K.SHAFT_TIP - 0.001], q, fan_pos, q)["catches"]            # only just set on the end
    assert not K.thread([0, 0, K.SHAFT_TIP + 0.002], q, fan_pos, q)["catches"]            # held above the end
    assert not K.thread([0.004, 0, K.SHAFT_TIP - 0.004], q, fan_pos, q)["catches"]        # beside it
    tipped = K._quat(K._ry(math.radians(20.0)))
    assert not K.thread([0, 0, K.SHAFT_TIP - 0.004], tipped, fan_pos, q)["catches"]       # crooked


def test_the_thread_draws_the_cap_down_by_its_pitch_and_stops_tight():
    s0 = K.SHAFT_TIP - K.CATCH_IN
    assert K.thread_target(s0, 0.5) == pytest.approx(s0 - 0.5 * K.PITCH)
    assert K.thread_target(s0, -0.25) == pytest.approx(s0 + 0.25 * K.PITCH)               # clockwise backs it off
    tight = (s0 - (K.CAP_SEAT - K.SNUG)) / K.PITCH
    assert 1.0 < tight < 2.0                                                              # over one turn to tight
    assert K.thread_target(s0, tight + 3.0) == pytest.approx(K.CAP_SEAT - K.SNUG)         # it does not wind further


def test_thread_friction_carries_a_free_shaft_round():
    # at the demo's turning rate the thread's drag is far more than the motor's: an unheld blade goes round with the cap
    assert K.THREAD_DRAG * math.radians(70.0) > 3 * K.ROTOR_DRAG


# -- the grilles and the clips ----------------------------------------------------------------------------------------

def test_grille_fit_seated_and_not():
    fan_pos, q = np.array([0.24, -0.10, 0.80]), (0.0, 0.0, 0.0, 1.0)
    down = K.grille_fit(fan_pos + [0, 0, K.GRILLE_SEAT], q, fan_pos, q)
    assert down["off"] < 1e-9 and down["up"] == pytest.approx(0.0) and down["under"]
    short = K.grille_fit(fan_pos + [-0.016, 0, K.GRILLE_SEAT], q, fan_pos, q)
    assert not short["under"] and short["off"] == pytest.approx(0.016)
    turned = K.grille_fit(fan_pos + [0, 0, K.GRILLE_SEAT], K.turn_z(math.radians(40.0)), fan_pos, q)
    assert not turned["under"] and turned["turn"] == pytest.approx(40.0)
    tipped = K.grille_fit(fan_pos + [0, 0, K.GRILLE_SEAT + 0.008], K._quat(K._ry(math.radians(2.6))), fan_pos, q)
    assert tipped["up"] > 0.012


def test_the_support_takes_the_rim_with_room():
    assert K.SUPPORT_WALL[0] - K.RIM_OUT == pytest.approx(0.0015)
    assert K.SUPPORT_LIP[0] - (K.RIM_Z + K.GRILLE_T) == pytest.approx(0.004)
    assert K.SUPPORT_REACH < K.RIM_OUT - 0.004


def test_a_shut_clip_hooks_over_the_rim_and_an_open_one_clears_it():
    for k in range(3):
        hinge = K.clip_hinge(k)
        assert np.hypot(hinge[0], hinge[1]) == pytest.approx(K.CLIP_HINGE_R)
        for deg, over in ((0.0, True), (K.CLIP_OPEN, False)):
            rot = K.clip_rot(k, deg)
            tip = hinge + rot @ np.array([K.CLIP_HOOK[0][0], 0.0, K.CLIP_HOOK[0][1]])
            r = float(np.hypot(tip[0], tip[1]))
            if over:
                assert r < K.RIM_OUT - 0.006 and tip[2] > K.RIM_Z + K.GRILLE_T                # over the seated rim's top
                assert tip[2] - (K.RIM_Z + K.GRILLE_T) < 0.0015
            else:
                assert r > K.RIM_OUT + 0.004                                                  # the rim can come down past it
            assert K.clip_deg(k, (0.0, 0.0, 0.0, 1.0), K._quat(rot)) == pytest.approx(deg, abs=1e-6)


def test_the_clip_spring_is_over_centre():
    assert K.clip_spring_target(K.CLIP_OPEN) > K.CLIP_OPEN
    assert K.clip_spring_target(K.CLIP_MID + 1.0) > K.CLIP_OPEN
    assert K.clip_spring_target(K.CLIP_MID - 1.0) < 0.0
    hardest = K.CLIP_K * (K.CLIP_OPEN + K.CLIP_PRE - K.CLIP_MID)              # N m, just before it goes over
    assert hardest / 0.015 < 1.5                                              # under 1.5 N at the top of the stem


def test_clips_are_clear_of_the_support_and_spread_round_the_rim():
    az = sorted(K.CLIP_AZ)
    assert all(40.0 <= a <= 320.0 for a in az)
    assert min(b - a for a, b in zip(az, az[1:])) >= 90.0


# -- the layouts ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lot", ["A", "B"])
def test_layout_keeps_the_parts_apart(lot):
    T = load(lot)
    fan = np.array(T.LAYOUT["fan"])
    blade, cap, grille = (np.array(T.LAYOUT[k]) for k in ("blade", "cap", "grille"))
    assert np.linalg.norm(blade - fan) > K.BLADE_R + K.RIM_OUT + 0.01
    assert np.linalg.norm(grille - fan) > 2 * K.RIM_OUT + 0.02
    assert np.linalg.norm(grille - blade) > K.RIM_OUT + K.BLADE_R + 0.02
    assert np.linalg.norm(cap - fan) > K.RIM_OUT + K.CAP_R + 0.02
    base = fan + [K.BASE_X, 0.0]
    assert abs(grille[1] - base[1]) > K.RIM_OUT + K.BASE_R + 0.01 or abs(grille[0] - base[0]) > K.RIM_OUT + K.BASE_T + 0.01
    for xy, r in ((blade, K.BLADE_R), (grille, K.RIM_OUT), (cap, K.CAP_R)):
        assert abs(xy[1]) + r < 0.65 and 0.02 < xy[0] - r and xy[0] + r < 0.80          # on the table
    for base_y in (0.3, -0.3):
        assert np.linalg.norm(grille - [0.0, base_y]) > K.RIM_OUT + 0.06                 # clear of the arms' bases


@pytest.mark.parametrize("lot", ["A", "B"])
def test_layout_objects(lot):
    T = load(lot)
    names = [o.name for o in T.OBJECTS]
    assert names == ["fan_towel", "blade_towel", "rack", "fan", "rotor", "clip_a", "clip_b", "clip_c", "blade", "cap", "grille"]
    by = {o.name: o for o in T.OBJECTS}
    fp, fq = T.fan_pose()
    assert by["fan"].pos[2] == pytest.approx(T.TOP + T.FAN_TOWEL[2])
    for k, name in enumerate(T.CLIPS):
        assert K.clip_deg(k, fq, by[name].quat) == pytest.approx(K.CLIP_OPEN, abs=1e-6)  # the clips start open
    bp, bq = T.blade_start()
    low = min((K.place(bp, bq, c + r @ np.array([sx * s[0] / 2, sy * s[1] / 2, -s[2] / 2])))[2]
              for c, r, s in K.blade_boxes() for sx in (-1, 1) for sy in (-1, 1) if np.hypot(*c[:2]) < 0.10)
    assert low - (T.TOP + T.BLADE_TOWEL[2]) == pytest.approx(T.REST, abs=0.0015)        # its trailing edges rest on the towel
    gp, gq = T.grille_start()
    assert gp[2] == pytest.approx(T.TOP + T.RACK[2] + T.REST)
    start = K.blade_fit(bp, bq, fp, fq, by["rotor"].quat)
    assert start["off"] > 0.2


@pytest.mark.parametrize("lot", ["A", "B"])
def test_the_blade_goes_on_with_its_slot_along_the_pin(lot):
    T = load(lot)
    fp, fq = T.fan_pose()
    rotor_q = K.turn_z(math.radians(T.LAYOUT["pin"]))
    on = K.blade_fit(fp + [0, 0, K.HUB_SEAT], K.turn_z(math.radians(T.blade_yaw_on())), fp, fq, rotor_q)
    assert on["slot"] == pytest.approx(0.0, abs=1e-6) and T.blade_seated(on)
    off = K.blade_fit(fp + [0, 0, K.HUB_RIDING], K.turn_z(math.radians(T.blade_yaw_on() + 60.0)), fp, fq, rotor_q)
    assert off["slot"] == pytest.approx(60.0) and not T.blade_seated(off)


def test_the_two_layouts_differ():
    a, b = load("A").LAYOUT, load("B").LAYOUT
    assert a["pin"] != b["pin"] and a["turn"] != b["turn"] and a["grille_turn"] != b["grille_turn"]
    assert all(a[k] != b[k] for k in ("fan", "blade", "cap", "grille"))


# -- the scoring rule ------------------------------------------------------------------------------------------------

def test_score_ladder():
    T = load("A")
    f = T.score_from
    assert f(False, False, False, 0, True, True) == 0.0
    assert f(False, True, True, 3, True, True) == 0.0                        # nothing counts over a blade off its pin
    assert f(True, False, False, 0, True, True) == 0.25
    assert f(True, False, True, 3, True, True) == 0.25                       # a grille clipped on over a loose cap
    assert f(True, True, False, 0, True, True) == 0.5
    assert f(True, True, False, 3, True, True) == 0.5
    assert f(True, True, True, 0, True, True) == 0.75
    assert f(True, True, True, 2, True, True) == 0.75
    assert f(True, True, True, 3, False, True) == 0.75                       # still held
    assert f(True, True, True, 3, True, False) == 0.75                       # still moving
    assert f(True, True, True, 3, True, True) == 1.0


def test_blade_seated_bounds():
    T = load("A")
    ok = {"off": 0.001, "tilt": 2.0, "up": 0.0008, "slot": 3.0}
    assert T.blade_seated(ok)
    assert not T.blade_seated({**ok, "off": 0.002})
    assert not T.blade_seated({**ok, "tilt": 4.0})
    assert not T.blade_seated({**ok, "up": K.SLOT_D})                         # riding on top of the pin


def test_cap_tight_needs_the_thread_and_the_hub():
    T = load("A")
    fit = {"off": 0.0002, "tilt": 0.5, "up": 0.0004}
    floor = K.CAP_SEAT - K.SNUG
    assert T.cap_tight(fit, True, floor)
    assert not T.cap_tight(fit, False, floor)                                 # resting on the hub, not on the thread
    assert not T.cap_tight(fit, True, floor + 0.001)                          # not turned down far enough
    assert not T.cap_tight({**fit, "up": 0.002}, True, floor)                 # held off the hub


def test_grille_down_bounds():
    T = load("A")
    ok = {"off": 0.002, "tilt": 0.3, "up": 0.0015, "low": -0.0005, "turn": 2.0, "under": True}
    assert T.grille_down(ok)
    assert not T.grille_down({**ok, "under": False})
    assert not T.grille_down({**ok, "up": 0.006})
    assert not T.grille_down({**ok, "off": 0.006})
    assert not T.grille_down({**ok, "turn": 30.0})


def test_score_text_matches_the_numbers():
    T = load("A")
    text = T.score.__doc__
    for word in ("1.5 mm", "1 mm", "7 mm", "0.8 mm", "3 mm", "2 mm", "6 degrees", "3 cm", "half second", "3 degrees"):
        assert word in text
    assert T.ON_AXIS == 0.0015 and T.PIN_HOME == 0.001 and T.CAP_GAP == 0.0008 and T.GRILLE_OFF == 0.003
    assert T.GRILLE_UP == 0.002 and K.CLIP_SHUT == 6.0 and T.CLEAR == 0.03 and T.REST_S == 0.5 and T.REST_MOVE == 0.002 and T.REST_TURN == 3.0
    assert K.SLOT_D == 0.007


# -- the demo's plan -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lot", ["A", "B"])
def test_demo_plan(lot):
    T = load(lot)
    import demo as D

    importlib.reload(D)
    p = D.plan(T)
    fan, _ = T.fan_pose()
    b = p["blade"]
    turn1 = K._rz(math.radians(b["yaw1"]))
    held = turn1 @ b["hold"]
    assert 0.06 < float(np.hypot(held[0], held[1])) < 0.10                    # on a blade, clear of the cap's hand
    az = math.degrees(math.atan2(held[1], held[0])) % 360.0
    assert 45.0 <= az <= 135.0                                                # on the fan's left
    assert b["over"][2] - fan[2] > K.SHAFT_TIP + 0.01                         # carried in over the shaft's end
    assert b["seat"][2] - fan[2] == pytest.approx(K.HUB_SEAT - D.PRESS)
    turns = D.BITES * D.BITE / 360.0
    need = (D.CAP_ON + K.CAP_SEAT - K.SHAFT_TIP + K.THREAD_LEN - (K.CAP_SEAT - K.SHAFT_TIP + K.THREAD_LEN) + (K.SHAFT_TIP - D.CAP_ON) - (K.CAP_SEAT - K.SNUG)) / K.PITCH
    need = (K.SHAFT_TIP - K.CATCH_IN - (K.CAP_SEAT - K.SNUG)) / K.PITCH       # turns from where the thread catches to tight
    assert need + 0.2 < turns < need + 0.6                                    # enough bites to tighten, not many more
    assert D.CAP_ON > K.CATCH_IN + 0.001
    g = p["grille"]
    assert np.linalg.norm(g["hold"][:2]) == pytest.approx(K.RIM_IN + K.BAND_T + 0.014)   # the fingertips stop short of the band
    assert g["hand_in"][2] - fan[2] == pytest.approx(K.GRILLE_SEAT + D.HOLD_UP + K.GRILLE_T / 2)
    assert D.HOLD_UP > K.FINGER_THICK + 0.003                                 # room for the lower finger over the rear rim
    far = g["tipped"] @ np.array([K.RIM_OUT, 0.0, 0.0]) - g["tipped"] @ np.array([g["hold"][0], g["hold"][1], 0.0])
    assert far[2] == pytest.approx(-(D.HOLD_UP - D.FAR_UP), abs=0.0005)       # tipped: its bottom edge 14 mm lower
    assert K.GRILLE_T + D.FAR_UP < K.SUPPORT_LIP[0] - K.RIM_Z                 # ... and goes in under the lip
    assert K.RIM_OUT - D.SHORT < K.SUPPORT_REACH - 0.0015                     # it comes down clear of the lip's end
    assert K.RIM_IN - D.SHORT > K.BLADE_R + 0.0025                            # ... and with its rim outside the blades' tips
    for k, cl in enumerate(p["clips"]):
        hinge = fan + K.clip_hinge(k)
        out = lambda v: float((np.asarray(v) - hinge) @ cl["r"])              # noqa: E731
        open_out = K.CLIP_STEM[1][0] * math.cos(math.radians(58.0)) + K.CLIP_STEM[1][1] * math.sin(math.radians(58.0))
        assert out(cl["from"]) - D.TIP_HALF[1] > open_out + 0.003            # the fingers start clear of the open clip
        assert out(cl["scoop"][0]) - D.TIP_HALF[0] == pytest.approx(open_out - D.SCOOP_IN, abs=0.0005)   # then meet its end
        assert cl["scoop"][-1][2] - cl["scoop"][0][2] > 0.008                 # and scoop it in and up
        over = math.hypot(*K.CLIP_STEM[1]) * math.sin(math.radians(K.CLIP_MID + math.degrees(math.atan2(*K.CLIP_STEM[1]))))
        assert out(cl["scoop"][-1]) - D.TIP_HALF[0] < over - 0.002           # past where it goes over centre
        assert out(cl["to"]) - D.TIP_HALF[0] == pytest.approx(K.CLIP_STEM[1][0], abs=0.001)   # and end on the shut clip's stem


def test_rest_is_judged_from_places():
    T = load("A")
    still = [(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))]
    n = int(round(T.REST_S * T.CONTROL_HZ))
    assert not T.at_rest([still] * n)                                         # not watched for long enough yet
    assert T.at_rest([still] * (n + 1))
    moved = [(np.array([0.0, 0.003, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))]
    assert not T.at_rest([moved] + [still] * n)                               # 3 mm within the last half second
    assert T.at_rest([moved] + [still] * (n + 1))                             # ... and longer ago than that
    turned = [(np.zeros(3), np.array(K.turn_z(math.radians(4.0))))]
    assert not T.at_rest([turned] + [still] * n)                              # 4 degrees
    slow = [(np.zeros(3), np.array(K.turn_z(math.radians(2.0))))]
    assert T.at_rest([slow] + [still] * n)                                    # 2 degrees in half a second: a shaft creeping round
