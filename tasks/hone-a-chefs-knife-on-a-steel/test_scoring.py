"""Offline checks of the honing task's scoring (no simulator):  python -m pytest -q test_scoring.py"""
import json
import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yam_arena import task_file  # noqa: E402


@pytest.fixture(scope="module")
def T():
    return task_file.load(str(HERE / "task.py")).module


UP = np.array([0.0, 0.0, 1.0])
FAR = [(np.array([0.0, 0.0, 2.0]), np.array([[0.0, 0.0, 2.0]])), (np.array([0.0, 0.5, 2.0]), np.array([[0.0, 0.5, 2.0]]))]
REST = {"knife": (0.0, 0.0), "steel": (0.0, 0.0)}


def steel_up(T):
    """The steel upright with its tip on the pad."""
    K = T.K
    tip = np.array([T.PAD_XY[0], T.PAD_XY[1], T.TOP + T.PAD_TOP])
    return tip, K._quat(np.column_stack([UP, (0.342, -0.940, 0.0), np.cross(UP, (0.342, -0.940, 0.0))]))


def knife_on(T, face, s, z, out=0.0, alpha=17.0, phi=25.0):
    """The knife with the point of its edge ``s`` from the heel against the upright steel, ``z`` up the rod."""
    K = T.K
    d = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
    r = face * np.cross(UP, d)
    a = math.radians(alpha)
    e = -math.sin(a) * r - math.cos(a) * UP
    rot = np.column_stack([d, -e, np.cross(d, -e)])
    c = np.array([T.PAD_XY[0], T.PAD_XY[1], T.TOP + T.PAD_TOP]) + z * UP + (K.ROD_R + out) * r
    return c - rot @ np.array([s, float(K.edge_y(s)), 0.0]), K._quat(rot)


def stroke(T, s, face, s0=0.02, s1=0.185, z0=0.20, z1=0.07, alpha=17.0, grams=12.0, n=40, arms=None):
    """Feed one stroke, then four steps off the steel."""
    st = steel_up(T)
    for k in range(n + 1):
        f = k / n
        T.advance(s, knife_on(T, face, s0 + f * (s1 - s0), z0 + f * (z1 - z0), alpha=alpha), st, grams, arms or FAR, REST)
    for _ in range(4):
        T.advance(s, knife_on(T, face, s1, z1, out=0.03, alpha=alpha), st, 0.0, arms or FAR, REST)
    return s


# -- the rule ------------------------------------------------------------------------------------------------------------
def test_tiers(T):
    f = T.score_from
    assert f(False, 0, False, False, True, True, True) == 0.0
    assert f(False, 1, True, True, True, True, True) == 0.0
    assert f(False, 2, False, False, True, False, True) == 0.25
    assert f(False, 7, True, True, True, True, True) == 0.25
    assert f(False, 8, False, False, True, False, True) == 0.5
    assert f(False, 8, True, False, True, True, True) == 0.5
    assert f(False, 8, True, True, True, False, True) == 0.75
    assert f(False, 8, True, True, False, True, True) == 0.75
    assert f(False, 8, True, True, True, True, False) == 0.75
    assert f(False, 8, True, True, True, True, True) == 1.0
    assert f(True, 8, True, True, True, True, True) == 0.0


def test_start_scores_nothing(T):
    s = T.new_state()
    for _ in range(5):
        T.advance(s, T.knife_start(), T.steel_start(), 0.0, FAR, REST)
    assert s["counted"] == 0 and not s["void"] and not s["strokes"]
    assert s["knife_down"] and s["steel_down"] and s["edge_away"] and s["let_go"] and s["still"]
    assert T.score_from(False, 0, True, True, True, True, True) == 0.0


# -- where the edge is on the steel ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("face", [1, -1])
def test_touch_reads_place_angle_and_face(T, face):
    c = T.touch(knife_on(T, face, 0.05, 0.15), steel_up(T))
    assert abs(c["gap"]) < 2e-4 and abs(c["s"] - 0.05) < 0.002 and abs(c["t"] - 0.15) < 0.002
    assert abs(c["angle"] - 17.0) < 1.5 and c["face"] == face and c["lead"]


def test_touch_off_the_steel(T):
    assert T.touch(knife_on(T, 1, 0.05, 0.15, out=0.02), steel_up(T))["gap"] > 0.015
    assert T.touch(T.knife_start(), T.steel_start())["gap"] > 0.02


def test_edge_up_is_not_leading(T):
    K = T.K
    pos, quat = knife_on(T, 1, 0.05, 0.15)
    rot = K._rotation(quat)
    flipped = np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])          # turned over about its length: edge up
    assert not T.touch((pos, K._quat(flipped)), steel_up(T))["lead"]


# -- strokes ---------------------------------------------------------------------------------------------------------------
def test_a_good_stroke_counts(T):
    s = stroke(T, T.new_state(), 1)
    assert len(s["strokes"]) == 1 and s["strokes"][0]["valid"] and s["counted"] == 1


def test_eight_alternating(T):
    s = T.new_state()
    for i in range(8):
        stroke(T, s, 1 if i % 2 == 0 else -1)
    assert s["counted"] == 8 and not s["void"]


def test_same_face_twice_counts_once(T):
    s = T.new_state()
    for _ in range(8):
        stroke(T, s, 1)
    assert s["counted"] == 1 and all(k["valid"] for k in s["strokes"])
    assert T.score_from(False, s["counted"], True, True, True, True, True) == 0.0


def test_middle_only_is_not_a_stroke(T):
    s = stroke(T, T.new_state(), 1, s0=0.06, s1=0.14)
    assert s["counted"] == 0 and "heel" in s["strokes"][0]["why"] and "tip" in s["strokes"][0]["why"]


def test_shallow_and_wide_angles_do_not_count(T):
    assert stroke(T, T.new_state(), 1, alpha=6.0)["counted"] == 0
    s = stroke(T, T.new_state(), 1, alpha=24.0)             # the edge's curve adds about 4 degrees toward the tip
    assert s["counted"] == 0 and not s["void"] and "angle" in s["strokes"][0]["why"]


def test_steep_voids(T):
    s = stroke(T, T.new_state(), 1, alpha=40.0)
    assert "degrees" in s["void"]


def test_scale_limits(T):
    assert stroke(T, T.new_state(), 1, grams=T.LIGHT_G - 5)["counted"] == 1
    s = stroke(T, T.new_state(), 1, grams=T.LIGHT_G + 20)
    assert s["counted"] == 0 and not s["void"] and "scale" in s["strokes"][0]["why"]
    assert "scale" in stroke(T, T.new_state(), 1, grams=T.HEAVY_G + 10)["void"]


def test_scale_rise_is_from_the_reading_before_the_stroke(T):
    s = T.new_state()
    st = steel_up(T)
    for _ in range(3):
        T.advance(s, knife_on(T, 1, 0.02, 0.20, out=0.03), st, 300.0, FAR, REST)      # the steel hand leaning on the board
    stroke(T, s, 1, grams=320.0)
    assert s["counted"] == 1 and abs(s["strokes"][0]["rise_g"] - 20.0) < 1.0


def test_up_the_steel_does_not_count(T):
    s = stroke(T, T.new_state(), 1, s0=0.02, s1=0.185, z0=0.15, z1=0.15)
    assert s["counted"] == 0 and "down the steel" in s["strokes"][0]["why"]


def test_steel_must_stand_on_the_pad(T):
    K = T.K
    s = T.new_state()
    tip, quat = steel_up(T)
    off = (tip + np.array([0.10, 0.0, 0.0]), quat)
    for k in range(41):
        f = k / 40
        pos, q = knife_on(T, 1, 0.02 + f * 0.165, 0.20 - f * 0.13)
        T.advance(s, (pos + np.array([0.10, 0.0, 0.0]), q), off, 12.0, FAR, REST)
    for _ in range(4):
        T.advance(s, T.knife_start(), off, 0.0, FAR, REST)
    assert s["counted"] == 0 and "upright" in s["strokes"][0]["why"]


# -- voids about the hand on the steel ---------------------------------------------------------------------------------------
def _steel_hand(T):
    K = T.K
    tip, quat = steel_up(T)
    at = tip + K._rotation(quat) @ np.array([T.HOLD_AT, 0.0, 0.0])
    return (at, np.array([at, at + [0.0, 0.03, 0.0], at - [0.0, 0.0, 0.02]]))


def test_blade_near_the_steel_hand_voids(T):
    K = T.K
    hand = _steel_hand(T)
    s = T.new_state()
    T.advance(s, knife_on(T, 1, 0.05, 0.15), steel_up(T), 0.0, [hand, FAR[1]], REST)
    assert not s["void"] and s["nearest_hand_mm"] > 20
    T.advance(s, knife_on(T, 1, 0.05, K.ROD_L - 0.02), steel_up(T), 0.0, [hand, FAR[1]], REST)
    assert "hand on the steel" in s["void"] or "guard" in s["void"]


def test_blade_past_the_guard_voids(T):
    K = T.K
    tip, quat = steel_up(T)
    hand = _steel_hand(T)
    far_hand = (hand[0], hand[1] + np.array([0.0, 0.2, 0.0]))                 # its points kept away: only the guard rule fires
    s = T.new_state()
    T.advance(s, knife_on(T, 1, 0.10, K.ROD_L + 0.03, out=0.02), (tip, quat), 0.0, [far_hand, FAR[1]], REST)
    assert "guard" in s["void"]


def test_no_hand_rule_while_the_steel_is_carried(T):
    s = T.new_state()
    hand = (np.asarray(T.steel_start()[0]), np.array([T.knife_start()[0]]))     # a hand right at the blade, steel lying
    T.advance(s, T.knife_start(), T.steel_start(), 0.0, [hand, FAR[1]], REST)
    assert not s["void"]


# -- how things are left -------------------------------------------------------------------------------------------------
def test_end_state_flags(T):
    K = T.K
    s = T.new_state()
    T.advance(s, T.knife_start(), steel_up(T), 0.0, FAR, REST)
    assert s["knife_down"] and not s["steel_down"]
    kp, kq = T.knife_start()
    T.advance(s, (kp + [0.0, 0.0, 0.05], kq), T.steel_start(), 0.0, FAR, REST)
    assert not s["knife_down"] and s["steel_down"]
    turned = K._quat(np.column_stack([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)]))    # edge toward the robot
    T.advance(s, (kp + [-K.BH, 0.0, 0.0], turned), T.steel_start(), 0.0, FAR, REST)
    assert s["knife_down"] and not s["edge_away"]
    near = [(kp, np.array([kp + [0.0, 0.0, 0.01]])), FAR[1]]
    T.advance(s, T.knife_start(), T.steel_start(), 0.0, near, REST)
    assert not s["let_go"]
    T.advance(s, T.knife_start(), T.steel_start(), 0.0, FAR, {"knife": (0.05, 0.0), "steel": (0.0, 0.0)})
    assert not s["still"]


# -- the layout ------------------------------------------------------------------------------------------------------------
def test_layout(T):
    K = T.K
    kp, kq = T.knife_start()
    sp, sq = T.steel_start()
    blade = K.blade_points(kp, kq)
    assert T.over_board(blade, 0.01)
    handle = K.place(kp, kq, np.array([[-0.02, K.BH - K.HANDLE_H / 2, 0.0], [-K.HL, K.BH - K.HANDLE_H / 2, 0.0]]))
    assert handle[:, 1].max() < T.BOARD_Y0                                    # the handle is clear of the board's right edge
    guard = K.place(sp, sq, K.steel_pieces()[0][0])
    assert T.over_board(guard) and guard[:, 1].max() > T.BOARD_RECT[3] - 0.012          # the guard is on the board, at its left edge
    assert abs(guard[:, 2].min() - T.TOP - T.BOARD_TOP) < 0.002                          # ... lying on its flat
    grip = K.place(sp, sq, np.array([[T.HOLD_AT - 0.02, 0.0, 0.0], [T.HOLD_AT + 0.02, 0.0, 0.0]]))
    assert grip[:, 1].min() > T.BOARD_RECT[3] + 0.02                                      # where it is taken is past the edge
    assert T.over_board(np.array([sp])) and abs(sp[2] - T.TOP - T.BOARD_TOP - K.ROD_R) < 0.002
    px, py = T.PAD_XY
    assert T.BOARD_RECT[0] + 0.04 < px < T.BOARD_RECT[1] - 0.04 and T.BOARD_RECT[2] + 0.04 < py < T.BOARD_RECT[3] - 0.04
    assert abs(K.scale_grams(-0.001, 0.0) - 1000 * K.SPRING * 0.001 / 9.81) < 1e-6
    names = [o.name for o in T.OBJECTS]
    assert names == [T.SCALE, T.BOARD, T.KNIFE, T.STEEL]


def test_scoring_text_matches_constants(T):
    doc = T.score.__doc__
    for word in ("1 mm", "3 cm", "10 to 25 degrees", "5 cm", "15 degrees", "30 g", "90 g", "2 cm", "30 degrees", "eight"):
        assert word in doc, word
    assert (T.TOUCH, T.NEAR_END, T.ANGLE, T.TRAVEL, T.UPRIGHT, T.LIGHT_G, T.HEAVY_G, T.HAND_NEAR, T.STEEP, T.EACH) == \
        (0.001, 0.03, (10.0, 25.0), 0.05, 15.0, 30.0, 90.0, 0.02, 30.0, 4)


# -- task.json -----------------------------------------------------------------------------------------------------------
def test_task_json(T):
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    j = json.loads(path.read_text())
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j.get(key), key
    assert j["instruction"] == T.INSTRUCTION
    assert len(j["title"]) <= 120 and len(j["fidelity"]) <= 500 and len(j.get("context", "")) <= 1000 and len(j["scores"]) <= 300
    assert j["category"] == "Impact on jobs"
