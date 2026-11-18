"""Checks of the timer task's sizes, dial reading, marks, layouts, demo plan and scoring rule. No simulator needed:

    python -m pytest -q test_scoring.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yam_arena import task_file  # noqa: E402


def load(lot: str):
    os.environ["TIMER_LOT"] = lot
    return task_file.load(HERE / "task.py").module


T = load("A")
K = T.K
I = (0.0, 0.0, 0.0, 1.0)


def turned(deg: float) -> tuple:
    """The top's quaternion when a timer whose base has not moved reads ``deg`` degrees: the top has gone that far
    anticlockwise (seen from above) round the base, which is the base gone clockwise under the top."""
    return K.turn_z(math.radians(deg))


# -- the timer -----------------------------------------------------------------------------------------------------------

def test_timer_is_the_stated_size():
    assert 2 * K.RADIUS == pytest.approx(0.060) and K.HEIGHT == pytest.approx(0.060)
    assert K.SPLIT + K.SEAM + K.TOP_H == pytest.approx(K.HEIGHT)
    assert K.BASE_MASS + K.TRAIN_MASS + K.TOP_MASS == pytest.approx(0.110)


def test_both_parts_fit_between_the_open_fingers_with_room():
    assert 2 * K.RADIUS < 0.0939 - 0.020


def test_each_part_is_tall_enough_for_a_fingertip():
    assert min(K.SPLIT, K.TOP_H) >= 0.016          # the fingertips are 16 mm wide at the end


def test_scale_runs_one_turn_for_an_hour():
    assert K.DEG_PER_MIN * 60 == pytest.approx(360.0)
    assert K.minutes(60.0) == pytest.approx(10.0) and K.minutes(330.0) == pytest.approx(55.0)


def test_clockwork_runs_down_a_minute_a_minute():
    assert K.RUN_DEG_S * 60 == pytest.approx(K.DEG_PER_MIN)


def test_far_stop_is_past_the_winding_mark_and_short_of_a_full_turn():
    assert K.DEG_PER_MIN * 55 < K.STOP_DEG < 360.0
    assert K.STOP_DEG / 2 < 180.0                   # each half of the hinge keeps its stops under half a turn apart


def test_mainspring_pulls_harder_the_further_it_is_wound():
    pulls = [K.spring_torque(d) for d in (0.0, 60.0, 180.0, 330.0)]
    assert pulls == sorted(pulls) and pulls[0] == pytest.approx(K.SPRING_0)
    assert K.spring_torque(-20.0) == pytest.approx(K.SPRING_0) and K.spring_torque(400.0) == pytest.approx(K.SPRING_1)


def test_going_train_is_set_to_the_clocks_rate_and_holds_far_more_than_it_lets_slip():
    # left alone the governor allows the mainspring to add only a small part of the clock's rate
    assert K.spring_torque(K.STOP_DEG) / K.GOVERNOR < 0.1 * K.RUN_DEG_S
    # at the demo's 60 degrees a second it would resist with far more than the clutch lets through
    assert K.GOVERNOR * 60.0 > 20 * K.CLUTCH


def test_clutch_holds_the_dial_against_the_mainspring_everywhere():
    assert K.CLUTCH > K.spring_torque(K.STOP_DEG)


def test_timer_turned_by_one_hand_where_it_stands_slips_on_the_table():
    # the most the table's friction can resist about the timer's axis, were all its weight carried at the rim
    weight = (K.BASE_MASS + K.TRAIN_MASS + K.TOP_MASS) * 9.81
    assert max(K.FRICTION) * weight * K.RADIUS < K.CLUTCH


def test_a_pinch_across_the_timer_can_carry_the_winding_torque():
    squeeze = 5.0 * 1000 * 0.00175                 # N a finger: 5 N per mm of commanded squeeze
    assert 2 * 0.4 * squeeze * K.RADIUS > 3 * (K.CLUTCH + K.SPRING_1)


# -- reading the dial -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0.0, 30.0, 60.0, 150.0, -45.0])
def test_relative_turn_is_the_base_clockwise_under_the_top(deg):
    assert K.relative_turn(I, turned(deg)) == pytest.approx(deg, abs=1e-9)
    # the same reading with the top held still and the base turned clockwise (seen from above) under it
    assert K.relative_turn(K.turn_z(-math.radians(deg)), I) == pytest.approx(deg, abs=1e-9)


def test_numerals_rise_from_left_to_right_seen_from_the_side():
    # seen from outside at the zero mark, the viewer's right is the way the azimuth grows
    assert 0.0 < K.mark_azimuth(5.0) < K.mark_azimuth(10.0) < math.pi


def test_relative_turn_does_not_depend_on_how_the_timer_is_held():
    tip = (math.sin(0.6), 0.0, 0.0, math.cos(0.6))                    # the timer on its side
    def mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)
    assert K.relative_turn(mul(tip, K.turn_z(1.1)), mul(tip, mul(K.turn_z(1.1), turned(75.0)))) == pytest.approx(75.0, abs=1e-6)


def test_unwrap_follows_the_dial_past_half_a_turn():
    seen, deg = 0.0, 0.0
    for step in range(1, 70):
        deg = 5.0 * step
        seen = K.unwrap(K.relative_turn(I, turned(deg)), seen)
    assert seen == pytest.approx(345.0, abs=1e-6)
    for step in range(57):
        deg -= 5.0
        seen = K.unwrap(K.relative_turn(I, turned(deg)), seen)
    assert seen == pytest.approx(60.0, abs=1e-6)


def test_tilt():
    assert K.tilt(I) == pytest.approx(0.0, abs=1e-6)
    assert K.tilt((math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))) == pytest.approx(90.0, abs=1e-6)
    assert K.tilt((1.0, 0.0, 0.0, 0.0)) == pytest.approx(180.0, abs=1e-6)


def test_distance_from_the_timer():
    pos = np.array([0.3, 0.1, 0.75])
    assert K.axis_distance(pos + (K.RADIUS + 0.030, 0.0, 0.030), pos, I) == pytest.approx(0.030)
    assert K.axis_distance(pos + (0.0, 0.0, K.HEIGHT + 0.050), pos, I) == pytest.approx(0.050)
    assert K.axis_distance(pos + (0.0, 0.0, 0.030), pos, I) < 0.0
    assert K.axis_distance(pos + (K.RADIUS + 0.030, 0.0, K.HEIGHT + 0.040), pos, I) == pytest.approx(0.050)


# -- the marks -------------------------------------------------------------------------------------------------------------

def test_marks_lie_on_the_metal_within_each_part():
    for quads, height in ((K.scale_quads(), K.SPLIT), (K.index_quads()[:1], K.TOP_H)):
        for q in quads:
            for x, y, z in q:
                assert math.hypot(x, y) == pytest.approx(K.RADIUS + K.PROUD, abs=1e-9)
                assert 0.0 < z < height


def test_scale_has_a_tick_a_minute_and_numerals_every_five():
    ticks = [q for q in K.scale_quads() if max(p[2] for p in q) == pytest.approx(K.TICK_TOP)]
    assert len(ticks) == 60
    long = [q for q in ticks if K.TICK_TOP - min(p[2] for p in q) == pytest.approx(K.TICK_FIVE)]
    assert len(long) == 12
    bars = sum(len(K._SEGMENTS[ch]) for m in range(0, 60, 5) for ch in str(m))
    assert len(K.scale_quads()) == 60 + bars


def test_index_points_at_the_minute_the_dial_reads():
    # the index is on the top's x; with the timer reading 10 minutes it stands over the base's 10-minute mark
    index = K._rotation(turned(60.0)) @ np.array([1.0, 0.0, 0.0])
    mark = K.mark_azimuth(10.0)
    assert index[:2] == pytest.approx([math.cos(mark), math.sin(mark)], abs=1e-9)


def test_numerals_do_not_run_into_each_other():
    width = 2 * K.DIGIT_W + K.DIGIT_GAP
    assert width < K.RADIUS * math.radians(30.0) - 0.002


# -- layouts -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lot", ["A", "B"])
def test_layout_stands_the_timer_on_the_table_reading_zero(lot):
    M = load(lot)
    pos, quat = M.timer_start()
    base, train, top = M.OBJECTS
    assert base.name == M.TIMER and top.name == M.DIAL and train.name == M.TRAIN
    assert top.base == M.TIMER and top.train == M.TRAIN and train.base == M.TIMER
    assert base.pos[2] - M.TOP == pytest.approx(M.REST) and top.pos[2] - base.pos[2] == pytest.approx(K.SPLIT + K.SEAM)
    assert train.pos[2] - base.pos[2] == pytest.approx(K.SPLIT + K.SEAM / 2)
    assert K.relative_turn(base.quat, top.quat) == pytest.approx(0.0, abs=1e-9)
    assert K.relative_turn(base.quat, train.quat) == pytest.approx(0.0, abs=1e-9)
    assert M.standing(pos, quat)["standing"]
    assert 0.20 <= pos[0] <= 0.36 and abs(pos[1]) <= 0.12
    assert not base.fixed


def test_the_two_layouts_differ():
    a, b = load("A"), load("B")
    assert np.linalg.norm(np.subtract(a.LAYOUT[0], b.LAYOUT[0])) >= 0.10
    assert abs(a.LAYOUT[1] - b.LAYOUT[1]) >= 45.0
    load("A")


def test_instruction_gives_nothing_away():
    text = T.INSTRUCTION.lower()
    for word in ("55", "wind", "past", "back", "hold", "both", "clockwise", "ring"):
        assert word not in text


# -- the demo's plan -------------------------------------------------------------------------------------------------------

def test_demo_winds_past_the_mark_and_short_of_the_stop():
    import demo as D

    assert T.WOUND_MIN + 2.0 <= D.WIND_TO <= K.minutes(K.STOP_DEG) - 1.5
    p = D.plan(T)
    assert math.degrees(p["wind"]) == pytest.approx(K.DEG_PER_MIN * D.WIND_TO)
    assert math.degrees(p["wind"] - p["back"]) == pytest.approx(K.DEG_PER_MIN * D.SET_TO)
    assert abs(D.SET_TO - T.SET_MIN) < T.EXACT / 2


def test_demo_hands_take_one_part_each_and_stay_apart():
    import demo as D

    p = D.plan(T)
    tip = 0.013                                               # the pads reach this far past the tool point
    assert D.T_IN + tip <= K.TOP_H - 0.003                    # the right hand's pads stay on the upper part
    assert D.B_IN + tip <= K.SPLIT - 0.003                    # the left hand's on the lower
    assert np.linalg.norm(p["left_at"] - p["right_at"]) == pytest.approx(K.HEIGHT - D.T_IN - D.B_IN)
    assert K.HEIGHT - (D.T_IN + tip) - (D.B_IN + tip) >= 0.010   # the two hands' fingertips, end to end


def test_demo_wrists_have_the_range():
    import demo as D
    from yam_arena import ik

    p = D.plan(T)
    span = math.degrees(ik.UPPER[5] - ik.LOWER[5])
    assert math.degrees(D.HALF * p["wind"]) <= span - 40 and math.degrees((1 - D.HALF) * p["wind"]) <= span - 40


# -- winding ---------------------------------------------------------------------------------------------------------------

def test_reaching_the_mark_winds_the_bell():
    assert not T.wound_after(False, 53.9)
    assert T.wound_after(False, 54.0) and T.wound_after(False, 57.5)


def test_winding_lasts_on_the_way_back_to_the_time_wanted():
    wound = False
    for minute in (20.0, 57.0, 40.0, 10.0, 9.0):
        wound = T.wound_after(wound, minute)
    assert wound


def test_back_at_zero_the_winding_is_spent():
    wound = False
    for minute in (57.0, 20.0, 0.4, 10.0):
        wound = T.wound_after(wound, minute)
    assert not wound


def test_standing():
    pos, quat = T.timer_start()
    assert T.standing(pos, quat)["standing"]
    lean = (math.sin(math.radians(6.0)), 0.0, 0.0, math.cos(math.radians(6.0)))          # 12 degrees
    assert not T.standing(pos, lean)["standing"]
    assert not T.standing(pos + (0.0, 0.0, 0.020), quat)["standing"]                       # held up
    assert T.standing(pos + (0.0, 0.0, 0.020), quat)["on_table"]
    assert not T.standing(pos - (0.0, 0.0, 0.200), quat)["on_table"]                       # on the floor
    side = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
    assert not T.standing(pos + (0.0, 0.0, K.RADIUS), side)["standing"]                    # lying down


# -- the scoring rule --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("minute", [9.0, 10.0, 10.3, 11.0])
def test_full_marks_need_winding_the_time_and_letting_go(minute):
    assert T.score_from(True, minute, True, True, True) == 1.0


@pytest.mark.parametrize("minute", [7.0, 8.9, 11.1, 13.0])
def test_a_time_within_three_minutes_scores_three_quarters(minute):
    assert T.score_from(True, minute, True, True, True) == 0.75


@pytest.mark.parametrize("minute", [0.6, 6.9, 13.1, 30.0, 57.5])
def test_wound_and_left_standing_at_another_time_scores_half(minute):
    assert T.score_from(True, minute, True, True, True) == 0.5


@pytest.mark.parametrize("standing,released,still", [(True, False, True), (True, True, False), (False, True, True), (False, False, False)])
def test_wound_but_not_left_standing_scores_a_quarter(standing, released, still):
    assert T.score_from(True, 10.0, standing, released, still) == 0.25


@pytest.mark.parametrize("minute", [9.0, 10.0, 11.0])
def test_straight_to_the_time_without_winding_scores_a_quarter(minute):
    assert T.score_from(False, minute, True, True, True) == 0.25


@pytest.mark.parametrize("minute", [0.0, 8.9, 11.1, 30.0, 53.9])
def test_unwound_at_any_other_time_scores_nothing(minute):
    assert T.score_from(False, minute, True, True, True) == 0.0


def test_unwound_and_still_held_scores_nothing():
    assert T.score_from(False, 10.0, True, False, True) == 0.0
    assert T.score_from(False, 10.0, False, True, True) == 0.0


def test_untouched_timer_scores_nothing():
    assert T.score_from(False, 0.0, True, True, True) == 0.0


def test_off_the_table_scores_nothing_whatever_else():
    assert T.score_from(True, 10.0, False, True, True, on_table=False) == 0.0
    assert T.score_from(True, 10.0, True, True, True, on_table=False) == 0.0


def test_score_is_one_of_five_values_everywhere():
    seen = set()
    for wound in (False, True):
        for minute in np.arange(0.0, 60.0, 0.5):
            for flags in range(16):
                seen.add(T.score_from(wound, float(minute), bool(flags & 1), bool(flags & 2), bool(flags & 4), bool(flags & 8)))
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_score_reads_the_trial_state():
    T._clear()
    assert T.score(None) == 0.0 and not T.success(None)
    T._STATE.update(deg=61.0, deg_max=345.0, wound=True, standing=True, released=True, still=True, on_table=True)
    assert T.score(None) == 1.0 and T.success(None)
    T._STATE.update(wound=False)
    assert T.score(None) == 0.25
    rep = T.report(None)
    assert rep["minutes"] == pytest.approx(10.17) and rep["max_minutes"] == pytest.approx(57.5) and rep["score"] == 0.25
    T._clear()


# -- task.json -----------------------------------------------------------------------------------------------------------

def test_task_json_matches_the_task_file():
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    j = json.loads(path.read_text())
    assert j["instruction"] == T.INSTRUCTION
    assert len(j["title"]) <= 120 and len(j["fidelity"]) <= 500 and len(j.get("context", "")) <= 1000 and len(j["scores"]) <= 300
    assert j["category"] in ("Impact on jobs", "Impact on AI timelines", "Safety and alignment", "Societal relatability", "Other")
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j[key]
