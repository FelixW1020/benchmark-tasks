"""Offline checks of the gearing, the lid's fit, the scoring rule, the drying rule and the spinner's geometry (no simulator):
python -m pytest test_scoring.py"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spinnerkit as K  # noqa: E402
import task as T  # noqa: E402

UPRIGHT = (0.0, 0.0, 0.0, 1.0)
CW = K.FORWARD                     # the sign of a clockwise (forward) joint speed


@pytest.fixture(autouse=True)
def fresh():
    T._clear()
    yield
    T._clear()


# -- the gearing: 1:5 through a one-way clutch ---------------------------------------------------------------------------

def test_cranking_forward_drives_the_basket_forward_and_the_crank_feels_it():
    on_basket, on_crank = K.gear(CW * 4.0, 0.0)
    assert on_basket * CW > 0                       # the basket is pushed forward ...
    assert on_crank * CW < 0                        # ... and the crank is resisted
    assert abs(on_crank) > K.RATIO * abs(on_basket) * 0.99 - 1e-9
    assert abs(on_crank) < 0.15                     # N m: about 2 N at the knob, as by hand


def test_at_its_geared_speed_the_basket_is_not_driven_further():
    on_basket, _ = K.gear(CW * 4.0, CW * 4.0 * K.RATIO)
    assert on_basket * CW < 0                       # only the lettuce's drag acts


def test_turning_the_crank_backward_does_not_drive_the_basket():
    on_basket, on_crank = K.gear(-CW * 4.0, 0.0)
    assert on_basket == 0.0
    assert on_crank * CW > 0                        # the gears still resist the crank (now going backward)


def test_the_clutch_spins_the_basket_up_over_a_second_or_two():
    on_basket, _ = K.gear(CW * 4.0, 0.0)
    assert abs(on_basket) <= K.CLUTCH_MAX
    assert K.RATIO * 4.0 * K.BASKET_I / K.CLUTCH_MAX == pytest.approx(1.7, abs=0.3)   # s to the geared speed


def test_a_spinning_basket_coasts_down():
    on_basket, on_crank = K.gear(0.0, CW * 20.0)
    assert on_basket * CW < 0 and on_crank == 0.0
    assert K.BASKET_I / K.DRAG == pytest.approx(3.9, abs=0.3)      # seconds: it coasts for several seconds


def test_with_the_lid_off_the_crank_drives_nothing():
    on_basket, on_crank = K.gear(CW * 6.0, 0.0, meshed=False)
    assert on_basket == 0.0                         # the lid's gear is out of the basket's hub
    assert on_crank * CW < 0                        # the gears' own friction is still felt
    spinning, _ = K.gear(CW * 6.0, CW * 20.0, meshed=False)
    assert spinning * CW < 0                        # a basket already spinning only coasts down


# -- the lid on the bowl --------------------------------------------------------------------------------------------------

def test_a_lid_in_place_is_on():
    s = T.spinner_start()
    sit = K.seat(s, UPRIGHT, s, UPRIGHT)
    assert sit["lift"] == pytest.approx(0.0) and sit["off"] == pytest.approx(0.0) and sit["tilt"] == pytest.approx(0.0)
    assert K.lid_on(sit)
    turned = (0.0, 0.0, math.sin(0.4), math.cos(0.4))                       # the whole spinner turned on the table
    assert K.lid_on(K.seat(s, turned, s, turned))


def test_a_lifted_or_tipped_lid_is_off():
    s = T.spinner_start()
    assert K.lid_on(K.seat(s, UPRIGHT, s + [0.0, 0.0, 0.003], UPRIGHT))     # 3 mm up: its rim still round the bowl's
    assert not K.lid_on(K.seat(s, UPRIGHT, s + [0.0, 0.0, 0.006], UPRIGHT))
    tipped = (math.sin(math.radians(5)), 0.0, 0.0, math.cos(math.radians(5)))    # 10 degrees about x
    sit = K.seat(s, UPRIGHT, s, tipped)
    assert math.degrees(sit["tilt"]) == pytest.approx(10.0, abs=0.01) and not K.lid_on(sit)


def test_the_lids_rim_fits_round_the_bowl_and_over_its_lugs():
    assert K.LID_R > K.BOWL_R + K.SKIRT[0] + 0.003                           # a rim at least 3 mm thick
    assert K.BOWL_R + K.LUG[0] <= K.LID_R                                    # the lugs stay inside the rim's thickness
    assert K.LUG[2] <= K.SKIRT[1]                                            # ... and inside its height
    assert K.SKIRT[1] > 2 * K.ON_LIFT                                        # "off" comes well before the rim is clear
    # with the rim round the bowl, the lid cannot tip as far as "off": it has to be lifted to come off
    wedge = math.asin(2 * K.SKIRT[0] / K.SKIRT[1])
    assert wedge < K.ON_TILT


def test_the_rim_leaves_a_notch_at_each_lug():
    pieces = K.rim_pieces()
    r_mid = K.BOWL_R + K.SKIRT[0] + (K.LID_R - K.BOWL_R - K.SKIRT[0]) / 2
    half_lug = K.LUG[1] / 2 / r_mid
    covered = 0.0
    for ang, width in pieces:
        half = math.atan(width / 2 / r_mid)
        covered += 2 * half
        for lug in (0.0, math.pi):
            gap = abs(math.remainder(ang - lug, 2 * math.pi)) - half        # from the piece's end to the lug's middle
            assert gap > half_lug + 0.5 * K.LUG_PLAY / r_mid                 # clear of the lug, with play
    assert covered == pytest.approx(2 * math.pi - 4 * (half_lug + K.LUG_PLAY / r_mid), abs=1e-6)


# -- scoring --------------------------------------------------------------------------------------------------------------

def test_tiers_are_by_the_share_of_the_water_spun_out():
    assert T.score_from(0.0, False, False, True) == 0.0
    assert T.score_from(0.32, False, False, True) == 0.0                     # under a third: still wet
    assert T.score_from(1 / 3, False, False, True) == 0.25
    assert T.score_from(0.49, False, False, True) == 0.25
    assert T.score_from(0.5, False, False, True) == 0.5
    assert T.score_from(0.66, False, False, True) == 0.5                     # not quite two-thirds
    assert T.score_from(2 / 3, False, False, True) == 1.0
    assert T.score_from(0.72, False, False, False) == 0.5                    # still holding the knob: not let go


def test_knocking_the_spinner_about_or_losing_the_lid_zeroes_the_trial():
    assert T.score_from(0.9, True, False, True) == 0.0
    assert T.score_from(0.9, False, True, True) == 0.0


def test_score_reads_the_trial_record():
    T._clear()
    assert T.score({}) == 0.0                                                 # nothing done: all the water still on
    T._STATE.update(water=0.30, let_go=True)
    assert T.score({}) == 1.0 and T.success({})
    T._STATE.update(lid_off=True)
    assert T.score({}) == 0.0
    T._clear()


# -- the drying rule ------------------------------------------------------------------------------------------------------

def spin(rps: float, seconds: float, water: float = 1.0, dt: float = 0.05) -> float:
    for _ in range(round(seconds / dt)):
        water = T.drain(water, rps, dt)
    return water


def test_the_rule_is_in_terms_of_the_push_at_the_wall_not_of_rpm():
    # the same rpm presses twice as hard in a basket of twice the radius ...
    assert T.wall_g(5.0, 0.176) == pytest.approx(2 * T.wall_g(5.0, 0.088))
    # ... and this basket at 300 rpm presses at 8.9 times gravity
    assert T.wall_g(5.0) == pytest.approx(8.9, abs=0.1)
    assert T.holds(0.5) == 1.0 and T.holds(1.0) == 1.0                       # nothing leaves at 1 g: it has already dripped
    assert T.holds(4.0) == pytest.approx(0.5) and T.holds(9.0) == pytest.approx(1 / 3)
    assert T.holds(100.0) == pytest.approx(0.1)


def test_water_only_ever_leaves_and_faster_spinning_takes_more():
    assert spin(0.0, 60.0) == 1.0                                             # a still basket dries nothing
    left = [spin(rps, 60.0) for rps in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]
    assert all(a >= b for a, b in zip(left, left[1:]))
    w = spin(6.0, 3.0)
    assert spin(2.0, 10.0, water=w) == w                                      # slowing down puts none back


def test_slow_cranking_however_long_earns_nothing():
    # a basket kept at 2.4 turns a second (144 rpm: 2.0 times gravity at its wall) never spins out a third of the
    # water, however long it is cranked
    assert T.wall_g(2.4) == pytest.approx(2.0, abs=0.05)
    assert 1.0 - spin(2.4, 600.0) < T.SOME
    assert T.score_from(1.0 - spin(2.4, 600.0), False, False, True) == 0.0


def test_what_each_tier_takes():
    some, half, dry = (T.turns_a_second_for(x) for x in (T.SOME, T.HALF, T.DRY))
    assert some == pytest.approx(2.5, abs=0.1)                                # 151 rpm kept up for long
    assert half == pytest.approx(3.4, abs=0.1)                                # 202 rpm
    assert dry == pytest.approx(5.0, abs=0.1)                                 # 302 rpm, and only after a long time
    assert some / K.RATIO == pytest.approx(0.5, abs=0.03)                     # half a crank turn a second, for a third
    # at exactly the speed whose limit is two-thirds, 5 seconds is not enough; well over it, 6 seconds is
    assert 1.0 - spin(5.0, 5.0) < T.DRY
    assert 1.0 - spin(6.0, 6.0) >= T.DRY
    assert 1.0 - spin(6.0, 2.0) < T.DRY                                       # a short burst is not


def test_the_instruction_says_what_full_marks_take():
    assert "two-thirds" in T.INSTRUCTION and "5 turns a second" in T.INSTRUCTION and "6 seconds" in T.INSTRUCTION
    assert 0.0 < T.SOME < T.HALF < T.DRY < 1.0
    assert T.DRAIN_S > 0 and T.HOLD_EXP == pytest.approx(0.5)


# -- geometry -------------------------------------------------------------------------------------------------------------

def test_the_knob_goes_round_the_lid_and_inside_its_edge():
    s = T.spinner_start()
    for a in np.linspace(0.0, 2 * math.pi, 9):
        p = K.knob_point(s, UPRIGHT, a)
        assert np.linalg.norm(p[:2] - s[:2]) == pytest.approx(K.KNOB_AT)
        assert p[2] == pytest.approx(s[2] + K.CRANK_Z + K.ARM[2])
    assert K.KNOB_AT + K.KNOB_R < K.LID_R


def test_the_handle_stands_clear_of_the_knobs_circle():
    """The hand on the handle and the hand on the knob never come closer than 8.5 cm (across the table)."""
    handle_end = K.BOWL_R + K.TAB[1] - 0.017
    assert handle_end - K.KNOB_AT > 0.085


def test_lid_sits_on_the_bowl_and_the_crank_on_the_lid():
    assert K.LID_Z[0] == pytest.approx(K.BOWL_H)
    assert K.HUB_Z[0] == pytest.approx(K.LID_Z[1]) and K.CRANK_Z == pytest.approx(K.HUB_Z[1])
    assert K.BASKET_Z[1] < K.BOWL_H and K.BASKET_R < K.BOWL_R - K.WALL


def test_the_scene_is_a_bowl_and_a_separate_lid():
    names = [o.name for o in T.OBJECTS]
    assert names == [T.NAME, T.LID]
    bowl, lid = T.OBJECTS
    assert isinstance(bowl, K.Spinner) and isinstance(lid, K.Lid) and bowl.lid == lid.name
    assert tuple(bowl.pos) == tuple(lid.pos)                                 # the lid starts on the bowl
