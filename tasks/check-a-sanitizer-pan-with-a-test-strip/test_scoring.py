"""Scoring and geometry rules of the sanitizer-pan task, checked without the simulator.

    python -m pytest test_scoring.py -q        (needs yam_arena on the path, as on the evaluation instance)
"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import sanikit as K  # noqa: E402
import task as T  # noqa: E402

UPRIGHT = (0.0, 0.0, 0.0, 1.0)


def at(x, y, h):
    return np.array([x, y, T.TOP + h])


# -- the scoring ladder ------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    assert T.score_from(False, False, 0, False, True) == 0.0
    assert T.score_from(False, False, 200, False, True) == 0.0       # a made-up pan left untested earns nothing


def test_tested_but_still_weak():
    assert T.score_from(True, False, 0, False, False) == 0.25
    assert T.score_from(True, False, 0, True, True) == 0.25          # binning the strip does not make up for no tablet


def test_right_strength_after_a_test():
    assert T.score_from(True, False, 200, False, False) == 0.5
    assert T.score_from(True, False, 200, True, False) == 0.75
    assert T.score_from(True, False, 200, True, True) == 1.0


def test_blind_dose_scores_zero_even_at_the_right_strength():
    assert T.score_from(False, True, 200, True, True) == 0.0
    assert T.score_from(True, True, 200, True, True) == 0.0          # testing afterwards does not redeem it


def test_too_strong_scores_zero():
    assert T.score_from(True, False, 400, True, True) == 0.0
    assert T.score_from(True, False, 600, True, True) == 0.0


@pytest.mark.parametrize("ppm, level", [(0, 0), (100, 100), (200, 200), (400, 400), (600, 400), (40, 0), (160, 200)])
def test_reading_is_the_nearest_chart_level(ppm, level):
    assert T.reading(ppm) == level


# -- the liquid --------------------------------------------------------------------------------------------------------
def test_one_gallon_depth():
    assert T.DEPTH * T.PAN_IN[0] * T.PAN_IN[1] == pytest.approx(0.003785, abs=1e-6)
    assert 0.050 < T.SURFACE < 0.060 < T.PAN_H


def test_under_needs_the_pan_and_the_depth():
    x, y = T.PAN_XY
    assert T.under(at(x, y, T.SURFACE - 0.002))
    assert not T.under(at(x, y, T.SURFACE + 0.002))
    assert not T.under(at(x + T.PAN_IN[0] / 2 + 0.01, y, 0.01))      # beside the pan, low down
    assert not T.under(at(x, y + T.PAN_IN[1] / 2 + 0.01, 0.01))


def _held_strip(end_height):
    """A strip held at the rest's slant with its pad end at this height over the dip point."""
    return at(0.290, -0.105, end_height), T.STRIP_QUAT


def test_pad_wet_needs_all_of_the_pad_under():
    pos, quat = _held_strip(T.SURFACE - 0.012)
    assert T.pad_wet(pos, quat) and T.handle_up(pos, quat)
    pos, quat = _held_strip(T.SURFACE - 0.004)                       # only the pad's lower part is in
    assert not T.pad_wet(pos, quat)
    pos, quat = _held_strip(T.SURFACE + 0.010)
    assert not T.pad_wet(pos, quat)


def test_a_strip_lying_in_the_pan_is_not_a_dip():
    flat = K.turn_x(math.pi / 2)                                     # lying along -y on the pan's floor
    pos = at(0.30, 0.0, T.PAN_WALL + 0.001)
    assert T.pad_wet(pos, flat)
    assert not T.handle_up(pos, flat)


def test_strip_points_follow_the_slant():
    pts = K.strip_points(T.strip_start(), T.STRIP_QUAT)
    assert np.allclose(pts["top"] - pts["foot"], K.STRIP_L * T.UP_STRIP, atol=1e-9)
    normal = K._rotation(T.STRIP_QUAT)[:, 1]                         # the pad faces up and toward the pan
    assert normal[1] > 0.8 and normal[2] > 0.4


# -- the waste cup -----------------------------------------------------------------------------------------------------
def test_in_cup():
    cup = at(T.CUP_XY[0], T.CUP_XY[1], 0.0)
    assert T.in_cup(cup + [0.01, 0.0, 0.03], cup, UPRIGHT)
    assert not T.in_cup(cup + [0.06, 0.0, 0.03], cup, UPRIGHT)       # beside it
    assert not T.in_cup(cup + [0.0, 0.0, T.CUP_H + 0.02], cup, UPRIGHT)   # still held above it
    tipped = K.turn_x(math.radians(80.0))
    assert not T.in_cup(cup + [0.0, 0.0, 0.03], cup, tipped)         # the cup knocked over


# -- the scene ---------------------------------------------------------------------------------------------------------
def test_scene_loads_and_names_are_unique():
    names = [o.name for o in T.OBJECTS]
    assert len(names) == len(set(names))
    for o in T.OBJECTS:
        o._check()
    assert {T.STRIP, T.CUP, T.PAN, *T.TABLETS} <= set(names)


def test_strip_rest_holds_the_strip_clear_of_the_table_and_the_pinch_clear_of_the_rest():
    rot = K._rotation(T.STRIP_QUAT)
    lowest = []
    for o in T.OBJECTS:
        if o.name.startswith("rest_"):
            r = K._rotation(getattr(o, "quat", UPRIGHT))
            half = np.abs(r) @ (np.asarray(o.size) / 2)
            lowest.append(o.pos[2] - half[2] - T.TOP)
    assert min(lowest) >= -1e-9                                       # nothing sunk into the worktop
    rib = next(o for o in T.OBJECTS if o.name == "rest_rib")
    assert rib.size[0] < K.STRIP_W - 0.002                           # fingers on the strip's edges do not meet the rib
    stop = next(o for o in T.OBJECTS if o.name == "rest_stop")
    face = np.asarray(stop.pos) + rot @ np.array([0.0, 0.0, stop.size[2] / 2])
    assert abs(float((T.strip_start() - face) @ rot[:, 2])) < 0.001   # the strip's end sits on the stop


def test_chart_digits():
    assert len(K.digit_boxes("0", 0.008, 0.0045, 0.0011, 0.0016)) == 6
    assert len(K.digit_boxes("100", 0.008, 0.0045, 0.0011, 0.0016)) == 2 + 6 + 6
    us = [b[0] for b in K.digit_boxes("400", 0.008, 0.0045, 0.0011, 0.0016)]
    assert min(us) < 0 < max(us)                                     # centred


def test_tablets_start_on_the_board_and_outside_the_pan():
    for o in T.OBJECTS:
        if o.name in T.TABLETS:
            assert not T.in_pan(o.pos)
            assert o.pos[2] - T.TOP == pytest.approx(T.BOARD[2] + T.TABLET_H / 2, abs=0.001)
            assert abs(o.pos[0] - T.BOARD_XY[0]) < T.BOARD[0] / 2 and abs(o.pos[1] - T.BOARD_XY[1]) < T.BOARD[1] / 2
