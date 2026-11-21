"""Offline checks of the tube task: shapes, the lid's catch, the station's rule and the scoring rule.

    python -m pytest -q test_scoring.py        (no simulator needed)
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task as T  # noqa: E402
import tubekit as K  # noqa: E402

Q0 = (0.0, 0.0, 0.0, 1.0)


def _radius(points) -> float:
    p = np.asarray(points, float)
    return float(np.hypot(p[:, 0], p[:, 1]).max())


# -- sizes -------------------------------------------------------------------------------------------------------------

def test_carrier_is_the_listed_size():
    pts = np.vstack([p for p, _ in K.carrier_pieces()])
    assert abs(pts[:, 2].min()) < 1e-9
    assert abs(pts[:, 2].max() - K.MOUTH_Z) < 1e-9 and abs(K.MOUTH_Z + K.LID_T - 0.228) < 1e-9
    assert K.BORE == 0.080 and K.RING_D == 0.104
    assert abs(2 * K.BODY_R - 0.086) < 1e-9          # what a gripper closes round


def test_body_fits_between_open_pads():
    assert 2 * K.BODY_R < 0.0939 - 0.006             # at least 3 mm a side


def test_rings_stand_out_and_pass_up_the_tube():
    assert K.RING_R > K.BODY_R + 0.005
    assert K.RING_R < K.TUBE_R                       # the (compressed) rings pass up the send tube
    assert K.RING_D / 2 > K.TUBE_R - 0.002           # ... and the listed rings nearly fill it
    assert K.COLLAR_R > K.RING_D / 2 + 0.010         # the entry collar is wider than the carrier


def test_vial_is_the_listed_size_and_fits_the_bore():
    pts = np.vstack([p for p, _ in K.vial_pieces()])
    assert abs(pts[:, 2].max() - 0.069) < 1e-9 and abs(2 * K.VIAL_R - 0.032) < 1e-9
    assert 2 * K.CAP_R < K.BORE - 0.030
    assert K.VIAL_H < K.MOUTH_Z - K.BASE_T


def test_pieces_are_small_convex_sets():
    for pieces in (K.carrier_pieces(), K.lid_pieces(), K.vial_pieces(), K.station_pieces()):
        for piece in pieces:
            assert 4 <= len(piece[0]) <= 40
            assert len(K.hull_faces(piece[0])) >= 4


# -- the lid -------------------------------------------------------------------------------------------------------------

def test_shut_lid_covers_the_mouth():
    mid = K.lid_middle(0.0)
    assert abs(mid[0]) < 1e-9 and abs(mid[1]) < 1e-9 and abs(mid[2] - K.LENGTH) < 1e-9
    assert K.LID_R > K.BORE / 2 + 0.003


def test_open_lid_leaves_room_for_the_vial():
    import demo as D

    mid = K.lid_middle(K.LID_MAX)[:2]
    drop = np.array(D.DROP_OFF)
    assert np.linalg.norm(drop - mid) - K.LID_R - K.CAP_R > 0.008        # clear of the open lid
    assert K.BORE / 2 - np.linalg.norm(drop) - K.CAP_R > 0.008           # and of the bore's wall


def test_lid_pose_and_angle_agree():
    c = np.array([0.3, -0.1, 0.75])
    for q in (Q0, K.turn_z(0.7)):
        for deg in (0.0, 3.0, 41.0, K.LID_MAX):
            pos, quat = K.lid_pose(c, q, deg)
            assert abs(K.lid_angle(q, quat) - deg) < 1e-6
            assert np.allclose(pos, K.place(c, q, K.PIN))
            # the lid's middle, placed from the lid's own pose, is where lid_middle says
            top = K.place(pos, quat, (0.0, -K.PIN_R, K.LID_T))
            assert np.allclose(top, K.place(c, q, K.lid_middle(deg)), atol=1e-9)


def test_catch_law():
    assert K.lid_torque(0.0) == -K.T_SHUT and K.lid_torque(K.SHUT_AT) == -K.T_SHUT        # held shut
    assert abs(K.lid_torque(K.BUMP_AT) - K.T_BUMP) < 1e-12                                # the bump's top
    assert K.lid_torque(K.BUMP_FROM) == 0.0 and K.lid_torque(K.LID_MAX) == 0.0            # free beyond it
    assert all(K.lid_torque(a) >= 0.0 for a in np.linspace(K.BUMP_AT, K.LID_MAX, 50))      # never pulled shut from afar
    crossing = K.SHUT_AT + (K.BUMP_AT - K.SHUT_AT) * K.T_SHUT / (K.T_SHUT + K.T_BUMP)
    assert K.SHUT_AT < crossing < K.BUMP_AT and abs(K.lid_torque(crossing)) < 1e-9
    assert K.T_SHUT > 3 * K.LID_DRAG and K.T_BUMP > 2 * K.LID_DRAG


def test_an_unheld_carrier_cannot_take_the_push():
    # the push on the rim that gets the lid over the bump, with the rim's friction on the finger working against it
    push = (K.T_BUMP + K.LID_DRAG) / (K.PIN_R - K.FRICTION[0] * K.LID_R)
    weight = (K.CARRIER_MASS + K.LID_MASS + K.VIAL_MASS) * 9.81
    assert push * K.MOUTH_Z > weight * K.BODY_R          # it tips a carrier standing free on its closed end


def test_a_push_on_the_rim_shuts_the_lid_against_friction():
    # A push on the lid's rim along the way the lid swings has the pin's offset as its arm; the rim sliding along the
    # finger drags the other way with the lid's radius as its arm. The push wins while PIN_R > mu * LID_R.
    assert K.PIN_R > K.FRICTION[0] * K.LID_R + 0.010


# -- the vial in the carrier ---------------------------------------------------------------------------------------------

def test_vial_in_carrier():
    c = np.array([0.27, -0.08, T.TOP])
    assert K.vial_in_carrier(c + (0.0, 0.0, K.BASE_T), Q0, c, Q0)
    assert K.vial_in_carrier(c + (0.012, -0.010, K.BASE_T), Q0, c, Q0)
    assert not K.vial_in_carrier(c + (0.0, 0.0, K.MOUTH_Z - 0.030), Q0, c, Q0)       # sticking out of the mouth
    assert not K.vial_in_carrier(c + (0.10, 0.0, 0.0), Q0, c, Q0)                    # beside the carrier
    assert not K.vial_in_carrier(np.array([0.28, 0.12, T.TOP]), Q0, c, Q0)           # where it starts
    lean = (0.0, math.sin(0.10), 0.0, math.cos(0.10))                                # leaning inside the bore
    assert K.vial_in_carrier(c + (-0.008, 0.0, K.BASE_T + 0.002), lean, c, Q0)


# -- the station ---------------------------------------------------------------------------------------------------------

def test_station_height_and_start():
    s = T.station_pos()
    assert abs(s[2] - T.TOP - 0.300) < 1e-9          # 1.2 m above the floor over a 0.9 m counter
    up, off, tilt = K.insertion(T.carrier_pos(), Q0, s)
    assert up < -0.05 and off > 0.10 and tilt == 0.0
    assert not K.taken(T.carrier_pos(), Q0, s)


def test_take_rule():
    s = T.station_pos()
    base = lambda up, dx=0.0: s + (dx, 0.0, up - K.LENGTH)       # noqa: E731
    assert not K.taken(base(K.TAKE - 0.002), Q0, s)
    assert K.taken(base(K.TAKE), Q0, s)
    assert not K.taken(base(K.TAKE + 0.01, dx=K.TAKE_OFF + 0.002), Q0, s)
    tilted = (math.sin(0.07), 0.0, 0.0, math.cos(0.07))          # 8 degrees
    top = s + (0.0, 0.0, K.TAKE + 0.01)
    assert not K.taken(top - K.place((0, 0, 0), tilted, (0.0, 0.0, K.LENGTH)), tilted, s)
    pos, quat = K.held_pose(s)
    assert K.taken(pos, quat, s)
    # where the station holds it, the upper ring is in the tube's bore and the lower ring is below the collar
    assert pos[2] + K.RING_AT[1] > s[2] + K.COLLAR_H and pos[2] + K.RING_AT[0] < s[2]


def test_in_tube_thresholds():
    assert T.ENTERED < T.IN_TUBE < K.TAKE


def test_body_distance():
    c = np.array([0.27, -0.08, T.TOP])
    assert K.body_distance([c + (0.0, 0.0, 0.1)], c, Q0) == 0.0
    assert abs(K.body_distance([c + (K.RING_D / 2 + 0.03, 0.0, 0.1)], c, Q0) - 0.03) < 1e-9
    assert abs(K.body_distance([c + (0.0, 0.0, K.LENGTH + 0.05)], c, Q0) - 0.05) < 1e-9


# -- scoring -------------------------------------------------------------------------------------------------------------

def test_score_tiers():
    f = T.score_from
    assert f(False, False, False, False, True, True, None) == 0.0
    assert f(True, False, False, False, True, True, None) == 0.25
    assert f(True, True, False, False, True, True, None) == 0.5
    assert f(True, True, True, False, True, True, None) == 0.75
    assert f(True, True, True, True, False, True, None) == 0.75        # taken but still held
    assert f(True, True, True, True, True, False, None) == 0.75        # taken but moving
    assert f(True, True, True, True, True, True, None) == 1.0


def test_no_vial_scores_nothing_whatever_else():
    for secured, in_tube, taken, released, still in itertools.product((False, True), repeat=5):
        assert T.score_from(False, secured, in_tube, taken, released, still, None) == 0.0


def test_broken_scores_nothing():
    for flags in itertools.product((False, True), repeat=6):
        assert T.score_from(*flags, "the carrier entered the send tube with its lid 40 deg open") == 0.0


def test_every_outcome_has_a_score():
    seen = set()
    for flags in itertools.product((False, True), repeat=6):
        for broken in (None, "x"):
            seen.add(T.score_from(*flags, broken))
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_open_lid_cannot_reach_a_loaded_tier():
    for in_tube, taken, released, still in itertools.product((False, True), repeat=4):
        assert T.score_from(True, False, in_tube, taken, released, still, None) == 0.25


def test_layouts():
    assert T.LOT in ("A", "B")
    names = [o.name for o in T.OBJECTS]
    assert names == ["wall", T.STATION, T.CARRIER, T.LID, T.VIAL]
    lid = T.OBJECTS[3]
    assert abs(K.lid_angle(Q0, lid.quat) - T.LID_START) < 1e-6


def test_text_matches_code():
    path = HERE / "task.json"
    if not path.exists():
        return
    doc = json.loads(path.read_text())
    assert doc["instruction"] == T.INSTRUCTION
    s = doc["scoring"]
    for phrase in ("0.25", "0.5", "0.75", "1.0", "3 degrees", "2 cm", "7 cm", "3 cm", "1 cm", "5 cm"):
        assert phrase in s, phrase
    assert (K.SHUT_AT, T.IN_TUBE, K.TAKE, T.CLEAR, T.ENTERED, T.FLOOR, T.STILL) == (3.0, 0.020, 0.070, 0.030, 0.010, 0.050, 0.020)
    assert len(doc["fidelity"]) <= 500 and len(doc["title"]) <= 120 and len(doc.get("context", "")) <= 1000
    assert len(doc["scores"]) <= 300
    assert "1,300 steps" in doc["run_limit"]
    for phrase in ("x 27.0, y -8.0", "28.5, -6.5", "x 28.0, y 12.0", "31.0, 10.0", "x 41.0, y -8.0", "30.0 up"):
        assert phrase in doc["setup"], phrase
    assert (tuple(T.MOUTH), K.LID_MAX) == ((0.410, -0.080, 0.300), 80.0)
