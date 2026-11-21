"""Offline checks of the scoring rule and the kit's geometry (no simulator needed).

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


def _load(lot: str):
    from yam_arena import task_file

    os.environ["DOOR_LOT"] = lot
    for k in ("DOOR_THETA", "DOOR_FLIP"):
        os.environ.pop(k, None)
    return task_file.load(str(HERE / "task.py")).module


@pytest.fixture(scope="module")
def A():
    return _load("A")


@pytest.fixture(scope="module")
def B():
    return _load("B")


# -- the scoring rule ------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero(A):
    assert A.score_from(cam=False, seated=False, down=False, released=True, still=True) == 0.0


def test_cam_only(A):
    assert A.score_from(cam=True, seated=False, down=False, released=True, still=True) == 0.25


def test_handle_seated_keeper_up(A):
    assert A.score_from(cam=True, seated=True, down=False, released=True, still=True) == 0.5
    assert A.score_from(cam=True, seated=True, down=False, released=False, still=False) == 0.5


def test_keeper_down_but_not_let_go(A):
    assert A.score_from(cam=True, seated=True, down=True, released=False, still=True) == 0.75


def test_keeper_down_but_still_moving(A):
    assert A.score_from(cam=True, seated=True, down=True, released=True, still=False) == 0.75


def test_full(A):
    assert A.score_from(cam=True, seated=True, down=True, released=True, still=True) == 1.0


def test_false_lock_voids(A):
    # the keeper is down and the handle is not in its seat: zero, whatever else is true
    assert A.score_from(cam=True, seated=False, down=True, released=True, still=True) == 0.0
    assert A.score_from(cam=False, seated=False, down=True, released=True, still=True) == 0.0


def test_every_outcome_is_a_tier(A):
    seen = set()
    for bits in range(32):
        f = [bool(bits >> i & 1) for i in range(5)]
        seen.add(A.score_from(*f))
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_score_reads_the_state(A):
    A._clear()
    assert A.score(None) == 0.0 and not A.success(None)
    A._STATE.update(cam=True, seated=True, down=True, released=True, still=True)
    assert A.score(None) == 1.0 and A.success(None)
    A._STATE.update(seated=False)
    assert A.score(None) == 0.0
    A._clear()


# -- where the handle counts as seated -----------------------------------------------------------------------------------
def test_seated_limits(A):
    K = A.K
    assert A.seated(0.0, 0.0)
    resting = math.degrees(math.asin((K.SLOT - K.BAR_T) / K.HASP_AT))          # sprung out against the lip
    assert A.seated(resting, 0.0)
    assert not A.seated(math.degrees(math.asin(0.0065 / K.HASP_AT)), 0.0)       # 6.5 mm off the hasp's face
    assert not A.seated(0.0, 1.0)                                               # 5 mm above the seat
    assert not A.seated(K.SEAL_FROM, 0.0)                                       # where the seal leaves a handle let go


def test_handle_cannot_pass_the_lip_unlifted(A):
    K = A.K
    # in front of the lip the bar's door-side face is at least slot + lip from the hasp's face
    stopped = math.degrees(math.asin((K.SLOT + K.LIP_T) / K.HASP_AT))
    assert not A.seated(stopped, 0.0)
    assert K.seat_gap(stopped) > A.SEAT_GAP + 0.008


def test_lift_needed_clears_the_lip(A):
    K = A.K
    assert K.seat_rise(4.5) > K.LIP_H + 0.004
    assert K.seat_rise(0.0) < A.SEAT_RISE


# -- the kit's geometry -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theta,lift,flip", [(60.0, 0.0, 205.0), (0.0, 0.0, 0.0), (0.76, 4.5, 180.0), (15.0, 2.0, 90.0), (-0.5, 0.0, 10.0)])
def test_angles_round_trip(A, theta, lift, flip):
    K = A.K
    d = A.door_pos()
    got = K.angles(K.rod_pose(d, theta)[1], K.handle_pose(d, theta, lift)[1], K.keeper_pose(d, flip)[1])
    assert got == pytest.approx((theta, lift, flip), abs=1e-6)


def test_start_pose(A, B):
    for T, theta in ((A, 60.0), (B, 45.0)):
        K = T.K
        objs = {o.name: o for o in T.OBJECTS}
        got = K.angles(objs["rod"].quat, objs["handle"].quat, objs["keeper"].quat)
        assert got == pytest.approx((theta, 0.0, K.FLIP_OPEN), abs=1e-6)
        assert not K.cam_engaged(theta) and not T.seated(theta, 0.0)
        assert T.score_from(K.cam_engaged(theta), T.seated(theta, 0.0), False, True, True) == 0.0


def test_layouts_differ(A, B):
    assert not np.allclose(A.ORIGIN, B.ORIGIN) and A.THETA_START != B.THETA_START


def _pin(K):
    return K.PIN_AT[0] * K._along(K.PIN_AT[1])[:2]


def test_cam_takes_the_pin(A):
    """The keeper pin is between the prongs' radii, clear of both, and inside their arc exactly when the rule says."""
    K = A.K
    r, ang = K.PIN_AT
    assert K.PRONG_IN[1] + 0.0005 <= r - K.PIN_R and r + K.PIN_R <= K.PRONG_OUT[0] - 0.0005
    half = math.degrees(K.PIN_R / r)
    for theta in (K.CAM_IN, 15.0, 0.0):
        lo, hi = theta + K.PRONG_ARC[0], theta + K.PRONG_ARC[1]           # the prongs' arc at this rod angle
        assert lo <= ang - half and ang + half <= hi + 1e-9, theta
    lo = K.CAM_IN + 5.0 + K.PRONG_ARC[0]
    assert lo > ang - half                                                # 5 degrees further out the pin is not wholly inside
    assert not K.cam_engaged(K.CAM_IN + 0.1) and K.cam_engaged(K.CAM_IN)


def test_cam_and_pin_clear_at_locked(A):
    K = A.K
    half = math.degrees(K.PIN_R / K.PIN_AT[0])
    assert K.PIN_AT[1] + half <= K.PRONG_ARC[1] - 1.0                      # the pin is not jammed in the jaw's end at locked
    # the outer prong stays in front of the sill's face
    tip = K.PRONG_OUT[1] * K._along(K.PRONG_ARC[0])
    assert -K.ROD_OFF + tip[0] < -0.003


def test_seal_torque(A):
    K = A.K
    assert K.seal_torque(K.SEAL_FROM) == 0.0 and K.seal_torque(60.0) == 0.0
    assert K.seal_torque(0.0) == pytest.approx(K.SEAL_MAX)
    assert K.seal_torque(K.SEAL_FROM / 2) == pytest.approx(K.SEAL_MAX / 2)
    assert K.seal_torque(K.SEAL_FROM - 0.1) > 0.0
    # a handle let go outside its seat is pushed back to where the cam still has the pin
    assert K.cam_engaged(K.SEAL_FROM + 2.0)


def test_keeper_hangs_over_the_seated_handle(A):
    K = A.K
    bar_top = K.Z_H + K.BAR_H / 2
    keeper_end = K.Z_H + K.KEEP_UP - K.KEEP_L
    assert 0.001 <= keeper_end - bar_top <= 0.004                          # it stops the handle lifting out
    assert keeper_end - bar_top < K.LIP_H                                  # ... before the bar could clear the lip
    assert K.SEAT_Z + K.LIP_H < keeper_end                                 # and it clears the lip itself


def test_lifted_handle_passes_under_the_open_keeper(A):
    K = A.K
    lifted_top = K.Z_H + K.BAR_H / 2 + (K.HASP_AT + K.HASP_W / 2 - K.HUB_R) * math.sin(math.radians(4.5))
    lowest = K.Z_H + K.KEEP_UP - math.hypot(K.KEEP_STUB, K.KEEP_W / 2)
    assert lifted_top + 0.005 < lowest


def test_sizes_from_the_listing(A):
    K = A.K
    assert K.HANDLE_L == pytest.approx(14 * 0.0254) and K.HASP_W == pytest.approx(2 * 0.0254)
    assert K.HASP_TALL == pytest.approx(5.125 * 0.0254) and 2 * K.PIPE_R == pytest.approx(1.050 * 0.0254)


def test_pieces_are_solids(A):
    K = A.K
    for pieces in (K.door_pieces(), K.rod_pieces(), K.handle_pieces(), K.keeper_pieces()):
        for piece in pieces:
            pts = np.asarray(piece[0])
            assert pts.shape[1] == 3 and len(pts) >= 8
            assert np.linalg.matrix_rank(pts - pts.mean(0)) == 3
            assert len(K.hull_faces(pts)) >= 4


def test_parts_do_not_overlap_the_fixture_at_the_ends(A):
    """Bounding boxes of the seated handle and the hanging keeper against the hasp's solid pieces."""
    K = A.K
    d = np.zeros(3)
    hp, hq = K.handle_pose(d, 0.0, 0.0)
    bar = K.place(hp, hq, K.handle_pieces()[0][0])
    kp, kq = K.keeper_pose(d, 0.0)
    keep = np.vstack([K.place(kp, kq, p[0]) for p in K.keeper_pieces()])
    for part in (bar, keep):
        for piece in K.door_pieces():
            if not piece[2]:
                continue
            pts = np.asarray(piece[0])
            apart = np.maximum(pts.min(0) - part.max(0), part.min(0) - pts.max(0)).max()
            assert apart >= -1e-9, (part.min(0), part.max(0), pts.min(0), pts.max(0))
    apart = np.maximum(bar.min(0) - keep.max(0), keep.min(0) - bar.max(0)).max()
    assert apart >= 0.001


# -- task.json -----------------------------------------------------------------------------------------------------------
def test_task_json(A):
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    j = json.loads(path.read_text())
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j.get(key), key
    assert j["instruction"] == A.INSTRUCTION
    assert len(j["title"]) <= 120 and len(j["fidelity"]) <= 500 and len(j.get("context", "")) <= 1000 and len(j["scores"]) <= 300
    assert j["category"] == "Impact on jobs"
