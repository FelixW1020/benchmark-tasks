"""Offline checks of the compass-calibration task: the scoring rule, the turn counter, the drone's shape, both layouts
and the demo's grasps.

    python -m pytest -q test_scoring.py        (needs yam_arena importable; no simulator is started)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(lot: str):
    from yam_arena import task_file

    os.environ["DRONE_LOT"] = lot
    sys.modules.pop("task", None)
    return task_file.load(str(HERE / "task.py")).module


@pytest.fixture(params=["A", "B"])
def T(request):
    return load(request.param)


@pytest.fixture
def A():
    return load("A")


def att(nose=0.0, top=0.0, up_tilt=0.0, plumb_tilt=90.0):
    return {"nose": nose % 360.0, "top": top % 360.0, "up_tilt": up_tilt, "plumb_tilt": plumb_tilt}


def level_turn(t, start=0.0, sweep=380.0, low=0.10, tilt=0.0, step=4.0):
    a = 0.0
    while abs(a) <= abs(sweep):
        t.step(low, 0.05, att(nose=start + a, up_tilt=tilt))
        a += math.copysign(step, sweep)


def upright_turn(t, start=0.0, sweep=380.0, low=0.08, tilt=0.0, step=4.0):
    a = 0.0
    while abs(a) <= abs(sweep):
        t.step(low, 0.05, att(top=start + a, up_tilt=90.0, plumb_tilt=tilt))
        a += math.copysign(step, sweep)


# -- the scoring rule ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level, both, dropped, stands, pad, clear, want", [
    (False, False, False, True, True, True, 0.0),      # nothing done: the drone still on its pad
    (False, False, False, False, False, False, 0.0),
    (True, False, False, True, True, True, 0.25),      # the level turn only
    (True, False, True, False, False, True, 0.25),
    (True, True, False, False, False, False, 0.5),     # both turns, still in the hand at the end
    (True, True, True, True, True, True, 0.5),         # both turns, then dropped (even if it landed on the pad)
    (True, True, False, False, True, True, 0.5),       # both turns, lying on its side on the pad
    (True, True, False, True, False, True, 0.75),      # stood on the table beside the pad
    (True, True, False, True, True, False, 0.75),      # on the pad with a gripper still on it
    (True, True, False, True, True, True, 1.0),
])
def test_score_ladder(A, level, both, dropped, stands, pad, clear, want):
    assert A.score_from(level, both, dropped, stands, pad, clear) == want


def test_every_outcome_is_scored(A):
    seen = set()
    for bits in range(64):
        f = [bool(bits >> k & 1) for k in range(6)]
        if f[1] and not f[0]:
            continue                                   # both turns includes the level turn
        seen.add(A.score_from(*f))
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_start_scores_nothing(T):
    pos, quat = T.drone_start()
    low = T.K.lowest(pos, quat) - T.TOP
    t = T.Turns()
    for _ in range(50):
        t.step(low, 0.0, T.K.attitude(quat))
    feet = [T.K.place(pos, quat, f)[:2] for f in T.K.FEET]
    assert not t.level_done and not t.held and not t.dropped
    assert T.standing(low, T.K.attitude(quat)["up_tilt"], 0.0, 0.0) and T.on_pad(feet)
    assert T.score_from(t.level_done, t.both_done, t.dropped, True, True, True) == 0.0
    assert T.score(None) == 0.0 and not T.success(None)


# -- the turn counter --------------------------------------------------------------------------------------------------
def test_both_turns_in_one_hold(A):
    t = A.Turns()
    level_turn(t)
    assert t.level_done and not t.both_done
    upright_turn(t)
    assert t.both_done and not t.dropped


@pytest.mark.parametrize("sweep", [380.0, -380.0])
def test_either_way_round_counts(A, sweep):
    t = A.Turns()
    level_turn(t, start=137.0, sweep=sweep)
    upright_turn(t, start=20.0, sweep=-sweep)
    assert t.both_done


def test_a_turn_short_of_full_does_not_count(A):
    t = A.Turns()
    level_turn(t, sweep=340.0)
    assert not t.level_done and len(t.level) == 35


def test_turning_a_wrist_back_and_forth_does_not_count(A):
    t = A.Turns()
    for _ in range(4):
        level_turn(t, sweep=230.0)
        level_turn(t, start=230.0, sweep=-230.0)
    assert not t.level_done and len(t.level) == 24


def test_spinning_it_on_the_table_does_not_count(A):
    t = A.Turns()
    level_turn(t, low=A.PAD_T + 0.0005)
    level_turn(t, low=0.015)
    assert not t.level_done and not t.held and len(t.level) == 0


@pytest.mark.parametrize("tilt, counts", [(0.0, True), (15.0, True), (19.9, True), (20.1, False), (30.0, False)])
def test_level_means_within_twenty_degrees(A, tilt, counts):
    t = A.Turns()
    level_turn(t, tilt=tilt)
    assert t.level_done is counts


@pytest.mark.parametrize("tilt, counts", [(0.0, True), (19.9, True), (20.1, False), (45.0, False)])
def test_upright_means_within_twenty_degrees_of_plumb(A, tilt, counts):
    t = A.Turns()
    level_turn(t)
    upright_turn(t, tilt=tilt)
    assert t.both_done is counts


def test_tipping_instead_of_turning_gains_nothing(A):
    t = A.Turns()
    for k in range(200):
        t.step(0.10, 0.05, att(nose=40.0, up_tilt=10.0 * math.sin(k / 5.0)))
    assert len(t.level) == 1


def test_putting_it_down_mid_turn_starts_the_turn_again(A):
    t = A.Turns()
    level_turn(t, sweep=200.0)
    t.step(0.004, 0.02, att(nose=200.0))               # stood on the table to change grip
    assert len(t.level) == 0 and not t.held
    level_turn(t, start=200.0, sweep=200.0)
    assert not t.level_done


def test_upright_turn_has_to_follow_the_level_turn_in_the_same_hold(A):
    t = A.Turns()
    upright_turn(t)                                     # upright first: nothing
    assert not t.level_done and len(t.plumb) == 0
    level_turn(t)
    assert t.level_done
    t.step(0.004, 0.02, att())                          # put down between the turns
    upright_turn(t)
    assert t.level_done and not t.both_done             # the level turn keeps its 0.25; the upright turn does not count
    level_turn(t)
    upright_turn(t)
    assert t.both_done


def test_a_drop_is_remembered(A):
    t = A.Turns()
    level_turn(t)
    upright_turn(t)
    t.step(0.012, 1.2, att(up_tilt=40.0))               # falling as it comes within 2 cm of the table
    assert t.both_done and t.dropped
    assert A.score_from(t.level_done, t.both_done, t.dropped, True, True, True) == 0.5


def test_setting_it_down_gently_is_not_a_drop(A):
    t = A.Turns()
    level_turn(t)
    upright_turn(t)
    for low in (0.05, 0.03, 0.019, 0.008, 0.0045):
        t.step(low, 0.06, att())
    assert not t.dropped and t.both_done


def test_standing_and_pad(A):
    assert A.standing(A.PAD_T + 0.0004, 0.3, 0.0, 0.0)             # on the pad
    assert A.standing(0.0005, 2.0, 0.0, 0.0)                       # on the table
    assert not A.standing(0.060, 0.0, 0.0, 0.0)                    # in the air
    assert not A.standing(A.PAD_T, 80.0, 0.0, 0.0)                 # on its side
    assert not A.standing(A.PAD_T, 0.0, 0.05, 0.0)                 # still moving
    px, py = A.PAD_XY
    feet = lambda dx, dy, yaw: [np.array([px + dx, py + dy]) + (A.K._rz(yaw) @ f)[:2] for f in A.K.FEET]   # noqa: E731
    assert A.on_pad(feet(0.0, 0.0, 0.0)) and A.on_pad(feet(0.0, 0.0, math.radians(45.0)))
    assert not A.on_pad(feet(0.06, 0.0, 0.0))                      # one pair of feet off the edge


# -- the drone -----------------------------------------------------------------------------------------------------------
def test_drone_matches_the_published_sizes(A):
    K = A.K
    c = K.corners()
    lo, hi = c.min(0), c.max(0)
    fus = c[:16]                                                    # the fuselage and the camera block
    assert fus[:, 0].max() - fus[:, 0].min() == pytest.approx(0.138, abs=1e-6)       # folded length
    assert hi[1] - lo[1] == pytest.approx(0.289, abs=1e-6)          # unfolded width across the propellers
    assert hi[2] - lo[2] == pytest.approx(0.056, abs=5e-4)          # unfolded height
    assert K.MASS == 0.249 and K.MASS < 0.300
    assert -lo[2] == pytest.approx(K.STAND, abs=1e-9)
    assert np.allclose(K.FEET[:, 2], lo[2])


def test_drone_stands_on_four_feet_with_its_weight_between_them(A):
    K = A.K
    assert len(K.FEET) == 4
    xs, ys = K.FEET[:, 0], K.FEET[:, 1]
    vol = np.array([s[0] * s[1] * s[2] for _, _, s, _, _ in K.boards()])
    com = (np.array([c for _, c, _, _, _ in K.boards()]) * vol[:, None]).sum(0) / vol.sum()
    assert xs.min() + 0.02 < com[0] < xs.max() - 0.02 and abs(com[1]) < 1e-9
    assert abs(com[0]) < 0.010                                      # the side grasp at the middle is under the weight


def test_attitude(A):
    import demo as D

    K = A.K
    a = K.attitude(K._quat(D.drone_rot(40.0, 0.0)))
    assert a["nose"] == pytest.approx(40.0) and a["up_tilt"] == pytest.approx(0.0, abs=1e-6) and a["plumb_tilt"] == pytest.approx(90.0)
    a = K.attitude(K._quat(D.drone_rot(-70.0, 90.0)))               # hanging nose down, top facing -70
    assert a["plumb_tilt"] == pytest.approx(0.0, abs=1e-5) and a["top"] == pytest.approx(290.0) and a["up_tilt"] == pytest.approx(90.0)
    a = K.attitude(K._quat(D.drone_rot(10.0, 30.0)))
    assert a["up_tilt"] == pytest.approx(30.0) and a["plumb_tilt"] == pytest.approx(60.0)


def test_lowest_point(A):
    import demo as D

    K = A.K
    pos = np.array([0.2, 0.0, 1.0])
    assert K.lowest(pos, K._quat(D.drone_rot(0.0, 0.0))) == pytest.approx(1.0 - K.STAND)
    assert K.lowest(pos, K._quat(D.drone_rot(0.0, 90.0))) == pytest.approx(1.0 - (K.MOTOR_X + K.MOTOR / 2))


def test_drone_distance(A):
    K = A.K
    pos, quat = np.zeros(3), (0.0, 0.0, 0.0, 1.0)
    assert K.drone_distance([[0.0, 0.0, 0.10]], pos, quat) == pytest.approx(0.10 - K.TOP_Z)
    assert K.drone_distance([[0.0, K.WID / 2 + 0.03, 0.0]], pos, quat) == pytest.approx(0.03)
    assert K.drone_distance([[K.MOTOR_X, K.SPAN / 2 + 0.02, K.TOP_Z + 0.0065]], pos, quat) == pytest.approx(0.02)


# -- the layouts -----------------------------------------------------------------------------------------------------
def test_layout(T):
    pos, quat = T.drone_start()
    K = T.K
    assert K.lowest(pos, quat) - T.TOP == pytest.approx(T.PAD_T + T.GAP)
    feet = [K.place(pos, quat, f)[:2] for f in K.FEET]
    assert T.on_pad(feet)
    assert K.attitude(quat)["nose"] == pytest.approx(math.degrees(T.YAW) % 360.0)
    names = [o.name for o in T.OBJECTS]
    assert names == [T.PAD, T.DRONE]
    pad = T.OBJECTS[0]
    assert pad.mass is None and pad.size == (T.PAD_SIZE, T.PAD_SIZE, T.PAD_T)
    assert T.OBJECTS[1].mass == K.MASS


def test_the_two_layouts_are_not_the_same():
    a, b = load("A"), load("B")
    assert a.PAD_XY != b.PAD_XY and a.YAW != b.YAW
    assert a.INSTRUCTION == b.INSTRUCTION


def test_pad_is_smaller_than_the_drone_but_holds_its_feet(A):
    K = A.K
    assert A.PAD_SIZE < K.SPAN
    reach = max(float(np.hypot(f[0], f[1])) for f in K.FEET)
    assert reach < A.PAD_SIZE / 2                                    # whichever way it faces, centred, its feet are on


# -- the demo's grasps ---------------------------------------------------------------------------------------------------
def _finger_boxes(D, g, opened: bool):
    """Rough boxes for the two fingers of a hand with grasp ``g``, in the drone's frame: (centre, half sizes)."""
    rot, at, wide, closed = g
    half = (wide if opened else closed) / 2
    out = []
    for s in (-1.0, 1.0):
        # a finger: from 13 mm past the tool point back 60 mm; 8 mm thick outside its pad; 16 mm wide at its end
        c_tool = np.array([s * (half + 0.004), 0.0, 0.013 - 0.030])
        out.append((at + rot @ c_tool, np.abs(rot @ np.array([0.004, 0.008, 0.030]))))
    return out


@pytest.mark.parametrize("kind, s, a", [("side", 1, 1), ("side", -1, 1), ("tail", 1, 1), ("tail", -1, 1),
                                         ("mid_y", 1, 1), ("mid_y", 1, -1), ("mid_z", 1, 1), ("mid_z", 1, -1)])
def test_a_grasp_holds_the_fuselage_and_touches_nothing_else(A, kind, s, a):
    import demo as D

    K = A.K
    g = D.grasp(kind, s, a)
    rot, at, wide, closed = g
    assert np.allclose(rot.T @ rot, np.eye(3)) and np.linalg.det(rot) == pytest.approx(1.0)
    faces = {"side": K.WID, "mid_z": K.WID, "tail": K.HGT, "mid_y": K.HGT}[kind]
    assert closed < faces < wide <= D.FULL_OPEN                     # closes on the faces, opens clear of them
    closing = np.abs(rot[:, 0])
    assert closing.argmax() == (1 if faces == K.WID else 2)         # the pads close across the width, or the height
    # opened, the fingers touch nothing; the parts that are not fuselage stay clear when closed too
    for opened in (True, False):
        for centre, halfsize in _finger_boxes(D, g, opened):
            for tag, c, size, _, held in K.boards():
                if held and not opened:
                    continue
                gapv = np.abs(centre - np.asarray(c)) - halfsize - np.asarray(size) / 2
                assert gapv.max() > 0.0005, (kind, tag, opened)


def test_tail_and_side_grasps_do_not_share_a_face(A):
    import demo as D

    side, tail = D.grasp("side", 1), D.grasp("tail", 1)
    assert np.abs(side[0][:, 0]).argmax() != np.abs(tail[0][:, 0]).argmax()
    assert abs(tail[1][0]) > 0.040                                    # the tail grasp is clear of the middle


def test_expected_pose_and_tool(A):
    import demo as D

    est = D.Est(A)
    assert est.yaw == pytest.approx(math.degrees(A.YAW)) and est.pitch == 0.0
    g = D.grasp("side", 1)
    pos, rot = est.tool(g)
    assert np.allclose(pos, est.pos) and rot[2, 2] == pytest.approx(-1.0)        # straight down over the middle
    g = D.grasp("tail", 1)
    pos, rot = est.tool(g, yaw=0.0)
    assert np.allclose(rot[:, 2], [1.0, 0.0, 0.0]) and pos[0] == pytest.approx(est.pos[0] - (A.K.LEN / 2 - D.TAIL_IN))
    pos, rot = est.tool(g, yaw=0.0, pitch=90.0)                                  # hanging nose down: from above
    assert rot[2, 2] == pytest.approx(-1.0) and pos[2] == pytest.approx(est.pos[2] + A.K.LEN / 2 - D.TAIL_IN)


def test_a_half_turn_back_lands_on_the_same_faces(A):
    import demo as D

    g = D.grasp("side", 1)
    g2 = (g[0] @ D._rz(math.pi), g[1], g[2], g[3])
    assert np.allclose(g2[0][:, 2], g[0][:, 2]) and np.allclose(g2[0][:, 0], -g[0][:, 0])


# -- the description -----------------------------------------------------------------------------------------------------
def test_task_json(A):
    d = json.loads((HERE / "task.json").read_text())
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert d.get(key), key
    assert d["category"] in ("Impact on jobs", "Impact on AI timelines", "Safety and alignment", "Societal relatability", "Other")
    assert len(d["title"]) <= 120 and len(d["fidelity"]) <= 500 and len(d.get("context", "")) <= 1000 and len(d["scores"]) <= 300
    assert d["instruction"] == A.INSTRUCTION
    assert "3,000 steps" in d["run_limit"]
    setup = d["setup"]
    for lot in ("A", "B"):
        L = load(lot)
        assert f"({L.PAD_XY[0] * 100:g}, {L.PAD_XY[1] * 100:g})" in setup, lot
        assert f"{math.degrees(L.YAW) % 360:g} degrees" in setup, lot
    for number in ("249 g", "13.8", "5.8", "4", "28.9", "5.6", "24 x 24"):
        assert number in setup, number
    for word in ("36", "20 degrees", "2 cm", "0.25 m/s", "3 cm", "0.25", "0.5", "0.75", "1.0"):
        assert word in d["scoring"], word
    text = json.dumps(d).lower()
    for brand in ("dji", "mavic", "mini 2"):
        assert brand not in text
