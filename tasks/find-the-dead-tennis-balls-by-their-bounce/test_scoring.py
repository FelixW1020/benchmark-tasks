"""Offline checks of the dead-tennis-ball task: the balls, the tray, the bin and the box, the layouts, the scoring
rule and the demo's plan. No simulator.

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

import ballkit as K  # noqa: E402


def load(lot: str):
    os.environ["BALL_LOT"] = lot
    from yam_arena import task_file

    for name in ("task", "demo"):
        sys.modules.pop(name, None)
    return task_file.load(str(HERE / "task.py")).module


@pytest.fixture(params=["A", "B"])
def T(request):
    return load(request.param)


def demo_module():
    sys.modules.pop("demo", None)
    import demo

    return demo


# -- the balls -----------------------------------------------------------------------------------------------------------

def test_ball_is_within_the_rule():
    assert 0.0654 <= K.BALL_D <= 0.0686              # ITF: 6.54 to 6.86 cm
    assert 0.0560 <= K.BALL_MASS <= 0.0594           # ITF: 56.0 to 59.4 g


def test_a_good_ball_comes_back_as_the_rule_says():
    back = K.rebound(K.DROP_ITF, K.BOUNCE_GOOD)
    assert back == pytest.approx(1.41, abs=1e-9)     # the ITF's high-specification ball: 141 cm from 254 cm
    assert K.REBOUND_ITF[0] <= back <= K.REBOUND_ITF[1]


def test_a_flat_ball_fails_the_rule():
    assert K.BOUNCE_DEAD == pytest.approx(0.8 * K.BOUNCE_GOOD)        # 20% less, as measured for a punctured ball
    back = K.rebound(K.DROP_ITF, K.BOUNCE_DEAD)
    assert back == pytest.approx(0.64 * 1.41)
    assert back < K.REBOUND_ITF[0] - 0.40            # 90 cm: far below the 135 cm a ball in play has to reach


def test_rebound_is_a_fixed_share_of_the_drop():
    for drop in (0.1, 0.35, 2.54):
        assert K.rebound(drop, K.BOUNCE_GOOD) / drop == pytest.approx(0.555, abs=0.001)
        assert K.rebound(drop, K.BOUNCE_DEAD) / drop == pytest.approx(0.355, abs=0.001)


def test_seam_lies_on_the_ball_and_closes():
    pts = K.seam_points(64)
    assert np.allclose(np.linalg.norm(pts, axis=1), K.BALL_R)
    steps = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert steps.max() < 0.012                       # no jump where the curve closes
    assert pts[:, 2].max() > 0.5 * K.BALL_R and pts[:, 2].min() < -0.5 * K.BALL_R    # it swings both ways


@pytest.mark.parametrize("dots", [1, 2, 3, 4])
def test_pen_dots_sit_on_both_poles(dots):
    pts = K.dot_points(dots)
    assert len(pts) == 2 * dots
    assert np.allclose(np.linalg.norm(pts, axis=1), K.BALL_R)
    assert (pts[:dots, 2] > 0.9 * K.BALL_R).all() and (pts[dots:, 2] < -0.9 * K.BALL_R).all()


# -- the tray, the bin and the box -------------------------------------------------------------------------------------------

def test_tray_holds_four_balls_in_a_row():
    assert K.TRAY_INNER[0] == pytest.approx(4 * K.TRAY_PITCH)
    assert K.TRAY_INNER[1] > K.BALL_D and K.TRAY_PITCH > K.BALL_D
    slots = [K.tray_slot((0.0, 0.0), 0.0, k) for k in range(4)]
    for a, b in zip(slots, slots[1:]):
        assert np.linalg.norm(b - a) == pytest.approx(K.TRAY_PITCH)
    for s in slots:
        assert K.inside(s, (0.0, 0.0), 0.0, K.TRAY_INNER, margin=K.BALL_R - 1e-9)   # the whole ball is over the floor
        assert s[2] == pytest.approx(K.TRAY_FLOOR + K.BALL_R)


def test_a_ball_can_be_taken_over_the_trays_rim():
    middle = K.TRAY_FLOOR + K.BALL_R
    assert middle - 0.013 > K.TRAY_FLOOR + K.TRAY_RIM + 0.004      # fingertips reach 13 mm below the ball's middle


@pytest.mark.parametrize("boxes,inner,rim", [(K.tray_boxes, K.TRAY_INNER, K.TRAY_RIM), (K.bin_boxes, K.BIN_INNER, K.BIN_RIM)])
def test_open_box_boards(boxes, inner, rim):
    boards = {tag: (np.asarray(c), np.asarray(s)) for tag, c, s in boxes((0.3, -0.1), 0.0)}
    assert set(boards) == {"floor", "wall_a", "wall_b", "wall_c", "wall_d"}
    floor_c, floor_s = boards["floor"]
    assert floor_c[2] - floor_s[2] / 2 == pytest.approx(0.0)       # it stands on the table
    for tag in ("wall_a", "wall_b", "wall_c", "wall_d"):
        c, s = boards[tag]
        assert c[2] - s[2] / 2 == pytest.approx(0.0)
        assert c[2] + s[2] / 2 == pytest.approx(floor_s[2] + rim)
    # the inside is clear: the walls' inner faces are the inner size apart
    assert boards["wall_a"][0][1] - boards["wall_a"][1][1] / 2 - 0.0 == pytest.approx(-0.1 + inner[1] / 2)
    assert boards["wall_c"][0][0] - boards["wall_c"][1][0] / 2 == pytest.approx(0.3 + inner[0] / 2)


def test_two_balls_lie_side_by_side_in_the_bin():
    assert K.BIN_INNER[0] >= 2 * K.BALL_D + 0.01 and K.BIN_INNER[1] >= 2 * K.BALL_D + 0.01
    assert K.BIN_RIM > K.BALL_D                      # a ball on the floor is wholly below the rim


def test_inside():
    assert K.inside((0.30, -0.10, 0.5), (0.3, -0.1), 0.0, K.BIN_INNER)
    assert K.inside((0.374, -0.10, 0.0), (0.3, -0.1), 0.0, K.BIN_INNER)
    assert not K.inside((0.376, -0.10, 0.0), (0.3, -0.1), 0.0, K.BIN_INNER)
    assert not K.inside((0.374, -0.10, 0.0), (0.3, -0.1), 0.0, K.BIN_INNER, margin=0.01)
    turned = ((0.3, -0.1), math.pi / 2, (0.15, 0.10))                # turned a quarter: its long side now runs along y
    assert K.inside((0.30, -0.03, 0.0), *turned) and not K.inside((0.36, -0.10, 0.0), *turned)


# -- the layouts ---------------------------------------------------------------------------------------------------------

def test_objects(T):
    names = [o.name for o in T.OBJECTS]
    assert len(names) == len(set(names)) == 19
    balls = [o for o in T.OBJECTS if isinstance(o, K.Ball)]
    assert [b.name for b in balls] == list(T.BALLS)
    assert [b.dots for b in balls] == [1, 2, 3, 4]
    assert all(b.mass == K.BALL_MASS for b in balls)             # nothing but the bounce tells them apart
    assert sorted(b.name for b in balls if b.bounce == K.BOUNCE_DEAD) == sorted(T.DEAD)
    assert sorted(b.name for b in balls if b.bounce == K.BOUNCE_GOOD) == sorted(T.GOOD)
    assert len(T.DEAD) == len(T.GOOD) == 2


def test_the_two_layouts_have_different_flat_balls():
    a, b = load("A"), load("B")
    assert set(a.DEAD) != set(b.DEAD)
    assert not set(a.DEAD) & set(b.DEAD)             # a ball that is flat in one layout is good in the other
    assert a.TRAY_XY != b.TRAY_XY and a.BIN_XY != b.BIN_XY and a.BOX_XY != b.BOX_XY


def test_balls_start_at_rest_in_the_tray(T):
    for k, name in enumerate(T.BALLS):
        pos = T.ball_start(k)
        assert T.place_of(pos) == "tray"
        assert pos[2] - T.TOP == pytest.approx(K.TRAY_FLOOR + K.BALL_R + T.GAP)
    assert T.score_from(*T.tally({b: "tray" for b in T.BALLS}, {b: True for b in T.BALLS})) == 0.0


def test_tray_bin_and_box_stand_apart(T):
    def outline(xy, inner, wall):
        half = np.array(inner) / 2 + wall
        return np.array(xy) - half, np.array(xy) + half

    boxes = [outline(T.TRAY_XY, K.TRAY_INNER, K.TRAY_WALL), outline(T.BIN_XY, K.BIN_INNER, K.BIN_WALL),
             outline(T.BOX_XY, K.BIN_INNER, K.BIN_WALL)]
    for i, (lo, hi) in enumerate(boxes):
        for lo2, hi2 in boxes[i + 1:]:
            gap = np.maximum(lo2 - hi, lo - hi2).max()
            assert gap > 0.02                        # at least 2 cm of table between any two


# -- where a ball is -----------------------------------------------------------------------------------------------------

def test_place_of(T):
    on_floor = T.TOP + K.BIN_FLOOR + K.BALL_R
    assert T.place_of((*T.BIN_XY, on_floor)) == "bin"
    assert T.place_of((*T.BOX_XY, on_floor)) == "box"
    assert T.place_of((T.BIN_XY[0] + 0.03, T.BIN_XY[1] - 0.03, on_floor + 0.05)) == "bin"      # lying on another ball
    assert T.place_of((*T.BIN_XY, T.TOP + K.BIN_FLOOR + K.BIN_RIM + 0.01)) == "elsewhere"     # held above the rim
    assert T.place_of((T.BIN_XY[0] + K.BIN_INNER[0] / 2 + 0.02, T.BIN_XY[1], T.TOP + K.BALL_R)) == "elsewhere"   # beside it
    assert T.place_of((T.TRAY_XY[0], T.TRAY_XY[1] + 0.18, T.TOP + K.BALL_R)) == "elsewhere"   # on the bare table


def test_tally_counts_only_balls_let_go_and_at_rest(T):
    where = {b: ("bin" if b in T.DEAD else "box") for b in T.BALLS}
    free = {b: True for b in T.BALLS}
    assert T.tally(where, free) == (4, 0)
    free[T.DEAD[0]] = False                          # still in a gripper, or still moving
    assert T.tally(where, free) == (3, 0)
    where[T.GOOD[0]] = "bin"
    assert T.tally(where, free) == (2, 1)
    free[T.GOOD[0]] = False
    assert T.tally(where, free) == (2, 0)


# -- the scoring rule ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("right,wrong,want", [(0, 0, 0.0), (1, 0, 0.25), (2, 0, 0.5), (3, 0, 0.75), (4, 0, 1.0),
                                              (3, 1, 0.0), (2, 2, 0.0), (0, 4, 0.0), (1, 1, 0.0), (0, 1, 0.0)])
def test_score_from(right, wrong, want):
    T = load("A")
    assert T.score_from(right, wrong) == want


def sort_score(T, to_bin, to_box) -> float:
    where = {b: "tray" for b in T.BALLS}
    where.update({b: "bin" for b in to_bin})
    where.update({b: "box" for b in to_box})
    return T.score_from(*T.tally(where, {b: True for b in T.BALLS}))


def test_sorted_right_is_full_score(T):
    assert sort_score(T, T.DEAD, T.GOOD) == 1.0


def test_sorted_the_wrong_way_round_is_zero(T):
    assert sort_score(T, T.GOOD, T.DEAD) == 0.0


def test_everything_in_the_bin_is_zero(T):
    assert sort_score(T, T.BALLS, ()) == 0.0
    assert sort_score(T, (), T.BALLS) == 0.0


def test_binning_the_two_nearest_untested_is_zero_in_both_layouts(T):
    assert sort_score(T, T.BALLS[:2], ()) == 0.0
    assert sort_score(T, T.BALLS[:2], T.BALLS[2:]) == 0.0


def test_no_fixed_sort_scores_in_both_layouts():
    a, b = load("A"), load("B")
    import itertools

    best = 0.0
    for pair in itertools.combinations(a.BALLS, 2):
        rest = tuple(x for x in a.BALLS if x not in pair)
        best = max(best, min(sort_score(a, pair, rest), sort_score(b, pair, rest)))
    assert best == 0.0                               # sorting by position, the same way each time, fails one layout at least


def test_chance_of_a_blind_sort(T):
    import itertools

    scores = [sort_score(T, pair, tuple(x for x in T.BALLS if x not in pair)) for pair in itertools.combinations(T.BALLS, 2)]
    assert sorted(scores) == [0.0] * 5 + [1.0]       # one blind two-and-two sort in six is right; the other five score 0


# -- the demo's plan -------------------------------------------------------------------------------------------------------

def test_drop_spot_is_on_bare_table(T):
    D = demo_module()
    p = D.plan(T)
    spot = p["spot"]
    for xy, inner, wall in ((T.TRAY_XY, K.TRAY_INNER, K.TRAY_WALL), (T.BIN_XY, K.BIN_INNER, K.BIN_WALL), (T.BOX_XY, K.BIN_INNER, K.BIN_WALL)):
        half = np.array(inner) / 2 + wall
        away = np.maximum(np.abs(spot[:2] - np.array(xy)) - half, 0.0)
        assert np.linalg.norm(away) > K.BALL_R + 0.03          # a ball lying there is at least 3 cm from each


def test_the_left_hand_waits_where_a_good_ball_comes_back_to(T):
    D = demo_module()
    p = D.plan(T)
    assert p["hold"][2] - T.TOP == pytest.approx(D.DROP + K.BALL_R)
    assert p["ring"][2] - T.TOP == pytest.approx(0.555 * D.DROP + K.BALL_R, abs=0.001)       # 22.7 cm
    assert np.allclose(p["ring"][:2], p["hold"][:2]) and np.allclose(p["ring"][:2], p["spot"][:2])
    assert D.RING_OPEN - K.BALL_D >= 0.020           # a centimetre each side for the ball to fall through


def test_the_catch_is_timed_for_the_top_of_a_good_balls_bounce(T):
    D = demo_module()
    p = D.plan(T)
    apex = D.apex_time(D.DROP, K.BOUNCE_GOOD)
    assert apex == pytest.approx(0.466, abs=0.002)
    meet = p["lead"] / D.CONTROL_HZ + (D.RING_OPEN - K.BALL_D) / 2 / D.FINGER_SPEED + 0.020 - D.LET_GO_LAG
    assert abs(meet - apex) <= 0.030                 # the pads meet the ball within 30 ms of the top of its bounce ...
    fall = 0.5 * D.G * (meet - apex) ** 2
    assert fall < 0.005                              # ... when it is within 5 mm of its highest


def test_a_flat_ball_is_out_of_the_catch(T):
    D = demo_module()
    p = D.plan(T)
    t = p["lead"] / D.CONTROL_HZ + (D.RING_OPEN - K.BALL_D) / 2 / D.FINGER_SPEED + 0.020 - D.LET_GO_LAG
    fall = math.sqrt(2 * D.DROP / D.G)
    up = K.BOUNCE_DEAD * math.sqrt(2 * D.G * D.DROP)
    since = t - fall
    bottom = up * since - 0.5 * D.G * since ** 2     # where a flat ball's bottom is when the pads close
    top = bottom + K.BALL_D
    pad_low = (p["ring"][2] - T.TOP) - 0.0135        # the lower edge of the pads (27 mm wide at the tool point)
    assert pad_low - top > 0.02                      # the flat ball's top is more than 2 cm below them
    assert D.HELD < K.BALL_D - 0.004 and D.HELD > D.CATCH + 0.004


def test_flat_ball_has_settled_before_it_is_picked_up(T):
    D = demo_module()
    p = D.plan(T)
    fall = math.sqrt(2 * D.DROP / D.G)
    bouncing = fall * (1 + K.BOUNCE_DEAD) / (1 - K.BOUNCE_DEAD)    # until the bounces have died away
    assert p["settle"] + 0.5 + 2.0 > bouncing + 0.5                # the wait, the verdict pause and the hand's way down


def test_balls_are_let_go_inside_the_bin_and_the_box(T):
    D = demo_module()
    p = D.plan(T)
    rim = T.TOP + K.BIN_FLOOR + K.BIN_RIM
    for at in p["over_bin"]:
        assert K.inside(at, T.BIN_XY, 0.0, K.BIN_INNER, margin=K.BALL_R)
        assert at[2] - K.BALL_R > rim + 0.02         # carried in over the rim
    for at in p["in_box"]:
        assert K.inside(at, T.BOX_XY, 0.0, K.BIN_INNER, margin=K.BALL_R)
        assert at[2] - K.BALL_R - (T.TOP + K.BIN_FLOOR) == pytest.approx(D.OVER_FLOOR)
    assert np.linalg.norm(p["over_bin"][0] - p["over_bin"][1]) > K.BALL_D
    assert np.linalg.norm(p["in_box"][0] - p["in_box"][1]) > K.BALL_D


def test_frames_are_right_handed():
    D = demo_module()
    for rot in (D.leaning(0.0, 66.0), D.leaning(-90.0, 90.0), D.leaning(-90.0, 30.0), D.frame((0, 0, -1.0), (0, 1.0, 0))):
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9) and np.linalg.det(rot) == pytest.approx(1.0)
    level = D.leaning(-90.0, 90.0)
    assert np.allclose(level[:, 2], (0.0, -1.0, 0.0), atol=1e-9)      # pointing to the robot's right
    assert abs(level[2, 0]) < 1e-9                                    # closing axis level
    back = D.leaning(0.0, 66.0)
    assert back[2, 2] == pytest.approx(-math.cos(math.radians(66.0))) and back[0, 2] > 0


# -- task.json -----------------------------------------------------------------------------------------------------------

def test_task_json():
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    d = json.loads(path.read_text())
    T = load("A")
    assert d["instruction"] == T.INSTRUCTION
    assert d["category"] in ("Impact on jobs", "Impact on AI timelines", "Safety and alignment", "Societal relatability", "Other")
    assert len(d["title"]) <= 120 and len(d["fidelity"]) <= 500 and len(d["context"]) <= 1000 and len(d["scores"]) <= 300
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert d.get(key)
