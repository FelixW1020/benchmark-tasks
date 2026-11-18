"""Checks of the transfer-forceps task's shapes, contamination record and scoring rule. No simulator needed:

    python -m pytest -q test_scoring.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yam_arena import task_file  # noqa: E402

T = task_file.load(HERE / "task.py").module
K = T.K
I = (0.0, 0.0, 0.0, 1.0)
FAR = np.array([[0.0, 0.0, T.TOP + 0.40]])                # an arm point nowhere near anything
AT_REST = {"forceps": 0.0, "cup_a": 0.0, "cup_b": 0.0}


def blades(tail=None, yaw: float = 0.0, pitch: float = 0.0, gap: float = K.F_OPEN):
    """The two blades' poses for forceps whose tail is at ``tail``, turned ``yaw`` about z and tipped ``pitch`` rad
    nose down, with the tips ``gap`` apart."""
    tail = T.tail_start() if tail is None else np.asarray(tail, float)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    base = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    out = []
    for side in (1.0, -1.0):
        a = side * gap / K.F_LEN / 2
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        out.append((tail, K._quat(base @ rz)))
    return tuple(out)


def cups(**moved) -> dict:
    """Both cups where they start, except those given as name=(x, y, height of the base above the table)."""
    out = {n: (T.cup_start(i), I) for i, n in enumerate(T.CUPS)}
    for n, (x, y, z) in moved.items():
        out[n] = (np.array([x, y, T.TOP + z]), I)
    return out


def step(s=None, arm=FAR, bl=None, cp=None, speeds=AT_REST, n: int = 1) -> dict:
    s = T.new_state() if s is None else s
    for _ in range(n):
        T.advance(s, arm, blades() if bl is None else bl, cups() if cp is None else cp, speeds)
    return s


def total(s) -> float:
    return T.score_from(not s["dirty"]["field"], s["cups_on"], s["down"], s["ready"])


ON_FIELD = {"cup_a": (0.392, 0.035, T.FIELD_TOP), "cup_b": (0.392, -0.045, T.FIELD_TOP)}


# -- shapes ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pieces", [lambda: K.blade_pieces(1.0), lambda: K.blade_pieces(-1.0), K.cup_pieces])
def test_every_piece_is_a_solid_convex_hull(pieces):
    for pts, color in pieces():
        assert 4 <= len(pts) <= 64 and len(color) == 3
        faces = K.hull_faces(pts)
        assert len(faces) >= 4
        for f in faces:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            assert ((pts - a) @ np.cross(b - a, c - a)).max() < 1e-9      # every other point is behind the face
        assert np.ptp(pts, axis=0).min() > 0.0005                         # not flat


def test_forceps_are_the_stated_size():
    pts = np.vstack([p for p, _ in K.blade_pieces(1.0)])
    assert np.ptp(pts[:, 0]) == pytest.approx(0.140)
    assert np.ptp(pts[:, 2]) == pytest.approx(0.011)
    f = K.forceps_points(*blades()[0], *blades()[1])
    assert f["gap"] == pytest.approx(K.F_OPEN, abs=1e-4) and f["up"] == pytest.approx(0.0, abs=1e-9)
    assert K.forceps_points(*blades(gap=0.0)[0], *blades(gap=0.0)[1])["gap"] == pytest.approx(0.0, abs=1e-9)
    # with the tips together the bowed blades still stand apart where the pads hold them: 8 mm over both blades
    assert 2 * (K.bow(K.F_BOW_AT) + K.F_TH) == pytest.approx(0.008)
    assert K.WORK + K.HANDLE < K.F_LEN


def test_spring_gives_the_stated_squeeze():
    assert K.F_STIFF * K.F_ANGLE / K.F_GRIP == pytest.approx(K.F_CLOSE_N)


def test_cup_is_the_stated_size_and_hollow():
    pts = np.vstack([p for p, _ in K.cup_pieces()])
    assert np.ptp(pts[:, 0]) == pytest.approx(0.051, abs=0.0005) and pts[:, 2].max() == pytest.approx(0.0445)
    inside = np.array([[0.0, 0.0, 0.02]])
    for p, _ in K.cup_pieces()[1:]:                                       # no wall piece reaches the axis
        assert np.hypot(p[:, 0], p[:, 1]).min() > 0.015
    assert K.in_cup(inside, (0, 0, 0), I, 0.0)[0] and not K.in_cup([[0.04, 0, 0.02]], (0, 0, 0), I, 0.004)[0]


def test_arm_shapes_are_read_from_the_robots_urdf():
    links = K.link_points()
    assert set(links) == set(K.ARM_LINKS) and all(len(v) > 30 for v in links.values())
    assert np.ptp(links["gripper"], axis=0).max() < 0.2


# -- the layout --------------------------------------------------------------------------------------------------------

def test_layout_starts_sterile_and_scores_nothing():
    assert T.LOT in ("A", "B")
    s = step(n=3)
    assert not any(s["dirty"].values()) and s["cups_on"] == 0 and total(s) == 0.0
    f = K.forceps_points(*blades()[0], *blades()[1])
    assert np.all(T.over(f["work"], "field")) and not np.any(T.over(f["handle"], "field"))
    assert T.touches(f["work"], "field")                                  # the working end rests on the field
    assert T.NEAR - f["tail"][0] == pytest.approx(T.OVERHANG)             # the handle sticks out over the edge
    for i in range(2):
        c = T.cup_start(i)
        assert np.all(T.over(K.cup_points(c, I), "pack"))
    assert s["down"] and s["ready"]                                       # the forceps start where they should end


def test_resting_arms_are_clear_of_both_sterile_areas():
    # the arms' rest pose reaches x = 0.337 m at most (the fingertips); both sterile parts start beyond that
    assert T.FIELD_IN[0] > 0.337 + 0.015 and T.PACK_IN[0] > 0.337 + 0.015


def test_demo_places_keep_the_hand_in_front_of_the_field():
    import demo as D

    lever = K.F_LEN - D.PINCH
    reach = lever * math.cos(D.L_PICK - D.L_WORK)                         # the tips' distance ahead of the tool point
    for sx, sy in D.SLOTS:
        assert T.FIELD_IN[0] + K.C_TOP_R + 0.008 <= sx and T.FIELD_IN[2] + K.C_TOP_R + 0.02 <= sy <= T.FIELD_IN[3] - K.C_TOP_R
        assert sx + D.PAST - reach + 0.014 < T.FIELD_IN[0] - 0.015        # fingertips (1.4 cm ahead of the tool point)
    assert abs(D.SLOTS[0][1] - D.SLOTS[1][1]) > 2 * K.C_TOP_R + 0.02
    assert math.degrees(D.L_PICK - D.L_WORK) == pytest.approx(44.0)
    assert D.pads_for(K, D.TIPS_HOLD) < D.pads_for(K, D.TIPS_OPEN) < 2 * (K.F_BOW + K.F_TH) + K.F_OPEN * D.PINCH / K.F_LEN


# -- the contamination record ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("area,xy", [("field", (0.40, 0.0)), ("pack", (0.40, -0.40))])
def test_an_arm_above_a_sterile_area_contaminates_it(area, xy):
    s = step(arm=np.array([[xy[0], xy[1], T.TOP + 0.30]]))
    assert s["dirty"][area] and s["arm_over"]
    other = "pack" if area == "field" else "field"
    assert not s["dirty"][other]


def test_an_arm_above_a_border_contaminates_nothing():
    s = step(arm=np.array([[T.NEAR + 0.020, 0.0, T.TOP + 0.05], [T.NEAR + 0.020, -0.40, T.TOP + 0.05]]))
    assert not any(s["dirty"].values())


def test_field_contaminated_scores_zero_whatever_stands_on_it():
    s = step(cp=cups(**ON_FIELD), n=2)
    assert s["cups_on"] == 2 and total(s) == 1.0
    step(s, arm=np.array([[0.50, 0.20, T.TOP + 0.30]]), cp=cups(**ON_FIELD))
    step(s, cp=cups(**ON_FIELD))
    assert s["dirty"]["field"] and s["dirty"]["cup_a"] and s["dirty"]["forceps"] and total(s) == 0.0


def test_pack_contaminated_takes_the_cups_still_on_it():
    s = step(arm=np.array([[0.45, -0.40, T.TOP + 0.20]]))
    assert s["dirty"]["pack"] and s["dirty"]["cup_a"] and s["dirty"]["cup_b"] and not s["dirty"]["field"]
    step(s, cp=cups(**ON_FIELD))                                          # ... and they then contaminate the field
    assert s["dirty"]["field"] and total(s) == 0.0


def test_tips_above_the_handle_contaminate_held_forceps():
    up = T.tail_start() + (-0.10, 0.0, 0.08)                               # held in the air in front of the field
    hand = np.array([up + (0.03, 0.012, 0.0)])                             # a gripper point at the handle
    assert not step(arm=hand, bl=blades(tail=up, pitch=math.radians(20)))["dirty"]["forceps"]     # nose down
    assert not step(arm=hand, bl=blades(tail=up, pitch=math.radians(-1.5)))["dirty"]["forceps"]   # 3.7 mm up: allowed
    s = step(arm=hand, bl=blades(tail=up, pitch=math.radians(-4.0)))                              # 9.8 mm up
    assert s["dirty"]["forceps"] and "tips raised" in s["why"]["forceps"]
    # forceps that have been let go may rock on the tray's edge: the rule is about how they are held
    rocking = blades(tail=T.tail_start() + (0.0, 0.0, -0.004), pitch=math.radians(-4.0))
    assert not step(bl=rocking)["dirty"]["forceps"]


def test_working_end_on_the_table_or_a_border_contaminates_the_forceps():
    on_table = np.array([0.20, 0.0, T.TOP + 0.006])
    assert step(bl=blades(tail=on_table))["dirty"]["forceps"]
    # lying along the field's near border, 1 cm in from the edge
    border = np.array([T.NEAR + 0.010, 0.10, T.TOP + T.FIELD_TOP + 0.006])
    assert step(bl=blades(tail=border, yaw=math.pi / 2))["dirty"]["forceps"]
    # the same lying wholly inside the border: the handle on the field contaminates the field instead
    inside = np.array([T.NEAR + 0.080, 0.10, T.TOP + T.FIELD_TOP + 0.006])
    s = step(bl=blades(tail=inside, yaw=math.pi / 2))
    assert s["dirty"]["field"] and "handle" in s["why"]["field"]


def test_an_arm_on_the_working_end_contaminates_the_forceps():
    f = K.forceps_points(*blades()[0], *blades()[1])
    near_tip = f["tip"] + (0.0, 0.0, 0.006)                                # over the field, though: the field goes too
    s = step(arm=np.array([near_tip]))
    assert s["dirty"]["forceps"] and s["dirty"]["field"]
    by_handle = f["tail"] + (0.03, 0.012, 0.0)
    assert not any(step(arm=np.array([by_handle]))["dirty"].values())      # holding the handle is what it is for


def test_contaminated_forceps_contaminate_what_they_touch():
    s = T.new_state()
    T._soil(s, "forceps", "test")
    c = T.cup_start(0)
    tail = c + (-K.F_LEN, K.C_TOP_R - 0.002, 0.08)                         # nose down into cup_a's side wall
    bl = blades(tail=tail, pitch=math.asin(0.045 / K.F_LEN), gap=0.003)
    step(s, bl=bl)
    assert s["dirty"]["cup_a"] and "contaminated forceps" in s["why"]["cup_a"]
    # the contaminated cup stands on the pack: the pack goes, and the other cup on it; the field is untouched
    assert s["dirty"]["pack"] and s["dirty"]["cup_b"] and not s["dirty"]["field"]
    clean = step(bl=bl)
    assert not any(clean["dirty"].values())                                # clean forceps on a clean cup: nothing


def test_cup_on_the_table_or_a_border_is_contaminated():
    assert step(cp=cups(cup_a=(0.25, -0.20, 0.0005)))["dirty"]["cup_a"]
    s = step(cp=cups(cup_a=(T.NEAR + 0.030, 0.0, T.FIELD_TOP)))            # its base half on the field's border
    assert s["dirty"]["cup_a"] and not s["on"]["cup_a"]
    s = step(cp=cups(cup_a=(T.NEAR + 0.25, 0.0, T.FIELD_TOP + 0.06)))      # carried over the field: touches nothing
    assert not s["dirty"]["cup_a"]


def test_cup_against_the_trays_side_is_contaminated():
    s = step(cp=cups(cup_a=(T.NEAR - K.C_BASE_R - 0.0005, 0.0, 0.005)))
    assert s["dirty"]["cup_a"]


def test_an_arm_at_a_cup_contaminates_it():
    held = (0.25, -0.20, 0.10)                                             # a cup in the air in front of the pack
    c = np.array([held[0], held[1], T.TOP + held[2]])
    s = step(arm=np.array([c + (-K.C_TOP_R - 0.003, 0.0, 0.04)]), cp=cups(cup_b=held))
    assert s["dirty"]["cup_b"] and "arm" in s["why"]["cup_b"]
    assert not any(v for k, v in s["dirty"].items() if k != "cup_b")
    assert not step(arm=np.array([c + (-K.C_TOP_R - 0.03, 0.0, 0.04)]), cp=cups(cup_b=held))["dirty"]["cup_b"]
    inside = step(arm=np.array([c + (0.0, 0.0, 0.03)]), cp=cups(cup_b=held))      # a fingertip inside the cup
    assert inside["dirty"]["cup_b"]


# -- what counts as on the field ---------------------------------------------------------------------------------------

def test_cup_must_be_wholly_inside_the_border():
    x0 = T.FIELD_IN[0]
    s = step(cp=cups(cup_a=(x0 + K.C_TOP_R + 0.002, 0.0, T.FIELD_TOP)))
    assert s["on"]["cup_a"] and total(s) == 0.25
    s = step(cp=cups(cup_a=(x0 + K.C_BASE_R + 0.002, 0.0, T.FIELD_TOP)))  # base inside, rim over the border
    assert not s["on"]["cup_a"] and not s["dirty"]["cup_a"] and total(s) == 0.0


def test_cup_must_be_upright_let_go_and_at_rest():
    on = cups(cup_a=ON_FIELD["cup_a"])
    assert step(cp=on)["on"]["cup_a"]
    assert not step(cp=on, speeds={**AT_REST, "cup_a": 0.05})["on"]["cup_a"]
    tipped = dict(on)
    tipped["cup_a"] = (on["cup_a"][0], (math.sin(math.radians(6)), 0.0, 0.0, math.cos(math.radians(6))))   # 12 degrees
    assert not step(cp=tipped)["on"]["cup_a"]
    c = on["cup_a"][0]
    tail = c + (-K.F_LEN, K.C_TOP_R - 0.002, 0.08)
    held = blades(tail=tail, pitch=math.asin(0.045 / K.F_LEN), gap=0.003)  # the tips still on its wall
    assert not step(cp=on, bl=held)["on"]["cup_a"]
    lifted = cups(cup_a=(ON_FIELD["cup_a"][0], ON_FIELD["cup_a"][1], T.FIELD_TOP + 0.02))
    assert not step(cp=lifted)["on"]["cup_a"]


def test_forceps_must_be_let_go_and_left_on_a_sterile_area():
    both = cups(**ON_FIELD)
    assert total(step(cp=both)) == 1.0
    f = K.forceps_points(*blades()[0], *blades()[1])
    held = np.array([f["tail"] + (0.03, 0.012, 0.0)])
    s = step(cp=both, arm=held)
    assert not s["down"] and total(s) == 0.5
    assert total(step(cp=both, speeds={**AT_REST, "forceps": 0.1})) == 0.5
    on_table = blades(tail=np.array([0.20, 0.0, T.TOP + 0.006]))
    s = step(cp=both, bl=on_table)
    assert s["down"] and not s["ready"] and total(s) == 0.75
    on_pack = blades(tail=np.array([T.NEAR - 0.05, -0.30, T.TOP + T.PACK_TOP + 0.006]))
    assert total(step(cp=both, bl=on_pack)) == 1.0                        # the pack's sterile part will do as well


# -- the scoring rule --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("clean,n,down,ready,want", [
    (True, 0, True, True, 0.0), (False, 2, True, True, 0.0), (False, 1, False, False, 0.0),
    (True, 1, False, False, 0.25), (True, 1, True, True, 0.25),
    (True, 2, False, False, 0.5), (True, 2, True, False, 0.75), (True, 2, True, True, 1.0)])
def test_score_rule(clean, n, down, ready, want):
    assert T.score_from(clean, n, down, ready) == want


def test_score_only_reads_the_trial_state():
    T._clear()
    assert T.score({}) == 0.0 and not T.success({})
    T._STATE.update(cups_on=2, down=True, ready=True)
    assert T.score({}) == 1.0 and T.success({})
    T._STATE["dirty"]["field"] = True
    assert T.score({}) == 0.0
    T._clear()


def test_text_matches_code():
    doc = json.loads((HERE / "task.json").read_text())
    assert doc["instruction"] == T.INSTRUCTION
    s = doc["scoring"]
    for phrase in ("1.5 mm", "4 mm", "2.5 cm", "5 cm", "6.5 cm", "5 mm", "10 degrees", "3 cm", "2 cm", "0.25", "0.5", "0.75", "1.0"):
        assert phrase in s, phrase
    assert (T.TOUCH, T.NEAR_ARM, T.BORDER, K.WORK, K.HANDLE, T.TIPS_UP, T.CLEAR, T.TIP_CLEAR, T.STILL) == \
        (0.0015, 0.004, 0.025, 0.050, 0.065, 0.005, 0.030, 0.020, 0.02)
    assert math.degrees(T.UPRIGHT) == pytest.approx(10.0)
    assert "1,800 steps" in doc["run_limit"]
    assert len(doc["fidelity"]) <= 500 and len(doc["title"]) <= 120
