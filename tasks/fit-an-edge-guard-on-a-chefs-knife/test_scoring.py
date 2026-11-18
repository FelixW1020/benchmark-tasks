"""Checks of the knife edge-guard task's shapes, layouts, record and scoring rule. No simulator needed:

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
FAR = np.array([[0.0, 0.0, T.TOP + 0.50]])                # an arm point nowhere near anything
REST = {T.KNIFE: (0.0, 0.0), T.GUARD: (0.0, 0.0)}
KNIFE0 = T.knife_start()
SEATED = K.G_FOLD + 0.0015                                # the folded edge this far below the knife's edge line


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def guard_on(knife=KNIFE0, depth=SEATED, s_end=0.0015, off=0.0, turn=None):
    """The guard's pose for a place on the knife: its heel-end corner ``s_end`` along the blade from the heel,
    ``depth`` below the edge line and ``off`` out of the blade's plane, turned by ``turn`` (a rotation in the
    knife's frame)."""
    rk = K._rotation(knife[1])
    pos = np.asarray(knife[0], float) + rk @ np.array([s_end, -depth, off])
    return pos, K._quat(rk @ (np.eye(3) if turn is None else turn))


def knife_at(x=None, y=None, z=None, lift=0.0):
    p = np.array(KNIFE0[0], float)
    if x is not None:
        p[0] = x
    if y is not None:
        p[1] = y
    p[2] = (p[2] if z is None else T.TOP + z) + lift
    return p, KNIFE0[1]


def step(s=None, arm=FAR, knife=KNIFE0, guard=None, speeds=REST, n=1):
    s = T.new_state() if s is None else s
    for _ in range(n):
        T.advance(s, arm, knife, T.guard_start() if guard is None else guard, speeds)
    return s


def total(s) -> float:
    return T.score_from(s["touched"], s["covered"], s["all"], s["seated"], s["lying"], s["let_go"], s["still"])


def edge_point(knife=KNIFE0, s=0.10, side=0.0, up=0.0):
    """A world point ``side`` m from the edge (outward, in the blade's plane) and ``up`` out of the plane, ``s``
    along the blade."""
    return K.place(knife[0], knife[1], [[s, float(K.edge_y(s)) - side, up]])


# -- shapes ------------------------------------------------------------------------------------------------------------

def test_handle_pieces_are_solid_convex_hulls():
    for pts, color in K.handle_pieces():
        assert 4 <= len(pts) <= 64 and len(color) == 3
        faces = K.hull_faces(pts)
        assert len(faces) >= 4
        for f in faces:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            assert ((pts - a) @ np.cross(b - a, c - a)).max() < 1e-9
        assert np.ptp(pts, axis=0).min() > 0.005


def test_knife_is_the_stated_size():
    boxes = K.knife_boxes()
    lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    handle = np.vstack([p for p, _ in K.handle_pieces()])
    assert hi[0] - lo[0] == pytest.approx(0.200) and lo[0] == 0.0             # the blade: 7.9 in
    assert hi[1] - lo[1] == pytest.approx(0.053)                              # its height at the heel: 2.1 in
    assert hi[0] - handle[:, 0].min() == pytest.approx(0.340)                 # overall: 13.4 in
    assert np.ptp(handle[:, 2]) == pytest.approx(0.020)                       # the handle's thickness: 0.8 in
    assert hi[2] - lo[2] <= K.T_SPINE + 1e-9 and K.T_SPINE == pytest.approx(0.0023)
    assert K.K_MASS == pytest.approx(0.181)                                   # 6.4 oz


def test_blade_tapers_from_spine_to_edge_and_to_the_tip():
    for s in (0.0, 0.05, 0.10, 0.15, 0.19):
        assert float(K.spine_y(s)) > float(K.edge_y(s))
    assert float(K.spine_y(K.BL)) == pytest.approx(float(K.edge_y(K.BL)))      # edge and spine meet at the tip
    assert float(K.edge_y(K.STRAIGHT)) == 0.0 and float(K.edge_y(0.15)) > 0.0
    boxes = K.knife_boxes()
    first = [hi[2] - lo[2] for lo, hi in boxes[:5]]                           # the heel segment, edge strip to spine strip
    assert first == sorted(first) and first[0] <= 0.0005 and first[-1] <= K.T_SPINE


def test_blade_colliders_stay_inside_the_outline():
    for lo, hi in K.knife_boxes():
        sm = (lo[0] + hi[0]) / 2
        assert lo[1] >= float(K.edge_y(sm)) - 1e-9 and hi[1] <= float(K.spine_y(sm)) + 1e-9
        assert hi[2] - lo[2] >= 0.0004


def test_guard_is_the_stated_size_and_its_slot_takes_the_blade():
    boxes = K.guard_boxes()
    lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    assert hi[0] - lo[0] == pytest.approx(0.216)                              # 8.5 in
    assert hi[1] - lo[1] == pytest.approx(0.0492)                             # 1 15/16 in
    assert hi[2] - lo[2] == pytest.approx(K.G_T) and K.G_T == pytest.approx(0.0059)
    assert K.GL > K.BL                                                        # longer than the blade
    depth = K.GW - K.G_FOLD
    assert depth - K.MOUTH > K.TIP_RISE + 0.008                               # deep enough to take the tip
    assert K.T_SPINE * depth / K.BH < K.G_SLOT - 0.0005                       # the blade is thinner than the slot at its mouth


def test_edge_points_run_from_just_in_front_of_the_heel_to_the_tip():
    e = K.edge_local()
    assert e[0, 0] == pytest.approx(K.EDGE_FROM) and e[-1, 0] == pytest.approx(K.BL)
    assert np.diff(e[:, 0]).max() <= 0.004 + 1e-9
    assert K.STRAIGHT_EDGE.sum() >= 20 and not K.STRAIGHT_EDGE[-1]


def test_arm_links_are_named():
    assert set(K.FINE_LINKS) <= set(K.ARM_LINKS) and "tip_left" in K.FINE_LINKS


# -- layouts -----------------------------------------------------------------------------------------------------------

def test_board_is_the_stated_size_and_on_the_table():
    board = next(o for o in T.OBJECTS if o.name == "board")
    assert board.size == pytest.approx((0.305, 0.457, 0.038)) and board.mass is None
    assert board.pos[2] - board.size[2] / 2 == pytest.approx(T.TOP)
    assert board.pos[0] - board.size[0] / 2 == pytest.approx(T.BOARD_X0)


def test_knife_starts_on_its_side_on_the_board_edge_toward_the_robot():
    pos, quat = T.knife_start()
    pts = K.knife_points(pos, quat)
    assert T.on_board(pts)
    assert pts[:, 2].min() - T.TOP - T.BOARD_TOP == pytest.approx(0.0005, abs=1e-6)       # the handle lies on the board
    rk = K._rotation(quat)
    assert rk[:, 0] == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)             # the blade points right
    assert rk[:, 1] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)              # the spine is on the far side
    assert K.edge_points(pos, quat)[:, 0].min() == pytest.approx(T.HEEL_XY[0])


def test_guard_starts_leaning_on_the_boards_front_edge():
    pos, quat = T.guard_start()
    rg = K._rotation(quat)
    corners = np.array([[x, y, z] for x in (0.0, K.GL) for y in (0.0, K.GW) for z in (-K.G_T / 2, K.G_T / 2)])
    pts = K.place(pos, quat, corners)
    assert pts[:, 2].min() - T.TOP == pytest.approx(0.0005, abs=1e-6)        # it stands on the table
    assert rg[:, 1] @ (0.0, 0.0, -1.0) == pytest.approx(math.cos(T.GUARD_LEAN))            # fold up, open side down
    # the face toward the board passes within half a millimetre of the top of the board's front edge
    corner = np.array([T.BOARD_X0, 0.0, T.TOP + T.BOARD_TOP])
    assert (corner - pos) @ rg[:, 2] - K.G_T / 2 == pytest.approx(0.0003, abs=2e-4)
    # no part of it is inside the board
    inside = (pts[:, 0] > T.BOARD_X0 + 1e-6) & (pts[:, 2] < T.TOP + T.BOARD_TOP - 1e-6)
    assert not inside.any()
    assert pts[:, 2].max() - T.TOP > T.BOARD_TOP + 0.008                     # its folded edge stands above the board


def test_start_scores_nothing():
    s = step()
    assert total(s) == 0.0 and s["covered"] == 0.0 and not s["touched"] and not s["all"]


# -- the record: how far the guard is on ---------------------------------------------------------------------------------

def test_guard_seated_knife_lying_let_go_at_rest_is_full_score():
    s = step(guard=guard_on())
    assert s["all"] and s["seated"] and s["lying"] and s["let_go"] and s["still"]
    assert s["edge_mm"] == pytest.approx(1.5, abs=0.05)
    assert total(s) == 1.0


def test_guard_turned_end_for_end_still_covers():
    turned = guard_on(s_end=0.0015 + K.GL - 0.016, turn=rot_y(math.pi))       # the other way round along the blade
    s = step(guard=turned)
    assert s["all"] and total(s) == 1.0


@pytest.mark.parametrize("depth, score", [(K.G_FOLD + 0.0039, 1.0), (K.G_FOLD + 0.0045, 0.5), (0.0120, 0.5),
                                          (0.0125, 0.25), (K.GW - 0.012, 0.25), (K.GW - 0.0055, 0.25)])
def test_depth_bands(depth, score):
    assert total(step(guard=guard_on(depth=depth))) == score


def test_guard_only_just_started_scores_nothing():
    s = step(guard=guard_on(depth=K.GW - 0.004))                              # the edge 4 mm inside the open side
    assert s["covered"] == 0.0 and total(s) == 0.0


def test_a_quarter_of_the_edge_is_the_first_step():
    # slid on from the tip: 5.5 cm of the edge inside
    s = step(guard=guard_on(s_end=K.BL - 0.055, depth=K.G_FOLD + 0.0015 - 0.0))
    assert T.PART <= s["covered"] < 1.0 and total(s) == 0.25
    s = step(guard=guard_on(s_end=K.BL - 0.040))
    assert s["covered"] < T.PART and total(s) == 0.0


def test_guard_short_of_the_heel_leaves_edge_bare():
    assert total(step(guard=guard_on(s_end=0.0039))) == 1.0
    s = step(guard=guard_on(s_end=0.0060))
    assert not s["all"] and total(s) == 0.25


def test_guard_lying_on_the_blade_covers_nothing():
    s = step(guard=guard_on(off=K.G_T / 2 + 0.0015))                          # on top of the blade, not round its edge
    assert s["covered"] == 0.0 and total(s) == 0.0


def test_guard_on_the_spine_scores_nothing():
    backwards = guard_on(depth=-(K.BH + SEATED), turn=rot_x(math.pi))         # the slot over the spine: only the tip is in it
    s = step(guard=backwards)
    assert 0.0 < s["covered"] < T.PART and total(s) == 0.0


# -- the record: how things are left -------------------------------------------------------------------------------------

def test_still_held_is_three_quarters():
    k = KNIFE0
    hand = K.place(k[0], k[1], [[-0.04, K.BH - 0.015, 0.02]])                 # a point 2 cm above the handle
    s = step(arm=hand, guard=guard_on())
    assert s["seated"] and not s["let_go"] and not s["touched"] and total(s) == 0.75


def test_moving_is_three_quarters():
    for speeds in ({T.KNIFE: (0.03, 0.0), T.GUARD: (0.0, 0.0)}, {T.KNIFE: (0.0, 0.0), T.GUARD: (0.0, 0.6)}):
        s = step(guard=guard_on(), speeds=speeds)
        assert not s["still"] and total(s) == 0.75


def test_knife_in_the_air_is_three_quarters():
    k = knife_at(lift=0.05)
    s = step(knife=k, guard=guard_on(k))
    assert s["seated"] and not s["lying"] and total(s) == 0.75


def test_knife_off_the_board_is_three_quarters():
    k = knife_at(x=T.BOARD_X0 - 0.10, z=K.HANDLE_T / 2)                       # on the table in front of the board
    s = step(knife=k, guard=guard_on(k))
    assert s["seated"] and not s["lying"] and total(s) == 0.75
    k = knife_at(y=-0.10)                                                     # on the board, its tip over the right edge
    s = step(knife=k, guard=guard_on(k))
    assert s["seated"] and not s["lying"] and total(s) == 0.75


# -- the record: the bare edge -------------------------------------------------------------------------------------------

def test_arm_near_the_bare_edge_zeroes_the_trial_for_good():
    s = step(arm=edge_point(side=0.009))
    assert s["touched"] and "bare edge" in s["why"] and s["nearest_mm"] == pytest.approx(9.0, abs=0.3)
    step(s, guard=guard_on(), n=3)                                            # a perfect finish afterwards
    assert s["seated"] and s["lying"] and total(s) == 0.0


def test_arm_eleven_millimetres_from_the_edge_is_allowed():
    s = step(arm=edge_point(side=0.011))
    assert not s["touched"] and s["nearest_mm"] == pytest.approx(11.0, abs=0.3)


def test_arm_at_the_tip_counts():
    s = step(arm=K.place(KNIFE0[0], KNIFE0[1], [[K.BL + 0.008, float(K.edge_y(K.BL)), 0.0]]))
    assert s["touched"]


def test_arm_on_the_handle_or_the_spine_does_not_count():
    k = KNIFE0
    for local in ([-0.04, K.BH - 0.015, 0.011], [-0.04, K.BH + 0.002, 0.0], [0.08, K.BH + 0.003, 0.0]):
        s = step(arm=K.place(k[0], k[1], [local]))
        assert not s["touched"]


def test_arm_next_to_a_covered_edge_does_not_count():
    g = guard_on()
    s = step(arm=edge_point(s=0.05, up=K.G_T / 2 + 0.002), guard=g)           # a fingertip on the guard's face, over the edge
    assert not s["touched"] and s["all"]
    s = step(arm=edge_point(s=0.05, side=SEATED + 0.002), guard=g)            # ... or behind its folded edge
    assert not s["touched"]


def test_only_the_part_still_outside_the_guard_is_bare():
    g = guard_on(depth=K.GW - 0.012)                                          # the straight part is in, the tip is not
    s = step(arm=edge_point(s=0.05, up=K.G_T / 2 + 0.002), guard=g)
    assert not s["touched"]
    s = step(arm=edge_point(s=0.19, up=0.006), guard=g)
    assert s["touched"]


def test_reset_clears_the_record():
    T._STATE.update(step(arm=edge_point(side=0.005)))
    assert T._STATE["touched"]
    T._clear()
    assert T._STATE == T.new_state()


# -- the scoring rule ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("args, score", [
    ((False, 0.0, False, False, False, False, False), 0.0),
    ((False, 0.24, False, False, True, True, True), 0.0),
    ((False, 0.25, False, False, True, True, True), 0.25),
    ((False, 0.98, False, False, True, True, True), 0.25),
    ((False, 1.0, True, False, True, True, True), 0.5),
    ((False, 1.0, True, True, False, True, True), 0.75),
    ((False, 1.0, True, True, True, False, True), 0.75),
    ((False, 1.0, True, True, True, True, False), 0.75),
    ((False, 1.0, True, True, True, True, True), 1.0),
    ((True, 1.0, True, True, True, True, True), 0.0),
    ((True, 0.5, False, False, True, True, True), 0.0),
])
def test_score_table(args, score):
    assert T.score_from(*args) == score


def test_score_and_report_read_the_record():
    T._clear()
    T._STATE.update(step(guard=guard_on()))
    assert T.score({}) == 1.0 and T.success({})
    rep = T.report({})
    assert rep["score"] == 1.0 and rep["whole_edge_covered"] and rep["seated"] and rep["lot"] == T.LOT
    T._clear()
    assert T.score({}) == 0.0 and not T.success({})


# -- the demo's plan -----------------------------------------------------------------------------------------------------

def test_demo_plan_puts_the_guard_on_the_carried_knife():
    import demo as D

    P = D.plan(T)
    k = P["knife_carried"]
    knife = (k[:3, 3], K._quat(k[:3, :3]))
    assert k[:3, 0] == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)             # level along its length
    assert math.degrees(math.asin(-k[2, 1])) == pytest.approx(math.degrees(D.TILT))        # tipped, edge up
    for depth, yaw, along, whole in ((K.G_FOLD + D.SEAT_GAP, 0.0, D.S_END, True), (K.GW + D.LEAD, D.YAW, D.S_CLEAR, False)):
        p, r = P["right_for"](depth, yaw, along)
        g = D.pose(p, r) @ P["guard_in_right"]
        s = step(knife=knife, guard=(g[:3, 3], K._quat(g[:3, :3])))
        assert s["all"] is whole and s["seated"] is whole
    p, r = P["right_for"](K.G_FOLD + D.SEAT_GAP, 0.0)
    assert math.degrees(math.atan2(r[0, 2], -r[2, 2])) == pytest.approx(math.degrees(D.LEAN_FLAT))
    # before it goes on, the guard's open side is clear of the edge all along
    p, r = P["right_for"](K.GW + D.LEAD, D.YAW)
    g = D.pose(p, r) @ P["guard_in_right"]
    lip = K.place(g[:3, 3], K._quat(g[:3, :3]), [[x, K.GW, 0.0] for x in np.linspace(0.0, K.GL, 12)])
    assert K.to_local(lip, *knife)[:, 1].max() < -0.010


def test_demo_hands_take_handle_and_folded_edge():
    import demo as D

    P = D.plan(T)
    kp, kq = T.knife_start()
    local = K.to_local([P["l_pick"]], kp, kq)[0]
    assert -0.060 <= local[0] - 0.014 and local[0] + 0.014 <= -0.012         # on the flat part of the handle
    assert local[1] == pytest.approx(K.BH - K.HANDLE_H / 2)
    assert P["l_pick"][2] - T.TOP - T.BOARD_TOP >= D.PAD_TIP                  # its pads' ends do not go into the board
    assert 0.0 <= D.S_END < K.EDGE_FROM - 0.002 and D.S_CLEAR > D.S_END       # the guard ends within the scored length
    gp, gq = T.guard_start()
    g = K.to_local([P["r_pick"]], gp, gq)[0]
    assert g[0] == pytest.approx(D.S_GRIP) and g[1] == pytest.approx(D.U_TCP) and abs(g[2]) < 1e-9
    # the fingertip on the board's side ends above the board (finger up to 9 mm thick, leaning with the guard)
    tip = P["r_pick"] + D.PAD_TIP * P["r_rot"][:, 2] + 0.009 * P["r_rot"][:, 0]
    assert tip[2] - T.TOP - T.BOARD_TOP > 0.0005
    assert D.PAD_TIP + D.U_TCP > 0.007                                        # at least 7 mm of pad on the guard


# -- the text ------------------------------------------------------------------------------------------------------------

def test_task_json_matches_the_task():
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    j = json.loads(path.read_text())
    assert j["instruction"] == T.INSTRUCTION
    assert len(j["title"]) <= 120 and len(j["fidelity"]) <= 500 and len(j["context"]) <= 1000 and len(j["scores"]) <= 300
    assert j["category"] in ("Impact on jobs", "Impact on AI timelines", "Safety and alignment", "Societal relatability", "Other")
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j[key]
    setup, scoring = j["setup"], j["scoring"]
    for text in ("45.7", "30.5", "3.8", "21.6", "4.9", "34", "181 g", f"{100 * T.BOARD_X0:.1f}", f"{100 * T.HEEL_XY[0]:.1f}"):
        assert text in setup, text
    for text in ("1 cm", "4 mm", "3 cm", "2 cm/s", "0.25", "0.5", "0.75", "1.0"):
        assert text in scoring, text
