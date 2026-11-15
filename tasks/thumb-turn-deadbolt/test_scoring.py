"""Offline checks (no GPU): every scoring outcome, and the lock's geometry (start clearances, the bolt clearing the
strike only with the door held shut, and the cam ratio).

    python -m pytest test_scoring.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechkit as mk  # noqa: E402
import task as T  # noqa: E402

HOLD = T.HOLD


def crank_angle(bolt_mm: float) -> float:
    """The thumb-turn angle (rad) whose crank pin sits in the middle of the bolt's slot with the bolt ``bolt_mm`` out:
    the pin's y is 12.5 * (cos - sin) mm from the axis, and the slot moves with the bolt."""
    return math.acos(min(1.0, (12.5 - bolt_mm) / (12.5 * math.sqrt(2)))) - math.pi / 4


def states(gap_mm=12.11, bolt_mm=0.0):
    """Fake states with the door's latch edge ``gap_mm`` off its stop and the bolt ``bolt_mm`` out (turn follows)."""
    q = {"door_hinge": -gap_mm / 1000 / T.LATCH_R, "bolt_slide": bolt_mm / 1000, "turn": crank_angle(bolt_mm)}
    return T.kit.fake_states(q=q)


def run(seq):
    """Feed a sequence of states through success()/score() as the runtime does; return the last (success, score)."""
    T._reset_state()
    out = None
    for o in seq:
        s = T.success(o)
        out = (s, T.score(o))
    return out


# -- readouts --------------------------------------------------------------------------------------------------------


def test_readouts_at_start():
    o = T.kit.fake_states()
    assert T.door_gap(o) == pytest.approx(0.0121, abs=2e-4)
    assert T.bolt_out(o) == pytest.approx(0.0, abs=1e-9)
    assert T.turn_deg(o) == pytest.approx(0.0, abs=1e-9)


def test_readouts_locked():
    o = states(0.0, 25.0)
    assert T.door_gap(o) == pytest.approx(0.0, abs=1e-9)
    assert T.bolt_out(o) == pytest.approx(0.025, abs=1e-9)
    assert T.turn_deg(o) == pytest.approx(90.0, abs=1e-6)


def test_quarter_turn_throws_25_mm():
    assert math.degrees(crank_angle(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(crank_angle(12.5)) == pytest.approx(45.0)
    assert math.degrees(crank_angle(25.0)) == pytest.approx(90.0)


# -- scoring outcomes ------------------------------------------------------------------------------------------------


def test_untouched_scores_zero():
    assert run([states()] * 50) == (False, 0.0)


def test_door_pushed_shut_only():
    assert run([states(1.5)] * 50) == (False, 0.25)
    assert run([states(2.5)] * 50) == (False, 0.0)


def test_bolt_into_strike():
    assert run([states(1.0, 8.1)] * 50) == (False, 0.5)
    assert run([states(2.9, 12.0)] * 50) == (False, 0.5)
    assert run([states(1.0, 7.0)] * 50) == (False, 0.25)


def test_bolt_out_with_door_open_is_not_in_the_strike():
    assert run([states(40.0, 25.0)] * 50) == (False, 0.0)
    assert run([states(3.5, 25.0)] * 50) == (False, 0.0)


def test_fully_thrown_briefly():
    seq = [states(0.5, 24.0)] * 5 + [states(0.5, 15.0)] * 30
    assert run(seq) == (False, 0.75)


def test_locked_needs_a_full_second_still():
    s, sc = run([states(0.5, 24.0)] * HOLD)        # the first step has no previous one, so it is not yet "still"
    assert (s, sc) == (False, 0.75)
    assert run([states(0.5, 24.0)] * (HOLD + 1)) == (True, 1.0)


def test_never_full_score_while_moving():
    seq = [states(0.5, 23.0 + 0.6 * (k % 2)) for k in range(80)]   # bolt chattering 0.6 mm per step
    assert run(seq) == (False, 0.75)


def test_locked_and_released_door_still_counts():
    # let go: the weatherstrip pushes the door out until the bolt bears on the strike, about 2 mm
    assert run([states(2.0, 25.0)] * 40) == (True, 1.0)


def test_score_reuses_success_once_per_step():
    T._reset_state()
    o = states(0.5, 24.0)
    for _ in range(HOLD):
        T.success(o)
        T.score(o)
    assert T._STATE["held"] == HOLD - 1


def test_reset_clears_latches():
    run([states(0.5, 24.0)] * 40)
    mk.new_episode()
    assert T._STATE["thrown"] is False and T._STATE["held"] == 0 and T._STATE["last"] is None


# -- geometry ---------------------------------------------------------------------------------------------------------


def _obbs(part, t):
    """(centre, rotation, half extents) of each box of ``part`` posed by world transform ``t``."""
    out = []
    for b in T.kit.parts[part].boxes:
        r = t[:3, :3] @ mk.quat_to_mat(b.quat)
        out.append((t[:3, :3] @ np.asarray(b.at) + t[:3, 3], r, np.asarray(b.size) / 2))
    return out


def _sep(a, b) -> float:
    """Separating distance (m) between two OBBs along the best SAT axis; negative = overlap depth."""
    (ca, ra, ha), (cb, rb, hb) = a, b
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    best = -np.inf
    d = cb - ca
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        pa = np.sum(ha * np.abs(ra.T @ ax))
        pb = np.sum(hb * np.abs(rb.T @ ax))
        best = max(best, abs(d @ ax) - pa - pb)
    return best


def _gap(pa, pb, q) -> float:
    poses = T.kit._fk(q)
    return min(_sep(a, b) for a in _obbs(pa, poses[pa]) for b in _obbs(pb, poses[pb]))


def _q(gap_mm, bolt_mm):
    return {"door_hinge": -gap_mm / 1000 / T.LATCH_R, "bolt_slide": bolt_mm / 1000, "turn": crank_angle(bolt_mm)}


# pairs that collide in the sim (jointed pairs and a fixed part's world-jointed children do not)
COLLIDING = [("bolt", "frame"), ("thumbturn", "frame"), ("weatherstrip", "door"), ("weatherstrip", "bolt"),
             ("weatherstrip", "thumbturn")]


@pytest.mark.parametrize("pair", COLLIDING)
def test_start_pose_clear(pair):
    q = {"door_hinge": math.radians(T.DOOR_START_DEG)}
    g = _gap(*pair, q)
    if pair == ("weatherstrip", "door"):
        assert g == pytest.approx(0.0, abs=2e-4)   # the door rests on the free weatherstrip
    else:
        assert g >= 0.002, f"{pair} only {g * 1000:.2f} mm apart at the start"


def test_crank_pin_drives_the_bolt():
    # all through the turn the round pin sits in the bolt's slot with 0.5 mm of play: the crank and slot are the only
    # link between thumb-turn and bolt (the pin is round, so this is checked on its centre rather than with boxes)
    walls = [b for b in T.kit.parts["bolt"].boxes if b.size[2] < T.BOLT_W]           # the two slot walls
    pin = next(b for b in T.kit.parts["thumbturn"].boxes if b.axis == "X" and b.size[1] == T.PIN)
    for bmm in np.arange(0.0, 25.01, 1.0):
        poses = T.kit._fk(_q(0.0, bmm))
        c = poses["thumbturn"][:3, :3] @ np.asarray(pin.at) + poses["thumbturn"][:3, 3]
        faces = sorted(float((poses["bolt"][:3, :3] @ np.asarray(w.at) + poses["bolt"][:3, 3])[1]) for w in walls)
        lo, hi = faces[0] + 0.002, faces[1] - 0.002          # the slot's two faces (walls are 4 mm thick)
        assert hi - lo == pytest.approx(T.SLOT)
        assert lo + T.PIN / 2 <= c[1] + 1e-9 and c[1] + T.PIN / 2 <= hi + 1e-9, bmm
        assert c[1] - T.PIN / 2 - lo == pytest.approx(0.0005, abs=1e-6)     # centred: 0.5 mm each side
        zl = poses["bolt"][2, 3] + walls[0].at[2] - walls[0].size[2] / 2
        zh = zl + walls[0].size[2]
        assert zl < c[2] - T.PIN / 2 and c[2] + T.PIN / 2 < zh, bmm      # the pin stays within the walls' height


def test_only_the_pin_touches_the_bolt():
    for bmm in np.arange(0.0, 25.01, 2.5):
        poses = T.kit._fk(_q(0.0, bmm))
        tt = [o for o, b in zip(_obbs("thumbturn", poses["thumbturn"]), T.kit.parts["thumbturn"].boxes)
              if not (b.axis == "X" and b.size[1] == T.PIN)]
        assert min(_sep(a, b) for a in tt for b in _obbs("bolt", poses["bolt"])) >= 0.001, bmm


def test_bolt_clears_strike_when_shut():
    for b in (0.0, 5.0, 12.5, 25.0):
        assert _gap("bolt", "frame", _q(0.0, b)) >= 0.0019


def test_bolt_clears_strike_within_seated_gap():
    # the 2 mm clearance round the bolt lets the door stand up to about 1.9 mm off its stop with the bolt in
    assert _gap("bolt", "frame", _q(1.5, 25.0)) > 0
    assert _gap("bolt", "frame", _q(2.5, 25.0)) < 0


def test_bolt_blocked_with_door_resting_open():
    gap = T.door_gap(T.kit.fake_states()) * 1000
    assert _gap("bolt", "frame", _q(gap, 2.5)) > 0           # 3 mm gap to the jamb: it moves 2.5 mm freely
    assert _gap("bolt", "frame", _q(gap, 4.0)) < 0           # then hits the strike plate beside the opening
    for g in (3.5, 6.0, 30.0):
        assert _gap("bolt", "frame", _q(g, 6.0)) < 0, g      # never in with the door 3.5 mm or more open


def test_thrown_bolt_stops_the_door_closing():
    # bolt thrown in the air with the door open: closing it, the bolt meets the jamb well before the door is shut
    blocked = [g for g in np.arange(0.0, 60.0, 0.5) if _gap("bolt", "frame", _q(g, 25.0)) < 0]
    assert blocked and min(blocked) >= 1.5 and max(blocked) >= 30.0


def test_door_swing_clears_jamb_corner():
    # the door's outside latch corner swings out toward the strike as it opens; keep it off the strike plate
    corner = np.array([T.XF + T.DOOR_T, T.LATCH_Y, 0.0])
    a = np.array([T.XF, T.HINGE_Y, 0.0])
    for deg in np.arange(0.0, 30.0, 0.5):
        r = mk.axis_angle(np.array([0.0, 0.0, 1.0]), -math.radians(deg))
        p = r @ (corner - a) + a
        assert p[1] > T.JAMB_Y + 0.0008


def test_masses_realistic():
    assert all((p.mass is None or p.mass >= 0.03) for p in T.kit.parts.values())
    assert T.kit.parts["door"].mass == pytest.approx(6.3)


def test_torques_feasible():
    # thumb-turn: the lock's friction (0.25 N*m), within the 0.2-0.5 N*m of real thumb-turns
    assert 0.2 <= T.TURN_FRICTION <= 0.5
    # holding the door shut against the weatherstrip and hinge friction, pressing 0.28 m from the hinge axis
    strip = T.STRIP_RATE * (T.STRIP_OUT + T.STRIP_PRELOAD)
    push = (strip * T.STRIP_R + T.HINGE_FRICTION) / 0.28
    assert 3.5 <= strip <= 6.0 and push <= 10.0


def test_text_matches_code():
    import json

    d = json.loads((Path(__file__).parent / "task.json").read_text())
    assert d["instruction"] == T.INSTRUCTION
    for k, v in d.items():
        assert len(v) <= {"title": 120, "context": 1000, "instruction": 1000, "fidelity": 500, "scores": 300,
                          "run_limit": 200, "setup": 4000, "scoring": 4000, "reset_notes": 1000,
                          "category": 100}.get(k, 4000) or k == "keywords"
    assert len(", ".join(d["keywords"])) <= 200
    for s in ("2 mm", "3 mm", "8 mm", "23 mm", "1 s", "0.5 mm"):
        assert s in d["scoring"], s
    assert all(not any(isinstance(v, str) and a in v for a in ("/Users", "/home", "ubuntu")) for v in d.values())
    assert list(itertools.chain()) == []
