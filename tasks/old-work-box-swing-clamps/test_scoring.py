"""Offline checks of the task (no GPU, no Isaac Sim): the layout's geometry and the clamp's swing path, every scoring
outcome from fabricated states (including attempts to score without doing the task), the runtime's ending, the
demo's reach, and that task.json states the numbers the code uses.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider test_scoring.py
"""

from __future__ import annotations

import importlib
import json
import math
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import boxparts as bp  # noqa: E402

T = importlib.import_module("task")
J = json.loads((HERE / "task.json").read_text())

I3 = np.eye(3)
MM = 1e-3
FAR = np.array([[0.0, 0.40, 1.20], [0.0, -0.40, 1.20]])      # both grippers well away
BOX_HOME = T.HOLE.copy()                                       # flush in the cut-out
DRIVER_BLOCK = (np.array(T.DRIVER_START), I3)                  # standing in its block
X_FLOOR = -bp.EAR_T - bp.HEAD_H + bp.RECESS_D


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


POINT_AT_WALL = rot_y(90)          # a driver's +z (toward its handle) along +x: pointing away from the wall ...
IN_SCREW = rot_y(-90)              # ... and along -x: its tip toward the wall, handle toward the arms


def driver_in(t, box_pos=BOX_HOME):
    z = bp.SCREW_Z if t == "t" else -bp.SCREW_Z
    return (np.asarray(box_pos, float) + np.array([X_FLOOR, 0.0, z]), IN_SCREW)


def clamp(turns, swing_deg, force):
    """Joint positions of one clamp: the screw turned ``turns``, the arm swung ``swing_deg``, pressing ``force``."""
    th = 2 * math.pi * turns
    ph = math.radians(swing_deg) - th
    return th, ph, ph / (2 * math.pi) * bp.LEAD, force / bp.K_ARM


def joints(top=(0.0, 0.0, 0.0), bottom=(0.0, 0.0, 0.0)):
    q = {}
    for t, c in (("t", top), ("b", bottom)):
        q[f"screw_{t}"], q[f"nut_{t}"], q[f"slide_{t}"], q[f"wing_{t}"] = clamp(*c)
    return q


SNUG = (3.4, 93.0, 30.0)
LOOSE = (0.0, 0.0, 0.0)


class Episode:
    """Feeds fabricated steps to the task as the runtime does (the step hook's observe(), then success() and
    score()) and keeps the runtime's own 1 s success count."""

    def __init__(self):
        T._reset_state()
        self.box = (np.array(T.BOX_START), bp.rotation(T.BOX_START_XYZW))
        self.q = joints()
        self.driver = DRIVER_BLOCK
        self.grippers = FAR
        self.held = 0
        self.ended = False
        self.log = []

    def step(self, n=1, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        for _ in range(n):
            assert not self.ended
            T.observe(*self.box, self.q, *self.driver, self.grippers)
            s, ok = T.score(None), T.success(None)
            assert ok == (s >= 1.0), "success() must hold exactly when score() is 1.0"
            self.held = self.held + 1 if ok else 0
            self.ended = self.held >= round(T.SUCCESS_HOLD_S * 20)
            self.log.append(s)
            if self.ended:
                break
        return self.log[-1]

    def drive(self, t, turns_from, turns_to, swing, force, steps=20, other=None):
        """Turn screw ``t`` with the screwdriver in it, from ``turns_from`` to ``turns_to`` turns."""
        self.driver = driver_in(t, self.box[0])
        for k in range(1, steps + 1):
            f = turns_from + (turns_to - turns_from) * k / steps
            c = (f, swing if k == steps else min(swing, 90.0 * f / 0.25), force if k == steps else 0.0)
            other_c = other if other is not None else self.q_clamp("b" if t == "t" else "t")
            self.q = joints(c, other_c) if t == "t" else joints(other_c, c)
            self.step()

    def q_clamp(self, t):
        m = bp.mechanism(self.q)[t]
        return (m["turns"], m["swing"], m["force"])


def full_episode() -> Episode:
    e = Episode()
    e.step(5)
    e.box = (BOX_HOME, I3)
    e.step(5)
    e.drive("t", 0.0, SNUG[0], SNUG[1], SNUG[2])
    e.drive("b", 0.0, SNUG[0], SNUG[1], SNUG[2])
    return e


# -- layout and geometry ----------------------------------------------------------------------------------------------

def test_box_fits_its_cut_out_and_its_ears_bear_on_the_wall():
    for x in np.linspace(0.0, bp.DRYWALL_T, 6):
        hw, hh = bp._body_half(x)
        assert bp.HOLE_W / 2 - hw >= 1.2 * MM and bp.HOLE_H / 2 - hh >= 1.2 * MM
    assert bp.RIM_HH - bp.HOLE_H / 2 >= 6.0 * MM                # each ear bears 6 mm or more beyond the cut-out
    assert abs(bp.RIM_HH * 2 - 4.125 * 25.4 * MM) < 0.5 * MM    # 4-1/8 in overall
    assert abs(bp.DEPTH + bp.EAR_T - 2.75 * 25.4 * MM) < 0.1 * MM
    assert abs(bp.BOX_MASS - 0.0567) < 0.001                   # the maker's 0.125 lb (56.7 g)


def test_swung_arm_reaches_behind_the_drywall_and_clamps_inside_its_slot():
    reach = bp.SCREW_Z + bp.ARM_LEN
    assert reach - bp.HOLE_H / 2 >= 8.0 * MM                    # 8 mm or more of arm behind the board
    assert bp.SLOT_X[0] < bp.ARM_X_MIN and bp.DRYWALL_T + bp.ARM_T < bp.SLOT_X[1]
    assert bp.ARM_X0 > bp.DRYWALL_T + 5 * MM                    # folded, it is behind the board as delivered
    assert bp.SCREW_Z + bp.ARM_HW < bp._body_half(bp.ARM_X0)[1] - bp.WALL   # folded, it is inside the box
    turns = (bp.ARM_X0 - bp.DRYWALL_T) / bp.LEAD + 0.25
    assert 2.5 < turns < 4.0


def test_arm_swings_freely_to_its_stop():
    spatial = pytest.importorskip("scipy.spatial")
    hulls = []
    for s in bp.box_shapes():
        if s.body != "body" or not s.material:
            continue
        p = s.points()
        if s.kind == "box":
            c, h = np.array(s.center), np.array(s.size) / 2
            p = np.array([c + h * np.array([a, b, d]) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
        hulls.append(spatial.Delaunay(p))
    for t, sign in (("t", 1), ("b", -1)):
        wing = [s for s in bp.box_shapes() if s.body == f"wing_{t}"][0]
        lo, hi = wing.points().min(0), wing.points().max(0)
        grid = np.stack(np.meshgrid(*[np.linspace(lo[i] - 0.4 * MM * (i > 0), hi[i] + 0.4 * MM * (i > 0), 7 if i == 0
                                                  else 21) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        axis = np.array([0.0, 0.0, sign * bp.SCREW_Z])
        first = None
        for deg in range(0, 100):
            pts = (grid - axis) @ rot_x(deg).T + axis
            if any(h.find_simplex(pts).max() >= 0 for h in hulls):
                first = deg
                break
        assert first is not None and 90 <= first <= 97, (t, first)


def test_wall_and_parts_layout():
    lo, hi = bp.bounds("wall")
    assert abs(lo[2] + T.HOLE_Z) < 1e-9                       # the wall section stands on the table
    assert T.XW > 0.31                                          # clear of both arms' ready poses (x 0.30 within it)
    assert T.driver_low(*DRIVER_BLOCK) > 0.0                    # the driver's tip clears the table in its block
    st = bp.rotation(T.BOX_START_XYZW)
    corners = np.array(T.BOX_START) + np.array([[x, y, z] for x in (bp.DEPTH,) for y in (-1, 1) for z in (-1, 1)]) \
        @ st.T
    assert np.allclose(corners[:, 2], T.TOP + 0.3 * MM)         # lying on its back on the table


# -- scoring outcomes ----------------------------------------------------------------------------------------------------

def test_nothing_done_scores_zero():
    e = Episode()
    assert e.step(40) == 0.0


def test_full_score_needs_everything_and_ends_the_episode():
    e = full_episode()
    assert e.log[-1] == 0.75                                    # both clamps tight, screwdriver still in the screw
    e.driver = (np.array([0.25, -0.05, T.TOP + 0.016]), rot_y(90))   # put down on the table
    e.step(19)
    assert e.log[-1] == 0.75                                    # not yet still for 1 s
    e.step(60)
    assert e.ended and e.log[-1] == 1.0
    assert e.log.index(1.0) >= 19


def test_partial_credit_ladder():
    e = Episode()
    e.step(3, box=(BOX_HOME, I3))
    assert e.log[-1] == 0.25                                    # flush in the cut-out
    e.drive("t", 0.0, SNUG[0], SNUG[1], SNUG[2])
    assert e.log[-1] == 0.5                                     # one clamp tight
    e.drive("b", 0.0, SNUG[0], SNUG[1], SNUG[2])
    assert e.log[-1] == 0.75                                    # both tight


def test_flush_is_latched_for_quarter_credit_only():
    e = Episode()
    e.step(3, box=(BOX_HOME, I3))
    e.step(3, box=(BOX_HOME - [0.02, 0, 0], I3))                # pulled back out
    assert e.log[-1] == 0.25


@pytest.mark.parametrize("gap_mm,ok", [(0.4, True), (0.6, False)])
def test_ear_gap_limit(gap_mm, ok):
    e = Episode()
    e.step(2, box=(BOX_HOME - [gap_mm * MM, 0, 0], I3))
    assert (e.log[-1] == 0.25) == ok


def test_tilted_box_is_not_flush():
    tilt = math.degrees(math.atan(0.8 * MM / (2 * 50 * MM)))    # one ear 0.8 mm off the wall
    e = Episode()
    e.step(2, box=(BOX_HOME - [0.0, 0, 0], rot_y(tilt * 2)))
    assert e.log[-1] == 0.0


def test_clamp_needs_swing_force_and_flush_box():
    for top in ((3.4, 70.0, 30.0), (3.4, 93.0, 15.0)):          # not swung out / pressing under 20 N
        e = Episode()
        e.step(2, box=(BOX_HOME, I3))
        e.drive("t", 0.0, top[0], top[1], top[2])
        assert e.log[-1] == 0.25
    e = Episode()                                                # clamps tight on a box that is not in the cut-out
    e.box = (np.array([0.25, 0.10, T.TOP + 0.05]), I3)
    e.drive("t", 0.0, *SNUG)
    e.drive("b", 0.0, *SNUG)
    assert e.log[-1] == 0.0


def test_screws_turned_without_the_screwdriver_earn_nothing():
    e = Episode()
    e.step(2, box=(BOX_HOME, I3))
    e.driver = DRIVER_BLOCK                                      # the screwdriver never leaves its block
    for k in range(1, 21):                                       # both screws turned by the fingers
        f = SNUG[0] * k / 20
        e.step(q=joints((f, min(93.0, 360 * f), SNUG[2] if k == 20 else 0.0),
                        (f, min(93.0, 360 * f), SNUG[2] if k == 20 else 0.0)))
    e.step(40, grippers=FAR)
    assert e.log[-1] == 0.25 and not e.ended


def test_screwdriver_in_the_wrong_place_does_not_count_as_driving():
    e = Episode()
    e.step(2, box=(BOX_HOME, I3))
    tip, r = driver_in("t")
    e.driver = (tip + [0.0, 3.0 * MM, 0.0], r)                   # 3 mm off the screw's axis: beside the head
    for k in range(1, 21):
        f = SNUG[0] * k / 20
        e.step(q=joints((f, min(93.0, 360 * f), SNUG[2] if k == 20 else 0.0)))
    assert e.log[-1] == 0.25


def test_held_screwdriver_or_hand_on_the_box_caps_at_three_quarters():
    e = full_episode()
    tip = np.array([0.25, -0.05, T.TOP + 0.016])
    e.driver = (tip, rot_y(90))
    e.grippers = np.array([tip + [bp.HANDLE_TOP / 2, 0.0, 0.0162 + 0.02], [0.0, 0.4, 1.2]])  # a finger 2 cm over it
    e.step(60)
    assert e.log[-1] == 0.75 and not e.ended
    e.grippers = np.array([T.HOLE + [-0.02, 0.0, 0.0], [0.0, -0.4, 1.2]])                    # a hand at the box
    e.step(60)
    assert e.log[-1] == 0.75 and not e.ended
    e.grippers = FAR
    e.step(60)
    assert e.ended


def test_screwdriver_left_on_the_box_or_held_up_is_not_put_down():
    e = full_episode()
    e.driver = (T.HOLE + [-0.05, 0.0, -0.035], rot_y(90))       # resting in front of the box, up off the table
    e.step(60)
    assert e.log[-1] == 0.75 and not e.ended


def test_loosening_after_tightening_loses_the_credit():
    e = full_episode()
    e.drive("t", SNUG[0], 1.0, 93.0, 0.0)
    assert e.log[-1] == 0.5


def test_moving_box_never_succeeds():
    e = full_episode()
    e.driver = (np.array([0.25, -0.05, T.TOP + 0.016]), rot_y(90))
    for k in range(80):
        e.step(box=(BOX_HOME + [0.0, 0.0, 0.0015 * (k % 2)], I3))
    assert not e.ended and max(e.log) < 1.0


# -- the demo's moves ---------------------------------------------------------------------------------------------------

def test_demo_plan_reaches_and_stays_clear():
    demo = importlib.import_module("demo")
    from yam_arena import ik, station
    from yam_arena.analysis import fk_link_poses_world, link_hulls
    from yam_arena.constants import ARM_JOINTS

    hulls = link_hulls()

    class Ctx:
        live = False
        left = demo.Arm("left", station.BASES["left"])
        right = demo.Arm("right", station.BASES["right"])

    def points(arm):
        q = {n: float(v) for n, v in zip(ARM_JOINTS, arm.q[:6])}
        # fingers opened inside the box stop on its side walls (24.5 mm from its middle): no wider than that
        q["joint7"] = q["joint8"] = -0.04695 * min(arm.grip, 0.38)
        P = fk_link_poses_world(q, arm.base)
        return np.vstack([h.vertices @ P[lk][:3, :3].T + P[lk][:3, 3]
                          for lk in ("link4", "link5", "gripper", "tip_left", "tip_right") for h in hulls[lk]])

    c = Ctx()
    n = 0
    for _ in demo.script(T, c):
        n += 1
        for a in (c.left, c.right):
            p, _r = ik.tcp_pose(a.q, a.base)
            assert np.linalg.norm(p - a.pos) < 1.0 * MM
            assert abs(a.q[5]) <= math.radians(120) + 1e-6
        if n % 4 == 0:
            pl, pr = points(c.left), points(c.right)
            assert np.min(np.linalg.norm(pl[::5, None] - pr[None, ::5], axis=2)) > 0.09
            for P in (pl, pr):
                assert P[:, 2].min() > T.TOP - 1 * MM
                inwall = ((P[:, 0] > T.XW + 1 * MM) & (P[:, 0] < T.XW + bp.DRYWALL_T + bp.STUD_D)
                          & (np.abs(P[:, 1]) < bp.WALL_W / 2) & (P[:, 2] < T.TOP + bp.WALL_H))
                in_hole = (np.abs(P[:, 1]) < bp.HOLE_W / 2) & (np.abs(P[:, 2] - T.HOLE[2]) < bp.HOLE_H / 2)
                assert not np.any(inwall & ~in_hole)
    limit = int(re.match(r"([\d,]+) steps", J["run_limit"]).group(1).replace(",", ""))
    assert n < 0.6 * limit


# -- task.json --------------------------------------------------------------------------------------------------------

def test_task_json_fields_and_numbers():
    limits = {"title": 120, "context": 1000, "scores": 300, "fidelity": 500, "setup": 4000, "scoring": 4000,
              "instruction": 1000, "run_limit": 200}
    for k, n in limits.items():
        assert k in J and 0 < len(J[k]) <= n, k
    assert len(", ".join(J["keywords"])) <= 200
    assert J["category"] == "Impact on jobs"
    assert J["instruction"] == T.INSTRUCTION
    s = J["scoring"]
    for text in ("0.5 mm", "80 degrees", "20 N", "3 cm", "5 cm", "1 mm", "1 s", "0.25 turn", "2.5 mm of the screw's axis"):
        assert text in s, text
    assert "2-1/4 x 3-5/8 in" in J["setup"] and "1/2 in" in J["setup"]
    assert "Fasten small metal or plastic boxes to walls to house electrical switches or outlets." in J["context"]


def test_no_identifying_text_or_bytecode():
    for p in HERE.iterdir():
        assert p.name != "__pycache__" and p.suffix != ".pyc"
    for name in ("task.json", "README.md"):
        text = (HERE / name).read_text()
        assert "/Users/" not in text and "/home/" not in text and "@" not in text
