"""Scripted full-score demo for the old-work box task, run in the yam_arena embodiment (CPU PhysX).

    python demo.py task.py demo_out [--max-steps 6000] [--no-record] [--cam 640x352]
    DEMO_MODE=letgo python demo.py task.py out      # a control run: still, inside, box, top, nodown, letgo, pullout
    DEMO_CALIBRATE=1 python demo.py task.py cal_out --no-record     # prints the CAL numbers for the screwdriver

1. The box hangs on its cable in front of the cut-out. The left hand comes from the left with its fingers one above
   the other, opened to 82 mm, and pushes the box into the cut-out by its two ears (one fingertip end on each); the
   cable goes back into the wall ahead of the box.
2. It draws back (the box springs out a few millimetres), closes its fingers and comes back lower and further round
   to the left to hold the box against the wall by its bottom ear.
3. The right arm takes the screwdriver from its block by the handle (fingers across it, pointing down along it), turns
   it to point at the wall and puts its tip into the top clamp screw.
4. It drives the screw in wrist strokes, as a hand does: it lines the tip up with the screw, presses it in and feels
   for the recess, turns the screwdriver 200 degrees clockwise about the tip, draws it out and turns back. It stops a
   stroke once the clamp presses the drywall with 25 N (a hand feels the screw stall). With the top clamp tight the
   left hand goes home. Then the bottom screw.
5. It stands the screwdriver back in its block, lets go, and goes home.

The left hand's moves are fixed, apart from taking out its own sag (read from its joint angles). The screwdriving is
closed loop (as an eye and hand are): it reads the driver's tip, the box, the screw's angle and the clamp's force
from the simulator. Writes one MP4 per camera, setup.png and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


def rot_axis(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rotvec(r) -> np.ndarray:
    """The rotation vector of a rotation matrix."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v / (2 * math.sin(a)) * a


def cols(x, z) -> np.ndarray:
    """A TCP rotation with the fingers closing along ``x`` and pointing along ``z``."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    return np.column_stack([x, np.cross(z, x), z])


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF) and wrist turns (joint 6 only), each starting where the last
    command left it. With ``solve=False`` it only records where each move ends (for the offline checks)."""

    def __init__(self, name: str, base_pos, solve: bool = True):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base, self.solve = name, np.asarray(base_pos, float), solve
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.ends: list[tuple[np.ndarray, np.ndarray, str]] = []
        self.j6: list[float] = []

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, label: str = ""):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        pos = np.asarray(pos, float)
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            p, r = p0 + s * (pos - p0), ik.slerp_matrix(r0, rot, s)
            if self.solve:
                self._solve(p, r)
            else:
                self.pos, self.rot = p, r
            yield
        self.ends.append((pos.copy(), np.asarray(rot, float).copy(), label))

    def turn(self, angle: float, duration: float, stop=None, center=None):
        """Generator: turn the hand by ``angle`` (rad; positive is clockwise seen from behind it) about its fingers'
        axis, or, with ``center``, about the parallel axis through that point (the screw's axis: the hand carries a
        tool held off its own axis round the screw), at a steady rate; ``stop()`` true ends the turn there."""
        n = max(1, round(duration * CONTROL_HZ))
        axis = self.rot[:, 2].copy()
        p0, r0 = self.pos.copy(), self.rot.copy()
        c = p0 if center is None else np.asarray(center, float)
        for k in range(1, n + 1):
            if stop is not None and stop():
                break
            rk = rot_axis(axis, angle * k / n)
            pos, rot = c + rk @ (p0 - c), rk @ r0
            if self.solve and center is not None:
                self._solve(pos, rot)
            else:
                self.q = self.q.copy()
                self.q[5] = self.q[5] + angle / n
                self.pos, self.rot = pos, rot
                self.cmd = np.r_[self.q, self.grip]
            self.j6.append(float(self.q[5]))
            yield
        self.ends.append((self.pos.copy(), self.rot.copy(), "turn"))

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip, "grip")


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


# Measured once with DEMO_CALIBRATE=1 (the layout is fixed and the simulation deterministic) and fixed here.
CAL = {
    "driver_in_hand": (-0.00068, -0.00297, 0.00106),  # the driver's tip, in the right TCP's frame, after the lift, less as planned
    "tip_at_screw": {"t": (-0.00564, 0.00085, 0.07825), "b": None},   # the driver's tip in the hand in front of each screw (TCP frame, m)
    "push_past": 0.020,                 # pushing: the hand is sent this far past the ears flat on the wall (m); the cable pushes back and the arm gives
    "hold_past": 0.020,                 # holding: the hand is sent this far past the bottom ear's face (about 6 N)
    "press_past": 0.005,                # the driver's tip is pressed this far past the recess floor (m)
}
CALIBRATE = bool(os.environ.get("DEMO_CALIBRATE"))
_MEASURED: dict = {}

MODE = os.environ.get("DEMO_MODE", "full")   # full | controls: still, inside, box, top, nodown, letgo, pullout
PUSH_YAW = math.radians(-45.0)   # the pushing hand points at the wall from the left, 45 degrees off square
PUSH_CLOSE = 1.0           # its fingers close along the vertical (one above the other)
PUSH_Y = 0.016             # its fingertip ends meet the ears this far left of the screws (the cable's tail is on the right)
PUSH_GRIP = 0.87           # opened to 82 mm between the pads: fingertip ends 43 mm above and below the box's middle
PUSH_TILT = math.radians(0.0)    # the hand can be tipped forward by this, so that the upper fingertip leads
HOLD_PITCH = math.radians(0.0)   # the holding hand is level
HOLD_CLOSE = -1.0          # which way its fingers close (across, level; the wrist then sits at zero)
HOLD_Y, HOLD_Z = 0.016, 0.044    # its closed fingertips meet the bottom ear this far left of and below the box's middle
PULL_PUSH = 0.060          # control "pullout": the fingertips are sent this far into the cable where it leaves the box
HOLD_YAW = math.radians(-50.0)   # and it points at the wall from the left, 50 degrees off square (clear of the screwdriver hand)
PUSH_S = 6.0               # seconds for the push
TIPS = 0.013               # the TCP sits this far short of the pad tips
BOX_FINGERS_IN = 0.025     # the pad tips go this far into the box past its front face
BOX_GRIP = 0.55            # the fingers opened to this (1 = 94 mm between the pads) press on the box's side walls
DRIVER_TCP_UP = 0.036      # holding the driver: the TCP this far above the handle's lower end
DRIVER_GRIP_OPEN = 0.6
STROKE_EXTRA = math.radians(10)    # each stroke runs from 10 degrees before ...
STROKE = math.pi + 2 * STROKE_EXTRA  # ... to 10 degrees past a half turn of the wrist (200 degrees)
STOP_N = 25.0              # a stroke ends once the clamp presses with this (N)
MAX_STROKES = 16


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    import boxparts as bp

    L, R = ctx.left, ctx.right
    T = task
    if MODE == "still":
        yield from wait(10.0)
        return
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    r_ins = cols((0, -1, 0), (1, 0, 0))                 # pointing at the wall, fingers closing across (wrist 0)
    r_pick = rot_axis((0, 1, 0), math.pi / 2) @ r_ins   # the same, turned to point down
    r_drive0 = rot_axis((1, 0, 0), -math.pi / 2 - STROKE_EXTRA) @ r_ins   # pointing at the wall (wrist -100)

    def measure(key, value):
        if CALIBRATE:
            _MEASURED[key] = [round(float(v), 5) for v in value]
            print("CAL so far = " + json.dumps(_MEASURED), flush=True)

    if MODE == "inside":   # control: push the box in by a grip inside it, below the cable's line (it tips back)
        front = np.array(T.BOX_START) + np.array([-bp.EAR_T, 0.0, -0.008])
        tcp_in = front + (BOX_FINGERS_IN - TIPS) * ex
        yield from L.move(tcp_in - 0.07 * ex + 0.02 * ez, r_ins, 3.0, grip=0.0, label="in front of the box")
        yield from L.move(tcp_in - 0.07 * ex, r_ins, 1.0, label="level with it")
        yield from L.move(tcp_in, r_ins, 2.0, label="fingers into the box")
        yield from L.set_grip(BOX_GRIP, 1.0)
        yield from wait(0.5)
        yield from L.move(tcp_in + (bp.START_OUT + 0.02) * ex, r_ins, PUSH_S, label="push it in")
        yield from wait(0.8)
        _MEASURED["pushed_in"] = _snap(ctx, T) if ctx.live else None
        yield from L.set_grip(0.0, 0.8)
        yield from L.move(L.pos - 0.08 * ex, r_ins, 1.5, label="draw back")
        yield from L.move(*L.home, 3.0, grip=1.0, label="home")
        yield from wait(3.0)
        return

    # 1. Left arm: the box hangs on its cable in front of the cut-out. The hand comes from the left, turned so its
    #    fingers are one above the other and opened so that one fingertip end meets the top ear and the other the
    #    bottom ear, beside the screws (two points, so the box cannot tip back as the cable pushes on its top).
    d_push = np.array([math.cos(PUSH_YAW), math.sin(PUSH_YAW), 0.0])              # the way the fingers point
    r_push = rot_axis(np.cross(ez, d_push), -PUSH_TILT) @ cols((0, 0, PUSH_CLOSE), d_push)   # tipped: the top fingertip leads

    def tcp_at(ear_x):   # the tool point with the fingertip ends on the ears' face at world x = ear_x
        return np.array([ear_x, T.HOLE[1] + PUSH_Y, T.HOLE[2]]) - TIPS * d_push
    x_ears0 = T.BOX_START[0] - bp.EAR_T
    yield from L.move(tcp_at(x_ears0 - 0.05) + 0.02 * ez, r_push, 3.5, grip=PUSH_GRIP, label="in front of the box")
    yield from L.move(tcp_at(x_ears0 - 0.05), r_push, 1.0, label="level with it")
    yield from wait(0.3)
    dz = 0.0
    if ctx.live:   # the level hand sags: bring it back to the height planned (its own measured position)
        dz = float(L.pos[2] - ctx.tcp("left")[2])
        _MEASURED["hand_sag_mm"] = round(-1000 * dz, 2)
        yield from L.move(tcp_at(x_ears0 - 0.05) + dz * ez, r_push, 0.8, label="sag taken out")
    yield from L.move(tcp_at(x_ears0 - 0.006) + dz * ez, r_push, 1.5, label="fingertips at the ears")

    # 2. Push it into the cut-out until the ears are on the wall (the cable goes back into the wall ahead of it), and
    #    keep pushing: the bent cable springs back a little, so the box is held there until its first clamp is tight.
    x_flush = T.XW - bp.EAR_T
    yield from L.move(tcp_at(x_flush + CAL["push_past"]) + dz * ez, r_push, PUSH_S, label="push it in")
    yield from wait(0.8)
    _MEASURED["pushed_in"] = _snap(ctx, T) if ctx.live else None

    # 3. Let go, the box springs back out a little (the bent cable's give), so it has to be held against the wall
    #    while its first clamp is driven. The two-finger hand would be in the screwdriver's way: it draws back, closes,
    #    and comes back lower and further round to the left to hold the box by its bottom ear with its fingertips.
    yield from L.move(L.pos - 0.06 * d_push, r_push, 1.2, label="draw back")
    d_hold = np.array([math.cos(HOLD_PITCH) * math.cos(HOLD_YAW), math.cos(HOLD_PITCH) * math.sin(HOLD_YAW),
                       math.sin(HOLD_PITCH)])
    x_hold = np.cross(ez, d_hold) / np.linalg.norm(np.cross(ez, d_hold))
    r_hold = cols(HOLD_CLOSE * x_hold, d_hold)

    def tcp_hold(ear_x):   # the tool point with the closed fingertips on the bottom ear's face at world x = ear_x
        return np.array([ear_x, T.HOLE[1] + HOLD_Y, T.HOLE[2] - HOLD_Z]) - TIPS * d_hold
    yield from L.move(tcp_hold(x_flush - 0.06), r_hold, 3.0, grip=0.0, label="beside the box")
    yield from L.move(tcp_hold(x_flush + CAL["hold_past"]), r_hold, 2.5, label="hold it by the bottom ear")
    yield from wait(0.8)
    _MEASURED["held"] = _snap(ctx, T) if ctx.live else None

    def left_away():
        yield from L.move(L.pos - 0.08 * d_hold, r_hold, 1.5, label="draw back")
        yield from L.move(*L.home, 3.0, grip=1.0, label="home")
    if MODE == "box":      # control: hold, then let go without driving a screw
        yield from wait(2.0)
        yield from left_away()
        yield from wait(3.0)
        return
    if MODE == "letgo":    # control: the hand leaves before any clamp is tight
        yield from left_away()

    # 3. Right arm: take the screwdriver from its block by the handle, pointing down along it.
    handle_low = np.array(T.DRIVER_START) + bp.SHAFT_TOP * ez
    tcp_d = handle_low + DRIVER_TCP_UP * ez
    yield from R.move(tcp_d + 0.06 * ez, r_pick, 3.0, grip=DRIVER_GRIP_OPEN, label="over the driver")
    yield from R.move(tcp_d, r_pick, 1.5, label="round its handle")
    yield from R.set_grip(0.0, 0.8)
    yield from wait(0.3)
    up = tcp_d + 0.08 * ez
    yield from R.move(up, r_pick, 1.5, label="out of the block")
    tip_local = r_pick.T @ (np.array(T.DRIVER_START) + 0.08 * ez - up)            # planned: the tip in the hand
    if CALIBRATE:
        s = ctx.states()["driver"]
        measure("driver_in_hand", r_pick.T @ (s.pos - up) - tip_local)
    tip_local = tip_local + np.array(CAL["driver_in_hand"])

    def tcp_for_tip(tip, rot):
        return np.asarray(tip, float) - rot @ tip_local

    def clamp_force(t):
        st = ctx.states() if ctx.live else None
        if st is None:
            return 0.0
        return bp.mechanism(st["box"].joints)[t]["force"]

    def head_now(t):
        """The screw's recess floor (world), from the box's pose read from the simulator."""
        b = ctx.states()["box"]
        z = bp.SCREW_Z if t == "t" else -bp.SCREW_Z
        return b.pos + bp.rotation(b.quat_xyzw) @ np.array([-bp.EAR_T - bp.HEAD_H + bp.RECESS_D, 0.0, z])

    def aim(t, back):
        """Closed loop, as an eye and hand do: put the driver's tip ``back`` metres out from the screw's recess (both
        read from the simulator) with its shaft along the screw's axis and its cross turned to the recess's."""
        for _ in range(2):
            st = ctx.states()
            d = st["driver"]
            rd = bp.rotation(d.quat_xyzw)
            target = head_now(t) - back * ex
            a = rd[:, 2]                                       # tip to handle: should point at the arms (-x)
            v = np.cross(a, -ex)
            tilt = rot_axis(v, math.asin(min(1.0, np.linalg.norm(v)))) if np.linalg.norm(v) > 1e-6 else np.eye(3)
            blade = rd[:, 0]
            screw = bp.mechanism(st["box"].joints)[t]["turns"] * 2 * math.pi
            delta = (screw - math.atan2(blade[2], blade[1]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
            # of the equivalent turns (a quarter turn apart), the one that leaves the wrist nearest its stroke start
            j6 = float(R.q[5])
            delta = min((delta + k * math.pi / 2 for k in (-2, -1, 0, 1)), key=lambda x: abs(j6 + x - j6_start))
            corr = rot_axis(ex, delta) @ tilt
            tip = np.asarray(d.pos, float)
            rot = corr @ R.rot
            pos = R.pos + (target - tip) + (corr - np.eye(3)) @ (R.pos - tip)
            if np.linalg.norm(target - tip) < 0.0003 and abs(delta) < math.radians(2) and np.linalg.norm(v) < 0.02:
                break
            yield from R.move(pos, rot, 0.5, label="aim")
            yield from wait(0.2)

    def tip_depth(t):
        """How far (m) the driver's tip is into the recess past the head's face (read from the simulator)."""
        st = ctx.states()
        b, d = st["box"], st["driver"]
        return float((bp.rotation(b.quat_xyzw).T @ (np.asarray(d.pos) - b.pos))[0]) - (-bp.EAR_T - bp.HEAD_H)

    def seat(t, seated):
        """Closed loop, as a hand feels for the recess: while pressing, small circles of the tip and small turns of
        the wrist until the tip has dropped into the cross (read from the simulator)."""
        x_head = T.XW - bp.EAR_T - bp.HEAD_H + bp.RECESS_D
        base_pos, base_rot = R.pos.copy(), R.rot.copy()
        for k in range(16):
            st = ctx.states()
            b, d = st["box"], st["driver"]
            depth = float((bp.rotation(b.quat_xyzw).T @ (np.asarray(d.pos) - b.pos))[0]) - (-bp.EAR_T - bp.HEAD_H)
            lat = (bp.rotation(b.quat_xyzw).T @ (np.asarray(d.pos) - head_now(t)))[1:]
            if depth >= 1.0e-3 and np.linalg.norm(lat) <= 0.8e-3:   # 1 mm or more into the recess, near its axis
                seated[0] = True
                return
            # move the tip back over the axis (the soft arm sags as it presses), plus a small circle and a small turn
            base_pos = base_pos - bp.rotation(b.quat_xyzw) @ np.r_[0.0, lat]
            ang = k * math.pi / 4
            off = 0.3e-3 * np.array([0.0, math.cos(ang), math.sin(ang)])
            roll = rot_axis(ex, math.radians(10.0 * (1 if k % 2 else -1)))
            yield from R.move(base_pos + off, roll @ base_rot, 0.3, label="feel for the recess")
            yield from wait(0.1)

    # 4. Drive each clamp screw in half-turn wrist strokes.
    x_floor = T.XW - bp.EAR_T - bp.HEAD_H + bp.RECESS_D                    # the recess floor, world x
    for t, zs in (("t", bp.SCREW_Z), ("b", -bp.SCREW_Z)):
        head = np.array([x_floor, T.HOLE[1], T.HOLE[2] + zs])
        out = head - 0.012 * ex
        yield from R.move(tcp_for_tip(out - 0.04 * ex, r_drive0), r_drive0, 3.0, label=f"to the {t} screw")
        yield from R.move(tcp_for_tip(out, r_drive0), r_drive0, 1.0, label="lined up")
        yield from wait(0.5)
        if CALIBRATE:
            s = ctx.states()["driver"]
            measure(f"tip_at_screw_{t}", r_drive0.T @ (s.pos - R.pos))
        if CAL["tip_at_screw"][t] is not None:
            tip_local = np.asarray(CAL["tip_at_screw"][t], float)
            yield from R.move(tcp_for_tip(out, r_drive0), r_drive0, 1.0, label="lined up on the tip")
        j6_start = float(R.q[5])
        for k in range(MAX_STROKES):
            seated = [not ctx.live]
            for _try in range(3 if ctx.live else 1):
                if ctx.live:
                    if abs(float(R.q[5]) - j6_start) > math.radians(5):
                        yield from R.turn(j6_start - float(R.q[5]), 0.6)       # the wrist back to its stroke start
                    yield from aim(t, 0.012)
                yield from R.move(R.pos + (0.012 + CAL["press_past"]) * ex, R.rot, 0.8, label="tip in")
                if ctx.live:
                    yield from seat(t, seated)
                    if seated[0]:
                        break
                    yield from R.move(R.pos - (0.012 + CAL["press_past"]) * ex, R.rot, 0.6, label="tip out, again")
            yield from wait(0.2)
            done = [False]

            def stalled():
                done[0] = done[0] or clamp_force(t) >= STOP_N
                if CALIBRATE and ctx.live:
                    st = ctx.states()
                    rd = bp.rotation(st["driver"].quat_xyzw)
                    print(f"CAL turn: j6 {math.degrees(R.q[5]):.0f} blade {math.degrees(math.atan2(rd[2, 0], rd[1, 0])):.0f}"
                          f" screw {360 * bp.mechanism(st['box'].joints)[t]['turns']:.0f}"
                          f" tip {np.round(1000 * (st['driver'].pos - head), 1).tolist()}", flush=True)
                return done[0]
            pivot = head
            if ctx.live:   # turn about the tip where it actually is in the recess (as a wrist turns about the tool)
                pivot = np.asarray(ctx.states()["driver"].pos, float)
            stroke = max(0.0, min(STROKE, math.radians(118) - float(R.q[5])))   # within the wrist's +-120 degrees
            yield from R.turn(stroke, 2.4 * stroke / STROKE + 0.05, stop=stalled, center=pivot)
            if CALIBRATE:
                print(f"CAL stroke {t}{k}: force {clamp_force(t):.1f} N", flush=True)
            if done[0] or (not ctx.live and k == 7):
                break
            yield from R.move(R.pos - (0.012 + CAL["press_past"]) * ex, R.rot, 0.8, label="tip out")
            for _w in range(20 if ctx.live else 4):          # until the tip is clear of the recess
                if ctx.live and tip_depth(t) < -0.8e-3:
                    break
                yield
            yield from R.turn(j6_start - float(R.q[5]), 1.0)  # the wrist back to its stroke start
        yield from wait(0.5)
        yield from R.move(R.pos - 0.03 * ex, R.rot, 1.0, label="tip out, done")
        yield from R.turn(j6_start - float(R.q[5]), 1.2)                        # the wrist back
        yield from R.move(tcp_for_tip(out - 0.03 * ex, r_drive0), r_drive0, 1.5, label="hand square again")
        if t == "t" and MODE != "letgo":   # the top clamp holds the box now: the left hand lets go and goes home
            yield from left_away()
        if t == "t" and MODE == "top":     # control: stop after the first screw
            yield from wait(2.0)
            return

    if MODE == "nodown":   # control: both clamps tight, the screwdriver kept in the hand
        yield from wait(3.0)
        return

    # 5. Stand the screwdriver back in its block, let go, go home.
    yield from R.move(tcp_d + 0.10 * ez, r_pick, 3.5, label="over the block")
    if MODE == "pullout":   # control: with the box clamped, the left fingertips push the cable back through the box
        c0 = np.array([T.XW - bp.EAR_T - 0.016, T.HOLE[1] + bp.CABLE_Y, T.HOLE[2] + bp.CABLE_Z])   # where it leaves the box
        yield from L.move(c0 - 0.06 * d_hold - TIPS * d_hold, r_hold, 3.5, grip=0.0, label="beside the cable")
        yield from L.move(c0 - TIPS * d_hold, r_hold, 1.5, label="at the cable")
        yield from L.move(c0 + PULL_PUSH * ex - TIPS * d_hold, r_hold, 5.0, label="push the cable back into the box")
        yield from wait(0.5)
        _MEASURED["cable_pushed"] = _snap(ctx, T) if ctx.live else None
        yield from L.move(c0 - 0.06 * d_hold - TIPS * d_hold, r_hold, 2.0, label="draw back")
        yield from L.move(*L.home, 3.0, grip=1.0, label="home")
    yield from R.move(tcp_d + 0.004 * ez, r_pick, 2.0, label="into the block")
    yield from R.set_grip(DRIVER_GRIP_OPEN, 0.8)
    yield from R.move(tcp_d + 0.08 * ez, r_pick, 1.5, label="up off it")
    yield from R.move(*R.home, 3.0, grip=1.0, label="home")


def _snap(ctx, T):
    """The box as the simulator has it: ear gaps (mm), its origin less the cut-out's (mm), cable fed (mm)."""
    import boxparts as bp
    b = ctx.states()["box"]
    r = bp.rotation(b.quat_xyzw)
    return {"gaps_mm": np.round(1000 * T.ear_gaps(b.pos, r), 2).tolist(),
            "tipped_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(r[0, 0]))))), 1),
            "box_minus_hole_mm": np.round(1000 * (b.pos - np.asarray(T.HOLE)), 2).tolist(),
            "cable_fed_mm": round(1000 * bp.cable_fed(b.joints), 1)}


def offline_plan(task):
    """The demo's moves without the simulator: each arm's list of (pos, rot, label) at the end of every move."""
    from yam_arena import station

    class _Ctx:
        live = False
        left = Arm("left", station.BASES["left"], solve=False)
        right = Arm("right", station.BASES["right"], solve=False)

    ctx = _Ctx()
    steps = sum(1 for _ in script(task, ctx))
    return {"left": ctx.left.ends, "right": ctx.right.ends, "steps": steps, "j6": ctx.right.j6}


class Ctx:
    live = True

    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def tcp(self, arm: str):
        """Where the arm's tool point is now, from its own joint angles (proprioception)."""
        from yam_arena import ik
        import boxparts as bp

        robot = self.emb._env.unwrapped.scene[f"{arm}_robot"]
        q = bp.as_numpy(robot.data.joint_pos)[0][:6]
        a = self.left if arm == "left" else self.right
        return ik.tcp_pose(q, a.base)[0]

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 6000, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 6.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height, device=os.environ.get("DEMO_DEVICE", "cpu"))
    obs = emb.reset(Scene(id="demo", instruction=emb.task.module.INSTRUCTION))
    task = emb.task.module
    ctx = Ctx(emb)
    try:   # the start as the simulator has it: the box, and the cable's bent hinges
        import boxparts as bp
        obj = emb._env.unwrapped.scene["box"]
        q0 = dict(zip(obj.joint_names, bp.as_numpy(obj.data.joint_pos)[0].round(4).tolist()))
        b0 = ctx.states()["box"]
        _MEASURED["start"] = {"box_minus_planned_mm": np.round(1000 * (b0.pos - np.array(task.BOX_START)), 2).tolist(),
                              "box_quat_xyzw": np.round(b0.quat_xyzw, 4).tolist(),
                              "hinges": {k: v for k, v in q0.items() if k[0] == "h" or k in ("feed", "twist", "roll", "yaw")}}
    except Exception as e:  # noqa: BLE001
        _MEASURED["start_error"] = repr(e)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info, first_full = 0, None, False, {}, None
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        if first_full is None and float(info.get("score", 0.0) or 0.0) >= 1.0:
            first_full = steps
        snap(res.observation)
        if steps % 100 == 0:
            inf = getattr(task, "_EP", {}).get("info", {})
            m = inf.get("mech", {})
            print(f"step {steps} score {info.get('score')} gaps {inf.get('gaps_mm')} fed {inf.get('cable_fed_mm')} "
                  + " ".join(f"{t}: turns {m[t]['turns']:.2f} swing {m[t]['swing']:.0f} F {m[t]['force']:.0f}" for t in m)
                  + f" eng {inf.get('engaged')} L {[e[2] for e in ctx.left.ends][-1:]} R {[e[2] for e in ctx.right.ends][-1:]}",
                  flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    # The trial's score is what the task file's score() reported on the last step; never set here.
    final = float(info.get("score", 0.0) or 0.0)
    last = dict(getattr(task, "_EP", {}).get("info", {}))
    for k in ("mech",):
        last.pop(k, None)
    try:
        obj = emb._env.unwrapped.scene["box"]
        q1 = dict(zip(obj.joint_names, bp.as_numpy(obj.data.joint_pos)[0].round(4).tolist()))
        b1 = ctx.states()["box"]
        _MEASURED["finish"] = {"box_minus_hole_mm": np.round(1000 * (b1.pos - np.asarray(task.HOLE)), 2).tolist(),
                               "box_quat_xyzw": np.round(b1.quat_xyzw, 4).tolist(),
                               "hinges": {k: v for k, v in q1.items() if k[0] == "h" or k in ("feed", "twist", "roll", "yaw")}}
    except Exception as e:  # noqa: BLE001
        _MEASURED["finish_error"] = repr(e)
    result = {"mode": MODE, "task_score": final, "success": success, "steps": steps, "first_full_score_step": first_full,
              "end": json.loads(json.dumps(last, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))),
              "measured": _MEASURED,
              "script_ended_step": ended,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if CALIBRATE:
        result["CAL"] = _MEASURED
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam", default="640x352", help="camera width x height")
    a = ap.parse_args()
    w, h = (int(v) for v in a.cam.split("x"))
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=w, cam_height=h)
