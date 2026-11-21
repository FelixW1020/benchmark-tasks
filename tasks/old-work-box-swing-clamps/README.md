# Install an old-work electrical box in a drywall cut-out and drive both swing-clamp screws

A wall section stands across the far side of the table, facing the arms: 1/2 in drywall on 2x4 studs, 16 in on
centre, with a 1-gang cut-out (2-1/4 x 3-5/8 in, the box's template) in the stud bay. On the left of the table a blue
1-gang old-work box (PVC, 4-1/8 x 2-1/4 x 2-3/4 in, 57 g) lies on its back with its two swing clamps folded inside; on
the right a #2 Phillips stubby screwdriver stands in a wooden block. The instruction:
*Install the blue electrical box in the hole in the wall: push it in until its ears are flat on the wall, then drive
both of its clamp screws with the screwdriver until the box is held tight. Put the screwdriver down and take both hands
away.*

**Status:** the scripted demo reaches the full score (1.0 from step 4,100, success at step 4,119 of 6,000; CPU PhysX). Dream-1: see
*Dream-1 run*. Mechanism bench results under *Mechanism bench*.

## Why it matters

Electricians (O*NET 47-2111.00, 821,000 US jobs in 2025) have as a Core task: *"Fasten small metal or plastic boxes to
walls to house electrical switches or outlets."* A box is fastened in one of two ways: nailed to a stud before the
drywall is hung (new work), or, in a finished wall, as an old-work (remodel) box held to the drywall by its own
clamps, a stock item at every electrical supplier. So nearly every switch or outlet added in an existing building
uses one; remodelers are 56% of US residential building firms (NAHB, 2025). The maker's installation steps are *"Trace Box, Cut on Lines, Insert Wires, Place Box in Wall,
Secure Box in Wall"* (Carlon nonmetallic box catalogue). General maintenance and repair workers (49-9071.00, 1.6 million US jobs in
2025; Core task: *"Assemble, install, or repair wiring, electrical or electronic components, ..."*) do the same job in
existing buildings.

What makes it hard for a robot is physical, not a choice:

- **A tight, square insertion.** The cut-out is 1.6 mm wider than the box on each side (1.5 mm above and below); the
  box has to go in square, by a grip that stays out of the hole, until both ears are flat on the wall.
- **Horizontal Phillips driving in strokes.** The #2 tip has to be pressed into a 1.8 mm recess on a horizontal axis
  and turned; a wrist turns only so far, so each half turn the tip comes out, the wrist turns back and the tip goes in
  again, lined up with the cross.
- **A hidden mechanism.** The clamps are behind the board. A turned screw first swings its clamp out, then draws it to
  the back of the drywall; the only sign that it has gripped is the screw stalling.

**How it differs from neighbouring tasks.** The pending "Fold an outlet's three stiff copper wires back into its wall
box ..." starts with the box already mounted and drives a device's screws into it. The final-rejected steel-stud box
bracket task (prebending sheet flaps of a specialised wraparound bracket) and the withdrawn, never-reviewed old-work box
clamps submission are the only box-mounting tasks found; this one uses a standard stock old-work box with its own
clamps. Accepted screwdriving tasks (end bracket on a DIN rail, lever set on a door, drawer knob with a powered driver)
turn screws whose effect is in view; none puts a box into a wall. No box or drywall installation task was found in
LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench or RoboDojo.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: layout, `success()`, `score()` and the per-step bookkeeping (which turns were made with the screwdriver) |
| `boxparts.py` | The parts (wall section, box with its two swing clamps, screwdriver, block): geometry, USD writer, and the hooks that let yam_arena spawn them and run the step function |
| `demo.py` | The scripted demo (and `DEMO_CALIBRATE=1` to measure its `CAL` numbers) |
| `mech_bench.py` | Bench for the clamps alone: steady torques on each screw, turns to swing out and to clamp, force per torque, a 20 N pull, backing out |
| `test_scoring.py` | Offline checks (no GPU): geometry and the clamp's swing path, every scoring outcome and attempts to score without doing the task, the runtime's ending, the demo's reach and clearances, task.json's numbers |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder. **CPU PhysX
(`-E device=cpu`) is required**, as for the end-bracket task: the Phillips tip's sub-millimetre contact in the recess
is what turns the screw.

```bash
INSTRUCTION=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score --max-steps 6000 --store-frames --log-dir ~/yam-eval/owb-dream
```

`task_score` gives 1.0 on success, otherwise `score()` at the end of the episode. `OWB_DEBUG=1` prints the ear gaps,
each clamp's turns, swing, depth and force, and the score's conditions every 10 steps.

## Scripted demo

```bash
python mech_bench.py --out bench_out                               # the clamps alone (bench.json)
DEMO_CALIBRATE=1 python demo.py task.py cal_out --no-record        # prints the CAL numbers
python demo.py task.py demo_out --max-steps 6000 --cam 640x352     # demo_*_cam.mp4, setup.png, result.json
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider test_scoring.py
```

1. The left arm puts its closed fingers 25 mm into the box (open face up), opens them against the side walls, lifts the
   box, turns it to face the wall and pushes it into the cut-out until the ears are on the wall; it closes its fingers,
   draws back and goes home.
2. The right arm takes the screwdriver out of its block by the handle (fingers across it, pointing down along it),
   turns it to point at the wall and puts the tip into the top clamp screw.
3. It drives the screw in strokes: it lines the tip up with the screw, presses it in (feeling for the recess with small
   circles and small turns until the tip has dropped into the cross), turns the wrist 200 degrees clockwise about the
   tip, draws the tip out and turns back. A stroke ends early once the clamp presses with 25 N, as a hand stops when
   the screw stalls. Then the bottom screw.
4. It stands the screwdriver back in its block, lets go and goes home.

The box's moves are fixed: where the box sits in the hand after it is picked up and held in front of the cut-out is
measured once with `DEMO_CALIBRATE=1` and fixed in `CAL`. The screwdriving is closed loop, as an eye and a hand are
(the end-bracket task's demo is too): before each stroke it reads the driver's tip, the box's pose, the screw's angle
and the clamp's force from the simulator, aims the tip, feels for the recess and stops at 25 N. Offline (no
simulator, eight strokes per screw) the plan's IK error is under 0.2 mm, the wrist stays within +-100 degrees, the
arms stay at least 19 cm apart and no part of either arm enters the wall outside the cut-out
(`test_demo_plan_reaches_and_stays_clear`).

### Mechanism bench (`mech_bench.py`, CPU PhysX)

The box flush in the cut-out, a steady torque on each screw in turn (as a screwdriver would):

| measurement | top / bottom |
| --- | --- |
| arm swung out to its stop (98 degrees) | 0.27 turns / 0.27 turns |
| arm on the back of the drywall | 3.1 turns / 2.9 turns |
| force at 0.05 N m (the screw stalls) | 91 N / 91 N |
| force at 0.1 / 0.2 N m | 150 / 152 N; 150 / 164 N (the arm's 3 mm spring travel is used up at ~120 N) |
| force left once the torque is taken off | 51 N / 27 N (the thread holds) |
| the clamped box pulled straight out with 20 N | moved 0.05 mm |

A kinematic probe (the screwdriver moved by pose, its tip 0.5 mm off the recess floor, turned 3 degrees a step)
turned the screw with it, 9 degrees behind (the play of the blades in the cross).

### Demo run

`demo.mp4` (top, left and right cameras side by side, 640 x 352 each) and `setup.png`: the box is flush in the cut-out
at step 720, the top clamp tight (20 N) at step 1,990, the bottom at 3,660; the screwdriver is back in its block at
step 4,030 and both hands away just after; score 1.0 from step 4,100, and the episode ends with success at step 4,119.
Screw turns: top 3.46 (3.34 with the screwdriver in it), bottom 3.38; clamp forces 46 N and 36 N; ear gaps 0.01 mm.

### Dream-1 run

One trial with the command under *Run a trial* (`--policy dream`, CPU PhysX, cameras 320 x 176, 6,000 steps):
**task_score 0.0** (ran to the step limit). Both arms reach for the box and the screwdriver; the box is picked up and
carried toward the wall but never put into the cut-out, and ends on the table; the screwdriver ends up out of its
block. `dream.mp4` (top, left and right cameras side by side).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Flush:** the box's centre within 2 mm of the cut-out's centre and each ear's back face within 0.5 mm of the wall at
  both its corners (four points, 52 mm apart across and 100 mm apart up the box).
- **Clamp tight:** its arm swung out at least 80 degrees (90 at its stop) and pressing the back of the drywall with
  20 N or more, and its screw turned no more than 0.25 turn without the screwdriver in it (tip in the recess's cross,
  within 2.5 mm of the screw's axis, shaft within 25 degrees of the axis). Turning the heads with the fingers earns
  nothing.
- **Put down:** the screwdriver's lowest point within 5 cm of the table top (lying on the table or standing in its
  block) and not in a screw. **Hands off:** every point of both grippers' collision hulls 3 cm or more from the box's
  outline and from the screwdriver. **Still:** the box and the screwdriver's tip moved 1 mm or less over the last 1 s.

| Score | Condition |
|---|---|
| 1.0 | flush, both clamps tight, screwdriver put down, hands off and still, all held for 1 s (20 steps); `success()` is true only then, and the runtime ends the trial after its own 1 s |
| 0.75 | flush and both clamps tight |
| 0.5 | flush and one clamp tight |
| 0.25 | the box was flush in the cut-out at some point |
| 0 | otherwise (clamps tightened on a box that is not flush in the cut-out earn nothing) |

**In a real trial:** the wall section's back is open, so after the trial each clamp is checked from behind (swung out
over the board; a 0.1 mm feeler will not pass under it); a clamp is tight when a further quarter turn with the
screwdriver takes force and the box moves 0.5 mm or less (feeler gauge) when pulled straight out with 20 N on a
luggage scale hooked over its rim; ear gaps with a 0.5 mm feeler gauge; distances with a ruler; that each screw was turned only with the
screwdriver, from the video.

## How it is modelled

All rigid bodies at real size and mass; the wall section and the block are fixed. The box is one PhysX articulation:
its shell, and for each clamp a screw, a threaded hub and an arm.

- **Screw** (`screw_*`): a revolute joint along its axis, against 0.02 N m of friction (head on the ear plate) and a
  little viscous drag. Its head is four lobes round a #2 Phillips cross (slots 1.6 mm wide, closed 2.95 mm from the
  axis, 1.8 mm deep, on a floor); the screwdriver's tip is two tapered blades crossing (0.6 to 1.0 mm thick). The tip
  turns the screw only by contact in the recess. (The lobes are boxes: a bench showed that PhysX made no contact
  between the tip and lofted, round-edged lobes this small. A real recess is conical, which centres the tip; the
  boxes' recess is not, so the tip has to be put on the cross more accurately than in real life.)
- **Thread** (`nut_*`, `slide_*`): the hub turns on the screw (revolute, 0.01 N m of thread friction) and slides along
  it (prismatic), the two coupled by a PhysX mimic joint: a turn of the hub on the screw moves it 4 mm along it.
- **Arm** (`wing_*`): on the hub, on a stiff spring along the screw (40 N/mm, the arm's bending: a 19 x 7 x 3 mm PVC
  cantilever, 3EI/L^3), up to 3 mm. Its force on the board is the spring's force.
- Nothing is scripted. Turned clockwise, the thread's friction carries the arm round with the screw; it swings out
  through the slot in the box's end wall and stops on the slot's edge (contact, at about 93 degrees). The screw then
  turns inside the stopped hub, and the thread draws the arm toward the ears until it presses on the back of the
  drywall (contact) and the screw stalls. Turned back, the arm backs off and swings in again. A box not in the wall
  has nothing behind it: the arm runs to the end of its travel and presses nothing.

**Simplifications.** The thread's lead is 4 mm per turn, five times a real #6-32 or #8-32 screw's 0.8 mm: the arm
travels from folded to clamped in about 3.4 turns (bench: swung out at 0.27 turns, on the drywall at 3.1) where a real
one takes about 15 (an estimate from its 11 mm of travel). A shorter thread would put the full run-down beyond a
reasonable run limit for half-turn wrist strokes; the accepted lever-set task makes the same kind of cut (two turns
for a dozen). The clamp force per unit of torque is about a fifth of an ideal real thread's, a little more than the
real torque per newton once a real thread's ~25% efficiency is counted, so the simulated clamp is not easier to
tighten. With 4 mm the thread still holds: on the bench a clamp left alone keeps 27-51 N, and the clamped box moves
0.05 mm under a 20 N pull (at 8 mm per turn the clamps backed off when let go).
Gypsum is rigid: an over-tightened clamp cannot crush the board's back or pull through it (in a real trial
that shows from behind as a crushed board and a loose box, and fails the pull test). The thread's friction does not
rise with the load. **No cable.** The maker's sequence is "Trace Box, Cut on Lines, Insert Wires, Place Box in Wall, Secure Box in Wall":
the NM cable is fed into the box before it goes into the wall, and then hangs out of its front. This task is only the
last two steps; the box goes in empty. A stiff cable through the box would make the insertion harder (it has to be
kept out of the way) and is not modelled.
The box's shell is our simplified design of a generic old-work box at the maker's published size and mass.

## Numbers and where they come from

| Quantity | Value | Source |
|---|---|---|
| Box | 4-1/8 L x 2-1/4 W x 2-3/4 D in, 14 cu in, PVC, 0.125 lb (56.7 g), ears and swing clamps | Carlon B114R-UPC product sheet (ABB) |
| Cut-out | 2-1/4 x 3-5/8 in | Wall-box template sold for the B114R (Jonard WTL-12: "One-gang cutout dimensions: 2 1/4" wide x 3 5/8" high") |
| Body section, wall 2.5 mm, 2 mm draft, ears 2.5 mm | as set | Estimates from the published size; measure the box used |
| Screws | 7.4 mm pan head, #2 Phillips | Estimate (#6/#8 pan head); the maker's FAQ: "A Number 6 screw is used with Carlon's old-work boxes" |
| Clamp arm | 19 mm from the axis, 7 x 3 mm, starts 24 mm behind the ears | Estimates: measure |
| Real turns, folded to clamped on 1/2 in drywall | about 15 (simulation: 3.4) | Estimate from 11 mm of travel at 0.8 mm per turn: count them |
| Clamped | 20 N per arm | Estimate: the box does not move under a 20 N pull, a further quarter turn takes force |
| Frictions | 0.02 N m (head), 0.01 N m (thread) | Estimates: measure the free-running torque with a torque screwdriver |
| Drywall, studs | 1/2 in; 2x4 = 1-1/2 x 3-1/2 in, 16 in on centre | Standard sizes |
| Screwdriver | #2 Phillips stubby, 92 mm, 50 g | Typical stubby; weigh the one used |

**Measuring the clamp.** With the box clamped in the cut-out, hook a luggage scale over the box's rim and pull
straight out; note the pull at which the box first moves (more than 0.5 mm on a feeler gauge). Then, with a torque
screwdriver on one clamp screw, note the torque at which the screw stalls (the hand's "snug") and count the turns from
folded to stalled. Five boxes, median values.

## Real-world set-up

Bought or cut from stock: a 480 x 400 mm piece of 1/2 in drywall screwed to a 2x4 frame (bottom and top plates and two
studs 16 in on centre), its back open, the bottom plate clamped to the table; the cut-out traced from the box's
template and cut with a drywall saw; a stock 1-gang old-work box with swing clamps; a #2 Phillips stubby screwdriver;
a 50 x 50 x 44 mm hardwood block with a 10 mm hole for it.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`); there are no seeds. A trial runs at most 6,000 steps (5 minutes at
20 Hz).
