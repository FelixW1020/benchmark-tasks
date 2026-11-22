# Install an old-work electrical box in a drywall cut-out and drive both swing-clamp screws

A wall section stands across the far side of the table, facing the arms: 1/2 in drywall on 2x4 studs, 16 in on
centre, with a 1-gang cut-out (2-1/4 x 3-5/8 in, the box's template) in the stud bay. A 14/2 NM-B cable comes down inside
the bay from the top plate, out through the cut-out and into the back of a blue 1-gang old-work box (PVC, 4-1/8 x 2-1/4 x
2-3/4 in, 57 g), which hangs on it in front of the cut-out with its two swing clamps folded inside. On the right of the
table a #2 Phillips stubby screwdriver stands in a wooden block. The instruction:
*The cable is already in the blue electrical box. Install the box in the hole in the wall: push it in until its ears are
flat on the wall, with the cable going back into the wall behind it, then drive both of its clamp screws with the
screwdriver until the box is held tight. Put the screwdriver down and take both hands away.*

**Status:** the scripted demo reaches the full score (1.0 from step 3,570, success at step 3,589 of 6,000; CPU PhysX). Dream-1 scores 0.0. Controls and timings under *Results*.

## Why it matters

Electricians (O*NET 47-2111.00, 821,000 US jobs in 2025) have as a Core task: *"Fasten small metal or plastic boxes to
walls to house electrical switches or outlets."* A box is fastened in one of two ways: nailed to a stud before the
drywall is hung (new work), or, in a finished wall, as an old-work (remodel) box held to the drywall by its own
clamps, a stock item at every electrical supplier. So nearly every switch or outlet added in an existing building
uses one; remodelers are 56% of US residential building firms (NAHB, 2025). The box goes in with its cable already in
it: *"Once the fit is right, remove the box, run wires into it, and position the box back in the hole. Then use a
screwdriver to tighten the screws that engage the box's tabs and lock it in place."* (Fine Homebuilding); a maker's sheet
for another rework box says the same: *"Pull the cables through the opening and secure them to the outlet box."* (ABB). General maintenance and repair workers (49-9071.00, 1.6 million US jobs in
2025; Core task: *"Assemble, install, or repair wiring, electrical or electronic components, ..."*) do the same job in
existing buildings.

What makes it hard for a robot is physical, not a choice:

- **A box on a stiff cable.** The cable holds its bends and pushes back on the box 30 mm above its middle. It has to
  go back into the wall ahead of the box without being drawn out of the box's entry. Pushed in and let go, the box
  springs back out 5 to 7 mm (two controls), so it is not flat on the wall until a clamp draws it there; held by its
  bottom ear, its top ears still stand 2 mm off the wall.
- **Holding while driving.** The demo holds the box against the wall with one hand while the other drives the first
  clamp screw, and the two hands work within 5 cm of each other in front of a 57 mm box (a first attempt with the
  holding hand on both ears collided with the screwdriver hand and knocked the box out of the wall). The hold is not
  strictly needed in the simulation: with the box let go, the screwdriver's own press on the screw seats it again
  (control, full score 400 steps later).
- **A tight, square insertion.** The cut-out is 1.6 mm wider than the box on each side (1.5 mm above and below); the
  box has to go in square until both ears are flat on the wall.
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
turn screws whose effect is in view; none puts a box into a wall or handles a cable that is already in the part. No box or drywall installation task was found in
LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench or RoboDojo.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: layout, `success()`, `score()` and the per-step bookkeeping (which turns were made with the screwdriver) |
| `boxparts.py` | The parts (wall section, box with its two swing clamps and its cable, screwdriver, block): geometry, USD writer, the cable's bending law, and the hooks that let yam_arena spawn them and run the step function |
| `demo.py` | The scripted demo (`DEMO_MODE=letgo`, `pullout` or `still` for the control runs; `DEMO_CALIBRATE=1` to measure its `CAL` numbers) |
| `mech_bench.py` | Bench for the clamps alone (the bare box, `OWB_NO_CABLE=1`): steady torques on each screw, turns to swing out and to clamp, force per torque, a 20 N pull, backing out |
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
python mech_bench.py --out bench_out                               # the clamps alone, on the bare box (bench.json)
DEMO_CALIBRATE=1 python demo.py task.py cal_out --no-record        # prints the CAL numbers
python demo.py task.py demo_out --max-steps 6000 --cam 640x352     # demo_*_cam.mp4, setup.png, result.json
DEMO_MODE=letgo python demo.py task.py out                         # a control: still, inside, box, top, nodown, letgo, pullout
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider test_scoring.py
```

1. The left hand comes from the left, turned so that its fingers are one above the other and opened to 82 mm, and
   pushes the box into the cut-out by its two ears, one fingertip end on each, beside the screws; the cable goes back
   into the wall ahead of the box.
2. It draws back (the box springs out a few millimetres), closes its fingers and comes back lower and further round
   to the left to hold the box against the wall by its bottom ear.
3. The right arm takes the screwdriver out of its block by the handle (fingers across it, pointing down along it),
   turns it to point at the wall and puts the tip into the top clamp screw.
4. It drives the screw in strokes: it lines the tip up with the screw, presses it in (feeling for the recess with small
   circles and small turns until the tip has dropped into the cross), turns the wrist 200 degrees clockwise about the
   tip, draws the tip out and turns back. A stroke ends early once the clamp presses with 25 N, as a hand stops when
   the screw stalls. With the top clamp tight the left hand lets go and goes home. Then the bottom screw.
5. It stands the screwdriver back in its block, lets go and goes home.

The left hand's moves are fixed (open loop), apart from taking out its own sag from its joint angles. The screwdriving
is closed loop, as an eye and a hand are (the end-bracket task's demo is too): before each stroke it reads the
driver's tip, the box's pose, the screw's angle and the clamp's force from the simulator, aims the tip, feels for the
recess and stops at 25 N.

### Results

Every row is one run on the files in this folder (`result.json` in each run's folder); the simulation is deterministic.

| Run | Score | What happened |
|---|---|---|
| Scripted demo | 1.0 | success at step 3,589 of 6,000 (full score from step 3,570); all four ear corners within 0.01 mm of the wall; cable 152.5 mm fed (152.4 at the start); screws turned 3.34 and 3.42 turns with the screwdriver |
| Control: nothing moves | 0.0 | the box hangs where it started (ears 74.3 to 76.3 mm from the wall) |
| Control: pushed in by a grip inside the box, then let go | 0.25 | a two-finger grip inside the box seats it (ears 0.0 to 0.1 mm from the wall while pushed), then lets go with no screw driven: it springs back out (ears 4.6 to 4.9 mm from the wall); it was flat on the wall for a moment, hence 0.25 |
| Control: pushed in, held, then let go | 0.0 | pushed in by the ears and held by the bottom ear (ears 0.1 to 2.1 mm from the wall: the top ears stand off), then let go with no screw driven: it springs back out (ears 5.3 to 6.9 mm from the wall); never flat on the wall |
| Control: stops after the top screw | 0.5 | flat on the wall, the top clamp tight (3.34 turns), the bottom one not |
| Control: screwdriver not put down | 0.75 | both clamps tight, the screwdriver kept in the hand |
| Control: box not held while the first screw is driven | 1.0 | the box springs out about 6 mm when the hand leaves, and the screwdriver's own press on the screw seats it again; success at step 3,989, 400 steps later than the demo |
| Control: fingertips pushed 60 mm into the cable where it leaves the clamped box | 1.0 | an attempt to push the cable back out: the clamp's 8 N holds, the cable is still 152.5 mm fed and the score is unchanged |
| Dream-1 | 0.0 | 6,000 steps (after a 200-step test, also 0.0): both arms folded down near their bases; midway the left gripper came up beside the hanging box and went down again; the box was never pushed in and the screwdriver stayed in its block |

No run drew the cable out of the box, so the 15 mm cable rule is exercised by the offline tests only
(`test_cable_drawn_out_of_the_box_caps_the_score`). The clamp bench was not rerun: the clamps are unchanged.

Demo timeline (steps at 20 Hz): the box is pushed in by step 400 (bottom ears 0.05 mm from the wall, top ears 2.1 mm: the cable pushes on the top) and held from step 500; the top screw is driven from about step 750 and its clamp is tight (20 N) by step 1,900, when all four ear corners are within 0.01 mm of the wall; the left hand is home by step 2,100; the bottom clamp is tight by step 3,400; the screwdriver is back in its block and both hands away by step 3,570.

### Mechanism bench (`mech_bench.py`, CPU PhysX)

Measured on the bare box before the cable was added (`OWB_NO_CABLE=1 python mech_bench.py`; the clamps are unchanged). The box flush in the cut-out, a steady torque on each screw in turn (as a screwdriver would):

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

**The cable.** The three higher scores (0.5, 0.75, 1.0) also need the cable still fed into the box: no more than 15 mm
of it drawn back out through the port. A box clamped in the wall with its cable drawn out has to come out again, so
it keeps only the 0.25 for having been flush. A cable caught between an ear and the wall already fails "flush" (the
0.5 mm feeler).

**In a real trial:** the wall section's back is open, so after the trial each clamp is checked from behind (swung out
over the board; a 0.1 mm feeler will not pass under it); a clamp is tight when a further quarter turn with the
screwdriver takes force and the box moves 0.5 mm or less (feeler gauge) when pulled straight out with 20 N on a
luggage scale hooked over its rim; ear gaps with a 0.5 mm feeler gauge; distances with a ruler; that each screw was turned only with the
screwdriver, from the video. The cable: a ruler on its tail, whose end stands 51 mm above the top ear's upper edge at the
start; it counts as fed in if the end is no more than 15 mm lower at the end of the trial.

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

**The cable** is part of the box's articulation, which is rooted where the cable leaves the top plate: 13 links of
14.6 mm between the plate and the box, one long link that runs through the box (in through the port, under the top
wall, out of the front) and a two-piece tail.

- **Bends.** Copper holds a bend, so each joint between links is a hinge with a spring of 8 N m/rad (the cable's
  elastic stiffness over one link) whose torque is capped at a plastic moment of 0.12 N m; once a control step, a
  hinge that has been forced past its elastic give (0.86 degrees) has its set angle moved along with it. A bend
  therefore resists with up to 0.12 N m, stays where it is put, and springs back by under a degree. This is a
  material law applied by the kit's code (`_cable_follow`), not a contact.
- **The entry's clamp** is a sliding joint along the cable with the same law: it grips with up to 8 N and then the
  cable slips. The cable cannot leave the box in the simulation; being drawn more than 15 mm out is scored as out.
- **Twist and edgewise bend** are springs (0.65 N m/rad at the plate, 0.5 at the entry; 30 N m/rad edgewise).
- **Shape at the start.** The joints' zero is the cable with no weight on it; each hinge is bent up by the wind-up
  that the hanging box takes out again (7.2 degrees in all), so the loaded box hangs level, 78 mm in front of the
  wall. The cable's links collide with the wall and with each other; the long link is held in the box by its joints.

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
rise with the load. **The cable** bends in one plane only (about its flat face), with the twist and edgewise springs above standing in
for the rest; its jacket cannot crease; the port's clamp grips equally both ways, where a real tab lets the cable in
more easily than out. Every stiffness and force in it is an estimate (see the table). The stud bay's back is open
(for checking the clamps), so the cable's slack bows out behind the bay where a second wall face would hold it in.
The box's shell is our simplified design of a generic old-work box at the maker's published size and mass, with one
cable port in the top of its back wall (real boxes have clamp entries at the back of the top and bottom).

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
| Cable | 14/2 NM-B with ground: 0.372 in (9.4 mm) wide, 91 g/m; 14 AWG solid conductors 0.064 in across, 20 mil insulation, 25 mil jacket, annealed copper | Maker's product page (Southwire Romex SIMpull 14-2): "Approx. OD (inches) (cm) 0.372 0.94", "Approx. Weight (lbs/1000') (kgs/km) 61 91", "Diameter Over Conductor (in) (cm) 0.064 0.16", "ASTM B3 Soft or Annealed Copper Wire" |
| Cable thickness | 4.5 mm (0.176 in) | Estimate: the page read gives one dimension only; conductor, insulation and jacket add up to about 4 mm |
| Cable past the back wall | 152 mm (6 in) | Our choice, after the wiring code's 6 in of free conductor at a box (seen only as a code site's summary of NEC 300.14, not the rule's text) |
| Plastic moment of a bend | 0.12 N m | Estimate: three 1.63 mm annealed copper conductors at 70 MPa yield give 3 x (70 MPa x d^3 / 6) = 0.15 N m; 0.12 used. Check: hang weights on a 10 cm length clamped at one end until it stays bent |
| Elastic stiffness | 8 N m/rad per 14.6 mm link (EI 0.11 N m^2) | Estimate: three copper conductors, E 110 GPa |
| Entry clamp's grip | 8 N | Estimate: pull the cable out of a real box's clamp with a luggage scale |
| Twist, edgewise springs | 0.65, 0.5 and 30 N m/rad | Estimates |

**Measuring the clamp.** With the box clamped in the cut-out, hook a luggage scale over the box's rim and pull
straight out; note the pull at which the box first moves (more than 0.5 mm on a feeler gauge). Then, with a torque
screwdriver on one clamp screw, note the torque at which the screw stalls (the hand's "snug") and count the turns from
folded to stalled. Five boxes, median values.

## Real-world set-up

Bought or cut from stock: a 480 x 400 mm piece of 1/2 in drywall screwed to a 2x4 frame (bottom and top plates and two
studs 16 in on centre), its back open, the bottom plate clamped to the table; the cut-out traced from the box's
template and cut with a drywall saw; a stock 1-gang old-work box with swing clamps; a #2 Phillips stubby screwdriver;
a 50 x 50 x 44 mm hardwood block with a 10 mm hole for it.

**The cable.** About 60 cm of 14/2 NM-B. Drill a 3/4 in hole down through the top plate, 60 mm behind the back of the
drywall (73 mm behind its face) and 17 mm to the robot's right of the cut-out's centre line, and staple the cable to
the top of the plate so that 365 mm hangs below the plate's underside. Bend it by hand to the start shape: 112 mm
straight down, a quarter bend of about 37 mm radius toward the room, then straight out through the cut-out 30 mm
above its centre. Feed it through the box's top rear entry until 152 mm is past the back wall, lay it under the box's
top wall, and bend the 80 mm that comes out of the front up through a 12 mm radius so that it stands in front of the
top ear, to the right of the top screw, within 25 mm of the wall. Hang the box on it, square and level, with its ears'
back faces 78 mm in front of the wall. A cardboard template of the side view makes the shape repeatable. Replace the
cable when its jacket creases.

**Clamp turns and the time limit.** In the simulation a clamp goes from folded to tight in about 3.4 turns, and the
demo takes about a minute a screw (57 s for the top one, 62 s for the bottom), 17 to 18 s a turn. A real clamp as delivered needs about 15 turns (an estimate, from 11 mm of travel at 0.8 mm a
turn): at the demo's pace that is about 27 strokes a screw and about four and a half minutes a screw, nine minutes for the two, more than the 5-minute limit.
So for real trials each clamp is run up its screw beforehand, its arm held folded, until 3.5 turns are left (the arm's
front face 15 mm behind the ears, still clear of the 12.7 mm board); that is the state the simulation's coarser thread
stands for, and the reset note says so.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`); there are no seeds. A trial runs at most 6,000 steps (5 minutes at
20 Hz).

## Sources

- Fine Homebuilding, "How to Install an 'Old-Work' Electrical Box"
  (finehomebuilding.com/project-guides/wiring/install-old-work-electrical-box): "Once the fit is right, remove the
  box, run wires into it, and position the box back in the hole. Then use a screwdriver to tighten the screws that
  engage the box's tabs and lock it in place."
- ABB, rework boxes, mounting instructions for the CI820-DX
  (tnb.ca.abb.com/en/pdf-catalogues/boxes-and-covers-meter-sockets/boxes-and-covers/rework-Boxes.pdf): "Pull the
  cables through the opening and secure them to the outlet box."; "Push the outlet box into the opening ensuring that
  the mounting ears are flush to the drywall".
- Southwire, Romex SIMpull Type NM-B cable, 14-2 with ground
  (southwire.com/wire-cable/building-wire/romex-brand-simpull-type-nm-b-cable/p/28827480): the figures quoted in the
  table above.
- A code site's summary of NEC 300.14 (up.codes): "A minimum of 150 mm (6 in.) of free conductor must be available at
  outlets, junctions, and switch points for connections or splices." This is the site's summary, not the rule's text.

