# Spin lettuce dry in a hand-crank salad spinner: hold the bowl, keep the loose lid on, basket at 300 rpm for 5 seconds

A 2-quart hand-crank salad spinner stands on the worktop with washed lettuce in its basket and its lid on. The robot
has to spin the lettuce dry: crank fast enough to keep the basket at 5 turns a second (300 rpm) or more for 5 seconds
with the lid still on the bowl, then let go.

**Instruction:** "Spin the washed lettuce dry in the salad spinner: turn its crank clockwise fast enough to keep the
basket spinning at 5 turns a second or more for 5 seconds, keep the lid on, then let go of the crank."

A hand-crank salad spinner is in most home kitchens and in many restaurant prep kitchens. Its crank turns the basket
through gears (here 1:5) and a one-way clutch, so the basket spins up while the crank is turned and coasts on when it
stops. Drying the leaves takes three things at once:

1. **Cranking fast, round and round.** Water leaves the leaves only when the basket spins fast. It takes just over
   one crank turn a second to hold 300 rpm at the basket; a slow crank turns the basket without drying anything. The hand holds a small
   knob that has to travel round a 5.5 cm circle more than once a second, a dozen times over, without losing it. The
   knob is fixed to the crank, so the hand cannot choose its own path: it has to go where the crank lets it.
2. **Holding the bowl.** The gears push back on the crank, and that push goes into the lid and the bowl. A light
   plastic bowl on a smooth worktop turns and slides with the crank unless the other hand holds it by its handle.
3. **Keeping the lid on.** The lid is a separate part that only lies on the bowl. Two lugs stop it turning, but
   nothing stops it lifting: a hand that pulls up on the knob, or drags it sideways hard enough, lifts or tips the
   lid, and then the crank drives nothing. Cranking the lid off scores 0.

## What changed in this revision

The first version counted crank turns with the lid fixed to the bowl. The review pointed out that a real lid can be
cranked off, and that a crank turned too slowly does not dry lettuce, yet both still earned credit. Now:

- **The lid is loose.** It is its own body, lying on the bowl's rim with its own rim round the bowl's. The gears mesh
  only while it is on the bowl. If it is ever lifted 5 mm or tipped 8 degrees off the rim, the trial scores 0.
- **Credit is for time at drying speed, not for turns.** Every score above 0 is for time the basket spent at 5 turns
  a second (300 rpm) or more with the lid on: 5 seconds for full score, and no less than 1 second for any credit. The
  first version's own demo (six turns at a quarter of a turn a second, basket at 155 rpm at most) now scores 0.
- **Where 300 rpm and 5 seconds come from.** Machines sold to dry salad give the scale: a commercial electric salad
  spinner (Sammic ES-100) has two settings, 350 and 900 rpm, and an industrial leaf dryer (KRONEN K850) spins at up
  to 660 rpm for 5 to 30 seconds (manufacturers' specifications). The task's floor is set just under the slower
  machine's low setting, for the shortest of those cycle times. At this basket's wall, 8.8 cm from the axis, 300 rpm
  presses water outward at 8.9 times gravity. With 1:5 gears that takes just over one full crank turn a second. We found no
  published measurement of a hand-cranked spinner's speed, so the threshold is set from those machines, and a real
  trial measures the basket's speed directly (see Scoring).

## Why this task is different

**Sustained fast motion under a constraint, not a reach to a pose.** Almost every task on the board, and in the task
sets it is compared with, is a sequence of point-to-point moves: reach, grasp, carry, place, press. Here the hand has
to follow a path that the object imposes, continuously, for thirteen revolutions at more than one a second. It is the
same skill as winding a hand crank, turning a valve handwheel, cranking a window or an awning, winding a hose reel or
a pencil sharpener. A policy that plans the hand's path in advance and ignores what the crank is doing loses the knob
(see the development runs below). One that follows the crank keeps it.

**The second hand is needed by physics.** The instruction mentions only the crank and the lid. Holding the bowl is a
precondition the robot has to work out: without it, the reaction of the gears goes into the loose spinner instead of
the crank.

**Force has to go the right way.** The same hand that turns the crank can lift the lid off by it. In a development
run the hand only held the height at which it took the knob: the basket reached 5.6 turns a second, but at that speed
the lid had lifted 7 mm and tipped 4 degrees, and the trial scored 0. The final demo leans lightly on the knob, as a
person does, and the lid stays within 2.3 mm and 1.3 degrees of its seat.

**A moving part with its own dynamics.** The basket spins up through a clutch and keeps spinning after the crank
stops (about 4 seconds to slow down). A policy has to let go of a knob attached to a mechanism that is still moving.

**Nothing like it in the benchmarks we checked.** We searched all task titles and descriptions on the task board,
the accepted-task list, the reviewers' comments, RoboCasa's 365 kitchen tasks and the 1,018 activity definitions of
BEHAVIOR-1K. None has a salad spinner or a geared crank in the kitchen. RoboCasa's kitchen tasks open and close
appliances, turn knobs and move food between containers. BEHAVIOR-1K's salad and cooking activities are symbolic
state changes (diced, cooked) with no mechanism.

## Where this comes up

- **Home kitchens.** Washing and spinning salad greens, herbs and berries is a routine part of cooking at home, and
  hand-crank spinners are among the most common kitchen gadgets. People with limited grip strength or tremor find
  the crank hard to keep going while steadying the bowl, which makes it a natural task for assistive robots.
- **Food preparation work.** Washing and drying produce is a core task of food preparation workers (O*NET
  35-2021.00, nearly 900,000 US workers) and restaurant cooks (O*NET 35-2014.00). Small kitchens use the same hand
  spinners; larger ones use the same mechanism at bigger scale.
- **The same skill elsewhere.** Hand cranks drive pencil sharpeners, egg beaters, pasta machines, hose and cable
  reels, awnings, manual winches, hand drills and valve handwheels. Each needs a hand to follow the crank's circle
  while something holds the base.

## Potential impact

- **Constrained, continuous manipulation at speed.** Measures whether a policy can keep hold of a moving handle and
  go where the mechanism goes, over many fast cycles, rather than planning a path in free space.
- **Using feel, not just sight.** The crank's circle is wherever the bowl has ended up, and small slips in the
  fingers add up within a turn or two. Keeping hold means following where the hand is actually being carried: a
  closed loop through the robot's own sense of its position.
- **Bimanual bracing that the instruction leaves unsaid.** The policy has to infer that the bowl must be held.
- **An outcome that has to be reached, not a motion that has to be made.** The score asks whether the basket got up
  to drying speed and stayed there with the lid on, which a slow or careless crank does not do.
- **Graded progress.** Partial credit separates a policy that never gets the basket up to drying speed (0), one
  that gets it there for a second (0.25), one that keeps it there half the time needed (0.5) and one that holds it
  for the full 5 seconds and lets go (1.0).

## Files

| File | What it is |
|---|---|
| `task.py` | The task file: the spinner and its layout, the instruction, what happened during the trial, scoring |
| `spinnerkit.py` | Builds the salad spinner as two loose parts (the bowl with its basket; the lid with its crank and knob), works out how the lid sits on the bowl (`seat()`, `lid_on()`) and runs the gearing (`gear()`, every control step) |
| `demo.py` | The scripted demo, and its control runs |
| `test_scoring.py` | Offline tests of the gearing, the lid's fit, the scoring rule and the spinner's geometry (no simulator) |
| `task.json` | The submission's details |

## Requirements

The yam_arena embodiment and Isaac Lab-Arena, as set up by the program's GPU instance image (an NVIDIA RTX GPU is
needed). `test_scoring.py` needs only Python with numpy, pytest and the `yam_arena` package.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Spin the washed lettuce dry in the salad spinner: turn its crank clockwise fast enough to keep the basket spinning at 5 turns a second or more for 5 seconds, keep the lid on, then let go of the crank." \
  --scorer task_score --max-steps 1200
```

Run limit: 1,200 steps (60 s at 20 Hz). The demo reaches full score at step 484 (24 s), so the limit leaves room
for a slower start and a second try.

## Scripted demo

```bash
python demo.py task.py out_dir                  # the demo
python demo.py task.py out_dir --mode slow      # control: four turns at about 0.4 turns a second
python demo.py task.py out_dir --mode lift      # control: pick the lid up by the knob, put it back, then crank
python demo.py task.py out_dir --mode nohold    # control: the same cranking with the left hand kept away
python demo.py task.py out_dir --mode still     # control: the arms do not move
```

1. The left hand comes straight down on the bowl's side handle and holds it between its fingers.
2. The right hand comes down on the crank's knob leaning 25 degrees out from its arm's base, its two finger faces
   upright either side of the knob, and takes it. Held straight down, this arm cannot keep its hand upright on the
   far side of the crank's circle; leaning, every joint has room all the way round.
3. It turns the crank clockwise thirteen times at about 1.3 turns a second. The knob carries the hand round the
   crank's real circle, wherever the bowl has ended up, so every control step the hand fits a circle to its own last
   second of measured positions and aims 0.8 rad further round it. So it pulls the knob round without pushing it
   sideways, and a small slip in the fingers cannot add up. It is sent 6 mm lower than where it took the knob: it
   cannot slide down the knob, so it leans on it and keeps the lid down. The knob turns freely in its fingers. If its
   fingers ever close on nothing (the knob has slipped out), it opens them and takes the knob again.
4. It lets go of the knob and rises clear; then the left hand lets go of the handle.

The demo uses only the layout constants in `task.py`, the spinner's geometry in `spinnerkit.py` and the robot's own
measured hand positions and finger openings (proprioception). It never reads the spinner's state to act.

## Scoring

Tools: the trial video; a phone's slow-motion video (or a contact-free tachometer) aimed at the red stripe on the
basket's rim, seen through the clear bowl; a stopwatch; a ruler; a tape mark round the bowl's base on the worktop.

- **Time at drying speed:** the time in all, in seconds, that the basket turned at 5 turns a second (300 rpm) or
  more with the lid on the bowl. From the slow-motion video: the time covered by passes of the red stripe that come
  0.2 s or less apart.
- **Lid on:** the lid lies on the bowl's rim. It has come off if at any time (from 0.25 s after the start) the
  middle of the lid is 5 mm or more above its place on the rim, or the lid is tipped 8 degrees or more from the rim,
  or the lid's middle is 10.5 cm or more to the side of the bowl's axis (beside the bowl, not on it). On the trial
  video, with a ruler: measure the gap between the lid's underside and the bowl's rim on opposite sides of the bowl.
  The lid is off if the two gaps average 5 mm or more, or differ by 28 mm or more (8 degrees across the 20 cm rim).
  The lid's rim can tip only about 6 degrees while it is still round the bowl's, so a lid tipped 8 degrees has come
  out of its seat on one side.
- **Knocked about:** at any time, the spinner's base moves 8 cm or more from its place, or the spinner tilts 15
  degrees or more.
- **Let go:** at the end, both grippers' tool points are 4 cm or more from the crank's knob.

| Outcome | Score |
|---|---|
| Knocked about, or the lid came off, at any time | **0** for the whole trial |
| 5 seconds or more at drying speed, and let go at the end | **1.0** |
| 2.5 seconds or more at drying speed | **0.5** |
| 1 second or more at drying speed | **0.25** |
| Anything else | **0** |

Every outcome falls into exactly one row (the first that applies). A basket that never reaches 300 rpm scores 0
however many times the crank is turned. The basket coasts on after the crank stops, so the 5 seconds can include the
first moments after letting go. In code: `_watch()`, `score_from()` and `score()` in `task.py`; the basket's
speed and the lid's seat come from `_mechanism()`, `seat()` and `lid_on()` in `spinnerkit.py`.

## Verification

All runs below use the submitted scoring code.

| Run | Score | Notes |
|---|---|---|
| Scripted demo | 1.0 | 13.1 crank turns at up to 1.3 turns a second. The basket peaked at 6.2 turns a second (372 rpm) and spent 7.5 s at 300 rpm or more. The lid stayed within 2.3 mm and 1.3 degrees of its seat, the bowl moved 1.5 cm, both hands let go. Full score at step 484 (24 s) |
| Slow cranking (control) | 0.0 | 4.0 crank turns at about 0.4 turns a second with the bowl held and the lid on: the basket peaked at 2.5 turns a second (149 rpm) and spent no time at drying speed |
| Lid lifted (control) | 0.0 | The hand picked the lid up by the knob (13.8 mm, tipped 7.7 degrees), put it back and gave the same 13 turns. The basket again passed 6 turns a second, but the lid had come off |
| No holding hand (control) | 0.0 | The same cranking with the left hand kept away: the hand dragged the loose bowl 12.5 cm (knocked about at step 165) and the basket peaked at 4.4 turns a second |
| Arms still (control) | 0.0 | Nothing moves; the lid stays on and the basket stays still |
| Dream-1, 1,200 steps | 0.0 | The right gripper went to the crank's knob and hovered at it for several seconds, then drew back beside the bowl; the left arm stayed back; the crank was not turned |
| `pytest test_scoring.py` | 19 passed | The gearing (with the lid on and off), the lid's seat and its fit on the rim and lugs, the score tiers, the zero for a lost lid or a knocked bowl, the drying speed, the geometry |

What we learned building the demo, which shows what the task asks of a policy:

| Development run | What happened |
|---|---|
| Hand follows a pre-planned circle (first version, a quarter of a turn a second) | The knob slipped along the finger pads, the crank fell behind the hand and the knob came out after 2.7 turns |
| Aiming round where the layout puts the crank's axis | The bowl shifts about 1.3 cm in the holding hand, so the crank's real circle is not where the layout says; the hand has to find it from its own motion |
| Hand pointing straight down | On the far side of the crank's circle the wrist ran onto its joint stop |
| Aiming 1.0 rad ahead | Lost the knob; 0.8 rad keeps it |
| Fitted circle, hand held at the height it took the knob | Basket at 5.6 turns a second, but the lid lifted 7 mm and tipped 4 degrees: lid off, 0 |
| Leaning 12 mm into the knob, aiming 0.7 rad ahead | Lid stayed down (1.0 mm), but the basket peaked at 4.9 turns a second: no time at drying speed |
| Leaning 6 mm into the knob, aiming 0.8 rad ahead (final) | 7.5 s at drying speed with the lid on: 1.0 |

## Runtime and seeds

About 2 minutes per demo run and about 4 minutes per Dream-1 run on one L40S, including about a minute of
start-up. `object_jitter=0`; the layout is fixed, and there is no randomness in the task.

## Setup

Positions in cm from the midpoint between the two arm bases, on the table (x forward, away from the arms; y to the
left).

- **The salad spinner:** a 2-quart (about 1.9 L) hand-crank salad spinner, base centred at x 32, y -7.
  - The bowl: clear plastic, 20 cm across and 11 cm tall, 0.13 kg. Its base is plain plastic on the worktop (friction
    about 0.25). A green rubber-faced side handle stands out 6 cm from the bowl on its left (+y) side, 5 mm thick and
    2.5 cm tall, its middle 5 cm above the table. Two green lugs (4 mm out, 8 mm wide, 12 mm tall) under the rim,
    front and back.
  - The lid: clear, 21 cm across and 1 cm thick, 0.20 kg with its gears. It lies loose on the bowl's rim; its own rim
    reaches 1.5 cm down round the bowl's with 0.8 mm of clearance and a notch at each lug, so it cannot turn on the
    bowl but lifts straight off. A green gear housing (4 cm across) in its middle, and a red mark near its edge.
  - The crank: a grey arm 6.6 cm long on top of the housing, turning about the spinner's axis, 13.5 cm above the
    table at its top. At its end, 5.5 cm from the axis, a green knob 2.2 cm across and 3.4 cm tall that turns freely
    on the arm. It starts pointing forward (+x). Crank and knob weigh 0.03 kg.
  - The basket: white, 17.6 cm across, inside the bowl, with 150 g of washed lettuce (0.27 kg in all) and a red
    stripe on its rim.
  - The gearing: with the lid on, one clockwise turn of the crank turns the basket 5 times through a one-way clutch.
    Turned backward, or with the lid off the bowl, the crank does not drive the basket.
- **Robot:** both arms start in the raised ready pose with their grippers open.

## Fidelity

- **Two loose rigid parts.** The bowl (with the basket on a joint about its axis) and the lid (with the crank on a
  joint about its axis and the knob on one about its own) are separate bodies. The lid is held on the bowl only by
  its weight, its rim and the two lugs; contact between them is simulated, nothing pins it.
- **A rule for the gears** (`gear()` in `spinnerkit.py`), applied each control step:
  - while the lid is on the bowl and the crank, geared up 1:5, would spin the basket faster than it is going, the
    clutch drives the basket toward that speed, up to 0.01 N m: from rest it takes 4.6 s or more to reach 5 turns a
    second;
  - otherwise the basket coasts against the lettuce's air drag (2.2 x 10^-4 N m s/rad, slowing over about 4 s);
  - the crank feels the clutch's torque times 5, plus the gears' friction (0.008 N m s/rad, and 0.006 N m of holding
    friction so that a crank left alone stays put): at one turn a second, 0.10 N m (1.8 N at the knob) while the
    basket is spinning up and 0.085 N m (1.5 N) once it is at speed. This resistance acts between the crank and the
    lid, as in a real gearbox.
  - The basket's own drive torque (at most 0.01 N m) is applied between the basket and the bowl; in a real spinner
    it comes through the lid.
- **No water or loose leaves.** The lettuce is drawn as leaves fixed in the basket, and its mass is in the basket.
  Water is not simulated: dryness is judged by the basket's speed and the time at it, as set out under "What changed
  in this revision".
- **The bowl's collision shape** is a solid cylinder: nothing has to reach inside it.

## Real-world build

- **The spinner:** any 2-quart hand-crank salad spinner with a crank on a lift-off lid and a handle or rim the
  gripper can hold (or a small handle added to the bowl). Put a red stripe on the basket's rim where it shows
  through the bowl, and a tape mark round the bowl's base.
- **Each trial:** about 150 g of washed lettuce in the basket, lid on, crank pointing away from the robot.
- **Scoring:** film the red stripe in slow motion and count its passes per second (5 or more for 5 seconds in all);
  check on the trial video that the lid stayed on the rim; check the bowl against its tape mark and the grippers'
  distance from the knob at the end.

## Third-party assets

None. Every part is built from primitive shapes in `spinnerkit.py`.
