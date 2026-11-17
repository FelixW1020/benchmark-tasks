# Spin lettuce dry in a hand-crank salad spinner: hold the bowl, keep the loose lid on, spin out two-thirds of its water

A 2-quart hand-crank salad spinner stands on the worktop with washed lettuce in its basket and its lid on. The robot
has to spin the lettuce dry: crank fast enough, for long enough, that at least two-thirds of the water the lettuce
carries has been spun out of it, with the lid still on the bowl, and then let go.

**Instruction:** "Spin the washed lettuce dry in the salad spinner: crank it clockwise fast enough and long enough to
spin out at least two-thirds of the water the lettuce carries (the basket well over 5 turns a second for about 6
seconds), keep the lid on, then let go of the crank."

A hand-crank salad spinner is in most home kitchens and in many restaurant prep kitchens. Its crank turns the basket
through gears (here 1:5) and a one-way clutch, so the basket spins up while the crank is turned and coasts on when it
stops. Drying the leaves takes three things at once:

1. **Cranking fast, round and round.** Water leaves the leaves only when the basket spins fast: how much comes off
   is set by how hard the basket's wall presses outward, which goes with the square of its speed. It takes more than
   one crank turn a second, kept up for several seconds, to spin out two-thirds of it; a slow crank turns the basket
   and leaves the lettuce wet. The hand holds a small
   knob that has to travel round a 5.5 cm circle more than once a second, a dozen times over, without losing it. The
   knob is fixed to the crank, so the hand cannot choose its own path: it has to go where the crank lets it.
2. **Holding the bowl.** The gears push back on the crank, and that push goes into the lid and the bowl. A light
   plastic bowl on a smooth worktop turns and slides with the crank unless the other hand holds it by its handle.
3. **Keeping the lid on.** The lid is a separate part that only lies on the bowl. Two lugs stop it turning, but
   nothing stops it lifting: a hand that pulls up on the knob, or drags it sideways hard enough, lifts or tips the
   lid, and then the crank drives nothing. Cranking the lid off scores 0.

## What changed in this revision

The second review said that the 300 rpm for 5 seconds threshold "is extrapolated from larger machines without
showing that it dries lettuce in this spinner", and asked for a calibration on measured drying of washed lettuce "or
another measurable drying condition". This revision takes the second course:

- **What is scored is water, not speed.** The score is by the share of the water the washed lettuce carried in that
  has been spun out of it: two-thirds for full score, half for 0.5, a third for 0.25. In a real trial that share is
  weighed on a kitchen scale (see Scoring). No speed or time threshold is used to score a real trial.
- **The rpm figures borrowed from larger machines are gone**, from the scoring, the instruction's reasoning and the
  context. They compared speeds across baskets of different sizes, which is not a like-for-like comparison: what
  takes water off a leaf is the outward push at the basket's wall, which goes with the radius as well as the speed.
- **In the simulator, which has no water, the share is worked out from that push**, in multiples of gravity, by a
  stated rule (next section). The rule has two numbers. One is a physical argument, the other an estimate. Neither
  has been measured on a spinner, and the section says so and gives the kitchen procedure for measuring them.
- **Slow cranking still earns nothing:** a basket kept at 144 rpm (2.0 times gravity at its wall) or slower never
  spins out a third of the water, however long it is cranked. The slow-cranking control run (149 rpm at its fastest)
  spun out 24% and scores 0.

Earlier changes, kept: the lid is a loose part that the crank can lift off (a trial in which it comes off scores 0),
and turning the crank is not credited for its own sake.

## The drying rule

**Why the push and not the rpm.** Water is flung off a leaf by the outward push on it, `(2 pi f)^2 r`: the square of
the basket's turning rate f times the radius r. At 300 rpm that is 8.9 times gravity at the wall of this 8.8 cm
basket and 20 times gravity in a 20 cm one. The rule below is written in multiples of gravity (n), so it means the
same thing in any basket.

**How much water a leaf can hold.** Washed leaves that have stopped dripping still carry water, held on by surface
tension against gravity. The size of drop that surface tension can hold against a push goes with the capillary
length, `sqrt(surface tension / (density x push))`, so it shrinks as one over the square root of the push; the water a
wetted leaf can carry per unit of its area goes the same way. The rule takes that as it stands: pressed outward at n
times gravity, the lettuce can hold `n^-0.5` of the water it held dripping at 1 g (`holds()` in `task.py`).

**How fast it leaves.** Water above what the leaves can hold leaves with a time constant of 2 s (`DRAIN_S`): after
2 s at a steady speed 63% of what that speed can remove has gone, after 6 s 95%. Water never goes back on.

| Basket speed | Push at this basket's wall | Share of the water spun out, if kept up |
|---|---|---|
| 151 rpm (2.5 turns a second; half a crank turn a second) | 2.25 g | a third |
| 202 rpm (3.4) | 4.0 g | half |
| 300 rpm (5.0; one crank turn a second) | 8.9 g | 66%: just short of two-thirds |
| 360 rpm (6.0) | 12.7 g | 72% |
| 600 rpm (10; beyond this demo's arm) | 35 g | 83% |

So full score takes the basket well over 5 turns a second for about 6 seconds, as the instruction says: held at 6
turns a second it reaches two-thirds in 5.2 s; at 5.5 turns a second in 6.5 s; at 5 turns a second it never quite
does.

**What has been measured, and how the rule compares.** We found four published measurements that bear on this.
None is a drying curve for a hand-crank household spinner, and one of them cuts against the rule:

| Measurement | What was measured | What the rule gives |
|---|---|---|
| A hand-pumped household spinner, pumped steadily by nine people ([Brown et al., Am J Trop Med Hyg 2011;85:327-332](https://pmc.ncbi.nlm.nih.gov/articles/PMC3144833)) | Basket speed 602 rpm on average (400 to 730), by laser tachometer; 31 g where their samples sat | Speed only: a hand-worked household spinner runs at several hundred rpm. The simulated one, at the pace this robot's arm keeps up, runs slower (374 rpm, 13.8 g) |
| Another hand-pumped household spinner, with washed mixed greens, in a product test ([Food Network, "Best Salad Spinners", updated June 2026](https://www.foodnetwork.com/how-to/packages/shopping/product-reviews/best-salad-spinner)) | "It removed at least 85 percent of the water from our lettuce in less than 25 seconds." Its speed was not measured | 85 to 86% if it ran at the 602 rpm above in a basket of 11 to 12 cm radius (44 to 48 g), reached within a few seconds. Consistent, but the speed is borrowed from a different spinner |
| An industrial drum dryer at 500 to 700 rpm, full baskets, 2 to 15 minutes ([US patent 8,234,795](https://patents.google.com/patent/US8234795B2/en), examples 1 and 2; moisture by air drying and weighing) | Surface moisture of romaine 8.5% before and 3.2% after (62% removed); chopped romaine, iceberg and two blends, 51 to 62% removed; spring mix 79%. In the patent's divided basket, which keeps the layer of leaves thin: romaine 81%, spring mix 89% | More than was measured in the full baskets: a drum that size at that speed presses harder than a household basket, for which the rule already gives about 85% |
| A draining stage and then a dryer on a fresh-cut line ([US patent 6,055,741](https://patents.google.com/patent/US6055741A/en)) | Lettuce and other vegetables at 4 to 10% residual water after draining, about 2% after the dryer: between half and four-fifths taken off | No speed given. The task's "dry" (two-thirds) sits inside that range |

A fifth source gives the shape but not numbers we could read: for washed spinach in a spin dryer, "the higher the
centrifugal speed, the less water excess was present at a given time, centrifugal speed having a greater influence
on water excess than operation time" (0 to 1080 rpm, 1 to 9 minutes;
[Pirovani et al., J Food Quality 2003;26:231-242](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-4557.2003.tb00241.x)).
That is the shape the rule has: a limit set by the speed, approached with time.

What this adds up to:

- The rule agrees with the one household measurement, on the assumption about that spinner's speed stated in the
  table.
- It overstates drying in a packed basket. In the industrial full baskets about 60% came off, at a push well above
  anything in this task; the runs of 5 to 15 minutes in the patent's second example took off no more than the
  2-minute run in its first. The same dryer with the leaves kept in a thin layer took off 81 to 89%. Water held between packed leaves does not leave the way the
  capillary argument says, which considers a leaf's open surface.
- So the rule describes leaves in a thin layer against the basket's wall. The task's 150 g in a 2-quart basket is
  somewhere between that and a packed basket, and nothing published says where. We have not measured it. The
  exponent (0.5) and the 2 s are therefore an argument and an estimate, and the simulator's score is a model of
  drying, not a measurement of it.
- A real trial does not depend on any of this: it is scored by weighing the lettuce. The procedure below measures
  the rule's two numbers on a real spinner, and they can be replaced in `task.py` without touching the scoring.

### Measuring the rule on a real spinner

Twenty minutes with the spinner, 150 g of torn lettuce, a kitchen scale that reads 1 g or better, a metronome and a
phone's slow-motion video gives both numbers:

1. Weigh the lettuce dry (D). Dunk it for 10 s, lift it out in the basket, let it drip for 10 s and weigh it wet
   (W). It carries W - D of water.
2. Crank at a steady pace (metronome) for 2 s, stop, let the basket stop, pour the water from the bowl into a cup on
   the scale and note the total. Repeat for running totals at 5, 10, 20 and 40 s. Film a few seconds in slow motion
   with a mark on the basket's rim in view, and count the basket's turns a second.
3. Do the same at two more paces (for example one crank turn every 2 s, one a second, two a second), re-wetting the
   same lettuce each time.
4. For each pace, the share spun out after 40 s is the limit at that push (the basket's turns a second and its
   radius give n): fit the exponent in `holds()` to the three limits. The time taken to get 63% of the way to each
   limit is `DRAIN_S`.

Put the two numbers in `task.py` (`HOLD_EXP`, `DRAIN_S`) and the table above follows from them. The scoring rule,
by share of water spun out, does not change.

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
- **An outcome that has to be reached, not a motion that has to be made.** The score asks how much of the water has
  been spun out of the lettuce with the lid on, which a slow or careless crank does not do.
- **Graded progress.** Partial credit separates a policy that leaves the lettuce wet (0), one that spins out a third
  of its water (0.25), one that spins out half (0.5) and one that spins out two-thirds and lets go (1.0).

## Files

| File | What it is |
|---|---|
| `task.py` | The task file: the spinner and its layout, the instruction, what happened during the trial, scoring |
| `spinnerkit.py` | Builds the salad spinner as two loose parts (the bowl with its basket; the lid with its crank and knob), works out how the lid sits on the bowl (`seat()`, `lid_on()`) and runs the gearing (`gear()`, every control step) |
| `demo.py` | The scripted demo, and its control runs |
| `test_scoring.py` | Offline tests of the gearing, the lid's fit, the scoring rule, the drying rule and the spinner's geometry (no simulator) |
| `task.json` | The submission's details |

## Requirements

The yam_arena embodiment and Isaac Lab-Arena, as set up by the program's GPU instance image (an NVIDIA RTX GPU is
needed). `test_scoring.py` needs only Python with numpy, pytest and the `yam_arena` package.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Spin the washed lettuce dry in the salad spinner: crank it clockwise fast enough and long enough to spin out at least two-thirds of the water the lettuce carries (the basket well over 5 turns a second for about 6 seconds), keep the lid on, then let go of the crank." \
  --scorer task_score --max-steps 1200
```

Run limit: 1,200 steps (60 s at 20 Hz). The demo reaches full score at step 465 (23 s), so the limit leaves room
for a slower start and a second try.

## Scripted demo

```bash
python demo.py task.py out_dir                  # the demo
python demo.py task.py out_dir --mode short     # control: the demo's pace, stopped after six turns
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

Tools: a kitchen scale that reads 1 g or better (0.1 g is better); the trial video; a ruler; a tape mark round the
bowl's base on the worktop.

- **Water carried in.** Before the trial, weigh the lettuce dry (D, about 150 g). Dunk it in water for 10 s, lift it
  out in the basket, let it drip for 10 s and weigh it wet (W). It carries W - D of water into the spinner.
- **Share spun out.** After the trial, weigh the lettuce again (E). The share of its water that has been spun out is
  (W - E) / (W - D). The water lying in the bowl, poured off and weighed, checks it. In the simulator there is no
  water: the share is worked out every control step from the basket's speed by the drying rule above, and never goes
  back down.
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
| Two-thirds or more of the water spun out, and let go at the end | **1.0** |
| Half or more of the water spun out | **0.5** |
| A third or more of the water spun out | **0.25** |
| Anything else | **0** |

Every outcome falls into exactly one row (the first that applies). Turning the crank earns nothing by itself: a
basket turned slowly leaves the lettuce wet however many turns it is given. The basket coasts on after the crank
stops and goes on spinning water out, so the share can still rise in the first moments after letting go. In code:
`_watch()`, `holds()`, `drain()`, `score_from()` and `score()` in `task.py`; the basket's speed and the lid's seat
come from `_mechanism()`, `seat()` and `lid_on()` in `spinnerkit.py`.

## Verification

All runs below use the submitted scoring code.

| Run | Score | Notes |
|---|---|---|
| Scripted demo | 1.0 | 13.1 crank turns at up to 1.3 turns a second. The basket peaked at 6.2 turns a second (374 rpm, 13.8 g at its wall) and spent 7.5 s at 5 turns a second or more: 71.5% of the water spun out, two-thirds reached at step 375. The lid stayed within 2.3 mm and 1.3 degrees of its seat, the bowl moved 1.5 cm, both hands let go. Full score at step 465 (23 s) |
| Short cranking (control) | 0.5 | The demo's pace, stopped after 6.1 turns: the basket peaked at 5.6 turns a second and was above 5 for 1.7 s. 60% of the water spun out: over half, short of two-thirds |
| Slow cranking (control) | 0.0 | 4.0 crank turns at about 0.4 turns a second with the bowl held and the lid on: the basket peaked at 2.5 turns a second (149 rpm, 2.2 g). 24% of the water spun out: under a third |
| Lid lifted (control) | 0.0 | The hand picked the lid up by the knob (13.8 mm, tipped 7.7 degrees), put it back and gave the same 13 turns: the lid had come off (step 215) |
| No holding hand (control) | 0.0 | The same cranking with the left hand kept away: the hand dragged the loose bowl 12.5 cm (knocked about at step 165). The basket peaked at 4.5 turns a second and 56% of the water had gone, which would have been 0.5 with the bowl in its place |
| Arms still (control) | 0.0 | Nothing moves; the lid stays on and no water leaves (run before this revision; a still basket spins nothing out under the new rule either) |
| Dream-1, 1,200 steps | 0.0 | Run with this revision's instruction. The right gripper came over the lid and hovered beside the crank; the left came to the bowl's handle side. Neither took hold, the crank was left where it was, and both arms drew back. No water was spun out |
| `pytest test_scoring.py` | 22 passed | The gearing (with the lid on and off), the lid's seat and its fit on the rim and lugs, the score tiers by share of water, the zero for a lost lid or a knocked bowl, the drying rule (the push at the wall and not the rpm, water only ever leaves, slow cranking never reaches a third, what each tier takes), the geometry |

What we learned building the demo, which shows what the task asks of a policy:

| Development run | What happened |
|---|---|
| Hand follows a pre-planned circle (first version, a quarter of a turn a second) | The knob slipped along the finger pads, the crank fell behind the hand and the knob came out after 2.7 turns |
| Aiming round where the layout puts the crank's axis | The bowl shifts about 1.3 cm in the holding hand, so the crank's real circle is not where the layout says; the hand has to find it from its own motion |
| Hand pointing straight down | On the far side of the crank's circle the wrist ran onto its joint stop |
| Aiming 1.0 rad ahead | Lost the knob; 0.8 rad keeps it |
| Fitted circle, hand held at the height it took the knob | Basket at 5.6 turns a second, but the lid lifted 7 mm and tipped 4 degrees: lid off, 0 |
| Leaning 12 mm into the knob, aiming 0.7 rad ahead | Lid stayed down (1.0 mm), but the basket peaked at 4.9 turns a second: too slow for two-thirds of the water |
| Leaning 6 mm into the knob, aiming 0.8 rad ahead (final) | 7.5 s at 5 turns a second or more with the lid on: 1.0 |

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
  Water is not simulated as a fluid: the share of it spun out is worked out from the basket's speed by the drying
  rule (a limit that goes as one over the square root of the push at the wall, in multiples of gravity, approached
  with a 2 s time constant). The rule is a physical argument and an estimate, not a measurement on a spinner; "The
  drying rule" above gives its sources, its limits and the kitchen procedure that would replace its two numbers.
  The basket's mass does not fall as water leaves.
- **The bowl's collision shape** is a solid cylinder: nothing has to reach inside it.

## Real-world build

- **The spinner:** any 2-quart hand-crank salad spinner with a crank on a lift-off lid and a handle or rim the
  gripper can hold (or a small handle added to the bowl). A tape mark round the bowl's base.
- **Each trial:** about 150 g of torn lettuce, weighed dry, dunked for 10 s, dripped for 10 s and weighed wet (see
  Scoring), in the basket; the bowl wiped dry; lid on, crank pointing away from the robot.
- **Scoring:** weigh the lettuce after the trial for the share of its water spun out; check on the trial video that
  the lid stayed on the rim; check the bowl against its tape mark and the grippers' distance from the knob at the
  end. No speed has to be measured to score a trial. To compare a real spinner with the simulator's drying rule, or
  to measure the rule's two numbers, see "Measuring the rule on a real spinner".

## Third-party assets

None. Every part is built from primitive shapes in `spinnerkit.py`.
