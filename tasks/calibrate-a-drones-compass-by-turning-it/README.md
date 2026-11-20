# Calibrate a drone's compass by turning it

A small folding camera drone stands on a landing pad, unfolded, with its propellers on. The instruction is
"Calibrate the drone's compass: pick it up, hold it level and turn it one full turn, then hold it vertically, nose
down or nose up, and turn it one full turn. Do not put it down in between. Then stand it back on its pad."

This is the compass calibration that the maker's manual asks for before flying at a new site. The drone is at full
size: 249 g, 138 mm long, 289 mm across its propellers.

## What the task asks of a robot

- **A full turn of a free object.** One wrist of these arms turns 240 degrees, and only a hand pointing straight
  down can use all of it to turn something about the vertical. A full turn therefore needs the drone passed to the
  other hand in the air while the first hand turns back, and then taken again. Putting it down to change grip is not
  allowed: the turn starts again.
- **Two attitudes, two grasps.** Held level, the drone is turned by a hand holding the sides of its fuselage from
  above. Held vertically, it hangs from a hand holding the top and bottom of its tail. Going from one to the other
  is a third exchange.
- **Nothing fixes the drone.** Whichever hand has it carries its whole 249 g, sometimes by the tail with the weight
  45 mm out from the pads. The hand that comes in has to close on the drone where the other hand's sag has left it.
- **Narrow places to hold.** The fuselage is 58 mm wide and 40 mm tall, and its sides are free for about 110 mm
  between the arms. The propellers, arms, legs and camera are in the way of every approach.
- **Finishing.** The drone has to be stood back on its pad, on its feet, and let go.

Both arms are used throughout: each is a turner in one of the two turns and the holder in the other.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `dronekit.py` | The drone as one rigid body of boxes. |
| `demo.py` | A scripted demo that reaches full score, and six control runs. |
| `test_scoring.py` | 55 checks of the scoring rule, the turn counter, the drone's shape, both layouts and the demo's grasps; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 3,000 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 3000 \
  --instruction "Calibrate the drone's compass: pick it up, hold it level and turn it one full turn, then hold it vertically, nose down or nose up, and turn it one full turn. Do not put it down in between. Then stand it back on its pad."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
DRONE_LOT=B python demo.py task.py out_dir     # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the drone starts standing on the pad.
- There are no seeds. `DRONE_LOT` picks the layout: A (pad at x 26, y -7 cm, nose pointing 270 degrees, toward the
  robot's right) or B (pad at x 23.5, y -10 cm, nose pointing 180 degrees, toward the robot).
- A trial is at most 3,000 steps (150 s). It ends early once full score has held for 1 s. The demo takes 2,077
  steps (104 s) on layout A and 2,031 steps (102 s) on layout B; a simulated trial of that length took 8 to 11
  minutes of wall time on a GPU shared with other jobs, part of it start-up.
- `result.json` from the demo holds the score, the task's own report at the start and end (directions covered in
  each turn, the steps at which each turn was finished, whether the drone stands, is on its pad and has been let
  go), the demo's plan, and a trace every 5 steps.

## Scoring

Directions are compass directions seen from above, in 36 ten-degree sectors.

- A turn counts over one **unbroken hold**: from when the drone's lowest point rises 2 cm above the table until it
  comes back within 2 cm of it.
- The **level turn** is done when the nose has pointed into all 36 sectors while the drone's top was within
  20 degrees of straight up.
- The **upright turn** is done when, after the level turn and in the same hold, the drone's top has faced all 36
  sectors while its nose-to-tail line was within 20 degrees of plumb (either end down).
- If the drone comes down before a turn is finished, that turn starts again from nothing in the next hold. A level
  turn already finished keeps its credit, but an upright turn only counts in the hold in which the level turn was
  made.
- A **drop** is coming within 2 cm of the table faster than 0.25 m/s.
- At the end the drone **stands** if its top is within 10 degrees of straight up, its lowest point is within 3 mm
  of the pad's top or of the table, and it is at rest (under 2 cm/s and 0.5 rad/s).

| Score | What it takes |
|---|---|
| 1.0 | Both turns done, never dropped, and at the end the drone stands with all four feet over the pad and both grippers at least 3 cm from it. |
| 0.75 | Both turns done, never dropped, and the drone stands at the end, but off the pad or with a gripper still within 3 cm. |
| 0.5 | Both turns done, but the drone was dropped, or is not standing at the end (still in a hand, or on its side). |
| 0.25 | The level turn done, the upright turn not. |
| 0 | The level turn not done. |

Other ways to try it, and what they score: spinning the drone on the table, 0 (it is not held); turning a wrist back
and forth, 0 (240 degrees covers 24 sectors however often it is done); standing the drone down to change grip
half way round, 0 (the turn starts again); the upright turn first, 0; holding the drone 30 degrees off level, 0.

## Scoring a real trial

From the video and one photograph at the end:

- Level turn: with the drone clear of the table and level, follow its nose once round. A protractor drawn on the
  table, or the 36 marks of a printed dial under the work area, shows whether any sector was missed.
- Upright turn: the same with the drone held vertically, following the way its top faces.
- The drone was never set down between lifting it and the end of the upright turn.
- At the end: the drone on its feet, all four on the pad, both grippers 3 cm away (a tape measure).

On a real drone the status light is the judge: the manual says it is solid yellow when calibration starts, turns
solid green after the level turn, and blinks red if calibration fails. The simulated drone has no light and no
compass; its score comes from its measured motion, by the rule above. The task has not been run on a physical rig.

## The demo

`demo.py` is **open loop**. It plans from the layout constants in `task.py` and the sizes in `dronekit.py`, keeps
its own account of where the drone should be from what it has commanded, and never reads the drone's state. The
only feedback is each hand's own measured position and direction (the robot's proprioception), used to correct for
the sag that the drone's weight gives the arm holding it.

How far each wrist can turn, where the exchange can be made and which way round each hand holds are worked out by
the script from the arms' kinematics, so the two layouts come out differently.

1. The right hand comes straight down over the drone, takes the fuselage by its sides (pads 58 mm apart), lifts it
   and carries it to where both layouts are turned (x 27, y -7 cm), its middle 13.5 cm above the table.
2. Level turn. The right wrist turns the drone to one end of its travel and then 200 degrees to the other. The left
   hand, level, comes in behind the tail and takes the fuselage top and bottom, 20 mm in from the tail's end. The
   right hand opens, rises 5.5 cm, turns back half a turn, comes down and takes the sides again. The left hand
   opens and draws back 10 cm. The right wrist turns the drone on 180 degrees: 380 degrees in all.
3. The right wrist turns the drone back to where the left hand can reach its tail. The left hand takes the tail,
   the right lets go and withdraws, and the left hand swings from level to pointing straight down while carrying
   the drone across: the drone ends hanging nose down, its middle 15.5 cm above the table.
4. Upright turn. The left wrist turns the hanging drone 200 degrees. The right hand, level, takes its middle. The
   left hand opens; because a hand pointing straight down cannot rise any further here, the right hand lowers the
   drone 3.8 cm, the left hand turns back half a turn above it, and the right hand raises it again. The left hand
   closes, the right lets go and withdraws, and the left wrist turns the drone on 180 degrees.
5. The left hand turns the drone to where it can swing it back to level, and swings it. The right hand takes the
   sides from above, the left lets go, and the right hand turns the drone square to its pad, stands it down and
   lets go.

| | Layout A | Layout B |
|---|---|---|
| Right hand closed on the sides, drone still on its pad | step 110 | step 110 |
| Drone lifted and carried to where it is turned; level turn begins | step 170 | step 170 |
| Left hand has the tail: both hands on the drone | step 420 | step 375 |
| Right hand has let go, turned back and taken the sides again | step 540 | step 495 |
| Level turn done (all 36 directions): score 0.25 | step 636 | step 591 |
| Left hand has the tail again, for the change to vertical | step 830 | step 785 |
| Drone hanging nose down; upright turn begins | step 980 | step 930 |
| Right hand has the middle: both hands on the drone | step 1,290 | step 1,245 |
| Left hand has let go, turned back and taken the tail again | step 1,430 | step 1,385 |
| Upright turn done (all 36 directions): score 0.5 | step 1,567 | step 1,522 |
| Drone level again in the left hand | step 1,775 | step 1,730 |
| Right hand has the sides, from above | step 1,880 | step 1,835 |
| Drone standing on its pad, right hand open | step 2,030 | step 1,980 |
| Hands clear: score 1.0 | step 2,040 | step 1,995 |
| Trial ends with full score | step 2,077 | step 2,031 |
| Drone's top off straight up during the level turn, at most | 2.4 degrees | 2.4 degrees |
| Nose-to-tail line off plumb during the upright turn, at most | 1.8 degrees | 2.9 degrees |
| Drone's lowest point between lifting and standing down, at least | 39 mm | 37 mm |
| Drone away from where the script expected it while held, at most | 18 mm | 18 mm |

(Steps of "turn done" are exact; the others are read from a trace written every 5 steps.)

The largest gap between the drone and where the script expected it, 18 mm, comes for half a second each time the
left hand closes on the tail: its upper pad meets the fuselage first and presses the drone down in the right
hand's grip until the lower pad arrives. The right hand's own correction brings it back.

After the pick the two layouts are turned at the same place, so their plans agree: the exchange of the level turn
is made with the nose pointing 335 degrees and that of the upright turn with the drone's top facing 290 degrees.
They differ in the pick, in how far the drone is turned before the first sweep (135 degrees in layout A, 45 in
layout B) and in the way back to the pad.

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves. The drone stays on its pad. |
| `--mode short`: the level turn stopped at the exchange, then everything else as in the demo | 0.0 | The nose pointed into 21 of the 36 sectors. The upright turn was then made in full, but does not count without the level turn. The drone ended standing on its pad. |
| `--mode tilted`: the level turn made with the drone held 30 degrees nose down | 0.0 | In the first sweep the drone's top was 29 to 31 degrees off straight up and no sector counted. After the exchange the right hand could not reach the tilted poses asked of it and the drone hung between 2 and 27 degrees off; 14 sectors counted where it was under 20. 16 of 36 in all. |
| `--mode down`: the drone stood on its pad for the level turn's exchange, instead of being passed in the air | 0.0 | 21 sectors before it was stood down, which ended the hold and the count. Picked up again and turned on, 22 in the second hold: neither covered all 36. The upright turn that followed does not count. |
| `--mode level_only`: the drone put back on its pad after the level turn | 0.25 | Level turn done at step 636. No upright turn. |
| `--mode off_pad`: the full demo, but the drone stood on the bare table 21 cm from the pad's centre | 0.75 | Both turns done (steps 636 and 1,567), the drone standing at rest and let go, none of its feet on the pad. |

## Dream-1

One run, 3,000 steps, layout A (after a 200-step test, which also scored 0): **0.0**. The right arm went to the
drone several times, reaching over its fuselage and at the propellers on its right side, and drew back each time.
The left arm folded down beside its own base and stayed there. The drone was not lifted in any frame examined, and
at the end it stood on its pad where it had started.

## Building it for real

- A sub-250 g folding camera drone, unfolded, propellers on, battery in (249 g). The model is 138 mm long folded and
  289 mm across its propellers. Remove the gimbal protector, as for a flight.
- A square rubber mat, 24 x 24 cm and 4 mm thick, fixed to the table, with a mark for the nose.
- A dial of 36 sectors printed under or beside the work area, for scoring from the video.

Resetting: stand the drone back on the pad with its nose on the mark. After a drop, check the propellers, the legs
and the camera.

Safety: the drone is not powered for flight in a trial and its motors never run. If the real calibration is to be
checked by the status light, the drone is switched on with the calibration started from the app, which does not
start the motors.

## What the simulation simplifies

- The drone is one rigid body made of boxes.
- **Sourced** (the maker's user manual): takeoff weight under 249 g; folded 138 x 81 x 58 mm; unfolded with
  propellers 245 x 289 x 56 mm.
- **Estimates, not measurements** (from photographs; no drawing was found): the fuselage's 58 x 40 mm section; the
  14 mm camera block at the nose; where the arms leave the fuselage (62.5 mm ahead of and behind its middle) and
  where the motors are (67.5 mm fore and aft, 89.5 mm out); the 110 mm propellers; the legs; friction (0.9 between
  the fuselage and a gripper pad, 0.45 between the drone and the pad or table); the patch of 8 mm radius over
  which a pad meets the fuselage, which is what resists the drone's twisting in a two-pad hold.
- **The 20-degree limits** on "level" and "vertical", and accepting either end down in the upright turn, are this
  task's choices. The manual says "horizontally" and "vertically" and gives no tolerance.
- The propellers are fixed, with their blades pointing out sideways, which makes the model 289 mm wide and 147 mm
  long. Real propellers turn freely and their blades fold; with the blades fore and aft the real drone is 245 mm
  long.
- The camera, its gimbal and the legs are rigid blocks. The scoring does not check whether a gripper touched them;
  the demo's grasps are on the fuselage only.
- There is no status light, no compass and no magnetic field. The score follows the drone's measured motion.
- The pad is fixed to the table.

## Related tasks on the board

No accepted or pending task turns a free object through a full turn, or changes the hand that holds it more than
once (board checked 19 November 2026).

- "Replace a drone's chipped propeller with the new one and tighten its prop nut" (accepted) is the other drone
  task. It is a fastening task on a drone that stays where it is; here the drone itself is the thing handled.
- "Arm-to-arm handover" (pending) passes a sponge once from one gripper to the other to carry it across. Here the
  exchanges are what make a full turn possible: three of them, with three kinds of grasp, on a rigid 249 g body
  that must stay level or plumb through each.
- "Turn a self-centring steering wheel to full lock hand over hand, hold it, then let it spring back to centre"
  (pending) also takes more than one wrist's travel. The wheel is on a fixed shaft: the hands walk round a part
  that cannot fall or tilt. Here the part is free and is carried by whichever hand has it.
- "Close a wall belt barrier: draw the belt, pass its end hand to hand against the spring, and click it into the
  receiver" (pending) has one exchange, of a belt end under tension.
- "End-of-day chairs up: lift a preschool chair, turn it over and set it seat down, legs up, on the table"
  (pending) turns a free object over once, within the reach of one hold.

## Sources

- The maker's user manual for a sub-250 g folding camera drone (DJI Mini 2 User Manual v1.0, 2020.11,
  dl.djicdn.com/downloads/DJI_Mini_2/DJI_Mini_2_User_Manual-EN.pdf), read directly. Page 48, "Calibrating
  the Compass": "Hold the aircraft horizontally and rotate it 360°. The aircraft status indicator will turn solid
  green." and "Hold the aircraft vertically and rotate it 360° around a vertical axis."; it is recommended when
  "Flying at a location further than 31 miles (50 km) away from the location the aircraft was last flown", when
  "The aircraft has not been flown for more than 30 days", or on an interference warning. Specifications: "Takeoff
  Weight < 249 g"; "Folded: 138×81×58 mm"; "Unfolded (with propellers): 245×289×56 mm". Page 7: "Make sure the gimbal
  protector is removed and all arms are unfolded before powering on the aircraft."
- The United States aviation regulator's "Drones by the Numbers (as of November 2025)" (faa.gov/node/54496), read
  directly: "837,513* Total Drones Registered", "481,760 Certificated Remote Pilots".
- Neither gives the size of the fuselage, where its arms are, or a tolerance for "horizontally" and "vertically".
  Those are estimates and choices (see "What the simulation simplifies").
