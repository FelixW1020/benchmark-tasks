# Put a blood-draw needle into a training arm's vein

An adult venipuncture training arm lies strapped down, palm up, on the table. Beside it a single-use tube holder with
a 21 gauge, 1 inch needle on it lies in a cradle, uncapped and bevel up. The robot has to do the stick of a routine
blood draw: hold the vein with one hand on the forearm below the site, and with the other put the needle into the vein
at the elbow crease, bevel up, at a shallow angle, stop when the tip is in the vein and keep it still.

Everything is at full size. Nothing marks the site on the skin; the vein is drawn in blue.

## What makes it hard

- **A small target in three directions.** The vein is 4 mm across and its middle is 3.5 mm under the skin. At 25
  degrees the tip is in it from 3.5 mm to 13 mm of needle; further and it is through the far wall. The needle has to
  cross the skin within 8 mm of the vein's line, and a tip more than 2 mm to the side of the vein's middle misses it.
- **An attitude, not just a place.** 10 to 30 degrees to the skin, pointing up the arm, bevel up.
- **Two hands 13 cm apart.** The vein has to be held before the needle reaches it and until the end, and the uncapped
  needle may never come within 15 mm of the hand that holds it.
- **A tool taken from a rest.** The holder is a 19 mm barrel lying in a V groove; how it sits in the fingers decides
  where the needle's tip is, 6 cm ahead of them.
- **Stopping.** The stick ends with the needle held still in the vein, not with letting go.

## The needle and the vein are rules, not physics

This has to be said plainly, because the score rests on it.

- **The needle has no collider.** A 0.8 mm shaft cannot be simulated as a body here, and nothing models skin being
  pierced. The needle is drawn; the holder and its hub are solid. Where the needle crosses the skin, at what angle,
  with which side up, how deep its tip is and whether its shaft is swept sideways in the skin are all read from the
  holder's pose every control step (`veinkit.Stick`). The skin does not push back, so a needle held at the wrong
  angle is not stopped: it goes in, and the trial scores 0 for a bad stick.
- **The vein rolling aside is a rule.** If no fingertip of the other gripper is on the forearm's skin in the anchoring
  window at the moment the tip gets as deep as the vein's near wall, the vein is taken to have moved 5 mm to the side,
  away from the needle, and it stays there while the needle is in. Why a rule: the reason the second hand is needed
  in a real draw is what skin and vein do under traction (the manual quoted below: "to prevent the vein from
  moving"; the training arm's listing: "life-like durable vein roll-over"), and nothing here simulates skin. A rigid
  arm would not need holding, so a model in which the arm itself had to be steadied would have been dishonest in the
  other direction. The 5 mm, and that the vein always rolls when it is not held, are choices, not measurements: a real
  vein rolls sometimes.
- **The anchoring fingertip is 8 to 14 cm below the site, not the 2 to 5 cm a thumb is.** The holder is 7 cm long
  behind the skin and lies almost on the forearm at these angles, and the hand on it fills the space above. A
  gripper's fingertips fit only below that.

## Files

| File | What it is |
|---|---|
| `task.py` | The layout, the instruction, the record kept every control step, the scoring rule. |
| `veinkit.py` | The training arm, the holder and needle, the cradle, and the needle's rule (`Stick`). |
| `demo.py` | A scripted demo that reaches full score, and the control runs. |
| `test_scoring.py` | 21 checks of the shapes, the needle's rule and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 600 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 600 \
  --instruction "Put the needle into the training arm's vein at the elbow crease for a blood draw: hold the vein with your other hand on the forearm below the site, go in bevel up at a shallow angle, and stop when the needle is in the vein. Keep it still there."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
VEIN_LOT=B python demo.py task.py out_dir      # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the arm and the cradle are fixed and the layouts are exact.
- There are no seeds. `VEIN_LOT` picks the layout: A, or B (the arm 1 cm nearer the robot and 1.5 cm to the left, the
  cradle 1 cm further and 2 cm to the right).
- A trial ends at full score held for 1 s, or at the run limit of 600 steps (30 s).
- Time per trial on one L40S: about 4.5 minutes for the demo (372 steps, with start-up) while three other runs shared
  the GPU; 6 minutes for the 600-step Dream-1 run. No run was timed alone.
- `result.json` from the demo holds the score, the needle's record and a trace every five steps.

## Scoring

Scored at the end of the trial from the needle's record.

| Score | State at the end |
|---|---|
| 0 | The needle is not in the skin, or the trial is void (below). |
| 0.25 | The needle is in the skin (tip at least 1 mm under it) after a good stick, but its tip is not in the vein. |
| 0.5 | Its tip is in the vein, but the holder is not in a gripper, or the tip moved 0.5 mm or more in the last second, or the needle is not at 5 to 30 degrees to the skin. |
| 0.75 | In the vein, held in a gripper, still, at 5 to 30 degrees; but the vein was let go at some time after the tip reached it. |
| 1.0 | In the vein, held, still, at 5 to 30 degrees, and the vein held ever since the tip reached it. |

The numbers:

- **A good stick:** the needle crosses the skin within 8 mm of the vein's line and 12 mm of the elbow crease, at 10 to
  30 degrees to the skin, with its bevel facing away from the skin within 30 degrees.
- **In the vein:** the tip is inside the vein's lumen, 4 mm across, its middle 3.5 mm under the skin on the vein's line
  (or 5 mm to the side of it, if the vein was not held when the tip reached it).
- **The vein is held** while a fingertip of the gripper that is not holding the holder is on the forearm's skin
  (within 2.5 mm of it), 8 to 14 cm below the crease and within 2 cm of the vein's line.
- **Still:** the tip has moved less than 0.5 mm over the last second.

The whole trial scores 0, whatever the end state, if at any time:

- the needle (tip to hub) came within 15 mm of the gripper that is not holding the holder: a needlestick;
- the needle touched (came within 1.5 mm of) the table, or the arm away from the site, before it went in;
- it made a bad stick: into the skin away from the site, outside 10 to 30 degrees, or with the bevel not up;
- its shaft was swept more than 2 mm sideways where it crosses the skin;
- its tip went deeper than the vein's far wall under the vein: through and through.

Doing nothing scores 0. The score depends on the state at the end and on those events; nothing is added up over time.

Ways through that were considered, and what they score:

- One hand only, no anchor: 0.25. The stick is good but the vein has rolled aside (run below).
- Anchoring only once the needle is in: 0.25. The vein rolled when the tip reached it.
- Going in steep to be sure of reaching the vein: 0 (bad stick; run below). Going in deep: 0 (through; run below).
- Going in anywhere on the arm and steering the tip to the vein under the skin: 0 (away from the site, or swept).
- Letting go of the vein once the needle is in: 0.75 (run below). Letting go of the holder: 0.5.

## Scoring a real trial

Tools: a phone on a stand at the arm's side, level with the elbow, with a protractor card behind the site; a second
phone above; a ruler.

1. Site: on the overhead video, the needle goes into the skin within 8 mm of the vein's line and 12 mm of the crease
   (rule a faint pencil cross on the skin after the trial, not before).
2. Angle and bevel: on the side video, the needle is 10 to 30 degrees to the skin as it goes in; the hub's mark is up.
3. In the vein: the training arm gives a flash of blood in the needle's hub when the tip enters a vein (its listing:
   "Realistic flashback"); no flash, or a flash that stops, is not in the vein. Through and through shows the same way.
4. Held: on the overhead video, a fingertip is on the forearm 8 to 14 cm below the crease, within 2 cm of the vein's
   line, from before the flash to the end.
5. Still: the needle's tip end does not move in the last second of the side video (0.5 mm is less than a needle's
   width; on video, no visible movement).
6. Void: on either video, the needle touches the table or the arm before the site, or passes within 15 mm of the
   other gripper (hold the ruler in the frame for scale).

On the real arm the vein's rolling is the arm's own behaviour, not the rule used here. This procedure has not been
tried on a physical rig.

## The demo

`demo.py` reaches 1.0 on both layouts. It never reads a part's state: every target comes from the layout constants
and the kit's sizes, and the only feedback is each hand's own measured position, used to correct for the arm's sag.

The right hand anchors and the left hand sticks. The right hand's closed fingertips lean 45 degrees toward the wrist
and press on the forearm 13 cm below the crease. The left hand takes the holder from above, across the barrel,
fingers pointing straight down; with the needle pointing up the arm it leans 25 degrees, so that the needle is 25
degrees below level and the hub's mark is up.

| Step (layout A) | What happens |
|---|---|
| 90 | The right hand's fingertips are on the forearm: the vein is held. |
| 115 | The left hand, having corrected for its sag over the cradle, has closed on the barrel. |
| 140 | The holder is lifted 10 cm out of its cradle. |
| 215 | It is over the arm, tipped: the needle points up the arm, 25 degrees below level. |
| 260 | The tip is 2 cm short of the skin on the needle's own line. |
| 325 | The needle has gone in along that line (it crossed the skin at 24.6 degrees, bevel 0.9 degrees off, 0.6 mm from the vein's line and 1.3 mm past the crease); the tip is in the vein: 0.5. |
| 355 | The tip has been still for a second, 3.1 mm under the skin and 0.5 mm from the vein's line: 1.0. |
| 372 | Full score has held for 1 s: the trial ends as a success. |

Layout B: the same steps to 325 (crossing at 24.5 degrees, 0.5 mm from the line), 1.0 at 375 with the tip 2.5 mm
under the skin, success at 393. Steps are read from a trace kept every five steps. In both runs the needle's nearest
approach to the right gripper was 46 mm, and it touched nothing before the skin.

The tip is sent 4.0 mm under the skin and ends 2.5 to 3.1 mm under it: the holder sits about a millimetre back in
the fingers from where the plan has it. The vein is 1.5 to 5.5 mm under the skin, so that is inside it with 1 mm to
spare on the shallow side. Once the needle is in, the left hand's command is left alone; while the hand was still
correcting for sag there, in a first run, the tip wandered a millimetre and ended up through the far wall.

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves (200 steps). |
| `--mode noanchor` (the same stick, the right hand left at rest) | 0.25 | A good stick at 24.6 degrees, but the vein was not held when the tip reached it (step 325): it is 5 mm aside and the tip, 3.1 mm under the skin, is not in it. |
| `--mode release` (the right hand lifts off as soon as the needle is in) | 0.75 | In the vein at step 325; the fingertips left the skin by step 345; still from step 355. |
| `--mode steep` (the same stick at 45 degrees) | 0.0 | The needle went in at 44.8 degrees: a bad stick (void from step 326). |
| `--mode through` (18 mm of needle at 25 degrees) | 0.0 | In the vein at step 315, through its far wall at step 326, ending 6.5 mm under the skin. |
| `VEIN_BEVEL=down` (the holder starts turned over; the same moves) | 0.0 | The needle went in with its bevel 179 degrees off: a bad stick (void from step 321). |

The first version of the release run let go 1.5 s after the needle was in; by then the trial had already ended as a
success (step 372), so the run above lets go at once.

## Dream-1

One run, 600 steps, layout A, after a 200-step test (also 0.0): **0.0**. Both arms reached out over the training
arm: the left gripper came to rest over the upper arm, near the cradle, and the right gripper over the hand end of
the forearm. The holder was never taken out of its cradle, so the needle did not come near the skin.

## Building it for real

- **Training arm.** An adult venipuncture and injection training arm with a vein at the elbow crease and flashback:
  the Life/form Venipuncture and Injection Training Arm (LF00698U). Strap it to the table palm up, shoulder end to the
  robot's left. Its listing gives no arm, vein or weight figures: measure the arm, find its vein at the crease, and
  set the site, the anchoring window and the heights from what is measured.
- **Holder and needle.** A single-use tube holder for 13 and 16 mm tubes with a 21 gauge, 1 inch multi-sample needle
  screwed in, the sheath taken off just before the trial by the person setting up.
- **Cradle.** Any block with a V groove that holds the barrel level with its axis 3 cm above the table, needle
  pointing left, hub mark up, and leaves the barrel's sides free.
- **Safety.** The needle is sharp and uncapped. Use a new sterile needle on a training arm only, keep people out of
  the arms' reach during a trial, and discard the needle in a sharps container afterwards without recapping it.

## What the simulation simplifies

- The needle and the vein, as said above: rules, not physics.
- **Uncapping is not part of the task.** The needle starts uncapped in its cradle; in practice the sheath comes off
  just before the stick.
- **No tourniquet, no cleaning of the site, no tube.** The site is taken as prepared; the stick ends before a tube is
  pushed on.
- **The arm** is rigid and fixed, built from round sections: upper arm 8.4 cm across at the elbow to 10 cm, 27 cm long, with a 2 cm shoulder cap; elbow 8.4 cm, level for 6 cm either side of the crease; forearm narrowing to 5.8 cm at the wrist over 19 cm; a flat hand 18 cm long, 2.6 cm thick, with a thumb. All of these are estimates of an adult arm,
  not measurements of the named training arm. The vein is straight, 4 mm across and 3.5 mm deep (estimates).
- **The holder**: barrel 19 mm across and 48 mm long, flange 34 x 24 x 2 mm, hub 10 mm long (estimates of a
  single-use holder for 13 and 16 mm tubes); needle 21 gauge, 1 in (the catalogue designation: 0.8 mm across, 25.4 mm
  long). Holder and needle weigh 5 g here (an estimate); the simulated body weighs 20 g and a steady upward force
  carries 15 g of that, so that it rests in the fingers as a 5 g part does while the solver has a heavier body.
- **Touching** (the needle against the table or the arm, the fingertip on the skin, the needle near a gripper) is
  judged by distance once per control step (20 times a second), from the kit's shapes and from sample points on the
  grippers' fingers and palm.
- Friction: 0.6 between the holder and the grippers' pads, 0.8 for the skin (estimates).

## Related tasks on the board

- "Fill a blood tube on a needle already in a training arm's vein and take it off without moving the needle"
  (accepted) starts with the needle in the vein. This task is the step before it: getting the needle there.
- "Give a 1 mL intramuscular injection into a deltoid trainer with a retractable safety syringe" (accepted) puts a
  needle in at 90 degrees, anywhere in a muscle site found from a landmark, and pushes a plunger. Here the needle goes
  in at a shallow angle into a 4 mm vein that the other hand has to hold.
- "Fill four tubes and tick each one off" (accepted) fills sample tubes with a transfer pipette and keeps count; no
  needle and no arm.
- A pending task whose title begins "Fingerstick a trainer hand: twist the lancet's tab off, hold the ring finger"
  lances a fingertip while the finger is held; no vein is entered and no needle is held at an angle and a depth.

## Sources

- O*NET OnLine, 31-9097.00 Phlebotomists: "143,900 employees" (2025); "Draw blood from veins by vacuum tube, syringe,
  or butterfly venipuncture methods." 31-9092.00 Medical Assistants: "833,900 employees" (2025); "Collect blood,
  tissue, or other laboratory specimens, log the specimens, and prepare them for testing."
- MSD Manual Professional Edition, "How To Do Venous Blood Sampling"
  (msdmanuals.com/professional/critical-care-medicine/how-to-do-peripheral-vascular-procedures/how-to-do-venous-blood-sampling):
  "Apply gentle traction to the vein distally using the thumb of your nondominant hand to prevent the vein from
  moving." "Insert the needle proximally (ie, in the direction of venous blood flow), with the bevel facing up,
  along the midline of the vein at a shallow angle (about 10 to 30 degrees) to the skin." "Blood will appear in the
  needle hub (called a blood flash ) when the needle tip enters the lumen of the vein." "Take care not to puncture
  too deeply and go through the vein." "Keep the needle motionless."
- Training arm: a distributor's listing of the Life/form Venipuncture and Injection Training Arm, LF00698U
  (boundtree.com/training-simulation/trainers/life-form-venipuncture-and-injection-training-arm/p/group004652):
  "35in x 12in x 6in"; "Veins are located at common sites, including the antecubital fossa, along the forearm, and at
  the back of the hand."; "life-like durable vein roll-over"; "Realistic flashback". It gives no weight and no arm or
  vein sizes.
- The World Health Organization's "WHO guidelines on drawing blood: best practices in phlebotomy" is the usual
  reference for these steps; its pages could not be fetched when this was written, so nothing here is quoted from it.
