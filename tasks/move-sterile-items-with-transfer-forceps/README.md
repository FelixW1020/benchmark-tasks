# Move two sterile medicine cups onto a sterile field with thumb forceps

A treatment-room counter is being set up for a minor procedure. A Mayo instrument tray under a sterile drape is the
sterile field. Beside it lies a sterile supply pack that has been opened flat, with two stainless medicine cups on the
inside of its wrapper. Sterile thumb dressing forceps lie across the near edge of the field, their handle sticking out
over the edge. The robot has to move both cups onto the field with the forceps and lay the forceps back down.

The robot's grippers stand for a nurse's or medical assistant's bare hands: they are not sterile.

## Why it needs the tool, and what makes it hard

Sterile technique as nursing and medical-assisting texts teach it (sources at the end) comes down to four rules, all
of them scored here:

- **Sterile touches only sterile.** The cups, the working end of the forceps and the inside of the field and the pack
  are sterile. The grippers, the table and the forceps' handle are not. Whatever sterile thing touches an unsterile
  one is contaminated, and contaminates what it touches next.
- **The outer 2.5 cm (1 in) of a sterile field is not sterile.** A cup has to stand wholly inside that border.
- **Nothing unsterile passes over the field.** No part of either arm may be above the sterile part of the field or of
  the pack, at any height. Only the forceps reach in.
- **Forceps are held tips down.** While they are held, the tips may not be raised above the handle.

So the gripper cannot take a cup, cannot hover over one, and cannot hold the forceps the easy way (straight down from
above, which puts the hand over the field). What works is what a person does:

- Take the forceps by the handle that sticks out over the edge and hold them as an extension of the fingers, pointing
  forward and 44 degrees down, so that the tips are 6.5 cm ahead of the fingertips and 7.6 cm below the hand's tool
  point.
- Work them by squeezing: the pads' gap sets the tips' gap (3.2 mm at the pads moves the tips 15 mm), and the pads'
  force sets the grip (about 16 N on each pad gives under 3 N at the tips; worked out from the fingers' position
  gain and the lever, not measured).
- Take each cup by its wall, one tip inside and one outside just under the rolled rim, reaching in from the front of
  the pack. A 35 g steel cup held at one point of its rim hangs tipped (22 to 30 degrees in the demo runs); it has to
  be set down from that tilt and rights itself.
- Lay the forceps back with their working end on the field and their handle out over the edge.

## Files

| File | What it is |
|---|---|
| `task.py` | Layout, the contamination record kept every control step, the scoring rule. |
| `xferkit.py` | The forceps (two bowed blades on a sprung hinge), the cup, the arms' shapes read from the robot's URDF. |
| `demo.py` | A scripted demo that reaches full score, and four control runs. |
| `test_scoring.py` | 35 checks of the shapes, the contamination record and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,800 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1800 \
  --instruction "Using the sterile forceps, move both medicine cups from the opened sterile pack onto the sterile field without contaminating anything, then lay the forceps back across the field's edge and let go."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
XFER_LOT=B python demo.py task.py out_dir      # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the field and the pack are fixed and the layouts are exact.
- There are no seeds. `XFER_LOT` picks the layout: A or B (the cups 1 to 2.5 cm from where they stand in A, the forceps
  1 cm to the left).
- A trial of 1,800 steps is 90 s of robot time. The demo takes 1,126 steps (56 s). On a shared GPU a demo run took about
  15 minutes of wall time.
- `result.json` from the demo holds the score, what was contaminated and why, and a trace every five steps.

## Scoring

Scored at the end of the trial, from a record kept at every control step.

| Score | State at the end |
|---|---|
| 0 | The field is contaminated, or no cup stands on it. |
| 0.25 | One cup stands on the field's sterile part, uncontaminated, let go, at rest. |
| 0.5 | Both cups do. |
| 0.75 | Both cups, and the forceps have been put down, let go and are at rest. |
| 1.0 | Both cups, and the forceps lie uncontaminated with their whole working end resting on the sterile part of the field or of the pack. |

The numbers:

- **Touching** is being within 1.5 mm; for any part of an arm, within 4 mm.
- **Sterile:** the two cups; the forceps' working end (the 5 cm from the tips); the tops of the field and of the pack
  more than 2.5 cm from their edges.
- **Contaminated** is anything sterile that touches the table, a 2.5 cm border, the tray's side, an arm, the forceps'
  handle (the 6.5 cm from the tail), or anything already contaminated. The field or the pack is also contaminated when
  any part of an arm is above its sterile part. The forceps are also contaminated when, while they are held (an arm
  within 3 cm of them), their tips are more than 5 mm higher than the end of their handle.
- **A cup stands on the field** when it is upright within 10 degrees, on the field's surface, with its whole outline
  inside the border. **Let go:** both arms 3 cm from it and the forceps' tips 2 cm from it. **At rest:** under 2 cm/s.
- **The forceps are put down** when both arms are 3 cm from them and they are at rest.

"Any part of an arm" means the collision shapes of every link from the upper arm to the fingertips, both arms,
placed where the simulator has them (`xferkit.arm_points`). A contaminated field scores 0 whatever stands on it:
in practice the set-up is thrown away and started again.

Unintended ways through, and what they score: carrying a cup in the gripper, 0 (run below); pushing or dragging a cup
across with the forceps, 0, because it crosses the pack's border, the table and the tray's side; reaching deep into
the field with the forceps, 0, because the hand follows them over it (run below).

## Scoring a real trial

From an overhead video (which shows whether an arm was above a sterile part: mark the 2.5 cm borders on the drape and
the wrapper with tape under the cloth, or measure them on the video) and a side view (touches, and whether the tips
were raised). The end state is read from the last frame with a ruler. This has not been tried on a physical rig.

## The demo

Only the right arm works. Its only feedback is its own measured tool position, used to correct for the arm's sag; it
never reads where a cup or the forceps are.

| Step | What happens (layout A) |
|---|---|
| 115 | The hand, leaning 72 degrees forward, has closed on the handle; the tips are 6.5 mm apart. |
| 180 | The handle has been lifted and the forceps drawn back: they point 44 degrees down. |
| 345 | The tips close on the first cup's wall. |
| 510 | The first cup stands on the field: score 0.25. |
| 705 | The tips close on the second cup's wall. |
| 860 | The second cup stands on the field: score 0.5. |
| 1,110 | The forceps lie across the field's edge again and the hand is 3 cm away: score 1.0. |
| 1,126 | Held for a second: success. |

Layout B runs to the same steps: 1.0 at step 1,110, success at 1,126. Steps are read from a trace kept every five steps.

## Control runs

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves. |
| `--mode hand` (the gripper takes a cup and sets it on the field) | 0.0 | The cup ends up standing on the field, but the hand passed over the pack and the field and held the cup. |
| `--mode over` (the same demo, the cups set 5 cm further into the field) | 0.0 | Both cups end up standing on the field, uncontaminated by touch, but the hand followed the forceps over the field's sterile part (first at step 407). |
| `--mode one` (the same demo with one cup) | 0.25 | One cup on the field, the forceps laid back; nothing contaminated. |

## Dream-1

One run, 1,800 steps, layout A, after a 200-step test: **0.0**. The right gripper went first to the cups and stayed
above the pack, which by the rules contaminates the pack and the cups on it. It then went to the forceps' handle at
the field's edge, did not carry the forceps away, and ended above the cups again. Both cups were still on the pack
at the end.

## Building it for real

- A Mayo instrument tray (49 x 32.5 x 1.9 cm, the standard size) under a sterile drape or towel folded to the tray's
  size, taped so that it does not slide.
- A 30.5 cm (12 in) square of sterilization wrap taped flat.
- Two 2 oz stainless graduated medicine cups (5.1 cm across, 4.45 cm tall).
- 14 cm (5.5 in) stainless thumb dressing forceps with serrated tips, laid on edge across the tray's near edge with
  5 cm of handle overhanging. A strip of tape on the drape marks where the tips go.

## What the simulation simplifies

- Everything is rigid. The drape and the wrapper are flat surfaces; there is no cloth and no fluid.
- The forceps are two rigid bowed blades on a sprung hinge at the tail. Real ones are one piece of spring steel that
  bends along its length. The closing force (2.5 N at the finger grip), the blade thickness (1.5 mm), the tips' gap at
  rest (15 mm) and the mass (25 g) are estimates, not measurements.
- The cup's diameter and height are the catalogue's. Its base diameter (3.7 cm), wall (0.8 mm), rolled rim and mass
  (35 g) are estimates. Its wall is made of 20 flat pieces.
- Friction between the serrated tips and the cup is 0.6 (an estimate).
- A gripper pad is rubber and meets a blade over a patch; the simulator's pads are rigid, so the blades are given a
  4 mm contact patch.
- Touching is judged by distance once per control step (20 times a second), not from contact forces: a touch shorter
  than a step can be missed, and passing within 1.5 mm counts as a touch.
- The forceps start lying on edge, the two blades side by side. Dropped, they may fall onto one blade's flat face,
  where a parallel gripper cannot squeeze them.

## Related tasks on the board

- "Peel open a sterile pouch with both hands and drop its forceps onto the sterile field" (pending) and "Transfer
  sterile forceps from their peel pouch onto a sterile drape" open a package and drop its contents. Here nothing is
  dropped: the forceps are the tool, and they carry something else.
- "Lay a surgical instrument set on a sterile field without touching its unsterile border" and "Set instruments on a
  sterile field and keep them off its unsterile border" place things on a field with the gripper. Here the gripper
  may touch neither the things nor the field, and may not pass over it.
- "Draw a splinter out of a fingertip with tweezers" and "Take out an interrupted suture: forceps lift the knot, a
  stitch cutter cuts flush" (both pending) also work a pair of forceps. They pull a small thing out; this carries two
  cups under sterile rules.

## Sources

- O*NET OnLine, 29-1141.00 Registered Nurses and 31-9092.00 Medical Assistants (task statements and 2025 employment).
- G. R. Doyle and J. A. McCutcheon, *Clinical Procedures for Safer Patient Care* (BCcampus), section 1.5 "Surgical
  Asepsis and the Principles of Sterile Technique". The four rules scored here, in its words: "Place large items on
  the sterile field using sterile gloves or sterile transfer forceps." "Once a sterile field is set up, the border of
  one inch at the edge of the sterile drape is considered non-sterile." "Non-sterile items should not cross over the
  sterile field. For example, a non-sterile person should not reach over a sterile field." "Keep the tips of forceps
  down during a sterile procedure to prevent fluid travelling over entire forceps and potentially contaminating the
  sterile field."
- OpenStax, *Fundamentals of Nursing* (2024), section 10.3 "Sterile Technique": the 1 in (2.5 cm) border of a sterile
  package.
- Open RN, *Nursing Skills*, section 4.4 "Sterile Fields": "Any part of the field within 1 inch from the edge" is
  treated as contaminated, and no part of the body is to be directly over the field.
- Thompson Rivers University School of Nursing, "Principles of Asepsis" (video transcript): the 2.5 cm border, sterile
  forceps tips arranging and adding items, the handles kept at the border.
- Catalogue sizes: 2 oz stainless graduated medicine cup, 2 in across by 1 3/4 in tall; Mayo instrument tray,
  19 1/4 x 12 3/4 x 3/4 in; thumb dressing forceps, 5 1/2 in.
