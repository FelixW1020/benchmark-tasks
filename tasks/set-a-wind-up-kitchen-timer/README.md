# Set a wind-up kitchen timer for 10 minutes

A clockwork kitchen timer stands on the counter reading zero: a steel cylinder 6 cm across and 6 cm tall, in two parts
that turn on each other. The instruction is only "Set the kitchen timer for 10 minutes." The robot has to leave it
standing on the counter, running, reading 10, and wound so that it will ring.

The table stands in for the counter. Nothing else is on it.

## What the task asks of a robot

- **A rule the instruction does not give.** A clockwork timer has to be turned all the way to a mark near the end
  of its scale before it is turned back to the time wanted. The maker's instructions for this timer: "For the timer
  to ring, you must always turn it clockwise to 55 minutes [...] Then, turn the timer anticlockwise to the desired
  time", and "If the timer does not ring properly, the spring inside the clockwork is not tight enough." Turned
  straight to 10 it counts down and does not ring. That is the `direct` control run, and it scores 0.25.
- **Two hands on one small object.** The two parts are each about 3 cm tall and turn against a clutch and a
  mainspring (0.04 to 0.055 N m here). The timer weighs about 110 g, so the table's friction holds it against less
  than half of that. Turned by one hand where it stands, the whole timer goes round with the hand: after 345 degrees of
  turning it read 15 minutes, not 57.5. That is the `onehand` control run, and it scores 0. One part has to be held
  while the other is turned.
- **Almost a full turn, and most of the way back.** 330 degrees to the 55-minute mark, then 270 degrees back to 10.
  A wrist of this robot rolls 240 degrees. The demo shares each turn between the two wrists, rolling them in
  opposite senses, so that neither has to let go: 345 degrees out, 283 back.
- **A place to do it.** While the timer stands on the table, a second gripper can take the 2.6 cm lower part only
  by leaning in low over the table, and its fingers then stand in the way of the hand that turns the upper part.
  The demo lifts the timer by its upper part and lays it on its side in the air, so that the two hands can face
  each other along its axis with one part each.
- **Reading a scale.** Full marks need the timer left reading 9 to 11 minutes: 6 degrees, or 3 mm at the rim, either
  side of the 10-minute mark.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `timerkit.py` | The timer: its two parts, the hinge with its stops, the going train with its clutch, the mainspring, the printed scale. |
| `demo.py` | A scripted demo that reaches full score, and three control runs. |
| `test_scoring.py` | 67 checks of the sizes, the dial's reading, the marks, the layouts, the demo's plan and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,800 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1800 \
  --instruction "Set the kitchen timer for 10 minutes."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
TIMER_LOT=B python demo.py task.py out_dir     # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the timer's two parts and the idler wheel inside it are placed to match each other
  exactly.
- There are no seeds. `TIMER_LOT` picks the layout: A (the timer at x 30 cm, y -6 cm, its zero mark facing the robot)
  or B (at x 26, y 5, its zero mark facing the robot's left and turned 30 degrees toward the robot).
- A trial is at most 1,800 steps (90 s). It ends early once full score has held for 1 s. The demo takes 816 steps
  (41 s); a simulated trial of that length took 6 to 9 minutes of wall time on a GPU shared with other jobs, much
  of it start-up.
- `result.json` from the demo holds the score, the task's own report at the start and end, and a trace every 5 steps.

## Scoring

Scored at the end of the trial.

- **Reading:** the scale mark under the index, in minutes. It is the angle the lower part has been turned clockwise
  (seen from above) against the upper part, 6 degrees to the minute.
- **Wound:** the reading has reached 54 minutes or more (the 55-minute mark, less a minute's tolerance) at some time
  in the trial, and has not been back to 0.5 or less since (at zero the bell rings and the winding is spent).
- **Left standing:** the timer stands on the table with its axis within 10 degrees of upright and its underside
  within 5 mm of the table; both grippers are at least 3 cm from it; it is at rest (under 2 cm/s and 0.5 rad/s).

| Score | What it takes |
|---|---|
| 0 | None of the rows below, or the timer has left the table. An untouched timer; a timer turned to some time without being wound and not left reading 9 to 11. |
| 0.25 | Wound, but not left standing (still held, lying on its side, moving). Or: never wound, but left standing reading 9 to 11 minutes. It will count down and not ring. |
| 0.5 | Wound and left standing, at a reading more than 3 minutes from 10 (left at 55, say). |
| 0.75 | Wound and left standing, reading 7 to 13 minutes but not 9 to 11. |
| 1.0 | Wound and left standing, reading 9 to 11 minutes. |

The timer runs once it is off zero, so its reading falls slowly while it stands (a real timer's by one minute a
minute; the simulated one's by about a quarter of that). A trial ends when full score has held for a second, so a
timer set to 10 is scored within a few seconds of being let go.

## Scoring a real trial

- **Reading:** from a photograph of the timer at the end of the trial. On a timer with a window, the number in the
  window; on one with an outside scale, the mark under the index. One minute is 6 degrees.
- **Wound:** from the video, whether the reading reached 55 during the trial. Or let the timer run out after the
  trial: a wound timer rings loudly when it reaches zero, an unwound one clicks or stays silent.
- **Left standing:** from the last second of the video.

The task has not been run on a physical rig.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `timerkit.py`, and the hands' own measured
positions (to correct for arm sag). It never reads where a part is or what the timer reads: how far the timer has
been turned is known only from how far the two wrists have been told to roll.

1. The right hand comes down over the timer with its fingers 8.5 cm apart, closes on the upper part (tool point
   1.3 cm below the top face), lifts it 7 cm, and swings it onto its side 14 cm above the table, its underside toward
   the left hand.
2. The left hand comes in along the timer's axis and closes on the lower part (tool point 1.0 cm in from the
   underside). The two hands' fingertips are then 11 mm apart, end to end.
3. Both wrists roll 172.5 degrees, in opposite senses: 345 degrees between the parts, 57.5 minutes. Then both roll
   141.5 degrees back: 10.3 minutes.
4. The left hand lets go and draws back. The right hand swings the timer upright again, stands it where it stood,
   lets go, and lifts away.

| | Layout A | Layout B |
|---|---|---|
| Right fingers closed on the upper part | step 110 | step 110 |
| Timer off the table | step 120 | step 120 |
| Timer on its side, 14 cm up | step 225 | step 225 |
| Left fingers closed on the lower part | step 365 | step 365 |
| Reading passes 54 minutes: wound (score 0.25) | step 475 | step 475 |
| Largest reading | 57.4 min, step 495 | 57.5 min, step 495 |
| Back at 10 minutes | step 590 | step 590 |
| Left hand lets go | step 610 | step 610 |
| Timer standing on the table again | step 755 | step 755 |
| Let go and clear (score 1.0) | step 800 | step 800 |
| Trial ends with full score | step 816 | step 816 |
| Reading at the end | 10.25 min | 9.76 min |
| Timer's place at the end, from where it stood | 1.0 mm | 0.4 mm |

(Steps are read from a trace written every 5 steps.)

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves; the timer reads zero. |
| `--mode onehand`: the timer left standing, its upper part turned by the right hand alone, two bites of 172.5 degrees and two bites back | 0.0 | The whole timer goes round on the table with the hand. After 345 degrees of turning it reads 15.1 minutes, not 57.5: never wound. After the two bites back it reads 1.4. |
| `--mode direct`: the full demo's two-handed hold, turned straight to 10 minutes, then left for 57 s | 0.25 | The timer is left standing reading 10.0 (9.8 a minute later), but its reading never passed 10.2: not wound, so it will not ring. |

## Dream-1

One run, 1,800 steps, layout A (after a 200-step test): **0.0**. In the first 8 seconds both arms drew back toward
their bases and closed their grippers, and they stayed there for most of the trial. In its last 4 seconds the right
arm reached out; the trial ended with its closed gripper over the timer. The timer stood where it had been put.

## Building it for real

- A two-part wind-up kitchen timer of the twist kind, 6 cm across and 6 cm tall, with a 60-minute scale. The model
  is the size of the IKEA ORDNING timer and follows its maker's setting rule. On that timer the minutes show in a
  window in the upper part; a timer with a scale printed on the outside can be used as it is, and the scoring is
  the same for both.
- A flat counter or table top.
- Tape marks for the two places, and for which way the zero mark faces.

Resetting: turn the lower part back to zero (or let the timer run out) and stand it on its mark.

## What the simulation simplifies

- **Sourced:** the timer's diameter and height (6 cm each); the setting rule (to 55 minutes, then back); that it is
  the lower part that the maker's drawing shows being turned, clockwise seen from above; the 60-minute scale.
- **Estimates, not measurements:** the height of the lower part (2.6 cm, taken from the maker's drawing); the
  timer's weight (110 g, of which the upper part 45 g); the far stop (59.5 minutes); the clutch's slipping torque
  (0.035 N m); the mainspring's pull (0.004 N m at zero rising to 0.020 N m at a full turn); friction between the
  timer and the table (0.5 static, 0.4 sliding). No maker publishes the torques, and none was measured.
- Both parts are plain rigid cylinders with square edges. The real timer's edges are rounded.
- The scale is printed round the outside of the lower part with an index mark on the upper part. The ORDNING timer
  shows the minutes through a window in its upper part instead.
- The clockwork is three things: a hinge between the two parts; the mainspring's pull toward zero, set from the
  reading 20 times a second; and a going train on the hinge, a drive set to run the timer back at 6 degrees a
  minute that slips above the clutch's torque. There are no gears, escapement or bell. "Wound" stands for the
  bell: it is the reading having reached 54 minutes.
- The simulated timer runs down more slowly than its drive is set to. Standing on the table after the `direct`
  run it fell from 9.98 to 9.76 minutes in 57 s, about a quarter of a real timer's rate. Over a trial that is a
  few tenths of a minute at most.
- The stops at zero and at 59.5 minutes are carried by two extra hinges in series through a 4 g idler wheel inside
  the timer, because the simulator's hinge stops cannot be more than half a turn apart. The idler has no shape and
  touches nothing.
- Turning the timer backwards from zero meets a hard stop. On the real timer the maker warns that this damages the
  clockwork; no damage is modelled.

## Sources

- IKEA, ORDNING timer, instruction sheet AA-2096903-1 (2018),
  https://www.ikea.com/us/en/manuals/ordning-timer__AA-2096903-1_pub.pdf: "For the timer to ring, you must always
  turn it clockwise to 55 minutes, see Figure 1. (This pulls the spring inside the clockwork to the correct
  countdown). Then, turn the timer anticlockwise to the desired time (see Figure 2). Now the timer will ring loudly
  and clearly." "Never turn the timer anticlockwise directly from the zero position. This damages the clockwork."
  "If the timer does not ring properly, the spring inside the clockwork is not tight enough. Always turn the timer
  clockwise to 55 minutes to ensure the timer rings properly." Its two figures show the lower part being turned,
  and the split between the parts.
- Remodelista, product listing "Ikea 300.667.25 Ordning Timer",
  https://www.remodelista.com/products/ikea-30066725-ordning-timer/: diameter 6 cm, height 6 cm; stainless steel,
  synthetic rubber, ABS plastic and reinforced polyamide.
- CDN (Component Design Northwest), mechanical timer MTM3, instruction sheet,
  https://cdn.restaurantsupply.com/pdf/E/n/g/English_MTM3en_0.pdf: "Turn knob clockwise to 50 and then back to the
  desired time." "The further the knob is turned clockwise, the longer the alarm will sound when it reaches 0."
- AvaTime mechanical 60-minute kitchen timer 914MT60M, specification sheet,
  https://argo.webstaurantstore.com/documents/specsheets/avatime_914mt60m.pdf: "It must be wound up to the 50
  minute mark first, then back to the desired set time for the alarm to properly sound."

## Related tasks on the board

Nothing on the board sets a mechanical timer.

- "Refit a gas burner after cleaning: turn the head until its slot drops round the igniter, then seat the cap"
  (accepted) turns a loose part until it seats. Here the turn is against a clutch and a spring, the thing it turns
  on weighs 110 g and has to be held, and where to stop is a reading on a scale.
- "Fold a lever nail clipper shut for storage" (accepted) also works a small household mechanism. Its instruction
  gives the end state; there is nothing to wind and no reading to hit.
- "Reset a jack-in-the-box and latch its lid" (pending) loads a spring by pushing a figure down against it.
- "Timed wafer immersion with a manual countdown" uses a timer to wait by; the timer is not set by hand.
