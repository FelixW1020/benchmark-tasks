# Test a pan of sanitizer with a test strip, and add a tablet only if it reads weak

A steel food pan stands on a kitchen worktop holding one gallon of water for sanitizing. It may or may not already
have its sanitizer tablet dissolved in it. A fresh test strip leans in a rest beside the pan, two sanitizer tablets
lie on a board, a colour chart lies in front, and a waste cup stands at the side. The robot has to test the pan with
the strip, bring it to 200 ppm if it is weak, leave it alone if it is not, and bin the used strip.

**Instruction:** "Test the pan of sanitizer with the test strip: dip the strip's pad in the solution for one to two
seconds and compare its colour with the chart. If it reads below 200 ppm, put one sanitizer tablet in the pan; if it
reads 200 ppm, add none. Then drop the used strip in the waste cup."

## What this task measures that others do not

Many "read it, then act" tasks can be answered from a picture: the gauge, the label or the level is already in view,
and the robot's part is ordinary placement. This one is built so that it cannot be.

- **The answer is not in the scene.** Plain water and made-up sanitizer look the same. The two layouts' first camera
  frames are identical pixel for pixel (checked on the recorded runs). Nothing the robot can see tells it which pan
  it has until it has made the measurement itself.
- **The measurement is a manipulation, and it can be done wrong.** The strip is 6.4 mm wide and 0.35 mm thick and has
  to be taken by its edges out of a slanted rest. Its 6.4 mm pad has to go fully under the surface for a second
  while the strip is held by its handle end, at least 2 cm above the liquid, with no part of the gripper in the
  solution. A strip dropped into the pan, a half-wetted pad or a wet gripper is not a test, and the scorer says so.
- **The reading has to be brought to a camera.** The pad is under two pixels across in the overhead camera. It can
  only be read up close, which here means holding the wet strip where a wrist camera sees it.
- **The right action differs from trial to trial, and one of the two right actions is to add nothing.** A pan that
  reads 0 gets one tablet. A pan that reads 200 gets none: the tablets are labelled one to the gallon, and a second
  one makes the pan twice its labelled strength, which scores 0. Dosing before testing also scores 0.
- **No fixed routine passes both layouts.** The control runs below show it: "always add a tablet" scores 1.0 on one
  layout and 0.0 on the other; "never add" scores 0.25 and 1.0. Only acting on the strip scores 1.0 on both.

The scripted demo makes its decision the way a policy would have to: from the right wrist camera's picture of the pad
before and after the dip. It does not read the pan's strength or the pad's state from the simulator.

No sanitizer-strength, test-strip or measure-then-dose task was on the task board when this was written (searched 15
November 2026).

## Where this comes up

- **Commercial kitchens, every shift.** Food codes require a test kit for the sanitizer in use and require its
  concentration to be measured with it (FDA Food Code 4-302.14 and 4-501.116, as adopted by the states). Suppliers'
  guidance is to test at the start of each shift and for each new batch. Dishwashers and food preparation workers do
  it as routine work: the US Bureau of Labor Statistics counted 463,940 dishwashers in May 2023, and in restaurants
  alone 382,640 dishwashers and 439,240 food preparation workers in May 2019.
- **Why both directions matter.** Too weak and the pan does not sanitize. Too strong and it is outside the
  product's labelled use for food-contact surfaces. The check exists because the two cannot be told apart by eye.
- **The same pattern elsewhere.** Dip strips are used the same way for fryer oil, pool and spa water, dialysis
  water and urinalysis: wet a small pad correctly, read it at once, act on the reading.

## Files

| File | What it is |
|---|---|
| `task.py` | The task file: scene, what counts as a test, the pad's colour, scoring, the instruction |
| `sanikit.py` | Builds the pan, the strip, the cup, the chart and the fixed parts |
| `demo.py` | The scripted demo, and its control runs |
| `test_scoring.py` | Offline tests of the scoring rule and the scene's geometry (no simulator) |
| `task.json` | The submission's details |

## Requirements

The yam_arena embodiment and Isaac Lab-Arena, as set up by the program's GPU instance image (an NVIDIA RTX GPU is
needed). `test_scoring.py` needs only Python with numpy, pytest and the `yam_arena` package.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Test the pan of sanitizer with the test strip: dip the strip's pad in the solution for one to two seconds and compare its colour with the chart. If it reads below 200 ppm, put one sanitizer tablet in the pan; if it reads 200 ppm, add none. Then drop the used strip in the waste cup." \
  --scorer task_score --max-steps 1800
```

Run limit: 1,800 steps (90 seconds at 20 Hz). There are two layouts, and a policy should be run on both: `SANI_LOT=A`
(the default) is plain water, `SANI_LOT=B` is a pan already at 200 ppm. Everything else is the same.

## Scripted demo

```bash
python demo.py task.py out_dir                    # layout A; SANI_LOT=B for layout B
python demo.py task.py out_dir --mode testonly    # control: test and bin the strip, never add a tablet
python demo.py task.py out_dir --mode always      # control: test, then add one tablet whatever the strip showed
python demo.py task.py out_dir --mode blind       # control: add a tablet without testing
python demo.py task.py out_dir --mode double      # control: test, then add both tablets
```

1. The right hand takes the strip out of its rest by its handle end, fingers on its two edges, and carries it over
   the pan's rim. Above the liquid it stops and looks at the dry pad through its own wrist camera.
2. It lowers the strip until the pad is 1.2 cm under the surface, holds it there for a second and a half, and lifts
   it back to the same place. It looks again. The pad's colour now, against its colour before, is the reading.
3. If the pad has not turned green, the left hand takes one tablet off the board and drops it in the pan. If it has,
   the left hand does nothing.
4. The right hand drops the used strip in the waste cup.

Every target comes from the layout constants in `task.py` and the sizes in `sanikit.py`. The feedback the demo uses
is what a robot has: each hand's own measured position, to correct an arm's sag before a placement, and the right
wrist camera's picture, to read the pad. It never reads the strip's, the tablets' or the solution's state.

## Scoring

Tools: the trial video, a stopwatch and a ruler; for the pan's strength, the count of tablets put in it (or a fresh
test strip).

- **0** if the pan was never tested, if a tablet went into the liquid before it was tested, or if the pan ends above
  200 ppm (a tablet in a pan that was already made up, or two tablets).
- Otherwise:
  - **0.25** tested: the strip's pad was fully under the liquid for 1 second, with the strip held by its handle end
    at least 2 cm above the surface and no part of a gripper in the liquid;
  - **0.5** and the pan ends at 200 ppm (its starting strength plus 200 ppm for each tablet in the liquid);
  - **0.75** and the used strip is inside the upright waste cup;
  - **1.0** and the strip, the cup and the tablets are at rest (under 2 cm/s), with both grippers at least 3 cm from
    the strip and out of the pan.

In code: `_watch()` and `score_from()` in `task.py`. A trial ends as soon as it reaches 1.0.

## Verification

Each row is one simulator run of the files in this folder.

| Run | Layout A (plain water) | Layout B (already 200 ppm) |
|---|---|---|
| Scripted demo: act on the strip | 1.0 (604 steps; read 0, one tablet added) | 1.0 (480 steps; pad seen to turn green, none added) |
| Control: test, never add a tablet | 0.25 | 1.0 |
| Control: test, always add one tablet | 1.0 | 0.0 (400 ppm) |
| Control: add a tablet without testing | 0.0 | not run |
| Control: test, then add both tablets | 0.0 (400 ppm) | not run |
| Arms still | 0.0 | not run |
| Dream-1, 1,800 steps | 0.0 (both arms reached out, the right toward the strip's rest and the left over the tablet board; nothing was picked up and the strip was never dipped) | not run |

- The two layouts' first overhead frames are the same file, byte for byte.
- In the layout B demo the wrist camera's picture of the pad changed from pale yellow to green over 112 pixels; in
  layout A no pixel met the test.
- `pytest test_scoring.py`: 22 passed (the score's tiers, a blind dose, an overdose, what counts as a wet pad and as
  a dip, the one-gallon depth, the cup, the scene and its starting clearances).

## Runtime and seeds

About 2 minutes per demo run and 6 minutes per Dream-1 run on one L40S, including about a minute of start-up.
`object_jitter=0`; the layout is fixed per lot (A or B), and there is no randomness in the task.

## Fidelity

- **The solution is not simulated as a fluid.** It is a level surface in a rigid pan: the task file works out from
  positions what is under it. Nothing floats, drips or is slowed, and the strip does not carry liquid away.
- **The pad** takes the chart colour for the pan's strength once all of it has been under for 1 second, as the
  strips' directions give (dip 1 to 2 seconds, compare within 10 seconds). The colour does not drift afterwards.
- **A tablet counts as dissolved the moment it is in the liquid** (200 ppm in one gallon). Real tablets take
  minutes, so a real trial is scored on the tablets put in the pan.
- **A dissolved tablet does not tint the water.** Some real sanitizer tablets carry a dye that shows the sanitizer
  is present, though not how strong it is; these are modelled without one, so the two pans look the same.
- **The strip** is a rigid plate (a real one bends a little), weighted 0.5 g so that the simulator holds it
  steadily; the rest it leans in is fixed to the worktop. The pan does not move.
- The renderer shows the inside of the pan dark; the liquid's surface shows as a highlight.

## Real-world build

A half-size, 4 in deep steel food pan (inside about 24 x 30 cm) with one US gallon of water, 5.3 cm deep. Plastic
quaternary-ammonium sanitizer test strips (6.4 x 64 mm, reagent pad at one end, chart at 0, 100, 200 and 400 ppm) and
sanitizer tablets labelled one tablet per gallon for 200 ppm. For the made-up layout dissolve one tablet in the pan
well before the trial. A strip rest that holds one strip at 60 degrees, pad end down, a 10 cm wide waste cup, and the
strips' colour chart, at the positions in `task.json`. Check the pan's starting strength with a strip of your own
before each trial, and use a fresh strip for each one.

## Third-party assets

None: every object is built from primitive shapes in `sanikit.py` and `task.py`. No brand, logo or product name is
modelled.
