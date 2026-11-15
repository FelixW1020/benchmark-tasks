# Mix a bowl of dough with a wooden spoon, steadying the bowl

A stainless mixing bowl stands loose on the worktop with the ingredients of a small bread dough in it: flour, with
egg and water poured into a well in the middle. A wooden spoon with a built-up grip stands in a spoon clip on the
bowl's near rim, its head down in the flour. The robot has to mix the flour and the liquid into a dough.

**Instruction:** "Mix the flour and the liquid in the bowl into a dough with the wooden spoon."

Mixing is judged the way a baker judges it: by how much dry flour is left. Flour only takes up liquid where the spoon
works it into the wet part of the batch, so the spoon has to be driven through the whole batch, round the bowl and
across it, not just dabbed in the middle. The bowl is a light steel bowl on a laminate worktop that stirring pushes
about, so the natural way, the way people do it, is to hold the bowl by its tab with one hand while the other stirs.

## Where this comes up

- **Home cooking and baking.** Mixing a dough, batter or stuffing in a bowl with a spoon is one of the most common
  steps in a recipe. It is a core task for any home or assistive kitchen robot.
- **Assistive cooking.** People with reduced grip strength or tremor often find stirring stiff mixtures, and holding
  the bowl still at the same time, among the hardest kitchen jobs. Built-up spoon handles and bowls with grips (as in
  this scene) are standard adaptive equipment for them; a robot that can use the same equipment can help or take over.
- **Food service.** Cooks, food preparation workers and bakers mix small batches by hand every shift (stuffings,
  batters, doughs too small for a mixer).

## Potential impact

- **Two hands with different jobs.** One hand steadies; the other works a tool against a load that pushes back. The
  same pattern runs through a great deal of household work: holding a pan while scraping it, a cutting board while
  slicing, a jar while scooping.
- **Tool use with process feedback.** Progress is not a pose. The robot has to keep working until the batch is
  mixed throughout, which needs coverage of the whole bowl, not a single scripted motion.
- **Granular and mixing materials.** Most manipulation benchmarks use rigid objects. Here the state that matters is
  the condition of a few hundred particles, and the score rewards how much of the batch has been worked.

## What it measures

- **Coverage.** Flour away from the spoon's path stays dry, so the spoon has to be worked round the middle, out
  toward the wall and across the bowl, not just in one place.
- **Holding the tool firmly.** A spoon pinched loosely swings in the fingers and rides over the top of the batch.
- **Keeping the batch in the bowl and the bowl upright.** Flinging more than 5% of the batch out, or tipping the
  bowl, scores 0.

## Files

| File | What it is |
|---|---|
| `task.py` | The task file: scene, the mixing rule, scoring, the instruction |
| `doughkit.py` | Builds the bowl (with its grip tab and spoon clip), the spoon (with its grip) and the batch's beads; reads and resets the beads through one PhysX view |
| `demo.py` | The scripted demo |
| `test_scoring.py` | Offline tests of the mixing and scoring rules (no simulator) |
| `task.json` | The submission's details |

## Requirements

The yam_arena embodiment and Isaac Lab-Arena, as set up by the program's GPU instance image (an NVIDIA RTX GPU is
needed). `test_scoring.py` needs only Python with numpy, pytest and the `yam_arena` package.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mix the flour and the liquid in the bowl into a dough with the wooden spoon." \
  --scorer task_score --max-steps 2400
```

Run limit: 2,400 steps (2 minutes at 20 Hz).

## Scripted demo

```bash
python demo.py task.py out_dir
```

1. The left hand comes straight down on the bowl's grip tab and takes it between its fingers.
2. The right hand comes straight down on the spoon's grip, takes it, and lifts the spoon 3 cm up out of its clip.
3. It moves the spoon in until the middle of its head is over the middle of the bowl, down in the batch, and stirs:
   five circles each way 2.5 cm round the middle, five each way 4.5 cm round it, then strokes across the bowl.
4. It puts the spoon back in its clip and lets go, and both hands rise clear.

The script is open loop: every pose comes from the layout constants in `task.py`, and object states are read only for
the result file.

## Scoring

Tools include the trial video, a kitchen scale, a fine sieve, tared collection containers and a dry brush.
A few drops of food colouring in the egg and water help distinguish wet dough from dry flour.

**In the real world**, start with 240 g of flour. Check the video for bowl tilt over 30 degrees or more than
19 g of the batch outside the bowl at any time. Either condition gives a score of zero.

1. Collect and weigh material that was removed or spilled onto the worktop, spoon handle or robot. Record
   its mass as `O`. Deduct the full mass as a conservative allowance when a spill contains liquid or wet dough.
2. Drain freely flowing liquid into a tared container. Record its mass separately. It does not enter the flour
   calculation.
3. Transfer the bowl contents to the sieve and scrape the bowl and spoon head into it. Sieve loose dry flour
   into a tared container and weigh it as `L`.
4. Extract enclosed dry flour in three passes over the fine sieve. For the first pass, tear open every retained
   lump, including those under 1 cm, and break the pieces smaller than a pea. Tap and brush the exposed powder
   through the sieve. For the second and third passes, spread out the retained pieces, split them again to expose
   the interiors, then tap and brush again. Use a tared container for each pass and sum the three dry-flour masses
   as `E`.
5. Calculate the raw estimate `R = 240 - O - L - E` in grams. Calibrate with three untouched batches prepared and
   measured with the same setup and extraction procedure. Round the largest raw estimate up to a whole gram
   and fix that allowance before testing fresh batches. The calibration below gives 34 g. The calibrated share is
   `clamp((R - 34) / 206, 0, 1)` and the score is `min(1, calibrated share / 0.95)`.

The allowance covers the apparent mixing seen in the untouched batches, including any flour lost during recovery
or wetted on contact. Report the recovered masses and raw estimate alongside the score. When changing the setup
or extraction procedure, repeat calibration and check the fixed allowance on separate no-mix removal trials.

**In the simulator:** each bead of flour is dry until the spoon has worked it into the liquid: while the spoon's head
is moving through the batch (faster than 3 cm/s) and the bead is within 4 cm of the head's middle, in contact with liquid (or flour already mixed in) and sliding against it at more
than 2 cm/s, it adds up the distance it slides, and at 1 cm in total it is mixed in (and turns from white to tan).
Flour lying against the liquid without being worked stays dry, and so does flour that is shaken, carried round
with the bowl or stirred by anything but the moving spoon. In code: `wet_step()`, `_watch()` and `score()` in
`task.py`.

## Verification

| Run | Score | Notes |
|---|---|---|
| Scripted demo | 1.0 | 96% of the flour mixed in and 1.2% of the batch out of the bowl at step 1,218 (61 s) |
| Dream-1, 2,400 steps | 0.0 | The right gripper reached into the bowl and knocked the spoon out of its clip; the left hand did not touch the bowl and nothing was mixed |
| Arms still | 0.0 | Nothing mixed, nothing spilled |
| Short stir, `--mode brief` (control) | 0.34 | One circle each way: 31.9% of the flour mixed in, nothing spilled; a controlled partial mix |
| Half the stirring, `--mode half` (control) | 0.43 | The small circles only: 41.2% mixed in, 0.8% out of the bowl; a controlled partial mix |
| `pytest test_scoring.py` | Offline arithmetic and simulator-rule checks | The table below records the physical measurements |

### Initial physical batches

Each batch started with 240 g of flour. Batches 1 and 2 had no flour intentionally mixed into the liquid.
About 15 g of dry flour was removed from batch 2. Batches 3 and 5 each used 120 g of flour in the mix and left
120 g dry. Batch 4 used all 240 g. Batch 6 used 200 g in the mix, left 20 g loose and enclosed another 20 g
in approximately ten lumps, each under 1 cm. The lumps were made by pinching pieces of dough around dry flour.
They were torn apart over the fine sieve into pieces smaller than a pea, then tapped and brushed to recover
the dry flour. True mixed flour is the quantity allocated to mixing during preparation.

All mass columns are in grams. The raw estimate comes from the measured masses. Historical scores use the original 31 g allowance
derived from batches 1 and 2.

| Batch | Removed | Free liquid recovered | Loose dry flour recovered | Enclosed dry flour recovered | True mixed flour | Raw estimate | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 108 | 213 | 0 | 0 | 27 | 0.000 |
| 2 | 15 | 105 | 194 | 0 | 0 | 31 | 0.000 |
| 3 | 0 | 0 | 115 | 0 | 120 | 125 | 0.473 |
| 4 | 0 | 0 | 0 | 0 | 240 | 240 | 1.000 |
| 5 | 0 | 29 | 113 | 0 | 120 | 127 | 0.484 |
| 6 | 0 | 0 | 19 | 17 | 200 | 204 | 0.871 |

The two 120 g mixes read 125 g and 127 g before calibration. The batch with 29 g of free liquid read 2 g higher.
In batch 6 the original extraction recovered 17 of 20 g of enclosed flour and 19 of 20 g of loose flour.

In the first independent removal checks, removing 5.1 and 15.0 g without mixing gave raw estimates of 28.3 and
30.2 g and scores of zero. Removing 18.1 g gave a 33.1 g raw estimate and a score of 0.0106 under the original
31 g allowance. That failure prompted the recalibration below.

### Three-pass extraction

Each trial started with 20.0 g of dry flour enclosed in sub-centimeter dough lumps and a separate loose dry-flour
fraction. The extraction pass columns record the dry flour recovered from the retained lumps. All masses are in grams.

| Trial | Lumps | Largest diameter (mm) | Loose recovered | Pass 1 | Pass 2 | Pass 3 | Enclosed recovered | Enclosed unrecovered | Enclosed recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 205 | 9.7 | 19.4 | 16.8 | 1.8 | 0.8 | 19.4 | 0.6 | 97.0% |
| 2 | 211 | 9.8 | 19.5 | 17.1 | 1.5 | 0.9 | 19.5 | 0.5 | 97.5% |
| 3 | 198 | 9.5 | 19.6 | 16.5 | 2.0 | 1.1 | 19.6 | 0.4 | 98.0% |
| 4 | 207 | 9.9 | 19.3 | 17.2 | 1.7 | 0.7 | 19.6 | 0.4 | 98.0% |

The third pass recovered 0.7–1.1 g across all four trials. The unrecovered enclosed mass is the difference
between the 20.0 g put into the lumps and the three-pass total.

### Updated calibration and independent removal checks

The calibration used three untouched batches with 240 g of flour and no material removed.

| Calibration batch | Dry flour recovered (g) | Raw estimate (g) |
|---|---:|---:|
| 1 | 207.6 | 32.4 |
| 2 | 206.9 | 33.1 |
| 3 | 206.4 | 33.6 |

The maximum was 33.6 g, rounded up to a fixed allowance of 34 g. The following separate no-mix removal trials
all stayed below the 19 g spill limit. All three scored zero with the fixed allowance.

| Trial | Target removal (g) | Actual removal (g) | Dry flour recovered (g) | Raw estimate (g) | Credited flour (g) | Score |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 | 5.0 | 202.0 | 33.0 | 0.0 | 0.0000 |
| 2 | 15.0 | 15.1 | 192.2 | 32.7 | 0.0 | 0.0000 |
| 3 | 18.5 | 18.7 | 188.0 | 33.3 | 0.0 | 0.0000 |

The 34 g allowance sets the zero point for scoring. It can reduce credited progress on partial mixes whose
recovery shortfall is smaller than in the calibration batches.

## Runtime and seeds

About 4 minutes per demo run and 8 minutes per Dream-1 run on one L40S, including about a minute of start-up.
`object_jitter=0`; the layout is fixed, and there is no randomness in the task.

## Fidelity

- The batch is 200 beads of 12 mm: 160 of flour (1.5 g each, 240 g) and 40 of egg and water (3.5 g each, 140 g),
  rigid spheres with friction. Dough's stickiness and its growing stiffness as it comes together are not modelled;
  "mixed in" is decided by the rule above, and the bead changes colour.
- The bowl, its grip tab and spoon clip, and the spoon are rigid. The bowl's wall is 32 flat segments.

## Real-world build

A 2.5 qt (2.4 L) stainless mixing bowl, 21 cm across the rim and 8.5 cm deep, with a flat grip tab riveted to its
left side (4 x 2.5 cm, 5 mm thick) and a clip-on spoon rest on its near rim (two posts 14 mm apart, 2 cm tall). A
26 cm beech spoon with a built-up silicone grip, 50 mm long, 24 x 30 mm, its middle 19.9 cm from the spoon's tip. 240 g of
bread flour with 100 g of water and one egg (about 40 g) poured into a well in the middle. Put the bowl on a laminate
worktop and stand the spoon in its clip, its head in the flour.

## Third-party assets

None: every object is built from primitive shapes in `doughkit.py` and `task.py`.
