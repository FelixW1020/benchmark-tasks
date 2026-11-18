# Put a clip-on shade on a small candlestick lamp

A small brass candlestick accent lamp stands on the table with its candelabra bulb in and its shade off. The shade,
a six-sided clip-on shade, stands on the table beside it. The instruction is "Put the shade on the lamp."

A clip-on shade has no screw, ring or harp. Inside its top a wire fitter carries two sprung wire loops that slide
down over the bulb and grip it, and between them a small ring that comes to rest on the bulb's tip. Putting it on
means lifting the shade over the bulb, lowering it straight down the bulb's axis, and pressing lightly until the
ring is home.

## What the task asks of a robot

- **A thin-walled part.** The shade is a 10.8 cm tall cone of flat panels with an open
  top 7.6 cm across, crossed by the fitter's three spokes. The demo takes it from outside, a fingertip on each of
  two opposite panels just under the top edge.
- **Height.** With the shade on, the lamp is 11 inches (27.9 cm) tall. To get over the bulb the shade's bottom has
  to pass above the bulb's tip, 27.3 cm up, which puts the hand that holds its top about 40 cm above the table. A
  hand pointing straight down cannot reach that high over the lamp; the demo's hand leans 66 degrees back toward
  the robot. Carried across 4 cm lower, the shade's bottom edge catches the bulb and the shade is knocked out of
  the hand (the `low` control run).
- **A fitter that cannot be seen from the side.** The ring and loops are 1.3 cm down inside the shade's top. The
  ring's opening is 11 mm; the bulb's tip has to enter it. Brought down 1.5 cm to one side, the bulb passes beside
  the ring and the shade ends hanging round the lamp (the `aside` control run).
- **A narrow lamp.** The lamp stands on an 8.3 cm base with its bulb's tip 27 cm up and weighs about 0.8 kg (an
  estimate): by that weight and base, a sideways push of about 1.5 N at the bulb would start it tipping. The demo's
  left hand holds its stem while the shade goes on. With the shade brought down exactly on the axis the right hand
  alone also does it (the `alone` control run).
- **A press, not a drop.** The loops are sprung apart by the bulb as the shade goes down (8 to 10 degrees each in
  the demo). The shade's own weight does not carry it home: let go 3 cm above its seat, it stays hanging on the
  loops 1.6 cm short (the `drop` control run).
- **A level finish.** The shade has to end with its axis within 5 degrees of the lamp's, the lamp upright where it
  stood, and both hands away.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `shadekit.py` | The lamp with its bulb, the shade with its fitter, the two sprung loops and their hinges. |
| `demo.py` | A scripted demo that reaches full score, and five control runs. |
| `test_scoring.py` | 30 checks of the shapes, the layouts, the demo's plan and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,200 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1200 \
  --instruction "Put the shade on the lamp."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
SHADE_LOT=B python demo.py task.py out_dir     # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the two loops are placed to match the shade exactly.
- There are no seeds. `SHADE_LOT` picks the layout: A (lamp at x 30, y -13 cm; shade at x 30, y -33 cm) or B (lamp
  at x 28, y -10.5 cm; shade at x 32.5, y -32 cm, turned 10 degrees clockwise seen from above).
- A trial is at most 1,200 steps (60 s). It ends early once full score has held for 1 s. The demo takes 440 steps
  (22 s); a simulated trial of that length took 5 to 14 minutes of wall time on a GPU shared with other jobs, part
  of it start-up.
- `result.json` from the demo holds the score, the task's own report at the start and end, and a trace every 5
  steps.

## Scoring

Scored at the end of the trial.

| Score | What it takes |
|---|---|
| 0 | The lamp leaned 20 degrees or more at any time in the trial (it went over). Or: none of the rows below. A shade put on upside down scores 0. |
| 0.25 | The bulb's tip is inside the shade, the shade the right way up, but none of the rows below: the shade is still held, or rests crooked on the lamp. |
| 0.5 | The shade hangs on the bulb without being home (the fitter's ring is within 1 cm of the lamp's axis and no more than 4.5 cm above its place, the shade within 30 degrees of the lamp's axis), or it is home but 12 to 30 degrees off level. Both grippers are at least 3 cm from the shade and from the lamp, and both are at rest (under 2 cm/s and 0.5 rad/s). |
| 0.75 | The shade is seated (the ring within 4 mm of its place on the bulb's tip and within 4 mm of the lamp's axis), let go and at rest as above, but it is 5 to 12 degrees off level, or the lamp has been moved more than 3 cm or left leaning more than 3 degrees. |
| 1.0 | Seated, the shade's axis within 5 degrees of the lamp's, the lamp upright within 3 degrees and within 3 cm of where it started, let go and at rest as above. |

The lamp's lean is read at every control step (20 times a second); everything else is read at the end.

Other ways through, and what they score: dropping the shade over the bulb from above, 0.5 (run below); bringing it
down beside the axis, 0 (run below); laying the lamp down to put the shade on, 0, because the lamp has gone over;
turning the shade over and standing it on the bulb by its ring, 0, because it is upside down.

## Scoring a real trial

- **Lamp went over:** from the video.
- **Seated:** the shade's top is 27.9 cm above the table when the ring is home; with a ruler, no more than 28.3 cm.
  From above, the bulb's tip stands through the middle of the ring.
- **Level:** measure the highest and the lowest point of the shade's top edge above the table. 5 degrees is a
  difference of 7 mm across the 7.6 cm top; 12 degrees is 16 mm.
- **Lamp in its place and upright:** a tape mark round the base at the start; a small level or a plumb line
  against the stem.
- **On the bulb but not home:** the shade's top between 28.3 and 32.4 cm above the table, the bulb inside it.
- **Let go and at rest:** from the last second of the video.

The task has not been run on a physical rig.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `shadekit.py`, and each hand's own measured
position: to correct for the arms' sag, and, once the left hand has closed on the stem, to tell where the stem is
across its pads. It never reads where a part is.

1. The left hand comes in low from the left, leaning 75 degrees from straight down, and closes on the lamp's stem
   6.5 cm above the table, under where the shade will hang.
2. The right hand comes down over the shade's top with its pads 9.2 cm apart, leaning 66 degrees back toward the
   robot, and closes on the outside of two opposite panels 1 cm under the top edge. At that lean the lower edge of
   each fingertip is level and lies along its panel.
3. It lifts the shade until its bottom is 2 cm above the bulb's tip (tool point 40 cm above the table), carries it
   over the lamp, lowers it down the lamp's axis, and presses 3 mm past the seated height.
4. It lets go, lifts away and moves aside. The left hand lets go of the stem and draws back.

| | Layout A | Layout B |
|---|---|---|
| Left hand closed on the stem | step 100 | step 100 |
| Right hand closed on the shade | step 105 | step 105 |
| Shade off the table | step 125 | step 125 |
| Shade at its highest, tool point 40.3 cm above the table | step 245 | step 245 |
| Bulb's tip inside the shade (score 0.25) | step 265 | step 265 |
| Loops begin to open | step 295 | step 295 |
| Ring home | step 320 | step 320 |
| Right hand lets go | step 355 | step 355 |
| Left hand lets go | step 400 | step 400 |
| Both hands clear (score 1.0) | step 425 | step 425 |
| Trial ends with full score | step 440 | step 440 |
| Ring above its place at the end | 0.2 mm | 0.1 mm |
| Ring off the lamp's axis | 0.0 mm | 0.3 mm |
| Shade off level | 0.7 degrees | 1.2 degrees |
| Loops sprung open | 7.6 and 8.5 degrees | 8.3 and 9.7 degrees |
| Lamp moved; leaned at most | 0.0 mm; 0.1 degrees | 0.0 mm; 0.1 degrees |

(Steps are read from a trace written every 5 steps.)

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves. |
| `--mode drop`: the same carry, the shade let go 3 cm above its seat | 0.5 | The loops catch the bulb and open 3.6 and 3.9 degrees; the shade stays hanging 16.4 mm above its seat, 1.4 degrees off level. |
| `--mode alone`: the full motion without the left hand | 1.0 | Brought down exactly on the lamp's axis, the shade goes home and the lamp stays where it was. Full score at step 375. |
| `--mode aside`: without the left hand, the shade brought down 1.5 cm to one side of the lamp's axis | 0.0 | The bulb's tip misses the ring and passes beside it through the shade's top; let go, the shade ends hanging round the lamp with its fitter 8 cm below its place. The lamp did not move. |
| `--mode low`: without the left hand, the shade carried across with its bottom 2 cm below the bulb's tip | 0.0 | The shade's bottom edge meets the bulb at step 210. The lamp leans 0.2 degrees and stays up; the shade is knocked out of the fingers and drops to the table. |

## Dream-1

One run, 1,200 steps, layout A (after a 200-step test): **0.0**. In the first 6 seconds the right gripper came
down over the shade's top with its fingers open. By 10 seconds it had closed them beside the shade's top edge
without taking hold of it, and it stayed there for the rest of the trial. The left arm drew back toward its base.
The shade was still standing on the table at the end, and the lamp was not touched.

## Building it for real

- A small candlestick accent lamp of the kind sold with a clip-on shade: 11 to 12 inches tall with the shade on,
  round base 3 1/4 inches wide, candelabra (E12) socket. Unplug it and lead the cord off the back of the table.
- A B10 torpedo candelabra bulb, LED so that it stays cold, screwed in.
- The lamp's clip-on shade, or any six-sided candelabra clip-on shade about 3 x 5 x 4 1/4 inches (the listed
  lamp's own shade is 4 1/2 inches high). A round shade of the same size does as well for a robot that holds it the
  way the demo does, by two opposite points of its top.
- Seat the shade once by hand and measure the height of its top above the table. Use that height in place of
  27.9 cm in the scoring if the lamp or shade differs from the sizes here.
- Tape marks for the lamp's base and the shade's bottom edge in the two layouts.

## What the simulation simplifies

- Everything is rigid. The shade's panels are flat plates 2 mm thick, each built from six bars; a real shade is
  fabric on a styrene backing with wire rings top and bottom, and gives a little under the fingers. The shade of
  the lamp these sizes come from is bell-shaped (its panels curve); here they are straight.
- **Sourced sizes:** the lamp's height with its shade on (11 in: the listing gives 11.00 inches under its technical
  specifications and 12 inches in its description), its base (3 1/4 in wide), the shade's top and bottom (3 in and
  5 in across), the bulb's type and overall size (B10, E12 base, 1.26 in across and 3.72 in high with its base).
- **The shade's height** here is 4 1/4 in (10.8 cm). The listing gives 4 1/2 in, so the shade here is 6 mm shorter
  than the listed one.
- **Estimates, not measurements:** the lamp's weight (0.8 kg) and the height of its centre of mass (5 cm); the
  thickness of its base, foot, stem, drip pan and candle cup; the bulb's outline between its tip and its widest
  point; the shade's weight (50 g) and wall thickness; how far the fitter sits below the shade's top (1.3 cm), the
  ring's opening (11 mm), the loops' size (16 x 32 mm of 1.6 mm wire), their spring (0.2 N m per radian with
  0.012 N m of preload, which makes each loop press on the glass with about 1.3 N when the shade is home) and the
  friction of wire on glass (0.3).
- Each loop is a rigid rectangle of wire on a sprung hinge at the ring. A real fitter is bent from spring wire and
  flexes along its length.
- Each loop weighs 2 g (heavier than the wire, so that the contact solver holds it); the shade is lighter by the
  same amount.
- The bulb is one convex shape with a 12-sided section. It is part of the lamp: it cannot come unscrewed or break.
- No cord, no switch, no light.

## Sources

- Destination Lighting, "What Is a Clip-On Lamp Shade?"
  (destinationlighting.com/fliptheswitch/what-is-a-clip-on-lamp-shade): "Modern clip-on lamp shades feature wire
  assemblies that slide down over the bulb." "These assemblies attach directly to the shade and clip to the bulb
  for a secure fit." "Modern clip-on lamp shades typically fit night lights, candlestick chandeliers and some
  accent lamps".
- Lux Lampshades, "The Complete Guide to Lampshade Fitters"
  (luxlampshades.com/blogs/lux-blog/the-complete-guide-to-lampshade-fitters-how-to-choose-the-right-one-for-your-lamp):
  "Clip-on fitters attach directly to the bulb itself using a spring-loaded clip."
- Lamp Shade Pro, clip-on chandelier shades (lampshadepro.com/clip-on-chandelier-shades.html): "All of our
  chandelier shades clip on to a candelabra light bulb type E12." "For the best fit always use torpedo shape bulbs,
  not flame tip."
- Lamps Plus, "Beacon Falls 12" High Brass Small Candlestick Base Accent Table Lamp", item 9X865
  (lampsplus.com/p/beacon-falls-12-inch-high-brass-small-candlestick-base-accent-table-lamp__9x865/), the lamp the
  sizes are taken from: "12" high overall. Round base is 3 1/4" wide. Shade is 3" across the top x 5" across the
  bottom x 4 1/2" high." "Off-white silk hexagon bell candelabra clip shade." Its technical specifications give
  "Height 11.00 inches". The listing gives no weight.
- Sunlite, data sheet of the 01372-SU torpedo-tip chandelier bulb (shape B10, candelabra screw base E12): "Each bulb
  is 1.26 inches in diameter and 3.72 inches high."

## Related tasks on the board

Nothing on the board has a lampshade.

- "Fit a two-pin lamp tube into its sockets and turn it to lock" (accepted) fits a lamp's tube into fixed sockets
  with both hands. Here the part goes onto a free-standing lamp that can be pushed over, from above, onto a hidden
  fitter.
- "Move a lamp closer before pulling its plug to a tray" (accepted) is about a desk lamp's cord. Nothing is fitted
  to the lamp.
- "Replace an electric toothbrush's worn brush head with the new one from its pack" and "Put a CD back in its
  jewel case: press it onto the hub until the teeth grip, then shut the lid" (both accepted) are also push-on fits
  of a household part. Their parts are small, are held close to the table and go onto something that lies or is
  held still; the shade is wide and thin-walled, goes on 27 cm up, and what it goes onto falls over.
- "Hold a stick umbrella by its handle with one hand while the other pushes the runner up until it latches open"
  (accepted) also has one hand steadying while the other works a household fitting.
