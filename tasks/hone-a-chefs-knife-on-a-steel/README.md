# Hone a chef's knife on a steel

A walnut cutting board stands on a flat kitchen scale. An 8 in chef's knife lies on the board with its handle over the
board's right edge, and a 10 in honing steel lies across the board with its handle over the left edge. A green
non-slip pad is on the board. The instruction is "Hone the chef's knife on the honing steel: stand the steel on its
tip on the green pad and draw the knife's edge down it from heel to tip at 15 to 20 degrees with light pressure,
four strokes on each side of the blade, alternating sides. Then lay the knife and the steel back on the board, the
knife's edge away from you."

Everything is at full size. The table stands in for a kitchen counter.

## What the task asks of a robot

- **Two hands with different jobs.** One hand stands a 41 cm steel upright on its tip and keeps it there by its
  handle, behind the guard. The other holds the knife by its handle and makes the strokes.
- **A held angle against a round rod.** The blade's face has to stay 10 to 25 degrees off the steel all the way down
  a stroke. The edge curves up toward the tip, which by itself adds about 4 degrees there.
- **Contact that runs the whole edge.** The knife moves down the steel and back toward the robot at once, so that the
  place where the edge touches the rod runs from the heel to the tip. A stroke that only covers the middle of the edge
  does not count.
- **Light pressure, measured.** The board stands on a kitchen scale. The reading may rise by no more than 30 g during
  a stroke.
- **Both faces in turn.** A stroke on the steel's right side hones one face and a stroke on its left side the other.
  Going from one side to the other means drawing the knife back until its tip is clear of the steel.
- **A hand next to a blade.** The blade may never come within 2 cm of the gripper on the steel, and never above the
  steel's guard.

A honing steel does not sharpen: it straightens an edge that use has bent over, and takes no metal off. That is why
nothing about the result can be seen afterwards and the strokes themselves are scored.

## Running it

    # one trial with a policy (Dream-1 shown), scored at the end
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 2800 \
      --instruction "Hone the chef's knife on the honing steel: stand the steel on its tip on the green pad and draw the knife's edge down it from heel to tip at 15 to 20 degrees with light pressure, four strokes on each side of the blade, alternating sides. Then lay the knife and the steel back on the board, the knife's edge away from you."

    python demo.py task.py out_dir                 # the scripted demo (layout A); writes result.json and videos
    HONE_LOT=B python demo.py task.py out_dir_b    # layout B
    python demo.py task.py out_dir --mode heavy    # a control run; see "Results" for the modes
    python -m pytest -q test_scoring.py            # the offline checks (no simulator)

A trial is 2,800 control steps at most (140 s at 20 Hz); the demo takes 2,643 (132 s). On an L40S a trial takes about
9 minutes. Nothing is random: `object_jitter=0`, no seeds.

## The scene

Centimetres from the midpoint between the arm bases, x away from the robot, y to its left, heights above the table.
Layout A; layout B (`HONE_LOT=B`) moves the board, the pad, the knife and the steel by 0.6 to 1.2 cm and turns the
steel 3 degrees.

| Object | Size | Where |
|---|---|---|
| Kitchen scale, fixed | 15 x 20 x 2 cm | centred at (35.8, 31.0) |
| Cutting board, on the scale | 30.5 x 45.7 x 3.8 cm, 3.4 kg | near edge x 20.5, right edge y 14.6; top 6.1 up |
| Non-slip pad, on the board | 8 x 8 x 0.3 cm | centred at (37.0, 22.0) |
| Chef's knife, loose | blade 20 cm, 5.3 cm high at the heel; handle 14 x 3 x 2 cm; 181 g | flat on the board, blade to the left, edge away from the robot along x 47.8, heel at y 14.4; handle over the right edge |
| Honing steel, loose | rod 25.9 x 1.1 cm; guard 5.3 x 4.1 cm; handle 12.5 cm; 275 g | across the board, tip at (36.0, 33.0), guard on the board 0.6 cm in from the left edge, handle over that edge |

## Scoring

The scorer keeps a record through the trial (`advance()` in `task.py`) and scores it with the end state
(`score_from()`).

The edge is **on the steel** when it is within 1 mm of the rod's surface. A **stroke** is one unbroken spell of that
(a gap of up to 0.1 s does not end it). A stroke is **valid** if all of these hold:

- it starts within 3 cm of the heel and ends within 3 cm of the tip;
- the blade's face stays 10 to 25 degrees off the steel's surface at the place they touch (0 would be the blade lying
  flat against the rod, 90 the blade square to it);
- it stays on one face of the blade;
- the place of contact runs at least 5 cm down the steel, toward its tip, with the cutting edge pointing that way;
- the steel stands within 15 degrees of upright with its tip on the pad;
- the scale never reads more than 30 g above what it read just before the stroke.

A valid stroke is **counted** if it is on the other face from the last stroke counted (the first valid stroke always
counts). Strokes that are not valid do not count and cost nothing, with the exceptions below.

| Score | What it takes |
|---|---|
| 0 | fewer than two counted strokes, or a void |
| 0.25 | two counted strokes (one on each face) |
| 0.5 | eight counted strokes (four on each face, in turn) |
| 0.75 | eight, and at the end the knife lies flat on the board and the steel lies on the board |
| 1.0 | as 0.75, and the knife's edge is not turned toward the robot, both grippers are at least 3 cm from the knife and the steel, and both are at rest (under 2 cm/s and 0.5 rad/s) |

**Voids** (the whole trial scores 0):

- while the steel stands on the pad, the blade comes within 2 cm of the gripper holding it, or any part of the blade
  passes above the guard within 5 cm of the steel;
- the edge is on the steel at more than 30 degrees for 0.15 s (an edge driven into the steel);
- the scale rises 90 g during a stroke.

In a real trial the strokes are judged from the video (where the edge is on the rod, which side, the count), a
photo taken along the knife for the angle (or a 10/25 degree card held behind the steel), and the scale's display,
which should be in the camera's view.

## Getting the real parts

| Item | Product | Sizes from | Price as shown |
|---|---|---|---|
| Chef's knife | Victorinox Fibrox Chef's Knife Extra Wide, 8 in, item 5.2063.20 | maker's page: blade 7.9 in, overall 13.4 in, 2.2 in wide, 6.4 oz | $67.00 |
| Honing steel | Victorinox 10 in sharpening steel with finger guard, item 7.8991.32 | maker's page: 16.1 in overall, rod 10.2 in, 2.1 in wide, 1.6 in high, 9.7 oz | $42.00 |
| Cutting board | John Boos R-Board, walnut, 18 x 12 x 1.5 in | retailer's listing; 7.4 lb | $193.95 |
| Kitchen scale | Beurer KS 19 glass kitchen scale (5 kg, reads to 1 g), or any flat scale of that kind | retailer's listing: 14.5 x 20 x 1.65 cm | EUR 25.90 |
| Non-slip pad | an 8 cm square cut from non-slip shelf liner or a silicone mat | cut to size | a few dollars (estimate) |

Any 12 x 18 in board about 1.5 in thick will do; what matters is that its top is 6.1 cm above the counter once
it stands on the scale, and that the scale can carry it. The simulated scale is 2.0 cm high: under the one named,
which is 1.65 cm, put a 3.5 mm sheet of card. Total about $330 with the board named; a plain board of the same
size brings it under $200 (estimate).

Setting up: tape the scale down, stand the board on it and tare it. Put the pad on its mark. Lay the knife flat with
its heel 2 mm outside the board's right edge, edge away from the robot. Lay the steel across the board on its tip and
its guard, the guard 6 mm in from the left edge. For rig safety a blunted knife of the same model can be used; with a
sharp one, keep the edge away from the gripper pads when setting up.

## What the simulation simplifies

- Everything is rigid. A steel takes no metal off and the simulation changes nothing about the edge: the strokes are
  scored from where the parts are, 20 times a second.
- **Real contact, no rules.** The knife and the steel are held by friction between the gripper pads and their
  handles, under gravity. The rod is a capsule, the blade a wedge built of boxes (1 mm thick at the edge, 2.3 mm at
  the spine), the guard and the handles convex pieces. No part is attached to a hand, and no state is set by code.
- **The scale** is the board hanging on a vertical spring (10 N per mm, an estimate) with no weight of its own, which
  is what taring does. Its reading is the spring's travel. It cannot tip, where a real board on a small scale could
  if pressed near an edge.
- **Sourced sizes:** the knife's blade length, overall length and weight; the steel's overall length, rod length,
  guard width and height, and weight; the board; the named scale.
- **Estimates, not measurements:** the rod's thickness (11 mm: the maker gives none); the guard's thickness and the
  steel's handle (12.5 x 3 x 2.6 cm); the knife's blade height at the heel (5.3 cm; the maker gives 2.2 in, 5.6 cm,
  as its width), thickness, edge curve and handle section; where the knife and the steel balance; every friction
  coefficient (pads on handles 0.9, steel on steel 0.3, pad 1.1); the 6 mm patch over which a rubber pad meets a
  handle; the simulated scale's body (15 x 20 x 2 cm, against 14.5 x 20 x 1.65 cm for the one named).
- **The 30 g limit and the 90 g void are estimates, not calibrated.** The sources say only "very little pressure"
  and "light pressure". The reasoning: with steel on steel at a friction of 0.3 (itself an estimate), 30 g of drag
  on the scale is about 1 N of push on the rod, about half the knife's own weight; the void is three times that.
  The demo's strokes, aimed 4 mm into the rod's place, raise the reading by 0 to 21 g; strokes aimed 3 cm into it
  raise it by 36 g. These arms cannot push much harder than that sideways (their wrists give), so the 90 g
  void was not reached in any run. The reading also moves by a few grams with how the other hand leans on the steel.

### Calibrating the limit on a real rig

This has not been done. Before real trials:

1. Stand the board on the scale, tare it, and put the pad on its mark.
2. A person who hones knives as taught (steel upright on the pad, 15 to 20 degrees, light pressure) holds the steel
   with one hand, lets the reading settle and notes it.
3. The person makes ten strokes with the named knife on the named steel, five on each face, while the scale's
   display is filmed.
4. For each stroke, read the highest value on the display during the stroke and subtract the value just before it.
5. Set the limit at twice the largest of the ten rises and the void at three times the limit, and use those two
   numbers in place of 30 g and 90 g (`LIGHT_G` and `HEAVY_G` in `task.py`).

## The scripted demo

`demo.py` is open loop: every target comes from the layout constants and the parts' sizes, and it never reads where
a part is. Its only feedback is each hand's own place and tilt from the robot's joint readings.

1. The right hand comes in line with the lying knife's handle, from the right, one finger over it and one under,
   closes on its two flat sides 4.5 cm from its end, lifts the knife and parks it to the right of the pad.
2. The left hand takes the lying steel's handle from the robot's side, 5 cm behind the guard, one finger over and one
   under, lifts it, rolls its wrist a quarter turn so that the steel hangs tip down, stands the tip on the pad and
   holds it there.
3. Eight strokes, the first on the steel's right. For each, the right hand rolls its wrist so that the blade's face
   is 15 degrees off the steel, brings the edge against the rod near the heel, 20 cm above the pad, and draws the
   knife down and back toward the robot until the contact has run to the tip, 7 cm above the pad. It then takes the
   knife off, draws it back until its tip is clear, crosses to the steel's other side and goes back up.
4. The right hand parks. The left hand lays the steel back on the board and lets go; the right hand lays the knife
   back and lets go.

Notes on it:

- **Full grip on the knife's handle.** The knife is taken at the end of its handle (nearer the blade, the handle's
  end meets the palm) with the gripper's full 50 N on each pad, and then stays within 0.2 degrees of where it was
  taken. At half that force it turned in the pinch, up to 16 degrees tip-down over a trial, and could not be laid
  back straight.
- **The wrists give under load.** Holding the steel, the left hand tilted 3 to 4 degrees, which took the stroke's
  angle out of range. The demo corrects each hand's place and tilt from the robot's own joint readings.
- **A hand that leans on something stops correcting in that direction.** Otherwise it pushes harder every step: the
  hand on the steel pressed 1.2 kg onto the scale before this was added. The steel's tip is aimed 2 mm below the
  pad's top and the edge 4 mm into the rod's place, and both are left there.
- The knife's edge curves up toward its tip, so the angle between the blade's face and the rod rises by about 4
  degrees along a stroke; the demo aims at 15 degrees and its strokes measure 13 to 21.

## Related tasks on the board

No accepted task hones or sharpens a knife.

- A pending task, "Hone a 200 mm chef's knife on a steel at 20 degrees, both sides, then park it", has the same
  subject. It was sent back for revision and has had none since; under the program's rule its hold has lapsed. This
  task was built from scratch. It differs where that review asked for change: the steel has a collision surface and
  the knife is held by friction under gravity, the angle is measured between the blade's face and the rod's surface
  at the contact, the contact has to run from heel to tip on both faces, and pressure is a number read from a scale.
- A pending task, "Sharpen a kitchen knife on a whetstone at 15 to 20 degrees, both sides, then lay it down edge
  away", grinds an edge on a flat stone lying on the bench. A steel is a rod held in the other hand, it removes no
  metal, and the stroke runs down a round surface.
- "Hold a loaf with one hand, clear of the blade, while the other saws a slice 1 to 2 cm thick off its end"
  (accepted) uses a knife on food with the other hand kept clear. Here the other hand holds the tool the knife works
  on.

## Results

| Run | Score |
|---|---|
| Scripted demo, layout A | 1.0 (ends on success at step 2,643 of 2,800) |
| Scripted demo, layout B (board, pad, knife and steel moved; steel turned 3 degrees) | 1.0 (step 2,643) |
| Control: nothing moves | 0.0 |
| Control: two good strokes, one on each face, then both tools laid down | 0.25 (scale rose 8.3 g and 0.0 g) |
| Control: two strokes aimed 3 cm into the steel's place (the scale rose 36 g on each) | 0.0 |
| Control: two strokes at 40 degrees | 0.0 (void) |
| Control: the blade taken up past the steel's guard | 0.0 (void) |
| Control: a stroke over the middle of the edge only (5.3 cm from the heel to 6.6 cm from the tip) | 0.0 (not counted; this control's cross-over then caught the steel and the run ended there) |
| Dream-1, 2,800 steps | 0.0 (both grippers went to the board and hovered at the knife's handle and over the steel; neither tool was picked up) |

In the two demo runs all eight strokes were valid: contact from 11 mm from the heel to 19 or 20 mm from the tip,
129 to 136 mm down the steel, the blade's face 12.9 to 20.9 degrees off the steel, the scale's rise 0 to 20.7 g, and
the blade never nearer than 30 mm to the gripper on the steel. 23 offline checks pass
(`python -m pytest -q test_scoring.py`).

## Sources

- Victorinox, 10 in sharpening steel, item 7.8991.32
  (victorinox.com/us/en/Products/Cutlery/Accessories/Sharpening-Steel/p/7.8991.32): "Height: 1.6 in", "Length:
  16.1 in", "Width: 2.1 in", "Weight: 9.7 oz", "Length of blade: 10.2 in"; "10'' Knife Sharpener with Finger Guard";
  "$ 42.00".
- Victorinox, "Fibrox Chef's Knife Extra Wide, 8 in", item 5.2063.20
  (victorinox.com/us/en/Products/Cutlery/Chefs-Knives/Fibrox-Pro-8-Chef%27s-Knife/p/5.2063.20): "Height: 0.8 in",
  "Length: 13.4 in", "Width: 2.2 in", "Weight: 6.4 oz", "Length of blade: 7.9 in"; "$ 67.00".
- Beurer KS 19 kitchen scale, as listed by Kärkkäinen (karkkainen.com/en/beurer-ks19-kitchen-scale-white):
  "Ultra-slim: only 16 mm"; "Capacity 5 kg, accuracy 1 g"; "Dimensions: 14.5 x 20 x 1.65 cm (D x W x H)"; "€25.90".
- John Boos R-Board, walnut, 18 x 12 x 1.5 in, as listed by Everything Kitchens
  (everythingkitchens.com/john-boos-r-board-series-18-x-12-x-1-5-cutting-board-walnut.html): "Weight (in Lbs)
  7.400000"; "Price: $193.95".
- America's Test Kitchen, "Knife Tune-Up: How to Use a Honing Rod"
  (americastestkitchen.com/how-tos/10751-knife-tune-up-how-to-use-a-honing-rod): "Place tip of honing steel on counter
  and place heel of blade against top of steel, pointing knife tip slightly upward."; "Hold blade at 15-degree angle
  away from steel."; "Maintaining light, consistent pressure and 15-degree angle between knife blade and honing steel,
  slide blade down length of steel in sweeping motion, pulling knife toward your body so that entire edge of blade
  makes contact with steel."; "Repeat this motion on other side of blade."; "Four or five strokes on each side of
  blade (total of eight to 10 alternating passes) should realign edge."
- America's Test Kitchen, "How to Use a Sharpening Steel"
  (americastestkitchen.com/articles/276-how-to-use-a-sharpening-steel): "Despite what many cooks believe, proper
  honing requires very little pressure."; "A harsh rasping noise indicates that too much pressure is being applied."
- Consumer Reports, "How to Use a Honing Steel"
  (consumerreports.org/home-garden/kitchen-knives/how-to-use-a-honing-steel-a6993476314/): "Hold the handle of the
  honing steel with your non-dominant hand and plant the tip straight down onto a cutting board."; "Gently place the
  heel of the knife against the top of the honing steel at an approximate 20-degree angle."; "draw the knife down the
  full length of the honing steel while applying light pressure and pulling across the full length of the knife
  blade, until the tip of the knife and the tip of the honing steel meet."
- Auguste Escoffier School of Culinary Arts, "How to Care for Your Kitchen Knives"
  (escoffier.edu/blog/culinary-arts/how-to-sharpen-your-knife/): "You should hone a knife before or after each use to
  keep the blade straight, while you only need to sharpen your knife once or twice a year, depending on use."
- O*NET OnLine: 35-2014.00 Cooks, Restaurant: "1,417,500 employees" (2025); "Carve and trim meats such as beef, veal,
  ham, pork, and lamb for hot or cold service, or for sandwiches." 35-2021.00 Food Preparation Workers: "908,500
  employees"; "Wash, peel, cut, and seed fruits and vegetables to prepare them for consumption." 51-3021.00 Butchers
  and Meat Cutters: "138,400 employees"; "Cut, trim, bone, tie, and grind meats, such as beef, pork, poultry, and
  fish, to prepare in cooking form." None of these lists names honing.
