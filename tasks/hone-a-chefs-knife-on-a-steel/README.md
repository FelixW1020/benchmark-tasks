# Hone a chef's knife on a steel

A polyethylene cutting board lies on a bench scale whose display is a separate indicator. An 8 in chef's knife lies on the board with its handle over the
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
- **Contact that runs the whole edge, in order.** The knife moves down the steel and back toward the robot at once,
  so that the place where the edge presses on the rod runs from the heel to the tip and down the steel. The edge has
  to be pressed on the rod the whole way: a pass that hovers beside it, one that only covers the middle of the edge,
  and one that runs from the tip to the heel do not count.
- **Light pressure, measured.** The board lies on a scale. The reading may rise by no more than 30 g during a
  stroke.
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
    python demo.py task.py out_dir --mode heavy    # a control run: two, still, heavy, steep, guard, middle, hover, reverse
    python -m pytest -q test_scoring.py            # the offline checks (no simulator)

A trial is 2,800 control steps at most (140 s at 20 Hz); the demo takes 2,645 (132 s). On an L40S a trial takes about
9 minutes. Nothing is random: `object_jitter=0`, no seeds.

## The scene

Centimetres from the midpoint between the arm bases, x away from the robot, y to its left, heights above the table.
Layout A; layout B (`HONE_LOT=B`) moves the board, the pad, the knife and the steel by 0.6 to 1.2 cm and turns the
steel 3 degrees.

| Object | Size | Where |
|---|---|---|
| Bench scale, standing on the table | base and pan 30 x 30 x 5 cm; its display is a separate indicator on a cable | centred at (35.8, 31.0) |
| Cutting board, on the scale's pan | 30.5 x 45.7 x 1.27 cm, about 1.5 kg | near edge x 20.5, right edge y 14.6; top 6.6 up |
| Non-slip pad, on the board | 8 x 8 x 0.3 cm | centred at (37.0, 22.0) |
| Chef's knife, loose | blade 20 cm, 5.3 cm high at the heel; handle 14 x 3 x 2 cm; 181 g | flat on the board, blade to the left, edge away from the robot along x 47.8, heel at y 14.4; handle over the right edge |
| Honing steel, loose | rod 25.9 x 1.1 cm; guard 5.3 x 4.1 cm; handle 12.5 cm; 275 g | across the board, tip at (36.0, 33.0), guard on the board 0.6 cm in from the left edge, handle over that edge |

## Scoring

The scorer keeps a record through the trial (`advance()` in `task.py`) and scores it with the end state
(`score_from()`).

The edge is **on the steel** while the blade and the rod are pressed together: in the simulator, while it reports a
contact force of 0.02 N or more between the knife and the steel (PhysX's contact view for the two bodies, read after
every physics step and averaged over the control step) and the edge's nearest point is within 2 mm of the rod, which
says where the contact is; on a real rig, while the contact lamp is lit (see "Checking a real trial").
Being close to the rod is not contact: a blade held half a millimetre off it is not on the steel. A **stroke** is one
unbroken spell of contact; it ends at the first control step (0.05 s) without it, so a break splits a pass into two
spells. Where the contact is (how far along the edge, how far up the rod, at what angle) is read from the two parts'
poses at each step, in time order. A stroke is **valid** if all of these hold:

- the contact begins within 3 cm of the heel and ends within 3 cm of the tip;
- it moves toward the tip and down the steel in that order: at no step is it more than 1.5 cm nearer the heel, or more
  than 1.5 cm higher on the steel, than it has already been (so a pass from the tip to the heel, a pass up the steel
  and a sawing pass do not count; the 1.5 cm allows for the contact settling as the edge first meets the rod);
- the blade's face stays 10 to 25 degrees off the steel's surface at the place they touch (0 would be the blade lying
  flat against the rod, 90 the blade square to it);
- it stays on one face of the blade;
- the contact ends at least 5 cm lower on the steel than it began, with the cutting edge pointing that way;
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

### Checking a real trial

None of this has been tried on a physical rig.

- **Contact: a lamp.** The knife and the steel are both steel; the board, the pad and the gripper pads do not
  conduct. Wire a 3 V coin cell, a 100 ohm resistor and an LED in series with two thin, limp leads (30 AWG silicone
  wire, about 60 cm each); clip one to the blade's spine just in front of the handle and the other to the rod just
  below the guard, both clear of the part of the edge and the 13 cm of rod that a stroke uses. Stand the LED where
  the camera sees it. It is lit exactly while steel touches steel, so a stroke is one unbroken lit spell in the
  video. This takes the place of the simulator's contact force. (A multimeter on its continuity range with its
  screen in view does the same.) A camera at 30 frames a second sees breaks of a frame or more; the simulator's rule
  breaks a stroke on 0.05 s without contact.
- **Where the contact is, and which way it goes.** Put a mark every centimetre along the blade's face near the edge
  and along the rod. In the frames where the lamp comes on and goes off, read where the rod crosses the edge and
  how high on the rod: within 3 cm of the heel at the start, within 3 cm of the tip at the end, at least 5 cm lower
  on the rod at the end. Step through the lit frames to check that the crossing never moves back by more than 1.5 cm.
- **Angle.** A photo or a second camera looking along the knife, or a card cut to 10 and 25 degrees stood behind the
  steel.
- **Pressure.** The scale's indicator stands in the camera's view (that is why a scale with a separate display is
  named: a board on an ordinary kitchen scale covers its display). The rise for a stroke is the highest reading
  while the lamp is lit less the reading in the frame before it came on. The named scale reads in steps of 2 g.

## Getting the real parts

| Item | Product | Sizes from | Price as shown |
|---|---|---|---|
| Chef's knife | Victorinox Fibrox Chef's Knife Extra Wide, 8 in, item 5.2063.20 | maker's page: blade 7.9 in, overall 13.4 in, 2.2 in wide, 6.4 oz | $67.00 |
| Honing steel | Victorinox 10 in sharpening steel with finger guard, item 7.8991.32 | maker's page: 16.1 in overall, rod 10.2 in, 2.1 in wide, 1.6 in high, 9.7 oz | $42.00 |
| Cutting board | Choice 18 x 12 x 1/2 in brown polyethylene cutting board | retailer's listing: 18 in, 12 in, 1/2 in; shipping weight 3.24 lb | $6.99 |
| Scale | Adam Equipment CPWplus 6 bench scale (6 kg, reads to 2 g), whose indicator is separate from the pan, on a cable | maker's page: pan 300 x 300 mm, overall 300 x 300 x 50 mm | not read (the maker's page shows no price) |
| Contact lamp | a 5 mm LED, a CR2032 cell in a holder, a 100 ohm resistor, two 60 cm leads of 30 AWG silicone wire, two small clips | made up | a few dollars (estimate) |
| Non-slip pad | an 8 cm square cut from non-slip shelf liner or a silicone mat | cut to size | a few dollars (estimate) |

Any scale will do that reads to 2 g or better, carries 3 kg, has a pan at least 30 cm square and 5 cm above the
table, and shows its reading on a display that the board does not cover (a separate indicator, a pull-out display or
a remote one). A board lying on an ordinary kitchen scale covers its display, which is why one is not used. The
simulated board's top is 6.6 cm above the table with a 3 mm gap under it; on the named scale and board it is 6.3 cm,
so put a 3 mm sheet of card or non-slip mat between the pan and the board (the mat also stops the board sliding).
The knife, steel and board come to about $116 at the prices read; the scale's price was not on the page read.

Setting up: stand the scale on the table with its indicator beside it, facing the camera; lay the board on the pan
and tare it. Put the pad on its mark. Lay the knife flat with its heel 2 mm outside the board's right edge, edge away
from the robot. Lay the steel across the board on its tip and its guard, the guard 6 mm in from the left edge. Clip
the lamp's leads on (see "Checking a real trial") and lay them so that they hang slack behind the tools. For rig
safety a blunted knife of the same model can be used; with a sharp one, keep the edge away from the gripper pads when
setting up.

## What the simulation simplifies

- Everything is rigid. A steel takes no metal off and the simulation changes nothing about the edge: the strokes are
  scored from where the parts are, 20 times a second.
- **Real contact, no rules.** The knife and the steel are held by friction between the gripper pads and their
  handles, under gravity. The rod is a capsule, the blade a wedge built of boxes (1 mm thick at the edge, 2.3 mm at
  the spine), the guard and the handles convex pieces. No part is attached to a hand, and no state is set by code.
- **Contact is the simulator's own.** "On the steel" is PhysX's reported contact force between the blade's colliders
  and the rod, not a distance. The lamp on a real rig shows touching, not force; the scale shows the force.
- **The scale** is the board hanging on a vertical spring (10 N per mm, an estimate) with no weight of its own, which
  is what taring does. Its reading is the spring's travel. It cannot tip. The scale's base and pan are one fixed
  block, and its indicator and cable are not drawn. The lamp and its two leads are not simulated.
- **Sourced sizes:** the knife's blade length, overall length and weight; the steel's overall length, rod length,
  guard width and height, and weight; the board's size; the scale's pan, height and readability.
- **Estimates, not measurements:** the rod's thickness (11 mm: the maker gives none); the guard's thickness and the
  steel's handle (12.5 x 3 x 2.6 cm); the knife's blade height at the heel (5.3 cm; the maker gives 2.2 in, 5.6 cm,
  as its width), thickness, edge curve and handle section; where the knife and the steel balance; every friction
  coefficient (pads on handles 0.9, steel on steel 0.3, pad 1.1); the 6 mm patch over which a rubber pad meets a
  handle; the board's weight (about 1.5 kg, from its 3.24 lb shipping weight).
- **The 30 g limit and the 90 g void are estimates, not calibrated.** The sources say only "very little pressure"
  and "light pressure". The reasoning: with steel on steel at a friction of 0.3 (itself an estimate), 30 g of drag
  on the scale is about 1 N of push on the rod, about half the knife's own weight; the void is three times that.
  The demo's strokes, aimed 4 to 8 mm into the rod's place (0.4 to 0.7 N of contact force as the simulator reports
  it), raise the reading by 0 to 20 g; a stroke aimed 3 cm into it (up to 3 N) raised it by 34 g on one side of the
  steel and lowered it on the other (see "Results"). The 90 g void was not reached in any run.

### Calibrating the limit on a real rig

This has not been done. Before real trials:

1. Lay the board on the scale's pan, tare it, and put the pad on its mark.
2. A person who hones knives as taught (steel upright on the pad, 15 to 20 degrees, light pressure) holds the steel
   with one hand, lets the reading settle and notes it.
3. The person makes ten strokes with the named knife on the named steel, five on each face, while the scale's
   indicator and the contact lamp are filmed.
4. For each stroke, read the highest value on the indicator while the lamp is lit and subtract the value just
   before it came on.
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
   is 15 degrees off the steel, presses the edge against the rod near the heel, 20 cm above the pad, and draws the
   knife down and back toward the robot until the contact has run to the tip, 7 cm above the pad. It then takes the
   knife off, draws it back until its tip is clear, crosses to the steel's other side and goes back up.
4. The right hand parks. The left hand lays the steel back on the board and lets go; the right hand lays the knife
   back and lets go.

Notes on it:

- **Grip force, kept under the real gripper's.** The real gripper gives about 40 N on each finger; the simulated
  one can give 50 N. The demo commands a 6 mm gap on the knife's 20 mm handle and a 16 mm gap on the steel's 26 mm
  handle, which the finger drive (5 N for each millimetre a finger is short of its command) turns into about 35 N
  and 25 N a finger. `demo.py` records the force each finger's motor applies at every step (`grip_per_finger` in
  `result.json`): in the two demo runs the knife hand's fingers never applied more than 36.3 N and the steel hand's
  never more than 27.4 N. The knife is taken at the end of its handle (nearer the blade, the handle's end meets the
  palm). At this force the knife creeps in the pinch, about 8 degrees over the eight strokes; the strokes stay valid
  and it is laid back flat with its edge away. In an earlier trial at about 25 N a finger it turned 16 degrees and
  could not be laid back straight; at 50 N it did not creep, but the real gripper does not give that.
- **The wrists give under load.** Holding the steel, the left hand tilted 3 to 4 degrees, which took the stroke's
  angle out of range. The demo corrects each hand's place and tilt from the robot's own joint readings.
- **A hand that leans on something stops correcting in that direction.** Otherwise it pushes harder every step: the
  hand on the steel pressed 1.2 kg onto the scale before this was added. The steel's tip is aimed 2 mm below the
  pad's top and left there.
- **Pressing harder toward the tip.** The edge is aimed 4 mm inside the rod's surface at the heel, rising to 8 mm at
  the tip, where the edge curves away from the rod. Aimed a constant 4 mm in, the contact broke near the tip on one
  face (the blade chatters on the rod there), and the scorer split those strokes. Aimed 12 mm in at the tip, one
  stroke on layout B raised the scale 31 g and did not count.
- The knife's edge curves up toward its tip, so the angle between the blade's face and the rod rises by about 4
  degrees along a stroke; the demo aims at 15 degrees and its strokes measure 13 to 20.

## Related tasks on the board

No accepted task hones or sharpens a knife. The nearest accepted task is "Hold a loaf with one hand, clear of the
blade, while the other saws a slice 1 to 2 cm thick off its end", which uses a knife on food with the other hand kept
clear. Here the other hand holds the tool the knife works on, and the knife cuts nothing.

## Results

All runs are on the files in this folder. Scores are as the scorer reported them at the end of each run.

| Run | Score |
|---|---|
| Scripted demo, layout A | 1.0 (ends on success at step 2,645 of 2,800) |
| Scripted demo, layout B (board, pad, knife and steel moved; steel turned 3 degrees) | 1.0 (step 2,645) |
| Control: nothing moves | 0.0 |
| Control: two good strokes, one on each face, then both tools laid down | 0.25 |
| Control: a pass on each face aimed 3 mm clear of the rod (`hover`) | 0.0 (the blade brushed the rod ten times for 0.05 to 0.3 s each; no spell was a stroke) |
| Control: two passes run backwards, tip to heel and up the steel, with real contact (`reverse`) | 0.0 (not counted: the contact went 126 and 168 mm back toward the heel and 105 and 149 mm back up the steel) |
| Control: a stroke on each face over the middle of the edge only (`middle`) | 0.0 (not counted: started 54 mm from the heel, ended 66 and 67 mm from the tip) |
| Control: two strokes aimed 3 cm into the steel's place (`heavy`) | 0.0 (one stroke not counted: the scale rose 34 g; see the note below) |
| Control: two strokes at 40 degrees (`steep`) | 0.0 (void) |
| Control: the blade taken up past the steel's guard (`guard`) | 0.0 (void) |
| Dream-1, 2,800 steps | 0.0 (both arms folded down near their bases; the left gripper went out to the steel's handle at the board's left edge twice and came back; neither tool was picked up) |

In the two demo runs all sixteen strokes were valid and counted. Contact was reported at every control step of each
stroke (57 to 62 steps), 0.02 to 1.7 N and typically 0.4 to 0.7 N; it began 12 to 17 mm from the heel and ended 16
to 22 mm from the tip, 130 to 146 mm lower on the steel, and never went back up the steel by more than 7.4 mm. The
blade's face was 13.0 to 19.5 degrees off the steel, the scale rose by 0 to 19.5 g, and the blade came no nearer than
33 mm to the gripper on the steel. The knife hand's fingers applied at most 36.3 N and the steel hand's 27.4 N.
30 offline checks pass (`python -m pytest -q test_scoring.py`).

A note on the `heavy` control, and a limit of the scale as a gauge. Both of its strokes pressed with up to 3 N, four
to eight times the demo's. On the steel's right side the scale rose 34 g and the stroke was not counted. On its left
side the same push lowered the reading by 110 g at its lowest and raised it by only 18.5 g, so that stroke counted;
the run scores 0 because one counted stroke is not two. The reading falls on that side because the push tilts the
steel against the hand that holds it, and the hand then carries part of the steel's weight. The rule looks only at
how far the reading rises, so as it stands it catches heavy pressure on one side of the steel only. A rule on how far
the reading moves either way would catch both sides but needs calibrating on a real rig first: in the demo runs the
reading already falls by up to 36 g on that side.

## Sources

- Victorinox, 10 in sharpening steel, item 7.8991.32
  (victorinox.com/us/en/Products/Cutlery/Accessories/Sharpening-Steel/p/7.8991.32): "Height: 1.6 in", "Length:
  16.1 in", "Width: 2.1 in", "Weight: 9.7 oz", "Length of blade: 10.2 in"; "10'' Knife Sharpener with Finger Guard";
  "$ 42.00".
- Victorinox, "Fibrox Chef's Knife Extra Wide, 8 in", item 5.2063.20
  (victorinox.com/us/en/Products/Cutlery/Chefs-Knives/Fibrox-Pro-8-Chef%27s-Knife/p/5.2063.20): "Height: 0.8 in",
  "Length: 13.4 in", "Width: 2.2 in", "Weight: 6.4 oz", "Length of blade: 7.9 in"; "$ 67.00".
- Adam Equipment, CPWplus 6 bench scale (adamequipment.com/cpwplus-6): "Capacity 13lb / 6kg"; "Readability 0.005lb /
  0.002kg"; "Pan size 11.8" x 11.8" / 300 x 300mm"; "Overall dims 11.8"x11.8"x2.0" / 300x300x50mm"; "Wall-mount
  bracket for placing indicator on vertical surface".
- Choice 18" x 12" x 1/2" Brown Polyethylene Cutting Board, as listed by WebstaurantStore
  (webstaurantstore.com/choice-18-x-12-x-1-2-brown-polyethylene-cutting-board/40718125BR.html): "Length 18 Inches",
  "Width 12 Inches", "Thickness 1/2 Inch", "Shipping Weight 3.24 lb."; "$6.99 /Each".
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
