# Find the two flat tennis balls among four by their bounce

Four tennis balls lie in a wooden tray. Two have lost their pressure and gone flat ("dead"); two are good. They look
the same and weigh the same. The instruction is "Two of these four tennis balls have gone flat. Find them by their
bounce: put the two flat ones in the grey bin and the two good ones in the blue box."

Players sort old balls this way. Dropped onto something hard, a good ball comes back a little over half the height it
fell; a flat one comes back about a third.

## What the task asks of a robot

- **Nothing to see and nothing to weigh.** All four balls are 6.6 cm across and 57.6 g. A flat ball differs from a good
  one only in how it bounces, so each ball that is sorted has to be tested first (or its kind worked out from the
  others: two of the four are flat).
- **A clean release from a known height.** The ball has to be let go without a push, high enough for the two kinds
  to come back to clearly different heights. From 35 cm (bottom of the ball above the table) a good ball's bottom
  comes back to 19.4 cm and a flat one's to 12.4 cm. Over the spot the demo uses, a hand pointing straight down
  gets its tool point 20 cm above the table, so a ball's bottom 17 cm; the demo's hand leans 66 degrees back toward
  the robot to hold a ball twice that high.
- **A judgement made in half a second.** A good ball is at the top of its bounce 0.47 s after it is let go, and
  within 5 mm of it for 60 ms. The rebound can be watched, or, as in the demo, felt: a second hand waits round the
  line of the fall at the height a good ball comes back to, and closes there at that moment. A good ball is caught.
  A flat one has turned back 7 cm lower and is already falling again, and the fingers close on nothing.
- **Getting the ball back.** A ball that was not caught bounces a few more times and comes to rest on the table
  about a second later, where it has to be found and picked up.
- **Two places to put things, and a cost for the wrong one.** A flat ball in the box or a good ball in the bin
  scores 0 for the whole trial.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `ballkit.py` | The balls (size, mass, bounce, seam and pen dots), the boards of the tray, the bin and the box. |
| `demo.py` | A scripted demo that reaches full score, and three control runs. |
| `probe.py` | A check of the simulator's bounces against the rule (see below). It is not used by the task or the demo. |
| `test_scoring.py` | 61 checks of the balls, the layouts, the scoring rule and the demo's plan; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,800 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1800 \
  --instruction "Two of these four tennis balls have gone flat. Find them by their bounce: put the two flat ones in the grey bin and the two good ones in the blue box."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
BALL_LOT=B python demo.py task.py out_dir      # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the balls start in a row in the tray, 1 cm apart.
- There are no seeds. `BALL_LOT` picks the layout: A (balls 2 and 3 flat) or B (balls 1 and 4 flat; the tray, the
  bin and the box 1 to 1.5 cm from where they stand in A).
- A trial is at most 1,800 steps (90 s). It ends early once full score has held for 1 s. The demo takes 1,475 steps
  (74 s) in layout A and 1,234 (62 s) in layout B; a simulated trial of that length took 5 to 10 minutes of wall
  time on a GPU shared with other jobs.
- `result.json` from the demo holds the score, what the demo found for each ball, the task's own report at the
  start and end, and a trace every 5 steps.

## Scoring

Scored at the end of the trial, from where each ball has been left.

| Score | What it takes |
|---|---|
| 0 | Any ball is in the wrong one of the two: a flat ball in the blue box, or a good ball in the grey bin. Or no ball is in its right place. |
| 0.25 | One ball is in its right place (a flat ball in the grey bin, or a good ball in the blue box) and none is in the wrong one. |
| 0.5 | Two are, and none is in the wrong one. |
| 0.75 | Three are, and none is in the wrong one. |
| 1.0 | All four: both flat balls in the grey bin, both good balls in the blue box. |

The numbers:

- **A ball counts** once it is let go (no part of either gripper or wrist within 3 cm of it) and at rest (under
  2 cm/s). A ball still held, still rolling or still bouncing counts for nothing yet.
- **In the bin, in the box:** the ball's middle is over the inside of that one (15 x 15 cm) and below its rim. A ball
  lying on another ball inside counts.
- **Anywhere else** (left in the tray, on the table, off the table) counts for nothing either way.
- **Which balls are flat** is fixed by the layout and marked by the pen dots: in layout A the balls with two and
  three dots, in layout B those with one and four.

What a trial without the test scores:

- Sorting two and two by guesswork is right one time in six (1.0) and scores 0 the other five, because a wrong pair
  always puts a good ball in the bin.
- Putting a single ball in the bin by guesswork scores 0.25 half the time and 0 the other half.
- No fixed habit (always the same two balls to the bin) scores above 0 in both layouts: every ball that is flat in
  one layout is good in the other.
- Everything into the bin, or everything into the box: 0.

## Scoring a real trial

- Photograph the bin and the box from above at the end, then take the balls out and read their dots.
- A ball counts if nothing is touching it and it lies still in the last second of the video.
- Before a session, check that the balls are what the layout says: drop each from 35 cm (bottom of the ball) beside
  a ruler onto the place where the robot will test them. A good ball's bottom should come back to about 19 cm, a
  flat one's to about 12 cm.

The task has not been run on a physical rig.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `ballkit.py`, and the robot's own readings: each
hand's measured position (to correct for arm sag) and the left hand's measured finger opening (to tell whether it
holds a ball). It never reads where a ball is or how it bounces.

For each ball, nearest first:

1. The right hand takes the ball out of the tray from above, a fingertip on each side over the tray's rim, and
   carries it to a spot on the bare table 18 cm to the tray's left. There it holds the ball with its bottom 35 cm
   above the table, the hand leaning 66 degrees back.
2. The left hand comes in level from the left and waits round the line of the fall, 15 cm below the ball, its pads
   8.8 cm apart at the height the middle of a good ball comes back to (22.7 cm).
3. The right hand opens. The ball falls between the left hand's fingers and bounces. Eight control steps (0.4 s)
   after the right hand was told to open, the left hand is told to close to 5 cm.
4. Half a second later the left hand's fingers are read. Further apart than 5.8 cm: they hold a ball, and it is a
   good one. In the runs here they read 65.5 or 65.6 mm on a ball and 50.0 mm on nothing.
5. A good ball goes to the blue box in the left hand, which lowers it to 4.5 cm above the box's floor and lets go.
   A flat ball is left to settle (it came to rest 3 to 6 mm from the drop spot, within 1.5 s of being let go);
   the left hand draws back, and the right hand comes down, picks the ball up off the table and drops it in the
   grey bin.

Once two good balls have been found, the rest are flat and go straight from the tray to the bin. Once two flat ones
have been found, the rest are good; they still go by the drop and the catch, which is also how a ball gets from
the right hand to the left.

| | Layout A (balls 2 and 3 flat) | Layout B (balls 1 and 4 flat) |
|---|---|---|
| Ball 1 let go; verdict | step 170; caught, good | step 170; not caught, flat |
| Ball 1 in its place | box, step 260 | bin, step 405 |
| Ball 2 let go; verdict | step 485; not caught, flat | step 595; caught, good |
| Ball 2 in its place | bin, step 720 | box, step 690 |
| Ball 3 let go; verdict | step 910; not caught, flat | step 910; caught, good |
| Ball 3 in its place | bin, step 1,145 | box, step 1,000 |
| Ball 4 | let go at step 1,335; caught, good; box, step 1,430 | not tested (the other two are good); bin, step 1,210 |
| Score 0.25, 0.5, 0.75 first reached | steps 290, 725, 1,155 | steps 415, 720, 1,030 |
| All four let go and at rest (score 1.0) | step 1,460 | step 1,215 |
| Trial ends with full score | step 1,475 | step 1,234 |

(Steps are read from a trace written every 5 steps. When a second ball lands in the box beside the first, the first
is knocked and does not count again until it is at rest, so the score dips for a few steps on the way to 1.0.)

## Control runs

| Run | Score | What happened |
|---|---|---|
| `--mode still` (layout A) | 0.0 | Nothing moves. |
| `--mode guess` (layout A): the two nearest balls go to the bin untested | 0.0 | Ball 1 is good and is now in the bin, with ball 2, which is flat. |
| `--mode guess` (layout B) | 0.0 | 0.25 after ball 1, which is flat in this layout, went in; 0 once ball 2, which is good, followed it. |
| `--mode half` (layout A): the first two balls are tested and sorted, then the demo stops | 0.5 | Ball 1 (good) is in the box and ball 2 (flat) in the bin; balls 3 and 4 are still in the tray. |

## Dream-1

One run, 1,800 steps, layout A (after a 200-step test): **0.0**. The right gripper moved to the tray in the first
5 seconds and stayed over it for the rest of the trial, coming down among the balls several times. The balls turned
and shifted in the tray; none was lifted out. The left arm drew back toward its base and stayed there. At the end all
four balls were still in the tray.

## Does a ball bounce in the simulator as the rule says?

`python probe.py task.py` places a good and a flat ball in the air and lets them fall onto the table, reading their
height 120 times a second. (Placing a ball in the air directly is something only this check does.)

| Drop (bottom of the ball) | Good ball came back | Rule (55.5%) | Flat ball came back | Rule (35.5%) |
|---|---|---|---|---|
| 15 cm | 8.27 cm | 8.33 cm | 5.29 cm | 5.33 cm |
| 25 cm | 13.86 cm | 13.88 cm | 8.88 cm | 8.88 cm |
| 35 cm | 19.40 cm | 19.43 cm | 12.42 cm | 12.43 cm |
| 45 cm | 24.82 cm | 24.98 cm | 15.84 cm | 15.99 cm |

Every rebound is within 1% of the rule. From 35 cm the good ball was at its highest 0.458 s after it started to
fall and the flat one 0.425 s; the good ball had stopped bouncing after 1.5 s and the flat one after 0.9 s. Neither
moved sideways.

## Building it for real

- Four tennis balls of one make and colour. Two good ones from a newly opened can. Two flat ones: balls of the same
  make that have lost their pressure (old ones, or ones let down with a hollow needle through the seam). Check all
  four with the 35 cm drop described under "Scoring a real trial".
- Mark the balls with one, two, three and four dots of a permanent pen at two opposite points, the flat ones as the
  layout says.
- A shallow wooden tray about 30.4 x 7.6 cm inside with a 1.5 cm rim (a pen tray or a cutlery-drawer insert), taped
  down.
- Two open plastic boxes about 15 x 15 cm inside and 9 cm deep, one grey and one blue, taped down.
- A table top that does not give under a bouncing ball. If good balls come back less than 17 cm from a 35 cm drop,
  put a stone tile on the table where the balls are tested.
- Tape marks for the tray, the bin and the box in the two layouts.

## What the simulation simplifies

- **A ball is a rigid sphere** with one coefficient of restitution on every surface. A real ball flattens against
  what it hits; its bounce depends a little on speed, temperature and the surface. The good ball's coefficient
  (0.745) is the one that gives the ITF's 141 cm from 254 cm; the flat ball's (0.596) is 20% lower, the loss
  measured for a punctured ball in impacts on a rigid surface and on a clamped racket. Using the same two numbers
  for a 35 cm drop onto a table is an assumption.
- **A flat ball cannot be told by squeezing it here.** A real flat ball also gives more under the fingers; the
  simulated balls are equally hard. Squeeze stiffness is not modelled, and neither are the felt, spin or air drag.
- **Sourced:** the ball's diameter (6.60 cm) and mass (57.6 g), the drop and rebound of the rule (254 cm; 135 to
  147 cm; 141 cm for the ITF's test ball), the 20% loss of a punctured ball, the fingers' top speed (0.146 m/s,
  from the robot's description).
- **Estimates, not measurements:** the friction of felt (0.6 static, 0.5 moving); the angular damping that stands in
  for rolling resistance (4 per second); the sizes of the tray, the bin and the box; the 20 ms allowed between the
  command to open and the ball falling, and the 20 ms added to the fingers' closing time.
- The ball's moment of inertia is that of a thin shell. Its seam and pen dots are drawn only; they do not touch
  anything.
- The tray, the bin and the box are fixed to the table and rigid.
- The demo counts on a flat ball coming to rest within a centimetre of where it landed, which holds in the
  simulator. On a real table a ball can roll further and would have to be looked for.

## Sources

- International Tennis Federation, "ITF Approved Tennis Balls, Classified Surfaces & Recognised Courts: a guide to
  products and test methods", 2025 (world.tennis/media/14104/2025-technical-booklet.pdf). Its table for a type 2
  ball: "MASS (WEIGHT) 56.0-59.4 grams", "SIZE 6.54-6.86 cm", "REBOUND 135-147 cm". The rebound test: "The ball is
  dropped from a height of 254 ± 0.3 cm (100 ± 0.1 inches) (measured from the bottom of the ball) onto a smooth,
  rigid and horizontal block of high mass and the rebound height is measured (again to the bottom of the ball)."
  Its test ball for court testing: "Mass 57.6 ± 0.3 g", "Diameter 6.60 ± 0.05 cm", "Rebound 141 ± 1 cm".
- S. J. Haake, M. J. Carré and S. R. Goodwill, "The dynamic impact characteristics of tennis balls with tennis
  rackets", Journal of Sports Sciences 21(10), 2003, pages 839-850 (abstract, PubMed 14620027): "Pressurized balls
  had the highest coefficient of restitution, which decreased by 20% when punctured."
- Tennis Creative, "How Long Do Tennis Balls Last?" (tenniscreative.com/?p=1993): "To find out if a tennis ball has
  lost its bounce, simply compare it to another one." "Drop both tennis balls from the same height." "If one drops
  significantly lower than the other it's probably wrecked and needs replacing."

## Related tasks on the board

No task on the board has a tennis ball, and none judges an object by how it bounces.

- "Sort kiwis by squeezing them gently: the ripe ones into the bowl, the hard ones into the paper bag" (pending)
  also sorts things that look alike by a physical test. Its test is a slow squeeze of a fruit lying in a tray; this
  one lets a ball go and judges half a second of flight, and the ball has to be got back afterwards.
- "Ping-pong keep-ups: 10 bounces in a row on the bat" (pending) keeps a ball bouncing as a skill. Nothing is judged
  from the bounce and nothing is sorted.
- "Find the used battery" (pending) finds the odd one out with a meter reading.
