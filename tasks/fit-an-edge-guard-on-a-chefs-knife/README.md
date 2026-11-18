# Slide an edge guard onto a chef's knife

A kitchen counter after the cooking is done. A 20 cm chef's knife lies on its side on a walnut cutting board, cutting
edge toward the robot. Its plastic edge guard, a folded PVC channel, leans against the front of the board. Before the
knife goes into a drawer or a knife roll the guard has to go on: the instruction is "Put the edge guard on the chef's
knife without touching the knife's cutting edge, then leave the knife lying on the cutting board."

The table stands in for the counter.

## What the task asks of a robot

- **Two hands, each with one safe place to hold.** The safe place to hold the knife is its handle, which leaves
  20 cm of blade sticking out of the hand. The guard has to be held so that its slot stays open along its whole length: by
  its folded edge, a finger on each face.
- **A long thin slot onto a long thin edge.** The slot is 3.5 mm wide and 21.6 cm long. Where its mouth ends up, the
  blade is 1.9 mm thick. The two parts have to be brought into one plane to about a millimetre and a degree while
  both are held in the air.
- **A curved edge.** The straight 9 cm next to the heel goes in first. The edge then curves up 3.2 cm to the tip,
  which is the last part to be covered: the guard has to go on to its full depth.
- **A part too thin to pick up off the counter.** The guard is 5.9 mm thick. It can be taken only where it stands
  clear of a surface: here by the centimetre of its folded edge that stands above the board.
- **A rule that holds for the whole trial.** No part of either arm may come within 1 cm of any part of the edge
  that is still bare. The grasp a parallel gripper would naturally make on a blade lying flat, one finger on the
  spine and one on the edge, ends the trial at zero (`--mode blade`).

## Files

| File | What it is |
|---|---|
| `task.py` | Layout, the record kept every control step, the scoring rule. |
| `guardkit.py` | The knife and the guard, the points on the edge and on the arms that the record uses. |
| `demo.py` | A scripted demo that reaches full score, and four control runs. |
| `test_scoring.py` | 50 checks of the shapes, the layouts, the record, the scoring rule and the demo's plan; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,500 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1500 \
  --instruction "Put the edge guard on the chef's knife without touching the knife's cutting edge, then leave the knife lying on the cutting board."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
GUARD_LOT=B python demo.py task.py out_dir     # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the guard starts leaning on the board's edge and the layouts are exact.
- There are no seeds. `GUARD_LOT` picks the layout: A, or B (the knife 0.8 cm further away and 1.5 cm to the right,
  the guard 2.5 cm to the right).
- A trial is at most 1,500 steps (75 s). It ends early once full score has held for 1 s. The demo takes 816 steps
  (41 s); on a GPU shared with other jobs a demo run took about 7 minutes of wall time, start-up included.
- `result.json` from the demo holds the score, the task's own report at the start and the end, and a trace every
  five steps.

## Scoring

The rule about the bare edge is watched at every control step. Everything else is read from the end of the trial.

| Score | What it takes |
|---|---|
| 0 | Some part of an arm came within 1 cm of a bare point of the edge, at any time. Or: less than a quarter of the edge is covered at the end. |
| 0.25 | At least a quarter of the edge is covered. |
| 0.5 | The whole edge is covered. |
| 0.75 | The whole edge is covered and the guard is seated. |
| 1.0 | As 0.75, and the knife lies on the board, let go and at rest. |

The numbers:

- **The edge** is the cutting edge from 4 mm in front of the heel to the tip, taken as points 4 mm apart.
- **Covered:** a point of the edge is covered when it is inside the guard's slot: between the two walls, within the
  guard's length, at least 5 mm in from the slot's open side. Any other point is **bare**. Which way round the guard
  is on does not matter.
- **Near the bare edge:** any part of either arm, from the upper arm to the fingertips, within 1 cm of a bare point.
  The arms' collision shapes are used, placed where the simulator has them (points 4 mm apart on the grippers and
  fingertips, 12 mm apart on the other links).
- **Seated:** every point of the straight 9 cm of edge next to the heel is within 4 mm of the slot's bottom. The
  spine then stands no more than 11 mm out of the guard at the heel.
- **Lying on the board:** the knife's lowest point is within 4 mm of the board's top, and no part of the knife or
  the guard is beyond the board's edge. **Let go:** both arms at least 3 cm from the knife and the guard. **At
  rest:** knife and guard under 2 cm/s and 0.5 rad/s.

Other ways to try it, and what they score: holding the knife by its blade, 0 (run below); laying the guard flat on
top of the blade, 0, because no point of the edge is in the slot; pushing the guard on over the spine, 0, because
only the tip then lies in the slot (a fifth of the edge).

## Scoring a real trial

- **Near the bare edge:** from the video. A gripper or arm closer than 1 cm to a part of the edge that is still
  outside the guard scores 0. Mark 1 cm on the board for scale.
- **Covered:** the guard is translucent, so the edge can be seen inside it. No part of the edge is outside the guard
  between the heel and the tip (4 mm at the heel does not count), and the edge is at least 5 mm inside the guard's
  open side everywhere (ruler).
- **Seated:** measure how far the spine stands out of the guard at the heel: 11 mm or less.
- **Lying, let go, at rest:** from the last second of the video.

The task has not been run on a physical rig.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `guardkit.py`, and each hand's own measured
position (to correct for arm sag). It never reads where a part is.

1. The right hand takes the guard by its folded edge, a finger on each face, 7 cm from its heel end. It lifts the
   guard and leans forward 85 degrees, until the guard lies flat in its fingers, open side forward.
2. The left hand comes straight down over the handle, 4 cm behind the heel, closes on the handle's back and belly,
   lifts the knife 6 cm off the board and tips it 5 degrees, edge up, into the guard's plane. The edge is then
   1.5 cm from the guard's open side.
3. The right hand pushes the guard onto the edge, heel end first (the guard is turned 1.5 degrees and straightens on
   the way), until the edge is at the slot's bottom (aimed half a millimetre short of it). It slides the guard
   4.5 mm along the blade, up to the handle, lets go and draws back.
4. The left hand lays the knife, guard on, back on the board and lets go.

| | Layout A | Layout B |
|---|---|---|
| Right hand closed on the guard's folded edge | step 110 | step 110 |
| Guard lying flat in the right hand | step 235 | step 235 |
| Left hand closed on the handle | step 355 | step 355 |
| Knife lifted and tipped | step 410 | step 410 |
| A quarter of the edge covered (score 0.25) | step 465 | step 470 |
| Whole edge covered (score 0.5) | step 495 | step 510 |
| Guard seated (score 0.75) | step 545 | step 510 |
| Right hand lets go of the guard | step 565 | step 565 |
| Left hand lets go of the knife | step 775 | step 775 |
| Let go and clear (score 1.0) | step 800 | step 800 |
| Trial ends with full score | step 816 | step 817 |
| Edge from the slot's bottom at the end | 3.5 mm | 2.4 mm |
| Guard's end from the heel at the end | 1.9 mm | 0.8 mm |
| Nearest an arm came to the bare edge | 29.9 mm | 30.0 mm |

(Steps are read from a trace written every 5 steps. The nearest approach is the left hand's finger on the handle's
belly, next to the heel.)

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves. |
| `--mode blade`: the left hand takes the knife by its blade, one finger on the spine and one on the edge, lifts it 3 cm and puts it back | 0.0 | A fingertip came within 1 cm of the bare edge at step 86, on the way down, and within 0.7 mm of it as the fingers closed. |
| `--mode half`: the same demo, the guard stopped with its open side 1.2 cm past the heel's edge | 0.25 | 70% of the edge is in the slot at the end; the last 6 cm, up to the tip, are bare. No arm near the bare edge. |
| `--mode shallow`: the same demo, the guard stopped 7 mm short of full depth | 0.5 | The whole edge is covered, 6.4 mm from the slot's bottom: not seated. The knife lies on the board, let go. |

## Dream-1

One run, 1,500 steps, layout A (after a 200-step test): **0.0**. In the first 15 seconds both arms came down toward
the table in front of the board. They stayed there for the rest of the trial, shifting a little, their grippers
turned toward the knife. Neither gripper closed on the knife or on the guard: the knife stayed where it lay and the
guard stayed leaning against the board.

## Building it for real

- An 8 in (20 cm) stamped chef's knife with a moulded handle (the sizes here are those of the Victorinox Fibrox Pro,
  item 5.2063.20) and an 8 in chef's-knife edge guard of folded PVC (Messermeister EGS-08C, 8.5 x 1 15/16 in).
  For trials with people nearby, use an unsharpened training knife of the same shape, or tape the edge.
- A hardwood cutting board 18 x 12 x 1.5 in (John Boos WAL-R01, walnut).
- Tape marks on the board for the knife's two places, and on the table for the guard's.
- Lean the guard against the board's front edge on its open side, folded edge up, 8 degrees from upright.

## What the simulation simplifies

- Everything is rigid.
- **The guard does not grip.** It is a rigid channel whose walls stand 3.5 mm apart. A real PVC guard's walls lie
  against the blade and hold it by friction, so a real guard needs a light push to go on and stays on when the
  knife is turned over. The simulated guard goes on without force and stays on by resting on the blade, which is
  kept within a few degrees of level from the moment the guard is let go.
- The guard's entry notch (a cut-out at the end of its open side) is not modelled.
- **Sourced sizes:** the knife's blade length, overall length and weight (maker; the maker's "Height: 0.8 in" is
  taken as the handle's thickness); its blade height, spine thickness and handle length (a retailer's listing); the
  guard's length and width (a retailer's listing, which gives them as exact); the board (a retailer's listing).
- **Estimates, not measurements:** the shape of the edge (straight for 9 cm, then rising 3.2 cm to the tip) and of
  the spine; the blade's thickness near the tip (1.2 mm); the handle's height (3 cm) and outline; where the knife
  balances (2 cm behind the heel); the guard's wall thickness (1.2 mm), slot width (3.5 mm), fold (3 mm) and weight
  (30 g); every friction coefficient.
- The blade's collider is a wedge built of boxes in five steps from the edge (0.4 mm) to the spine; the picture
  shows a smooth wedge.
- A gripper pad is rubber and meets the handle or the guard over a patch; the simulator's pads are rigid, so the
  handle is given a contact patch of 4 mm radius and the guard's faces one of 6 mm (estimates).
- Nearness to the edge is judged by distance once per control step (20 times a second), not from contact forces.

## Related tasks on the board

Nothing on the board puts a guard, sheath or cover on a kitchen knife.

- "Retract a utility knife's blade before putting the knife away" (accepted) also makes a blade safe before it is
  stored. There the blade slides into its own handle with one thumb slide; here a loose cover has to be brought
  onto a loose 20 cm blade with two hands.
- "Lock a rotary microtome's handwheel and cover its knife edge before taking the tissue block" (pending) covers
  a knife that is clamped in a machine.
- Two pending tasks work a kitchen knife's edge on a tool (a honing steel, a whetstone). They hold the knife by
  its handle too; neither brings anything onto the edge that has to stay there.

## Sources

- Victorinox, "Fibrox Pro 8" Chef's Knife", item 5.2063.20
  (victorinox.com/us/en/Products/Cutlery/Chefs-Knives/Fibrox-Pro-8-Chef%27s-Knife/p/5.2063.20): "Length of blade:
  7.9 in", "Length: 13.4 in", "Height: 0.8 in", "Weight: 6.4 oz".
- Cutlery and More, listing of the same knife, item 5.2063.20BK
  (cutleryandmore.com/products/victorinox-fibrox-chefs-knife-42192): "Blade Height: 2.1" (53mm)", "Handle Length:
  5.5" (140mm)", "Spine Thickness: 2.3mm".
- Messermeister, "8 Inch Chef's Edge-Guard", EGS-08C (messermeister.com/products/8-inch-chef-s-edge-guard): "Ideal
  for drawer storage, knife rolls, or travel, this durable guard keeps your blade—and your fingers—safe and sound.";
  "Translucent black PVC so you see the knife"; "Blade entry notch". A customer review on that page: "Easy to slip
  on and off, but yet it securely stays on."
- House of Knives, listing of the EGS-08C (houseofknives.ca/messermeister-chefs-edge-guard-8-5-egs-08c/): "The exact
  dimensions are 8.5" long by 1.9375" wide."; "an ideal fit for chef's knives or wide bladed knives 8" in length or
  less".
- Victorinox, "Blade Protection" (victorinox.com/hk/en/Products/Cutlery/Accessories/Blade-Protection/p/7.4014): "Many
  of us store our knives in kitchen drawers."; "As they knock against other utensils they can become blunt, and if
  left uncovered they may cut your hand as you rummage in the drawer."
- America's Test Kitchen, "How to Choose a Guard for Your Knife", Miye Bromberg, 15 June 2021
  (americastestkitchen.com/articles/3271-how-to-choose-a-guard-for-your-knife): "If you store your knife in a drawer
  or travel with it in a knife roll, you might want to get a knife guard."; "Look for a guard that has a length that
  is equal to or slightly longer than that of your blade."
- John Boos, R-Board WAL-R01, American black walnut, 18 x 12 x 1.5 in, as listed by Everything Kitchens
  (everythingkitchens.com/john-boos-r-board-series-18-x-12-x-1-5-cutting-board-walnut.html).
