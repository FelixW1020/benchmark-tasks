# Guard two bare rebar stubs with impalement-rated caps

The top of a footing on a building site. Three reinforcing-bar stubs stand in a row, 14 cm tall. One already wears an
impalement-rated safety cap. The other two are bare. On the table in front of them lie two more rated caps, on their
sides as they came out of the box, and three plain plastic mushroom caps. The instruction is only "The crew is about
to start work around this footing. Make the rebar safe."

The table stands in for the footing; a strip of plywood holds the bars.

## What the task asks of a robot

- **A choice the instruction does not give.** Bare bar ends are an impalement hazard, and the rule is that they be
  guarded. The mushroom caps are small, light and easy to pick up, and they are not a guard: an OSHA memorandum on
  them says "Therefore, the mushroom caps do not meet the "guarded" requirement of 1926.701(b)." A bar under a
  mushroom cap scores nothing (the `mushroom` control run).
- **A part wider than the hand opens.** A rated cap's top is 4 in (10.2 cm) square; the gripper opens 9.4 cm. It
  cannot be taken from above. It can be taken by its neck, 3.6 cm across, and only near the mouth: a finger is 6.4 cm
  wide 5 cm back from where it grips, and the top overhangs the neck by 3.3 cm all round.
- **A turn in the air.** A cap lying on its side rests on an edge of its top and the rim of its mouth, its neck 30
  degrees from the table. The demo takes the neck from above with the hand leaning 25 degrees, lifts, and leans
  forward to 85 degrees: the cap swings mouth-down under a level hand, its top above the fingers. The wrist then
  carries 227 g on a 4.3 cm lever and droops; the demo reads its own hand's lean and corrects it.
- **Onto the bar, and home.** A 12.7 mm bar goes into a 30 mm bore, where four sprung fins centre it. The fins have
  to be spread by the bar's end, which takes more than the cap's weight: set on the bar and let go, the cap is left
  perched on its fins, 22 degrees off plumb (the `perch` control run). A maker's instructions: "Push firmly until the [cap's] internal
  metal insert comes into contact with the end of the protruding rebar." Only a cap whose insert is on the bar's end
  counts as guarding it.
- **A way out.** The fingers are under the cap's top. The hand has to let go, draw 9.5 cm straight back, and only
  then lift.
- **Leaving what is already right.** The bar that is capped at the start has to stay capped. If its cap comes off,
  the trial scores 0.

One arm does it. The left arm is not used.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `rebarkit.py` | The rated cap with its four sprung fins and steel insert, the mushroom cap, the bars in their board. |
| `demo.py` | A scripted demo that reaches full score, and three control runs. |
| `test_scoring.py` | 60 checks of the shapes, the layouts, the scoring rule and the demo's plan; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,500 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1500 \
  --instruction "The crew is about to start work around this footing. Make the rebar safe."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
REBAR_LOT=B python demo.py task.py out_dir     # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: each cap's four fins are placed to match the cap exactly, and one cap starts on its
  bar.
- There are no seeds. `REBAR_LOT` picks the layout: A (the bar on the robot's left is capped; the loose caps lie
  on the right and in the middle) or B (the bar on the right is capped; the loose caps lie in the middle and on the
  left; the mushroom caps are on the other side).
- A trial is at most 1,500 steps (75 s). It ends early once full score has held for 1 s. The demo takes 1,049
  steps (52 s); a simulated trial of that length took 7 to 10 minutes of wall time on a GPU shared with other
  jobs, part of it start-up.
- `result.json` from the demo holds the score, the task's own report at the start and end, and a trace every 5
  steps.

## Scoring

Scored at the end of the trial.

A bar is **guarded** when a rated cap is seated on it, both grippers are at least 3 cm from that cap, and the cap is
at rest (under 2 cm/s and 0.5 rad/s). **Seated:** the bar's end is within 2 mm of the cap's steel insert, and the
cap's axis is within 5 degrees of the bar's.

| Score | What it takes |
|---|---|
| 0 | The bar that was capped at the start is no longer guarded. Or: neither of the other two bars has a rated cap on it. |
| 0.25 | One of the two bars has a rated cap on it that is not seated. |
| 0.5 | One of the two bars is guarded. Or: both have a rated cap on them, neither seated. |
| 0.75 | One is guarded and the other has a rated cap on it that is not seated. |
| 1.0 | Both are guarded. |

In numbers: each of the two bars that were bare adds 0.5 if it is guarded, and 0.25 if a rated cap is **on it but not
seated**: the bar's end at least 1 cm up the cap's bore, the cap within 30 degrees of upright, let go and at rest. A
bar left bare adds nothing. A bar wearing a mushroom cap adds nothing. A cap that is still held, or still moving,
adds nothing.

Other ways to try it, and what they score: mushroom caps on both bars, 0; a rated cap laid on a bar upside down, 0,
because the bar is not in its bore; moving the cap from the capped bar to a bare one, 0, because the bar it came from
is no longer guarded.

## Scoring a real trial

- **Which caps:** from a photograph at the end. Rated caps have the 4 in square top; mushroom caps are round and
  5.7 cm across.
- **Seated:** the bars' ends are 16.0 cm above the table, and a seated cap's top is 17.8 cm above it (the insert is
  1.8 cm under the top). With a ruler: no more than 18.0 cm. On another make of cap, seat one by hand first and
  measure it.
- **Within 5 degrees:** measure the highest and the lowest edge of the cap's top above the table. 5 degrees is a
  difference of 9 mm across the 10.2 cm top.
- **On but not seated:** the top between 18.0 and 22.0 cm above the table, the cap no more than 30 degrees from
  upright.
- **Let go and at rest:** from the last second of the video.

The task has not been run on a physical rig. Real bar ends are sharp: for trials with people nearby, grind the
ends smooth or use lengths of 1/2 in steel rod.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `rebarkit.py`, and the right hand's own measured
position and lean (the robot's proprioception): to correct for the arm's sag, for the droop of its wrist under the
cap, and to push by a set amount beyond the point where it carries no load. It never reads where a part is.

For each of the two bare bars:

1. The right hand comes down its own line, leaning 25 degrees forward, onto the neck of a lying cap, 12 mm from the
   mouth, pads 6.2 cm apart, and closes on the 3.6 cm neck.
2. It lifts the cap 5 cm, then rises to 20 cm while leaning forward to 85 degrees. The cap now hangs mouth-down, its
   top over the fingers.
3. It carries the cap to a point in front of its bar, then over the bar, the mouth 1.5 cm above the bar's end, and
   lowers it. When the bar's end is about to meet the fins it stops correcting its height, sends its command 1 cm
   below the point where it would carry no load, and so pushes the cap home.
4. It eases off, opens, draws 9.5 cm straight back, and lifts.

| | Layout A | Layout B |
|---|---|---|
| Fingers closed on the first cap's neck | step 140 | step 140 |
| First cap mouth-down, within 8 degrees of plumb | step 225 | step 225 |
| The bar's end enters its mouth | step 365 | step 365 |
| Fins spread 2 mm | step 405 | step 410 |
| Bar's end within 2 mm of the insert | step 425 | step 425 |
| Let go and clear: first bar guarded (score 0.5) | step 495 | step 500 |
| Fingers closed on the second cap's neck | step 670 | step 670 |
| Bar's end within 2 mm of the insert | step 960 | step 960 |
| Let go and clear: second bar guarded (score 1.0) | step 1,030 | step 1,030 |
| Trial ends with full score | step 1,049 | step 1,049 |
| Bars' ends from the inserts at the end | 0.0 and 0.1 mm | 0.1 and 0.1 mm |
| Caps off plumb at the end | 0.0 and 0.7 degrees | 0.9 and 0.7 degrees |
| Caps off plumb in the hand, before they go onto their bars | 5.4 and 4.1 degrees | 4.4 and 4.7 degrees |

(Steps are read from a trace written every 5 steps.)

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves. |
| `--mode perch`: the first cap set on its bar with the bar's end 1.7 cm up the bore, and let go without a push; the second cap pushed home | 0.75 | Let go, the first cap drops onto its fins and tips: it ends hanging on the bar 22 degrees off plumb, the bar's end 2.5 cm up its bore, two fins pushed out 1.6 mm. On the bar, not seated: 0.25. The second cap is seated: 0.5. |
| `--mode mushroom`: the mushroom cap that lies on its side put on the first bar in the same way; a rated cap pushed home on the second | 0.5 | The mushroom cap goes onto the first bar (the bar's end is in its stem from step 365) and ends hanging on it, 44 degrees off plumb. A bar under a mushroom cap adds nothing. The second bar is guarded: 0.5. The rated cap meant for the first bar is left lying where it was. |

## Dream-1

One run, 1,500 steps, layout A (after a 200-step test): **0.0**. In the first 20 seconds the left gripper went to
the cap on the bar that was already capped and stayed beside it. From about 25 seconds the right arm reached out
over the loose caps to the bare bars and moved its gripper among them and above the caps for the rest of the
trial, without taking hold of a cap. At the end the two rated caps and the three mushroom caps lay where they had
started, and both bars were bare.

## Building it for real

- Three 16 cm lengths of #4 (1/2 in) reinforcing bar, their top ends ground smooth, set upright 13 cm apart in
  1/2 in holes in a strip of 3/4 in plywood 10 x 60 cm, standing 14.1 cm out of it. Fix the strip to the table.
- Three impalement-rated rebar safety caps of the 4 x 4 in kind for #3 to #8 bar, with a steel insert (the model
  here is the size of a cap listed at 4 x 4 x 2.75 in). Seat one on a bar by hand. Lay two on their sides on their
  marks, mouths toward the robot.
- Three plain plastic mushroom caps for #3 to #8 bar: two standing on their heads, one on its side.
- Tape marks for the loose caps in the two layouts.

Resetting: pull the caps off the two bars and lay everything back on its marks.

## What the simulation simplifies

- Everything is rigid.
- **Sourced:** the rated cap's 4 x 4 in top and 2.75 in height (a retailer's listing of a cap with these approval
  numbers); that it has a steel insert which seats it square on the bar, and fins that centre it (its maker); that
  it is pushed on until the insert meets the bar's end (another maker's instructions); 0.5 lb, the weight that
  second maker lists for its own 4 x 4 in cap for #3 to #8 bar; the mushroom cap's 0.02 lb (the same maker); the
  #4 bar's 1/2 in.
- **Estimates, not measurements:** how the rated cap's height divides into top (1.3 cm) and neck (5.7 cm), the
  neck's width (3.6 cm) and wall (3 mm), both read off the maker's photograph; the insert's depth in the bore
  (5.2 cm); where the cap's weight sits (5.5 cm above the mouth); the fins' shape, their springs (0.70 N at rest,
  100 N/m, so 0.94 N each on a #4 bar) and friction (0.3 for a fin, 0.5 elsewhere); every size of the mushroom cap
  (head 5.7 cm, stem 2.5 cm across and 4.5 cm long). No maker publishes the force needed to push a cap on, and
  none was measured: here the fins take about 2.6 N to spread, against the cap's weight of 2.2 N.
- The rated cap is a square top on a neck of 12 flat staves. The real one is a single moulding, its top ribbed and
  its edges rounded.
- Each fin is a rigid rounded bar on a sprung slide along a radius of the bore, with a 13 degree lead-in. Real
  fins are thin plastic vanes that bend. Each weighs 2 g (heavier than a vane, so that the contact solver holds
  it); the cap is lighter by the same amount.
- Once the fins are spread the cap slides the rest of the way down under its own weight and the hand's push: the
  fins' friction on the bar (about 1.5 N) is less than the cap's weight. On a real cap the fins may hold harder.
- The mushroom cap's fins are fixed ribs: it drops onto a #4 bar and sits on the bar's end.
- The bars are smooth 12-sided prisms with flat ends. Real bar is ribbed, and its sheared end is rough.
- The bars stand in plywood, not in concrete, and are 14 cm tall; stubs on a site are often longer.
- A gripper pad is rubber and meets the neck over a patch; the simulator's pads are rigid, so the neck is given a
  contact patch of 9 mm radius (an estimate), which is what lets a pinch near the mouth carry the top-heavy cap
  through its turn. The cap still turned about 5 degrees in the fingers.

## Related tasks on the board

No accepted or pending task has rebar or an impalement cap.

- Two earlier tasks about capping rebar were not accepted. "Cap upright rebar with impalement-rated caps, not
  mushroom caps" used one-third-scale bars and loose sleeves. "Cap the bare rebar stubs on a footing form with
  impalement-rated steel-plate caps, leaving the capped ones as they are" added a grip adapter to its caps and gave
  full credit to a cap 4 mm short of its stop. Here the bars and caps are full size, a cap is taken by its own neck
  with nothing added, its fins are sprung parts that the bar has to spread, and a cap counts only when the bar's end
  is within 2 mm of its insert.
- "Retract a utility knife's blade before putting the knife away" (accepted) also makes a hazard safe before
  others meet it. Its safe state is one thumb slide away; here the guard is a separate part that has to be chosen,
  carried and pushed home.
- "Clear away a used procedure tray that still has a needle in it" (accepted) has the same pattern of an ordinary
  request with an unstated hazard. There the hazard has to be noticed and put in the right bin; here the wrong
  guard lies ready beside the right one.
- "Snap a sip lid onto a full 16 oz paper hot cup all round the rim, without tipping or squeezing the cup"
  (accepted) is also a push-on closure. Its lid is held from above; this cap is wider than the gripper and has to
  be taken from under its own top and turned over in the air.

## Sources

- 29 CFR 1926.701(b), "Reinforcing steel" (osha.gov/laws-regs/regulations/standardnumber/1926/1926.701): "All
  protruding reinforcing steel, onto and into which employees could fall, shall be guarded to eliminate the hazard
  of impalement."
- OSHA, memorandum "Mushroom Style Plastic Rebar Covers Used For Impalement Protection", 15 January and 29 May 1997
  (osha.gov/laws-regs/standardinterpretations/1997-05-29-0): "Therefore, the mushroom caps do not meet the
  "guarded" requirement of 1926.701(b)."
- OSHA, letter "Protection of impalement hazards: rebar and other hazards", 3 August 1999
  (osha.gov/laws-regs/standardinterpretations/1999-08-03-2): "protective devices (covers or wooden troughs) for
  rebar that are capable of withstanding at least 250 pounds dropped from a height of ten feet".
- California Code of Regulations, Title 8, section 1712 (dir.ca.gov/title8/1712.html): "Protective covers shall
  have a minimum 4-inch by 4-inch square surface area".
- Deslauriers, DISC-10 impalement safety cover (deslinc.com/disc-10-impalement-safety-cover): "Fits rebar #3-#8";
  "Self-centering formed steel insert seats the cover perpendicular to rebar"; "Strategically designed fins help
  ensure a secure, centered fit"; "16 sq. in. top surface aids in worker protection"; approved by California OSHA
  #C1718AG. Its photograph is what the neck's proportions are read from.
- Sandbaggy, listing of an OSHA rebar safety cap with the same approval number
  (sandbaggy.com/products/osha-rebar-safety-caps-impalement-protection): "Dimensions (L x W x H): 4” x 4”x 2.75”";
  "Rebar size: #3-#8".
- OCM, OSHA Rebar Cap REBARCAP-38OSHA (ocm-inc.com/?p=3112): "Push firmly until the OCM OSHA Rebar Cap’s internal
  metal insert comes into contact with the end of the protruding rebar."; "Please check that the cap is secured
  properly on the rebar such that the top surface of the cap is set in a perpendicular manner to the vertical
  exposed rebar end."; "Weight Per Piece (lbs) 0.5"; size #3 to #8.
- OCM, Rebar Mushroom Cap RC-MUSH (ocm-inc.com/shop/rebar-tools-and-accessories/rebar-mushroom-cap/rc-mush/):
  "Weight Per Piece (lbs) 0.02"; size #3 to #8.
