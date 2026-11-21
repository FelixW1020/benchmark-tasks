# Send a prescription vial by hospital pneumatic tube

A hospital pharmacy's counter. On the wall behind it hangs a pneumatic tube station of the bottom-loading kind: the
lower end of its send tube, with an entry collar, is 30 cm above the counter. The station has been given its
destination and is waiting for a carrier. On the counter stand an empty carrier with its swivel lid swung open, and a
filled, capped prescription vial.

The robot has to put the vial in the carrier, swing the lid shut until its catch holds, and push the carrier up into
the send tube until the station takes it, then let go.

The table stands in for the counter. Everything is at full size.

## Why it matters

Pneumatic tubes are how much of a hospital's medication reaches its wards: one maker's system is used in over 80% of
North American hospitals, and pharmaceuticals are about 30% of what hospital tube systems carry (see Sources). The
tube moves a carrier by itself, but every carrier is still loaded, shut and sent from a station by a person. That
step is what a robot has to take over before medication transport in a hospital can run without staff, and it is a
two-handed one: the carrier must be held while its lid is swung shut, and then fed into a tube barely wider than it.

## Why it takes two hands, and what makes it hard

- **The carrier has to be held while its lid is shut.** It weighs about 290 g and stands on an end 8.6 cm across.
  The lid's catch is a snap that has to be pushed past, 22 cm above the counter. A carrier standing free rocks under
  that push and the lid springs back open (control run below).
- **The lid swings, it does not press down.** A swivel lid turns in its own plane about a pin beside the mouth. It
  has to be pushed round the pin, first sideways and then away from the robot, with the pushing finger staying on
  its rim; a straight push jams it against the pin.
- **The carrier is nearly as wide as a gripper opens.** Its body is 8.6 cm across between its two sealing rings and
  the grippers open to 9.4 cm, so a hand goes round it with 4 mm to spare on each side, between the rings.
- **The send tube is only slightly wider than the carrier.** The rings are 10.4 cm across and the tube's bore is
  10.5 cm. The entry collar widens to 13.2 cm, which is all the room there is for the arm's sag and for a carrier
  that hangs a little crooked in the hand.
- **Order matters.** A carrier that goes into the tube with its lid not secured voids the trial, whatever happens
  afterwards.

## Files

| File | What it is |
|---|---|
| `task.py` | The layout, the instruction, the record kept every control step, the scoring rule. |
| `tubekit.py` | The carrier, its lid and catch, the vial, the station's send tube and cabinet: shapes, sizes, the two rules. |
| `demo.py` | A scripted demo that reaches full score, and seven control runs. |
| `test_scoring.py` | 23 checks of the shapes, the catch, the station's rule and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 1,300 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1300 \
  --instruction "Send this prescription vial to the ward by tube: put it in the carrier, swing the carrier's lid shut until it catches, then push the carrier up into the station's send tube until the station takes it, and let go."

python demo.py task.py out_dir                 # the scripted demo: videos, setup.png, result.json
TUBE_LOT=B python demo.py task.py out_dir      # the second layout
python -m pytest -q test_scoring.py            # the offline checks
```

- `object_jitter=0` is required: the station is fixed and the layouts are exact.
- There are no seeds. `TUBE_LOT` picks the layout: A, or B (the carrier 1.5 cm further from the robot and 1.5 cm to
  the left; the vial 3 cm further and 2 cm to the right). The station does not move.
- A trial ends at full score held for 1 s, or at the run limit of 1,300 steps (65 s).
- Time per trial on one L40S: about 3 minutes for a demo run by itself (an earlier version of the demo, 863 steps,
  with start-up); about 7 minutes for the 1,300-step Dream-1 run with three demo runs sharing the GPU.
- `result.json` from the demo holds the score, the record and a trace every five steps.
- The overhead camera sees the counter, the carrier and the vial, and the top of the station's cabinet. The send
  tube's mouth is under the cabinet, out of its sight: the wrist cameras see it (the demo video and the setup picture show the left wrist camera beside the overhead one).

## Scoring

Scored at the end of the trial.

| Score | State at the end |
|---|---|
| 0 | The vial is not wholly inside the carrier. |
| 0.25 | The whole vial is inside the carrier. |
| 0.5 | And the lid is secured: swung shut to within 3 degrees, where its catch holds it. |
| 0.75 | And the carrier's top is in the station's send tube (2 cm or more above the collar's lower edge), but the station has not taken it, or a gripper is within 3 cm of it, or it is still moving. |
| 1.0 | The station has taken the carrier; both grippers are at least 3 cm from it; it is at rest (under 2 cm/s). |

The whole trial scores 0, whatever the end state, if at any time:

- the carrier's top enters the send tube (1 cm above the collar's lower edge, within the collar) with the lid not
  secured;
- the vial or the carrier falls off the counter (5 cm below its top).

"Wholly inside" means both ends of the vial are within the carrier's bore, between its closed end and the rim of its
mouth. "The station has taken the carrier" is the kit's rule (see "What the simulation simplifies"): the carrier's
top is 7 cm up the send tube, within 8 mm of the tube's axis, and the carrier is within 6 degrees of upright.

So the possible scores are 0, 0.25, 0.5, 0.75 and 1.0, and doing nothing scores 0. The score depends only on the
state at the end and on those two events; nothing is added up over time.

Ways through that were considered, and what they score:

- Loading the carrier empty, lid shut: 0 (run below). The vial is not in it.
- Loading the carrier with the lid not secured: 0 for the whole trial (run below).
- Pushing the vial up the send tube without a carrier: 0. It is not in the carrier, and nothing holds it there.
- Laying the carrier down to shut the lid: allowed; the tiers do not ask for the carrier to stand until it is loaded.

## Scoring a real trial

Tools: the trial video, a ruler, a protractor or a square-on photo of the carrier's top.

1. Vial: look through the clear carrier. The vial is wholly below the rim of the mouth.
2. Lid: it lies over the mouth and does not swing when the carrier is tilted or its edge is touched lightly (the
   catch holds); on a photo from above, the lid's edge is within 2.5 mm of where it lies when shut (3 degrees at the
   lid's far edge, 4.4 cm from its middle).
3. In the tube: the carrier's top is 2 cm or more above the collar's lower edge (a mark on the carrier 2 cm below its
   top is hidden by the collar).
4. Taken: the station holds the carrier when the hand lets go (it does not fall), and both grippers are 3 cm away.
5. The lid rule: on the video, the lid was shut before the carrier's top reached the collar.

This procedure has not been tried on a physical rig.

## The demo

`demo.py` reaches 1.0 on both layouts. It never reads a part's state: every target comes from the layout constants
and the kit's sizes, and the only feedback is each hand's own measured position and attitude, used to correct for the
arm's sag and droop under load.

| Step (layout A) | What happens |
|---|---|
| 120 | The left hand, leaning 30 degrees down, has closed on the vial's body. |
| 250 | The vial has been let go over the carrier's mouth, 12 mm to the side away from the open lid, and is in: 0.25. |
| 420 | The right hand, coming from the right and leaning 15 degrees down, has closed round the carrier's body between its rings. |
| 505 | The left hand's closed fingertip is on the lid's rim. |
| 575 | It has followed the rim round the pin; the lid is over its catch's bump and held: 0.5. |
| 730 | The carrier has been lifted and brought under the send tube, its top 2 cm below the collar. The left hand has moved aside and points its wrist camera at the tube's mouth. |
| 770 | The carrier's top is 2 cm up the send tube: 0.75. |
| 795 | Its top is 7 cm up: the station takes it. |
| 860 | The right hand has opened and is 3 cm clear: 1.0. |
| 877 | Full score has held for 1 s: the trial ends as a success. |

Layout B: 0.25 at step 250, 0.5 at 575, 0.75 at 770, 1.0 at 860, success at 877. Steps are read from a
trace kept every five steps. While it was carried the carrier stayed within 1.5 degrees of upright, and its top was
4 mm or less from the tube's axis as it went in (layout A).

Three things the demo's moves are chosen for, all found in the simulator:

- A straight push on the lid's edge jams it: the rim slides along the finger and that friction turns the lid back
  against its pin. The fingertip has to follow the rim round the pin.
- The right hand comes at the carrier from the right, turned 70 degrees from straight ahead, so that the left arm
  can reach the lid across the front of the carrier (the arms pass 6.5 cm apart by the offline check; with the right
  hand straight on they came within 2 mm).
- Under the carrier's weight the right hand sags by about 2 cm and droops by 3 to 5 degrees, which would put the
  carrier's top about 2 cm off the tube's axis (a run without the correction jammed on the collar). The demo corrects
  both from the hand's own measured pose.

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still` | 0.0 | Nothing moves (200 steps). |
| `--mode vial` (the vial is put in; nothing else) | 0.25 | The vial is in the carrier from step 250; the lid is still open. |
| `--mode lid` (vial in, carrier held, lid shut; the carrier is left on the counter) | 0.5 | The lid is secured from step 575; the carrier stands on the counter. |
| `--mode held` (everything, but the hand keeps hold of the loaded carrier) | 0.75 | The station has the carrier (top 7.5 cm up the tube), but the gripper is still round it. |
| `--mode open` (the lid is swung only to 12 degrees from shut, short of its catch, and the carrier is loaded) | 0.0 | The carrier's top entered the send tube with the lid 13 deg open: the trial is void from step 764. The station took the carrier all the same. |
| `--mode empty` (lid shut and carrier loaded, without the vial) | 0.0 | The station takes the carrier, lid shut, but the vial is still on the counter. |
| `--mode loose` (vial in; the lid is pushed without the other hand on the carrier) | 0.25 | The carrier rocks under the push (up to 3.2 degrees) and the lid stops 7 degrees from shut, short of the top of its catch's bump, and springs back to 10. |

One more run, made with the lid's catch set half as stiff (a bump of 0.03 N m): `--mode loose` then scored 0.5, the lid shut on a carrier standing free. See "What the simulation simplifies".

## Dream-1

One run, 1,300 steps, layout A, after a 200-step test (also 0.0): **0.0**. The left gripper went to the vial and stayed beside it for the first half of the run without lifting it, while the right gripper came up beside the carrier; both then drew back toward their bases. At the end the vial stood where it started, the lid was still open and the carrier had not been moved.

## Building it for real

- **Carrier.** A short swivel-lid carrier for a 110 mm pneumatic tube line, the size NHS hospital systems use
  (a UK maker lists it as "approx ext. 104mmD x 228mmL"). Stand it on its closed end with its lid swung open.
- **Station.** A bottom-loading station for a 110 mm line, mounted so that the lower edge of its entry is 1.2 m
  above the floor (the height the NHS design guide gives), over a counter 0.9 m high. A working tube system is
  costly and fixed to a building, so a low-cost bench stand-in is specified under **Building a bench station**
  below.
- **Vial.** A 13 dram amber prescription vial with a child-resistant cap, with thirty tablets in it.
- **Check before a trial:** the carrier's lid swings freely and snaps shut; the station accepts and holds a carrier
  pushed up into it by hand.
- **Safety.** Light plastic parts and forces of a few newtons. Run the station with its blower off, or with a
  destination that returns the carrier to the same station.

Measure the real carrier and station first. Apart from the sizes listed under Sources, every size, mass and force
here is an estimate (next section).

## Building a bench station

The task needs only the part of a station that a person loads: the mouth of its send tube, and something that
takes the carrier and holds it. This bench rig is a length of the line's own tube on a board, with a sprung catch.
**It is a design drawn from the simulated station's numbers; it has not been built.** The checks at the end are part
of the build.

| No. | Part | Size | Notes |
|---|---|---|---|
| 1 | Send tube | 300 mm of 110 mm pneumatic-tube pipe (the makers of the carriers sell it by the length), or any rigid tube with a bore of 105 to 106 mm | A carrier must slide through it freely. Ordinary 110 mm drain pipe has a thicker wall and may be too tight: try a carrier in it before buying |
| 2 | Entry collar | A funnel-shaped ring 20 mm tall, opening from the tube's bore to 132 mm across at its lower edge | Cut from a plastic funnel or printed, taped or glued to the tube's lower end |
| 3 | Wall board | 18 mm plywood, 600 mm wide and 660 mm tall, standing upright at the back of the table | The simulated wall |
| 4 | Brackets, two | Stand-off pipe clips or shelf brackets with band clamps | Hold the tube upright with its axis 140 mm in front of the board and the collar's lower edge 300 mm above the table |
| 5 | Catch, two fingers | Strips of spring steel or stiff plastic sheet, about 15 mm wide and 40 mm long, riveted inside the tube opposite each other, pointing up and inward | Their free tips stand 9 mm in from the wall, 20 mm above the collar's lower edge. A carrier's upper sealing ring pushes them aside going up and they spring back under it |
| 6 | Cover (optional) | A box 215 x 220 x 260 mm round the tube, its underside 40 mm above the collar's lower edge | Makes the rig look like the simulated station's cabinet |

Checks:

- **Free fit.** A carrier with its lid shut slides up the tube by hand without binding.
- **Take-up.** Pushed up until its top is 70 mm above the collar's lower edge, the carrier is held by the catch and
  stays when let go, loaded with the vial (about 320 g in all, an estimate). Bend the fingers until it does.
- **Height.** The collar's lower edge is 300 mm above the table and the tube's axis is where the setup gives it.
- **Reset.** Push the carrier on up and take it out of the tube's open top.

The carrier itself and the prescription vial are stock items (see Sources).

## What the simulation simplifies

- **Why a UK carrier and station.** The carriers used in most US hospitals are for 4 in and 6 in lines, open along
  their side and close with latches. No maker's page that could be read gives their body size, weight or latch, and
  a 4 in carrier is wider than these grippers open. The NHS-standard 110 mm short carrier has published sizes and a
  body a gripper can close round, so it is used here.
- **The station is mostly estimates.** From the design guide: that bottom-loading stations exist, the 1.2 m mounting
  height, the 110 mm line. Estimates: the entry collar's shape (2 cm tall, widening to 13.2 cm), the send tube's bore
  (105.4 mm, from a 2.3 mm wall), the cabinet's size and how far the tube hangs below it, the display.
- **The station taking the carrier is a rule, not contact physics.** A real station holds a carrier that is pushed
  up into its send tube with a catch. Here the kit holds the carrier (at the tube's axis, upright, its weight carried)
  once its top is 7 cm up the tube, within 8 mm of the axis and within 6 degrees of upright. The 7 cm is an estimate.
  The station does not check the lid; the scoring does.
- **The lid's catch is a torque law on its pin,** a snap: a bump that resists shutting, beginning 10 degrees from
  shut and reaching 0.06 N m at 5 degrees, then 0.06 N m holding the lid shut within 3 degrees. Further out the lid
  stays where it is left (0.012 N m of friction). All of it is an estimate; the bump is taken to be as high as the
  hold. It matters: with a bump half as high (0.03 N m) the lid could be shut on a carrier standing free (that
  control scored 0.5), so "the carrier has to be held" rests on this estimate. The lid and the carrier's rim do not
  collide with each other: the pin holds the lid at the rim's height.
- **Carrier sizes that are estimates:** wall 3 mm (so the body is 86 mm across), the rings' places and width, the
  closed end, the lid's size (88 mm across, 12 mm thick), the pin's place (47 mm from the axis, on a lug), the lid's
  80 degree stop, and the weights (260 g and 30 g).
- **The sealing rings are rigid** and 101 mm across: a real ring is 104 mm of hook-and-loop pile that gives a little
  in the tube.
- **The vial is one rigid piece** (no loose tablets), and there is no bag or padding round it, which a hospital's
  procedure would usually ask for.
- **Everything is rigid convex pieces**; round parts are 16-sided.
- A gripper pad is rubber and meets the carrier and the vial over a small patch (4 mm).

## Related tasks on the board

- "Count ten tablets out of a stock bottle on a pill counting tray" (accepted) is the pharmacy step before this one.
  It counts with a spatula; nothing is put in a container and sent.
- "Hand a drink out a self-closing drive-thru window" (accepted) passes something through an opening that shuts.
  Here a loaded container is pushed up into a tube that holds it.
- "Put a baked wafer from the hotplate into a twist-lock single-wafer carrier" (accepted) loads a carrier and locks
  its lid by twisting. This carrier's lid swings round a pin, and the carrier itself then has to be loaded into a
  station.

No accepted or pending task handles a pneumatic tube carrier or a tube station.

## Sources

- How widely hospitals use tube systems: Health-ISAC bulletin on the Translogic pneumatic tube system (2 August 2021):
  "This Translogic PTS system is used in over 80% of hospitals in North America"; American Hospital Association,
  Fast Facts on U.S. Hospitals, 2026: "There are 6,100 hospitals in the United States."
- What the tubes carry: MIT Technology Review, an article on hospital tube systems (19 June 2024): "Today, lab specimens are
  roughly 60% of what hospital tube systems carry"; "pharmaceuticals account for 30%, and blood products for phlebotomy
  make up 5%."; of one university hospital's system: "completes more than 6,000 transactions on an average day."
- NHS National Services Scotland, Scottish Health Technical Memorandum 08-04, "Pneumatic tube transport systems",
  Part B (2015): the carrier table's row "110 Short Swivel top 110 800 80 228" (nominal tube 110 mm, bend radius
  800 mm, internal diameter 80 mm, length 228 mm); "network tubing: 160mm or 110mm diameter"; 3.13: "The carrier lids
  should be of the swivel pattern designed so that the lid is positively secured before loading the carrier into the
  system (an insecure lid may be dislodged by the system)."; 3.25: "For bottom loading stations the mounting height
  from the bottom of the station should be 1.2m above finished floor level."
- A UK maker's listing of the carrier (quirepace.co.uk/product/110mm-short-carrier-blue/): "Short flip-top carrier
  for 110mm Pneumatic Tube Systems, with dimensions approx ext. 104mmD x 228mmL". Its hospitals page: "used
  extensively in over 100 NHS Hospitals to transport Samples, Drugs, Prescriptions"; "the primary applications for a
  Pneumatic Tube System are for transporting Laboratory Specimens and Pharmaceutical Products".
- A Canadian health authority's procedure (Alberta Health Services, REG01-110 Attachment C): "Key in the destination
  code and press SEND." before "Place carrier into dispatcher"; "Close carrier and secure latches"; "Never force a
  carrier closed." That order is why this scene starts with the station already waiting for a carrier.
- How much such systems are used: a US university hospital's facilities report
  (annualreport.fm.virginia.edu/2022/innovative-outcomes/tube-system-touch-up): "The system supports approximately
  4,000 transactions a day (a transaction is each time a carrier travels through the system)."; a trade article by a
  maker (hconews.com, 23 September 2020): "one of the 2,400 hospitals equipped with" its system; "Two out of every
  ten carriers contain medication intended for patient care areas." Neither says who loads the stations or how many
  workers do.
- Vial: a distributor's listing of a 13 dram amber vial with child-resistant cap (calpaclab.com, PSA-AVCR13):
  "Height (without cap): 65 mm", "Height (with cap): 69 mm", "Width: 32 mm".
- A maker's table of tube sizes (air-log.com, flip-cap carriers): "124 x 2,3 mm" for its 124 mm tube; the 2.3 mm wall
  is used here for the 110 mm tube as an estimate.
