# Put a table fan back together after washing

A 12 inch table fan lies on its back on a folded towel, unplugged, its rear grille still on and its motor shaft
pointing up. The three parts that come off for washing lie dry beside it: the blade on a tea towel, the blade cap on
the table, the front grille on a dish rack. The instruction is only "Put the fan back together."

## What the task asks of a robot

Four fits, in the order the maker's instructions give them. Each is made by colliding shapes; no rule places a part
or shuts a clip.

- **The blade onto the shaft's pin.** The shaft has a cross pin and the back of the blade's hub has a slot. The hub
  comes down onto the pin only when the slot lies along it. Put on 30 degrees out, it rides on top of the pin 7 mm
  high and covers the thread, so the cap has nothing to catch. That is the `offpin` control run: the cap never
  caught, and the run scored 0.25 only because the turning cap dragged the hub round until it dropped.
- **The cap, against a shaft that turns.** The thread is left-handed ("Tighten=Counter-clockwise"). A switched-off
  motor's shaft turns freely (0.004 N m here), and thread friction carries the shaft and the blade round with the
  cap. The blade has to be stopped while the cap is turned. Without a stop the cap never comes tight. That is the
  `nohold` control run, and it scores 0.25.
- **The front grille into the support.** The rear grille has a hook at its bottom whose lip stands 4 mm over a seated
  rim and reaches 5 mm in. The front grille's bottom edge has to come down between that lip and the blades' tips
  (a 5.5 mm window), slide in under the lip, and the rest of the rim be let down.
- **Three clips.** Each is a hooked lever on a hinge with an over-centre spring. It stands open until it is pushed in
  past half way and then shuts over the front grille's rim. Pushed straight in low down, a clip jams against a rubber
  fingertip; it has to be rolled in and up. A grille laid on and left unclipped is the `noclips` control run. It
  scored 0.5: the grille lay flat with its edge under the lip but 5.0 mm off centre, outside the 3 mm the rule
  allows. In the demo it is the clips that bring it to the middle.

The parts are large for the arms (the grille is 35 cm across, the gripper opens 9.4 cm), so each is taken where a
gripper can hold it: the blade by one blade, the cap by its flats, the grille by its rim.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the instruction and the scoring. |
| `fankit.py` | The fan: housing, rear grille, support and clips; the shaft and pin on its bearing; the blade, the cap and its thread, the front grille. |
| `demo.py` | A scripted demo that reaches full score, and four control runs. |
| `test_scoring.py` | 32 checks of the shapes and fits, the thread, the clips, the layouts, the demo's plan and the scoring rule; no simulator needed. |
| `task.json` | The task's description for the board. |

## Running and scoring a trial

```bash
# a policy (Dream-1 here), 3,000 steps at 20 Hz; the score is printed at the end and written to the log
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 3000 \
  --instruction "Put the fan back together."

python demo.py task.py out_dir --max-steps 3000      # the scripted demo: videos, setup.png, result.json
FAN_LOT=B python demo.py task.py out_dir --max-steps 3000    # the second layout
python demo.py task.py out_dir --mode offpin --upto cap      # a control run (offpin, nohold, noclips, still)
python -m pytest -q test_scoring.py                  # the offline checks
```

- `object_jitter=0` is required: the shaft, the clips and the fan are placed to match each other exactly.
- There are no seeds. `FAN_LOT` picks the layout, A or B (see `task.json` for both). They differ in where the fan
  and each part lie, how far the blade and the grille are turned, and which way the shaft's pin points.
- A trial is at most 3,000 steps (150 s). It ends early once full score has held for 1 s. The demo reaches full
  score at step 2,335 and its trial ends at step 2,354 (118 s); a simulated trial of that length took 9 to 15
  minutes of wall time on a GPU shared with other jobs.
- `result.json` from the demo holds the score, the task's own report at the start and end, and a trace every 5 steps.

## Scoring

Scored at the end of the trial; the highest row that holds.

| Score | What it takes |
|---|---|
| 0 | None of the rows below. Nothing done; or the blade not down on the pin, whatever else was done. |
| 0.25 | The blade is on the shaft with the pin at the bottom of the slot in its hub: the hub within 1.5 mm of the shaft's axis and 3 degrees of square, its back face within 1 mm of its seat. (Riding on top of the pin it stands 7 mm high.) |
| 0.5 | Also the cap is on the thread and tight: turned anticlockwise until the thread draws its face against the hub's, within 0.8 mm. |
| 0.75 | Also the front grille lies on the rear grille's rim: centred within 3 mm, no point of its rim more than 2 mm up, turned within 6 degrees of the way it goes, its bottom edge under the support's lip. |
| 1.0 | Also all three clips are shut over its rim (within 6 degrees of fully in), both grippers are at least 3 cm from every part, and everything is at rest: in the last half second no part or clip has moved more than 2 mm or turned more than 3 degrees. |

Rest is judged from where the parts are, not from the simulator's reported speeds: small sprung parts held against a
stop report speeds that chatter while the parts stay put.

## Scoring a real trial

- **Blade on the pin:** look along the shaft's pin from the side: the pin is inside the slot and the hub's back face is
  down on its seat. A hub riding on the pin stands 7 mm high, which a ruler against the hub shows.
- **Cap tight:** no gap between cap and hub (a 0.8 mm feeler does not enter), and the cap does not turn anticlockwise
  under a light fingertip.
- **Grille down:** a ruler across the two rims shows no step of more than 2 mm anywhere; the rims' edges are within
  3 mm of each other all round; a photograph of the support shows the front rim under its lip.
- **Clips:** each lies against the front rim (a photograph from the side; within 6 degrees of fully in).
- **Let go and at rest:** from the last second of the video.

The task has not been run on a physical rig.

## The demo

`demo.py` uses only the layout constants in `task.py`, the sizes in `fankit.py` and the hands' own measured positions
(to correct for arm sag). It never reads where a part is.

1. The right hand takes the blade by one of its three blades (pads on its two faces, hand coming down the blade's
   slope), lifts it, carries it over the shaft turned so the hub's slot lies along the pin, lowers the hub onto the
   shaft and the pin, lets go, backs off up the slope and rises clear.
2. The left hand puts a closed fingertip down between two blades, just ahead of one of them the way the cap will be
   turned. The right hand fetches the cap by its flats, sets its bore over the shaft's end and turns it anticlockwise
   in three half-turn bites. The shaft and blade go round with the cap until a blade meets the left fingertip; from
   there the cap turns on the shaft and comes tight.
3. The left hand takes the front grille from the rack by its rim (fingers above and below it), carries it over the
   fan's base, tips its bottom edge down, brings it in 7 mm short of the support (outside the blades' tips), lowers
   that edge onto the rear rim and slides it in under the lip. The left hand's lower finger is then under the rim.
   The right hand's fingertips come down on the grille's band on the far side and steady it while the left hand
   opens and draws its finger out; the rim drops the last 16 mm onto the rear grille's.
4. The left hand shuts the left clip and the right hand the right clip, each with the side of its closed fingertips:
   in onto the open clip's outer end, then in and up along the arc that end follows until the clip goes over centre
   and shuts, then its stem pressed home. The right hand goes round the outside of the rim, above the open clip, to
   the clip nearest the robot and shuts it the same way. Both hands go back to the ready pose.

| | Layout A | Layout B |
|---|---|---|
| Blade off its towel | step 135 | step 135 |
| Blade down on the pin (score 0.25) | step 425 | step 355 |
| Shaft turned by the hub as it went on | 20.5 degrees | 1.2 degrees |
| Left fingertip down between two blades | step 585 | step 585 |
| Cap off the table | step 615 | step 615 |
| Thread caught | step 790 | step 800 |
| Shaft carried round before a blade met the fingertip | 12.5 degrees | 32 degrees |
| Cap tight (score 0.5) | step 1,040 | step 1,035 |
| Cap turned on the shaft, in all | 1.46 turns | 1.39 turns |
| Grille off the rack | step 1,335 | step 1,335 |
| Its bottom edge under the support's lip | step 1,655 | step 1,655 |
| Its rim flat on the rear grille's | step 1,785, 5.0 mm off centre | step 1,785, 3.2 mm off centre |
| Left and right clips shut, grille within 3 mm (score 0.75) | step 2,020 | step 2,020 |
| Near clip shut | step 2,300 | step 2,295 |
| Hands clear, everything at rest (score 1.0) | step 2,335 | step 2,335 |
| Trial ends with full score | step 2,354 | step 2,354 |
| Grille off centre at the end | 0.8 mm | 1.4 mm |

The grille does not count as down until it is within 3 mm of the middle. The demo lays it 3 to 5 mm off, and the
first two clips push it in; that is why 0.75 comes at step 2,020 and not at step 1,785.

(Steps are read from a trace written every 5 steps.)

## Control runs

All on layout A.

| Run | Score | What happened |
|---|---|---|
| `--mode still`, 300 steps | 0.0 | Nothing moves. |
| `--mode offpin --upto cap`: the blade put on 30 degrees out of line with the pin, no fingertip stop, then the cap | 0.25 | The hub rode on top of the pin, 7.0 mm high, from step 400 to step 900. The cap, set on the shaft's end, found no thread. The turning cap dragged the hub and the shaft round until the slot met the pin and the hub dropped (step 905, 0.25). The cap was left loose on the shaft's end, tilted 12 degrees, 4 mm above the hub. |
| `--mode nohold --upto cap`: the demo's blade and cap steps without the left fingertip | 0.25 | Blade on the pin (step 425), thread caught (step 790). The shaft and the blade went round with the cap, 306 degrees in all. Of the hand's 1.5 turns the cap turned 0.65 on the shaft and stopped 1.3 mm above the hub. Not tight. |
| `--mode noclips`: the demo up to the grille, clips left open | 0.5 | Blade on, cap tight, grille flat on the rim with its bottom edge under the lip, but 5.0 mm off centre: outside the 3 mm rule, so the grille row is not met. |

The 0.75 row is reached inside the demo itself, from step 2,020 to step 2,335, with two clips shut and the third open.

## Dream-1

One run, 3,000 steps, layout A (after a 200-step test, which also scored 0.0): **0.0**. In the first 15 seconds both
arms drew back toward their bases. The left arm stayed there for the whole trial, its gripper facing the front
grille on its rack. From about 75 seconds the right arm reached out and held its gripper over the fan's shaft and
rear grille; in the last seconds it moved off toward the cap. No part was picked up: the blade, the cap and the
front grille lay where they started.

## Building it for real

- A 12 inch oscillating table fan of the common pattern: rear grille held by a locknut, blade keyed by a cross pin,
  a left-hand-thread blade cap, a front grille that hooks into a support at the bottom of the rear grille, and grille
  clips on the rear grille. The Comfort Zone CZ121 is one; its manual gives these steps. Check that the fan used has
  three clips and one support, or adapt the count.
- Unplug it and tape the plug to the base. Lay it on its back on a folded towel, base away from the robot, and keep
  it from sliding (a strap over the base to the table).
- A folded tea towel for the blade, a dish rack for the front grille, tape marks for all four places and for the way
  the pin points.

Resetting: open the clips, lift the grille off, unscrew the cap clockwise, lift the blade off, put the parts back on
their marks and turn the shaft until its pin points along its mark.

Real-world trials are safe as long as the fan stays unplugged.

## What the simulation simplifies

- **Sourced:** the fan's size class (12 inch); the order of the four steps; the pin and the slot; the reversed thread
  and its tightening sense; the support at the bottom of the rear grille; three clips on the rear grille.
- **Estimates, not measurements** (no maker publishes them and none was measured): every size of the fan and its
  parts in `task.json` and `fankit.py`, including the grilles' 34.9 cm (retailers' listings give the fan's width as
  about 13.75 in; the listings were seen only in search results), the shaft and pin, the hub's bore and slot, the
  cap, the support's lip, the clips and their 58 degree open angle; the parts' weights (blade 70 g, cap 10 g, grille
  110 g, clips 3 g); the motor's drag (0.004 N m); thread friction; the clips' springs (0.0004 N m a degree);
  friction between parts (0.6 static, 0.5 sliding). Where the three clips sit round the rim (left, right and the side
  opposite the support) is an assumption.
- **The thread is not cut.** Once the cap's bore is 2.5 mm over the shaft's end, within 1.5 mm of its axis and 10
  degrees of square, a joint keeps the cap on the shaft's axis, free to turn against thread friction, and a spring
  (8,000 N/m, at most 6 N) draws it along the shaft by 4 mm for every turn of the cap on the shaft, anticlockwise.
  The pitch is coarser than a real fan's, to keep a trial short: the cap comes tight in about 1.2 turns. Turned
  clockwise it backs off and comes free. Turning cap and shaft together moves nothing.
- **The clips** are rigid hooked levers on hinges. Each has a spring whose pull is switched from outward to inward
  when the clip is pushed in past 27 degrees, which is what an over-centre spring does.
- The blades are flat plates in four pieces each; the grilles' wires and ribs are drawn, and one in three collides.
  The front grille's rim is a flat ring 14 mm wide and 10 mm thick with a 3 cm band standing on it.
- The fan's body (housing, rear grille, pillar, base) is fixed in place. There is no cord, no motor and no
  oscillation knob. The rear grille is already on.
- **The shaft can creep.** After the cap is tight and the hands have left, the shaft, blade and cap sometimes turn
  together slowly (0.3 to 2 degrees a second in some runs, not at all in others; in the layout B demo, 17 degrees in
  the 63 s after the cap came tight). The motor's drag is modelled as
  viscous below its 0.004 N m limit, so a small residual torque from the contact solver is enough to turn it. The
  rest rule (3 degrees in half a second) allows it.
- The towels and the dish rack are fixed blocks.

## Sources

- Comfort Zone, 12" Table Fan CZ121 / CZ121BK, Owner's Manual (Rev. 080215),
  https://images.homedepot-static.com/catalog/pdfImages/56/5696792a-3273-43b3-a7fa-6015a152397b.pdf: "Place fan blade
  onto motor shaft. Notice the pin on motor shaft should fit into slots of fan blade." "Thread blade cap onto
  threaded end of motor shaft. Note: Threads are reversed at this connection. Tighten=Counter-clockwise." "Place
  front grille with logo facing upwards onto fan. Be sure the front grille is inserted into rear grille support on
  the bottom of rear grille." "Press the three (3) grille clips on rear grille to securely attach to front grille."
  "Clean the grille surface with a clean damp cloth." "Replace front grille before fan use."
- Midea, 12" Table Fan FT30-8MB, Instruction Manual,
  https://images.homedepot-static.com/catalog/pdfImages/3c/3c5c6cc3-b099-48c7-a1b8-76672e10e634.pdf: "Disconnect fan
  when removing grilles for cleaning." "Plastic parts should be cleaned with mild soap and damp cloth or sponge.
  Thoroughly remove soap film with clean water." "Dry all parts completely before reassembling and reconnecting to
  power source." "Install the blade onto the shaft until the retaining pin fits into the notch on the blade." "Screw
  spinner on to motor shaft counterclockwise." "Mount the Front Grill and join both Grills together using the Clips."
- 12" (30 cm) Oscillating Desk Fan, product no. 043-5693-8, instruction manual,
  https://media-www.canadiantire.ca/manual/product/0435693/0435693p-en-c4119fd1-50b4-4761-8ddc-fd6746a4ff16.pdf:
  "Remove the blade cap by twisting it CLOCKWISE while holding the metal shaft tightly. Note: Threads are reversed.
  Unscrew = Clockwise." This fan's front grille snaps on with eight clips ("Be sure all 8 clips are secure"); the
  three-clip arrangement here follows the CZ121.

## Related tasks on the board

Nothing on the board assembles a fan.

- "Replace a drone's chipped propeller with the new one and tighten its prop nut" (accepted) also stops a propeller
  on a free motor shaft while a nut is turned. Here the blade is keyed to the shaft by a cross pin and has to be
  turned until it drops, the thread is left-handed, and two more fits follow: a 35 cm grille hooked under a lip and
  three over-centre clips.
- "Refit a gas burner after cleaning: turn the head until its slot drops round the igniter, then seat the cap"
  (accepted) also turns a part until a slot drops over a fixed feature. Here the feature is a pin on a shaft that
  turns away, and the cap is threaded.
- "Close a picture frame's back: press it flush with one hand while the other turns its clips, swapping hands"
  (accepted) turns flat swivel clips. These clips are sprung levers that snap over a rim.
