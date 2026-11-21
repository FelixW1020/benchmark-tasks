# Lock a trailer's rear door with its cam-action lock rod

A bench section of a dry van's right-hand rear door stands upright on the table, facing the robot. The door is shut
but not locked: the lock rod's handle hangs swung out toward the robot and the hasp's flip keeper is up. The
instruction is "Lock this trailer door: swing the lock rod's handle in until it is flat against the door, drop it into
its hasp, and flip the hasp's keeper down over it. Then let go."

Everything is at full size. The rod is a short cut length of the real pipe, so that the handle, the hasp and the cam
at the rod's lower end are all on the table and in view.

## Why it matters

Heavy and tractor-trailer truck drivers (O*NET 53-3032.00, 2,221,200 employed in 2025) "Inspect loads to ensure that
cargo is secure." A federal rule bars a driver from operating the vehicle unless its "tailgate, tailboard, doors,
tarpaulins, spare tire and other equipment used in its operation, and the means of fastening the commercial motor
vehicle's cargo, are secured" (49 CFR 392.9(a)(2)). On a dry van the doors are secured with cam-action lock rods, and
a driver works them at every pick-up and every delivery.

## What the task asks of a robot

- **A double-action handle.** Turning the handle turns the rod. The same handle is pinned to the rod's hub so that it
  lifts. To lock, it has to be swung in, lifted over the hasp's lip, pressed flat on the hasp's face and let down into
  the seat. Swung in without the lift, it stops against the lip (a control run below).
- **Working against a spring.** Over the rod's last 15 degrees the cam draws the door onto its seal and the handle
  pushes back, 2 N at the grip when it is flat on the hasp. The lift and the lowering are done while holding that.
  A handle let go outside the hasp springs back out.
- **A constrained path.** The held handle can only move on its arc about the pipe and on its pin; the hand has to
  turn with it through 60 degrees (45 in layout B).
- **A small flap, last.** The keeper is a 28 mm wide plate leaning on its stop above the seat. It has to be pushed
  over its top so that it falls and hangs over the handle.
- **Order.** A keeper flipped down first blocks the seat. A door with its keeper down and its handle not under it
  looks locked and is not: that scores 0.

The demo uses both arms (the left for the handle, the right for the keeper). One arm could do both in turn; the task
does not claim to need two.

## Files

| File | What it is |
|---|---|
| `task.py` | the scene, the instruction and the scoring rule |
| `doorkit.py` | the fixture, rod, handle and keeper: shapes, joints, the seal's spring |
| `demo.py` | the scripted demo and the control runs |
| `test_scoring.py` | 28 offline checks of the scoring rule and the geometry |
| `task.json` | the task's details for the board |

## Run a trial

    inspect-robots run --embodiment yam_arena --policy dream \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Lock this trailer door: swing the lock rod's handle in until it is flat against the door, drop it into its hasp, and flip the hasp's keeper down over it. Then let go." \
      --scorer task_score --max-steps 900

Layout B: set `DOOR_LOT=B` in the environment (the fixture 1.5 cm further away and 1.5 cm to the right, the handle
starting 45 degrees out instead of 60).

## Scripted demo

    python demo.py task.py out_dir                 # layout A; DOOR_LOT=B for layout B
    python demo.py task.py out_dir --mode nolift   # a control run (see below)

1. The left hand, level and turned to face the handle's flat side, closes on the handle 20 cm out from the pipe, one
   finger above the bar and one below.
2. It swings the handle in on its arc, turning with it. The cam's prongs go either side of the keeper pin on the sill.
3. Eight degrees short of locked it lifts the handle 4.5 degrees on its pin, carries it over the hasp's lip against the
   seal's push-back until the bar is flat on the hasp's face, and lets it down into the seat. It lets go and draws
   back; the handle springs out 4 mm against the lip and stays.
4. The right hand, fingers closed, pushes the keeper's finger tab over its top. The keeper falls and hangs over the
   handle. Both hands go back to rest.

The script is open loop: every target comes from the layout constants in `task.py` and the sizes in `doorkit.py`. The
only feedback is each hand's own measured position and attitude, used to correct for the arm's sag under load. It
never reads a part's state.

## Scoring

Scored at the end of the trial, from where the parts are then.

| Score | State at the end |
|---|---|
| 1.0 | Handle in the hasp's seat, keeper down over it, both grippers at least 3 cm from the handle and the keeper, nothing moving |
| 0.75 | Handle in its seat and keeper down, but a gripper is within 3 cm, or something is still moving |
| 0.5 | Handle in its seat, keeper not down |
| 0.25 | Handle not in its seat and keeper not down, but the cam has hold of its pin |
| 0 | Keeper down while the handle is not in its seat; or the cam has no hold; or nothing done |

Definitions, each with how to check it on a real fixture:

- **Handle in its seat:** at the hasp, the bar's door-side face is within 6 mm of the hasp's face (a ruler or a 6 mm
  feeler; in the seat it can be at most 4 mm off) and its underside is within 3 mm of the seat (a 3 mm feeler). A
  handle in that place is behind the lip.
- **Keeper down:** within 10 degrees of hanging straight down (by eye against the hasp's edge; flipped up it leans
  25 degrees past upright).
- **Cam has hold of its pin:** the keeper pin on the sill is between the cam's two prongs, seen from above. In the
  simulator this is the rod within 25 degrees of locked.
- **Nothing moving:** the rod and the handle turn at under 0.05 rad/s and the keeper at under 0.3 rad/s (in a real
  trial: no visible movement between two video frames half a second apart).

## Verification

| Run | Score | What happened |
|---|---|---|
| Scripted demo, layout A | 1.0 | ends at step 469 of 900; handle 3.7 mm off the hasp's face, keeper 0.2 degrees from hanging |
| Scripted demo, layout B (fixture moved, handle starting 45 degrees out) | 1.0 | ends at step 469 of 900; handle 3.7 mm off the hasp's face, keeper 0.1 degrees from hanging |
| Control: nothing moves | 0.0 | handle and keeper where they started |
| Control: handle swung to 10 degrees and let go | 0.25 | handle comes to rest 23 degrees out; the cam has the pin |
| Control: handle swung all the way in without the lift | 0.25 | stops against the hasp's lip (3.5 degrees at its closest), ends 19 degrees out |
| Control: handle seated, keeper left up | 0.5 | handle 3.7 mm off the hasp's face, behind the lip |
| Control: everything, but the right hand stays where it pushed the keeper over | 0.75 | handle seated, keeper down, gripper 19.8 mm from the keeper |
| Control: keeper flipped down first, then the handle swung to 10 degrees | 0.0 | keeper down, handle 23 degrees out: looks locked, is not |
| Dream-1, 900 steps | 0.0 | the right gripper went to the handle's free end and stayed beside it; the handle was not swung in and the keeper was not touched |

28 offline checks pass (`python -m pytest -q test_scoring.py`). They cover every tier and the false-lock zero, the
seat's limits, the cam's hold on the pin, the seal's spring and the clearances between the handle, the lip and the
keeper.

## Runtime and seeds

One trial takes about two and a half minutes on an L40S, including about a minute of start-up; the demo's 469 steps
are 23 s of simulated time. Nothing is random: `object_jitter=0`, one fixed layout per lot, no seed.

## Getting the real parts

| Item | Product | Where the size comes from | Price shown |
|---|---|---|---|
| Handle, hasp with flip keeper, cams, keepers, pipe guides | Polar Hardware PLR258-002, "Cam-Action Lockable Door Latch Kit for Large Enclosed Trailers - Zinc-Plated Steel" | the listing: "Handle length: 14\"", "Hasp dimensions: 5-1/8\" tall x 2\" wide", "Retainer dimensions: 4\" tall x 1-1/2\" wide" | $146.25 |
| Lock rod | 32 cm of 3/4 in Schedule 40 steel pipe (1.050 in outside) | the listing names the pipe | a few dollars (estimate) |
| Door section, sill | 18 mm plywood, 67 x 28 cm, on a 6 x 6 cm timber sill | this task's own sizes | a few dollars (estimate) |
| Seal | a strip of closed-cell foam weatherstrip behind the door section's lower edge, or a torsion spring on the pipe | see below | a few dollars (estimate) |

The listing says the pipe "needs to be welded to the cam ends as well as the handle for complete installation"; for a
bench fixture the hub and the cam can be pinned or clamped to the pipe instead.

## Building the bench fixture

This is a design from the simulated fixture's numbers. It has not been built.

1. Stand the plywood door section on the sill, facing the robot, and brace it from behind. Fix the sill to the table.
2. Mount the two pipe guides one above the other so that the pipe's axis is 3.5 cm in front of the door's face.
3. Fix the hub and handle on the pipe with the handle's pin 16 cm above the table, and one cam on the pipe's lower end
   3 cm above the table. Fix that cam's keeper on the sill where the cam meets it.
4. Mount the hasp 30 cm along the locked handle from the pipe, at the height where the lowered handle rests in it.
5. Give the last 15 degrees of the rod's turn a push-back: with the keeper up and the handle held flat on the hasp at a
   point 20 cm from the pipe, a spring scale pulling the handle straight out should read 2.0 N (accept 1.5 to 2.5 N).
   Add or trim foam, or change the torsion spring, until it does. Let go: the handle should come to rest 15 to 25
   degrees out, with the cam still holding its keeper.
6. Mark the handle's start on the table: its tip at (16.1, 0.3) cm for layout A.

Check after building: the handle swung in level stops against the hasp's lip; lifted about 2 cm at the hasp it passes
over; let down, it stays in the seat with the hand off; with the keeper down it cannot be lifted out.

A kit's own hasp, cam and keeper will differ in detail from the simulated ones (see the next section). If they do,
measure them and change the constants at the top of `doorkit.py`; the scoring rule is written in terms of the hasp's
own face, seat and keeper, so it carries over.

## What the simulation simplifies

- **One force law.** The door seal's push-back is a torsion spring on the rod over its last 15 degrees, 0.4 N m at
  locked. It is an estimate, set to what these arms can hold (2 N at the grip); a loaded trailer door takes far more.
  Everything else that holds a part is contact between solid shapes: the seat and lip, the keeper over the handle,
  the cam's prongs either side of the keeper pin, the handle's stop on its hub.
- **From the listing:** the handle's 14 in length, the hasp's 5-1/8 x 2 in outline, the 3/4 in pipe.
- **Estimates, not measurements:** the bar's 8 x 25 mm section and 0.55 kg; the hasp's seat (12 mm deep), lip (12 mm
  high, 5 mm thick) and its 3 cm stand-off from the door; the keeper's size, 40 g and pin friction; the cam as a plate
  with two prongs and the keeper as a 6 mm pin (the kit's cam and keeper are castings whose shapes no listing gives);
  the hub's 3 cm offset and the handle's 30 degrees of lift; friction 0.5 on steel, 0.8 on the handle; 0.03 N m of
  drag in the pipe guides.
- The pipe guides are drawn but are not solid; the rod turns on a pin joint in their place.
- The rod is cut to 32 cm. On a trailer it runs the door's full height with a cam at each end, and the handle is
  about 1.2 m above the ground.
- The door itself does not move: it is already shut, and the cam's pull on it is the spring above.
- Not tried on a physical rig.

## Nearest tasks

- "Plug in a trailer's 7-way connector under its spring cover" (accepted) and "Couple a tractor's air line to the
  trailer's gladhand: off the dummy coupler, seals together at 90 degrees, turn to lock" (accepted) are coupling steps
  from the same job: a plug pushed in, and a quarter-turn coupling. Neither has a lever that is turned, lifted and
  dropped behind a lip.
- "Retract a grounded two-speed landing leg to its travel stop and stow the crank" (pending) turns a crank through
  many turns.

No accepted or pending task works a door's lock rod, a double-action handle or a hasp with a flip keeper.

## Sources

- O*NET OnLine, 53-3032.00 Heavy and Tractor-Trailer Truck Drivers (onetonline.org/link/summary/53-3032.00):
  "2,221,200 employees" (2025); task: "Inspect loads to ensure that cargo is secure."
- 49 CFR 392.9(a)(2) (law.cornell.edu/cfr/text/49/392.9): "The commercial motor vehicle's tailgate, tailboard, doors,
  tarpaulins, spare tire and other equipment used in its operation, and the means of fastening the commercial motor
  vehicle's cargo, are secured".
- A retailer's listing of the latch kit (etrailer.com, Polar Hardware PLR258-002): the sizes quoted above; "Flip the
  handle keeper on the hasp latch up and lift the handle to release"; "Rotate the handle to either release or engage
  the cams"; "Schedule 40 3/4\" plumbers pipe needs to be welded to the cam ends as well as the handle for complete
  installation."

## Third-party assets

None. Every shape is built in `doorkit.py`.
