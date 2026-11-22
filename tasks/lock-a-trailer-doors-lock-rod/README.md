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
- **Working against the seal.** Over the rod's last 15 degrees the cam draws the door onto its seal and the handle
  pushes back, rising evenly to 20 N m at locked: 53 N square to the handle at its free end (see "The handle's
  resistance"). The lift over the lip and the lowering into the seat are done while holding that. A handle let go
  outside the hasp is thrown back out.
- **More push than one pinch holds.** A gripper's pinch on the bar holds by pad friction, about 50 N at full grip. In
  the demo the right hand's closed fingertips push the bar near its tip (about 45 N) while the left hand, gripping,
  guides it, lifts it and sets it down (about 21 N of push).
- **A constrained path.** The held handle can only move on its arc about the pipe and on its pin; the hand has to
  turn with it through 60 degrees (45 in layout B).
- **A small flap, last.** The keeper is a 28 mm wide plate leaning on its stop above the seat. It has to be pushed
  over its top so that it falls and hangs over the handle.
- **Order.** A keeper flipped down first blocks the seat. A door with its keeper down and its handle not under it
  looks locked and is not: that scores 0.

The demo uses both arms on the handle and then the right hand on the keeper. The task does not claim that this is
the only way: a hand that hooks the bar instead of pinching it could carry more of the push.

## Files

| File | What it is |
|---|---|
| `task.py` | the scene, the instruction and the scoring rule |
| `doorkit.py` | the fixture, rod, handle and keeper: shapes, joints, the seal's spring (`DOOR_SEAL` sets its peak, 20 N m by default) |
| `demo.py` | the scripted demo and the control runs |
| `test_scoring.py` | 31 offline checks of the scoring rule, the geometry and the handle's resistance |
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

1. The left hand, level and turned to face the handle's flat side, closes firmly on the handle 20 cm out from the
   pipe, one finger above the bar and one below, and swings it in to 22 degrees from locked, turning with it. The
   cam's prongs go either side of the keeper pin on the sill.
2. The right hand, fingers closed, comes up behind the bar 35 cm out from the pipe, beyond the hasp, and sets its
   fingertips on the bar's outer face.
3. Both hands swing the handle in against the seal. Eight degrees short of locked they lift it 4.5 degrees on its pin
   (the right hand's fingertips rise with the bar), carry it over the hasp's lip to just short of the hasp's face, and
   let it down into the seat.
4. Both hands ease off. The seal pushes the bar 4 mm out against the lip, which holds it. The right hand draws back;
   the left hand opens and draws back.
5. The right hand, fingers closed, pushes the keeper's finger tab over its top. The keeper falls and hangs over the
   handle. Both hands go back to rest.

Holding the seal's force: each arm's joints are position-controlled springs, so a hand pushes by being commanded past
where it is. The demo adds to each joint's commanded angle the joint torque that the wanted push needs (the arm's
Jacobian transposed times the force) divided by that joint's stiffness. The push it asks for comes from the fixture's
stated spring (the table in "The handle's resistance") and the planned handle angle, 80% of the torque to the right
hand and 20% to the left.

The script is open loop: every target and every force comes from the layout constants in `task.py` and the sizes and
the spring in `doorkit.py`. The only feedback is each hand's own measured position and attitude, used to correct for
the arm's sag under load (limited to a few centimetres while a hand is on the handle, so that it cannot wind up
against the fixture). It never reads a part's state.

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
- **Nothing moving:** over the last half second the rod and the handle have each turned by no more than 0.5 degrees
  and the keeper by no more than 3 degrees (in a real trial: no visible movement between two video frames half a
  second apart). Positions are compared, not speeds: a bar pressed against the lip by the seal reports a jitter in
  speed in the simulator while it stays where it is.

## Verification

| Run | Score | What happened |
|---|---|---|
| Scripted demo, layout A | 1.0 | ends at step 647 of 900; handle 0.7 degrees from locked and 3.7 mm off the hasp's face, behind the lip; keeper 0.1 degrees from hanging |
| Scripted demo, layout B (fixture moved, handle starting 45 degrees out) | 1.0 | ends at step 647 of 900; handle 3.7 mm off the hasp's face, keeper 0.1 degrees from hanging |
| Control: nothing moves | 0.0 | handle and keeper where they started |
| Control: handle swung in to 20 degrees, short of the seal, and let go | 0.25 | handle stays 20 degrees out; the cam has the pin |
| Control: both hands push the handle all the way in without the lift | 0.25 | stops against the hasp's lip 3.6 degrees from locked; thrown back when the hands leave, it comes to rest 24.7 degrees out, just inside where the cam has the pin |
| Control: handle seated, keeper left up | 0.5 | handle 3.7 mm off the hasp's face, behind the lip, held there by the seal |
| Control: everything, but the right hand stays where it pushed the keeper over | 0.75 | handle seated, keeper down, gripper 19.6 mm from the keeper |
| Control: keeper flipped down first, then the handle swung in to 20 degrees | 0.0 | keeper down, handle 20 degrees out: looks locked, is not |
| Dream-1, 900 steps | 0.0 | the left gripper went to the handle beside the pipe and the right gripper to its free end; the handle was swung a little way in and ended more than 25 degrees out, short of the cam's hold; the keeper was not touched |

All of these were run with the seal at 20 N m. The same demo also seated the handle and dropped the keeper with the
seal at 10 N m.

31 offline checks pass (`python -m pytest -q test_scoring.py`). They cover every tier and the false-lock zero, the
seat's limits, the cam's hold on the pin, the rule for "nothing moving", the seal's spring and the forces it puts on
the handle and the lip, the demo's sharing of that force between the hands, and the clearances between the handle,
the lip and the keeper.

## Runtime and seeds

One trial takes about three minutes on an L40S, including about a minute of start-up; the demo's 647 steps are
32 s of simulated time. Nothing is random: `object_jitter=0`, one fixed layout per lot, no seed.

## The handle's resistance

No published measurement of the force on a trailer lock rod's handle was found (lock-rod makers' pages, patents,
container and trailer door guides and ergonomics studies were searched). The fixture's resistance is therefore a
**calculated estimate**, built from published figures, and not a measurement of this latch.

| Step | Figure | Where it comes from |
|---|---|---|
| Load on a door's latch edge | half the gasket's load: "gasket N/cm × (width + height)" | an enclosure-latch maker's design note |
| Gasket load, stiff end | 3 to 7 N/cm | the same note, for sealed enclosure doors |
| Gasket load, soft end | 0.44 to 0.77 N/cm | a 2025 study of car door seals: "4.38 N to 7.67 N" on 100 mm of seal; whole-door seal reaction "173 N to 212 N" |
| Door | one rear door leaf, about 1.25 x 2.79 m, so width + height = 4.0 m | an estimate |
| Lock rods sharing it | two on most dry van doors; one on a cargo trailer's door | an estimate |
| Rod torque per newton at the cams | 0.023 m: the cam draws the door about 6 mm onto its seal over the rod's last 15 degrees | "Approximately 6mm of compression all round is required" (a container-seal maker); the 15 degrees is an estimate |

Result, as torque on the rod at locked:

| Case | Latch-edge load | Per rod | Rod torque at locked |
|---|---|---|---|
| Car-type seals, two rods | 180 to 310 N | 90 to 155 N | 2 to 4 N m |
| Car-type seals, one rod | 180 to 310 N | 180 to 310 N | 4 to 7 N m |
| Enclosure-type gaskets, two rods | 1,200 to 2,800 N | 600 to 1,400 N | 14 to 32 N m |
| Enclosure-type gaskets, one rod | 1,200 to 2,800 N | 1,200 to 2,800 N | 28 to 65 N m |

**The task uses 20 N m at locked**, rising evenly from zero at 15 degrees from locked. That is inside the range for a
two-rod door with gaskets as stiff as a sealed enclosure's, and about six times the figure for car-type seals. At the
handle it is:

| Handle's angle from locked | Rod torque | Pull square to the handle at its free end (37.4 cm from the pipe's axis) |
|---|---|---|
| 15 degrees or more | 0 | 0 (the rod's own drag, under 1 N) |
| 10 degrees | 6.7 N m | 18 N |
| 7.5 degrees | 10 N m | 27 N |
| 5 degrees | 13.3 N m | 36 N |
| 0 (flat on the hasp's face) | 20 N m | 53 N |

Seated, the handle rests 0.7 degrees from locked against the hasp's lip, 30 cm from the pipe, with about 64 N.

What this does not cover. A steel shipping container is far stiffer: a container supplier writes that "Relatively
modest rotational force applied to the handle (typically 50-100 Nm of torque) is converted into large clamping force
in the door frame, often exceeding 1,000 kg of pressure on the doors." With four rods sharing that clamping force
those figures give 0.02 to 0.04 m of torque per newton, the same ratio as the 0.023 m used here; the difference is the
load. A container door, a racked door and a door with a load leaning on it are outside what this fixture represents,
and are beyond these arms: the two arms together can hold about 50 N m on this handle before a joint reaches its
torque limit (a static figure with no margin).

## Getting the real parts

| Item | Product | Where the size comes from | Price shown |
|---|---|---|---|
| Handle, hasp with flip keeper, cams, keepers, pipe guides | Polar Hardware PLR258-002, "Cam-Action Lockable Door Latch Kit for Large Enclosed Trailers - Zinc-Plated Steel" | the listing: "Handle length: 14\"", "Hasp dimensions: 5-1/8\" tall x 2\" wide", "Retainer dimensions: 4\" tall x 1-1/2\" wide" | $146.25 |
| Lock rod | 32 cm of 3/4 in Schedule 40 steel pipe (1.050 in outside) | the listing names the pipe | a few dollars (estimate) |
| Door section, sill | 18 mm plywood, 67 x 28 cm, on a 6 x 6 cm timber sill | this task's own sizes | a few dollars (estimate) |
| Seal stand-in | a steel arm 10 cm long clamped to the pipe, and a compression spring of 7.6 N/mm with 30 mm of travel (3.0 mm music wire, 20 mm mean coil diameter, 13 active coils, 80 mm free length) in a guide tube | "The handle's resistance" and step 5 below | a few dollars (estimate) |

The listing says the pipe "needs to be welded to the cam ends as well as the handle for complete installation"; for a
bench fixture the hub and the cam can be pinned or clamped to the pipe instead.

## Building the bench fixture

This is a design from the simulated fixture's numbers. It has not been built.

1. Stand the plywood door section on the sill, facing the robot, and brace it from behind. Clamp the sill to the
   table at both ends: the robot pushes on this fixture with up to 70 N.
2. Mount the two pipe guides one above the other so that the pipe's axis is 3.5 cm in front of the door's face.
3. Fix the hub and handle on the pipe with the handle's pin 16 cm above the table, and one cam on the pipe's lower end
   3 cm above the table. Fix that cam's keeper on the sill where the cam meets it.
4. Mount the hasp 30 cm along the locked handle from the pipe, at the height where the lowered handle rests in it.
5. Give the last 15 degrees of the rod's turn the seal's push-back. Clamp the 10 cm arm to the pipe above the upper
   guide, pointing over the door's top edge, and fix the spring in its guide tube to a block on the back of the door so
   that the arm's end meets the spring when the handle is 15 degrees from locked and has pressed it 26 mm when the
   handle is flat on the hasp's face (200 N on the arm: 20 N m). Bolt the block through the door.
   Check it with a luggage scale hooked on the handle's free end and pulled square to the bar, keeper up, handle
   lifted clear of the lip: 53 N (5.4 kg) as the bar reaches the hasp's face, 27 N at 7.5 degrees (the bar 4 cm from
   the hasp's face at the hasp), nothing but the rod's drag from 15 degrees out. Accept within 10%; shim the spring or
   move the arm until it does.
6. Mark the handle's start on the table: its tip at (16.1, 0.3) cm for layout A.

Check after building: the handle swung in level stops against the hasp's lip; lifted about 2 cm at the hasp it passes
over; let down, it stays in the seat with the hand off, pressed against the lip; with the keeper down it cannot be
lifted out. A handle let go inside the last 15 degrees is thrown back by the spring: keep hands and faces out of its
arc, and reset by holding the handle while lifting it out.

A kit's own hasp, cam and keeper will differ in detail from the simulated ones (see the next section). If they do,
measure them and change the constants at the top of `doorkit.py`; the scoring rule is written in terms of the hasp's
own face, seat and keeper, so it carries over.

## What the simulation simplifies

- **One force law.** The door seal's push-back is a spring on the rod over its last 15 degrees, 20 N m at locked,
  with a little loss in it (0.4 N m s/rad). The 20 N m is the calculated estimate of "The handle's resistance", not a
  measurement; a straight-line rise stands in for a real cam's curve. Everything else that holds a part is contact
  between solid shapes: the seat and lip, the keeper over the handle, the cam's prongs either side of the keeper pin,
  the handle's stop on its hub. The spring and its arm are not drawn.
- **From the listing:** the handle's 14 in length, the hasp's 5-1/8 x 2 in outline, the 3/4 in pipe.
- **Estimates, not measurements:** the bar's 8 x 25 mm section and 0.55 kg; the hasp's seat (12 mm deep), lip (12 mm
  high, 5 mm thick) and its 3 cm stand-off from the door; the keeper's size, 40 g and pin friction; the cam as a plate
  with two prongs and the keeper as a 6 mm pin (the kit's cam and keeper are castings whose shapes no listing gives);
  the hub's 3 cm offset and the handle's 30 degrees of lift; friction 0.5 on steel, 0.8 on the handle; 0.3 N m of
  drag in the pipe guides and at the cam.
- The pipe guides are drawn but are not solid; the rod turns on a pin joint in their place.
- The rod is cut to 32 cm. On a trailer it runs the door's full height with a cam at each end, and the handle is
  about 1.2 m above the ground.
- The door itself does not move: it is already shut, and the cam's pull on it is the spring above. A real door also
  racks and its seal ages; neither is modelled.
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

- An enclosure-latch maker's design note (hjprecisions.com/blog/swing-handle-vs-compression-latch/): "The latch edge of
  a hinged door carries exactly half the gasket load. That load is gasket N/cm × (width + height)."; its worked
  examples use "gasket force q = 3–7 N/cm".
- Yang Z, Li L, Huang J, Chen G, Liu Z, Wang Z (2025), "Prediction method for the reaction force of vehicle door
  sealing systems", PLoS One 20(11): e0336802: "When the CLD force value of the door frame seal on the door lock side
  under the D0 condition increases from 4.38 N to 7.67 N, the reaction force of the sealing system increases from
  173 N to 212 N."; its seal elements are "spaced at intervals of 100 mm to simulate the seal stiffness per 100 mm".
- A container-seal maker's note on compression seals (bloxwichgroup.com/news/container-door-seals-other-types/):
  "Approximately 6mm of compression all round is required."
- A container supplier's page on door openings (hz-containers.com/en/technical-information/door-opening-of-a-shipping-container/):
  "Relatively modest rotational force applied to the handle (typically 50-100 Nm of torque) is converted into large
  clamping force in the door frame, often exceeding 1,000 kg of pressure on the doors."
- Wikipedia, "Lockrod": trailer doors must "swing free and be safely opened and closed with minimal effort"; a cam's
  reach "helps to overcome resistance from compression seals and warped doors".
- The arms' limits used for the demo's forces: joint torque limits of 27 N m (shoulder and elbow) and 7 N m (wrist) and
  joint stiffnesses of 80 and 10 N m/rad, from the `yam_arena` package's constants.

## Third-party assets

None. Every shape is built in `doorkit.py`.
