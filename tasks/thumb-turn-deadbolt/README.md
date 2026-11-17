# Lock a door's deadbolt by its inside thumb-turn while pressing the door shut against its weatherstrip

A section of an inswing exterior door hangs in a fixed wooden frame on the table, its inside face toward the robot. It
is solid core, 44 mm thick, 53 cm from its hinge axis to its latch edge, 45 cm tall and 6.3 kg. A deadbolt is fitted
60 mm from the latch edge. The robot must lock the door with it.

On many exterior doors the weatherstrip pushes the door back off its stop, and the latch doesn't hold it shut. This
door has no working latch, and its weatherstrip holds it 12 mm open at the latch edge. Turned now, the bolt moves
2.5 mm, hits the jamb beside the strike opening and stops the thumb-turn. The opening has 2 mm of clearance round the
bolt, so the bolt only goes in while the door is held within about 2 mm of its stop. If the bolt is thrown with the
door swung open, it stands out of the door's edge and stops the door from closing.

What the task tests:

- **Holding a heavy door shut.** The 6.3 kg door has to be pressed shut against the weatherstrip and kept there
  throughout: about 8 N at 28 cm from the hinge axis, about halfway across.
- **Turning a small knob with the other hand.** The thumb-turn's wing is 42 mm long, 12 mm thick and stands 18 mm
  off a round hub. It has to be pinched and turned a quarter turn, its top toward the latch edge, against 0.25 N*m.
  That is a wrist rotation about an axis pointing into the door.
- **Order and coordination.** Turning before the door is shut, or letting it spring back mid-turn, leaves the bolt
  blocked against the jamb.

## What changed in this revision

The review found that the bolt's projection cannot be measured with a ruler or a photo once the bolt is inside the
door and the strike. The score no longer asks for it.

- **Every bolt threshold is now read from the thumb-turn's angle,** which stays in view with the door locked. Two
  angles, A and B, are found once per lock with the door open, where the bolt can be measured. On the simulated lock
  they are 33 and 84 degrees. See [Scoring a real trial](#scoring-a-real-trial).
- **The scorer reads what a real trial reads.** `score()` and `success()` now use the gap at the latch edge and the
  thumb-turn's angle. The stillness check is on the gap and the angle too.
- **The demo was rerun under the revised scorer:** 1.0 at step 341, as before. The scene and the demo's motion are
  unchanged, so the videos show the same run.
- **The Dream-1 run was not rerun.** Any score above 0 under the revised rule needs a state that also scored under
  the old one (the door within 2 mm of shut, or the bolt in the strike), and that run scored 0.0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, readouts, `success()` and `score()` |
| `mechkit.py` | Multi-box parts, hinges and spring sliders for yam_arena task files (with round pins added) |
| `demo.py` | The scripted demo |
| `test_scoring.py` | Offline checks: every scoring outcome, the two thumb-turn angles against the bolt's travel, and the lock's geometry (start clearances, the crank pin in the bolt's slot all through the turn, the bolt clearing the strike only with the door shut, the door's swing) (no GPU) |

`mechkit.py` hooks `yam_arena.scene` and the embodiment's reset and step without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer. It is the part kit used by other tasks in this
repository; this copy adds round pins (cylinders) to parts.

## The mechanism

- **Door:** a free 6.3 kg body on a revolute joint about the hinge axis (on its inside face at the hinge edge), with
  0.2 N*m of friction. It stops at shut and opens toward the robot.
- **Weatherstrip:** a rubber strip in a kerf in the latch jamb's stop, on a 250 N/m spring. Free, it stands 12 mm proud
  and holds the door 1.3 degrees open. The door shut compresses it to 3.9 N.
- **Thumb-turn:** a 50 g part on a revolute joint through the rose, with 0.25 N*m of friction (a real thumb-turn takes
  0.2 to 0.5 N*m). Its spindle carries a crank inside the door, and the crank's 6 mm round pin, 17.7 mm off the axis,
  runs in a 7 mm slot in the bolt's tail.
- **Bolt:** a 60 g steel bar (22 x 22 mm, 85 mm with its tail) on a prismatic joint in the door, from retracted (flush
  with the faceplate) to 25 mm out. A quarter turn swings the pin from 45 degrees on the hinge side of upright to 45
  degrees on the latch side, which carries the bolt 25 mm. The pin and slot are the only link between thumb-turn and
  bolt. So a blocked bolt stops the thumb-turn, as on a real lock.
- **Frame:** fixed. The latch jamb is 3 mm from the latch edge, with a steel strike plate and a 26 x 26 mm opening,
  32 mm deep, centred on the bolt. The stops are 2 mm off the shut door's outside face.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
INSTRUCTION=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/deadbolt-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200              # writes demo_*_cam.mp4, setup.png, result.json
DEMO_DEBUG=1 python demo.py task.py debug_out --no-record      # also prints the door gap, bolt, thumb-turn, TCPs,
                                                               # and writes them for every step to trace.csv
DEMO_DEBUG=1 python demo.py task.py slow_out --no-record --turn-s 30   # the quarter turn in 30 s: bolt against angle
python -m pytest test_scoring.py                               # offline checks
```

1. The left arm, its fingers closed and pointing level at the door, presses the door 28 cm from its hinge axis at the
   bolt's height. It is sent 14 mm past the shut face, so the compliant arm keeps pressing it shut.
2. The right arm comes in level with its fingers open 46 mm either side of the upright wing. It stops with the
   fingertips 1.5 mm short of the round hub, so the pads close on the flat wing, and it pinches the wing.
3. The right hand turns 100 degrees about the thumb-turn's axis, its top toward the latch edge. The thumb-turn stops
   at a quarter turn with the bolt fully out, and the compliant wrist lags. Both hands hold.

The demo runs open loop. It reads no simulator state while it runs; every target comes from the fixed layout in
`task.py`, and nothing needed calibrating. With `-E object_jitter=0` there are no seeds. It reaches 1.0 (success) at
step 341 (17 s): door gap 0.0 mm, thumb-turn at 91.5 degrees, bolt 24.7 mm out.

| Step | Score | What happened | Gap (mm) | Thumb-turn (degrees) | Bolt (mm) |
|---|---|---|---|---|---|
| 82 | 0.25 | door pressed shut | 0.4 | 0.0 | 0.0 |
| 243 | 0.5 | thumb-turn past A | 0.1 | 33.1 | 8.2 |
| 303 | 0.75 | thumb-turn past B, bolt 23 mm out | 0.0 | 84.8 | 23.1 |
| 322 | 1.0 | locked and still for 1 s | 0.0 | 90.5 | 24.4 |
| 341 | 1.0 | success: the episode ends | 0.0 | 91.5 | 24.7 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). It uses two readings,
both in view with the door locked:

- **Gap:** how far the door's latch edge stands off its shut position (12 mm at the start).
- **Turn:** the thumb-turn wing's angle from upright (0 at the start, about 90 degrees locked).

The bolt is out of sight once it is in the strike, so its travel is read from the turn:

- **Angle A** is where the bolt stands 8 mm out of the door's edge. That is 5 mm past the 3 mm gap between door and
  jamb, so the bolt is in the strike.
- **Angle B** is where the bolt stands 23 mm out, of its 25 mm.

On the simulated lock A is 33 degrees and B is 84 degrees. A real lock gets its own two angles, found with the door
open (next section).

- **1.0:** at the end, the turn is at least B with the gap at most 3 mm, and for the last 1 s (20 control steps) the
  gap has not moved more than 0.5 mm, nor the turn more than 2 degrees, between control steps. The episode ends as a
  success once the score has been 1.0 for 1 s, so after 2 s locked and still.
- **0.75:** the turn was at some point at least B with the gap at most 3 mm.
- **0.5:** the turn was at some point at least A with the gap at most 3 mm (the bolt in the strike).
- **0.25:** the gap was at some point at most 2 mm (the door pressed shut).
- Otherwise 0.

Geometrically, a thrown bolt lets the door stand at most about 1.9 mm off. So a door let go after locking, pushed out
by the weatherstrip until the bolt bears on the strike, still counts as shut.

A bolt thrown with the door open does not count: with a gap of 3.5 mm or more the bolt cannot be in the strike, and
once out it stops the door 2 mm or more short of shut. Nothing in the scene can be damaged or removed, so there is no
failure state.

The simulator can see the bolt, and the scorer also requires it to be where the angle puts it: 8 mm out at A and
23 mm out at B. On a working lock that is always so. The check only stops a thumb-turn that had come adrift of its
bolt from scoring.

## Scoring a real trial

Tools: a steel ruler, a protractor (or a phone photo taken square-on to the door), and a video of the trial. This
procedure is written from the lock's geometry and the simulated runs. It has not been tried on a physical door.

**Find angles A and B once per lock,** before the trials, with the door open so the bolt is in view:

1. Start with the thumb-turn upright and the bolt flush with the faceplate.
2. Turn the thumb-turn slowly toward locked. Stop when the bolt's end stands 8 mm out of the faceplate, by the ruler
   held along the bolt. Read the wing's angle from upright.
3. Go back to upright and repeat twice more. A is the largest of the three readings, rounded up to a whole degree.
4. Do the same for 23 mm. That gives B.
5. Optional: put a pencil mark on the rose at the wing's top end for A and for B, so both read at a glance in a video.

Always come from the upright side. A lock's linkage has some play, and turning toward locked the bolt trails the
thumb-turn. Found that way, a thumb-turn at A or beyond has its bolt at least 8 mm out, and at B or beyond at least
23 mm out.

The simulated lock was read the same way. The slow pass above turns the thumb-turn a quarter turn in 30 s and writes
the bolt's travel against the angle to `trace.csv`:

| Thumb-turn (degrees) | Bolt out of the door's edge (mm) |
|---|---|
| 10.6 | 1.9 |
| 20.6 | 4.5 |
| 30.8 | 7.5 |
| 32.5 | 8.1 (the first reading at or past 8 mm) |
| 40.9 | 10.7 |
| 51.1 | 13.7 |
| 61.2 | 16.9 |
| 71.2 | 19.7 |
| 81.2 | 22.4 |
| 83.7 | 23.0 (the first reading at or past 23 mm) |
| 89.4 | 24.3 |

Rounded up, that gives A = 33 degrees and B = 84 degrees. The lock's geometry puts the two points at 31.9 and 83.5
degrees; the simulated contacts give a little.

**Read the gap** as a step. Hold the ruler against the latch jamb's inside face and measure back to the door's inside
face, at the latch edge. On this frame the step is 28 mm with the door pressed fully shut and 16 mm at the start. The
gap is the step with the door shut minus the step now. For the video, stick a millimetre scale (a strip cut from a
paper ruler) on the jamb's side at the bolt's height, running back from the jamb's inside face. The door's inside
face then reads the step straight off the scale.

**Score the trial:**

1. Read the turn and the gap at the end of the trial, from the last video frame or on the fixture itself. (A
   thumb-turn stays where it is left, and a locked door stays within 2 mm of shut when the arms let go.) If the turn
   is at least B and the gap at most 3 mm, check the last second of the video: the wing and the door's edge must be
   at rest (no more than 2 degrees and 0.5 mm between frames 50 ms apart). That is 1.0.
2. Otherwise go through the video for the best state reached: turn at least B with the gap at most 3 mm (0.75), turn
   at least A with the gap at most 3 mm (0.5), or the gap at most 2 mm (0.25).

On the simulated lock B is 6 degrees short of the full quarter turn, and the demo ends at 91.5 degrees with the gap
at 0.0 mm. So reading the wing to 2 degrees and the gap to 0.5 mm is enough.

A cross-check, not part of the score: after a trial that ended locked, pull the door's latch edge toward the room by
hand. A thrown bolt holds the door within 2 mm of shut.

## Runtime and seeds

Expect about 5 minutes for a full 1,200-step Dream-1 trial on an L40S, including about 1.5 minutes for Isaac Sim to
start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Fidelity

Rigid boxes and cylinders at real size. The door is a 53 x 45 cm section of a full door, hung in a matching frame
section, so its mass (6.3 kg) is about a fifth of a full exterior door's. The weatherstrip is a sprung strip, not a
compressible bulb. The bolt's mechanism is a crank pin in a slot in the bolt's tail, a simple form of a deadbolt's cam.
The lock's friction is lumped at the thumb-turn (0.25 N*m), with light damping on the bolt. The deadlocking of a real
deadbolt, which stops the bolt being pushed back, is left off. The hinges are a frictional pivot with visual
knuckles, without leaves or screws. Grippy contact friction from the station's material. No contact force cap is
enforced in simulation.

## Real-world build

Use a pre-hung exterior door (or a door section in a bench frame) with a standard 60 mm backset deadbolt, the latch
removed or its strike unscrewed, and a compression weatherstrip that holds the door 12 mm open at the latch edge. The
trial is safe. Score it as in [Scoring a real trial](#scoring-a-real-trial): nothing has to be measured on the bolt
once the door is locked.

## Assets

All geometry is defined in `task.py` from boxes and cylinders; there are no external assets.
