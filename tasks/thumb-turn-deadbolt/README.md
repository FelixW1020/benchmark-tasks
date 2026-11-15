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

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, readouts, `success()` and `score()` |
| `mechkit.py` | Multi-box parts, hinges and spring sliders for yam_arena task files (with round pins added) |
| `demo.py` | The scripted demo |
| `test_scoring.py` | Offline checks: every scoring outcome, and the lock's geometry (start clearances, the crank pin in the bolt's slot all through the turn, the bolt clearing the strike only with the door shut, the door's swing) (no GPU) |

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
DEMO_DEBUG=1 python demo.py task.py debug_out --no-record      # also prints the door gap, bolt, thumb-turn, TCPs
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
step 341 (17 s): door gap 0.0 mm, bolt 24.7 mm out, thumb-turn at 91 degrees.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Measured with a ruler
at the latch edge. The gap is how far the edge stands off its shut position (12 mm at the start). The bolt is how far it
stands out of the door's edge (0 to 25 mm). The bolt is in the strike when the gap is at most 3 mm and the bolt at least
8 mm out, which is 5 mm past the 3 mm gap between door and jamb. Geometrically, a thrown bolt lets the door stand at
most about 1.9 mm off. So a door let go after locking, pushed out by the weatherstrip until the bolt bears on the
strike, still counts as shut.

- 1.0 (success, which must hold for 1 s): the bolt at least 23 mm out with the gap at most 3 mm. Neither may move more
  than 0.5 mm between control steps, for 20 control steps (1 s) in a row.
- 0.75: the bolt was at some point at least 23 mm out with the gap at most 3 mm.
- 0.5: the bolt was at some point in the strike.
- 0.25: the gap was at some point at most 2 mm (the door pressed shut).
- Otherwise 0.

A bolt thrown with the door open does not count: with a gap of 3.5 mm or more the bolt cannot be in the strike, and
once out it stops the door 2 mm or more short of shut. Nothing in the scene can be damaged or removed, so there is no
failure state.

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
trial is safe. Measure the gap at the latch edge and the bolt's projection with a ruler, or from a photo with a scale.

## Assets

All geometry is defined in `task.py` from boxes and cylinders; there are no external assets.
