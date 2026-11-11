# Sumo AI 🥋

Two AI fighters learn sumo by playing against each other (reinforcement learning + self-play)
in Unity with ML-Agents. Built to be trained on a MacBook Pro M3 Pro.

```
sumo-ai/
├── Assets/Scripts/
│   ├── SumoAgent.cs   # the fighter: senses, actions, manual controls
│   ├── SumoArena.cs   # the referee: ring-outs, rewards, round reset, scoreboard
│   └── Editor/
│       └── SumoSceneBuilder.cs  # "Sumo" menu: builds the whole scene in one click
└── config/
    └── sumo.yaml      # training settings (PPO + self-play)
```

---

## 1. Install (once)

**Python side** (in Terminal):

```bash
conda create -n mlagents python=3.10.12 -y
conda activate mlagents
pip install torch
pip install mlagents
mlagents-learn --help        # should print the help text
```

If `pip install mlagents` fails on `grpcio` or `numpy`, run `conda install grpcio numpy -y` and retry.

**Unity side:**

1. Install **Unity Hub**, then a recent LTS editor (Unity 6 or 2023 LTS, Apple Silicon build).
2. New project → **Universal 3D** template → name it `SumoAI`.
3. *Window → Package Manager → Unity Registry* → search **ML-Agents** → Install.
   The Unity package and the Python `mlagents` package must come from the same
   ML-Agents release (see the release table on the ML-Agents GitHub page). If training
   complains about versions, install the matching Python version, e.g. `pip install mlagents==1.1.0`.
4. Copy the `Assets/Scripts` folder of this repo into your Unity project's `Assets` folder.
5. Copy the `config` folder next to your Unity project (anywhere is fine, you just point to it).

---

## 2. Build the scene

### ⚡ One click (recommended)

In Unity's top menu bar: **Sumo → Build Scene (1 arena - test and record)**.
That's it: ring, fighters, eyes, every ML-Agents setting and the camera are created and the
scene is saved as `Assets/Scenes/Sumo.unity`.
For training, use **Sumo → Build Scene (12 arenas - training)** (saved as `Sumo Training.unity`).

Then skip to **step 3**. The manual steps below explain what the button does, if you're curious
or want to change something.

### By hand (≈ 20 minutes)

Create this hierarchy (*right-click in Hierarchy → 3D Object / Create Empty*):

```
Arena                 (Empty, position 0,0,0)      + SumoArena script
├── Ring              (Cylinder)
├── RingCenter        (Empty)
├── FighterBlue       (Cube)                       + SumoAgent script
└── FighterRed        (Cube)                       + SumoAgent script
```

### Ring
- Position `0, 0, 0`, Scale `10, 0.5, 10` → a disc of radius 5, top surface at y = 0.5.
- ⚠️ Unity's cylinder comes with a **Capsule Collider**, which has a rounded top.
  **Remove it** and add a **Mesh Collider** with **Convex** ticked.
- Give it a material (light colour, e.g. sand).

### RingCenter
- Position `0, 0.5, 0` (centre of the ring's top surface).

### FighterBlue and FighterRed (same setup, different colour)
- Position: Blue `-2.5, 1, 0`, Red `2.5, 1, 0` (the script moves them anyway).
- Add **Rigidbody**: Mass `1`, Interpolate `Interpolate`, Collision Detection `Continuous`.
- Add the **SumoAgent** script. Unity also adds **Behavior Parameters** automatically. Set:
  | Field | Value |
  |---|---|
  | Behavior Name | `Sumo` (exactly, it must match the yaml) |
  | Vector Observation → Space Size | `13` |
  | Stacked Vectors | `3` (gives the AI a short memory of movement) |
  | Actions → Continuous Actions | `2` |
  | Actions → Discrete Branches | `0` |
  | Behavior Type | `Default` |
  | **Team Id** | Blue = `0`, Red = `1` |
- Add **Decision Requester**: Decision Period `5`, *Take Actions Between Decisions* ✅.
- Give each a coloured material (blue / red). Optional: two small white spheres as eyes,
  **with their colliders removed**.

### Arena
- Add the **SumoArena** script and drag in:
  `Ring Center` → RingCenter, `Agent A` → FighterBlue, `Agent B` → FighterRed.
- Leave the defaults (radius 5, 2000 steps = 40 s per round).

---

## 3. Test it yourself first

1. Set **Behavior Type** to `Heuristic Only` on both fighters.
2. Press **Play**. Blue = **W A S D**, Red = **arrow keys**.
3. Push one out → the round should reset immediately. Tick *Show Scoreboard* on the arena
   to see the score.
4. Set both back to **Default** when done.

If the keys do nothing on Unity 6: *Edit → Project Settings → Player → Active Input Handling*
→ `Both`.

---

## 4. Train

### Quick start (inside the Editor)

1. Open the training scene (**Sumo → Build Scene (12 arenas - training)**).
   More arenas = more fights per second = faster learning.
   (Building by hand? Turn **Arena** into a prefab and duplicate it 12 times, 20 units apart.)
3. In Terminal:
   ```bash
   conda activate mlagents
   cd path/to/folder/containing/config
   mlagents-learn config/sumo.yaml --run-id=sumo1
   ```
4. When it says *"Start training by pressing the Play button"*, press **Play** in Unity.

### Faster (recommended for long runs on the M3 Pro)

1. *File → Build Settings* → macOS → Build → save as `Builds/Sumo.app`.
2. Train without graphics on 5 copies at once:
   ```bash
   mlagents-learn config/sumo.yaml --run-id=sumo1 --env=Builds/Sumo.app --num-envs=5 --no-graphics
   ```
3. Keep the Mac plugged in. Stop any time with **Ctrl+C**, and resume with `--resume`.

### Watch progress

```bash
tensorboard --logdir results
```
Open http://localhost:6006. With self-play, the key graph is **ELO**: it should keep rising.
(Cumulative reward hovers around 0, which is normal when the AI plays itself.)

Rough guide on an M3 Pro: they start pushing after ~500k steps, and real tactics
(dodges, edge tricks) appear somewhere in the 2–5M range.

---

## 5. Use the trained brains (and film the video)

Results are saved in `results/sumo1/Sumo/`:
- `Sumo-250000.onnx`, `Sumo-500000.onnx`, … are checkpoints, i.e. the "generations".
- `results/sumo1/Sumo.onnx` is the final brain.

To watch one: drag the `.onnx` into your Unity `Assets`, then on a fighter set
**Model** to that file and **Behavior Type** to `Inference Only`.

🎬 Video ideas:
- **Gen 1 vs Gen 1**: give both fighters an early checkpoint to show the chaos.
- **Old vs new**: Blue with `Sumo-250000`, Red with the final brain. Who wins?
- **You vs the AI**: Blue on `Heuristic Only` (you), Red on the final brain.
- **Upgrade episode**: make the ring smaller, add obstacles, or change a fighter's mass,
  then retrain with `--initialize-from=sumo1 --run-id=sumo2`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Couldn't connect to trainer` | Start `mlagents-learn` first, then press Play within ~60 s. |
| Fighters fly or bounce strangely | Ring must use a **convex Mesh Collider**, not the capsule. |
| `Behavior Name` warning / nothing learns | Name must be exactly `Sumo` on both fighters. |
| Observation size warning | Space Size must be `13`. |
| ELO stays flat | Check Team Ids are 0 and 1, and that both use Behavior Type `Default`. |
| They just spin in place for a long time | Normal early on. Give it ~1M steps. |
