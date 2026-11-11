# What we're actually doing: reinforcement learning, explained with Sumo

A refresher on reinforcement learning (RL), mapped line by line to this project.
🧒 boxes are the "explain like I'm 5" versions.

---

## 1. The big picture

> 🧒 **ELI5:** Imagine teaching a dog a trick without being able to speak. You can't explain
> anything. You can only give a treat when it does something good and say "no" when it does
> something bad. The dog tries random things, remembers what earned treats, and slowly does
> more of those things. That's reinforcement learning. Our fighters are the dog, and +1 / −1
> are the treat and the "no".

There are **no examples to copy**. Nobody shows the AI how to do sumo. It only gets a score and
has to figure out by itself what leads to a good score. That's the difference from supervised
learning ("here are 10,000 photos labelled cat/dog").

### The loop that repeats millions of times

```
        ┌──────────────── observation (13 numbers) ─────────────────┐
        │                                                            ▼
   ┌─────────┐        action (2 numbers: drive, turn)        ┌──────────────┐
   │  Unity  │ ◀──────────────────────────────────────────── │ Brain (neural │
   │ (world) │                                               │   network)   │
   └─────────┘ ─────────────── reward (+1 / −1 / small −) ──▶ └──────────────┘
                                                                     │
                                         every ~20,000 experiences: │
                                         "learn" = adjust the brain ◀┘
```

| RL term | Meaning | In our sumo |
|---|---|---|
| **Agent** | The one making decisions | A fighter (`SumoAgent.cs`) |
| **Environment** | The world it acts in | The ring and the physics (Unity) |
| **Observation (state)** | What the agent perceives | 13 numbers (positions, speeds…) |
| **Action** | What it can do | 2 numbers: drive, turn |
| **Reward** | The score signal | +1 win, −1 loss, a tiny − each moment |
| **Episode** | One attempt, start to finish | One round of sumo |
| **Policy** | The strategy: observation → action | The neural network (the "brain") |

### Why 12 arenas?

The training scene has **12 arenas** (a 4 × 3 grid, not 12 × 12). Each one is a separate match,
and all 12 feed the **same brain**, so it gets 12× more practice per second. RL is very
data-hungry: the AI needs millions of attempts.

---

## 2. The code, piece by piece

### 2a. Observations: what the fighter "sees" (`SumoAgent.CollectObservations`)

> 🧒 The fighter has no eyes. It doesn't see the 3D image. It receives a short list of numbers,
> like a robot reading its sensors.

```csharp
Vector3 toCenter   = ToLocal(center - transform.position) / r;      // where is the ring centre?
Vector3 myVel      = ToLocal(GetVelocity(rb)) / maxSpeed;          // how am I moving?
Vector3 toOpponent = ToLocal(opponent.position - position) / (2r); // where is the enemy?
Vector3 oppVel     = ...                                            // how is the enemy moving?
Vector3 oppFacing  = ...                                            // is the enemy facing me?
EdgeDistance01(me), EdgeDistance01(opponent)                         // how close to the edge are we?
rb.angularVelocity.y / 10f                                           // am I spinning?
```

Two design choices matter a lot:

1. **Everything is relative to the fighter (`ToLocal`).** "The enemy is 2 m ahead, slightly to
   my left" is easier to learn than "the enemy is at world position (13.4, 1, −7.2)". In the
   relative version the same situation gives the same numbers wherever the ring is and whichever
   way the fighter faces, so a lesson learned once applies everywhere.
2. **Every value is divided by something** (`/ r`, `/ maxSpeed`) to keep it roughly between −1
   and 1. Neural networks learn badly when one input is 0.01 and another is 500. The yaml line
   `normalize: true` adds automatic rescaling on top.

**Stacked Vectors = 3:** the brain receives the last 3 observations (3 × 13 = 39 numbers).

> 🧒 A single photo doesn't tell you whether a ball is moving towards you or away; three photos
> in a row do. It's a tiny short-term memory.

### 2b. Actions: what it can do (`SumoAgent.OnActionReceived`)

```csharp
float move = actions.ContinuousActions[0];           // -1 = full reverse ... +1 = full forward
float turn = actions.ContinuousActions[1];           // -1 = turn left    ... +1 = turn right
rb.MoveRotation(... turn * turnSpeed ...);           // rotate
rb.AddForce(transform.forward * move * moveForce);   // push
```

These are **continuous** actions: any value in between is allowed, like a joystick rather than
buttons. The alternative is **discrete** actions, choosing from a menu like {forward, left,
right, nothing}.

**Decision Requester, Period = 5:** physics runs 50 times a second, but the brain decides only
every 5th step (10 times a second) and repeats its last action in between.

> 🧒 Humans don't change their mind 50 times a second either. Fewer decisions means less work and
> each decision has a more visible effect, which makes learning easier.

### 2c. Rewards: how it knows what's good (`SumoArena.cs`)

```csharp
if (bOut) { winner.SetReward(1f); loser.SetReward(-1f); }   // the result
...
float p = -timePenalty / maxEnvironmentSteps;              // -0.5 / 2000 = -0.00025
agentA.AddReward(p); agentB.AddReward(p);                  // tiny cost each physics step
```

- **+1 / −1** is the real goal. It's a **sparse** reward: it only arrives at the end of a round.
- The **time penalty** totals −0.5 if the round lasts the full 40 seconds.
  🧒 It's a small "hurry up". Without it, "both stay still forever" costs nothing, and the AI
  could get stuck there.
- **Deliberately not added:** rewards like "move towards the enemy" or "stay near the centre".
  That kind of hint is called **reward shaping**. It speeds learning up, but the AI optimises
  exactly what you reward, not what you meant. Reward "get close to the enemy" and it might learn
  to hug the enemy without ever pushing. Keeping the reward close to the real goal lets the AI
  invent its own tactics, which is the surprising part that makes good videos.

> ⚠️ **The golden rule of RL:** the agent optimises the reward, not your intention. Every funny
> "AI cheats" moment on YouTube is really a reward that didn't say exactly what the creator meant.

### 2d. Episodes (`EndEpisode`, `EpisodeInterrupted`)

- A fighter falls out: `EndEpisode()`. The round is truly over and the result is known.
- 40 seconds pass: `EpisodeInterrupted()`.
  🧒 This tells the trainer "the game didn't finish, we just stopped it." The trainer then
  estimates what would have happened next, instead of pretending the future was worth 0
  (see section 3).

---

## 3. How the brain learns: PPO

### 3a. The brain itself

The **policy network** takes 39 numbers in, passes them through 2 layers of 256 neurons
(`hidden_units: 256`, `num_layers: 2`), and outputs 2 actions. For continuous actions it doesn't
output one exact action. It outputs a **mean and a spread** (a bell curve), and the action is
**sampled** from it.

> 🧒 Early on the spread is wide: "I'm not sure, let's try lots of things." As it learns, the
> spread narrows: "I know what works." That's how exploration happens.

There's also a second network, the **critic** (or *value network*). It doesn't choose actions.
It predicts: **"from this situation, how much reward will I probably get from now on?"**

### 3b. The key ideas, step by step

**① Return: what an action led to.** It's the sum of future rewards, with distant ones counting
less:

  G = r₁ + γ·r₂ + γ²·r₃ + …  with **γ (gamma) = 0.99**

> 🧒 A treat now is worth more than a treat later. With 0.99 per decision (10 decisions a
> second), the AI effectively "looks ahead" about 100 decisions, roughly 10 seconds. That's
> enough to connect "I charged at the edge" with "I won 3 seconds later".

**② Advantage: was the action better or worse than usual?**

  Advantage = (what actually happened) − (what the critic expected)

> 🧒 You expected a 5/10 on the test and got 8/10: whatever you did was better than usual, so do
> more of it. You got 2/10: do less of it.

Judging against *expected* instead of *absolute* reward makes learning much more stable.
ML-Agents computes this with **GAE** (`lambd: 0.95`), a blend of short-term and long-term
estimates.

**③ The update.** Push the policy towards actions with positive advantage and away from those
with negative advantage.

### 3c. The loss

The "loss" is the number the training tries to **make smaller**. In PPO it has three parts:

  **Total loss = Policy loss + 0.5 × Value loss − β × Entropy**

| Part | What it measures | 🧒 ELI5 |
|---|---|---|
| **Policy loss** | How the strategy should change, weighted by advantage | "Do more of what worked better than expected." |
| **Value loss** | Error of the critic: (predicted − actual return)² | "The judge practises predicting scores more accurately." |
| **Entropy bonus** (β = `beta: 0.005`) | How random or exploratory the policy still is | "Don't become a know-it-all too early; keep trying new things." Subtracting it rewards staying a bit random. |

**What the "Proximal" in PPO means:**

```
ratio       = (new probability of the action) / (old probability)
policy loss = −min( ratio × Advantage,  clip(ratio, 1−ε, 1+ε) × Advantage )    with ε = epsilon = 0.2
```

> 🧒 You learn something from one game. Rather than rewriting your entire strategy because of it
> (maybe it was luck), you only adjust by **at most about 20%** per update. The clipping is that
> safety limit. Before PPO, algorithms often took one giant step, broke the strategy, and never
> recovered.

### 3d. One training cycle in ML-Agents (yaml settings)

1. Play and collect **20,480 experiences** (`buffer_size`). Each experience is observation,
   action, reward.
2. Compute returns and advantages.
3. Shuffle them, cut them into **batches of 2,048** (`batch_size`), and do a learning step
   (gradient descent, `learning_rate: 0.0003`) on each.
4. Repeat step 3 for **3 passes** (`num_epoch: 3`). That's 10 batches × 3 = 30 small learning
   steps.
5. Throw the experiences away (PPO only learns from fresh data from the current strategy) and go
   back to step 1.

`time_horizon: 1000`: if a round runs long, the trainer cuts it into chunks of 1,000 experiences
and lets the critic estimate the rest.

---

## 4. Self-play: why two copies of the same brain

Both fighters have the **same Behavior Name (`Sumo`)**, so they're trained as one brain. But
**Team Id 0 and Team Id 1** tell ML-Agents they are opponents. That turns on **self-play**.

> 🧒 You learn chess by playing against yourself. When you get better, your opponent gets better
> too, so the challenge always matches your level. And you also play against old versions of
> yourself, so you don't forget how to beat older strategies.

| yaml | Meaning |
|---|---|
| `save_steps: 50000` | Every 50k steps, save a snapshot of the current brain into a pool of past opponents |
| `window: 10` | The pool keeps the last 10 snapshots |
| `swap_steps: 10000` | Change opponent every 10k steps |
| `play_against_latest_model_ratio: 0.5` | Half the time face the most recent self, otherwise a random older one |
| `team_change: 200000` | Every 200k steps the learning side switches (blue ↔ red), so both sides train |

**Why the final reward has to be exactly +1/−1/0:** self-play reads the **last reward of a
round** to decide who won, then updates an **ELO** score, the same rating system used in chess.

**Why "Mean Reward" stays around 0 in self-play:** whenever one side wins (+1), the other loses
(−1), so the average is about zero even when both are improving. That's why **ELO** is the graph
to watch, not reward.

---

## 5. Reading TensorBoard

```bash
tensorboard --logdir results
```
Then open http://localhost:6006.

| Graph | Healthy shape | 🧒 Meaning |
|---|---|---|
| **Self-play / ELO** | Goes up steadily | "Am I beating my old selves?" This is the main one |
| **Episode Length** | Often drops (quicker ring-outs), may rise again later (better defence) | How long rounds last |
| **Policy / Entropy** | Slowly goes down | Becoming less random, more confident |
| **Losses / Value Loss** | Goes up then down; noisy | The critic's prediction error |
| **Losses / Policy Loss** | Small and noisy, hovering near 0 | Normal; not a progress meter |
| **Cumulative Reward** | Around 0 (self-play!) | Not useful here |

> ⚠️ Unlike supervised learning, **loss going down doesn't mean the AI is getting better**. The
> data changes as the AI changes, so the loss mostly shows whether training is stable. Progress
> shows in **ELO** and in watching the fighters.

---

## 6. Experiments to try (each makes good video content)

1. **Remove the time penalty** (`timePenalty = 0` on the arenas) and see whether they become
   passive.
2. **Add shaping** (+0.001 per step for moving towards the opponent) and watch them learn faster
   but possibly dumber.
3. **Set `beta` to 0.05** (much more exploration) versus **0.0005** (very little) and compare the
   ELO curves.
4. **Set Stacked Vectors to 1** (no memory) and see whether they can still predict an incoming
   charge.
5. **Old vs new:** load an early checkpoint into blue and the final brain into red.

Use a new `--run-id` for each experiment so TensorBoard shows the curves side by side.
