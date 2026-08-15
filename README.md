# blackjack-rl

A full-rules blackjack engine, a Gymnasium environment with **bet sizing and card counting**,
and RL agents that learn to play (and bet) — plus a browser (and terminal) game so you can play
the exact same environment yourself.

![Blackjack RL web UI — split hand mid-decision with the advisor panel](docs/screenshot-play.png)

<details>
<summary>More screenshots</summary>

![A natural — result banner and per-hand result tags](docs/screenshot-result.png)

</details>

```
blackjack_rl/
├── engine/       pure-Python game engine (no RL deps): shoe + Hi-Lo count, hands, rules, round state machine
├── env/          Gymnasium env  "BlackjackFull-v0"  (bet phase → play phase, action mask, persistent shoe)
├── agents/       random, basic strategy (+ Hi-Lo bet spread), masked Double-DQN (PyTorch)
├── cli/          blackjack-play / blackjack-train / blackjack-eval / blackjack-strategy
├── web/          blackjack-web: browser UI (stdlib HTTP server + vanilla JS) on top of the env
└── evaluation.py EV / variance / win-rate / bet-by-count statistics
tests/            pytest suite for the engine, env and baselines
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Play it yourself

### In the browser (recommended)

```bash
blackjack-web                              # opens http://127.0.0.1:8000
blackjack-web --checkpoint checkpoints/dqn.pt --h17 --bets 5,10,25 --port 9000
```

A casino-style table served straight from the RL environment: click chips to bet, keyboard
shortcuts (`1-9` bet, `H/S/D/P/R` play, `Enter` next round), split hands, hole-card reveal,
running/true count and shoe penetration, and an **advisor panel** showing what basic strategy,
Hi-Lo betting and the trained DQN would do (with the DQN's Q-values per action). **Autoplay** lets
any agent (basic / Hi-Lo / DQN) play the table while you watch. ⚙ Table opens a form to start a
new table with different rules, bet sizes and bankroll. Zero front-end dependencies — a small
stdlib HTTP server (`blackjack_rl/web`) + vanilla HTML/CSS/JS.

### In the terminal

```bash
blackjack-play                # 6 decks, S17, DAS, late surrender, bets 1/2/4/8
blackjack-play --hint         # shows the basic-strategy play and Hi-Lo bet suggestion
blackjack-play --h17 --decks 2 --bets 5,10,25 --hide-count
```

Both front-ends drive `BlackjackEnv` directly: you see the same observation the agent gets
(rendered as a table), pick from the same legal actions, and get the same reward.

## The environment

```python
import gymnasium as gym
import blackjack_rl                       # registers the env

env = gym.make("BlackjackFull-v0")        # or BlackjackEnv(rules=Rules(...), bet_sizes=(1, 2, 4, 8))
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample(mask=info["action_mask"])   # always mask!
    obs, reward, done, truncated, info = env.step(action)
print(info["profit"], info["results"])
```

* **Episode = one round.** It starts in the *bet phase* (actions `5..5+len(bet_sizes)-1` pick a bet size),
  then continues in the *play phase* with `0 STAND, 1 HIT, 2 DOUBLE, 3 SPLIT, 4 SURRENDER` for each hand.
* **The shoe persists across episodes** (reshuffled at the cut card), so the running/true count in the
  observation carries real information. Pass `reshuffle_each_round=True` to disable counting.
* **Observation** (22 floats in [-1, 1]): phase, hand total, soft/pair flags, legal-action flags,
  dealer up-card one-hot, true count, fraction of shoe left, current bet, number of hands. See `env/observation.py`.
* **Reward:** 0 until the round ends, then the round's profit in bet units (+1.5 for a natural on 1 unit,
  −4 for a lost doubled 2-unit bet, ...).
* **Illegal actions raise** `ValueError` — use `info["action_mask"]` / `info["legal_actions"]`.
* Rules are a frozen dataclass (`Rules`): decks, penetration, S17/H17, blackjack payout, peek, DAS,
  double restrictions, max splits, RSA, hit split aces, late surrender.

## Baselines and evaluation

```bash
blackjack-eval --agent basic --rounds 1000000        # basic strategy, min bet
blackjack-eval --agent hilo  --rounds 1000000        # basic strategy + 1-2-4-8 Hi-Lo bet spread
blackjack-eval --agent random --rounds 100000
blackjack-strategy --agent basic                     # print the basic-strategy tables the agent is judged against
```

Sanity check: with the default rules basic strategy measures **−0.40 % ± 0.22 % of the initial bet
over 1M rounds**, matching the published house edge for 6D / S17 / DAS / LS.

## Basic strategy reference

`blackjack-strategy --agent basic` prints (and, with `--open`, renders) the chart the agents are
measured against — here for the default table (6 decks, S17, DAS, late surrender):

```
Legend: S=stand H=hit D=double P=split R=surrender   (2-card hands; dealer up-card across the top)

HARD          2   3   4   5   6   7   8   9   T   A
5              H   H   H   H   H   H   H   H   H   H
6              H   H   H   H   H   H   H   H   H   H
7              H   H   H   H   H   H   H   H   H   H
8              H   H   H   H   H   H   H   H   H   H
9              H   D   D   D   D   H   H   H   H   H
10             D   D   D   D   D   D   D   D   H   H
11             D   D   D   D   D   D   D   D   D   H
12             H   H   S   S   S   H   H   H   H   H
13             S   S   S   S   S   H   H   H   H   H
14             S   S   S   S   S   H   H   H   H   H
15             S   S   S   S   S   H   H   H   R   H
16             S   S   S   S   S   H   H   R   R   R
17             S   S   S   S   S   S   S   S   S   S
18             S   S   S   S   S   S   S   S   S   S
19             S   S   S   S   S   S   S   S   S   S

SOFT          2   3   4   5   6   7   8   9   T   A
A,2            H   H   H   D   D   H   H   H   H   H
A,3            H   H   H   D   D   H   H   H   H   H
A,4            H   H   D   D   D   H   H   H   H   H
A,5            H   H   D   D   D   H   H   H   H   H
A,6            H   D   D   D   D   H   H   H   H   H
A,7            S   D   D   D   D   S   S   H   H   H
A,8            S   S   S   S   S   S   S   S   S   S
A,9            S   S   S   S   S   S   S   S   S   S

PAIRS         2   3   4   5   6   7   8   9   T   A
2,2            P   P   P   P   P   P   H   H   H   H
3,3            P   P   P   P   P   P   H   H   H   H
4,4            H   H   H   P   P   H   H   H   H   H
5,5            D   D   D   D   D   D   D   D   H   H
6,6            P   P   P   P   P   H   H   H   H   H
7,7            P   P   P   P   P   P   H   H   H   H
8,8            P   P   P   P   P   P   P   P   P   P
9,9            P   P   P   P   P   S   P   P   S   S
T,T            S   S   S   S   S   S   S   S   S   S
A,A            P   P   P   P   P   P   P   P   P   P
```

`D` falls back to hit (hard 9–11, soft ≤ 17) or stand (soft 18) when doubling isn't allowed, `R` falls
back to hit — the same fallback logic `BasicStrategyAgent` uses. `--h17`, `--no-das`, `--no-surrender`,
`--decks N` etc. regenerate the chart for other tables; `--agent hilo` adds the Hi-Lo bet spread.

![Basic strategy report](docs/screenshot-basic-strategy.png)

## Train the DQN

```bash
blackjack-train --rounds 400000 --out checkpoints/dqn.pt
blackjack-eval --agent dqn --checkpoint checkpoints/dqn.pt --rounds 200000
blackjack-strategy --agent dqn --checkpoint checkpoints/dqn.pt     # learned tables vs basic strategy + bet spread
blackjack-strategy --agent dqn --checkpoint checkpoints/dqn.pt --html reports/dqn.html --open
```

`--html PATH` (or `--open`) writes a self-contained, light/dark-aware **HTML report**: colour-coded
hard/soft/pairs charts with every disagreement vs basic strategy outlined (hover a cell for the
Q-values), agreement tiles, the bet spread by true count against the Hi-Lo reference, and the
Q-value table per bet size. The same report is one click away in the browser game (📊 Strategy).

![Strategy report](docs/screenshot-strategy.png)

The agent is a masked **Double DQN** (`agents/dqn.py`): one MLP outputs Q-values for all 5 play
actions + all bet sizes; the mask picks which are valid in the current phase. Rewards are scaled by
`1/max_bet` for training. Because a round's outcome is very noisy compared with the EV differences
between actions (often < 1 % of the bet), learning the fine details of basic strategy — and especially
the bet spread — takes millions of rounds; `--rounds`, `--lr`, `--batch-size`, `--train-every`,
`--eps-decay-frac` and `--resume` are the knobs. `--bets 1` trains a flat-betting player if you only
care about the playing decisions.

Reference points (6D S17 DAS LS, bets 1/2/4/8, M2 Max CPU — the network is far too small for a
GPU to help, `--device mps` is slower):

| run | wall time | EV / unit wagered | agreement with basic strategy (hard / soft / pairs) | bet spread |
|---|---|---|---|---|
| basic strategy + Hi-Lo (benchmark) | – | ≈ −0.3 % … +0.3 % | 100 % | 1 → 8 |
| DQN, 400k rounds, batch 256 | 10 min | −4.6 % | 63 / 54 / 55 % | none |
| DQN, 3M rounds, batch 1024, `--train-every 8` | 36 min | **−1.2 %** | **79 / 65 / 62 %** | partial: ~2 units at TC ≤ −4, 4 units around 0, 8 units at TC ≥ +6 |

So the agent is clearly learning (it hits/stands correctly on almost every hard total and has started
to size bets with the count) but is still short of basic strategy on doubles, soft hands and pairs —
the rarest states with the smallest EV differences. More rounds, `--resume` from the last checkpoint,
and a lower final learning rate are the obvious next levers.

## Ideas / next steps

* Insurance as an action (becomes +EV at true count ≥ +3), bankroll in the observation, table limits
* PPO / policy-gradient agent, vectorised envs for faster training
* Multi-player table (other players' cards feed the count)
* A GUI / web front-end on top of the same env
