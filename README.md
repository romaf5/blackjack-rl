# blackjack-rl

A full-rules blackjack engine, a Gymnasium environment with **bet sizing and card counting**,
and RL agents that learn to play (and bet) — plus a terminal game so you can play the exact
same environment yourself.

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

## Train the DQN

```bash
blackjack-train --rounds 400000 --out checkpoints/dqn.pt
blackjack-eval --agent dqn --checkpoint checkpoints/dqn.pt --rounds 200000
blackjack-strategy --agent dqn --checkpoint checkpoints/dqn.pt     # learned tables vs basic strategy + bet spread
```

The agent is a masked **Double DQN** (`agents/dqn.py`): one MLP outputs Q-values for all 5 play
actions + all bet sizes; the mask picks which are valid in the current phase. Rewards are scaled by
`1/max_bet` for training. Because a round's outcome is very noisy compared with the EV differences
between actions (often < 1 % of the bet), learning the fine details of basic strategy — and especially
the bet spread — takes millions of rounds; `--rounds`, `--lr`, `--batch-size`, `--train-every`,
`--eps-decay-frac` and `--resume` are the knobs. `--bets 1` trains a flat-betting player if you only
care about the playing decisions.

Reference point: a first 400k-round run (≈10 min on an M2 Max CPU; the network is far too small for a
GPU to help — `--device mps` is slower) reaches −4.6 % of wager with ~63 % agreement with basic
strategy on hard totals and no usable bet spread yet, i.e. clearly under-trained. Longer runs with
bigger batches (`--rounds 3000000 --batch-size 1024 --train-every 8`) are the next step.

## Ideas / next steps

* Insurance as an action (becomes +EV at true count ≥ +3), bankroll in the observation, table limits
* PPO / policy-gradient agent, vectorised envs for faster training
* Multi-player table (other players' cards feed the count)
* A GUI / web front-end on top of the same env
