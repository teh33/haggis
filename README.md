# Haggis AI

A dependency-light Python playground for building strong computer players for
[Haggis](https://thespiel.net/files/haggis.pdf).

The project is intentionally split into small pieces:

- a deterministic rules engine with no UI dependencies;
- baseline, heuristic, search, and learned bots;
- tournament/game/ladder evaluation tools;
- self-play data export for ML experiments;
- a dependency-free linear imitation policy trainer;
- regression tests for rule behavior, legal moves, search, evaluation, and ML data.

## Status

Implemented today:

- two-player Haggis deal, trick, scoring, bomb, wild, betting, and target-score game support;
- optimized legal move generation checked against brute-force oracle tests on small hands;
- invariant validation for card conservation and illegal state detection;
- fixed-hand tournaments, official target-score games, and round-robin ladders;
- JSON ladder metrics export;
- self-play JSONL export with either perfect-information or player-observation records;
- linear imitation-policy training with averaged perceptron and validation metrics;
- an end-to-end experiment runner;
- an interactive player-vs-CPU hand mode.

## Quick start

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Run a fixed-hand tournament:

```bash
python3 -m haggis.tournament --bot-a point-aware --bot-b bomb-control --hands 100 --seed 1 --output-json runs/tournament.json
```

Run an interactive hand against the default CPU:

```bash
python3 -m haggis.play --cpu policy-rollout --search-root-moves 4 --search-rollout-turns 40
```

During your turn, enter a listed move number, `pass` when legal, exact card
names like `3C 3D`, or `q` to quit.

Run an official target-score game:

```bash
python3 -m haggis.tournament \
  --bot-a point-aware \
  --bot-b bomb-control \
  --target-score 250 \
  --max-hands 100 \
  --seed 1 \
  --output-json runs/game.json
```

Run a round-robin ladder and save machine-readable metrics:

```bash
python3 -m haggis.ladder \
  --bots random,greedy,point-aware,bomb-control \
  --hands 20 \
  --seed 1 \
  --output-json runs/ladder.json
```

Run a lightweight benchmark:

```bash
python3 -m haggis.benchmark --bots random,greedy,point-aware,bomb-control --states 5 --seed 1 --output-json runs/benchmark.json
```

Run a release-validation ladder with the included model:

```bash
python3 -m haggis.ladder \
  --bots policy-rollout,policy,point-aware,bomb-control,information-set,tree-information-set,ucb-information-set,greedy \
  --policy-model models/linear_policy.json \
  --hands 8 \
  --seed 20260521 \
  --search-simulations 8 \
  --search-root-moves 4 \
  --search-rollout-turns 40 \
  --output-json runs/release-validation-ladder.json
```

In the current 12-hand validation run for the promoted larger model,
`policy-rollout` ranked first with a 72.0% hand win rate and +8024 score margin
over 168 hands. The matching benchmark at this budget measured `policy-rollout`
at about 0.050s/decision, compared with 0.008s for the direct `policy` bot and
0.432s for `information-set`.

## Bot roster

Available bot names for tournament/ladder commands:

| Bot | Description |
| --- | --- |
| `random` | Uniform random legal move; always bets 0. |
| `greedy` | Sheds the most cards, then prefers lower commitment. |
| `point-aware` | Avoids unnecessary point-card/wild-card donation. |
| `bomb-control` | Conservative heuristic that saves bombs for threats/endgames. |
| `endgame-search` | Perfect-information minimax on small endgame states, heuristic fallback otherwise. |
| `monte-carlo` | Root rollout search using perfect-information playouts. |
| `information-set` | Rollout search that samples opponent-hand/haggis determinizations. |
| `tree-information-set` | Shallow information-set tree search over sampled determinizations; can use `--policy-model` for policy-guided leaf rollouts. |
| `ucb-information-set` | UCB1 root allocation over information-set determinizations. |
| `policy` | Loaded linear action-ranking policy model. |
| `policy-rollout` | Root rollout search using a loaded policy model during simulated playouts. |

Tournaments ask bots for pre-play bets by default. Use `--no-betting` to disable
this for compatibility experiments.

Search bots accept budget flags in tournament and ladder commands. `policy`,
`policy-rollout`, and `tree-information-set` use `models/linear_policy.json` by
default; pass `--policy-model` to load a different trained model:

```bash
--search-simulations 20
--search-root-moves 8
--search-rollout-turns 120
--policy-model models/linear_policy.json
```

A fast strong default uses the included model with policy-guided rollouts:

```bash
python3 -m haggis.tournament \
  --bot-a policy-rollout \
  --bot-b point-aware \
  --hands 20 \
  --search-root-moves 4 \
  --search-rollout-turns 40 \
  --seed 1
```

For stronger but slower play, use root `6` and rollout turns `80`.

## Strategy guide

Haggis is a climbing/shedding game: you win hands by going out, but the score is
mostly decided by captured point cards, leftover-card penalties, and successful
bets. Point cards are `3`, `5`, `7`, and `9` for 1 point each, `J` for 2, `Q`
for 3, and `K` for 5. When a player goes out, they also score 5 points for each
card left in the opponent's hand. Successful bets swing the combined wager: 15,
30, 45, or 60 points depending on whether one or both players bet and for how
much. Good play balances tempo, point control, and bomb timing.

### Core priorities

1. **Go out before the opponent.** Emptying your hand ends the hand and scores 5
   points for every card left in the opponent's hand. Moves that shed multiple
   cards are valuable when they do not donate too many captured points.
2. **Do not donate points cheaply.** Threes, fives, sevens, nines, jacks, queens,
   and kings carry points. Avoid spending them into tricks you are unlikely to
   win back.
3. **Control the lead.** Leading lets you choose the combination family. If your
   hand has strong pairs, sequences, or bombs, try to regain the lead before your
   opponent can shed freely.
4. **Save bombs for leverage.** Bombs can beat normal combinations and many lower
   bombs. They are strongest when they stop an opponent from going out, recapture
   a point-heavy trick, or let you immediately shed afterward.
5. **Respect hand size.** When the opponent has only a few cards, prefer moves
   that force an awkward response or preserve a bomb/pass option. When you are
   behind on cards, prioritize shedding and avoid long value fights.

### Opening and betting

- Bet only with real strength: multiple wilds, high point density, strong same-rank
  groups, or flexible sequences. If both players bet, the hand winner collects
  the combined wager, so a 15-vs-30 betting round is worth 45 points and a
  30-vs-30 round is worth 60.
- A speculative bet can erase a good hand if you fail to go out first, so avoid
  betting just because you have points.
- Early in the hand, lead low-commitment combinations that reveal little and keep
  your wilds flexible.

### Midgame tactics

- Prefer plays that either shed many cards or preserve future structure. Breaking a
  strong sequence to win a small trick is often bad.
- Passing is fine when the current trick is low value and winning would cost a
  bomb, wild, or high point card.
- If a trick contains many points, winning it can be worth spending a stronger
  combination, especially if it also gives you the lead.
- Track what shapes have been played. If your opponent keeps passing on pairs or
  sequences, lead that shape again when it helps you shed.

### Endgame tactics

- Count both hand sizes every turn. A one-card opponent can go out on almost any
  lead; a two-card opponent may be holding a pair, bomb, or two singles.
- If you can go out immediately, usually do it unless it gives away a catastrophic
  point trick.
- Use bombs defensively to stop an opponent's likely final play, then lead the
  shape that empties or nearly empties your own hand.
- Preserve low singles when possible. They are useful for safely leading after you
  win a trick with a bomb or high combination.

### Playing against the included CPU

The strongest default bot is `policy-rollout`. It is good at tempo and tactical
rollouts, so beat it by avoiding obvious point donations and by forcing awkward
combination families. The simpler heuristic bots are easier to exploit:

- `greedy` overvalues shedding; feed it low-value tricks and punish point dumps.
- `point-aware` protects points but may miss tempo wins; pressure it with hand-size
  leads.
- `bomb-control` holds bombs conservatively; bait passes with medium threats, then
  switch shapes.

For example:

```bash
python3 -m haggis.tournament \
  --bot-a ucb-information-set \
  --bot-b bomb-control \
  --hands 20 \
  --search-simulations 16 \
  --search-root-moves 6 \
  --search-rollout-turns 100 \
  --seed 1
```

## ML workflow

### 1. Export and inspect self-play records

```bash
python3 -m haggis.self_play export \
  --bot-a point-aware \
  --bot-b bomb-control \
  --hands 100 \
  --seed 1 \
  --observation-mode player \
  --output data/self_play.jsonl

python3 -m haggis.self_play summary \
  --input data/self_play.jsonl \
  --output-json data/self_play_summary.json
```

Observation modes:

- `perfect`: records both exact hands and haggis point totals. Useful for debugging.
- `player`: records the acting player's hand, hides opponent hand identities, and hides haggis point totals. This is the safer default for hidden-information policy learning.

Self-play exports include bot pre-play bets by default; pass `--no-betting` to
write zero-bet compatibility records.

### 2. Train a linear imitation policy

```bash
python3 -m haggis.policy train \
  --input data/self_play.jsonl \
  --output models/linear_policy.json \
  --epochs 5 \
  --averaged \
  --validation-fraction 0.2
```

The trainer is dependency-free. It uses deterministic feature extraction from the
visible state/action summaries, supports averaged perceptron weights, and reports
training plus validation accuracy.

### 3. Evaluate the trained policy

```bash
python3 -m haggis.tournament \
  --bot-a policy \
  --bot-b greedy \
  --policy-model models/linear_policy.json \
  --hands 100 \
  --seed 1
```

Or include it in a ladder:

```bash
python3 -m haggis.ladder \
  --bots policy,greedy,point-aware,bomb-control \
  --policy-model models/linear_policy.json \
  --hands 20 \
  --seed 1 \
  --output-json runs/policy_ladder.json
```

### 4. Run the full experiment loop

```bash
python3 -m haggis.experiment \
  --output-dir runs/demo \
  --data-hands 20 \
  --epochs 5 \
  --averaged \
  --validation-fraction 0.2 \
  --eval-hands 10 \
  --ladder-hands 5 \
  --eval-opponents greedy,point-aware,bomb-control \
  --observation-mode player \
  --evaluate-policy-rollout \
  --rollout-simulations 2 \
  --rollout-root-moves 6 \
  --rollout-turns 80 \
  --seed 1
```

This writes:

- `self_play.jsonl` — training records;
- `linear_policy.json` — trained model;
- `metrics.json` — training, direct-policy evaluation, and optional policy-rollout metrics;
- `ladder.json` — optional policy/policy-rollout-vs-baselines ladder ratings when `--ladder-hands` is set.

A larger local run at `runs/strength-family-pruning/` trained on 80 player-observation
hands and evaluated `policy-rollout` against the baselines. In that sample,
`policy-rollout` swept the ladder at 24/24 hand wins with about 0.12s/decision at
`--rollout-root-moves 6 --rollout-turns 80`.

## Architecture map

- `haggis/cards.py` — ranks, suits, deck/deal helpers, point values.
- `haggis/combinations.py` — sets, sequences, bombs, validation, comparison.
- `haggis/engine.py` — game state, legal moves, trick resolution, scoring, invariants.
- `haggis/bots.py` — baseline, heuristic, search, and policy bots.
- `haggis/tournament.py` — fixed-hand tournaments and target-score games.
- `haggis/ladder.py` — round-robin evaluation and JSON metrics.
- `haggis/self_play.py` — JSONL decision-record export.
- `haggis/policy.py` — linear action-ranking policy training/loading.
- `haggis/experiment.py` — end-to-end data/train/evaluate runner.
- `tests/` — behavior, oracle, CLI, evaluation, and ML regression tests.

## Testing notes

The full suite is intentionally broad and may take around 1–2 minutes because it
runs self-play, experiment, and search regression checks:

```bash
python3 -m unittest discover -s tests
```

The legal move generator is protected by brute-force oracle tests for curated and
seeded small hands, including bombs, wilds, and sequences.

## Roadmap

Remaining useful next steps:

1. Mature the shallow `tree-information-set` prototype into stronger ISMCTS; it now has reusable deeper child nodes, but still trails simpler `information-set` in small ladders.
2. Run even larger ladders before treating `models/linear_policy.json` as a release-strength model.
3. Add more evaluation runs against stronger/search opponents and tune policy-rollout/tree-search budgets.
4. Continue profiling sequence generation if larger ladders need more throughput.
