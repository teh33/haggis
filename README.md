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
- an end-to-end experiment runner.

## Quick start

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Run a fixed-hand tournament:

```bash
python3 -m haggis.tournament --bot-a point-aware --bot-b bomb-control --hands 100 --seed 1 --output-json runs/tournament.json
```

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

In the larger 12-hand validation run, `policy-rollout` ranked first with a 67.3%
hand win rate and +6756 score margin over 168 hands. A smaller 8-hand validation
run also ranked `policy-rollout` first at 71.4% over 112 hands. The matching
benchmark at this budget measured `policy-rollout` at about 0.073s/decision,
compared with 0.012s for the direct `policy` bot and 0.751s for
`information-set`.

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

1. Mature the shallow `tree-information-set` prototype into fuller ISMCTS with deeper reusable child nodes.
2. Run even larger ladders before treating `models/linear_policy.json` as a release-strength model.
3. Add more evaluation runs against stronger/search opponents and tune policy-rollout/tree-search budgets.
4. Continue profiling sequence generation if larger ladders need more throughput.
