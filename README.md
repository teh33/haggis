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
  --target-score 350 \
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

In the extended 20-hand validation run for the promoted mixed-teacher model,
`policy-rollout` ranked first with a 73.2% hand win rate and +14603 score margin
over 280 hands. The matching 50-state benchmark at this budget measured
`policy-rollout` at about 0.036s/decision, compared with 0.006s for the direct
`policy` bot and 0.381s for `information-set`. Artifacts are under
`runs/promoted-candidate-002-validation/`.

## Champion/challenger promotion gates

Use a fixed-seed promotion gate before replacing the default model. Small gates
are useful for smoke tests only; require a larger confirmation block before
promotion because 20-game gates have produced false positives in local
experiments. The torch gate CLI accepts comma-separated seeds and inclusive seed
ranges such as `7600:7659`:

```bash
python3 -m haggis.promotion \
  --champion models/linear_policy.json \
  --challenger runs/candidate/linear_policy.json \
  --output-dir runs/promotion/candidate \
  --hands 12 \
  --seed 100 \
  --search-root-moves 4 \
  --search-rollout-turns 40
```

For PyTorch policy candidates, use a larger champion-vs-challenger confirmation
gate before copying a candidate over the current champion:
```bash
python3-torch -m haggis.torch_gate \
  --champion runs/torch-champions/current.pt \
  --challenger runs/candidate/candidate.pt \
  --output-dir runs/candidate/gate-confirm-60g \
  --seeds 7600:7659 \
  --target-score 350 \
  --max-hands 30 \
  --search-root-moves 3 \
  --search-rollout-turns 16 \
  --require-wins 31
```
Treat a 20-game pass as a screening signal, not promotion evidence. In May 2026,
self-play candidates that passed 20-game torch gates later failed 60-game
confirmation blocks, so promotion should require a positive larger gate before
updating `runs/torch-champions/current.pt`.


## CPU ratings and improvement tracking

`python3 -m haggis.ladder` maintains an Elo ladder for CPU-vs-CPU comparison.
Each ordered matchup updates ratings from the hand win rate in that match, using
an initial rating of 1500 and configurable `--k-factor` (default 32). JSON output
includes the rating system metadata, standings, per-bot entries, and each match's
before/after ratings, so future model or bot changes can be compared over time.

Example:

```bash
python3 -m haggis.ladder \
  --bots policy-rollout,policy,point-aware,bomb-control,information-set,greedy \
  --policy-model models/linear_policy.json \
  --hands 12 \
  --search-root-moves 4 \
  --search-rollout-turns 40 \
  --output-json runs/ratings/current.json
```

## Bot roster

Available bot names for tournament/ladder commands:

| Bot | Description |
| --- | --- |
| `random` | Uniform random legal move; uses the default hand-strength betting heuristic. |
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

## Rules audit

A source-backed implementation audit is available in
[`docs/rules-audit.md`](docs/rules-audit.md). It compares the implementation
against the published two-player rules and notes remaining confidence risks.

## Strategy guide

Haggis is a climbing/shedding game: you win hands by going out, but the score is
mostly decided by captured point cards, leftover-card penalties, and successful
bets. Point cards are `3`, `5`, `7`, and `9` for 1 point each, `J` for 2, `Q`
for 3, and `K` for 5. When a player goes out, they also score 5 points for each
card left in the opponent's hand, plus any point cards left in that opponent's
hand and any point cards in the haggis. Successful bets swing the combined wager: 15,
30, 45, or 60 points depending on whether one or both players bet and for how
much. Good play balances tempo, point control, and bomb timing.

### Core priorities

1. **Go out before the opponent.** Emptying your hand ends the hand and scores 5
   points for every card left in the opponent's hand, plus any unplayed point
   cards in the opponent hand and haggis. Moves that shed multiple
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

To generate stronger AlphaGo-style policy targets, use search-guided teachers and
record the teacher metadata in each JSONL row:

```bash
python3 -m haggis.self_play export \
  --bot-a policy-rollout \
  --bot-b policy-rollout \
  --policy-model models/linear_policy.json \
  --search-root-moves 4 \
  --search-rollout-turns 40 \
  --hands 100 \
  --seed 2 \
  --observation-mode player \
  --output data/search_improved.jsonl
```

Those records include `dataset_source: "search_improved"` and a `teacher` object
with the bot/model/search settings that produced the selected action.

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

You can also train a value model from the same records. It predicts final outcome
from the selected state-action features, starting with normalized score margin or
win/loss targets:

```bash
python3 -m haggis.policy train-value \
  --input data/search_improved.jsonl \
  --output models/value_model.json \
  --epochs 5 \
  --validation-fraction 0.2 \
  --target actor_score_margin_normalized
```

The value model is not wired into search yet; it is the training artifact needed
for future value-guided rollouts.

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

### 5. Iterative self-play training schedule

Use a staged loop inspired by AlphaGo/AlphaZero: generate stronger self-play,
train, evaluate against the current champion, and promote only through a gate.
Keep failed candidates as artifacts, but do not overwrite `models/linear_policy.json`.

| Stage | Purpose | Data | Teacher/search budget | Gate | Expected runtime |
| --- | --- | --- | --- | --- | --- |
| Smoke | Verify the pipeline after code or rules changes. | 10-20 hands | `policy-rollout`, root 2-3, turns 8-20 | Trainer succeeds and tiny promotion gate writes artifacts. | < 5 min |
| Candidate | Produce a plausible local challenger. | 200-500 hands | `policy-rollout`, root 4, turns 40 | Challenger `policy-rollout` rating and win-rate deltas are >= 0; speed ratio <= 1.25. | 15-60 min |
| Serious | Get a meaningful strength signal. | 1k-5k hands | Mixed `policy-rollout`, `information-set`, and `tree-information-set`; root 4-6, turns 40-80 | Positive Elo/win-rate deltas across fixed seeds; no speed regression; full tests pass. | Hours |
| Release | Promote only a stable champion. | 10k+ hands or multiple serious runs | Best accepted teacher mix and budget | Passes champion/challenger gate on at least two seeds and README/model metadata are updated. | Overnight+ |

Recommended candidate loop:

```bash
python3 -m haggis.self_play export \
  --bot-a policy-rollout \
  --bot-b policy-rollout \
  --policy-model models/linear_policy.json \
  --search-root-moves 4 \
  --search-rollout-turns 40 \
  --hands 300 \
  --seed 10 \
  --observation-mode player \
  --output runs/candidate/search_improved.jsonl

python3 -m haggis.policy train \
  --input runs/candidate/search_improved.jsonl \
  --output runs/candidate/linear_policy.json \
  --epochs 8 \
  --averaged \
  --validation-fraction 0.2

python3 -m haggis.policy train-value \
  --input runs/candidate/search_improved.jsonl \
  --output runs/candidate/value_model.json \
  --epochs 5 \
  --validation-fraction 0.2 \
  --target actor_score_margin_normalized

python3 -m haggis.promotion \
  --champion models/linear_policy.json \
  --challenger runs/candidate/linear_policy.json \
  --output-dir runs/candidate/promotion \
  --hands 12 \
  --seed 100 \
  --search-root-moves 4 \
  --search-rollout-turns 40
```

Promotion criteria:

- challenger `policy-rollout` rating delta >= 0 against the champion;
- challenger hand win-rate delta >= 0 on the fixed gate;
- challenger speed ratio <= 1.25 at the recommended budget;
- `python3 -m compileall haggis tests` and `python3 -m unittest discover -s tests` pass;
- `promotion.json`, ladder JSON, benchmark JSON, and model artifacts are kept under `runs/`.

Rollback/rejection criteria:

- If the promotion gate fails, keep the candidate in `runs/` but do not copy it to
  `models/linear_policy.json`.
- If strength improves but speed regresses, reject or rerun with a cheaper rollout
  budget before promotion.
- If direct `policy` improves but `policy-rollout` regresses, keep the model as an
  experiment only; the default CPU is `policy-rollout`.
- If rules or scoring change, retrain before treating old ratings as comparable.

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
