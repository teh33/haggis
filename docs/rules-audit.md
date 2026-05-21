# Haggis rules audit

Source checked: published two-player rules PDF (`haggis.pdf`, Sean Ross; text extracted locally with `pdftotext`). This audit focuses on the implemented two-player game.

## Audited rule areas

- Deck/setup: 42 cards total: ranks 2-10 in four suits plus two each of J/Q/K; each player has face-up J/Q/K wild cards plus 14 dealt cards; 8 undealt haggis cards.
- Betting: before playing cards, each player may bet 0/15/30; successful bets score for bettor, failed bets score for opponent.
- Combination types: sets, sequences, and six bomb types.
- Wilds: J/Q/K can stand for lower ranks in non-bomb sets/sequences and may be played alone as their printed rank.
- Trick play: responses must match type/card count and outrank, except bombs; after a bomb only a higher bomb may beat it.
- Bomb capture: bomb winner leads next trick, but trick cards are captured by the opponent.
- Hand end: player who sheds final card wins the hand; final trick capture follows the normal bomb-capture exception.
- Scoring: hand winner scores 5 per opponent leftover card; captured point cards score for capturer; opponent leftover point cards and haggis point cards score for hand winner; bets are scored as successful/failed bet amounts.
- Game continuation: trailing player leads next hand; if tied, previous hand winner leads.

## Findings

### Confirmed correct after this audit

- Card point values match rules and are now explicitly tested: 3/5/7/9 = 1, J = 2, Q = 3, K = 5, other ranks = 0.
- Hand scoring includes captured point cards, 5-per-leftover-card, opponent leftover point cards, haggis point cards, and bet swings.
- Bomb capture behavior is covered by tests.
- Tie continuation now makes the previous hand winner lead by setting dealer to the previous loser.

### Fixed during this audit

- Reinstated scoring of opponent leftover point cards and haggis point cards for the hand winner. The previous correction had removed these, but the official rules explicitly include them.
- Fixed tied-game dealer progression: when cumulative scores are tied, the previous hand winner should lead the next hand; internally this means dealer is `1 - last_hand_winner`.
- Updated README strategy text to accurately describe leftover-card and haggis point-card scoring.

### Remaining confidence notes

- The engine has legal-move oracle/invariant tests plus focused tests for scoring, betting, bombs, and dealer progression.
- Rules involving sequence examples and wild substitution are the riskiest area because they are combinatorially broad; existing tests cover representative cases, not every printed player-aid example.
- Bot strength/model artifacts were trained under earlier scoring details, so they remain playable but should be retrained if model strength matters under the corrected official scoring.
