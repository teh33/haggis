import unittest

from haggis import (
    BombControlBot,
    Card,
    CombinationType,
    EndgameSearchBot,
    GreedySheddingBot,
    HaggisState,
    InformationSetRolloutBot,
    MonteCarloRolloutBot,
    Move,
    PolicyRolloutBot,
    TreeInformationSetBot,
    UCBInformationSetBot,
    PointAwareBot,
    RandomBot,
    Rank,
    Suit,
    legal_moves,
    validate_combination,
)


def c(rank, suit=Suit.CLUBS, wild=False):
    return Card(Rank(rank), suit, wild)


class EngineAndBotTests(unittest.TestCase):
    def test_legal_moves_include_pass_only_when_responding(self):
        hand = (c(7), c(8), c(9), c(11, wild=True), c(12, wild=True))
        lead_moves = legal_moves(hand)
        previous = validate_combination((c(7, Suit.HEARTS),))
        response_moves = legal_moves(hand, previous)

        self.assertTrue(all(not move.is_pass for move in lead_moves))
        self.assertTrue(response_moves[-1].is_pass)
        self.assertTrue(any(move.combination.type == CombinationType.BOMB for move in response_moves if not move.is_pass))

    def test_playing_and_passing_awards_non_bomb_trick_to_highest_player(self):
        state = HaggisState(hands=((c(7), c(2)), (c(8), c(9))), current_player=0)
        state = state.apply_move(Move((c(7),), validate_combination((c(7),))))
        state = state.apply_move(Move((c(8),), validate_combination((c(8),))))
        state = state.apply_move(Move.pass_turn())

        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.captured[1], (c(7), c(8)))
        self.assertIsNone(state.last_combination)

    def test_bomb_winner_leads_but_opponent_captures_trick_cards(self):
        jq_bomb = (c(11, wild=True), c(12, wild=True))
        state = HaggisState(hands=((c(9), c(2)), (*jq_bomb, c(10))), current_player=0)
        state = state.apply_move(Move((c(9),), validate_combination((c(9),))))
        state = state.apply_move(Move(jq_bomb, validate_combination(jq_bomb)))
        state = state.apply_move(Move.pass_turn())

        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.captured[0], (c(9), *jq_bomb))
        self.assertEqual(state.captured[1], ())

    def test_final_bomb_is_captured_by_opponent_but_bomb_player_wins_hand(self):
        jq_bomb = (c(11, wild=True), c(12, wild=True))
        state = HaggisState(hands=(jq_bomb, (c(3),)), current_player=0)
        state = state.apply_move(Move(jq_bomb, validate_combination(jq_bomb)))

        self.assertEqual(state.hand_winner, 0)
        self.assertEqual(state.captured[1], jq_bomb)

    def test_score_hand_counts_loser_cards_points_haggis_captures_and_bets(self):
        state = HaggisState(
            hands=((), (c(3), c(4), c(13, wild=True))),
            haggis=(c(5),),
            captured=((c(9),), (c(7),)),
            bets=(15, 30),
            hand_winner=0,
        )

        score = state.score_hand()

        # Winner: 3 loser cards * 5 + own captured 9 + loser hand points 3/K + haggis 5 + own bet + failed opponent bet.
        self.assertEqual(score.points, (15 + 1 + 6 + 1 + 15 + 30, 1))

    def test_bots_return_legal_moves(self):
        state = HaggisState(hands=((c(7), c(8), c(9)), (c(3),)), current_player=0)
        random_move = RandomBot(seed=1).choose_move(state)
        greedy_move = GreedySheddingBot().choose_move(state)
        point_aware_move = PointAwareBot().choose_move(state)
        bomb_control_move = BombControlBot().choose_move(state)
        legal = set(state.legal_moves())

        self.assertIn(random_move, legal)
        self.assertIn(greedy_move, legal)
        self.assertIn(point_aware_move, legal)
        self.assertIn(bomb_control_move, legal)
        self.assertEqual(len(greedy_move.cards), 3)

    def test_point_aware_bot_avoids_point_cards_when_it_can_still_beat(self):
        previous = validate_combination((c(6, Suit.HEARTS),))
        state = HaggisState(
            hands=((c(7), c(8), c(9)), (c(2),)),
            current_player=0,
            last_combination=previous,
            last_player=1,
        )

        move = PointAwareBot().choose_move(state)

        self.assertEqual(move.cards, (c(8),))

    def test_point_aware_bot_still_takes_a_hand_emptying_play(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(2),)), current_player=0)

        move = PointAwareBot().choose_move(state)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_bomb_control_bot_saves_bomb_when_non_bomb_can_beat(self):
        previous = validate_combination((c(8, Suit.HEARTS),))
        jq_bomb = (c(11, wild=True), c(12, wild=True))
        state = HaggisState(
            hands=((c(9), *jq_bomb), (c(2),)),
            current_player=0,
            last_combination=previous,
            last_player=1,
        )

        move = BombControlBot().choose_move(state)

        self.assertEqual(move.cards, (c(9),))
        self.assertNotEqual(move.combination.type, CombinationType.BOMB)

    def test_bomb_control_bot_uses_lowest_bomb_against_bomb(self):
        previous_bomb = validate_combination((c(11, Suit.HEARTS, wild=True), c(12, Suit.HEARTS, wild=True)))
        higher_jk_bomb = (c(11, wild=True), c(13, wild=True))
        higher_qk_bomb = (c(12, wild=True), c(13, wild=True))
        state = HaggisState(
            hands=((*higher_jk_bomb, *higher_qk_bomb, c(2)), (c(3),)),
            current_player=0,
            last_combination=previous_bomb,
            last_player=1,
        )

        move = BombControlBot().choose_move(state)

        self.assertEqual(move.combination.type, CombinationType.BOMB)
        self.assertEqual(move.combination.bomb_rank, 3)

    def test_endgame_search_bot_takes_immediate_hand_win(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(9),)), current_player=0)

        move = EndgameSearchBot(max_cards=4).choose_move(state)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_endgame_search_bot_returns_playable_moves(self):
        state = HaggisState(
            hands=((c(8), c(9), c(11, wild=True)), (c(7, Suit.HEARTS), c(2))),
            current_player=1,
        )
        state = state.apply_move(Move((c(7, Suit.HEARTS),), validate_combination((c(7, Suit.HEARTS),))))

        move = EndgameSearchBot(max_cards=6).choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_monte_carlo_rollout_bot_takes_immediate_hand_win(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(9),)), current_player=0)

        move = MonteCarloRolloutBot(seed=1).choose_move(state)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_monte_carlo_rollout_bot_is_deterministic_for_same_seed(self):
        state = HaggisState(
            hands=((c(7), c(8), c(9), c(10)), (c(3), c(4), c(5))),
            current_player=0,
        )

        first = MonteCarloRolloutBot(seed=7, simulations_per_move=1, max_root_moves=4).choose_move(state)
        second = MonteCarloRolloutBot(seed=7, simulations_per_move=1, max_root_moves=4).choose_move(state)

        self.assertEqual(first, second)
        self.assertIn(first, state.legal_moves())

    def test_monte_carlo_rollout_bot_returns_playable_move_when_responding(self):
        state = HaggisState(
            hands=((c(8), c(9), c(11, wild=True)), (c(7, Suit.HEARTS), c(2))),
            current_player=1,
        )
        state = state.apply_move(Move((c(7, Suit.HEARTS),), validate_combination((c(7, Suit.HEARTS),))))

        move = MonteCarloRolloutBot(seed=3, simulations_per_move=1, max_root_moves=4).choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_information_set_rollout_bot_takes_immediate_hand_win(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(9),)), current_player=0)

        move = InformationSetRolloutBot(seed=1).choose_move(state)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_information_set_rollout_bot_samples_unknown_opponent_and_haggis_cards(self):
        opponent_cards = (c(3), c(4), c(5))
        haggis = (c(6), c(7), c(8))
        state = HaggisState(hands=((c(9), c(10)), opponent_cards), haggis=haggis, current_player=0)

        determinized = InformationSetRolloutBot(seed=2).sample_determinization(state)

        self.assertEqual(determinized.hands[0], state.hands[0])
        self.assertEqual(len(determinized.hands[1]), len(opponent_cards))
        self.assertEqual(len(determinized.haggis), len(haggis))
        self.assertEqual(set(determinized.hands[1]) | set(determinized.haggis), set(opponent_cards) | set(haggis))
        self.assertNotEqual(determinized.hands[1], opponent_cards)

    def test_information_set_rollout_bot_is_hidden_info_invariant_for_immediate_wins(self):
        state_a = HaggisState(
            hands=((c(7), c(7, Suit.HEARTS)), (c(3), c(4))),
            haggis=(c(5), c(6)),
            current_player=0,
        )
        state_b = HaggisState(
            hands=((c(7), c(7, Suit.HEARTS)), (c(5), c(6))),
            haggis=(c(3), c(4)),
            current_player=0,
        )

        first = InformationSetRolloutBot(seed=4).choose_move(state_a)
        second = InformationSetRolloutBot(seed=4).choose_move(state_b)

        self.assertEqual(first, second)

    def test_information_set_rollout_bot_returns_playable_move_when_responding(self):
        state = HaggisState(
            hands=((c(8), c(9), c(11, wild=True)), (c(7, Suit.HEARTS), c(2))),
            current_player=1,
        )
        state = state.apply_move(Move((c(7, Suit.HEARTS),), validate_combination((c(7, Suit.HEARTS),))))

        move = InformationSetRolloutBot(seed=3, simulations_per_move=1, max_root_moves=4).choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_ucb_information_set_bot_takes_immediate_hand_win(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(9),)), current_player=0)

        move = UCBInformationSetBot(seed=1).choose_move(state)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_ucb_information_set_bot_is_deterministic_for_same_seed(self):
        state = HaggisState(
            hands=((c(7), c(8), c(9), c(10)), (c(3), c(4), c(5))),
            haggis=(c(6), c(2)),
            current_player=0,
        )

        first = UCBInformationSetBot(seed=7, simulations=5, max_root_moves=4).choose_move(state)
        second = UCBInformationSetBot(seed=7, simulations=5, max_root_moves=4).choose_move(state)

        self.assertEqual(first, second)
        self.assertIn(first, state.legal_moves())

    def test_ucb_information_set_bot_returns_playable_move_when_responding(self):
        state = HaggisState(
            hands=((c(8), c(9), c(11, wild=True)), (c(7, Suit.HEARTS), c(2))),
            current_player=1,
        )
        state = state.apply_move(Move((c(7, Suit.HEARTS),), validate_combination((c(7, Suit.HEARTS),))))

        move = UCBInformationSetBot(seed=3, simulations=5, max_root_moves=4).choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_policy_rollout_bot_uses_policy_model_for_playable_rollouts(self):
        import tempfile
        from pathlib import Path

        from haggis.policy import LinearPolicy

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "policy.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)
            state = HaggisState(hands=((c(7), c(8), c(9)), (c(3), c(4))), haggis=(c(5),), current_player=0)
            bot = PolicyRolloutBot(model_path, seed=3, simulations_per_move=1, max_root_moves=3, max_rollout_turns=8)

            move = bot.choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_tree_information_set_bot_returns_playable_move(self):
        state = HaggisState(hands=((c(7), c(8), c(9), c(10)), (c(3), c(4), c(5))), haggis=(c(6),), current_player=0)
        bot = TreeInformationSetBot(seed=5, simulations=4, max_root_moves=3, max_child_moves=2, max_rollout_turns=8)

        move = bot.choose_move(state)

        self.assertIn(move, state.legal_moves())

    def test_tree_information_set_search_visits_root_candidates(self):
        state = HaggisState(hands=((c(7), c(8), c(9), c(10)), (c(3), c(4), c(5))), haggis=(c(6),), current_player=0)
        bot = TreeInformationSetBot(seed=5, simulations=4, max_root_moves=3, max_child_moves=2, max_rollout_turns=8)

        root_moves, visits, values = bot.search_root(state)

        self.assertEqual(len(root_moves), len(visits))
        self.assertEqual(len(root_moves), len(values))
        self.assertGreaterEqual(sum(visits), len(root_moves))
        self.assertTrue(all(visit_count >= 1 for visit_count in visits))

    def test_ucb_information_set_search_visits_each_root_candidate(self):
        state = HaggisState(hands=((c(7), c(8), c(9), c(10)), (c(3), c(4), c(5))), haggis=(c(6),), current_player=0)
        bot = UCBInformationSetBot(seed=5, simulations=3, max_root_moves=4)

        root_moves, visits, values = bot.search_root(state)

        self.assertEqual(len(root_moves), len(visits))
        self.assertEqual(len(root_moves), len(values))
        self.assertGreaterEqual(sum(visits), len(root_moves))
        self.assertTrue(all(visit_count >= 1 for visit_count in visits))

if __name__ == "__main__":
    unittest.main()
