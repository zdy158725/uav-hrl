import unittest
from unittest.mock import patch

import numpy as np

from exp_article_test.battlefield import BattleField


class SimultaneousDamageTests(unittest.TestCase):
    def _check_after_step(self, battlefield, attack_agent_ids, attack_targets=None):
        targets = attack_targets or {agent_id: "0" for agent_id in attack_agent_ids}
        battlefield.check_after_step(attack_agent_ids, targets)

    def test_both_sides_can_destroy_each_other_in_one_physics_step(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()
        red = battlefield.red_uav[0]
        blue = battlefield.blue_uav[0]
        red.pos = np.array([8000.0, 8000.0, -4500.0])
        blue.pos = np.array([8500.0, 8000.0, -4500.0])
        red.blood = blue.blood = 0.05

        self._check_after_step(battlefield, {"red_0", "blue_0"})
        battlefield.check_winner()

        self.assertFalse(red.alive)
        self.assertFalse(blue.alive)
        self.assertEqual(battlefield.battlefield_info, "draw")
        self.assertGreater(battlefield.record["Red"]["0"]["damage"], 0.0)
        self.assertGreater(battlefield.record["Blue"]["0"]["damage"], 0.0)

    def test_multiple_hits_are_summed_before_destruction(self):
        battlefield = BattleField(2, 1)
        battlefield.reset()
        blue = battlefield.blue_uav[0]
        blue.blood = 1.0

        def damage(attacker, target):
            return 0.6 if attacker.team == "Red" else 0.0

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            self._check_after_step(battlefield, {"red_0", "red_1", "blue_0"})

        self.assertFalse(blue.alive)
        self.assertAlmostEqual(battlefield.record["Blue"]["0"]["hurt"], 1.2)
        for red in battlefield.red_uav:
            self.assertAlmostEqual(battlefield.record["Red"][red.id]["damage"], 0.6)
            self.assertEqual(battlefield.record["Red"][red.id]["target_blood"], 0.0)

    def test_evading_uav_does_not_deal_damage(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()
        red = battlefield.red_uav[0]
        blue = battlefield.blue_uav[0]
        red.pos = np.array([8000.0, 8000.0, -4500.0])
        blue.pos = np.array([8500.0, 8000.0, -4500.0])
        red.blood = blue.blood = 0.05

        self._check_after_step(battlefield, {"red_0"})

        self.assertTrue(red.alive)
        self.assertTrue(blue.alive)
        self.assertLess(blue.blood, 0.05)
        self.assertGreater(battlefield.record["Red"]["0"]["damage"], 0.0)
        self.assertEqual(battlefield.record["Blue"]["0"]["damage"], 0.0)

    def test_evading_defender_receives_half_damage(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()
        with patch.object(battlefield, "cal_damage", return_value=0.6):
            self._check_after_step(battlefield, {"red_0"})

        self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.7)
        self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.3)
        self.assertAlmostEqual(battlefield.record["Blue"]["0"]["hurt"], 0.3)
        self.assertEqual(battlefield.red_uav[0].blood, 1.0)

    def test_attacking_defender_receives_full_damage(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()

        def damage(attacker, target):
            return 0.6 if attacker.team == "Red" else 0.0

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            self._check_after_step(battlefield, {"red_0", "blue_0"})

        self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.4)
        self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.6)

    def test_only_locked_target_receives_damage_even_if_another_would_take_more(self):
        battlefield = BattleField(1, 2)
        battlefield.reset()

        def damage(attacker, target):
            if attacker.team == "Blue":
                return 0.0
            return 0.2 if target.id == "0" else 0.8

        with patch.object(battlefield, "cal_damage", side_effect=damage) as calculate:
            self._check_after_step(
                battlefield,
                {"red_0"},
                {"red_0": "0"},
            )

        self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.9)
        self.assertEqual(battlefield.blue_uav[1].blood, 1.0)
        self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.1)
        calculate.assert_called_once_with(battlefield.red_uav[0], battlefield.blue_uav[0])

    def test_locked_target_miss_does_not_switch_to_another_enemy(self):
        battlefield = BattleField(1, 2)
        battlefield.reset()

        def damage(attacker, target):
            return 0.0 if target.id == "0" else 0.8

        with patch.object(battlefield, "cal_damage", side_effect=damage) as calculate:
            self._check_after_step(battlefield, {"red_0"}, {"red_0": "0"})

        self.assertEqual(battlefield.blue_uav[0].blood, 1.0)
        self.assertEqual(battlefield.blue_uav[1].blood, 1.0)
        self.assertNotIn("red_0", battlefield.last_damage_time)
        calculate.assert_called_once_with(battlefield.red_uav[0], battlefield.blue_uav[0])

    def test_destroyed_target_is_not_replaced_mid_step(self):
        battlefield = BattleField(1, 2)
        battlefield.reset()
        battlefield.blue_uav[0].destroy("hit")
        with patch.object(battlefield, "cal_damage", side_effect=AssertionError("unexpected attack")):
            self._check_after_step(battlefield, {"red_0"}, {"red_0": "0"})
        self.assertEqual(battlefield.blue_uav[1].blood, 1.0)

    def test_step_skips_damage_calculation_without_attackers(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()
        with patch.object(battlefield, "cal_damage", side_effect=AssertionError("unexpected attack")):
            battlefield.step(
                {"red_0": 0},
                {"blue_0": 0},
                attack_agent_ids=set(),
                attack_targets={},
            )
        self.assertEqual(battlefield.red_uav[0].blood, 1.0)
        self.assertEqual(battlefield.blue_uav[0].blood, 1.0)

    def test_successful_hits_respect_per_uav_fire_cooldown(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()

        def damage(attacker, target):
            return 0.2 if attacker.team == "Red" else 0.0

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            battlefield.t = 0.1
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.8)

            battlefield.t = 0.2
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.8)
            self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.2)

            battlefield.t = 0.6
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.6)
            self.assertAlmostEqual(battlefield.last_damage_time["red_0"], 0.6)
            self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.4)

    def test_step_records_all_hits_for_low_level_reward(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()

        def damage(attacker, target):
            return 0.1 if attacker.team == "Red" else 0.0

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            battlefield.step(
                {"red_0": 0},
                {"blue_0": 0},
                attack_agent_ids={"red_0", "blue_0"},
                attack_targets={"red_0": "0", "blue_0": "0"},
            )

        self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.8)
        self.assertAlmostEqual(battlefield.record["Red"]["0"]["damage"], 0.2)
        self.assertAlmostEqual(battlefield.record["Blue"]["0"]["hurt"], 0.2)

    def test_missed_attack_does_not_start_cooldown(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()

        def damage(attacker, target):
            if attacker.team == "Blue":
                return 0.0
            return 0.0 if battlefield.t == 0.1 else 0.2

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            battlefield.t = 0.1
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            self.assertNotIn("red_0", battlefield.last_damage_time)

            battlefield.t = 0.2
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.8)

    def test_fire_cooldown_is_independent_for_each_uav(self):
        battlefield = BattleField(2, 1)
        battlefield.reset()

        def damage(attacker, target):
            return 0.1 if attacker.team == "Red" else 0.0

        with patch.object(battlefield, "cal_damage", side_effect=damage):
            battlefield.t = 0.1
            self._check_after_step(battlefield, {"red_0", "blue_0"})
            battlefield.t = 0.2
            self._check_after_step(battlefield, {"red_1", "blue_0"})

        self.assertAlmostEqual(battlefield.blue_uav[0].blood, 0.8)
        self.assertAlmostEqual(battlefield.last_damage_time["red_0"], 0.1)
        self.assertAlmostEqual(battlefield.last_damage_time["red_1"], 0.2)

    def test_fire_cooldown_resets_with_new_episode(self):
        battlefield = BattleField(1, 1)
        battlefield.reset()
        with patch.object(battlefield, "cal_damage", return_value=0.2):
            battlefield.t = 0.1
            self._check_after_step(battlefield, {"red_0"})
        self.assertIn("red_0", battlefield.last_damage_time)

        battlefield.reset()
        self.assertEqual(battlefield.last_damage_time, {})


if __name__ == "__main__":
    unittest.main()
