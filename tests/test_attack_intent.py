import unittest
from unittest.mock import patch

import numpy as np

from exp_article_test.low_level_env import LowLevelBattleEnv
from exp_article_test.battlefield import BattleField
from exp_article_test.maneuver import make_decision_for_blue_team
from exp_article_test.scene import BattleScene


class AttackIntentTests(unittest.TestCase):
    def test_high_level_evade_and_attack_are_forwarded_to_battlefield(self):
        scene = BattleScene(1, 1, sub_steps=1)
        scene.reset()
        blue = scene.bf.blue_uav[0]
        with patch.object(scene, "_make_blue_actions", return_value=({"blue_0": 0}, set(), {})):
            with patch.object(scene, "infer_low_level_action", side_effect=[(0, None), (0, blue)]):
                with patch.object(scene.bf, "step") as battle_step:
                    scene.step({"red_0": 0})
                    self.assertEqual(battle_step.call_args.kwargs["attack_agent_ids"], set())
                    scene.step({"red_0": 1})
                    self.assertEqual(battle_step.call_args.kwargs["attack_agent_ids"], {"red_0"})
                    self.assertEqual(battle_step.call_args.kwargs["attack_targets"], {"red_0": "0"})

    def test_high_level_target_id_stays_locked_when_candidate_order_changes(self):
        scene = BattleScene(1, 2, sub_steps=2)
        scene.reset()
        red = scene.bf.red_uav[0]
        blue_0, blue_1 = scene.bf.blue_uav
        blue_0.pos = red.pos + np.array([1000.0, 0.0, 0.0])
        blue_1.pos = red.pos + np.array([2000.0, 0.0, 0.0])

        def reorder_after_first_step(*args, **kwargs):
            blue_0.pos = red.pos + np.array([2000.0, 0.0, 0.0])
            blue_1.pos = red.pos + np.array([1000.0, 0.0, 0.0])

        with patch.object(scene, "_make_blue_actions", return_value=({}, set(), {})):
            with patch.object(scene.bf, "step", side_effect=reorder_after_first_step) as battle_step:
                scene.step({"red_0": 1})

        self.assertEqual(battle_step.call_count, 2)
        for call in battle_step.call_args_list:
            self.assertEqual(call.kwargs["attack_targets"], {"red_0": "0"})

    def test_destroyed_high_level_target_is_not_replaced_before_next_decision(self):
        scene = BattleScene(1, 2, sub_steps=2)
        scene.reset()
        red = scene.bf.red_uav[0]
        blue_0, blue_1 = scene.bf.blue_uav
        blue_0.pos = red.pos + np.array([1000.0, 0.0, 0.0])
        blue_1.pos = red.pos + np.array([2000.0, 0.0, 0.0])

        def destroy_target_after_first_step(*args, **kwargs):
            blue_0.destroy("hit")
            scene.bf.check_winner()

        with patch.object(scene, "_make_blue_actions", return_value=({}, set(), {})):
            with patch.object(scene.bf, "step", side_effect=destroy_target_after_first_step) as battle_step:
                scene.step({"red_0": 1})

        self.assertEqual(battle_step.call_count, 2)
        self.assertEqual(battle_step.call_args_list[0].kwargs["attack_targets"], {"red_0": "0"})
        self.assertEqual(battle_step.call_args_list[1].kwargs["attack_targets"], {})

    def test_blue_script_marks_only_pursuit_as_attack(self):
        scene = BattleScene(1, 1)
        scene.reset()
        with patch("exp_article_test.maneuver.should_blue_evade", return_value=True):
            _, attack_agent_ids = make_decision_for_blue_team(scene.bf, return_attackers=True)
            self.assertEqual(attack_agent_ids, set())
        with patch("exp_article_test.maneuver.should_blue_evade", return_value=False):
            _, attack_agent_ids = make_decision_for_blue_team(scene.bf, return_attackers=True)
            self.assertEqual(attack_agent_ids, {"blue_0"})
            _, _, attack_targets = make_decision_for_blue_team(scene.bf, return_targets=True)
            self.assertEqual(attack_targets, {"blue_0": "0"})

    def test_blue_script_damages_its_chosen_target_only(self):
        battlefield = BattleField(2, 1)
        battlefield.reset()
        chosen_target = battlefield.red_uav[1]
        with patch(
            "exp_article_test.maneuver.select_coordinated_script_target",
            return_value=chosen_target,
        ):
            with patch("exp_article_test.maneuver.should_blue_evade", return_value=False):
                _, attackers, attack_targets = make_decision_for_blue_team(
                    battlefield, return_targets=True
                )

        def damage(attacker, target):
            return 0.8 if target.id == "0" else 0.2

        with patch.object(battlefield, "cal_damage", side_effect=damage) as calculate:
            battlefield.check_after_step(attackers, attack_targets)

        self.assertEqual(attack_targets, {"blue_0": "1"})
        self.assertEqual(battlefield.red_uav[0].blood, 1.0)
        self.assertAlmostEqual(chosen_target.blood, 0.9)
        calculate.assert_called_once_with(battlefield.blue_uav[0], chosen_target)

    def test_history_policy_uses_its_macro_action_for_attack_intent(self):
        scene = BattleScene(1, 1, blue_policy_mode="history_policy")
        scene.reset()
        with patch.object(scene.history_opponent, "compute_action", return_value=0):
            _, attack_agent_ids, attack_targets = scene._make_history_policy_blue_actions()
            self.assertEqual(attack_agent_ids, set())
            self.assertEqual(attack_targets, {})
        with patch.object(scene.history_opponent, "compute_action", return_value=1):
            _, attack_agent_ids, attack_targets = scene._make_history_policy_blue_actions()
            self.assertEqual(attack_agent_ids, {"blue_0"})
            self.assertEqual(attack_targets, {"blue_0": "0"})

    def test_fixed_history_policy_keeps_macro_decision_for_ten_low_level_steps(self):
        scene = BattleScene(1, 1, blue_policy_mode="history_policy", blue_high_level_sub_steps=10)
        scene.reset()
        with patch.object(scene.history_opponent, "compute_action", side_effect=[1, 0]) as decide:
            with patch.object(scene, "infer_low_level_action", return_value=(0, scene.bf.red_uav[0])):
                for _ in range(10):
                    _, attackers, targets = scene._make_history_policy_blue_actions()
                    self.assertEqual(attackers, {"blue_0"})
                    self.assertEqual(targets, {"blue_0": "0"})
                self.assertEqual(decide.call_count, 1)
                _, attackers, targets = scene._make_history_policy_blue_actions()
                self.assertEqual(decide.call_count, 2)
                self.assertEqual(attackers, set())
                self.assertEqual(targets, {})

    def test_low_level_training_mode_controls_red_attack(self):
        for mode in ("attack", "evade"):
            with self.subTest(mode=mode):
                env = LowLevelBattleEnv({"mode": mode, "num_red": 1, "num_blue": 1})
                env.reset()
                with patch(
                    "exp_article_test.low_level_env.make_decision_for_blue_team",
                    return_value=({"blue_0": 0}, {"blue_0"}, {"blue_0": "0"}),
                ):
                    with patch.object(env.bf, "step") as battle_step:
                        env.step({"red_0": 0})
                attack_agent_ids = battle_step.call_args.kwargs["attack_agent_ids"]
                self.assertEqual("red_0" in attack_agent_ids, mode == "attack")
                self.assertIn("blue_0", attack_agent_ids)
                attack_targets = battle_step.call_args.kwargs["attack_targets"]
                self.assertEqual("red_0" in attack_targets, mode == "attack")
                self.assertEqual(attack_targets["blue_0"], "0")


if __name__ == "__main__":
    unittest.main()
