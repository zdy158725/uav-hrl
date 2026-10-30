import unittest

import numpy as np

from exp_article_test.low_level_env import LowLevelBattleEnv
from exp_article_test.maneuver import (
    ACTION_COUNT,
    ACTION_LIBRARY,
    BASE_BANK,
    convert_to_contol_input,
    select_discrete_action,
)


class ActionLibraryTests(unittest.TestCase):
    def test_original_action_numbers_keep_their_controls(self):
        original = [
            (0, 1, 0), (4, 1, 0), (-4, 1, 0),
            (0, 7, 0), (4, 7, 0), (-4, 7, 0),
            (0, -5, 0), (4, -5, 0), (-4, -5, 0),
            (0, 7, BASE_BANK), (4, 7, BASE_BANK), (-4, 7, BASE_BANK),
            (0, 7, -BASE_BANK), (4, 7, -BASE_BANK), (-4, 7, -BASE_BANK),
        ]
        self.assertEqual(ACTION_COUNT, 15)
        self.assertEqual(len({tuple(action) for action in ACTION_LIBRARY}), ACTION_COUNT)
        for action_id, control in enumerate(original):
            np.testing.assert_allclose(convert_to_contol_input(action_id), control)

    def test_script_controls_map_to_original_actions(self):
        for control in ((4, 3.5, 0), (4, 1, BASE_BANK), (4, 1, -BASE_BANK)):
            action_id = select_discrete_action(control)
            self.assertGreaterEqual(action_id, 0)
            self.assertLess(action_id, ACTION_COUNT)
            distances = np.linalg.norm(ACTION_LIBRARY - control, axis=1)
            self.assertEqual(action_id, int(np.argmin(distances)))

    def test_low_level_environment_executes_original_action_number(self):
        env = LowLevelBattleEnv({"mode": "evade", "num_red": 1, "num_blue": 1})
        self.assertEqual(env.action_space.n, ACTION_COUNT)
        env.reset()
        action_id = select_discrete_action((4, 1, BASE_BANK))
        env.step({"red_0": action_id})
        self.assertEqual(env.bf.record["Red"]["0"]["action_index"], action_id)

    def test_removed_action_number_is_rejected(self):
        with self.assertRaises(ValueError):
            convert_to_contol_input(15)


if __name__ == "__main__":
    unittest.main()
