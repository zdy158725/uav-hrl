import unittest

from exp_article_test.config import HRLConfig
from exp_article_test.scene import BattleScene
from exp_diffusion_hrl.config import DiffusionHRLConfig


class HighLevelTimingTests(unittest.TestCase):
    def test_new_training_defaults_to_two_second_high_level_actions(self):
        self.assertEqual(HRLConfig["sub_steps"], 2)
        self.assertEqual(DiffusionHRLConfig["sub_steps"], 2)

        scene = BattleScene(1, 1)
        scene.reset()
        scene.step({"red_0": 0})
        self.assertAlmostEqual(scene.bf.t, 2.0)
        self.assertFalse(scene.done()["__all__"])


if __name__ == "__main__":
    unittest.main()
