import unittest

import numpy as np

from main_pys.grid_actions import action_mask_for_locs, get_action_dim
from main_pys.model_inputs import create_data_object, discrete_action_labels_from_positions
from main_pys.simulator import pibt


class Grid8ActionTests(unittest.TestCase):
    def test_grid8_labels_and_bd_width(self):
        positions = np.array([[[0, 0], [1, 1]], [[3, 3], [2, 4]]])
        labels = discrete_action_labels_from_positions(positions, 0, action_mode="grid8")
        self.assertEqual(labels.tolist(), [5, 7])

        grid = np.zeros((5, 5), dtype=int)
        bd = np.zeros((2, 5, 5), dtype=np.float32)
        data = create_data_object(
            np.array([[2, 2], [1, 1]]),
            bd,
            grid,
            k=1,
            m=1,
            goal_locs=np.zeros((2, 2), dtype=int),
            action_mode="grid8",
        )
        self.assertEqual(get_action_dim("grid8"), 9)
        self.assertEqual(data.bd_pred.shape[1], 9)

    def test_blocked_pair_diagonal_mask(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[2, 1] = 1
        grid[1, 2] = 1
        mask = action_mask_for_locs(
            grid,
            np.array([[1, 1]]),
            action_mode="grid8",
            diagonal_rule="blocked_pair",
        )
        self.assertTrue(mask[0, 5])

    def test_pibt_blocks_crossing_diagonals(self):
        grid = np.zeros((4, 4), dtype=int)
        prefs = np.array([
            [5, 0, 1, 2, 3, 4, 6, 7, 8],
            [7, 0, 1, 2, 3, 4, 5, 6, 8],
        ])
        move, ok = pibt(
            grid,
            prefs,
            np.array([[1, 1], [2, 1]]),
            np.array([2.0, 1.0]),
            [],
            0,
            999,
            action_mode="grid8",
        )
        self.assertTrue(ok)
        self.assertFalse(move[0].tolist() == [1, 1] and move[1].tolist() == [-1, 1])


if __name__ == "__main__":
    unittest.main()
