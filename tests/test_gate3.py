from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from gate3_read_write import gate3a, gate3b  # noqa: E402


class ReadWriteGateTests(unittest.TestCase):
    def test_state_write_has_exact_null_controls(self) -> None:
        result = gate3a(1)
        self.assertLess(result["max_hypothesis_spread"]["read_only_zero"], 1e-13)
        self.assertLess(result["max_hypothesis_spread"]["uniform_write"], 1e-13)
        self.assertGreater(result["max_hypothesis_spread"]["localized_write"], 0.25)

    def test_persistent_write_survives_fast_state_reset(self) -> None:
        result = gate3b(1)
        self.assertTrue(result["fast_state_erased_before_diagnostic"])
        self.assertFalse(result["write_identity_available_to_observer"])
        self.assertLess(result["max_hypothesis_spread"]["no_operator_write"], 1e-13)
        self.assertGreater(result["max_hypothesis_spread"]["persistent_operator_write"], 0.005)

    def test_noise_aware_policy_beats_naive_variance(self) -> None:
        result = gate3b(2)
        self.assertGreater(result["noise_aware"]["final_accuracy"], result["raw_variance"]["final_accuracy"])
        self.assertGreater(result["noise_aware"]["final_accuracy"], result["random"]["final_accuracy"])


if __name__ == "__main__":
    unittest.main()
