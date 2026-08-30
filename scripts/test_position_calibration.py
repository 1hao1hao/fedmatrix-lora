#!/usr/bin/env python3
"""Focused checks for the three C3 position-calibration variants."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.server import Server


class DummyClient:
    def __init__(self, block_scores):
        self.client_id = 0
        self.candidate_metadata = []
        scores = []
        for block_id, block_score in enumerate(block_scores):
            for module_type in ("query", "key", "value"):
                self.candidate_metadata.append(
                    {"block_id": block_id, "module_type": module_type}
                )
                scores.append(block_score)
        self.scores = np.asarray(scores, dtype=float)
        self.last_position_signal = 0.0
        self.last_position_weight = 0.0
        self.last_position_direction = "none"
        self.last_depth_prior = None
        self.last_agreement_scores = None
        self.last_final_utility = None

    def get_depth_prior(self, scale="one_ten"):
        return np.linspace(1.0, 10.0, len(self.scores))


def make_server(calibration_mode):
    server = Server.__new__(Server)
    server.args = SimpleNamespace(
        dataset="mnli",
        score_utility_mode="raw",
        position_calibration_mode=calibration_mode,
        adaptive_depth_mix=False,
        depth_prior_weight=0.0,
        position_band_fraction=1.0 / 3.0,
        position_signal_temperature=0.15,
        position_prior_max_weight=0.45,
        score_stability_eps=1e-8,
        agreement_weight=0.0,
    )
    return server


class PositionCalibrationTests(unittest.TestCase):
    def test_adaptive_prior_selects_deep_direction(self):
        server = make_server("adaptive")
        client = DummyClient([1, 1, 2, 2, 4, 4])

        prior, weight, signal = server._build_adaptive_position_prior(
            client, client.scores
        )

        self.assertGreater(signal, 0.0)
        self.assertGreater(weight, 0.0)
        self.assertGreater(prior[-1], prior[0])

    def test_adaptive_prior_selects_shallow_direction(self):
        server = make_server("adaptive")
        client = DummyClient([4, 4, 2, 2, 1, 1])

        prior, weight, signal = server._build_adaptive_position_prior(
            client, client.scores
        )

        self.assertLess(signal, 0.0)
        self.assertGreater(weight, 0.0)
        self.assertGreater(prior[0], prior[-1])

    def test_flat_profile_disables_position_prior(self):
        server = make_server("adaptive")
        client = DummyClient([2, 2, 2, 2, 2, 2])

        prior, weight, signal = server._build_adaptive_position_prior(
            client, client.scores
        )

        self.assertAlmostEqual(signal, 0.0)
        self.assertAlmostEqual(weight, 0.0)
        np.testing.assert_allclose(prior, np.ones_like(client.scores))

    def test_explicit_adaptive_mode_overrides_raw_utility_mode(self):
        server = make_server("adaptive")
        client = DummyClient([4, 4, 2, 2, 1, 1])

        final = server._build_client_utility_scores(
            client,
            client.scores,
            global_max=float(np.max(client.scores)),
            agreement_scores=np.zeros_like(client.scores),
            allocation_strategy="Random_FIM",
        )

        self.assertEqual(client.last_position_direction, "shallow")
        self.assertGreater(client.last_position_weight, 0.0)
        self.assertEqual(len(final), len(client.scores))


if __name__ == "__main__":
    unittest.main()
