"""学习率 warmup/cosine 调度与 LambdaLR 时序测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.schedules import (  # noqa: E402
    resolve_warmup_steps,
    warmup_cosine_factor,
)


class ResolveWarmupStepsTest(unittest.TestCase):
    def test_explicit_steps_override_ratio(self) -> None:
        self.assertEqual(
            resolve_warmup_steps(100, warmup_steps=7, warmup_ratio=0.5), 7
        )

    def test_ratio_is_rounded(self) -> None:
        self.assertEqual(resolve_warmup_steps(100, warmup_ratio=0.05), 5)
        self.assertEqual(resolve_warmup_steps(3, warmup_ratio=0.5), 2)
        self.assertEqual(resolve_warmup_steps(10, warmup_ratio=0.0), 0)

    def test_warmup_is_clamped_to_total(self) -> None:
        self.assertEqual(resolve_warmup_steps(10, warmup_steps=99), 10)
        self.assertEqual(resolve_warmup_steps(10, warmup_ratio=1.0), 10)

    def test_invalid_inputs_are_rejected(self) -> None:
        for total in (0, -1, 1.5, True):
            with self.subTest(total=total):
                with self.assertRaises(ValueError):
                    resolve_warmup_steps(total)
        for ratio in (-0.1, 1.1, math.nan, math.inf, -math.inf):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    resolve_warmup_steps(10, warmup_ratio=ratio)
        with self.assertRaises(ValueError):
            resolve_warmup_steps(10, warmup_steps=-1)
        with self.assertRaises(ValueError):
            resolve_warmup_steps(10, warmup_ratio="bad")


class WarmupCosineFactorTest(unittest.TestCase):
    def test_warmup_then_cosine_reaches_floor(self) -> None:
        total, warmup, floor = 12, 3, 0.1
        factors = [
            warmup_cosine_factor(step, total, warmup, floor)
            for step in range(total + 1)
        ]

        self.assertAlmostEqual(factors[0], 1.0 / 3.0)
        self.assertAlmostEqual(factors[1], 2.0 / 3.0)
        self.assertAlmostEqual(factors[2], 1.0)
        self.assertAlmostEqual(factors[3], 1.0)
        self.assertAlmostEqual(factors[-1], floor)
        self.assertTrue(all(floor <= value <= 1.0 for value in factors))
        self.assertTrue(
            all(a <= b + 1e-12 for a, b in zip(factors[:4], factors[1:4]))
        )
        self.assertTrue(
            all(a >= b - 1e-12 for a, b in zip(factors[3:], factors[4:]))
        )

    def test_pure_cosine_shape(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(0, 4, 0, 0.2), 1.0)
        self.assertAlmostEqual(
            warmup_cosine_factor(2, 4, 0, 0.2), 0.6, places=12
        )
        self.assertAlmostEqual(warmup_cosine_factor(4, 4, 0, 0.2), 0.2)

    def test_steps_after_end_stay_at_floor(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(50, 5, 2, 0.25), 0.25)

    def test_short_schedules_have_no_division_by_zero(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(0, 1, 1, 0.3), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(1, 1, 1, 0.3), 0.3)
        self.assertAlmostEqual(warmup_cosine_factor(0, 1, 0, 0.3), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(1, 1, 0, 0.3), 0.3)
        self.assertAlmostEqual(warmup_cosine_factor(0, 2, 2, 0.3), 0.5)
        self.assertAlmostEqual(warmup_cosine_factor(1, 2, 2, 0.3), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(2, 2, 2, 0.3), 0.3)

    def test_warmup_longer_than_total_is_clamped(self) -> None:
        factors = [
            warmup_cosine_factor(step, 3, 10, 0.1) for step in range(4)
        ]

        self.assertAlmostEqual(factors[0], 1.0 / 3.0)
        self.assertAlmostEqual(factors[1], 2.0 / 3.0)
        self.assertAlmostEqual(factors[2], 1.0)
        self.assertAlmostEqual(factors[3], 0.1)

    def test_zero_floor_reaches_zero(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(5, 5, 1, 0.0), 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        for floor in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    warmup_cosine_factor(0, 10, 0, floor)
        for total in (0, -1, 2.5):
            with self.subTest(total=total):
                with self.assertRaises(ValueError):
                    warmup_cosine_factor(0, total)
        with self.assertRaises(ValueError):
            warmup_cosine_factor(-1, 10)
        with self.assertRaises(ValueError):
            warmup_cosine_factor(0, 10, warmup_steps=-1)


class LambdaLRTimingTest(unittest.TestCase):
    def _scheduler(
        self, optimizer: torch.optim.Optimizer, total: int, warmup: int, floor: float
    ) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(
                step, total, warmup, floor
            ),
        )

    def test_lambda_lr_sequence_matches_schedule(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        total, warmup, floor = 6, 2, 0.2
        scheduler = self._scheduler(optimizer, total, warmup, floor)

        def current_lr() -> float:
            return float(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(
            current_lr(), warmup_cosine_factor(0, total, warmup, floor)
        )
        for step in range(1, total + 1):
            parameter.grad = torch.tensor(1.0)
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(
                current_lr(),
                warmup_cosine_factor(step, total, warmup, floor),
                msg=f"step {step}",
            )
        self.assertAlmostEqual(current_lr(), floor)

        scheduler.step()
        self.assertAlmostEqual(current_lr(), floor)

    def test_single_step_schedule_starts_above_floor(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = self._scheduler(optimizer, total=1, warmup=1, floor=0.3)

        self.assertAlmostEqual(float(optimizer.param_groups[0]["lr"]), 1.0)
        parameter.grad = torch.tensor(1.0)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(float(optimizer.param_groups[0]["lr"]), 0.3)


if __name__ == "__main__":
    unittest.main()
