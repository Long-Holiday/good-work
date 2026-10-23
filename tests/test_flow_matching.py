"""Flow Matching 公式、连续时间和随机 GRPO 策略契约（尚未执行）。"""

import unittest

import torch
from torch import nn

from diffpeak.models.flow_matching import ConditionalFlowMatching
from diffpeak.grpo import gaussian_kl_divergence


class ConstantVelocity(nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, state, time, conditions):
        return torch.ones_like(state) * self.value


class TestFlowMatching(unittest.TestCase):
    def features(self):
        return [torch.zeros(2, 1, max(1, 8 // 2**i), max(1, 8 // 2**i)) for i in range(4)]

    def test_euler_and_heun_constant_velocity(self):
        for solver in ("euler", "heun"):
            flow = ConditionalFlowMatching(sampling_steps=4, solver=solver, policy_std=0.1)
            state = torch.zeros(2, 1, 8, 8)
            mean, std = flow.transition(ConstantVelocity(), state, 0.25, 0.5, self.features())
            torch.testing.assert_close(mean, torch.full_like(state, 0.05))
            torch.testing.assert_close(std, torch.full_like(state, 0.05))
            self.assertTrue(torch.isfinite(flow.log_prob(mean, mean, std)).all())

    def test_heun_uses_endpoint_velocity(self):
        class LinearVelocity(nn.Module):
            def forward(self, state, time, conditions):
                return state

        state = torch.ones(2, 1, 8, 8)
        flow = ConditionalFlowMatching(solver="heun")
        mean, _ = flow.transition(LinearVelocity(), state, 0.0, 0.5, self.features())
        torch.testing.assert_close(mean, state * 1.625)

    def test_record_replay_and_reference_kl(self):
        flow = ConditionalFlowMatching(sampling_steps=3)
        model = ConstantVelocity()
        output, trace = flow.sample(model, self.features(), generator=torch.Generator().manual_seed(5), record=True)
        self.assertEqual(output.shape, (2, 1, 8, 8))
        self.assertEqual(len(trace), 3)
        for step in trace:
            self.assertLess(step["timestep"], step["previous_timestep"])
            mean, std = flow.transition(model, step["state"], step["timestep"], step["previous_timestep"], self.features())
            kl = gaussian_kl_divergence(mean, mean.detach(), std)
            torch.testing.assert_close(kl, torch.zeros_like(kl))
            replay = flow.log_prob(step["action"], mean, std)
            torch.testing.assert_close(replay, step["old_log_prob"], rtol=0, atol=0)
            expected_std = flow.policy_std * (step["previous_timestep"] - step["timestep"]) ** 0.5
            torch.testing.assert_close(std, torch.full_like(std, expected_std))
        self.assertEqual(trace[-1]["previous_timestep"], 1.0)
        model.value.data.add_(0.01)
        step = trace[0]
        mean, std = flow.transition(model, step["state"], step["timestep"], step["previous_timestep"], self.features())
        log_prob = flow.log_prob(step["action"], mean, std)
        self.assertGreater((log_prob - step["old_log_prob"]).abs().sum().item(), 0)
        (-log_prob.mean()).backward()
        self.assertGreater(model.value.grad.abs().item(), 0)

    def test_differentiable_sampling_matches_inference(self):
        flow = ConditionalFlowMatching(sampling_steps=3)
        model = ConstantVelocity()
        first = flow.sample(model, self.features(), generator=torch.Generator().manual_seed(10))
        second = flow.sample_differentiable(model, self.features(), generator=torch.Generator().manual_seed(10))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        second.sum().backward()
        self.assertTrue(torch.isfinite(model.value.grad))

    def test_masked_flow_loss_and_field_estimate(self):
        flow = ConditionalFlowMatching(sampling_steps=2, field_loss_min_time=0)
        model = ConstantVelocity()
        target = torch.zeros(2, 1, 8, 8)
        losses = flow.training_losses(model, self.features(), target, return_prediction=True)
        self.assertEqual(set(losses), {"flow", "heat", "heat_pred"})
        self.assertTrue(torch.isfinite(losses["flow"]))
        (losses["flow"] + losses["heat"]).backward()
        self.assertIsNotNone(model.value.grad)
        losses = flow.training_losses(model, self.features(), target, valid_mask=torch.zeros_like(target))
        self.assertEqual(losses["flow"].item(), 0)
        self.assertEqual(losses["heat"].item(), 0)

    def test_transition_requires_forward_continuous_time(self):
        flow = ConditionalFlowMatching()
        state = torch.zeros(2, 1, 8, 8)
        for start, end in ((0.5, 0.4), (0.5, 0.5), (-0.1, 0.5), (0.5, 1.1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                flow.transition(ConstantVelocity(), state, start, end, self.features())

    def test_stratified_time_covers_each_bin_and_randomizes_images(self):
        flow = ConditionalFlowMatching(time_sampling="stratified")
        torch.manual_seed(42)
        times = torch.stack([flow._sample_training_time(2, torch.device("cpu")) for _ in range(128)])
        self.assertTrue(((times >= 0) & (times < 1)).all())
        self.assertTrue(((times >= 0.5).sum(dim=1) == 1).all())
        for column in times.T:
            self.assertTrue((column < 0.5).any())
            self.assertTrue((column >= 0.5).any())
        for batch in (1, 3, 4):
            time = flow._sample_training_time(batch, torch.device("cpu"))
            torch.testing.assert_close((time * batch).long().sort().values, torch.arange(batch))

    def test_stratified_field_supervision_keeps_gradients(self):
        flow = ConditionalFlowMatching(time_sampling="stratified")
        model = ConstantVelocity()
        target = torch.zeros(2, 1, 8, 8)
        losses = flow.training_losses(model, self.features(), target)
        self.assertGreater(losses["heat"].item(), 0)
        losses["heat"].backward()
        self.assertTrue(torch.isfinite(model.value.grad))
        self.assertNotEqual(model.value.grad.item(), 0)
        with self.assertRaises(ValueError):
            ConditionalFlowMatching(time_sampling="invalid")

    def test_invalid_config(self):
        for kwargs in ({"policy_std": 0}, {"solver": "ddim"}, {"sampling_steps": 0}, {"time_scale": 0}, {"field_loss_min_time": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConditionalFlowMatching(**kwargs)


if __name__ == "__main__":
    unittest.main()
