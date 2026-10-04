"""Semantic tests for independent batch Mixup augmentation."""

import importlib
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


class TestMidMixup(unittest.TestCase):
    def setUp(self):
        self.mix_batch = importlib.import_module("models.mid_mixup").mix_batch

    def test_convex_combination_matches_returned_pairing_and_preserves_inputs(self):
        images = torch.arange(72, dtype=torch.float64).reshape(3, 2, 3, 4)
        labels = torch.tensor([2, 0, 1], dtype=torch.int32)
        original_images, original_labels = images.clone(), labels.clone()
        permutation = torch.tensor([1, 2, 0])
        torch.manual_seed(42)
        with patch("torch.randperm", return_value=permutation):
            mixed, labels_a, labels_b, lam = self.mix_batch(images, labels, 0.7)
        self.assertGreater(lam, 0)
        self.assertLess(lam, 1)
        torch.testing.assert_close(mixed, lam * images + (1 - lam) * images[permutation])
        self.assertTrue(torch.equal(labels_a, labels))
        self.assertTrue(torch.equal(labels_b, labels[permutation]))
        self.assertTrue(torch.equal(images, original_images))
        self.assertTrue(torch.equal(labels, original_labels))
        self.assertNotEqual(mixed.data_ptr(), images.data_ptr())
        self.assertEqual(mixed.dtype, images.dtype)
        self.assertEqual(mixed.device, images.device)
        self.assertEqual(labels_b.dtype, labels.dtype)
        self.assertEqual(labels_b.device, labels.device)

    def test_torch_seed_reproduces_images_pairing_and_lambda(self):
        images = torch.arange(160, dtype=torch.float32).reshape(5, 2, 4, 4)
        labels = torch.arange(5)
        outputs = []
        for seed in (123, 123, 124):
            torch.manual_seed(seed)
            outputs.append(self.mix_batch(images, labels, 1.2))
        for first, second in zip(outputs[0][:3], outputs[1][:3]):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(outputs[0][3], outputs[1][3])
        self.assertNotEqual(outputs[0][3], outputs[2][3])

    def test_gradient_reaches_both_members_of_each_pair(self):
        images = torch.arange(24, dtype=torch.float64).reshape(3, 1, 2, 4).requires_grad_()
        permutation = torch.tensor([1, 2, 0])
        weights = torch.tensor([1., 3., 7.], dtype=images.dtype).reshape(3, 1, 1, 1)
        torch.manual_seed(5)
        with patch("torch.randperm", return_value=permutation):
            mixed, _, _, lam = self.mix_batch(images, torch.arange(3), 0.8)
        (mixed * weights).sum().backward()
        expected = (lam * weights + (1 - lam) * weights[torch.tensor([2, 0, 1])]).expand_as(images)
        torch.testing.assert_close(images.grad, expected)

    def test_weighted_cross_entropy_uses_actual_pairing(self):
        torch.manual_seed(42)
        images = torch.arange(24, dtype=torch.float32).reshape(3, 1, 2, 4)
        labels = torch.tensor([2, 0, 1])
        with patch("torch.randperm", return_value=torch.tensor([1, 2, 0])):
            _, labels_a, labels_b, lam = self.mix_batch(images, labels, 0.7)
        logits = torch.tensor([[2., -1., 0.], [0., 3., 1.], [-2., 1., 4.]], requires_grad=True)
        loss = lam * F.cross_entropy(logits, labels_a) + (1 - lam) * F.cross_entropy(logits, labels_b)
        logp = logits.log_softmax(dim=1)
        expected = -(lam * (logp[0, 2] + logp[1, 0] + logp[2, 1])
                     + (1 - lam) * (logp[0, 0] + logp[1, 1] + logp[2, 2])) / 3
        torch.testing.assert_close(loss, expected)
        actual_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_partial_singleton_and_noncontiguous_batches_preserve_dtype(self):
        for batch in (1, 2, 3):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(batch=batch, dtype=dtype):
                    images = torch.arange(batch * 24, dtype=dtype).reshape(batch, 2, 3, 4).transpose(2, 3)
                    labels = torch.arange(batch)
                    torch.manual_seed(8)
                    mixed, labels_a, labels_b, lam = self.mix_batch(images, labels, 0.5)
                    self.assertEqual(mixed.shape, images.shape)
                    self.assertEqual(mixed.dtype, dtype)
                    self.assertTrue(torch.equal(labels_a, labels))
                    self.assertEqual(sorted(labels_b.tolist()), labels.tolist())
                    self.assertTrue(0 <= lam <= 1)
                    if batch == 1:
                        torch.testing.assert_close(mixed, images)

    def test_invalid_alpha_is_rejected(self):
        for alpha in (0, -0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.mix_batch(torch.zeros(2, 1, 3, 4), torch.arange(2), alpha)

    def test_invalid_rank_batch_shape_and_dtypes_are_rejected(self):
        cases = [
            (torch.zeros(2, 3, 4), torch.arange(2)),
            (torch.zeros(2, 1, 1, 3, 4), torch.arange(2)),
            (torch.zeros(2, 1, 3, 4), torch.arange(2).reshape(2, 1)),
            (torch.zeros(2, 1, 3, 4), torch.arange(3)),
            (torch.zeros(0, 1, 3, 4), torch.empty(0, dtype=torch.long)),
            (torch.zeros(2, 1, 0, 4), torch.arange(2)),
            (torch.zeros(2, 0, 3, 4), torch.arange(2)),
            (torch.zeros(2, 1, 3, 0), torch.arange(2)),
            (torch.zeros(2, 1, 3, 4, dtype=torch.int32), torch.arange(2)),
            (torch.zeros(2, 1, 3, 4), torch.arange(2, dtype=torch.float32)),
            (torch.zeros(2, 1, 3, 4), torch.tensor([True, False])),
        ]
        for images, labels in cases:
            with self.subTest(shape=images.shape, labels=labels.shape, dtype=labels.dtype):
                with self.assertRaises(ValueError):
                    self.mix_batch(images, labels, 0.5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_preserves_device_gradients_and_rejects_mismatched_devices(self):
        images = torch.randn(3, 2, 4, 5, device="cuda", requires_grad=True)
        labels = torch.arange(3, device="cuda")
        torch.manual_seed(42)
        mixed, labels_a, labels_b, _ = self.mix_batch(images, labels, 0.8)
        self.assertEqual(mixed.device, images.device)
        self.assertEqual(labels_a.device, labels.device)
        self.assertEqual(labels_b.device, labels.device)
        mixed.sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        with self.assertRaises(ValueError):
            self.mix_batch(images, labels.cpu(), 0.8)


if __name__ == "__main__":
    unittest.main()
