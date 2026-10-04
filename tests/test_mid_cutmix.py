"""Semantic and clipped-area tests for independent batch CutMix."""

import importlib
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


class TestMidCutmix(unittest.TestCase):
    def setUp(self):
        self.mix_batch = importlib.import_module("models.mid_cutmix").mix_batch

    def test_lambda_matches_real_replaced_rectangle_including_clipped_edges(self):
        images = torch.tensor([0., 10., 20.], dtype=torch.float64).reshape(3, 1, 1, 1).expand(3, 2, 9, 13).clone()
        labels = torch.tensor([2, 0, 1], dtype=torch.int32)
        permutation = torch.tensor([1, 2, 0])
        original_images, original_labels = images.clone(), labels.clone()
        saw_replacement = saw_edge = False
        for seed in range(16):
            torch.manual_seed(seed)
            with patch("torch.randperm", return_value=permutation):
                mixed, labels_a, labels_b, lam = self.mix_batch(images, labels, 0.7)
            mask = mixed[0, 0] != images[0, 0]
            area = int(mask.sum())
            self.assertEqual(lam, 1 - area / (9 * 13))
            self.assertTrue(torch.equal(mixed, torch.where(mask[None, None], images[permutation], images)))
            self.assertTrue(torch.equal(labels_a, labels))
            self.assertTrue(torch.equal(labels_b, labels[permutation]))
            self.assertTrue(torch.equal(images, original_images))
            self.assertTrue(torch.equal(labels, original_labels))
            self.assertNotEqual(mixed.data_ptr(), images.data_ptr())
            self.assertEqual(mixed.dtype, images.dtype)
            self.assertEqual(mixed.device, images.device)
            self.assertEqual(labels_b.dtype, labels.dtype)
            self.assertEqual(labels_b.device, labels.device)
            if area:
                saw_replacement = True
                rows, cols = mask.nonzero(as_tuple=True)
                self.assertEqual(area, int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)))
                saw_edge |= bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        self.assertTrue(saw_replacement, "fixture must exercise nonempty replacement")
        self.assertTrue(saw_edge, "fixture must exercise a rectangle clipped at an edge")

    def test_torch_seed_reproduces_images_pairing_and_lambda(self):
        images = torch.arange(5., dtype=torch.float32).reshape(5, 1, 1, 1).expand(5, 2, 17, 23).clone()
        outputs = []
        for seed in (123, 123, 124):
            torch.manual_seed(seed)
            outputs.append(self.mix_batch(images, torch.arange(5), 1.2))
        for first, second in zip(outputs[0][:3], outputs[1][:3]):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(outputs[0][3], outputs[1][3])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(outputs[0][:3], outputs[2][:3])))

    def test_gradient_follows_source_outside_and_paired_sample_inside_rectangle(self):
        images = torch.tensor([0., 10., 20.], dtype=torch.float64).reshape(3, 1, 1, 1).expand(3, 2, 9, 13).clone().requires_grad_()
        weights = torch.tensor([1., 3., 7.], dtype=images.dtype).reshape(3, 1, 1, 1)
        torch.manual_seed(42)
        with patch("torch.randperm", return_value=torch.tensor([1, 2, 0])):
            mixed, _, _, lam = self.mix_batch(images, torch.arange(3), 0.7)
        self.assertLess(lam, 1)
        mask = mixed.detach()[0, 0] != images.detach()[0, 0]
        (mixed * weights).sum().backward()
        expected = torch.where(mask[None, None], weights[torch.tensor([2, 0, 1])], weights).expand_as(images)
        torch.testing.assert_close(images.grad, expected)

    def test_weighted_cross_entropy_uses_corrected_lambda_and_pairing(self):
        images = torch.tensor([0., 10., 20.]).reshape(3, 1, 1, 1).expand(3, 1, 9, 13).clone()
        torch.manual_seed(42)
        with patch("torch.randperm", return_value=torch.tensor([1, 2, 0])):
            mixed, labels_a, labels_b, lam = self.mix_batch(images, torch.tensor([2, 0, 1]), 0.7)
        actual_fraction = float((mixed[0, 0] != images[0, 0]).sum()) / 117
        self.assertGreater(actual_fraction, 0)
        logits = torch.tensor([[2., -1., 0.], [0., 3., 1.], [-2., 1., 4.]], requires_grad=True)
        loss = lam * F.cross_entropy(logits, labels_a) + (1 - lam) * F.cross_entropy(logits, labels_b)
        logp = logits.log_softmax(dim=1)
        expected = -((1 - actual_fraction) * (logp[0, 2] + logp[1, 0] + logp[2, 1])
                     + actual_fraction * (logp[0, 0] + logp[1, 1] + logp[2, 2])) / 3
        torch.testing.assert_close(loss, expected)
        actual_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_partial_singleton_tiny_and_noncontiguous_batches_preserve_dtype(self):
        for batch in (1, 2, 3):
            for height, width in ((1, 1), (1, 7), (7, 1), (3, 4)):
                for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    with self.subTest(batch=batch, size=(height, width), dtype=dtype):
                        images = torch.arange(batch * 2 * height * width, dtype=dtype).reshape(batch, 2, height, width).transpose(2, 3)
                        original = images.clone()
                        labels = torch.arange(batch)
                        torch.manual_seed(8)
                        mixed, labels_a, labels_b, lam = self.mix_batch(images, labels, 0.5)
                        self.assertEqual(mixed.shape, images.shape)
                        self.assertEqual(mixed.dtype, dtype)
                        self.assertTrue(torch.equal(images, original))
                        self.assertTrue(torch.equal(labels_a, labels))
                        self.assertEqual(sorted(labels_b.tolist()), labels.tolist())
                        self.assertTrue(0 <= lam <= 1)
                        if batch == 1:
                            self.assertTrue(torch.equal(mixed, images))

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
