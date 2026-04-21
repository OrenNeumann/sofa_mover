"""Tests for encoder forward shape and rotation equivariance."""

from typing import cast

import torch

from sofa_mover.networks import (
    BoundaryEncoder,
    CircularBoundaryEncoder,
)

TEST_DEVICE = torch.device("cpu")


class TestCircularBoundaryEncoder:
    def test_forward_preserves_leading_batch_dims(self) -> None:
        encoder = CircularBoundaryEncoder(n_rays=32, channels=4, depth=2)
        sofa_view = torch.randn(2, 3, 32)
        pose = torch.randn(2, 3, 3)
        progress = torch.randn(2, 3, 1)
        out = encoder(sofa_view, pose, progress)
        assert out.shape == (2, 3, encoder.feature_dim)

    def test_manual_pad_matches_padding_mode_circular(self) -> None:
        """Inline torch.cat pad must produce identical conv outputs to
        nn.Conv1d(padding_mode='circular'). Locks in semantics while we use
        the faster manual variant."""
        torch.manual_seed(0)
        n_angles = 16
        encoder = CircularBoundaryEncoder(
            n_rays=2 * n_angles, channels=4, depth=2, kernel_size=5, stride=1
        )
        encoder.eval()

        # Mirror the encoder's convs with padding_mode="circular" and
        # copy weights over, so only the pad path differs.
        enc_convs = cast(list[torch.nn.Conv1d], list(encoder.convs))
        ref_convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    c.in_channels,
                    c.out_channels,
                    kernel_size=c.kernel_size[0],
                    padding=c.kernel_size[0] // 2,
                    padding_mode="circular",
                )
                for c in enc_convs
            ]
        )
        for c_ref, c_enc in zip(ref_convs, encoder.convs, strict=True):
            c_ref.load_state_dict(c_enc.state_dict())
        ref_convs.eval()

        profile = torch.randn(3, 2, n_angles)
        manual = profile
        reference = profile
        for c_enc, c_ref in zip(encoder.convs, ref_convs, strict=True):
            manual = encoder.relu(c_enc(encoder._circular_pad(manual)))
            reference = encoder.relu(c_ref(reference))
        assert torch.allclose(manual, reference, atol=1e-6)

    def test_conv_stack_is_rotation_equivariant(self) -> None:
        """Rolling the angular axis equally on both channels should shift the
        pre-flatten conv features by the same amount. This is the inductive
        bias the circular padding is supposed to buy us."""
        torch.manual_seed(0)
        n_angles = 16
        encoder = CircularBoundaryEncoder(
            n_rays=2 * n_angles, channels=4, depth=2, stride=1
        )
        encoder.eval()

        captured: dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module, _inp: object, output: torch.Tensor
        ) -> None:
            captured["out"] = output.detach().clone()

        handle = encoder.convs[-1].register_forward_hook(capture)
        try:
            sofa_view = torch.randn(1, 2 * n_angles)
            pose = torch.zeros(1, 3)
            progress = torch.zeros(1, 1)
            encoder(sofa_view, pose, progress)
            feat_original = captured["out"]

            shift = 5
            rolled = sofa_view.view(1, 2, n_angles).roll(shift, dims=-1).reshape(1, -1)
            encoder(rolled, pose, progress)
            feat_rolled = captured["out"]
        finally:
            handle.remove()

        assert torch.allclose(
            feat_rolled, feat_original.roll(shift, dims=-1), atol=1e-6
        )


class TestBoundaryEncoder:
    def test_forward_preserves_leading_batch_dims(self) -> None:
        encoder = BoundaryEncoder(n_rays=32, width=64, depth=2)
        sofa_view = torch.randn(2, 3, 32)
        pose = torch.randn(2, 3, 3)
        progress = torch.randn(2, 3, 1)
        out = encoder(sofa_view, pose, progress)
        assert out.shape == (2, 3, encoder.feature_dim)
