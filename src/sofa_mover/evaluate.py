"""Evaluate a trained policy and render a trajectory video."""

from pathlib import Path

import torch

from sofa_mover.env import make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaBoundaryEncoder, SofaEncoder
from sofa_mover.training.config import DEVICE, SofaEnvConfig, TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)
from tensordict.nn import TensorDictModule
from torchrl.modules import OneHotCategorical, ProbabilisticActor


def evaluate(
    checkpoint_path: str = "output/best_policy.pt",
    output_path: str = "output/agent_trajectory.gif",
    device: torch.device | None = None,
) -> Path:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE if device is None else device,
        weights_only=False,
    )
    training_config: TrainingConfig = checkpoint["config"]
    cfg: SofaEnvConfig = training_config.env
    device = training_config.device if device is None else device

    env = make_sofa_env(num_envs=1, cfg=cfg, device=device)
    normalizer = Normalizer.from_config(training_config, num_envs=1)
    normalizer.freeze = True
    if "vec_normalize" in checkpoint:
        normalizer.load_state_dict(checkpoint["vec_normalize"])

    # Rebuild actor with the correct encoder for this checkpoint's config
    encoder: SofaEncoder | SofaBoundaryEncoder
    if cfg.observation_type == "boundary":
        encoder = SofaBoundaryEncoder(
            n_rays=cfg.boundary_rays,
            normalizer=normalizer,
        )
    else:
        encoder = SofaEncoder()
    actor_net = SofaActorNet(encoder=encoder).to(device)
    actor_module = TensorDictModule(
        actor_net,
        in_keys=[
            ("observation", "sofa_view"),
            ("observation", "pose"),
            ("observation", "progress"),
        ],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # Deterministic rollout — pose lives in env._pose (internal state)
    H = cfg.sofa_config.grid_size
    frames: list[FrameData] = []

    def _collect_frame(step: int) -> None:
        pose_tuple = (
            env._pose[0, 0].item(),
            env._pose[0, 1].item(),
            env._pose[0, 2].item(),
        )
        # Reconstruct full grid from cropped sofa for visualization
        full_sofa = torch.zeros(1, 1, H, H, device=device)
        full_sofa[:, :, env._crop_y, env._crop_x] = env._sofa.float()
        corridor_full = env.rasterizer.corridor_mask(env._pose)
        frames.append(
            compute_frame_data(
                step, pose_tuple, full_sofa, corridor_full, cfg.sofa_config
            )
        )

    with torch.no_grad():
        td = env.reset()

        for step in range(cfg.max_steps):
            _collect_frame(step)

            # Greedy action: argmax logits
            td = actor_module(td)
            logits = td["logits"]
            action = torch.zeros_like(logits)
            action.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            td["action"] = action

            td = env.step(td)["next"]

            if td["done"].all():
                _collect_frame(step + 1)
                break

    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)
    result = render_trajectory(frames, cfg.sofa_config, out, fps=10)
    print(f"Saved {len(frames)}-frame trajectory to {result}")
    print(f"Final area: {frames[-1].area:.4f}")
    return result


if __name__ == "__main__":
    evaluate()
