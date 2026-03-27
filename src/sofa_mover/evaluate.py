"""Evaluate a trained policy and render a trajectory video."""

from pathlib import Path

import torch

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig, make_sofa_env
from sofa_mover.networks import SofaActorNet
from sofa_mover.obs_mode import make_encoder
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
    device: torch.device = DEVICE,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: SofaEnvConfig = checkpoint["cfg"]

    # Single-env for visualization
    env = make_sofa_env(num_envs=1, cfg=cfg, device=device)

    # Rebuild actor with the correct encoder for this checkpoint's config
    encoder = make_encoder(cfg)
    actor_net = SofaActorNet(encoder=encoder).to(device)
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "pose", "progress"],
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
    frames: list[FrameData] = []
    with torch.no_grad():
        td = env.reset()

        for step in range(cfg.max_steps):
            pose_tuple = (
                env._pose[0, 0].item(),
                env._pose[0, 1].item(),
                env._pose[0, 2].item(),
            )
            # For boundary mode, obs is 1D — use internal sofa grid for viz
            sofa = env._sofa.float()  # (1, 1, crop_h, crop_w)
            corridor_full = env.rasterizer.corridor_mask(env._pose)
            mask = env._crop(corridor_full)
            frames.append(
                compute_frame_data(step, pose_tuple, sofa, mask, cfg.sofa_config)
            )

            # Greedy action: argmax logits
            td = actor_module(td)
            logits = td["logits"]
            action = torch.zeros_like(logits)
            action.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            td["action"] = action

            td = env.step(td)["next"]

            if td["done"].all():
                # Final frame
                pose_tuple = (
                    env._pose[0, 0].item(),
                    env._pose[0, 1].item(),
                    env._pose[0, 2].item(),
                )
                sofa = env._sofa.float()
                corridor_full = env.rasterizer.corridor_mask(env._pose)
                mask = env._crop(corridor_full)
                frames.append(
                    compute_frame_data(
                        step + 1, pose_tuple, sofa, mask, cfg.sofa_config
                    )
                )
                break

    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)
    result = render_trajectory(frames, cfg.sofa_config, out, fps=10)
    print(f"Saved {len(frames)}-frame trajectory to {result}")
    print(f"Final area: {frames[-1].area:.4f}")
    return result


if __name__ == "__main__":
    evaluate()
