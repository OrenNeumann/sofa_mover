"""Evaluate a trained skrl_sofa policy and render a trajectory GIF.

Mirrors `sofa_mover.evaluate` but loads a skrl-format checkpoint and uses
`SharedSofaModel` for greedy rollout (argmax over categorical logits).
"""

from pathlib import Path

import torch

from sofa_mover.env import make_sofa_env
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)

from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.model import SharedSofaModel

DEFAULT_CHECKPOINT_PATH = "output_skrl/best_policy.pt"
DEFAULT_OUTPUT_PATH = "output_skrl/agent_trajectory.gif"


def evaluate(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    device: torch.device | None = None,
) -> Path:
    training_config_for_device: SkrlTrainingConfig = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["config"]
    device = training_config_for_device.device if device is None else device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_config: SkrlTrainingConfig = checkpoint["config"]
    cfg = training_config.env

    env = make_sofa_env(
        total_frames=training_config.total_frames,
        num_envs=1,
        cfg=cfg,
        device=device,
    )
    assert cfg.observation_type == "boundary"

    normalizer = Normalizer(
        obs_dim=cfg.boundary_rays + 3 + 1,
        num_envs=1,
        device=device,
        norm_obs=training_config.normalize_observation,
        norm_reward=False,
    )
    normalizer.freeze = True
    normalizer.load_state_dict(checkpoint["vec_normalize"])

    # Rebuild the shared actor-critic model and load weights.
    import gymnasium

    obs_space = gymnasium.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(cfg.boundary_rays + 3 + 1,),
        dtype="float32",
    )
    action_space = gymnasium.spaces.Discrete(27)
    model = SharedSofaModel(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        n_rays=cfg.boundary_rays,
        normalizer=normalizer,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    frames: list[FrameData] = []

    def _collect_frame(step: int) -> None:
        pose = (
            env._pose[0, 0].item(),
            env._pose[0, 1].item(),
            env._pose[0, 2].item(),
        )
        sofa = env._sofa[:1].float()
        corridor = env.rasterizer.corridor_mask(env._pose[:1])
        frames.append(compute_frame_data(step, pose, sofa, corridor, env.cell_area))

    with torch.no_grad():
        td = env.reset()

        for step in range(cfg.max_steps):
            _collect_frame(step)
            obs = torch.cat(
                [
                    td["observation", "sofa_view"],
                    td["observation", "pose"],
                    td["observation", "progress"],
                ],
                dim=-1,
            )
            logits, _ = model.compute({"observations": obs}, role="policy")
            action_idx = logits.argmax(dim=-1)
            one_hot = torch.zeros(1, 27, device=device)
            one_hot[0, action_idx] = 1.0
            td["action"] = one_hot

            td = env.step(td)["next"]
            if td["done"].all():
                _collect_frame(step + 1)
                break

    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)
    result = render_trajectory(
        frames, out, sofa_extent=env.sofa_extent, corridor_width=cfg.corridor_width
    )
    print(f"Saved {len(frames)}-frame trajectory to {result}")
    print(f"Final area: {frames[-1].area:.4f}")
    return result


if __name__ == "__main__":
    evaluate()
