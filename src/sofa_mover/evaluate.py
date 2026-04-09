"""Evaluate a trained policy and render a trajectory video."""

from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import OneHotCategorical, ProbabilisticActor

from sofa_mover.env import make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaBoundaryEncoder, SofaEncoder
from sofa_mover.training.config import DEVICE, TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)

DEFAULT_CHECKPOINT_PATH = "output/best_policy.pt"
DEFAULT_OUTPUT_PATH = "output/agent_trajectory.gif"


def evaluate(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    device: torch.device | None = None,
) -> Path:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE if device is None else device,
        weights_only=False,
    )
    training_config: TrainingConfig = checkpoint["config"]
    cfg = training_config.env
    device = training_config.device if device is None else device

    env = make_sofa_env(
        total_frames=training_config.total_frames,
        num_envs=1,
        cfg=cfg,
        device=device,
    )
    normalizer = Normalizer.from_config(training_config, num_envs=1, device=device)
    normalizer.freeze = True
    if "vec_normalize" in checkpoint:
        normalizer.load_state_dict(checkpoint["vec_normalize"])

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
    result = render_trajectory(
        frames, out, sofa_extent=env.sofa_extent, corridor_width=cfg.corridor_width
    )
    print(f"Saved {len(frames)}-frame trajectory to {result}")
    print(f"Final area: {frames[-1].area:.4f}")
    return result


if __name__ == "__main__":
    evaluate()
