"""Evaluate a trained policy and render a trajectory video.

Two trajectories may be rendered:
- The greedy rollout of the saved policy (argmax of the action logits).
- The recorded best-during-training trajectory, replayed from its stored
  action sequence. This is the trajectory that produced
  `best_trajectory_area` at the moment the checkpoint was saved.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.networks import (
    MultiDiscreteCategorical,
    SofaActorNet,
    build_encoder,
)
from sofa_mover.training.config import DEVICE, TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)

DEFAULT_CHECKPOINT_PATH = "output/best_policy.pt"
DEFAULT_OUTPUT_PATH = "output/agent_trajectory.gif"


def _collect_frame(env: SofaEnv, step: int) -> FrameData:
    pose = (
        env._pose[0, 0].item(),
        env._pose[0, 1].item(),
        env._pose[0, 2].item(),
    )
    sofa = env._sofa[:1].float()
    corridor = env.rasterizer.corridor_mask(env._pose[:1])
    return compute_frame_data(step, pose, sofa, corridor, env.cell_area)


def _rollout(
    env: SofaEnv,
    action_fn: Callable[[TensorDictBase, int], torch.Tensor],
    max_steps: int,
) -> list[FrameData]:
    frames: list[FrameData] = []
    with torch.no_grad():
        td = env.reset()
        for step in range(max_steps):
            frames.append(_collect_frame(env, step))
            td["action"] = action_fn(td, step)
            td = cast(TensorDictBase, env.step(td)["next"])
            if td["done"].all():
                frames.append(_collect_frame(env, step + 1))
                break
    return frames


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

    nvec = [2 * cfg.n_magnitude_levels + 1] * 3

    encoder = build_encoder(training_config, normalizer)
    actor_net = SofaActorNet(
        nvec=nvec,
        encoder=encoder,
        width=training_config.head_width,
        depth=training_config.head_depth,
    ).to(device)
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
        distribution_class=MultiDiscreteCategorical,
        distribution_kwargs={"nvec": nvec},
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)

    def greedy_action(td: TensorDictBase, step: int) -> torch.Tensor:
        logits = actor_module(td)["logits"]
        one_hots = [
            torch.zeros(logits.shape[0], n, device=device).scatter_(
                1, lg.argmax(dim=-1, keepdim=True), 1.0
            )
            for lg, n in zip(logits.split(nvec, dim=-1), nvec)
        ]
        return torch.cat(one_hots, dim=-1)

    greedy_frames = _rollout(env, greedy_action, cfg.max_steps)
    greedy_path = render_trajectory(
        greedy_frames,
        out,
        sofa_extent=env.sofa_extent,
        corridor_width=cfg.corridor_width,
    )
    print(
        f"Greedy: saved {len(greedy_frames)}-frame trajectory to {greedy_path} "
        f"(final area {greedy_frames[-1].area:.4f})"
    )

    if (
        "best_trajectory_actions" in checkpoint
        and checkpoint["best_trajectory_actions"] is not None
    ):
        replay_actions: torch.Tensor = checkpoint["best_trajectory_actions"].to(device)
        replay_frames = _rollout(
            env,
            lambda _td, step: replay_actions[step].unsqueeze(0),
            max_steps=len(replay_actions),
        )
        replay_path = render_trajectory(
            replay_frames,
            out.with_name(out.stem + "_best_train.gif"),
            sofa_extent=env.sofa_extent,
            corridor_width=cfg.corridor_width,
        )
        recorded = checkpoint["best_trajectory_area"]
        print(
            f"Best-during-training: saved {len(replay_frames)}-frame trajectory to "
            f"{replay_path} (replay final area {replay_frames[-1].area:.4f}, "
            f"recorded {recorded:.4f})"
        )

    return greedy_path


if __name__ == "__main__":
    evaluate()
