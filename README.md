# Sofa Mover

## Training

```bash
uv run python -m sofa_mover.train
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/sofa_ppo
```

## Visualization

A trajectory GIF is generated automatically at the end of training. To visualize a trained model manually:

```bash
uv run python -m sofa_mover.evaluate
```

This loads `output/best_policy.pt` and renders a greedy rollout to `output/agent_trajectory.gif`.
