# Sofa Mover

## Training

```bash
uv run python -m sofa_mover.training.train
```

## Visualization

A trajectory GIF is generated automatically at the end of training. To visualize a trained model manually:

```bash
uv run python -m sofa_mover.evaluate
```

This loads `output/best_policy.pt` and renders a greedy rollout to `output/agent_trajectory.gif`.

## Benchmarking

Find max batch size and FPS for three representative observation configs: (temporary)

```bash
uv run python bench_optimization.py
```
