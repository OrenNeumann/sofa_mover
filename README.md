# Sofa Mover

## Training

```bash
uv run python -m sofa_mover.train
```

### Observation modes

Three configurations trade off memory/speed vs observation richness:

| Mode | Observation | Max batch size | Notes |
|------|------------|---------------|-------|
| `baseline` | Grid (1, 256, 171) uint8 | ~256 | Full resolution, safest for learning |
| `safe` | Grid (1, 128, 85) uint8 | ~512 | 2x downsampled, good balance |
| `aggressive` | Boundary (128,) float32 | ~1024 | Polar ray-cast, fastest, least spatial info |

```bash
# Default (aggressive)
uv run python -m sofa_mover.train

# Downsampled grid
uv run python -m sofa_mover.train --obs-mode safe

# Boundary representation
uv run python -m sofa_mover.train --obs-mode aggressive

# Auto-detect max batch size for GPU
uv run python -m sofa_mover.train --obs-mode safe --num-envs auto
```

### Monitoring

```bash
tensorboard --logdir runs/sofa_ppo
```

## Visualization

A trajectory GIF is generated automatically at the end of training. To visualize a trained model manually:

```bash
uv run python -m sofa_mover.evaluate
```

This loads `output/best_policy.pt` and renders a greedy rollout to `output/agent_trajectory.gif`.

## Benchmarking

Find max batch size and FPS for all three obs modes:

```bash
uv run python bench_optimization.py
```
