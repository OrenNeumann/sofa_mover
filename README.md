# Sofa Mover

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![Status](https://img.shields.io/badge/status-active%20development-yellow)

A reinforcement learning environment for solving the Sofa-Moving Problem.


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

To render a specific checkpoint after training:

```bash
uv run python -c "from sofa_mover.evaluate import evaluate; evaluate('output/final_policy.pt', 'output/final_policy.gif')"
```


## Profiling

To generate a flame graph for the default training run:

```bash
uv run --with py-spy python -m sofa_mover.training.flamegraph
```

This writes `output/default_training_flamegraph.svg`. Open it in a browser to inspect the interactive flame graph.
