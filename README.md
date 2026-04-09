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
