# Tutorials

Step-by-step notebooks, from a first simulation to training and evaluating a
learning agent. They are saved with their outputs, so you can read them on
GitHub without running anything.

| notebook | what you learn | runtime |
|---|---|---|
| [`01_first_market.ipynb`](01_first_market.ipynb) | Run a scenario; prices, liquidity and funding; who made money; fat tails and volatility clustering; liquidations; what agents can see at each transparency level; saving and replaying runs. | ~15 s |
| [`02_agents_and_experiments.ipynb`](02_agents_and_experiments.ipynb) | Write an agent that reads the crowd; a paired experiment (ANONYMOUS vs FULL) over several seeds; why paired seeds matter; comparing setups; an exogenous crash as an intervention. | ~1 min |
| [`03_reinforcement_learning.ipynb`](03_reinforcement_learning.ipynb) | The Gymnasium environment and its exit-value reward; a short MaskablePPO run; evaluation on held-out seeds against a scripted market maker, random and hold; the trained agent inside any market; custom observation, ladder actions and reward; several learners through PettingZoo. | ~3 min |

To run them:

```bash
pip install -e ".[notebooks]"     # the simulator, RL, pandas, matplotlib and JupyterLab
jupyter lab notebooks/
```

Notebooks 1 and 2 only need `pip install -e ".[analysis]"` plus Jupyter.
Runs and trained agents created by the notebooks are written to
`notebooks/runs/`, which git ignores.
