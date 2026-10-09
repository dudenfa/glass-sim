# Examples

Each script is self-contained and runs from the repository root with the
package installed (`pip install -e ".[analysis]"`; example 1 prints pandas
frames):

```bash
python examples/01_quickstart.py
```

| script | what it shows | runtime |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Run the reference scenario for one simulated hour, print the report, read the summary, the best account, the stylized facts and the pandas frames. | ~2 s |
| [`02_custom_agent.py`](02_custom_agent.py) | Write an agent (`FadeTheCrowd` bets against the aggregate retail position, which only FULL transparency reveals) and add two of them to a scenario with `AgentSpec`. | ~5 s |
| [`03_transparency_experiment.py`](03_transparency_experiment.py) | A paired experiment: the same seeds under ANONYMOUS, ATTRIBUTED and FULL transparency, with medians of market metrics and returns by participant type. `python examples/03_transparency_experiment.py 10` uses 10 seeds. | ~1 min for 5 seeds |
| [`04_external_agent_loop.py`](04_external_agent_loop.py) | Drive an agent from outside the simulator with `run_until_decision()` / `act()`: the observation, action and reward loop that the RL environment is built on. | ~1 s |

The [user guide](../docs/user-guide.md) explains the concepts these examples
use; the RL layer (`glass train`, `glass evaluate`, `glass_sim.rl`) is
documented there as well.
