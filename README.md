# Welcome to JaxUED!
<p align="center">
<a href="#get-started">Get Started</a> &mdash; <a href="https://arxiv.org/abs/2403.13091">Paper</a> &mdash; <a href="https://dramacow.github.io/jaxued/">Docs</a>
</p>

<div align="center">
    <img src="figures/Labyrinth_299.gif" >
    <img src="figures/StandardMaze3_299.gif">
    <img src="figures/SixteenRooms_299.gif">
    <img src="figures/StandardMaze2_299.gif">
    <br>
    <img src="figures/SixteenRooms2_299.gif">
    <img src="figures/Labyrinth2_299.gif">
    <img src="figures/StandardMaze_299.gif">
    <img src="figures/LabyrinthFlipped_299.gif">
</div>

JaxUED is a Unsupervised Environment Design (UED) library with similar goals to [CleanRL](https://docs.cleanrl.dev): high-quality, single-file and understandable implementations of common UED methods.

## Why JaxUED?
- Single-file reference implementations of common-UED algorithms
- Allows easy modification and quick prototyping of ideas
- Understandable code with low levels of obscurity/abstraction
- Wandb integration and logging of metrics and generated levels

### What We Provide
JaxUED has several (Jaxified) utilities that are useful for implementing UED
methods, a `LevelSampler`, a general environment interface
`UnderspecifiedEnv`, typed environment-utility metric interfaces, and a Maze
implementation.

Environment utility measurement is kept separate from level storage and
sampling. The existing MaxMC and positive value-loss metrics are available from
`jaxued.metrics`, and compatible project-specific metrics can be selected
through `MetricRegistry`.

We also have understandable single-file implementations of DR, PLR, ACCEL,
PAIRED, and persistent-ensemble PLR with virtual learning-progress scoring.

### Who JaxUED is for
JaxUED is primarily intended for researchers looking to get *in the weeds* of UED algorithm development. Our minimal dependency implementations of the current state-of-the art UED methods expose all implementation details; helping researchers understand how the algorithms work in practise, and facilitating easy, rapid prototyping of new ideas. 

## Get Started
See the [docs](https://dramacow.github.io/jaxued/) for more examples and [explanations of arguments](https://dramacow.github.io/jaxued/maze_dr/), or simply read the documented code in `examples/`
### Installation

To install the core package:
```bash
pip install jaxued
```
To install optional dependencies required for the example scripts (found in the examples/ directory):
```bash
pip install "jaxued[examples]"
```

Follow instructions [here](https://jax.readthedocs.io/en/latest/installation.html) for jax GPU installation, and run something like the following 
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Training
The Maze examples include `examples/maze_{dr,plr,paired}.py` plus
`examples/maze_ensemble_plr.py`, which implements ensemble disagreement
reduction with isolated virtual PPO updates.

To run them, simply run the scripts directly. See the [documentation](https://dramacow.github.io/jaxued/) or the files themselves for arguments.

```bash
python examples/maze_plr.py
```

#### Modal launcher

The launchers in `modal-run/maze/` run on one L40S by default, log to W&B in
online mode, and persist checkpoints in the `jaxued-checkpoints` Modal Volume.
They share image, secret, volume, and typed training configuration in
`modal-run/maze/_common.py`.

```bash
uv pip install -e ".[modal]"
modal setup
modal secret create wandb-secret --from-dotenv .env
modal run modal-run/maze/robust_plr.py
modal run modal-run/maze/robust_plr_three_seeds.py
modal run modal-run/maze/mean_absolute_advantage_three_seeds.py
modal run modal-run/maze/mean_positive_delight_three_seeds.py
modal run modal-run/maze/plr_plus.py
modal run modal-run/maze/ensemble_epistemic_uncertainty.py
```

Arguments can be overridden through the local entrypoint, for example:

```bash
modal run modal-run/maze/robust_plr.py --seed 1 --run-name maze_robust_plr_seed_1
```

The three-seed launcher runs seeds `0,1,2` concurrently in one W&B group and
stores checkpoints under separate seed directories. Override the exact seeds
or use a short smoke-test budget with:

```bash
modal run modal-run/maze/robust_plr_three_seeds.py \
  --seeds 3,5,8 \
  --num-updates 100
```

Run the same three-seed Robust PLR configuration with mean absolute advantage
as its replay score using:

```bash
modal run modal-run/maze/mean_absolute_advantage_three_seeds.py \
  --run-name maze_mean_absolute_advantage_three_seeds \
  --seeds 0,1,2 \
  --num-updates 30000
```

Run the matching three-seed configuration with mean positive delight as its
replay score using:

```bash
modal run modal-run/maze/mean_positive_delight_three_seeds.py \
  --run-name maze_mean_positive_delight_three_seeds \
  --seeds 0,1,2 \
  --num-updates 30000
```

`plr_plus.py` selects the existing PLR mode with exploratory gradient updates.
The ensemble launcher defaults to 8 agents, 3 isolated virtual rollout-PPO
phases, 5 virtual PPO epochs per phase, a virtual level batch size of 32 on
L40S, and a checkpoint interval of 2,500 training updates. Both
single-agent launchers evaluate every 200 updates and checkpoint every 10
evaluation cycles, which is 2,000 training updates.

```bash
modal run modal-run/maze/ensemble_epistemic_uncertainty.py \
  --virtual-rollout-phases 3 \
  --virtual-epoch-ppo 5 \
  --virtual-level-batch-size 32
```

To sweep virtual level batch sizes `1, 2, 4, 8, 16, 32` on both supported
benchmark GPUs:

```bash
modal run modal-run/maze/benchmark_ensemble_epistemic.py --gpu both
```

The sweep reports compilation time, median and p95 steady-state latency,
scoring updates per second, peak VRAM, and updates per dollar. Apply its
fastest non-OOM values in `VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU`. The L40S default
uses all 32 candidate levels; the RTX PRO 6000 default remains 4 until its
sweep is recorded.

Set `JAXUED_MODAL_GPU` before invoking `modal run` to select another supported
GPU type.

### Evaluation
After the training is completed, it will store checkpoints in `./checkpoints/<run_name>/<seed>/models/<update_step>`, and the same file can be run to evaluate these checkpoints, storing the evaluation results in `./results/`.
The only things to change are `--mode=eval`, specifying `--checkpoint_directory` and `--checkpoint_to_eval`:
```
python examples/maze_plr.py --mode eval --checkpoint_directory=./checkpoints/<run_name>/<seed> --checkpoint_to_eval <update_step>
```


These result files are `npz` files with keys
```
states, cum_rewards, episode_lengths, levels
```


## Supported Methods
| Method                                                                              | Run Command                                                  |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [Domain Randomization (DR)](https://arxiv.org/abs/1703.06907)                       | `python examples/maze_dr.py`                             |
| [Prioritized Level Replay (PLR)](https://arxiv.org/abs/2010.03934)                  | `python examples/maze_plr.py --exploratory_grad_updates` |
| [Robust Prioritized Level Replay (RPLR)](https://arxiv.org/abs/2110.02439) | `python examples/maze_plr.py`                            |
| [ACCEL](https://arxiv.org/abs/2203.01302)                                           | `python examples/maze_plr.py --use_accel`                |
| [PAIRED](https://arxiv.org/abs/2012.02096)                                          | `python examples/maze_paired.py`                         |
| Ensemble virtual learning progress                                                  | `python examples/maze_ensemble_plr.py --num_agents 8`    |

## Modification
One of the core goals of JaxUED is that our reference implementations can easily be modified to add arbitrary functionality. All of the primary functionality is provided in the file, from the PPO implementation to the specifics of each method. 

So, to get started, simply copy one of the files, and start modifying the file directly.

## New Environments
To implement a new environment, simply subclass the `UnderspecifiedEnv` interface, and in the files themselves, change
```python
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
```

to 
```
    env = MyEnv(...)
```

And make any other changes necessary to the network architecture, etc.
## Supported Environments
### Craftax
`examples/craftax/craftax_plr.py` contains code to run DR, PLR and ACCEL in [Craftax](https://github.com/MichaelTMatthews/Craftax).
To use Craftax, install it using 
```bash
pip install git+https://github.com/MichaelTMatthews/Craftax.git@main
```

Run it using the following command (see [here](https://dramacow.github.io/jaxued/craftax/) for the full list of arguments):

```
python examples/craftax/craftax_plr.py --exploratory_grad_updates --num_train_envs 512 --num_updates 256
```

Currently, this only supports CraftaxSymbolic, but the following are coming soon:

- [ ] Support for Pixel Environments
- [ ] Support for Craftax-Classic
- [ ] Support for an RNN policy

### Gymnax
See `examples/gymnax/gymnax_plr.py` to run gymnax environments, currently supporting Acrobot, Pendulum and Cartpole. Use the `--env` flag with the name of the environment in lowercase to choose which is used. We have set the distribution of levels as somewhat arbitrary, changing two of the parameters of each environments (e.g. length and mass in Cartpole). This can easily be changed, however. The evaluation distribution is also somewhat arbitrary and can be easily changed.

The `examples/gymnax/gymnax_plr.py` can be modified to add additional environments as well.

## See Also
Here are some other libraries that also leverage Jax to obtain massive speedups in RL, which acted as inspiration for JaxUED.

RL Algorithms in Jax
- [Minimax](https://github.com/facebookresearch/minimax): UED baselines, with support for multi-gpu training, and more parallel versions of PLR/ACCEL
- [PureJaxRL](https://github.com/luchris429/purejaxrl) End-to-end RL implementations in Jax
- [JaxIRL](https://github.com/FLAIROx/jaxirl): Inverse RL
- [Mava](https://github.com/instadeepai/Mava): Multi-Agent RL
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): Lots of different multi-agent RL algorithms

RL Environments in Jax
- [Gymnax](https://github.com/RobertTLange/gymnax): Standard RL interface with several environments, such as classic control and MinAtar.
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): Lots of different multi-agent RL environments
- [Jumanji](https://github.com/instadeepai/jumanji): Combinatorial Optimisation
- [Pgx](https://github.com/sotetsuk/pgx): Board games, such as Chess and Go.
- [Brax](https://github.com/google/brax): Continuous Control (like Mujoco), in Jax
- [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid): Meta RL environments, taking ideas from XLand and Minigrid
- [Craftax](https://github.com/MichaelTMatthews/Craftax): Greatly extended version of [Crafter](https://github.com/danijar/crafter) in Jax.

## Projects using JaxUED
- [Craftax](https://github.com/MichaelTMatthews/Craftax): Using UED to generate worlds for learning an RL agent.
- [ReMiDi](https://github.com/Michael-Beukman/ReMiDi): JaxUED is the primary library used for baselines and the backbone for implementing ReMiDi.

## 📜 Citation
For attribution in academic contexts, please cite this work as
```
@article{coward2024JaxUED,
  title={JaxUED: A simple and useable UED library in Jax},
  author={Samuel Coward and Michael Beukman and Jakob Foerster},
  journal={arXiv preprint},
  year={2024},
}
```
