# Learning Adaptive Safety for Multi-Agent Systems

Preprint and supplementary material available [online](https://arxiv.org/abs/2309.10657).


[![Video](docs/video_thumbnail.png)](https://youtu.be/NDOsWzt1xWo?si=kloCob3V9R_BJRBW)

# Installation

The implementation has been tested with `Python 3.8` under `Ubuntu 20.04`.


# Installation:

1. Clone this repo.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
## Docker

For better reproducibility, we will release soon a Dockerfile 
to build a container with all the necessary dependencies. :construction_worker:


# Reproducing the Results

We assume that all the experiments are run from the project directory
and that the project directory is added to the `PYTHONPATH` environment variable as follows:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Experiment 1 - End-to-End Training

![exp1](docs/exp1.png)

1. For the multi-robot environment, run from the project directory:
```bash
./scripts/run_exp_baselines.sh [0-6]
```
where the exp-id `[0-6]` denotes runs with 
`PPOPID`, `PPOLag`, `CPO`, `IPO`, `DDPGLag`, `TD3Lag`, and `PPOSaute` respectively.

2. Similary, For the racing environment, run:
```bash
./scripts/run_exp_baselines.sh [7-13]
```

The results will be saved in the `logs/baselines` folder.


## Experiment 2 - Ablation Study

![exp2](docs/exp2.png)

We provide a couple of ablate models to augment built-in controllers with adaptive safety in the `checkpoints` folder.

To play with trained models with adaptive safety, run:
```bash
./scripts/run_checkpoint_eval.sh [0-1]
```
where the exp-id `[0-1]` denotes runs for particle-env and racing environments respectively.

# Citation
```
@misc{berducci2023learning,
      title={Learning Adaptive Safety for Multi-Agent Systems}, 
      author={Luigi Berducci and Shuo Yang and Rahul Mangharam and Radu Grosu},
      year={2023},
      eprint={2309.10657},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
