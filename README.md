# RL-Experiments

The motivation of this project is to compare and modify deep reinforcement learning algorithms easily **for experiments**.

# Implemented algorithms

* REINFORCE
* DQN
* DDPG
* A3C
* TRPO
* PPO + GAE
* RAINBOW

# Dependency

* Python: 3.5+
* Gym: 0.10.3+
* PyTorch: 0.4.0+
* The code was checked at Windows 10, Ubuntu 16.04 and Mac OS.

# Usage

```bash
git clone https://github.com/Officium/RL-Experiments.git
cd RL-Experiments/src
python -m baselines.algorithm.demo  # replace `algorithm` by `TRPO`, `PPO` ...
```
