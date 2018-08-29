# RL-Experiments

The motivation of this project is to compare and modify deep reinforcement learning algorithms easily **for experiments**.

# Implemented algorithms

* REINFORCE
* DQN
* DDPG
* A3C
* TRPO
* PPO + GAE

# Dependency

* Python: 3.5+
* Gym: 0.10.3
* PyTorch: 0.4.0
* We propose to build your environment by virtualenv or docker

# Usage

```bash
mkdir logs
cd src
python -m baselines.algorithm.demo  # replace `algorithm` by `TRPO`, `PPO` ...
```

# Plans

* Implement more algorithms such as Q-Prop, Rainbow and GPS
