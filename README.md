# RL-Experiments

The motivation of this project is to compare and modify deep reinforcement learning algorithms easily **for experiments**.

# Recent plan

1. Rewrite algorithms and add environment utils refer to [openai/baselines](https://github.com/openai/baselines).    
2. Highlight the differences between implementation and paper.    
3. Make codes following PEP8.    
4. Based on PyTorch 1.0.   

During that time, many internal or incomplete code will be committed. 
If you mind, please checkout to the [old but stable version](https://github.com/Officium/RL-Experiments/commit/255aa7a9c03e38349d7c03540769eb9dfa91d33d). 
Welcome any contributions!

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
