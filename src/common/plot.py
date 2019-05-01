""" Plot reward curve
Usage: set xscale and run `python common/plot.py ../logs/ppo_* ../logs/trpo_*`
"""
import os
import sys

import numpy as np
from matplotlib import pyplot as plt


xscale = {
    'ppo': 1024 * 4,  # 1024 timesteps/log, 4 skipped frames/observation
    'trpo': 2048 * 4,
    'dqn': 1000 * 4,
    'a2c': 8 * 4,
    'acer': 160 * 4,
}
COLORS = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple',
    'pink', 'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender',
    'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'
]
eprews = dict()
for path in sys.argv[1:]:
    alg, seed = os.path.split(path)[-1].split('_')
    if eprews.get(alg) is None:
        eprews[alg] = []
    eprews[alg].append([])
    nan = 0
    with open(path) as f:
        for line in f:
            if line.__contains__('eprewmean'):
                rew = line.strip().rsplit(':', 1)[1].strip()
                if rew == 'nan':
                    nan += 1
                else:
                    eprews[alg][-1].extend([float(rew)] * (nan + 1))
                    nan = 0

f, axarr = plt.subplots(1, len(eprews),
                        sharex=False, sharey=False, squeeze=False)
for i, (alg, rews) in enumerate(eprews.items()):
    ax = axarr[0][i]
    min_length = min(len(rs) for rs in rews)
    xs = np.arange(min_length) * xscale[alg]
    rews = np.asarray([rs[:min_length] for rs in rews])
    mean = np.mean(rews, 0)
    std = np.std(rews, 0) / np.sqrt(len(rews))
    ax.set_title(alg)
    ax.plot(xs, mean, color=COLORS[i])
    ax.fill_between(xs, mean - std, mean + std, color=COLORS[i], alpha=.4)
plt.show()
