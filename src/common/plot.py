""" Plot reward curve
Usage: set xscale and run `python common/plot.py ../logs/ppo_* ../logs/trpo_*`
"""
import os
import sys
from glob import glob

import numpy as np
from matplotlib import pyplot as plt


def smooth(y, radius=100, mode='two_sided'):
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        return np.convolve(y, convkernel, mode='same') / \
               np.convolve(np.ones_like(y), convkernel, mode='same')
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / \
              np.convolve(np.ones_like(y), convkernel, mode='full')
        return out[:-radius+1]


def one_sided_ema(xolds, yolds, low, high, n,
                  decay_steps, low_counts_threshold):

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low, high, n,
                  decay_steps=1., low_counts_threshold=1e-8):
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high,
                                       n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low,
                                       n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys


COLORS = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple',
    'pink', 'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender',
    'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'
]
eprews = dict()
for path in sys.argv[1:]:
    alg = os.path.split(path)[-2]
    if eprews.get(alg) is None:
        eprews[alg] = []
    eprews[alg].append([])
    actor_num = len(glob(os.path.join(path, 'actor.log*')))
    for actor in glob(os.path.join(path, 'actor.log*')):
        n = 0
        with open(actor) as f:
            for line in f:
                r, l = map(float, line.split('\t'))
                n += int(l) * actor_num
                eprews[alg][-1].append((n, r))
    eprews[alg][-1].sort()

fig, ax = plt.subplots()
for i, (alg, rews) in enumerate(sorted(eprews.items())):
    num_points = 512
    color = COLORS[i]
    all_ys = []
    low = max(x[0][0] for x in rews)
    high = min(x[-1][0] for x in rews)
    for rew in rews:
        xs, ys = np.asarray([x[0] for x in rew]), smooth([x[1] for x in rew])
        xs, ys = symmetric_ema(xs, ys, low, high, num_points)
        all_ys.append(ys)
    xs = np.linspace(low, high, num_points)
    mean = np.mean(all_ys, 0)
    std = np.std(all_ys, 0) / np.sqrt(len(all_ys))
    plt.plot(xs, mean, color=color, label=alg)
    plt.fill_between(xs, mean - std, mean + std, color=COLORS[i], alpha=.4)
plt.legend(fontsize=20, frameon=False)
plt.xticks([])
plt.yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
