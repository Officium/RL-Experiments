def atari():
    return dict(
        nsteps=128,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=4,
        log_interval=1,
        ent_coef=.01,
        lr=2.5e-4,
        cliprange=0.1,
    )
