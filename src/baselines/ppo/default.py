def atari():
    return dict(
        network='cnn',
        lr=2.5e-4,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=128,
        ent_coef=.01,
        vf_coef=0.5,
        gae_lam=0.95,
        nminibatches=4,
        opt_iter=4,
        cliprange=0.1,
    )
