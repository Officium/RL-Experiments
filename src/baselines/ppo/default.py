def atari():
    return dict(
        network='cnn',
        optimizer='Adam',
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
        ob_scale=1.0 / 255
    )


def classic_control():
    return dict(
        network='mlp',
        optimizer='Adam',
        lr=1e-2,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=2048,
        ent_coef=0,
        vf_coef=0.001,
        gae_lam=0.95,
        nminibatches=4,
        opt_iter=4,
        cliprange=0.2,
        ob_scale=1
    )
