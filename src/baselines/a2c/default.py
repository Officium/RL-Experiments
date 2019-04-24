def atari():
    return dict(
        network='cnn',
        optimizer='RMSprop',
        lr=7e-4,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=5,
        ent_coef=.01,
        vf_coef=0.5,
        ob_scale=1.0 / 255
    )


def classic_control():
    return dict(
        network='mlp',
        optimizer='RMSprop',
        lr=1e-2,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=5,
        ent_coef=0,
        vf_coef=0.5,
        ob_scale=1
    )
