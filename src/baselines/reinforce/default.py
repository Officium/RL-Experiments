def classic_control():
    return dict(
        network='mlp',
        optimizer='Adam',
        gamma=0.99,
        lr=0.01,
        timesteps_per_batch=100,
        ob_scale=1.0,
    )
