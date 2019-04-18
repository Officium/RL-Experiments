def classic_control():
    return dict(
        network='mlp',
        gamma=0.99,
        lr=0.01,
        timesteps_per_batch=100,
        reset_after_batch=True,
    )
