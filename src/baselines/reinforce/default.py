def classic_control():
    return dict(
        network='mlp',
        gamma=1.0,
        lr=0.01,
        batch_episode=1,
        batch_size=100,
    )
