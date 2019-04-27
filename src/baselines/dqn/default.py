def atari():
    return dict(
        network='cnn',
        optimizer='Adam',
        lr=1e-4,
        grad_norm=10,
        batch_size=32,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        param_noise=False,
        dueling=True,
        scale_ob=1 / 255.0
    )


def classic_control():
    return dict(
        network='mlp',
        optimizer='Adam',
        lr=1e-2,
        grad_norm=10,
        batch_size=32,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=200,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        param_noise=False,
        dueling=True,
        scale_ob=1
    )
