def atari():
    return dict(
        network='smallcnn',
        optimizer='Adam',
        gamma=0.98,
        timesteps_per_batch=512,
        cg_iters=10,
        cg_damping=1e-3,
        max_kl=0.001,
        gae_lam=1.0,
        vf_iters=3,
        vf_lr=1e-4,
        entcoeff=0.00,
        ob_scale=1 / 255.0
    )


def classic_control():
    return dict(
        network='mlp',
        optimizer='Adam',
        gamma=0.98,
        timesteps_per_batch=512,
        cg_iters=10,
        cg_damping=1e-3,
        max_kl=0.001,
        gae_lam=1.0,
        vf_iters=3,
        vf_lr=1e-2,
        entcoeff=0.00,
        ob_scale=1
    )
