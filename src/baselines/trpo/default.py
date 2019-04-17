def atari():
    return dict(
        network='small_cnn',
        gamma=0.98,
        timesteps_per_batch=512,
        cg_iters=10,
        cg_damping=1e-3,
        max_kl=0.001,
        lam=1.0,
        vf_iters=3,
        vf_stepsize=1e-4,
        entcoeff=0.00,
    )
