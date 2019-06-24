import copy
import os
import time
from collections import deque
from functools import partial

import torch
import torch.distributions
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader

from common.util import scale_ob, Trajectories


def learn(logger,
          device,
          env, nenv,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          gamma, timesteps_per_batch, cg_iters, cg_damping, max_kl,
          gae_lam, vf_iters, entcoeff, linear_search_iter=10):
    """
    Thesis:
    Schulman J. Optimizing Expectations: From Deep Reinforcement Learning to
    Stochastic Computation Graphs[D]. UC Berkeley, 2016.

    Official implementation:
    https://github.com/joschu/modular_rl

    There are a little differences between openai's implementation and official
    one in policy update. We follow the openai's version which is also committed
    by John Schulman.

    Parameters:
    ----------
        timesteps_per_batch (int): timesteps per gradient estimation batch
        cg_iters (int): number of iterations of conjugate gradient algorithm
        cg_damping (float): conjugate gradient damping
        max_kl (float): max KL(pi_old || pi)
        gae_lam (float): advantage estimation
        vf_iters (float): number of value update per policy update
        entcoeff (float): coefficient of policy entropy term

    """
    logger.warning('This implementation of trpo only '
                   'support discrete action spaces now!')

    policy, value = map(lambda net: net.to(device), network)
    number_timesteps = number_timesteps // nenv
    generator = _generate(
        device, env, policy, value, ob_scale,
        number_timesteps, gamma, gae_lam, timesteps_per_batch
    )

    n_iter = 0
    total_timesteps = 0
    infos = {'eplenmean': deque(maxlen=40), 'eprewmean': deque(maxlen=40)}
    start_ts = time.time()
    while True:
        try:
            batch = generator.__next__()
        except StopIteration:
            break
        b_o, b_a, b_r, b_v_old, info = batch
        total_timesteps += b_o.size(0)
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])
        # highlight: Normalization is a way to stabilize the BP gradients.
        adv = b_r - b_v_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # calculate current surrgain and kl
        b_logits = policy(b_o)
        dist = torch.distributions.Categorical(logits=b_logits)
        dist_old = torch.distributions.Categorical(logits=b_logits.detach())
        entropy = entcoeff * dist.entropy().mean()
        ratio = (dist.log_prob(b_a) - dist_old.log_prob(b_a)).exp()
        surrgain = (ratio * adv).mean()
        optimgain = surrgain + entropy

        # update policy
        policy.zero_grad()
        pg = _get_flatten_grad(optimgain, policy.parameters())
        if torch.allclose(pg, torch.zeros_like(pg)):
            logger.warn("got zero gradient. not updating")
        else:
            # highlight, only 20 percents of data is used to calculate fvp
            fvp = partial(_fvp, b_o=b_o[::5],
                          policy=policy, cg_damping=cg_damping)
            stepdir = _cg(fvp, pg, cg_iters)
            shs = 0.5 * stepdir.dot(fvp(stepdir))
            lm = torch.sqrt(shs / max_kl)
            fullstep = stepdir / lm
            surrbefore = optimgain.item()
            stepsize = 1.0
            thbefore = parameters_to_vector(policy.parameters())
            for _ in range(linear_search_iter):
                thnew = thbefore + fullstep * stepsize
                new_policy = copy.deepcopy(policy)
                vector_to_parameters(thnew, new_policy.parameters())
                with torch.no_grad():
                    old_logits = policy(b_o)
                    new_logits = new_policy(b_o)
                    dold = torch.distributions.Categorical(logits=old_logits)
                    dnew = torch.distributions.Categorical(logits=new_logits)
                    kl = _kl(old_logits, new_logits).mean()
                    entropy = entcoeff * dnew.entropy().mean()
                    ratio = (dnew.log_prob(b_a) - dold.log_prob(b_a)).exp()
                    surrgain = (ratio * adv).mean()
                    optimgain = surrgain + entropy
                improve = optimgain - surrbefore
                if not torch.isfinite(optimgain).all():
                    logger.info("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.info("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.info("surrogate didn't improve. shrinking step.")
                else:
                    vector_to_parameters(thnew, policy.parameters())
                    break
                stepsize *= .5

        # update baseline
        # highlight: minibatch value update here and drop the last
        loader = DataLoader(list(zip(b_o, b_r)), 64, True, drop_last=True)
        for _ in range(vf_iters):
            for bb_o, bb_r in loader:
                bb_v = value(bb_o)[:, 0]
                optimizer.zero_grad()
                vloss = (bb_v - bb_r).pow(2).mean()
                vloss.backward()
                optimizer.step()

        # log
        n_iter += 1
        logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
        fps = int(total_timesteps / (time.time() - start_ts))
        logger.info('Total timesteps {} FPS {}'.format(total_timesteps, fps))
        for k, v in infos.items():
            v = (sum(v) / len(v)) if v else float('nan')
            logger.info('{}: {:.6f}'.format(k, v))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.checkpoint'.format(n_iter)))


def _generate(device, env, policy, value, ob_scale,
              number_timesteps, gamma, gae_lam, timesteps_per_batch):
    """ Generate trajectories """
    record = ['o', 'a', 'r', 'done', 'vpred']
    export = ['o', 'a', 'r', 'vpred']
    trajectories = Trajectories(record, export,
                                device, gamma, ob_scale, gae_lam)

    o = env.reset()
    infos = []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            ob = scale_ob(o, device, ob_scale)
            logits = policy(ob)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().cpu().numpy()
            v = value(ob).cpu().numpy()[:, 0]

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        trajectories.append(o, a, r, done, v)
        if n % timesteps_per_batch == 0:
            with torch.no_grad():
                ob = scale_ob(o_, device, ob_scale)
                v_ = value(ob).cpu().numpy()[:, 0] * (1 - done)
            yield trajectories.export(v_) + (infos, )
            infos.clear()
        o = o_


def _kl(logits0, logits1):
    """ calculate KL(logits0||logits1) """
    a0 = logits0 - logits0.mean(-1, keepdim=True)
    a1 = logits1 - logits1.mean(-1, keepdim=True)
    ea0 = a0.exp()
    ea1 = a1.exp()
    z0 = ea0.sum(-1, keepdim=True)
    z1 = ea1.sum(-1, keepdim=True)
    p0 = ea0 / z0
    return (p0 * (a0 - z0.log() - a1 + z1.log())).sum(-1)


def _get_flatten_grad(loss, var, create_graph=False, **kwargs):
    grads = torch.autograd.grad(loss, var, create_graph=create_graph, **kwargs)
    return torch.cat([g.contiguous().view(-1) for g in grads])


def _fvp(v, policy, b_o, cg_damping):
    # calculate fisher information matrix of $ \bar{D}_KL(\theta_old, \theta) $
    # see more in John's thesis section 3.12 page 40
    b_logits = policy(b_o)
    kl_old_new = _kl(b_logits.detach(), b_logits).mean()
    kl_grads = _get_flatten_grad(kl_old_new, policy.parameters(), True)
    grads = _get_flatten_grad((kl_grads * v).sum(), policy.parameters())
    # for conjugate gradient, multiply v * cg_damping
    return grads + v * cg_damping


def _cg(fvp, b, cg_iters, cg_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = fvp(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < cg_tol:
            break
    return x
