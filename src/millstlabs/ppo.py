"""Recurrent PPO with independent personal and finite-window social accounting."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    observation: dict
    hidden: torch.Tensor
    action: int
    log_probability: float
    personal_value: float
    social_value: float
    personal_reward: float
    social_reward: float
    personal_terminal: bool
    social_terminal: bool
    temperature: float = 1.0
    intrinsic_value: float = 0.0
    intrinsic_reward: float = 0.0
    publication: int = -1
    publication_log_probability: float = 0.0
    publication_reward: float = 0.0


def advantages(transitions, cfg, personal_bootstrap=0.0, social_bootstrap=0.0):
    """At death, social_bootstrap may be the *realized* discounted post-death tail.

    Dead ticks have no policy loss or fabricated actions. Their exact discounted
    return enters the final living action's TD target, then GAE propagates it.
    """
    ap, ass = np.zeros(len(transitions)), np.zeros(len(transitions))
    next_p, next_s = personal_bootstrap, social_bootstrap
    gp, gs = 0.0, 0.0
    for k in reversed(range(len(transitions))):
        r = transitions[k]
        mp, ms = 1 - int(r.personal_terminal), 1 - int(r.social_terminal)
        dp = r.personal_reward + cfg.gamma * mp * next_p - r.personal_value
        ds = r.social_reward + cfg.gamma * ms * next_s - r.social_value
        gp = dp + cfg.gamma * cfg.gae_lambda * mp * gp
        gs = ds + cfg.gamma * cfg.gae_lambda * ms * gs
        ap[k], ass[k] = gp, gs
        next_p, next_s = r.personal_value, r.social_value
    return ap, ass


def intrinsic_advantages(transitions, cfg, bootstrap=0.0):
    result = np.zeros(len(transitions))
    gae, next_value = 0., bootstrap
    for k in reversed(range(len(transitions))):
        row = transitions[k]
        live = 1 - int(row.personal_terminal)
        delta = row.intrinsic_reward + cfg.gamma*live*next_value - row.intrinsic_value
        gae = delta + cfg.gamma*cfg.gae_lambda*live*gae
        result[k] = gae
        next_value = row.intrinsic_value
    return result


def update(bank, i, transitions, preference, personal_bootstrap=0.0, social_bootstrap=0.0, intrinsic_bootstrap=0.0):
    if not transitions:
        return {}
    cfg = bank.cfg
    ap, ass = advantages(transitions, cfg, personal_bootstrap, social_bootstrap)
    ai = intrinsic_advantages(transitions, cfg, intrinsic_bootstrap) if cfg.intrinsic_critic else np.zeros(len(transitions))
    actor = torch.tensor(ap + preference * ass + ai, dtype=torch.float32, device=bank.device)
    # Normalization is within one individual's experience only.
    if len(actor) > 1:
        actor = (actor - actor.mean()) / actor.std(unbiased=False).clamp_min(1e-6)
    tp = torch.tensor(ap + np.array([r.personal_value for r in transitions]), dtype=torch.float32, device=bank.device)
    ts = torch.tensor(ass + np.array([r.social_value for r in transitions]), dtype=torch.float32, device=bank.device)
    ti = torch.tensor(ai + np.array([r.intrinsic_value for r in transitions]), dtype=torch.float32, device=bank.device)
    metrics = []
    # Each transition is reused exactly `epochs` times; no agent shares a buffer.
    for _ in range(cfg.epochs):
        for start in range(0, len(transitions), cfg.sequence_length):
            seq = transitions[start:start + cfg.sequence_length]
            bank._activate(i)
            outputs, _ = bank.sequence(i, [r.observation for r in seq], seq[0].hidden.to(bank.device),
                                       [getattr(r, "temperature", 1.0) for r in seq])
            actions = torch.tensor([r.action for r in seq], device=bank.device)
            logp = torch.cat([o[0].log_prob(actions[j]) for j, o in enumerate(outputs)])
            old = torch.tensor([r.log_probability for r in seq], device=bank.device)
            ratio = torch.exp(logp - old)
            a = actor[start:start + len(seq)]
            policy_loss = -torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vp, vs = torch.cat([o[1] for o in outputs]), torch.cat([o[2] for o in outputs])
            value_loss = ((vp - tp[start:start + len(seq)]) ** 2).mean() + ((vs - ts[start:start + len(seq)]) ** 2).mean()
            if cfg.split_controller:
                value_loss = (torch.nn.functional.smooth_l1_loss(vp / cfg.value_scale, tp[start:start+len(seq)] / cfg.value_scale)
                              + torch.nn.functional.smooth_l1_loss(vs / cfg.value_scale, ts[start:start+len(seq)] / cfg.value_scale))
            if cfg.intrinsic_critic:
                vi = torch.cat([o[3] for o in outputs])
                value_loss = value_loss + torch.nn.functional.smooth_l1_loss(vi, ti[start:start+len(seq)])
            entropy = torch.cat([o[0].entropy() for o in outputs]).mean()
            loss = policy_loss + cfg.value_coefficient * value_loss - cfg.entropy * entropy
            publication_losses = []
            for j, row in enumerate(seq):
                if row.publication >= 0:
                    write_dist = outputs[j][-1]
                    write_logp = write_dist.log_prob(torch.tensor(float(row.publication), device=bank.device))
                    write_ratio = torch.exp(write_logp-row.publication_log_probability)
                    # Publishing also bears the task's individual/social outcome,
                    # with a small immediate information-delivery bonus.
                    write_advantage = a[j] + row.publication_reward
                    publication_losses.append(-torch.minimum(write_ratio*write_advantage,
                        write_ratio.clamp(1-cfg.clip, 1+cfg.clip)*write_advantage)-cfg.entropy*write_dist.entropy())
            if publication_losses:
                loss = loss + torch.cat(publication_losses).mean()
            if cfg.feed_retention:
                # Preserve a basic demonstrated skill without forcing actions.
                # No oracle state, path planner, or action override at deployment.
                eligible = [j for j, r in enumerate(seq) if r.observation["self"][0] < 80
                            and r.observation["action_mask"][4]
                            and not (r.observation["threats"][:, 0] >= 0).any()]
                if eligible:
                    retention = -torch.cat([outputs[j][0].log_prob(torch.tensor(4, device=bank.device))
                                            for j in eligible]).mean()
                    loss = loss + cfg.feed_retention*retention
            optimizer = bank.optimizers[i]
            optimizer.zero_grad(set_to_none=True)
            gpu_start = bank.gpu_start()
            loss.backward()
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if cfg.split_controller:
                controller = bank.controllers[i]
                actor_norm = torch.nn.utils.clip_grad_norm_(bank._adapter_parameters(i) + controller.actor_parameters(), cfg.max_grad_norm)
                critic_norm = torch.nn.utils.clip_grad_norm_(controller.critic_parameters(), cfg.max_grad_norm)
            else:
                actor_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
                critic_norm = float("nan")
            optimizer.step()
            bank.gpu_end(gpu_start)
            bank.gradient_updates += 1
            metrics.append([float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                            float(((ratio - 1) - (logp - old)).mean().detach()),
                            float(((ratio - 1).abs() > cfg.clip).float().mean()), float(actor_norm), float(critic_norm)])
    result = dict(zip(["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm", "critic_grad_norm"],
                      np.mean(metrics, axis=0).tolist()))
    if not cfg.split_controller:
        result["critic_grad_norm"] = None  # Shared gradient norm only for legacy controllers.
    for name, target, prediction in [("personal", tp, [r.personal_value for r in transitions]),
                                      ("social", ts, [r.social_value for r in transitions])]:
        y = target.cpu().numpy()
        result[f"{name}_explained_variance"] = float(1 - np.var(y - prediction) / np.var(y)) if np.var(y) > 1e-8 else None
    result["intrinsic_reward_sum"] = sum(r.intrinsic_reward for r in transitions)
    result["publication_opportunities"] = sum(r.publication >= 0 for r in transitions)
    result["publications"] = sum(r.publication == 1 for r in transitions)
    result["publication_reward"] = sum(r.publication_reward for r in transitions)
    return result
