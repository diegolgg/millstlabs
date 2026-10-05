"""Eight independent CPU actor-critics; frozen GPT agents retain country/reason generation."""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, width, actions):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.SiLU(), nn.Linear(96, actions))
        self.critic = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.SiLU(), nn.Linear(96, 1))
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, observation, mask):
        return Categorical(logits=self.actor(observation).masked_fill(~mask.bool(), -1e9)), self.critic(observation).squeeze(-1)


class Controllers:
    def __init__(self, cfg, width, actions):
        self.cfg = cfg
        torch.set_num_threads(1)
        torch.manual_seed(cfg.policy_seed)
        common = ActorCritic(width, actions)
        self.models = [copy.deepcopy(common) for _ in range(cfg.population)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.learning_rate) for m in self.models]
        self.updates = 0

    @torch.no_grad()
    def act(self, agent, observation, mask, rng):
        x, legal = torch.tensor(observation, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
        distribution, value = self.models[agent](x, legal)
        probabilities = distribution.probs.numpy().astype(float)
        action = int(rng.choice(len(mask), p=probabilities/probabilities.sum()))
        return action, float(distribution.log_prob(torch.tensor(action))), float(value)

    def learn(self, trajectories, correct):
        metrics = []
        c = self.cfg
        for i, rows in enumerate(trajectories):
            if not rows:
                continue
            peer_accuracy = (sum(correct)-correct[i])/(len(correct)-1)
            rewards = np.asarray([-c.tool_cost*bool(r["action"]) for r in rows], dtype=float)
            rewards[-1] += correct[i]+c.cooperation_weight*peer_accuracy
            values = np.asarray([r["value"] for r in rows])
            advantages = np.zeros(len(rows))
            gae, next_value = 0., 0.
            for k in reversed(range(len(rows))):
                gae = rewards[k]+c.gamma*next_value-values[k]+c.gamma*c.gae_lambda*gae
                advantages[k], next_value = gae, values[k]
            target = torch.tensor(advantages+values, dtype=torch.float32)
            adv = torch.tensor(advantages, dtype=torch.float32)
            if len(adv) > 1:
                adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
            x = torch.tensor(np.asarray([r["observation"] for r in rows]), dtype=torch.float32)
            mask = torch.tensor([r["mask"] for r in rows], dtype=torch.bool)
            action = torch.tensor([r["action"] for r in rows])
            old = torch.tensor([r["logp"] for r in rows])
            for _ in range(c.ppo_epochs):
                distribution, value = self.models[i](x, mask)
                ratio = (distribution.log_prob(action)-old).exp()
                policy = -torch.minimum(ratio*adv, ratio.clamp(1-c.clip, 1+c.clip)*adv).mean()
                critic = nn.functional.smooth_l1_loss(value, target)
                loss = policy+.5*critic-c.entropy*distribution.entropy().mean()
                self.optimizers[i].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.models[i].actor.parameters(), 1.)
                nn.utils.clip_grad_norm_(self.models[i].critic.parameters(), 1.)
                self.optimizers[i].step()
                self.updates += 1
            metrics.append({"agent": i, "samples": len(rows), "training_return": float(rewards.sum()),
                            "policy_loss": float(policy.detach()), "value_loss": float(critic.detach())})
        return metrics

    def state(self):
        return {"models": [m.state_dict() for m in self.models],
                "optimizers": [o.state_dict() for o in self.optimizers], "updates": self.updates}

    def restore(self, state):
        for m, saved in zip(self.models, state["models"]):
            m.load_state_dict(saved)
        for o, saved in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(saved)
        self.updates = state["updates"]
