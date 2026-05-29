"""
Agents and neural networks for Part 1

Implemented algorithms:
- REINFORCE without baseline
- REINFORCE with constant baseline
- one-step Actor-Critic
- n-step Actor-Critic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


Tensor = torch.Tensor


def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Compute Monte-Carlo discounted returns G_t for one episode"""
    returns = torch.zeros_like(rewards)
    running_return = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


class PolicyNetwork(nn.Module):
    """Gaussian policy network for continuous action spaces"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        init_log_std: float = -0.7,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Normal:
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_head(x)

        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)

        return Normal(mean, std)


class ValueNetwork(nn.Module):
    """State-value network V(s) used by Actor-Critic"""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Tensor:
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.value_head(x).squeeze(-1)


@dataclass
class ReinforceTransition:
    log_prob: Tensor
    reward: float


class ReinforceAgent:
    """
    REINFORCE / Vanilla Policy Gradient agent

    Note about the constant baseline:
    For the main comparison requested in the project, the baseline is subtracted from the unnormalized Monte-Carlo returns. This makes baseline=20 meaningful.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        baseline: float = 0.0,
        hidden_dim: int = 64,
        normalize_advantages: bool = False,
        grad_clip: Optional[float] = 5.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.baseline = baseline
        self.normalize_advantages = normalize_advantages
        self.grad_clip = grad_clip

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.memory: List[ReinforceTransition] = []

    def get_action(self, state: np.ndarray, evaluation: bool = False) -> Tuple[np.ndarray, Optional[Tensor]]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        dist = self.policy(state_t)

        if evaluation:
            action = dist.mean
            return action.detach().cpu().numpy(), None

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().cpu().numpy(), log_prob

    def store_outcome(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action_log_prob: Tensor,
        reward: float,
        done: bool,
    ) -> None:
        del state, next_state, done
        if action_log_prob is None:
            raise ValueError("action_log_prob cannot be None during training.")
        self.memory.append(ReinforceTransition(log_prob=action_log_prob, reward=float(reward)))

    def update_policy(self) -> Dict[str, float]:
        if len(self.memory) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "mean_return": 0.0, "mean_advantage": 0.0}

        log_probs = torch.stack([transition.log_prob for transition in self.memory]).to(self.device)
        rewards = torch.tensor([transition.reward for transition in self.memory], dtype=torch.float32, device=self.device)

        returns = discounted_returns(rewards, self.gamma)
        advantages = returns - self.baseline

        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        metrics = {
            "loss": float(policy_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "mean_return": float(returns.mean().detach().cpu().item()),
            "mean_advantage": float(advantages.mean().detach().cpu().item()),
        }

        self.memory.clear()
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "gamma": self.gamma,
                "baseline": self.baseline,
                "normalize_advantages": self.normalize_advantages,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class ActorCriticAgent:
    """Actor-Critic agent supporting one-step and n-step updates"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        value_coef: float = 0.7,
        entropy_coef: float = 0.02,
        grad_clip: Optional[float] = 5.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

    def get_action(self, state: np.ndarray, evaluation: bool = False) -> Tuple[np.ndarray, Optional[Tensor]]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        dist = self.policy(state_t)

        if evaluation:
            action = dist.mean
            return action.detach().cpu().numpy(), None

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().cpu().numpy(), log_prob

    def update_step(
        self,
        state: np.ndarray,
        action_log_prob: Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """One-step Actor-Critic update"""
        if action_log_prob is None:
            raise ValueError("action_log_prob cannot be None during training.")

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        done_t = torch.tensor(float(done), dtype=torch.float32, device=self.device)

        value = self.value(state_t)
        with torch.no_grad():
            next_value = self.value(next_state_t)
            td_target = reward_t + self.gamma * next_value * (1.0 - done_t)

        advantage = td_target - value

        # Recompute entropy from the current policy at state_t
        dist = self.policy(state_t)
        entropy = dist.entropy().sum()

        actor_loss = -(action_log_prob * advantage.detach()) - self.entropy_coef * entropy
        critic_loss = F.mse_loss(value, td_target.detach())
        total_loss = actor_loss + self.value_coef * critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "policy_loss": float(actor_loss.detach().cpu().item()),
            "value_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "td_target": float(td_target.detach().cpu().item()),
            "advantage": float(advantage.detach().cpu().item()),
        }

    def compute_n_step_targets(
        self,
        rewards: Sequence[float],
        next_states: Sequence[np.ndarray],
        dones: Sequence[bool],
        n_steps: int,
    ) -> Tensor:
        """
        Compute n-step bootstrapped targets for all transitions in one episode.

        For each t:
            G_t:t+n = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1} + gamma^n V(s_{t+n}) if the episode is not done before t+n.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1.")

        targets: List[Tensor] = []
        episode_len = len(rewards)

        for t in range(episode_len):
            g = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            discount = 1.0
            bootstrap_index: Optional[int] = None

            for k in range(n_steps):
                idx = t + k
                if idx >= episode_len:
                    break

                g = g + discount * float(rewards[idx])

                if dones[idx]:
                    bootstrap_index = None
                    break

                discount *= self.gamma
                bootstrap_index = idx

            # If we consumed exactly n non-terminal rewards, bootstrap from s_{t+n}
            # next_states[bootstrap_index] is the state after transition bootstrap_index
            if bootstrap_index is not None and (t + n_steps - 1) < episode_len and not dones[t + n_steps - 1]:
                next_state_t = torch.as_tensor(
                    next_states[t + n_steps - 1], dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    g = g + discount * self.value(next_state_t)

            targets.append(g)

        return torch.stack(targets)

    def update_n_step_episode(
        self,
        states: Sequence[np.ndarray],
        action_log_probs: Sequence[Tensor],
        rewards: Sequence[float],
        next_states: Sequence[np.ndarray],
        dones: Sequence[bool],
        n_steps: int = 5,
        normalize_advantages: bool = False,
    ) -> Dict[str, float]:
        """
        Batched n-step Actor-Critic update at the end of one episode

        This is still Actor-Critic because the targets bootstrap from V(s_{t+n}).
        It is less "myopic" than the one-step update and can propagate reward information over a longer horizon.
        """
        if len(states) == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "mean_return": 0.0,
                "mean_advantage": 0.0,
                "advantage": 0.0,
            }

        if any(log_prob is None for log_prob in action_log_probs):
            raise ValueError("action_log_probs cannot contain None during training.")

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        log_probs_t = torch.stack(list(action_log_probs)).to(self.device)

        n_step_targets = self.compute_n_step_targets(rewards, next_states, dones, n_steps=n_steps)
        values = self.value(states_t)
        advantages = n_step_targets.detach() - values

        if normalize_advantages and advantages.numel() > 1:
            advantages_for_actor = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        else:
            advantages_for_actor = advantages

        dist = self.policy(states_t)
        entropy = dist.entropy().sum(dim=-1).mean()

        actor_loss = -(log_probs_t * advantages_for_actor.detach()).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, n_step_targets.detach())
        total_loss = actor_loss + self.value_coef * critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "policy_loss": float(actor_loss.detach().cpu().item()),
            "value_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "mean_return": float(n_step_targets.mean().detach().cpu().item()),
            "mean_advantage": float(advantages.mean().detach().cpu().item()),
            "advantage": float(advantages.mean().detach().cpu().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "gamma": self.gamma,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])


Agent = ReinforceAgent