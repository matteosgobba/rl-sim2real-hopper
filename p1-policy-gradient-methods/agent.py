import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(r.size(0))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma) + 1e-5
        normal_dist = Normal(action_mean, sigma)
        return normal_dist


class ReinforceAgent(object):
    def __init__(self, policy, baseline=0.0, lr=1e-3, gamma=0.99, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.baseline = baseline

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.next_states = [], []
        self.action_log_probs, self.rewards, self.done = [], [], []

        returns = discount_rewards(rewards, self.gamma)

        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = returns - self.baseline

        loss = -(action_log_probs * advantages.detach()).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss.item(), returns.sum().item()

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy(x)

        if evaluation:
            action = normal_dist.mean
            return action.detach().cpu().numpy(), None
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action.detach().cpu().numpy(), action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)

#rete actor (policy)
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_mean = torch.nn.Linear(self.hidden, action_space)
        
        self.sigma_activation = F.softplus
        self.sigma = torch.nn.Parameter(torch.zeros(action_space) + 0.5)
        self.tanh = torch.nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        mean = self.fc3_mean(x)
        sigma = self.sigma_activation(self.sigma) + 1e-5
        return Normal(mean, sigma)

#rete critic (value function)
class ValueNetwork(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        self.tanh = torch.nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x).squeeze(-1) # Restituisce lo scalare V(s)


class ActorCriticAgent(object):
    def __init__(self, actor, critic, lr_actor=1e-3, lr_critic=3e-4, gamma=0.99, value_coef=0.5, entropy_coef=0.01, device='cpu'):
        self.train_device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        
        dist = self.actor(x)
        value = self.critic(x)

        if evaluation:
            return dist.mean.detach().cpu().numpy(), None

        action = dist.sample()
        return action.detach().cpu().numpy(), (dist.log_prob(action).sum(), dist.entropy().sum(), value)

    def update_step(self, next_state, action_log_prob, reward, done, value, entropy):
        next_x = torch.from_numpy(next_state).float().to(self.train_device)
        
        with torch.no_grad():
            next_value = self.critic(next_x)

        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.train_device)
        done_t = torch.tensor(float(done), dtype=torch.float32, device=self.train_device)


        td_target = reward_t + self.gamma * next_value * (1.0 - done_t)
        advantage = td_target - value 
        

        #update critic
        critic_loss = self.value_coef * F.mse_loss(value, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor
        actor_loss = -(action_log_prob * advantage.detach()) + self.entropy_coef * (-entropy)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return (actor_loss + critic_loss).item()