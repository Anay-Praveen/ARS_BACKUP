import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any
import gymnasium as gym

from models import Actor, Critic, MaskGenerator, AdaptiveNoiseInjector
from tensorboard_logger import AMNSTensorboardLogger
from replay_buffer import MultiEnvReplayBuffer


class LifelongSACAgent:
    """
    Lifelong Soft Actor-Critic (SAC) agent with Adaptive Mask-Noise Synchronization (AMNS).
    Implements continuous learning across multiple environments with dynamic mask and noise adaptation.
    """

    def __init__(self,
                 state_dims: List[int],
                 action_dims: List[int],
                 max_actions: List[float],
                 num_environments: int,
                 env_names: List[str],
                 env_action_spaces: List[gym.spaces.Space],
                 hyperparams: Dict[int, Dict]) -> None:
        """Initialize the lifelong SAC agent."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_names = env_names
        self.num_environments = num_environments
        self.env_action_spaces = env_action_spaces
        self.hyperparams = hyperparams
        self.state_dims = state_dims

        # Initialize networks and optimizers
        self.actors: List[nn.Module] = []
        self.critics1: List[nn.Module] = []
        self.critics2: List[nn.Module] = []
        self.critic1_targets: List[nn.Module] = []
        self.critic2_targets: List[nn.Module] = []

        # Initialize actor-critic networks for each environment
        for env_id in range(num_environments):
            env_hyperparams = self.hyperparams[env_id]
            hidden_dims = env_hyperparams["hidden_dims"]

            # Actor network
            actor = Actor(
                state_dims[env_id],
                action_dims[env_id],
                max_actions[env_id],
                hidden_dims=hidden_dims
            ).to(self.device)
            self.actors.append(actor)

            # Critics
            critic1 = Critic(
                state_dims[env_id],
                action_dims[env_id],
                hidden_dims=hidden_dims
            ).to(self.device)
            critic2 = Critic(
                state_dims[env_id],
                action_dims[env_id],
                hidden_dims=hidden_dims
            ).to(self.device)
            self.critics1.append(critic1)
            self.critics2.append(critic2)

            # Target critics
            critic1_target = Critic(
                state_dims[env_id],
                action_dims[env_id],
                hidden_dims=hidden_dims
            ).to(self.device)
            critic2_target = Critic(
                state_dims[env_id],
                action_dims[env_id],
                hidden_dims=hidden_dims
            ).to(self.device)

            # Initialize target networks with source weights
            critic1_target.load_state_dict(critic1.state_dict())
            critic2_target.load_state_dict(critic2.state_dict())
            self.critic1_targets.append(critic1_target)
            self.critic2_targets.append(critic2_target)

        # Initialize optimizers with environment-specific learning rates
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.hyperparams[i]["actor_lr"])
            for i, actor in enumerate(self.actors)
        ]
        self.critic1_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.hyperparams[i]["critic_lr"])
            for i, critic in enumerate(self.critics1)
        ]
        self.critic2_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.hyperparams[i]["critic_lr"])
            for i, critic in enumerate(self.critics2)
        ]

        # Initialize mask generator
        hidden_dims_list = [self.hyperparams[i]["hidden_dims"] for i in range(num_environments)]
        self.mask_generator = MaskGenerator(
            state_dims=state_dims,
            hidden_dims_list=hidden_dims_list,
            num_environments=num_environments
        ).to(self.device)
        self.mask_optimizer = torch.optim.Adam(
            self.mask_generator.parameters(),
            lr=self.hyperparams[0]["mask_lr"]
        )

        # Initialize noise injector
        initial_scales = [self.hyperparams[i]["initial_noise_scale"] for i in range(num_environments)]
        self.noise_injector = AdaptiveNoiseInjector(
            num_environments=num_environments,
            state_dims=state_dims,
            initial_scales=initial_scales
        ).to(self.device)
        self.noise_optimizer = torch.optim.Adam(
            self.noise_injector.parameters(),
            lr=self.hyperparams[0]["noise_lr"]
        )

        # Initialize replay buffer
        self.replay_buffer = MultiEnvReplayBuffer(
            capacity=1000000,
            num_environments=num_environments,
            env_action_spaces=env_action_spaces,
            alpha=0.6,
            beta=0.4
        )

        # Environment-specific parameters
        self.gamma = {i: self.hyperparams[i]["gamma"] for i in range(num_environments)}
        self.tau = {i: self.hyperparams[i]["tau"] for i in range(num_environments)}
        self.alpha = {i: self.hyperparams[i]["alpha"] for i in range(num_environments)}
        self.reward_scale = {i: self.hyperparams[i]["reward_scale"] for i in range(num_environments)}

        # Training metrics and history
        self.total_steps = 0
        self.env_steps = defaultdict(int)
        self.update_counts = defaultdict(int)
        self.episode_rewards = defaultdict(list)
        self.prev_masks = defaultdict(lambda: None)
        self.prev_noise_dist = defaultdict(lambda: None)
        self.performance_history = defaultdict(list)
        self.initial_noise_scales = initial_scales

        # Learning progress tracking
        self.learning_rates = {
            i: {
                'actor': self.hyperparams[i]["actor_lr"],
                'critic': self.hyperparams[i]["critic_lr"],
                'mask': self.hyperparams[i]["mask_lr"],
                'noise': self.hyperparams[i]["noise_lr"]
            } for i in range(num_environments)
        }

    def select_action(self,
                      state: np.ndarray,
                      env_id: int,
                      training: bool = True) -> Union[np.ndarray, Tuple[int, np.ndarray]]:
        """Select an action using the appropriate policy."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        masks = self.mask_generator(state, env_id)

        if training:
            noise = self.noise_injector.sample(env_id, state, self.actors[env_id].hidden_dims[-1])
        else:
            noise = None

        mean, log_std = self.actors[env_id](state, masks, noise)

        if isinstance(self.env_action_spaces[env_id], gym.spaces.Discrete):
            logits = mean
            if training:
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
            else:
                action = torch.argmax(logits, dim=-1)

            action_index = int(action.item())
            one_hot = F.one_hot(action, num_classes=self.env_action_spaces[env_id].n).float()
            return action_index, one_hot.cpu().numpy()[0]
        else:
            if training:
                std = torch.exp(log_std)
                distribution = torch.distributions.Normal(mean, std)
                action = distribution.rsample()
            else:
                action = mean
            return torch.tanh(action).cpu().numpy()[0]


    def train(self, batch_size: int, env_id: int, logger: AMNSTensorboardLogger) -> None:
        """Train the agent using SAC updates."""
        self.total_steps += 1
        self.env_steps[env_id] += 1

        # Sample from replay buffer
        sample = self.replay_buffer.sample(env_id, batch_size)
        if sample is None:
            return

        states, actions, rewards, next_states, dones, weights, indices = sample

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) * self.reward_scale[env_id]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Generate masks
        with torch.no_grad():
            current_masks = self.mask_generator(states, env_id)
            next_masks = self.mask_generator(next_states, env_id)

        # Compute target Q-values
        with torch.no_grad():
            next_mean, next_log_std = self.actors[env_id](next_states, next_masks)

            if isinstance(self.env_action_spaces[env_id], gym.spaces.Box):
                next_std = torch.exp(next_log_std)
                next_distribution = torch.distributions.Normal(next_mean, next_std)
                next_actions = torch.tanh(next_distribution.rsample())
                next_log_probs = next_distribution.log_prob(next_actions).sum(dim=1, keepdim=True)
            else:
                next_distribution = torch.distributions.Categorical(logits=next_mean)
                next_actions_idx = next_distribution.sample()
                next_actions = F.one_hot(next_actions_idx, num_classes=self.env_action_spaces[env_id].n).float()
                next_log_probs = next_distribution.log_prob(next_actions_idx).unsqueeze(1)

            target_Q1 = self.critic1_targets[env_id](next_states, next_actions, next_masks)
            target_Q2 = self.critic2_targets[env_id](next_states, next_actions, next_masks)
            target_Q = rewards + (1 - dones) * self.gamma[env_id] * (
                    torch.min(target_Q1, target_Q2) - self.alpha[env_id] * next_log_probs
            )

        # Update critics (no retain_graph needed here)
        current_Q1 = self.critics1[env_id](states, actions, current_masks)
        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())

        self.critic1_optimizers[env_id].zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics1[env_id].parameters(), 1.0)
        self.critic1_optimizers[env_id].step()

        current_Q2 = self.critics2[env_id](states, actions, current_masks)
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())

        self.critic2_optimizers[env_id].zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics2[env_id].parameters(), 1.0)
        self.critic2_optimizers[env_id].step()

        # Update actor (with fresh computation graph)
        mean, log_std = self.actors[env_id](states, current_masks)
        if isinstance(self.env_action_spaces[env_id], gym.spaces.Box):
            std = torch.exp(log_std)
            distribution = torch.distributions.Normal(mean, std)
            actions_new = torch.tanh(distribution.rsample())
            log_probs = distribution.log_prob(actions_new).sum(dim=1, keepdim=True)
        else:
            distribution = torch.distributions.Categorical(logits=mean)
            actions_idx = distribution.sample()
            actions_new = F.one_hot(actions_idx, num_classes=self.env_action_spaces[env_id].n).float()
            log_probs = distribution.log_prob(actions_idx).unsqueeze(1)

        # Compute actor loss with fresh Q-values
        Q1_new = self.critics1[env_id](states, actions_new, current_masks)
        Q2_new = self.critics2[env_id](states, actions_new, current_masks)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (self.alpha[env_id] * log_probs - Q_new).mean()

        self.actor_optimizers[env_id].zero_grad()
        actor_loss.backward()  # No retain_graph here
        torch.nn.utils.clip_grad_norm_(self.actors[env_id].parameters(), 1.0)
        self.actor_optimizers[env_id].step()

        # Update mask generator last (with fresh computation)
        current_masks = self.mask_generator(states, env_id)  # Recompute masks
        mask_sparsity_loss = self.compute_mask_sparsity_loss(current_masks)
        mask_loss = 0.001 * mask_sparsity_loss

        self.mask_optimizer.zero_grad()
        mask_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mask_generator.parameters(), 1.0)
        self.mask_optimizer.step()

        # Update target networks
        if self.update_counts[env_id] % self.hyperparams[env_id]["target_update_interval"] == 0:
            self.update_target_networks(env_id)

        # Log metrics
        with torch.no_grad():
            metrics = {
                'actor_loss': actor_loss.item(),
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'q_value_mean': Q_new.mean().item(),
                'q_value_std': Q_new.std().item(),
                'actor_grad_norm': self.compute_gradient_norm(self.actors[env_id]),
                'critic_grad_norm': np.mean([
                    self.compute_gradient_norm(self.critics1[env_id]),
                    self.compute_gradient_norm(self.critics2[env_id])
                ])
            }
            logger.log_training_step(env_id, self.total_steps, metrics)

        # Update previous states and counters
        self.prev_masks[env_id] = [mask.detach() for mask in current_masks]  # Store detached version

        self.update_counts[env_id] += 1

    def update_target_networks(self, env_id: int) -> None:
        """Update target network parameters."""
        with torch.no_grad():
            for param, target_param in zip(self.critics1[env_id].parameters(),
                                           self.critic1_targets[env_id].parameters()):
                target_param.data.copy_(
                    self.tau[env_id] * param.data + (1 - self.tau[env_id]) * target_param.data
                )

            for param, target_param in zip(self.critics2[env_id].parameters(),
                                           self.critic2_targets[env_id].parameters()):
                target_param.data.copy_(
                    self.tau[env_id] * param.data + (1 - self.tau[env_id]) * target_param.data
                )

    def compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute the norm of gradients for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return np.sqrt(total_norm)

    def compute_mask_sparsity_loss(self, masks: List[torch.Tensor]) -> torch.Tensor:
        """Compute sparsity loss for masks."""
        return sum(torch.mean(torch.abs(mask)) for mask in masks)

    def adjust_learning_rates(self, env_id: int, factor: float = 0.5) -> None:
        """Adjust learning rates for all optimizers of a specific environment."""
        for optimizer in [
            self.actor_optimizers[env_id],
            self.critic1_optimizers[env_id],
            self.critic2_optimizers[env_id]
        ]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor
                self.learning_rates[env_id]['actor'] = param_group['lr']

    def adjust_exploration(self, env_id: int, factor: float = 1.5) -> None:
        """Adjust exploration parameters for a specific environment."""
        self.alpha[env_id] *= factor
        self.noise_injector.increase_noise_scale(env_id, factor)

    def get_performance_stats(self, env_id: int) -> Dict[str, float]:
        """Get performance statistics for a specific environment."""
        if not self.performance_history[env_id]:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}

        recent_performance = self.performance_history[env_id][-100:]
        return {
            'mean': float(np.mean(recent_performance)),
            'std': float(np.std(recent_performance)),
            'max': float(np.max(recent_performance)),
            'min': float(np.min(recent_performance))
        }

    def update_performance_history(self, env_id: int, reward: float) -> None:
        """Update performance history for a specific environment."""
        self.performance_history[env_id].append(reward)
        if len(self.performance_history[env_id]) > 1000:  # Keep last 1000 episodes
            self.performance_history[env_id].pop(0)

    def get_learning_progress(self, env_id: int) -> Dict[str, float]:
        """Get learning progress metrics for a specific environment."""
        if len(self.performance_history[env_id]) < 2:
            return {'improvement_rate': 0.0, 'stability': 0.0}

        recent_100 = self.performance_history[env_id][-100:]
        prev_100 = self.performance_history[env_id][-200:-100] if len(self.performance_history[env_id]) > 200 else []

        improvement_rate = (np.mean(recent_100) - np.mean(prev_100)) if prev_100 else 0.0
        stability = 1.0 / (np.std(recent_100) + 1e-6)  # Higher value means more stable

        return {
            'improvement_rate': float(improvement_rate),
            'stability': float(stability)
        }

    def should_adjust_hyperparameters(self, env_id: int) -> bool:
        """Determine if hyperparameters should be adjusted based on performance."""
        if len(self.performance_history[env_id]) < 200:
            return False

        recent_50 = self.performance_history[env_id][-50:]
        prev_50 = self.performance_history[env_id][-100:-50]

        recent_mean = np.mean(recent_50)
        prev_mean = np.mean(prev_50)

        return recent_mean <= prev_mean or (recent_mean - prev_mean) / prev_mean < 0.01

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state dictionary for saving agent state."""
        return {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics1': [critic.state_dict() for critic in self.critics1],
            'critics2': [critic.state_dict() for critic in self.critics2],
            'critic1_targets': [critic.state_dict() for critic in self.critic1_targets],
            'critic2_targets': [critic.state_dict() for critic in self.critic2_targets],
            'mask_generator': self.mask_generator.state_dict(),
            'noise_injector': self.noise_injector.state_dict(),
            'hyperparams': self.hyperparams,
            'total_steps': self.total_steps,
            'env_steps': dict(self.env_steps),
            'update_counts': dict(self.update_counts),
            'performance_history': dict(self.performance_history),
            'learning_rates': self.learning_rates
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load complete state dictionary."""
        for i, actor_state in enumerate(state_dict['actors']):
            self.actors[i].load_state_dict(actor_state)
        for i, critic_state in enumerate(state_dict['critics1']):
            self.critics1[i].load_state_dict(critic_state)
        for i, critic_state in enumerate(state_dict['critics2']):
            self.critics2[i].load_state_dict(critic_state)
        for i, critic_state in enumerate(state_dict['critic1_targets']):
            self.critic1_targets[i].load_state_dict(critic_state)
        for i, critic_state in enumerate(state_dict['critic2_targets']):
            self.critic2_targets[i].load_state_dict(critic_state)

        self.mask_generator.load_state_dict(state_dict['mask_generator'])
        self.noise_injector.load_state_dict(state_dict['noise_injector'])
        self.hyperparams = state_dict['hyperparams']
        self.total_steps = state_dict['total_steps']
        self.env_steps = defaultdict(int, state_dict['env_steps'])
        self.update_counts = defaultdict(int, state_dict['update_counts'])
        self.performance_history = defaultdict(list, state_dict['performance_history'])

        if 'learning_rates' in state_dict:
            self.learning_rates = state_dict['learning_rates']
            # Update optimizer learning rates
            for env_id in range(self.num_environments):
                for param_group in self.actor_optimizers[env_id].param_groups:
                    param_group['lr'] = self.learning_rates[env_id]['actor']
                for param_group in self.critic1_optimizers[env_id].param_groups:
                    param_group['lr'] = self.learning_rates[env_id]['critic']
                for param_group in self.critic2_optimizers[env_id].param_groups:
                    param_group['lr'] = self.learning_rates[env_id]['critic']


