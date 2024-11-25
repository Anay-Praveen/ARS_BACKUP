"""
models.py

Contains neural network models for the Lifelong SAC agent, including:
- Actor network
- Critic network
- Mask generator
- Adaptive noise injector
"""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class Actor(nn.Module):
    """Actor network with residual connections and layer normalization."""

    def __init__(self, state_dim, action_dim, max_action, hidden_dims=None):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims else [512, 256, 128]

        # Layers for the main network
        self.layers = nn.ModuleList([
            nn.Linear(state_dim, self.hidden_dims[0]),
            *[
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                for i in range(len(self.hidden_dims) - 1)
            ]
        ])

        # Residual connections for gradient flow
        self.residual_layers = nn.ModuleList([
            nn.Linear(state_dim, self.hidden_dims[0]),
            *[
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                for i in range(len(self.hidden_dims) - 1)
            ]
        ])

        # Layer normalization for training stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.hidden_dims
        ])

        # Output layers for mean and log_std
        self.mean = nn.Linear(self.hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(self.hidden_dims[-1], action_dim)
        self.max_action = max_action

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, state, masks, noise=None):
        x = state
        residual = state

        for i, (layer, res_layer, norm) in enumerate(zip(self.layers, self.residual_layers, self.layer_norms)):
            x = layer(x) + res_layer(residual)  # Main path + residual connection
            x = norm(x)
            x = F.relu(x)
            x = x * masks[i]  # Apply mask
            residual = x

        if noise is not None:
            noise = noise.view(x.size())  # Reshape noise to match hidden dimensions
            x = x + noise

        mean = self.mean(x)
        if hasattr(self, 'max_action') and self.max_action != 1.0:
            mean = self.max_action * torch.tanh(mean)

        log_std = self.log_std(x).clamp(-20, 2)  # Clamp for numerical stability
        return mean, log_std


class Critic(nn.Module):
    """Critic network with residual connections and layer normalization."""

    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims else [512, 256, 128]

        self.input_dim = state_dim + action_dim

        # Layers for the main network
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            *[
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                for i in range(len(self.hidden_dims) - 1)
            ]
        ])

        # Residual connections for better gradient flow
        self.residual_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            *[
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                for i in range(len(self.hidden_dims) - 1)
            ]
        ])

        # Layer normalization for training stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.hidden_dims
        ])

        # Output layer for Q-value prediction
        self.q_out = nn.Linear(self.hidden_dims[-1], 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, state, action, masks):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        residual = x

        for i, (layer, res_layer, norm) in enumerate(zip(self.layers, self.residual_layers, self.layer_norms)):
            x = layer(x) + res_layer(residual)  # Main path + residual connection
            x = norm(x)
            x = F.relu(x)
            x = x * masks[i]  # Apply mask
            residual = x

        return self.q_out(x)


class MaskGenerator(nn.Module):
    """Generates environment-specific masks for network layers."""

    def __init__(self, state_dims, hidden_dims_list, num_environments):
        super().__init__()
        self.state_dims = state_dims
        self.hidden_dims_list = hidden_dims_list

        # Encoders for each environment's state
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU()
            ) for state_dim in state_dims
        ])

        # Learnable environment embeddings
        self.env_embeddings = nn.Parameter(torch.randn(num_environments, 64))

        # Mask networks for each layer in each environment
        self.mask_nets = nn.ModuleDict({
            str(env_id): nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128 + 64, 128),
                    nn.ReLU(),
                    nn.Linear(128, dim),
                    nn.Sigmoid()
                ) for dim in hidden_dims_list[env_id]
            ]) for env_id in range(num_environments)
        })

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights with orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, state, env_id):
        """Generate masks with gradient tracking."""
        features = self.encoders[env_id](state)
        env_embedding = self.env_embeddings[env_id].expand(state.size(0), -1)
        combined = torch.cat([features, env_embedding], dim=1)

        # Generate masks with gradient tracking
        masks = [net(combined) for net in self.mask_nets[str(env_id)]]
        return masks

    def get_sparsity_loss(self, masks):
        """Compute sparsity loss for the generated masks."""
        total_loss = 0
        for mask in masks:
            # L1 regularization for sparsity
            mask_loss = torch.mean(torch.abs(mask))
            total_loss = total_loss + mask_loss
        return total_loss


class AdaptiveNoiseInjector(nn.Module):
    """
    Handles dynamic noise injection based on environment state and performance.
    Adapts noise scales based on state-dependent factors and maintains history.
    """

    def __init__(self,
                 num_environments: int,
                 state_dims: List[int],
                 initial_scales: List[float]):
        """
        Initialize the noise injector.

        Args:
            num_environments: Number of environments to handle
            state_dims: List of state dimensions for each environment
            initial_scales: Initial noise scales for each environment
        """
        super().__init__()

        # Learnable environment-specific noise scales
        self.env_scales = nn.Parameter(torch.tensor(initial_scales, dtype=torch.float32))

        # State encoders for each environment
        self.state_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for state_dim in state_dims
        ])

        # Noise adaptation parameters
        self.decay_factor = 1.0
        self.min_scale = 0.01
        self.max_scale = 2.0
        self.adaptation_rate = 0.01

        # Track current noise states
        self.current_noise_state = defaultdict(lambda: None)
        self.noise_history = defaultdict(list)
        self.scale_history = defaultdict(list)

        # Environment-specific parameters
        self.state_dims = state_dims
        self.num_environments = num_environments
        self.initial_scales = initial_scales

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=1.414)
            nn.init.constant_(m.bias.data, 0.0)

    def get_env_scale(self, env_id: int, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the noise scale for a given environment and state.

        Args:
            env_id: Environment identifier
            state: Current environment state

        Returns:
            Computed noise scale as tensor
        """
        # Ensure positive scale through softplus
        base_scale = F.softplus(self.env_scales[env_id])

        # Get state-dependent scaling factor
        state_factor = self.state_encoders[env_id](state)

        # Compute final scale
        scale = base_scale * state_factor

        # Clip to reasonable range
        scale = torch.clamp(scale, self.min_scale, self.max_scale)

        # Store current state
        self.current_noise_state[env_id] = scale
        self.scale_history[env_id].append(scale.mean().item())

        return scale

    def sample(self, env_id: int, state: torch.Tensor, hidden_size: int) -> torch.Tensor:
        """
        Sample noise for the given environment and state.

        Args:
            env_id: Environment identifier
            state: Current environment state
            hidden_size: Size of the hidden layer to match

        Returns:
            Sampled noise tensor
        """
        # Get current scale
        scale = self.get_env_scale(env_id, state)

        # Generate noise
        noise = torch.randn(state.size(0), hidden_size, device=state.device)

        # Scale noise
        scaled_noise = noise * scale * self.decay_factor

        # Store in history
        self.noise_history[env_id].append(scaled_noise.mean().item())

        return scaled_noise

    def update_decay_factor(self, factor: float = 0.995) -> None:
        """
        Update the noise decay factor.

        Args:
            factor: Decay multiplier
        """
        self.decay_factor *= factor
        self.decay_factor = max(0.1, self.decay_factor)  # Ensure minimum exploration

    def increase_noise_scale(self, env_id: int, factor: float = 1.5) -> None:
        """
        Increase noise scale for specific environment.
        """
        with torch.no_grad():
            self.env_scales[env_id] *= factor
            # Ensure scale remains in a reasonable range
            device = self.env_scales.device  # Get the device of self.env_scales
            self.env_scales[env_id] = torch.clamp(
                self.env_scales[env_id],
                min=torch.log(torch.tensor(self.min_scale, device=device)),
                max=torch.log(torch.tensor(self.max_scale, device=device))
            )

    def decrease_noise_scale(self, env_id: int, factor: float = 0.75) -> None:
        """
        Decrease noise scale for specific environment.

        Args:
            env_id: Environment identifier
            factor: Scale decrease factor
        """
        with torch.no_grad():
            self.env_scales[env_id] *= factor
            # Ensure scale remains above minimum
            self.env_scales[env_id] = torch.clamp(
                self.env_scales[env_id],
                min=torch.log(torch.tensor(self.min_scale))
            )

    def get_current_noise(self, env_id: int) -> torch.Tensor:
        """
        Get current noise state for environment.

        Args:
            env_id: Environment identifier

        Returns:
            Current noise state or zero tensor if none exists
        """
        if self.current_noise_state[env_id] is None:
            return torch.zeros(1, device=self.env_scales.device)
        return self.current_noise_state[env_id]

    def get_noise_stats(self, env_id: int) -> Dict[str, float]:
        """
        Get noise statistics for environment.

        Args:
            env_id: Environment identifier

        Returns:
            Dictionary containing noise statistics
        """
        if not self.noise_history[env_id]:
            return {
                'mean': 0.0,
                'std': 0.0,
                'current_scale': self.initial_scales[env_id]
            }

        noise_history = torch.tensor(self.noise_history[env_id])
        scale_history = torch.tensor(self.scale_history[env_id])

        return {
            'mean': noise_history.mean().item(),
            'std': noise_history.std().item(),
            'current_scale': scale_history[-1].item() if len(scale_history) > 0 else self.initial_scales[env_id]
        }

    def reset_history(self, env_id: Optional[int] = None) -> None:
        """
        Reset noise history for specified or all environments.

        Args:
            env_id: Optional environment identifier. If None, resets all.
        """
        if env_id is None:
            self.noise_history.clear()
            self.scale_history.clear()
            self.current_noise_state.clear()
        else:
            self.noise_history[env_id].clear()
            self.scale_history[env_id].clear()
            self.current_noise_state[env_id] = None
