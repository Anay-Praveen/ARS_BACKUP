"""
replay_buffer.py

Defines a Multi-Environment Replay Buffer for Lifelong SAC Agent.
Handles both prioritized sampling and uniform sampling.
"""

import numpy as np
from collections import deque


class MultiEnvReplayBuffer:
    """
    Replay buffer supporting multiple environments with distinct action spaces.
    Includes functionality for prioritized sampling.
    """

    def __init__(self, capacity, num_environments, env_action_spaces, alpha=0.6, beta=0.4):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum buffer size.
            num_environments (int): Number of environments.
            env_action_spaces (list): List of action spaces for each environment.
            alpha (float): Priority exponent for prioritized sampling.
            beta (float): Importance sampling exponent.
        """
        self.capacity = capacity
        self.buffers = {i: deque(maxlen=capacity) for i in range(num_environments)}
        self.priorities = {i: deque(maxlen=capacity) for i in range(num_environments)}
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.env_action_spaces = env_action_spaces  # Action space details

        # Tracks insertion metrics
        self.insertion_counts = {i: 0 for i in range(num_environments)}

    def push(self, env_id, state, action, reward, next_state, done):
        """
        Store transitions in the buffer with consistent action representation.

        Args:
            env_id (int): ID of the environment.
            state (array): Current state.
            action: Action taken (can be int, float, or array).
            reward (float): Reward received.
            next_state (array): Next state.
            done (bool): Whether the episode ended.
        """
        # Convert action to consistent format
        if isinstance(self.env_action_spaces[env_id], int):  # Discrete actions
            if isinstance(action, (int, np.int32, np.int64)):
                one_hot_action = np.zeros(self.env_action_spaces[env_id])
                one_hot_action[action] = 1
                action = one_hot_action
            else:
                action = np.array(action, dtype=np.float32)
        else:  # Continuous actions
            action = np.array(action, dtype=np.float32)

        # Convert all inputs to numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        # Default maximum priority
        max_priority = max(self.priorities[env_id], default=1.0)

        # Store the transition and priority
        self.buffers[env_id].append((state, action, reward, next_state, done))
        self.priorities[env_id].append(max_priority)
        self.insertion_counts[env_id] += 1

    def sample(self, env_id, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            env_id (int): ID of the environment to sample from.
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices).
        """
        if len(self.buffers[env_id]) < batch_size:
            return None

        # Prioritized sampling
        priorities = np.array(self.priorities[env_id])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffers[env_id]), batch_size, p=probabilities)
        batch = [self.buffers[env_id][i] for i in indices]

        # Unpack batch
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Compute importance sampling weights
        weights = (len(self.buffers[env_id]) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, env_id, indices, new_priorities):
        """
        Update priorities for sampled transitions.

        Args:
            env_id (int): ID of the environment.
            indices (list): Indices of transitions to update.
            new_priorities (list): New priorities for each transition.
        """
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priorities[env_id]):
                self.priorities[env_id][idx] = max(priority, 1e-6)  # Avoid zero priority

    def get_stats(self):
        """
        Get buffer statistics for monitoring.

        Returns:
            dict: Buffer statistics including buffer sizes and insertion counts.
        """
        return {
            'buffer_sizes': {env_id: len(buffer) for env_id, buffer in self.buffers.items()},
            'insertion_counts': dict(self.insertion_counts)
        }

    def clear(self, env_id=None):
        """
        Clear the replay buffer for a specific environment or all environments.

        Args:
            env_id (int, optional): Environment ID to clear. If None, clears all buffers.
        """
        if env_id is None:
            for env_id in self.buffers.keys():
                self.buffers[env_id].clear()
                self.priorities[env_id].clear()
                self.insertion_counts[env_id] = 0
        else:
            self.buffers[env_id].clear()
            self.priorities[env_id].clear()
            self.insertion_counts[env_id] = 0
