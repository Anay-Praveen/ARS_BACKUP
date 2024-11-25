import os
import numpy as np
from datetime import datetime
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Dict, List, Any
"""tensorboard --logdir=C:/Users/anayp/PycharmProjects/PythonProject1/runs """

class AMNSTensorboardLogger:
    """
    Comprehensive TensorBoard logger for AMNS (Adaptive Mask-Noise Synchronized Learning).
    Tracks all metrics relevant to the research questions and learning progress.
    """

    def __init__(self, base_dir: str = "runs", env_names: List[str] = None, experiment_name: str = None):

        """Initialize the logger with environment-specific writers."""
        i=1
        self.env_names = env_names or []
        experiment_name = experiment_name or "AMNS_Experiment"
        self.log_dir = os.path.join(base_dir, f"{experiment_name}_{i}")

        # Initialize writers for each environment
        self.writers: Dict[int, SummaryWriter] = {}
        for env_id, env_name in enumerate(self.env_names):
            log_path = os.path.join(self.log_dir, f"{env_name}")
            self.writers[env_id] = SummaryWriter(log_path)

        # Metric history for analysis
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = defaultdict(list)
        self.noise_scales = defaultdict(list)
        self.mask_sparsity = defaultdict(list)

        # Performance tracking
        self.best_performance = defaultdict(lambda: float('-inf'))
        self.performance_window = 100

        # Create log directory structure
        os.makedirs(self.log_dir, exist_ok=True)
        i=+1

    def log_episode(self, env_id: int, episode: int, metrics: Dict[str, float]) -> None:
        """Log episode-level metrics with detailed breakdowns."""
        writer = self.writers[env_id]
        env_name = self.env_names[env_id]

        # Basic episode metrics
        writer.add_scalar(f'{env_name}/episode/reward', metrics['reward'], episode)
        writer.add_scalar(f'{env_name}/episode/length', metrics['length'], episode)

        # Track moving averages
        self.episode_rewards[env_id].append(metrics['reward'])
        self.episode_lengths[env_id].append(metrics['length'])

        if len(self.episode_rewards[env_id]) > self.performance_window:
            moving_avg_reward = np.mean(self.episode_rewards[env_id][-self.performance_window:])
            moving_avg_length = np.mean(self.episode_lengths[env_id][-self.performance_window:])
            writer.add_scalar(f'{env_name}/episode/moving_avg_reward', moving_avg_reward, episode)
            writer.add_scalar(f'{env_name}/episode/moving_avg_length', moving_avg_length, episode)

            # Update best performance
            if moving_avg_reward > self.best_performance[env_id]:
                self.best_performance[env_id] = moving_avg_reward

    def log_training_step(self, env_id: int, step: int, metrics: Dict[str, float]) -> None:
        """Log detailed training metrics per step."""
        writer = self.writers[env_id]
        env_name = self.env_names[env_id]

        # Actor-Critic losses
        writer.add_scalar(f'{env_name}/training/actor_loss', metrics['actor_loss'], step)
        writer.add_scalar(f'{env_name}/training/critic1_loss', metrics['critic1_loss'], step)
        writer.add_scalar(f'{env_name}/training/critic2_loss', metrics['critic2_loss'], step)

        # Value metrics
        writer.add_scalar(f'{env_name}/training/q_value_mean', metrics['q_value_mean'], step)
        writer.add_scalar(f'{env_name}/training/q_value_std', metrics['q_value_std'], step)

        # Gradient norms
        writer.add_scalar(f'{env_name}/training/actor_grad_norm', metrics['actor_grad_norm'], step)
        writer.add_scalar(f'{env_name}/training/critic_grad_norm', metrics['critic_grad_norm'], step)

    def log_research_metrics(self, env_id: int, step: int, metrics: Dict[str, Dict[str, float]]) -> None:
        """Log metrics specifically related to research questions."""
        writer = self.writers[env_id]
        env_name = self.env_names[env_id]

        # RQ1: Noise Adaptation
        noise_metrics = metrics['noise_adaptation']
        writer.add_scalar(f'{env_name}/noise/scale', noise_metrics['scale'], step)
        writer.add_scalar(f'{env_name}/noise/entropy', noise_metrics['entropy'], step)
        writer.add_scalar(f'{env_name}/noise/adaptation_rate', noise_metrics['adaptation_rate'], step)
        self.noise_scales[env_id].append(noise_metrics['scale'])

        # RQ2: Mask-Noise Coordination
        mask_metrics = metrics['mask_noise_coordination']
        writer.add_scalar(f'{env_name}/coordination/sync_rate', mask_metrics['sync_rate'], step)
        writer.add_scalar(f'{env_name}/coordination/stability', mask_metrics['stability'], step)
        self.mask_sparsity[env_id].append(mask_metrics['sparsity'])

        # RQ3: Component Protection
        protection_metrics = metrics['component_protection']
        writer.add_scalar(f'{env_name}/protection/preservation_rate',
                          protection_metrics['preservation_rate'], step)
        writer.add_scalar(f'{env_name}/protection/recovery_speed',
                          protection_metrics['recovery_speed'], step)

    def log_environment_transition(self, from_env_id: int, to_env_id: int,
                                   step: int, metrics: Dict[str, float]) -> None:
        """Log metrics during environment transitions."""
        writer = self.writers[to_env_id]
        to_env_name = self.env_names[to_env_id]

        # Performance transfer
        writer.add_scalar(f'{to_env_name}/transition/performance_drop',
                          metrics['performance_drop'], step)
        writer.add_scalar(f'{to_env_name}/transition/recovery_steps',
                          metrics['recovery_steps'], step)

        # Adaptation metrics
        writer.add_scalar(f'{to_env_name}/transition/mask_adaptation_time',
                          metrics['mask_adaptation_time'], step)
        writer.add_scalar(f'{to_env_name}/transition/noise_adaptation_time',
                          metrics['noise_adaptation_time'], step)

    def log_hyperparameters(self, env_id: int, hparams: Dict[str, Any],
                            metrics: Dict[str, float]) -> None:
        """Log hyperparameters and their corresponding metrics."""
        writer = self.writers[env_id]
        metric_dict = {f'hparam/{k}': v for k, v in metrics.items()}
        writer.add_hparams(hparams, metric_dict)

    def create_analysis_plots(self, env_id: int) -> None:
        """Generate and save analysis plots."""
        writer = self.writers[env_id]
        env_name = self.env_names[env_id]

        # Reward Distribution
        if self.episode_rewards[env_id]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.episode_rewards[env_id], bins=30, alpha=0.7)
            ax.set_title(f'Reward Distribution - {env_name}')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            writer.add_figure(f'{env_name}/analysis/reward_distribution', fig)
            plt.close(fig)

        # Noise Scale Evolution
        if self.noise_scales[env_id]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.noise_scales[env_id])
            ax.set_title(f'Noise Scale Evolution - {env_name}')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Noise Scale')
            writer.add_figure(f'{env_name}/analysis/noise_evolution', fig)
            plt.close(fig)

        # Mask Sparsity Evolution
        if self.mask_sparsity[env_id]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.mask_sparsity[env_id])
            ax.set_title(f'Mask Sparsity Evolution - {env_name}')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Sparsity')
            writer.add_figure(f'{env_name}/analysis/mask_sparsity', fig)
            plt.close(fig)

    def save_experiment_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration for reproducibility."""
        config_path = os.path.join(self.log_dir, 'experiment_config.txt')
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f'{key}: {value}\n')

    def close(self) -> None:
        """Create final analysis and close all writers."""
        for env_id in self.writers.keys():
            self.create_analysis_plots(env_id)

        # Save final performance summary
        summary_path = os.path.join(self.log_dir, 'performance_summary.txt')
        with open(summary_path, 'w') as f:
            for env_id, writer in self.writers.items():
                env_name = self.env_names[env_id]
                avg_reward = np.mean(self.episode_rewards[env_id][-self.performance_window:])
                best_reward = self.best_performance[env_id]
                f.write(f'{env_name}:\n')
                f.write(f'  Final Average Reward: {avg_reward:.2f}\n')
                f.write(f'  Best Average Reward: {best_reward:.2f}\n\n')
                writer.close()

