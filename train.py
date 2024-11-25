import os
import numpy as np
from datetime import datetime
import torch
import gymnasium as gym
from gymnasium import make as gym_make
from typing import Dict, List, Tuple, Optional

from tensorboard_logger import AMNSTensorboardLogger
from sac_agent import LifelongSACAgent
from hyperparameters import ENV_HYPERPARAMS, TRAINING_CONFIG


def compute_research_metrics(agent: LifelongSACAgent, env_id: int, eval_reward: float, episode: int) -> Dict:
    """Compute detailed research metrics."""
    # Get current noise state
    current_noise = agent.noise_injector.get_current_noise(env_id)
    noise_stats = agent.noise_injector.get_noise_stats(env_id)

    # Get mask statistics
    current_masks = agent.prev_masks[env_id] if agent.prev_masks[env_id] is not None else []
    performance_stats = agent.get_performance_stats(env_id)
    learning_progress = agent.get_learning_progress(env_id)

    return {
        'noise_adaptation': {
            'scale': noise_stats['current_scale'],
            'entropy': -torch.mean(current_noise * torch.log(current_noise + 1e-10)).item(),
            'adaptation_rate': noise_stats['std'] / (noise_stats['mean'] + 1e-10)
        },
        'mask_noise_coordination': {
            'sync_rate': len(current_masks) / agent.actors[env_id].hidden_dims[-1] if current_masks else 0.0,
            'stability': learning_progress['stability'],
            'sparsity': np.mean([mask.mean().item() for mask in current_masks]) if current_masks else 0.0
        },
        'component_protection': {
            'preservation_rate': performance_stats['mean'] / (performance_stats['max'] + 1e-10),
            'recovery_speed': learning_progress['improvement_rate']
        }
    }


def evaluate_policy(env: gym.Env, agent: LifelongSACAgent, env_id: int, num_episodes: int = 5) -> float:
    """Evaluate the current policy without exploration."""
    total_rewards = []
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            action_result = agent.select_action(state, env_id, training=False)

            if is_discrete:
                action_index, _ = action_result
                next_state, reward, done, truncated, _ = env.step(action_index)
            else:
                next_state, reward, done, truncated, _ = env.step(action_result)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def init_env_info() -> Tuple[List[str], Dict[int, Dict], List[int], List[int], List[float], List[gym.spaces.Space]]:
    """Initialize environment information and parameters."""
    env_names = [
        "CartPole-v1",
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3"
    ]

    hyperparams = {i: ENV_HYPERPARAMS[env_name] for i, env_name in enumerate(env_names)}
    state_dims = []
    action_dims = []
    max_actions = []
    env_action_spaces = []

    for env_name in env_names:
        env = gym_make(env_name)
        state_dims.append(env.observation_space.shape[0])
        env_action_spaces.append(env.action_space)

        if isinstance(env.action_space, gym.spaces.Box):
            action_dims.append(env.action_space.shape[0])
            max_actions.append(float(env.action_space.high[0]))
        else:
            action_dims.append(env.action_space.n)
            max_actions.append(1.0)
        env.close()

    return env_names, hyperparams, state_dims, action_dims, max_actions, env_action_spaces


def save_checkpoint(agent: LifelongSACAgent, env_id: int, episode: int, reward: float,
                    path: str, metrics: Optional[Dict] = None) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'env_id': env_id,
        'episode': episode,
        'agent_state': agent.get_state_dict(),
        'reward': reward,
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    torch.save(checkpoint, path)


def train_lifelong_agent(render: bool = False) -> LifelongSACAgent:
    """Main training loop for lifelong learning across multiple environments."""
    # Initialize environment information
    global avg_reward
    env_names, hyperparams, state_dims, action_dims, max_actions, env_action_spaces = init_env_info()

    # Initialize logger
    logger = AMNSTensorboardLogger(
        base_dir="runs",
        env_names=env_names,
        experiment_name=f"AMNS_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Initialize agent
    agent = LifelongSACAgent(
        state_dims=state_dims,
        action_dims=action_dims,
        max_actions=max_actions,
        num_environments=len(env_names),
        env_names=env_names,
        env_action_spaces=env_action_spaces,
        hyperparams=hyperparams
    )

    # Training loop for each environment
    for env_id, env_name in enumerate(env_names):
        print(f"\nStarting training on {env_name} (Environment {env_id + 1}/{len(env_names)})")

        env_params = ENV_HYPERPARAMS[env_name]
        env_config = TRAINING_CONFIG[env_name]
        env = gym_make(env_name, render_mode="human" if render else None)
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        best_reward = float('-inf')
        episode_rewards = []
        stable_performance_counter = 0
        plateau_counter = 0

        checkpoint_dir = os.path.join('models', f'env_{env_id}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            for episode in range(env_config["episodes_per_env"]):
                state, _ = env.reset()
                episode_reward = 0.0
                steps = 0
                done = truncated = False
                warmup_phase = episode < env_config.get("warmup_episodes", 0)

                # Episode loop
                while not (done or truncated) and steps < env_config["max_steps_per_episode"]:
                    # Select and execute action
                    action_result = agent.select_action(state, env_id, training=not warmup_phase)

                    if is_discrete:
                        action_index, action_onehot = action_result
                        next_state, reward, done, truncated, _ = env.step(action_index)
                        agent.replay_buffer.push(env_id, state, action_onehot, reward, next_state, done or truncated)
                    else:
                        next_state, reward, done, truncated, _ = env.step(action_result)
                        agent.replay_buffer.push(env_id, state, action_result, reward, next_state, done or truncated)

                    # Training step
                    if not warmup_phase and len(agent.replay_buffer.buffers[env_id]) > env_params["min_buffer_size"]:
                        agent.train(env_params["batch_size"], env_id, logger)

                    state = next_state
                    episode_reward += reward
                    steps += 1

                # Update metrics
                episode_rewards.append(episode_reward)
                agent.update_performance_history(env_id, episode_reward)
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(
                    episode_rewards)

                # Log episode metrics
                logger.log_episode(env_id, episode, {
                    'reward': episode_reward,
                    'length': steps,
                    'average_reward': avg_reward
                })

                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                      f"Average = {avg_reward:.2f}, Steps = {steps}")

                # Evaluation and checkpointing
                if (episode + 1) % env_config["eval_frequency"] == 0:
                    eval_reward = evaluate_policy(env, agent, env_id)
                    research_metrics = compute_research_metrics(agent, env_id, eval_reward, episode)
                    logger.log_research_metrics(env_id, episode, research_metrics)

                    # Save best model
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        save_checkpoint(
                            agent, env_id, episode, best_reward,
                            os.path.join(checkpoint_dir, 'best_model.pt'),
                            research_metrics
                        )

                        if best_reward >= env_config["early_stop_threshold"]:
                            print(f"Early stopping threshold reached for {env_name}!")
                            break

                # Check for stable performance
                if avg_reward >= env_config["stable_threshold"]:
                    stable_performance_counter += 1
                    if stable_performance_counter >= env_config["stable_episodes_required"]:
                        print(f"\nEnvironment {env_name} solved with stable performance!")
                        break
                else:
                    stable_performance_counter = 0

                # Save periodic checkpoints
                if (episode + 1) % env_config["save_frequency"] == 0:
                    save_checkpoint(
                        agent, env_id, episode, avg_reward,
                        os.path.join(checkpoint_dir, f'checkpoint_ep_{episode + 1}.pt')
                    )

                # Check for learning plateau
                if len(episode_rewards) >= 100:
                    recent_avg = np.mean(episode_rewards[-50:])
                    old_avg = np.mean(episode_rewards[-100:-50])
                    if abs(recent_avg - old_avg) < env_config["plateau_threshold"]:
                        plateau_counter += 1
                        if plateau_counter >= 5:  # 5 consecutive plateau detections
                            print(f"Learning plateaued for {env_name}, adjusting hyperparameters...")
                            agent.adjust_exploration(env_id)
                            agent.adjust_learning_rates(env_id)
                            plateau_counter = 0
                    else:
                        plateau_counter = 0

            # Save final model for environment
            save_checkpoint(
                agent, env_id, env_config["episodes_per_env"], avg_reward,
                os.path.join(checkpoint_dir, 'final_model.pt')
            )

        except Exception as e:
            print(f"Error during training on {env_name}: {str(e)}")
            raise
        finally:
            env.close()

    logger.close()
    print("\nTraining completed!")
    return agent


def main():
    """Main execution function."""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        # Train agent
        agent = train_lifelong_agent(render=True)

        # Save final model
        final_save_path = os.path.join('models', 'final_lifelong_agent.pt')
        save_checkpoint(
            agent, -1, -1, 0.0, final_save_path,
            {env_name: agent.get_performance_stats(env_id)
             for env_id, env_name in enumerate(agent.env_names)}
        )
        print(f"Final model saved to {final_save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()