"""Example demonstrating navigation arena with episode metrics tracking.

Shows how to use Gymnasium's RecordEpisodeStatistics wrapper for automatic
metrics collection without modifying the core environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import (
    GridEnvironment,
    NavigationArena,
    SimpleField,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


def main():
    print("=" * 70)
    print("NAVIGATION WITH METRICS TRACKING")
    print("=" * 70)
    
    # 1. Create configuration
    config = GridConfig(n_x=10, n_y=10, n_z=5, d_max=1)
    print(f"\nGrid size: {config.n_x} x {config.n_y} x {config.n_z}")
    
    # 2. Create components
    field = SimpleField(config)
    actor = GridActor(noise_prob=0.1)
    
    # 3. Define positions
    initial_position = GridPosition(2, 2, 2)
    target_position = GridPosition(8, 8, 4)
    vicinity_radius = 1.5
    
    print(f"Start: ({initial_position.i}, {initial_position.j}, {initial_position.k})")
    print(f"Target: ({target_position.i}, {target_position.j}, {target_position.k})")
    
    # 4. Create arena
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        boundary_mode='clip',
        distance_reward_weight=-0.1,
        vicinity_bonus=1.0,
        step_penalty=-0.1,
        terminate_on_reach=False,
        use_distance_decay=True,
        decay_rate=0.3
    )
    
    # 5. Create base environment
    base_env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=42
    )
    
    # 6. WRAP with RecordEpisodeStatistics
    # This wrapper automatically tracks:
    # - episode_return (cumulative reward)
    # - episode_length (number of steps)
    # - episode_time (wall-clock time)
    env = RecordEpisodeStatistics(base_env)
    
    print("\n✓ Environment wrapped with RecordEpisodeStatistics")
    print("  Automatic tracking: episode_return, episode_length, episode_time")
    
    # 7. Run multiple episodes to demonstrate metrics
    print("\n" + "=" * 70)
    print("Running 3 episodes with random policy")
    print("=" * 70)
    
    all_episode_stats = []
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=42 + episode)
        
        done = False
        step = 0
        episode_reward = 0.0
        
        while not done and step < 100:  # Run full episode
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: reward={episode_reward:.2f}")
        
        # At episode end, info dict contains episode statistics
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Final step: {step}")
        print(f"  Episode reward (manual): {episode_reward:.2f}")
        
        # RecordEpisodeStatistics adds 'episode' key to info at episode end
        if 'episode' in info:
            print(f"  Episode length (wrapper): {info['episode']['l']}")
            print(f"  Episode return (wrapper): {info['episode']['r']:.2f}")
            print(f"  Episode time (wrapper): {info['episode']['t']:.3f}s")
            
            all_episode_stats.append({
                'length': info['episode']['l'],
                'return': info['episode']['r'],
                'time': info['episode']['t']
            })
        else:
            print("  (Episode stats not available - episode didn't terminate)")
            all_episode_stats.append({
                'length': step,
                'return': episode_reward,
                'time': 0.0
            })
    
    # 8. Summary statistics
    if all_episode_stats:
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        avg_return = np.mean([s['return'] for s in all_episode_stats])
        avg_length = np.mean([s['length'] for s in all_episode_stats])
        avg_time = np.mean([s['time'] for s in all_episode_stats])
        
        print(f"\nAcross {len(all_episode_stats)} episodes:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f} steps")
        print(f"  Average Time: {avg_time:.3f}s")
    
    # 9. Show what's available in info dict
    print("\n" + "=" * 70)
    print("AVAILABLE METRICS IN INFO DICT")
    print("=" * 70)
    
    # Run one more step to get info
    obs, info = env.reset(seed=99)
    obs, reward, terminated, truncated, info = env.step(0)
    
    print("\nAt each step, info dict contains:")
    print("\nFrom Arena State (via to_dict()):")
    arena_keys = [
        'step_count', 'last_action', 'last_reward', 'position',
        'last_position', 'last_displacement', 'out_of_bounds',
        'cumulative_reward', 'target_reached', 'target_position',
        'vicinity_radius', 'distance_reward_weight', 'vicinity_bonus',
        'step_penalty', 'use_distance_decay', 'decay_rate'
    ]
    for key in arena_keys:
        if key in info:
            value = info[key]
            if hasattr(value, '_asdict'):  # NamedTuple
                print(f"  - {key}: {value}")
            elif isinstance(value, (int, float, bool)):
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: <{type(value).__name__}>")
    
    print("\nFrom Environment:")
    print(f"  - episode_step: {info.get('episode_step', 'N/A')}")
    print(f"  - is_terminal: {info.get('is_terminal', 'N/A')}")
    
    print("\nFrom RecordEpisodeStatistics (at episode end):")
    print("  - episode['r']: Total return")
    print("  - episode['l']: Episode length")
    print("  - episode['t']: Episode duration (seconds)")
    
    # 10. Example: Custom metrics collection
    print("\n" + "=" * 70)
    print("CUSTOM METRICS EXAMPLE")
    print("=" * 70)
    
    print("\nYou can easily extract custom metrics from info dict:")
    print("""
    # During training loop:
    obs, reward, done, truncated, info = env.step(action)
    
    # Extract navigation-specific metrics
    distance_to_target = np.linalg.norm([
        info['position'].i - info['target_position'].i,
        info['position'].j - info['target_position'].j,
        info['position'].k - info['target_position'].k
    ])
    
    in_vicinity = info['target_reached']
    cumulative_reward = info['cumulative_reward']
    
    # Log to TensorBoard/WandB (in training script):
    # writer.add_scalar('navigation/distance', distance_to_target, step)
    # writer.add_scalar('navigation/in_vicinity', in_vicinity, step)
    """)
    
    env.close()
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

