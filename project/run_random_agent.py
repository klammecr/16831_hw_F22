import gym
from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from navigation_aviary import NavigationAviary
from navigation_aviary_rgb import NavigationAviaryVision
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from gym.envs.registration import register

# Register our custom environment
register(
    id = "navigation-aviary-v0",
    entry_point='navigation_aviary:NavigationAviary',
)
register(
id = "vision-navigation-aviary-v0",
entry_point='navigation_aviary_rgb:NavigationAviaryVision',
)

# Create the gym environment
env = gym.make("navigation-aviary-v0")
env = NavigationAviary(record=True, gui = True)

# env = gym.make("vision-navigation-aviary-v0") 
# env = NavigationAviaryVision(record=True, gui = True, num_obstacles=6)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
check_env(env,
            warn=True,
            skip_render_check=True
            )

# Run the random agent in simulation
obs = env.reset()
rewards_all = {}
rewards = []
reward_traj = 0
i = 0

while len(rewards_all) < 10:
    obs, reward, done, info = env.step(env.action_space.sample())
    reward_traj += reward
    rewards.append(reward_traj)
    env.render()
    if done:
        obs = env.reset()
        rewards_all[i] = rewards
        i += 1
        rewards = []
        reward_traj = 0
env.close()

colors = iter(cm.rainbow(np.linspace(0, 1, len(rewards_all))))
start_iter = 0

for end_iter in rewards_all:
    # Get the color for the plot
    c = next(colors)

    # Get the trajectory information
    traj = rewards_all[end_iter]

    # Plot
    x = list(range(len(traj)))
    plt.plot(x, traj[:], color = c, linestyle="dashed")

    start_iter  = int(end_iter) + 1

plt.title(f"Reward of a Random Agent - Navigation")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig(f"results/Random NavAgentPerformance.png")