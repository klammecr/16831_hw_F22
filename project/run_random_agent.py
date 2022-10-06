import gym
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

# Create the gym environment
env = gym.make("takeoff-aviary-v0")
env = TakeoffAviary(record=True, gui = True)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
check_env(env,
            warn=True,
            skip_render_check=True
            )
obs = env.reset()
rewards_all = {}
rewards = []
reward_traj = 0
for i in range(10*240):
    obs, reward, done, info = env.step(env.action_space.sample())
    reward_traj += reward
    rewards.append(reward_traj)
    env.render()
    if done or reward < -2:
        obs = env.reset()
        rewards_all[i] = rewards
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
    x = list(range(start_iter, int(end_iter) + 1))
    plt.plot(x, traj[:], color = c, linestyle="dashed")

    start_iter  = int(end_iter) + 1

plt.title("Reward of a Random Agent - Takeoff")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig("results/RandomAgentPerformance.png")