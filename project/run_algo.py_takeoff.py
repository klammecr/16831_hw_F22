"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import gym
import numpy as np
from stable_baselines3 import A2C, TD3, PPO
from stable_baselines3.a2c import MlpPolicy as MlpPolicyA2C
from stable_baselines3.td3.policies import MlpPolicy as MlpPolicyTD3
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy

import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import os

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_RLLIB = False
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'C:/Users/chris/dev/16831_hw_F22/project/results'
DEFAULT_COLAB = False

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def run(algo = "A2C", rllib=DEFAULT_RLLIB,output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO):
    
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create the gym environment
    env = gym.make("takeoff-aviary-v0")
    env = TakeoffAviary(record=True, gui = True)
    env = Monitor(env, log_dir)

    # Give some information about the observation and action space
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env,
                warn=True,
                skip_render_check=True)

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Number of training steps
    train_steps = 2e5

    # Choose your algorithm
    if algo == "A2C":
        model = A2C(MlpPolicyA2C,
                    env,
                    verbose=1
                    )
        model.learn(total_timesteps=train_steps, callback=callback) # Typically not enough
        model.save("TakeoffAviary_A2C")
    elif algo == "TD3":
        model = TD3(MlpPolicyTD3, env)
        model.learn(total_timesteps=train_steps) # Typically not enough
        model.save("TakeoffAviary_TD3")
    elif algo == "PPO":
        model = PPO(MlpPolicy, env)
        model.learn(total_timesteps=train_steps) # Typically not enough
    elif algo == "SAC":
        model = SAC("MlpPolicy", env)
        model.learn(total_timesteps=train_steps, callback=callback) # Typically not enough

    # # Plot results
    # results_plotter.plot_results([log_dir], train_steps, results_plotter.X_TIMESTEPS, algo)
    # plt.show()


    # # Evaluate the policy of interest
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    # print(f"Mean Reward: {mean_reward}")
    # print(f"Std Reward: {std_reward}")

    model = SAC.load()

    # Run the model in simulation
    obs = env.reset()
    rewards_all = {}
    rewards = []
    reward_traj = 0
    for i in range(3*env.SIM_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
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

    plt.title(f"Reward of a {algo} Agent - Takeoff")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.savefig(f"OneDrive - Personal/{algo}AgentPerformance.png")


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--rllib',      default=DEFAULT_RLLIB,        type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument("--algo", default="SAC", type=str)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
