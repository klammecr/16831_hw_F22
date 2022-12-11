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
from stable_baselines3.common.noise import NormalActionNoise

import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import os

from navigation_aviary import NavigationAviary
from gym.envs.registration import register

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

# CEM Implementation
from rob831.scripts.run_hw4_mb import MB_Trainer


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

    # Register our custom environment
    register(
        id = "navigation-aviary-v0",
        entry_point='navigation_aviary:NavigationAviary',
    )

    # Create the gym environment
    env = gym.make("navigation-aviary-v0")
    env = NavigationAviary(record=False, gui = False)
    #env = NavigationAviary(record=True, gui = True)
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
    train_steps = 5e5
    action_noise = NormalActionNoise(np.array([0,0,0,0]), np.array([0.1, 0.1, 0.1, 0.1]))


    # Choose your algorithm
    if algo == "A2C":
        model = A2C(MlpPolicyA2C,
                    env,
                    verbose=1
                    )
        model.learn(total_timesteps=train_steps, \
                    callback=callback) # Typically not enough
        model.save("NavAviary_A2C")
    elif algo == "TD3":
        model = TD3(MlpPolicyTD3, env, action_noise=action_noise)
        model.learn(total_timesteps=train_steps, callback = callback) # Typically not enough
        model.save("NavAviary_TD3")
    elif algo == "PPO":
        model = PPO("MlpPolicy", env)
        model.learn(total_timesteps=train_steps, callback=callback) # Typically not enough
        model.save("NavAviary_PPO")
    elif algo == "SAC":
        model = SAC("MlpPolicy", env)
        model.learn(total_timesteps=train_steps, callback=callback) # Typically not enough
        model.save("NavAviary_SAC")
    elif algo == "CEM":
        # Set the parameters for CEM
        agent_params = {}
        agent_params["seed"]                          = 6969
        agent_params["no_gpu"]                        = False
        agent_params["which_gpu"]                     = 0
        agent_params['video_log_freq']                = -1
        agent_params['scalar_log_freq']               = 1
        agent_params['save_params']                   = False
        agent_params["env_name"]                      = "navigation-aviary-v0"
        agent_params["exp_name"]                      = "MB_Exp"
        agent_params['n_iter']                        = 20#train_steps
        agent_params['batch_size']                    = 5000
        agent_params['batch_size_initial']            = 5000
        agent_params['train_batch_size']              = 512
        agent_params['eval_batch_size']               = 400
        agent_params['ac_dim']                        = env.action_space.shape
        agent_params['ob_dim']                        = env.observation_space.shape
        agent_params['n_layers']                      = 2
        agent_params['mpc_horizon']                   = 15
        agent_params['ensemble_size']                 = 5
        agent_params['size']                          = 250
        agent_params['ep_len']                        = 500
        agent_params['num_agent_train_steps_per_iter']= 1500
        agent_params['mpc_action_sampling_strategy']  = "cem"
        agent_params['mpc_num_action_sequences']      = 1000
        agent_params['cem_iterations']                = 4
        agent_params['cem_num_elites']                = 8
        agent_params['cem_alpha']                     = 1.0
        agent_params['learning_rate']                 = 1e-3
        agent_params['add_sl_noise']                  = True

        # Setup log directory
        logdir_prefix = "results"

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        if not (os.path.exists(data_path)):
            os.makedirs(data_path)

        
        logdir = logdir_prefix + agent_params["exp_name"] + '_' + agent_params["env_name"] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        agent_params['logdir'] = logdir
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

        # Setup the Model Based Training to Run!
        model = MB_Trainer(agent_params)
        model.run_training_loop()

    # Plot results
    results_plotter.plot_results([log_dir], train_steps, results_plotter.X_TIMESTEPS, algo)
    plt.show()


    # Evaluate the policy of interest
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f"Mean Reward: {mean_reward}")
    print(f"Std Reward: {std_reward}")

    # if algo == "SAC":
    #     model = SAC.load("results/SAC_model.zip")

    # Run the model in simulation
    obs = env.reset()
    rewards_all = {}
    rewards = []
    reward_traj = 0
    i = 0
    while len(rewards_all) < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
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

    plt.title(f"Reward of a {algo} Agent - Navigation")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.savefig(f"results/{algo}NavAgentPerformance.png")


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning for navigating')
    parser.add_argument('--rllib',      default=DEFAULT_RLLIB,        type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument("--algo", default="CEM", type=str)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
