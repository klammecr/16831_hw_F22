from copy import copy
import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        q_vals     = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_vals, terminals)
        train_log  = self.actor.update(observations, actions, advantages, q_vals, len(rewards_list), num_updates=1)

        return train_log


    def calculate_q_vals(self, rewards_list):
        """
            Monte Carlo estimation of the Q function.
        """
        # Estimate for the q values
        q_values = []

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.
        if not self.reward_to_go:
            # For each trajectory, Q^{pi}(s_t, a_t) is the discounted expected return after taking action a_t at state s_t
            # If you think of an action-state tree, once taking this action at the state, this is the sum of the discounted returns following the policy
            fn = self._discounted_return
        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            fn = self._discounted_cumsum
        
        #use the whole traj for each timestep
        for traj_rewards in rewards_list:
            q_values.append(fn(traj_rewards))

        return np.concatenate(q_values)

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:

            values_normalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_normalized.ndim == q_values.ndim

            # Find the difference to adjust the mean and std to match the mean of the q values
            values = values_normalized * np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rewards = np.concatenate(rewards_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # If we are at the end of the trajectory, we will just have the reward(s) - V(s)
                    delta = rewards[i] + self.gamma * values[i+1] * float(not bool(terminals[i])) - values[i]

                    # If we are at the end of a trajectory, this will just be delta as the R(s) - V(s) how much better the reward is from the expected value of the state
                    advantages[i] = delta + self.gamma * self.gae_lambda * float(not bool(terminals[i])) * advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                # Compute advantage estimates using q_values, and values as baselines
                # The advantage is just the difference between the Q value and the value function
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        gamma_arr = np.array([self.gamma ** i for i in range(len(rewards))])
        # This is just the discounted sum of rewards
        return np.repeat(np.sum(rewards * gamma_arr), len(rewards))

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        # This one is a bit different, each index t' should tell us how much reward we have left to obtain
        # We should be making each index t', calculating the gammas needed for the future then doing the calculation

        # The gamma array will be different depending on where we are
        # if t = T-1, we will only have the return
        # if T = 0, we will have the full range of gammas
        list_of_discounted_cumsums = np.zeros(len(rewards)+1)

        for t in range(len(rewards) - 1, -1, -1):
            list_of_discounted_cumsums[t] = rewards[t] + self.gamma * list_of_discounted_cumsums[t+1]
        
        return list_of_discounted_cumsums[:-1]