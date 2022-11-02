import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # Return the action that maxinmizes the Q-value
        q_vals = self.critic.qa_values(observation)
        best_act = np.argmax(q_vals, axis = -1)

        # at the current observation as the output
        return best_act.squeeze()