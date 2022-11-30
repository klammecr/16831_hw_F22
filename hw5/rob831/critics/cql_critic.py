from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb

from rob831.infrastructure import pytorch_util as ptu


class CQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['cql_alpha']

    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """ Implement DQN Loss """

        # Precursor:
        # torch.gather will go along the non-specified dimension and take indexing for the specified dimension
        # For example, the code below gets the result from the q net which are (s,a) pairs
        # The gather part will index by the action taken for the rollout and gather the q value for each state action pair

        # Calculate the prediction and target from the networks
        pred = torch.gather(self.q_net(ob_no), 1, ac_na.unsqueeze(1)).squeeze(1) # Take the current action at the current state and getting the Q value
        max_next_qvals = self.q_net_target(next_ob_no).max(dim = 1)[0]
        target = reward_n + self.gamma * max_next_qvals * (1 - terminal_n) # One step look ahead using the target network to approximate the target q value

        # The DQN loss is the difference between the Q Network Output and the target
        loss = self.loss.forward(pred, target)

        return loss, target, self.q_net(ob_no)


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # Compute the DQN Loss 
        loss, q_target_vals, q_vals = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
            )
        
        # CQL Implementation

        # Push the Q values down, Push up on the samples (s,a) in the data
        # We need the sample actions from mu

        # This q_vals will be of size [num_states, num_actions] with each row having a q value for each action in the state
        q_val_logsumexp = torch.logsumexp(q_vals, dim = 1) 

        # Take the logsumexp (softmax of the q values at a state) minus the expected q value for each state
        cql_loss = (q_val_logsumexp - torch.mean(q_vals, dim = 1)).mean()

        # Weight the CQL Loss by alpha, weight the bellman loss by half via the paper
        loss = 0.5 *loss + self.cql_alpha * cql_loss

        # Take a gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        info = {'Training Loss': ptu.to_numpy(loss)}

        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(q_vals).mean()
        info['OOD q-values'] = ptu.to_numpy(q_val_logsumexp).mean()

        return info


    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
