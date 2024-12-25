import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

# class Actor(Network):
#     def __init__(self, name, state_size, action_size, hidden_size):
#         super(Actor, self).__init__(name)
#         # --- define layers here ---
#         # TODO: Add some layers to keep depth of model enough for the increased input size
#         self.fa1 = nn.Linear(state_size, hidden_size)
#         self.fa2 = nn.Linear(hidden_size, hidden_size)
#         self.fa3 = nn.Linear(hidden_size, hidden_size)
#         self.fa4 = nn.Linear(hidden_size, action_size)
#         # --- define layers until here ---
#
#         self.dropout = nn.Dropout(0.2)
#
#         self.apply(super().init_weights)
#
#     def forward(self, states, visualize=False):
#         # --- define forward pass here ---
#         # TODO: Connect layer if you add new ones
#         x1 = torch.relu(self.fa1(states))
#         x2 = self.dropout(x1)
#         x3 = torch.relu(self.fa2(x2))
#         x4 = self.dropout(x3)
#         x5 = torch.relu(self.fa3(x4))
#         x6 = self.dropout(x5)
#         action = torch.tanh(self.fa3(x6))
#
#         # -- define layers to visualize here (optional) ---
#         if visualize and self.visual:
#             self.visual.update_layers(
#                 states,
#                 action,
#                 [x1, x3, x5],
#                 [self.fa1.bias, self.fa2.bias, self.fa3.bias, self.fa4.bias]
#             )
#         # -- define layers to visualize until here ---
#         return action

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)

        # --- Define layers here ---
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, hidden_size)
        self.fa4 = nn.Linear(hidden_size + state_size, hidden_size)  # Include skip connection input
        self.fa5 = nn.Linear(hidden_size, action_size)
        # --- Define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- Define forward pass here ---
        x1 = F.leaky_relu(self.fa1(states))  # Apply LeakyReLU activation
        x2 = F.leaky_relu(self.fa2(x1))
        x3 = F.leaky_relu(self.fa3(x2))

        # Skip connection: Concatenate input states and the output of fa3
        x_combined = torch.cat((x3, states), dim=1)  # Combine with input states
        x4 = F.leaky_relu(self.fa4(x_combined))
        action = torch.tanh(self.fa5(x4))  # Final action output

        # -- Define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(
                states,
                action,
                [x1, x2, x3, x_combined, x4],
                [self.fa1.bias, self.fa2.bias, self.fa3.bias, self.fa4.bias, self.fa5.bias]
            )
        # -- Define layers to visualize until here ---
        return action


# class Critic(Network):
#     def __init__(self, name, state_size, action_size, hidden_size):
#         super(Critic, self).__init__(name)
#
#         # --- define layers here ---
#         # TODO: Add some layers to keep depth of model enough for the increased input size
#         self.l1 = nn.Linear(state_size, int(hidden_size / 2))
#         self.l2 = nn.Linear(action_size, int(hidden_size / 2))
#         self.l3 = nn.Linear(hidden_size, hidden_size)
#         self.l4 = nn.Linear(hidden_size, 1)
#         # --- define layers until here ---
#
#         self.apply(super().init_weights)
#
#     def forward(self, states, actions):
#         # --- define forward pass here ---
#         # TODO: Connect layer if you add new ones
#         xs = torch.relu(self.l1(states))
#         xa = torch.relu(self.l2(actions))
#         x = torch.cat((xs, xa), dim=1)
#         x = torch.relu(self.l3(x))
#         x = self.l4(x)
#         return x


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)

        # --- Define layers here ---
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(action_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size + hidden_size, hidden_size)  # Skip connection after combining
        self.l5 = nn.Linear(hidden_size, 1)
        # --- Define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, actions):
        # --- Define forward pass here ---
        xs = F.leaky_relu(self.l1(states))  # Process state inputs
        xa = F.leaky_relu(self.l2(actions))  # Process action inputs
        x_combined = F.leaky_relu(self.l3(xs + xa))  # Combine state and action features

        # Skip connection: Concatenate xs and x_combined
        x_skip_combined = torch.cat((x_combined, xs), dim=1)  # Combine with earlier state features
        x4 = F.leaky_relu(self.l4(x_skip_combined))
        q_value = self.l5(x4)  # Final Q-value output

        return q_value


class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        print(f'Actor:\n{self.actor}')
        print(f'Critic:\n{self.critic}')

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size

    def train(self, state, action, reward, state_next, done):
        # optimize critic
        action_next = self.actor_target(state_next)
        Q_next = self.critic_target(state_next, action_next)
        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q = self.critic(state, action)

        loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        pred_a_sample = self.actor(state)
        loss_actor = -1 * (self.critic(state, pred_a_sample)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]
