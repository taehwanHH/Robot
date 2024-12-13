import torch
import torch.nn as nn

from DDPG_module.MLP import MultiLayerPerceptron as MLP
from param import Hyper_Param,Robot_Param,Comm_Param
from robotic_env import RoboticEnv

DEVICE = Hyper_Param['DEVICE']
_iscomplex = Comm_Param['_iscomplex']
channel_type = Comm_Param['channel_type']
SNR = Comm_Param['SNR']
qam_order = Comm_Param['qam_order']

Sense_max = Robot_Param['Sense_max']
Act_max = Robot_Param['Act_max']



class Actor(nn.Module,RoboticEnv):

    def __init__(self):
        super(Actor, self).__init__()
        RoboticEnv.__init__(self)


        self.mlp = MLP(self.state_dim, self.action_dim,
                       num_neurons=Hyper_Param['num_neurons'],
                       hidden_act='ReLU',
                       out_act='Identity')


    def forward(self, state):
        # seed = torch.randint(low=0, high=70000, size=(1,)).to(DEVICE).item()

        ## Remote central unit
        action = self.mlp(state)
        action = action / (1+torch.abs(action)) # soft sign

        return action


class Critic(nn.Module, RoboticEnv):
    def __init__(self):
        super(Critic, self).__init__()
        RoboticEnv.__init__(self)
        self.state_encoder = MLP(self.state_dim, 32,
                                 num_neurons=[64],
                                 out_act='ReLU')  # single layer model
        self.action_encoder = MLP(self.action_dim, 32,
                                  num_neurons=[32],
                                  out_act='ReLU')  # single layer model
        self.q_estimator = MLP(64, 1,
                               num_neurons=Hyper_Param['critic_num_neurons'],
                               hidden_act='ReLU',
                               out_act='Identity')

    def forward(self, x, a):
        emb = torch.cat([self.state_encoder(x), self.action_encoder(a)], dim=-1)
        return self.q_estimator(emb)


class DDPG(nn.Module,RoboticEnv):

    def __init__(self,
                 critic: nn.Module,
                 critic_target: nn.Module,
                 actor: nn.Module,
                 actor_target: nn.Module,
                 epsilon: float = 1,
                 lr_critic: float = 0.0005,
                 lr_actor: float = 0.001,
                 gamma: float = 0.99):

        super(DDPG, self).__init__()

        RoboticEnv.__init__(self)
        self.critic = critic
        self.actor = actor
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.epsilon = epsilon

        # setup optimizers
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                           lr=lr_critic)

        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)

        # setup target networks
        critic_target.load_state_dict(critic.state_dict())
        self.critic_target = critic_target
        actor_target.load_state_dict(actor.state_dict())
        self.actor_target = actor_target

        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state, noise):
        with torch.no_grad():

            action = self.actor(state)*torch.tensor(self.action_space.high, dtype=torch.float32, device=DEVICE)+noise.to(DEVICE)
            clamped_action = torch.clamp(action, min=torch.tensor(self.action_space.low, dtype=torch.float32, device=DEVICE), max=torch.tensor(self.action_space.high, dtype=torch.float32, device=DEVICE))

        return clamped_action

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            critic_target = r + self.gamma * self.critic_target(ns, self.actor_target(ns)) * (1 - done)
        critic_loss = self.criteria(self.critic(s, a), critic_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        print(f"critic_loss: {critic_loss.item()}")

        # compute actor loss and update the actor parameters
        actor_loss = -self.critic(s, self.actor(s)).mean()  # !!!! Impressively simple
        print(f"actor_loss: {actor_loss.item()}")
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # return actor_loss.item()

