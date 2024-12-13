import numpy as np
import torch
import argparse
from param import Comm_Param, Hyper_Param

def to_tensor(np_array: np.array, size=None) -> torch.tensor:
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor


def to_numpy(torch_tensor: torch.tensor) -> np.array:
    return torch_tensor.cpu().detach().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse SNR and channel_type parameters")

    # argparse로 받을 파라미터들 정의 (snr과 channel_type만)
    parser.add_argument('--snr', type=int, default=Comm_Param['SNR'], help='Signal-to-noise ratio')
    parser.add_argument('--channel_type', type=str, default=Comm_Param['channel_type'], help='Type of communication channel')
    parser.add_argument('--latency', type=int, default=Comm_Param['comm_latency'], help='Communication latency')
    parser.add_argument('--_iscomplex', type=bool, default=Comm_Param['_iscomplex'], help='Channel complex')


    args = parser.parse_args()
    return args

class OrnsteinUhlenbeckProcess:
    """
    OU process; The original implementation is provided by minimalRL.
    https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    """

    def __init__(self, mu):
        self.theta, self.dt, self.sigma = Hyper_Param['theta'], Hyper_Param['dt'], Hyper_Param['sigma']
        self.mu = mu
        self.x_prev = torch.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(size=self.mu.shape)
        self.x_prev = x
        return x


def prepare_training_inputs(sampled_exps, device='cuda'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0].float())
        actions.append(sampled_exp[1].float())
        rewards.append(sampled_exp[2].float())
        next_states.append(sampled_exp[3].float())
        dones.append(sampled_exp[4].float())

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones



class EMAMeter:

    def __init__(self,
                 alpha: float = 0.5):
        self.s = None
        self.alpha = alpha

    def update(self, y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1 - self.alpha) * self.s


