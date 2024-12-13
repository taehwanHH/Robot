import torch
from datetime import datetime


TYPE = 'Agent.raw_DDPG'


Hyper_Param = {
    'today': datetime.now().strftime('%Y-%m-%d'),
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tau': 0.001,
    'discount_factor': 0.95,
    'theta': 0.05,
    'dt': 0.01,
    'sigma': 0.01,
    'epsilon': 0.01,
    'epsilon_decay': 0.999,
    'epsilon_min': 0.0001,
    'lr_actor': 0.005,
    'lr_critic': 0.001,
    'batch_size': 4096,
    'train_start': 5000,
    'num_episode': 5000,
    'memory_size': 10**4,
    'print_every': 1,
    'num_neurons': [64,32,16],
    'critic_num_neurons': [128,64,32,16],
    'penalty_coeff' : 10,
    'quat_relax' : 90
}

Robot_Param = {
    'Sensing_interval' : 100,
    'End_flag' : 2,
    'Sense_max': 1.2,
    'Act_max': 0.001,
    'Max_time': 400,
    'State_normalizer' : 500,
    'h_threshold' : 2,
    'rot_threshold' : 0.015,
    'target_Z' : 1.8
}

Comm_Param ={
    'channel_type': 'awgn',
    'SNR': 5,
    'comm_latency': 0,  # ms
    'qam_order': 256,
    '_iscomplex': False
}

