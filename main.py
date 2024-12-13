import torch
import os
from param import *
import importlib

from DDPG_module.train_utils import parse_args, prepare_training_inputs
from DDPG_module.train_utils import OrnsteinUhlenbeckProcess as OUProcess

args = parse_args()

Comm_Param['SNR'] = args.snr
Comm_Param['channel_type'] = args.channel_type
Comm_Param['comm_latency'] = args.latency
Comm_Param['_iscomplex'] = args._iscomplex

snr = Comm_Param['SNR']
channel_type = Comm_Param['channel_type']
comm_latency = Comm_Param['comm_latency']
_iscomplex = Comm_Param['_iscomplex']

print(f'SNR: {snr}')
print(f'channel: {channel_type}')
print(f'complex: {_iscomplex}')
print(f'communication latency: {comm_latency}ms')

simType = TYPE
SCENARIO = importlib.import_module(simType)
from DDPG_module.memory import ReplayMemory
from DDPG_module.target_update import soft_update

from scipy.io import savemat

from robotic_env import RoboticEnv
import time
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'


# Hyperparameters
DEVICE = Hyper_Param['DEVICE']
tau = Hyper_Param['tau']
lr_actor = Hyper_Param['lr_actor']
lr_critic = Hyper_Param['lr_critic']
batch_size = Hyper_Param['batch_size']
gamma = Hyper_Param['discount_factor']
memory_size = Hyper_Param['memory_size']
total_eps = Hyper_Param['num_episode']
sampling_only_until = Hyper_Param['train_start']
print_every = Hyper_Param['print_every']

# List storing the results
epi = []
lifting_time =[]
box_z_pos =[]
stable_lifting_time =[]
success_time = []
reward = []

sensor_val = []

# Create Environment

env = RoboticEnv()

s_dim = env.state_dim
a_dim = env.action_dim

# initialize target network same as the main network.
actor, actor_target = SCENARIO.Actor().to(DEVICE), SCENARIO.Actor().to(DEVICE)
critic, critic_target = SCENARIO.Critic().to(DEVICE), SCENARIO.Critic().to(DEVICE)

agent = SCENARIO.DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target,epsilon=Hyper_Param['epsilon'],
             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma).to(DEVICE)

memory = ReplayMemory(memory_size)


# Episode start
for n_epi in range(total_eps):
    ou_noise = OUProcess(mu=torch.zeros(a_dim))
    s = env.reset()
    epi.append(n_epi)
    episode_return = 0
    while True:
        if n_epi < 1000:
            sensor_val.append(s)
        a = agent.get_action(s, agent.epsilon*ou_noise()).view(-1)
        ns, r, done, info = env.step(a)

        episode_return += r.item()
        experience = (s.view(-1, s_dim),
                      a.view(-1, a_dim),
                      r.view(-1, 1),
                      ns.view(-1, s_dim),
                      torch.tensor(done, device=DEVICE).view(-1, 1))
        memory.push(experience)
        s = ns
        if done:
            break

    avg_r = episode_return / env.time_step
    final_pos = env.final_pos.item()
    # lifting_time.append(env.time_step - 1)
    box_z_pos.append(final_pos)
    stable_lifting_time.append(env.stable_time)
    success_time.append(env.task_success)
    reward.append(avg_r)

    if len(memory) >= sampling_only_until:
        # train agent
        agent.epsilon = max(agent.epsilon * Hyper_Param['epsilon_decay'], Hyper_Param['epsilon_min'])

        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

        soft_update(agent.actor, agent.actor_target, tau)
        soft_update(agent.critic, agent.critic_target, tau)

    if n_epi % print_every == 0:
        msg = (n_epi, env.stable_time,final_pos, avg_r)
        print("Episode : {:4.0f} | lift time : {:3.0f} | box pos : {:3.4f} | avg reward : {:3.4f}:".format(*msg))



# Base directory path creation
base_directory = os.path.join(Hyper_Param['today'])

# Subdirectory creation
sub_directory = os.path.join(base_directory, Comm_Param['channel_type'])
if not os.path.exists(sub_directory):
    os.makedirs(sub_directory)

snr_title = f"{Comm_Param['comm_latency']}ms"
sub_directory = os.path.join(sub_directory, snr_title)
if not os.path.exists(sub_directory):
    os.makedirs(sub_directory)
    index = 1

else:
    existing_dirs = [d for d in os.listdir(sub_directory) if os.path.isdir(os.path.join(sub_directory, d))]
    indices = [int(d) for d in existing_dirs if d.isdigit()]
    index = max(indices) + 1 if indices else 1

sub_directory = os.path.join(sub_directory,str(index))
os.makedirs(sub_directory)

# Store Hyperparameters in txt file
with open(os.path.join(sub_directory, 'Hyper_Param.txt'), 'w') as file:
    for key, value in Hyper_Param.items():
        file.write(f"{key}: {value}\n")

# Store score data (matlab data file)
savemat(os.path.join(sub_directory, 'data.mat'),{'stable_lifting_time': stable_lifting_time, 'box_z_pos': box_z_pos, 'success_time': success_time, 'average_reward' : reward})
