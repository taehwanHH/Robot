import torch
import gym
import glfw
import mujoco_py
from scipy.spatial.transform import Rotation as R
from param import Hyper_Param,Robot_Param,Comm_Param,TYPE

DEVICE = Hyper_Param['DEVICE']
penalty_coeff = Hyper_Param['penalty_coeff']
quat_relax = Hyper_Param['quat_relax']
comm_latency = Comm_Param['comm_latency']

Sensing_interval = Robot_Param['Sensing_interval']
End_flag = Robot_Param['End_flag']
Act_max = Robot_Param['Act_max']
Max_time= Robot_Param['Max_time']
State_normalizer = Robot_Param['State_normalizer']
h_threshold = Robot_Param['h_threshold']
rot_threshold = Robot_Param['rot_threshold']
target_Z = Robot_Param['target_Z']

# mujoco-py
xml_path = "multiRobot/sim_env.xml"
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
# # Ensure GLFW is initialized
# if not glfw.init():
#     raise Exception("GLFW initialization failed")
# # Set GLFW window hints BEFORE creating the viewer
# glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)  # Allow window resizing
# glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Make window invisible during setup
# glfw.window_hint(glfw.DECORATED, glfw.TRUE)  # Ensure the window has a border
# viewer = mujoco_py.MjViewer(sim)
# viewer.cam.azimuth = 180
# viewer.cam.elevation = -5
# viewer.cam.distance = 4
# viewer._run_speed = 512
# # Get the viewer's window handle
# window = viewer.window
#
# ## Ensure the window is not in fullscreen mode by setting monitor to None
#
# # Immediately reconfigure the window
# # Set the window to a specific size and position
# glfw.set_window_size(window, 800, 600)  # Set initial window size
#
# # Optionally, you can set the window title
# glfw.set_window_title(window, TYPE)


def sensor_read(num_robot, num_sensor):
    touch_vector = []
    for i in range(num_robot):
        for j in range(num_sensor):
            sensor_idx = sim.model.sensor_name2id(f"robot{i+1}_sensor{j + 1}")
            touch_vector.append(sim.data.sensordata[sensor_idx]/State_normalizer)

    return torch.tensor(touch_vector, dtype=torch.float32).to(DEVICE)


def actuator_write(num_robot, action):
    for i in range(num_robot):
        actuator_2_idx = sim.model.actuator_name2id(f"{i+1}_actuator_joint2")
        actuator_3_idx = sim.model.actuator_name2id(f"{i+1}_actuator_joint3")

        sim.data.ctrl[actuator_2_idx] = action[2*i]
        sim.data.ctrl[actuator_3_idx] = action[2*i+1]


def box_state():
    # Get box orientation quaternion
    box_idx = sim.model.body_name2id("box")
    object_quat = sim.data.body_xquat[box_idx]
    # Convert quaternion to Euler angles
    object_euler = torch.tensor(R.from_quat(object_quat).as_euler('xyz', degrees=True)/quat_relax, device=DEVICE,
                                dtype=torch.float32)

    # Get box position
    box_pos = sim.data.body_xpos[box_idx]
    box_z_pos = torch.tensor(box_pos[2], device=DEVICE, dtype=torch.float32)
    return object_euler[1:], box_z_pos


class RoboticEnv:
    def __init__(self, Max_time=Max_time):
        # Define the state space and action space
        self.num_sensor_output = 25  # pressure sensor output
        self.num_robot = 4
        self.num_joint = 2
        self.state_dim = self.num_sensor_output * self.num_robot + 1
        self.action_dim = self.num_robot * self.num_joint
        # self.state_space = gym.spaces.Box(low=0, high=1500, shape=(self.state_dim,))
        self.action_space = gym.spaces.Box(low=-Act_max, high=Act_max, shape=(self.action_dim,))

        self.ctrl_state = torch.zeros(self.num_robot*self.num_joint, device=DEVICE)
        # initialize
        self.Max_time = Max_time
        # self.target_Z = Robot_Param['target_Z']
        # self.penalty_coeff = Hyper_Param['penalty_coeff']
        self.time_step = 0
        self.stable_time = 0
        self.state = torch.tensor([0]*self.state_dim).view(1,-1)
        self.reward = torch.tensor([0]).view(1,-1)
        self.done = False
        self.flag = 0
        self.z_pos = 0
        self.final_pos = 0
        # self.sum_pos = 0
        self.task_success = 0
        self.past_quat = torch.zeros(2,device=DEVICE)



    def step(self, action):
        self.time_step += 1

        for _ in range(comm_latency):
            sim.step()
            # viewer.render()

        self.ctrl_state = torch.clamp(self.ctrl_state + action, min=torch.tensor(0,device=DEVICE), max=torch.tensor(0.45,device=DEVICE)).to(DEVICE)
        actuator_write(self.num_robot, self.ctrl_state)

        # past_euler,_ = box_state()
        for _ in range(Sensing_interval):
            sim.step()
            # viewer.render()

        object_euler, self.z_pos = box_state()

        # self.sum_pos += self.z_pos ##
        next_state = torch.cat([sensor_read(self.num_robot, self.num_sensor_output),self.z_pos.repeat(1).view(-1)],dim=0)

        # past_norm = torch.norm(past_euler)
        euler_norm = torch.norm(object_euler)

        # euler_var = past_norm - euler_norm

        # pos_penalty = torch.abs(self.z_pos - target_Z)*4
        quat_penalty = 1/ rot_threshold * (euler_norm ** 2)
        pos_penalty = 10*(torch.abs(self.z_pos - target_Z)) ** 2

        reward = -quat_penalty - pos_penalty
        if euler_norm < rot_threshold:
            self.final_pos = self.z_pos
            self.stable_time += 1
            if pos_penalty < 0.01:
                self.task_success += 1
                reward += 5  # Penalty term
                if pos_penalty < 0.001:
                    reward += 5

        if torch.sum(next_state) == 0:
            self.flag += 1
            reward -= 5  # Penalty term
        else:
            self.flag = 0

        reward = reward.to(DEVICE)

        if self.time_step > self.Max_time or self.flag == End_flag or self.z_pos < 0.1:
            self.done = True


        return next_state, reward, self.done, {}

    def reset(self):
        self.time_step = 0
        self.stable_time = 0
        self.task_success = 0
        # self.sum_pos = 0
        self.past_quat = torch.zeros(2,device=DEVICE)
        self.done = False

        sim.reset()

        init_pos = 0.45*torch.rand(1,device=DEVICE,dtype=torch.float32)
        self.ctrl_state = init_pos*torch.ones(self.num_robot*self.num_joint, device=DEVICE,dtype=torch.float32)
        actuator_write(self.num_robot, self.ctrl_state)
        for _ in range(Sensing_interval*50):
            sim.step()
            # viewer.render()

        object_euler, self.z_pos = box_state()

        state = torch.cat([sensor_read(self.num_robot, self.num_sensor_output),self.z_pos.repeat(1).view(-1)],dim=0)

        return state














