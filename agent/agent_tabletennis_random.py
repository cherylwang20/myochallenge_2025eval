import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

from utils import RemoteConnection
from stable_baselines3 import PPO

"""
Define your custom observation keys here
"""
custom_obs_keys = [ 
    'pelvis_pos', 
    'body_qpos', 
    'body_qvel', 
    'ball_pos', 
    'ball_vel', 
    'paddle_pos', 
    'paddle_vel', 
    'paddle_ori', 
    'reach_err', 
    'touching_info', 
    'act',
]

class ActionExpander:
    def __init__(self, full_action_dim=275):
        self.syn_action_shape = 89
        self.full_action_dim = full_action_dim
        self.action_mapping = {
            0: list(range(0, 11)),    1: list(range(11, 22)),   2: [22],        3: [23],
            4: list(range(24, 28)),   5: list(range(28, 32)),   6: list(range(32, 40)), 7: list(range(40, 48)),
            8: list(range(48, 69)),   9: list(range(69, 90)),   10: list(range(90, 95)), 11: list(range(95, 100)),
            12: list(range(100, 107)),13: list(range(107, 114)),14: list(range(114, 119)),15: list(range(119, 124)),
            16: list(range(124, 130)),17: list(range(130, 136)),18: list(range(136, 161)),19: list(range(161, 186)),
            20: list(range(186, 192)),21: list(range(192, 198)),22: list(range(198, 204)),23: list(range(204, 210))
        }

        # Add direct mapping: 24–88 → indices 210–274
        for i in range(24, 89):
            self.action_mapping[i] = [i + 169]  # 24 maps to 210, ..., 88 maps to 274

    def expand(self, reduced_action):
        assert len(reduced_action) == self.syn_action_shape
        full_action = np.zeros(self.full_action_dim, dtype=np.float32)
        for i, indices in self.action_mapping.items():
            full_action[indices] = reduced_action[i]
        return full_action


def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class Policy:

    def __init__(self, env):
        self.action_space = env.action_space

    def __call__(self, env):
        return self.action_space.sample()

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

#policy = Policy(rc)
path = '/'.join(os.path.realpath('baseline_mc25_tabletennis.zip').split('/')[:-1])

root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
print(root_path)
model = PPO.load(os.path.join(root_path, 'baseline_mc25_tabletennis'))

print('Loading Table Tennis Policy')

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

expander = ActionExpander()

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"PINGPONG: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        ################################################
        obs = get_custom_observation(rc, custom_obs_keys)
        action, _ = model.predict(obs)
        full_action = expander.expand(action)
        ################################################

        base = rc.act_on_environment(full_action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
