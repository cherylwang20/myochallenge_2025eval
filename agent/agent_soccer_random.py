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

"""
Define your custom observation keys here
"""
custom_obs_keys = [
    'internal_qpos',
    'internal_qvel',
    'grf',
    'torso_angle',
    'ball_pos',
    'model_root_pos',
    'model_root_vel',
    'muscle_length',
    'muscle_velocity',
    'muscle_force',
]

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

time.sleep(60) # DO NOT REMOVE. Required for EvalAI processing

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

policy = Policy(rc)

shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"Soccer: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset(osl_dict)
    obs = get_custom_observation(rc, custom_obs_keys)

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        ################################################
        ## Replace with your trained policy.
        action = rc.action_space.sample()
        ################################################

        base = rc.act_on_environment(action)

        # Get the observations you used here
        obs = get_custom_observation(rc, custom_obs_keys)

        #obs = base["feedback"][0]


        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
