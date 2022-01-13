import numpy as np
from gym import spaces

from navrep.tools.rings import generate_rings
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.ianenv import IANEnv

_L = 1080  # lidar size
_RS = 5  # robotstate size
_64 = 64  # ring size

class FlatLidarAndStateEncoder(object):
    """ Generic class to encode the observations of an environment into a single 1d vector """
    def __init__(self):
        self._N = _L + _RS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(362,1), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, -2.84]), high=np.array([0.22, 2.84]), shape=(2,), dtype=np.float32)

    def reset(self):
        pass

    def close(self):
        pass

    def _get_action(self, action):
        return np.array([action[0], action[1], 0.])

    def _get_goal_pose_in_robot_frame(self, goal_pos):
        x_relative, y_relative = goal_pos
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) + 4 * np.pi) % (2 * np.pi) - np.pi
        return rho, theta

    def _encode_obs(self, obs):
        lidar, state = obs

        lidar_upsampling = 1080 // 360
        downsampled_scan = lidar.reshape((-1, lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)

        new_lidar = [np.min([3.5, i]) for i in downsampled_scan]
        rho, theta = self._get_goal_pose_in_robot_frame(state[:2])

        obs = np.concatenate([new_lidar, [rho, theta]]).reshape(362, 1)
        return obs

class RosnavE2E1DNavRepEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, *args, **kwargs):
        self.encoder = FlatLidarAndStateEncoder()
        super(RosnavE2E1DNavRepEnv, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=np.array([0, -2.84]), high=np.array([0.22, 2.84]), shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], 0., action[1]])  # no rotation
        obs, reward, done, info = super(RosnavE2E1DNavRepEnv, self).step(action)
        h = self.encoder._encode_obs(obs)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(RosnavE2E1DNavRepEnv, self).reset()
        h = self.encoder._encode_obs(obs)
        return h

# class E2E1DIANEnv(IANEnv):
#     """ takes a (2) action as input
#     outputs encoded obs (1085) """
#     def __init__(self, *args, **kwargs):
#         self.encoder = FlatLidarAndStateEncoder()
#         super(E2E1DIANEnv, self).__init__(*args, **kwargs)
#         self.action_space = spaces.Box(low=[0, -2.84], high=[0.22, 2.84], shape=(2,), dtype=np.float32)
#         self.observation_space = self.encoder.observation_space

#     def step(self, action):
#         action = np.array([action[0], 0, action[1]])  # no rotation
#         obs, reward, done, info = super(E2E1DIANEnv, self).step(action)
#         h = self.encoder._encode_obs(obs, action)
#         return h, reward, done, info

#     def reset(self):
#         self.encoder.reset()
#         obs = super(E2E1DIANEnv, self).reset()
#         h = self.encoder._encode_obs(obs, np.array([0,0,0]))
#         return h

# class E2EIANEnv(IANEnv):
#     """ takes a (2) action as input
#     outputs encoded obs (1085) """
#     def __init__(self, *args, **kwargs):
#         self.encoder = RingsLidarAndStateEncoder()
#         super(E2EIANEnv, self).__init__(*args, **kwargs)
#         self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
#         self.observation_space = self.encoder.observation_space

#     def step(self, action):
#         action = np.array([action[0], action[1], 0.])  # no rotation
#         obs, reward, done, info = super(E2EIANEnv, self).step(action)
#         h = self.encoder._encode_obs(obs, action)
#         return h, reward, done, info

#     def reset(self):
#         self.encoder.reset()
#         obs = super(E2EIANEnv, self).reset()
#         h = self.encoder._encode_obs(obs, np.array([0,0,0]))
#         return h
