import numpy as np
from gym import spaces
import yaml
from scipy import interpolate
from navrep.envs.e2eenv import FlatLidarAndStateEncoder

from navrep.tools.rings import generate_rings
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.ianenv import IANEnv

class RosnavEncoder(object):
    """ Generic class to encode the observations of an environment into a single 1d vector """
    def __init__(self, encoder):
        self.encoder = encoder

        self.setup_by_configuration()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._laser_num_beams + 2,1), dtype=np.float32)

        min, max = self._get_action_space(self.encoder)

        self.action_space = spaces.Box(
            low=np.array(min),
            high=np.array(max),
            dtype=np.float,
        )

    def setup_by_configuration(
        self
    ):
        with open("robot/" + self.encoder + ".model.yaml", "r") as fd:
            robot_data = yaml.safe_load(fd)
            # get robot radius
            for body in robot_data["bodies"]:
                if body["name"] == "base_footprint":
                    for footprint in body["footprints"]:
                        if footprint["radius"]:
                            self._robot_radius = footprint["radius"] * 1.05
            # get laser related information
            for plugin in robot_data["plugins"]:
                if plugin["type"] == "Laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    self.laser_range = plugin["range"]

                    self._laser_num_beams = int(
                        round(
                            (laser_angle_max - laser_angle_min)
                            / laser_angle_increment
                        )
                        + 1
                    )
                    self._laser_max_range = plugin["range"]

            self.linear_range = robot_data["robot"]["continuous_actions"]["linear_range"]
            self.angular_range = robot_data["robot"]["continuous_actions"]["angular_range"]

    
    def _get_action_space(self, roboter):
        if roboter == "ridgeback":
            return [self.linear_range[0], 0, self.angular_range[0]], [self.linear_range[1], 0.5, self.angular_range[1]]

        return [self.linear_range[0], self.angular_range[0]], [self.linear_range[1], self.angular_range[1]] 

    def reset(self):
        pass

    def close(self):
        pass

    def _get_observation_from_scan(self, obs):
        if self.encoder == "tb3":
            lidar_upsampling = 1080 // 360
            downsampled_scan = obs.reshape((-1, lidar_upsampling))
            downsampled_scan = np.min(downsampled_scan, axis=1)
            return downsampled_scan
        if self.encoder == "jackal" or self.encoder == "ridgeback":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(810)
            downsampled[:405] = rotated_scan[135:540]
            downsampled[405:] = rotated_scan[540:945]

            f = interpolate.interp1d(np.arange(0, 810), downsampled)
            upsampled = f(np.linspace(0, 810 - 1, 944))

            lidar = upsampled.reshape((-1, 2))
            lidar = np.min(lidar, axis=1)

            return lidar
        if self.encoder == "agv":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(540)
            downsampled[:270] = rotated_scan[270:540]
            downsampled[270:] = rotated_scan[540:810]

            f = interpolate.interp1d(np.arange(0, 540), downsampled)
            return f(np.linspace(0.0, 540 - 1, 720))

    def _get_goal_pose_in_robot_frame(self, goal_pos):
        x_relative, y_relative = goal_pos
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) + 4 * np.pi) % (2 * np.pi) - np.pi
        return rho, theta

    def _encode_obs(self, obs):
        lidar, state = obs

        new_lidar = [np.min([self.laser_range, i]) for i in self._get_observation_from_scan(lidar)]
        rho, theta = self._get_goal_pose_in_robot_frame(state[:2])

        obs = np.concatenate([new_lidar, [rho, theta]]).reshape(self._laser_num_beams + 2, 1)
        return obs

    def _get_action(self, action):
        if self.encoder == "ridgeback":
            return np.array(action)

        return np.array([action[0], 0, action[1]])

    def _encode_action(self, action):
        new_action = self._get_action(action)
        
        return new_action

class RosnavNavRepEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, encoder="tb3", *args, **kwargs):
        if encoder == "e2e":
            self.encoder = FlatLidarAndStateEncoder()
        else:
            self.encoder = RosnavEncoder(encoder=encoder)
        super(RosnavNavRepEnv, self).__init__(*args, **kwargs)
        self.action_space = self.encoder.action_space
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = self.encoder._get_action(action) # np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(RosnavNavRepEnv, self).step(action)
        h = self.encoder._encode_obs(obs)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(RosnavNavRepEnv, self).reset()
        h = self.encoder._encode_obs(obs)
        return h