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
            low=np.array(min, dtype=np.float32),
            high=np.array(max, dtype=np.float32),
            dtype=np.float32,
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
                    )
                    self._laser_max_range = plugin["range"]

            self.linear_range_x = robot_data["robot"]["continuous_actions"]["linear_range"]["x"]
            self.linear_range_y = robot_data["robot"]["continuous_actions"]["linear_range"]["y"]
            self.angular_range = robot_data["robot"]["continuous_actions"]["angular_range"]
            self.is_holonomic = robot_data["isHolonomic"]
    
    def _get_action_space(self, roboter):
        # return [-1, -3], [1, 3]

        if self.is_holonomic:
            return [self.linear_range_x[0], self.linear_range_y[0], self.angular_range[0]], [self.linear_range_x[1], self.linear_range_y[1], self.angular_range[1]]

        return [self.linear_range_x[0], self.angular_range[0]], [self.linear_range_x[1], self.angular_range[1]] 

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
        if self.encoder == "rto":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(720)
            downsampled[:360] = rotated_scan[180:540]
            downsampled[360:] = rotated_scan[540:900]

            f = interpolate.interp1d(np.arange(0, 720), downsampled)

            return f(np.linspace(0, 720 - 1, 684))
        if self.encoder == "rto_new_lidar":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(826)
            downsampled[:413] = rotated_scan[127:540]
            downsampled[413:] = rotated_scan[540:953]

            f = interpolate.interp1d(np.arange(0, 826), downsampled)

            return f(np.linspace(0, 826 - 1, 552))
        if self.encoder == "cob4":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            f = interpolate.interp1d(np.arange(0, 1080), rotated_scan)

            return f(np.linspace(0, 1080 - 1, 720))
        if self.encoder == "jackal" or self.encoder == "ridgeback":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(810)
            downsampled[:405] = rotated_scan[135:540]
            downsampled[405:] = rotated_scan[540:945]

            f = interpolate.interp1d(np.arange(0, 810), downsampled)

            return f(np.linspace(0, 810 - 1, 720))
        if self.encoder == "youbot":
            rotated_scan = np.zeros_like(obs)
            rotated_scan[:540] = obs[540:]
            rotated_scan[540:] = obs[:540]

            downsampled = np.zeros(540)
            downsampled[:270] = rotated_scan[270:540]
            downsampled[270:] = rotated_scan[540:810]

            f = interpolate.interp1d(np.arange(0, 540), downsampled)

            return f(np.linspace(0, 540 - 1, 512))
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

        new_lidar = np.minimum(self.laser_range, self._get_observation_from_scan(lidar)) # self._get_observation_from_scan(lidar)]
        rho, theta = self._get_goal_pose_in_robot_frame(state[:2])

        obs = np.concatenate([new_lidar, [rho, theta]]).reshape(self._laser_num_beams + 2, 1)
        return obs

    def _get_action(self, action):
        if self.is_holonomic:
            return np.array(action)

        return np.array([action[0], 0, action[1]]) # , 0])

    def _encode_action(self, action):
        new_action = self._get_action(action)
        
        return new_action

class RosnavNavRepEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, encoder="tb3", *args, **kwargs):
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