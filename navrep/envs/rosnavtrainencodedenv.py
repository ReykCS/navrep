from gym import spaces
import numpy as np
import yaml

from navrep.envs.navreptrainenv import NavRepTrainEnv

from navrep.rosnav_models.utils.reward import RewardCalculator

class RosnavTrainEncodedEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, reward_fnc = "rule_02",
                 scenario='test', silent=False, adaptive=True,
                 gpu=False, max_steps_per_episode=100):

        super(RosnavTrainEncodedEnv, self).__init__(scenario=scenario, silent=silent, adaptive=adaptive,
                                                    legacy_mode=False)
        self.setup_by_configuration()

        self.action_space = spaces.Box(
            low=np.array([0, -2.7]),
            high=np.array([0.3, 2.7]),
            dtype=np.float,
        )

        self.observation_space = RosnavTrainEncodedEnv._stack_spaces(
            (
                spaces.Box(
                    low=0,
                    high=self._laser_max_range,
                    shape=(self._laser_num_beams,),
                    dtype=np.float32,
                ),
                spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
                spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
            )
        )

        self.reward_calculator = RewardCalculator(
            robot_radius=self._robot_radius,
            safe_dist=1.6 * self._robot_radius,
            goal_radius=0.1,
            rule=reward_fnc,
            extended_eval=False,
        )

        self._steps_curr_episode = 0
        self._max_steps_per_episode = max_steps_per_episode

        self.last_observation = None

    def step(self, action):
        self._steps_curr_episode += 1

        action = np.array([action[0], 0, action[1]])
        obs, reward, done, info = super(RosnavTrainEncodedEnv, self).step(action)

        lidar, rho, theta = self._encode_obs(obs)

        reward, reward_info = self.reward_calculator.get_reward(
            np.array(lidar),
            (rho, theta),
            action=np.array([action[0], action[2]]),
            global_plan=None,
            robot_pose=None
        )
        done = reward_info["is_done"]

        observation = np.hstack([lidar, np.array([rho, theta])])

        info = {}

        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if self._steps_curr_episode > self._max_steps_per_episode:
            done = True
            info["done_reason"] = 0
            info["is_success"] = 0

        return observation, reward, done, info

    def reset(self, *args, **kwargs):
        self.reward_calculator.reset()
        self._steps_curr_episode = 0

        obs = super(RosnavTrainEncodedEnv, self).reset(*args, **kwargs)

        observation, rho, theta = self._encode_obs(obs)

        return np.hstack([observation, np.array([rho, theta])])

    def _encode_obs(self, obs):
        scan, robotstate = obs

        # Downsample observation
        lidar_upsampling = 1080 // 360
        downsampled_scan = scan.reshape((-1, lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)
        lidar = [np.min([3.5, i]) for i in downsampled_scan]
        self.last_rosnav_scan = lidar

        rho, theta = self._get_goal_pose_in_robot_frame(robotstate[:2])

        return lidar, rho, theta

    def close(self):
        super(RosnavTrainEncodedEnv, self).close()

    def render(self, mode="human", close=False, save_to_file=False,
               robocentric=False, render_decoded_scan=True):
        super(RosnavTrainEncodedEnv, self).render(
            mode=mode, close=close, lidar_scan_override=self.last_rosnav_scan, save_to_file=save_to_file,
            robocentric=robocentric)

    def _get_goal_pose_in_robot_frame(self, goal_pos):
        y_relative = goal_pos[1]
        x_relative = goal_pos[0]
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) + 4 * np.pi) % (2 * np.pi) - np.pi
        return rho, theta

    def setup_by_configuration(
        self, robot_yaml_path="/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/robot/tb3.model.yaml"
    ):
        """get the configuration from the yaml file, including robot radius, discrete action space and continuous action space.

        Args:
            robot_yaml_path (str): [description]
        """
        with open(robot_yaml_path, "r") as fd:
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

                    self._laser_num_beams = int(
                        round(
                            (laser_angle_max - laser_angle_min)
                            / laser_angle_increment
                        )
                        + 1
                    )
                    self._laser_max_range = plugin["range"]
    
    @staticmethod
    def _stack_spaces(ss):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())