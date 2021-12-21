import os, time
import numpy as np
import pickle, yaml
from pyniel.python_tools.path_tools import make_dir_if_not_exists
# from stable_baselines3 import PPO
from stable_baselines import PPO2
from scipy import interpolate

from datetime import datetime, timedelta

from navrep.tools.commonargs import parse_common_args
from navrep.envs.ianenv import IANEnv
from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes

import os, sys, navrep

import navrep.scripts.custom_policy as cp

# from arena_local_planner_drl import *

sys.modules["arena_navigation"] = navrep
sys.modules["arena_navigation.arena_local_planner"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.scripts.custom_policy"] = cp


# constants. If you change these, think hard about what which assumptions break.
_1080 = 1080  # navrep scan size

class RosnavCPolicy():
    def __init__(self, path="/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav/rosnav_2021_12_13__13_54_51_rule_00_AGENT_21_tb3"): 
        self.model = PPO2.load("/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/newModels/" + path)  # noqa
        print(self.model.observation_space.shape)

    def act(self, obs):
        action, _state = self.model.predict(obs, deterministic=True)
        return action

def get_goal_pose_in_robot_frame(goal):
    y_relative = goal[1]
    x_relative = goal[0]
    rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
    theta = (
        np.arctan2(y_relative, x_relative + 4 * np.pi)
    ) % (2 * np.pi) - np.pi

    return rho, theta

class RosnavWrapperForIANEnv(IANEnv):
    def __init__(self, encoder="tb3", path="/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav/rosnav_2021_12_05__17_25_33_rule_01_AGENT_21", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

        self.setup_by_configuration()

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

    def _convert_obs(self, ianenv_obs):
        """
            scan
                1080 lidar values 
            robotstate
                (goal x [m], goal y [m], vx [m/s], vy [m/s], vth [rad/s]) - all in robot frame
        """
        return self._encode_obs(ianenv_obs)

    def _get_action(self, action):
        if self.encoder == "ridgeback":
            return np.array(action)

        return np.array([action[0], 0, action[1]])

    def _convert_action(self, rosnav_action):
        return self._get_action(rosnav_action)

    def step(self, rosnav_action):
        ianenv_action = self._convert_action(rosnav_action)
        ianenv_obs, reward, done, info = super(RosnavWrapperForIANEnv, self).step(ianenv_action)
        rosnav_obs = self._convert_obs(ianenv_obs)
        
        return rosnav_obs, reward, done, info

    def reset(self, *args, **kwargs):
        ianenv_obs = super(RosnavWrapperForIANEnv, self).reset(*args, **kwargs)
        rosnav_obs = self._convert_obs(ianenv_obs)
        return rosnav_obs

    def render(self, *args, **kwargs):
        # lidar_angles_downsampled = np.linspace(0, 2 * np.pi, 360) \
        #     + self.iarlenv.rlenv.virtual_peppers[0].pos[2]
        # kwargs["lidar_angles_override"] = lidar_angles_downsampled
        # kwargs["lidar_scan_override"] = self.last_rosnav_scan
        return self.iarlenv.render(*args, **kwargs)


models_to_test = [
    ["rosnavnavreptrainenv_2021_12_17__12_53_36_PPO_ROSNAV_tb3_ckpt", "tb3"],
    ["rosnavnavreptrainenv_2021_12_17__12_53_36_PPO_ROSNAV_tb3_ckpt_best_model", "tb3"],
    ["rosnavnavreptrainenv_2021_12_18__17_32_35_PPO_ROSNAV_jackal_ckpt", "jackal"],
    ["rosnavnavreptrainenv_2021_12_18__17_32_35_PPO_ROSNAV_jackal_ckpt_best_model", "jackal"]
]

if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 100
    collect_trajectories = False

    for model in models_to_test:

        path, encoder = model

        print("Testing model", path)

        env = RosnavWrapperForIANEnv(encoder=encoder, silent=True, collect_trajectories=collect_trajectories)
        policy = RosnavCPolicy(path=path)

        S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

        DIR = os.path.expanduser("~/navrep/eval/crosstest")
        if args.dry_run:
            DIR = "/tmp/navrep/eval/crosstest"
        make_dir_if_not_exists(DIR)

        NAME = "{}_{}.pckl".format(path, len(S))
        PATH = os.path.join(DIR, NAME)

        if collect_trajectories:
            S.to_pickle(PATH)
        else:
            S.to_csv(PATH)
        print("{} written.".format(PATH))
