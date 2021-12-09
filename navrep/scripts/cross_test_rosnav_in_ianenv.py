import os, time
import numpy as np
import pickle
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from stable_baselines3 import PPO

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
    def __init__(self, path="/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav/rosnav_2021_12_05__18_04_06_rule_01_AGENT_21"): 
        self.model = PPO.load(path + "/best_model")  # noqa
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
    def __init__(self, path="/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav/rosnav_2021_12_05__17_25_33_rule_01_AGENT_21", **kwargs):
        super().__init__(**kwargs)

        # vec_path = path + "/vec_normalize.pkl"
        # assert os.path.isfile(
        #     vec_path
        # ), f"VecNormalize file cannot be found at {vec_path}!"

        # with open(vec_path, "rb") as file_handler:
        #     vec_normalize = pickle.load(file_handler)

        # self._obs_norm_func = vec_normalize.normalize_obs

    def _convert_obs(self, ianenv_obs):
        """
            scan
                1080 lidar values 
            robotstate
                (goal x [m], goal y [m], vx [m/s], vy [m/s], vth [rad/s]) - all in robot frame
        """
        scan, robotstate = ianenv_obs

        rho, theta = get_goal_pose_in_robot_frame(
            robotstate 
        )
        print(robotstate, rho, theta)

        rosnav_obs = np.zeros(360 + 2)

        # Reshape the array of size 1080 to 360 and take the minimum of the merged values
        lidar_upsampling = _1080 // 360
        downsampled_scan = scan.reshape((-1, lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)
        
        # Turtlebot laser range is only 3.5 meters
        rosnav_obs[:360] = [np.min([3.5, i]) for i in downsampled_scan]
        
        self.last_rosnav_scan = rosnav_obs[:360]

        rosnav_obs[360] = rho
        rosnav_obs[361] = theta

        # Normalize values
        # rosnav_obs = self._obs_norm_func(rosnav_obs)

        return rosnav_obs

    def _convert_action(self, rosnav_action):
        vx, omega = rosnav_action
        ianenv_action = np.array([vx, 0., omega])
        return ianenv_action

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
        lidar_angles_downsampled = np.linspace(0, 2 * np.pi, 360) \
            + self.iarlenv.rlenv.virtual_peppers[0].pos[2]
        kwargs["lidar_angles_override"] = lidar_angles_downsampled
        kwargs["lidar_scan_override"] = self.last_rosnav_scan
        return self.iarlenv.render(*args, **kwargs)

if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 100
    collect_trajectories = False

    env = RosnavWrapperForIANEnv(silent=True, collect_trajectories=collect_trajectories)
    policy = RosnavCPolicy()

    S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

    DIR = os.path.expanduser("~/navrep/eval/crosstest")
    if args.dry_run:
        DIR = "/tmp/navrep/eval/crosstest"
    make_dir_if_not_exists(DIR)

    if collect_trajectories:
        NAME = "rosnav_in_ianenv_{}.pckl".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_pickle(PATH)
    else:
        NAME = "rosnav_in_ianenv_{}.csv".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_csv(PATH)
    print("{} written.".format(PATH))
