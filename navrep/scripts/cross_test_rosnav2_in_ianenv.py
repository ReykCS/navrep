import os, sys
from time import clock_getres
import numpy as np
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from stable_baselines.ppo2 import PPO2

# import rl_agent.common_custom_policies  # noqa


from navrep.tools.commonargs import parse_common_args
from navrep.envs.ianenv import IANEnv
from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes


# constants. If you change these, think hard about what which assumptions break.
_90 = 90  # guldenring downsampled scan size
_1080 = 1080  # navrep scan size
_540 = 1080 // 2
_8 = 8  # number of guldenrings waypoints

class Rosnav2CPolicy():
    def __init__(self, path=""):
        self.model = PPO2.load(os.path.join("/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav/rosnav_2021_12_15__21_10_49_rule_00_AGENT_21_tb3/newest"))  # noqa

    def act(self, obs):
        action, _state = self.model.predict(obs, deterministic=True)
        return action

class Rosnav2WrapperForIANEnv(IANEnv):
    def __init__(self, path="", **kargs):
        super().__init__(**kargs)
        pass

    def _convert_obs(self, ianenv_obs):
        # print(ianenv_obs)
        scan = ianenv_obs[0]
        robotstate = ianenv_obs[1]

        lidar_upsampling = 1080 // 360  
        downsampled_scan = scan.reshape((-1, lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)

        lidar = [np.min([3.5, i]) for i in downsampled_scan]

        self.last_scan = downsampled_scan

        rho, theta = self._get_goal_pose_in_robot_frame(robotstate[:2])

        return np.hstack([lidar, np.array([rho, theta])])

    def _convert_action(self, guldenring_action):
        vx, omega = guldenring_action
        ianenv_action = np.array([vx, 0., omega])
        # ianenv_action = np.array([0.5, 0., 0.5])
        return ianenv_action

    def _get_goal_pose_in_robot_frame(self, goal_pos):
        y_relative = goal_pos[1]
        x_relative = goal_pos[0]
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) + 4 * np.pi) % (2 * np.pi) - np.pi
        return rho, theta

    def step(self, guldenring_action):
        ianenv_action = self._convert_action(guldenring_action)
        ianenv_obs, reward, done, info = super(Rosnav2WrapperForIANEnv, self).step(ianenv_action)
        guldenring_obs = self._convert_obs(ianenv_obs)
        return guldenring_obs, reward, done, info

    def reset(self, *args, **kwargs):
        ianenv_obs = super(Rosnav2WrapperForIANEnv, self).reset(*args, **kwargs)
        guldenring_obs = self._convert_obs(ianenv_obs)
        return guldenring_obs

    def render(self, *args, **kwargs):
        lidar_angles_downsampled = np.linspace(0, 2 * np.pi, 360) \
            + self.iarlenv.rlenv.virtual_peppers[0].pos[2]
        kwargs["lidar_angles_override"] = lidar_angles_downsampled
        kwargs["lidar_scan_override"] = self.last_scan
        return self.iarlenv.render(*args, **kwargs)


if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 1000
    collect_trajectories = False

    env = Rosnav2WrapperForIANEnv(silent=True, collect_trajectories=collect_trajectories)
    policy = Rosnav2CPolicy()

    S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

    DIR = os.path.expanduser("~/navrep/eval/crosstest")
    if args.dry_run:
        DIR = "/tmp/navrep/eval/crosstest"
    make_dir_if_not_exists(DIR)

    if collect_trajectories:
        NAME = "rosnav2_in_ianenv_{}.pckl".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_pickle(PATH)
    else:
        NAME = "rosnav2_in_ianenv_{}.csv".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_csv(PATH)
    print("{} written.".format(PATH))
