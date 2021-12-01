from navrep.scripts.cross_test_rosnav_in_ianenv import RosnavWrapperForIANEnv, RosnavCPolicy
from navrep.scripts.cross_test_guldenring_in_ianenv import GuldenringCPolicy, GuldenringWrapperForIANEnv
from navrep.scripts.test_navrep import NavRepCPolicy
from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv

from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from navrep.tools.commonargs import parse_common_args

import os, sys, navrep

import navrep.scripts.custom_policy as cp

# from arena_local_planner_drl import *

sys.modules["arena_navigation"] = navrep
sys.modules["arena_navigation.arena_local_planner"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl"] = navrep
sys.modules["arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.scripts.custom_policy"] = cp

# Evaluate rosnav

RUNS = 100

class ModelPath:
    def __init__(self, base_path, agents, create_path, name, classes):
        self.base_path = base_path
        self.agents = agents
        self.create_path = create_path
        self.name = name,
        self.classes = classes

class NavrepModelPath(ModelPath):
    def __init__(self, base_path, agents, create_path, name, backend=[], encoding=[]):
        super().__init__(base_path, agents, create_path, name, None)

        self.backend = backend
        self.encoding = encoding

RosNavModelPath = ModelPath(
    "/home/reyk/Schreibtisch/Uni/VIS/arena-rosnav-2d/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/",
    ["rule_03", "AGENT_19_2021_04_12__13_17", "rule_03", "rule_00", "rule_01", "rule_02", "rule_04"],
    lambda base, agent: base + agent,
    "rosnav",
    (RosnavWrapperForIANEnv, RosnavCPolicy)
)

GuldenringModelPath = ModelPath(
    "/home/reyk/Schreibtisch/Uni/VIS/catkin_gring/src/gring/example_agents/",
    ["ppo2_1_raw_data_cont_0"],
    lambda base, agent: base + agent + "/" + agent + ".pkl",
    "guldenring",
    (GuldenringWrapperForIANEnv, GuldenringCPolicy),
)

# NavrepModelPath = ModelPath(
#     "/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/"
#     ["navreptrainencodedenv_2020_10_22__16_18_01_PPO_GPT_M_ONLY_V64M64_ckpt"],
#     lambda base, agent: base + agent,
#     "navrep",
#     backend=["GPT", "GPT1D", "VAE1DLSTM", "VAELSTM", "VAELSTM", "VAE1D_LSTM"],
#     encoding=["V_ONLY", "M_ONLY", "VM"]
# )

if __name__ == "__main__":
    args, _ = parse_common_args()

    collect_trajectories = False

    for models in [GuldenringModelPath, RosNavModelPath]:
        for m in models.agents:
            path_to_model = models.create_path(models.base_path, m)

            env = models.classes[0](path=path_to_model, silent=True, collect_trajectories=collect_trajectories)
            policy = models.classes[1](path=path_to_model)

            print(path_to_model)

            S = run_test_episodes(env, policy, render=args.render, num_episodes=RUNS)

            DIR = os.path.expanduser("~/navrep/eval/crosstest")
            if args.dry_run:
                DIR = "/tmp/navrep/eval/crosstest"
            make_dir_if_not_exists(DIR)

            if collect_trajectories:
                NAME = f"{models.name}_in_ianenv_{len(S)}.pckl"
                PATH = os.path.join(DIR, NAME)
                S.to_pickle(PATH)
            else:
                NAME = f"{models.name}_in_ianenv_{len(S)}.csv"
                PATH = os.path.join(DIR, NAME)
                S.to_csv(PATH)
            print("{} written.".format(PATH))

            print(models.name, m, "done")

    # env = RosnavWrapperForIANEnv(silent=True, collect_trajectories=collect_trajectories)
    # policy = RosnavCPolicy()


# S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

# DIR = os.path.expanduser("~/navrep/eval/crosstest")
# if args.dry_run:
#     DIR = "/tmp/navrep/eval/crosstest"
# make_dir_if_not_exists(DIR)

# if collect_trajectories:
#     NAME = "rosnav_in_ianenv_{}.pckl".format(len(S))
#     PATH = os.path.join(DIR, NAME)
#     S.to_pickle(PATH)
# else:
#     NAME = "rosnav_in_ianenv_{}.csv".format(len(S))
#     PATH = os.path.join(DIR, NAME)
#     S.to_csv(PATH)
# print("{} written.".format(PATH))