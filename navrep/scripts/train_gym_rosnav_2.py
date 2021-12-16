from datetime import datetime
import os, json
from navrep.envs.rosnavtrainencodedenv import RosnavTrainEncodedEnv

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from navrep.rosnav_models.agent_factory import AgentFactory
from navrep.rosnav_models.custom_sb3_policy import *
from navrep.rosnav_models.feature_extractors import init
from navrep.tools.sb_eval_callback import NavrepEvalCallback

from navrep.tools.commonargs import parse_common_args

params = {
    "gamma": 0.99,
    "n_steps": 1200,
    "ent_coef": 0.005,
    "learning_rate": 0.0003,
    "vf_coef": 0.22,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "m_batch_size": 15,
    "n_epochs": 3,
    "clip_range": 0.22,
    "normalize": True,
    "max_episode_steps": 1000,
    "agent_name": "AGENT_21",
    "timestep": 0.1,
    "rule": "rule_00",
    "n_eval_episodes": 100,
    "eval_freq": 50000
}

roboter_models = ["tb3", "jackal", "ridgeback", "agv"]
model_base_path = "./robot/"

if __name__ == "__main__":
    args, _ = parse_common_args()

    for robot in roboter_models:

        print("Training model", robot)

        model_path = os.path.join(os.getcwd(), "robot", robot + ".model.yaml")

        params["roboter"] = robot
        params["interrupt"] = False

        DIR = os.path.join(os.getcwd(), "models/gym/rosnav")
        EVALDIR = os.path.join(os.getcwd(), "logs/gym/rosnav/eval")
        
        START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    
        MODELPATH = os.path.join(DIR, "rosnav_" + START_TIME + "_" + params["rule"] + "_" + params["agent_name"] + "_" + params["roboter"])

        if not os.path.exists(DIR):
            os.makedirs(DIR)
        if not os.path.exists(EVALDIR):
            os.makedirs(EVALDIR)
        if not os.path.exists(MODELPATH):
            os.makedirs(MODELPATH)


        MILLION = 1000000
        TRAIN_STEPS = args.n
        if TRAIN_STEPS is None:
            TRAIN_STEPS = 40 * MILLION

        N_ENVS = args.envs

        env = SubprocVecEnv(
            [
                lambda: RosnavTrainEncodedEnv(
                    roboter_yaml_path=model_path, 
                    scenario="train", 
                    roboter=params["roboter"], 
                ) for _ in range(N_ENVS)
            ]
        )

        eval_env = RosnavTrainEncodedEnv(
                    roboter_yaml_path=model_path, 
                    scenario="train", 
                    roboter=params["roboter"], 
                )

        cb = NavrepEvalCallback(
            eval_env, 
            test_env_fn=lambda: RosnavTrainEncodedEnv(
                roboter_yaml_path=model_path, 
                scenario="test", 
                roboter=params["roboter"], 
            ),
            n_eval_episodes=50,
            logpath=EVALDIR + "/model.csv", savepath=MODELPATH, verbose=1
        )

        model = PPO2(MlpPolicy, env, verbose=0)

        try:
            model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
        except KeyboardInterrupt:
            pass

        if len(args.load) == 0:
            with open(MODELPATH + "/hyperparams.json", "a") as fp:
                json.dump(params, fp)
        
        model.save(MODELPATH + "/newest")

        print("Model '{}' saved".format(MODELPATH))

        del model
        env.close()