from datetime import datetime
import os, json
from navrep.envs.rosnavtrainencodedenv import RosnavTrainEncodedEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from navrep.rosnav_models.agent_factory import AgentFactory
from navrep.rosnav_models.custom_sb3_policy import *
from navrep.rosnav_models.feature_extractors import init

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
    "max_episode_steps": 700,
    "agent_name": "AGENT_21",
    "timestep": 0.1,
    "rule": "rule_00",
    "n_eval_episodes": 100,
    "eval_freq": 40000
}

_L = None

roboter_models = ["tb3", "jackal", "ridgeback", "agv"]
model_base_path = "./robot/"

if __name__ == "__main__":
    args, _ = parse_common_args()

    for robot in roboter_models:

        print("Training model", robot)

        model_path = os.path.join(os.getcwd(), "robot", robot + ".model.yaml")

        init(model_path)

        params["roboter"] = robot
        params["interrupt"] = False

        DIR = os.path.join(os.getcwd(), "models/gym/rosnav")
        LOGDIR = os.path.join(os.getcwd(), "logs/gym/rosnav")
        EVALDIR = os.path.join(os.getcwd(), "logs/gym/rosnav/eval")
        
        if len(args.load) == 0:
            START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        
            MODELPATH = os.path.join(DIR, "rosnav_" + START_TIME + "_" + params["rule"] + "_" + params["agent_name"] + "_" + params["roboter"])

            if not os.path.exists(DIR):
                os.makedirs(DIR)
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            if not os.path.exists(EVALDIR):
                os.makedirs(EVALDIR)
        else:
            MODELPATH = os.path.join(DIR, args.load)

            if not os.path.exists(DIR):
                print("Path to model", MODELPATH, "does not exist")
                os.exit(1)

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
                    reward_fnc=params["rule"], 
                    max_steps_per_episode=params["max_episode_steps"]
                )
            ] * N_ENVS,
            start_method="fork"
        )

        eval_env = DummyVecEnv(
            [
                lambda: Monitor(
                    RosnavTrainEncodedEnv(
                        roboter_yaml_path=model_path, 
                        scenario="test", 
                        roboter=params["roboter"], 
                        reward_fnc=params["rule"], 
                        max_steps_per_episode=params["max_episode_steps"]
                    ), 
                    EVALDIR, 
                    info_keywords=("done_reason", "is_success")
                )
            ]
        )
        
        ### Stop training on reward threshhold callback
        stop_training_cb = StopTrainingOnRewardThreshold(
            treshhold_type="succ", threshold=0.9, verbose=1
        )

        ### Evaluation callback
        eval_cb = EvalCallback(
            eval_env=eval_env,
            train_env=env,
            n_eval_episodes=params["n_eval_episodes"],
            eval_freq=params["eval_freq"],
            best_model_save_path=MODELPATH,
            deterministic=True,
            callback_on_new_best=stop_training_cb
        )

        agent = AgentFactory.instantiate(
            params["agent_name"]
        )   

        if len(args.load) == 0:
            model = PPO(
                agent.type.value,
                env,
                policy_kwargs=agent.get_kwargs(),
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                learning_rate=params["learning_rate"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                batch_size=params["m_batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                verbose=1,
            )
        else:
            model = PPO.load(os.path.join(MODELPATH, "newest"), env)

        try:
            model.learn(total_timesteps=TRAIN_STEPS+1, callback=eval_cb)
        except KeyboardInterrupt:
            pass
        except:
            params["interrupt"] = True

        if len(args.load) == 0:
            with open(MODELPATH + "/hyperparams.json", "a") as fp:
                json.dump(params, fp)
        
        model.save(MODELPATH + "/newest")

        print("Model '{}' saved".format(MODELPATH))

        del model
        env.close()