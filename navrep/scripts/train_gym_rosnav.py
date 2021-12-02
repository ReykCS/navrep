from datetime import datetime
import os, json
from navrep.envs.rosnavtrainencodedenv import RosnavTrainEncodedEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from navrep.rosnav_models.agent_factory import AgentFactory
from navrep.rosnav_models.custom_sb3_policy import *

from navrep.tools.commonargs import parse_common_args

AGENT_NAME = "AGENT_23"

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
    "rule": "rule_02"
}

def load_vec_normalize(params: dict, model_path, env: VecEnv, eval_env: VecEnv):
    if params["normalize"]:
        load_path = os.path.join(model_path, "vec_normalize.pkl")
        if os.path.isfile(load_path):
            env = VecNormalize.load(load_path=load_path, venv=env)
            eval_env = VecNormalize.load(load_path=load_path, venv=eval_env)
            print("Succesfully loaded VecNormalize object from pickle file..")
        else:
            env = VecNormalize(
                env, training=True, norm_obs=True, norm_reward=False, clip_reward=15
            )
            eval_env = VecNormalize(
                eval_env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
        return env, eval_env

if __name__ == "__main__":
    args, _ = parse_common_args()

    print(args)

    DIR = os.path.expanduser("/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/models/gym/rosnav")
    LOGDIR = os.path.expanduser("/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/logs/gym/rosnav")
    EVALDIR = os.path.expanduser("/home/reyk/Schreibtisch/Uni/VIS/catkin_navrep/src/navrep/logs/gym/rosnav/eval")
    
    if args.dry_run:
        DIR = "/tmp/navrep/models/gym/rosnav"
        LOGDIR = "/tmp/navrep/logs/gym/rosnav"
    
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    
    MODELPATH = os.path.join(DIR, "rosnav_" + START_TIME + "_ckpt_" + params["rule"])
    MODELPATH2 = os.path.join(DIR, "rosnav_latest_PPO_ckpt_" + params["rule"])

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(EVALDIR):
        os.makedirs(EVALDIR)

    MILLION = 1000000
    TRAIN_STEPS = args.n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 40 * MILLION

    N_ENVS = 6

    env = DummyVecEnv(
        [
            lambda: RosnavTrainEncodedEnv(scenario="train", reward_fnc=params["rule"])
        ] * N_ENVS
    )

    eval_env = DummyVecEnv(
        [
            lambda: Monitor(RosnavTrainEncodedEnv(scenario="test", reward_fnc=params["rule"]), EVALDIR, info_keywords=("done_reason", "is_success"))
        ]
    )

    # env, eval_env = load_vec_normalize(params, "MODELPATH", env, eval_env)
    
    ### Stop training on reward threshhold callback
    stop_training_cb = StopTrainingOnRewardThreshold(
        treshhold_type="succ", threshold=0.9, verbose=1
    )

    ### Evaluation callback
    eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=40,
        eval_freq=20000,
        best_model_save_path=MODELPATH,
        deterministic=True,
        callback_on_new_best=stop_training_cb
    )

    agent = AgentFactory.instantiate(
        AGENT_NAME
    )   

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

    try:
        model.learn(total_timesteps=TRAIN_STEPS+1, callback=eval_cb)
    except KeyboardInterrupt:
        pass

        
    with open(MODELPATH + "/hyperparams.json", "a") as fp:
        json.dump(params, fp)

    #model.save(MODELPATH)
    #model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model
    env.close()