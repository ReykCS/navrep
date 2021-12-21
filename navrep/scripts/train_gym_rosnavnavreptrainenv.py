from datetime import datetime
import os

from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from navrep.envs.rosnavencoder import RosnavNavRepEnv
from navrep.tools.sb_eval_callback import NavrepEvalCallback
from navrep.tools.commonargs import parse_common_args

roboters = ["tb3", "jackal", "ridgeback", "agv"]

if __name__ == "__main__":
    args, _ = parse_common_args()

    for roboter in roboters:

        print("Start training model", roboter)

        DIR = os.path.expanduser("~/navrep/models/gym")
        LOGDIR = os.path.expanduser("~/navrep/logs/gym")
        if args.dry_run:
            DIR = "/tmp/navrep/models/gym"
            LOGDIR = "/tmp/navrep/logs/gym"
        START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        LOGNAME = "rosnavnavreptrainenv_" + START_TIME + "_PPO2" + "_ROSNAV_" + roboter
        LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
        MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
        MODELPATH2 = os.path.join(DIR, "rosnavnavreptrainenv_latest_PPO2_ckpt_" + roboter)
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)

        MILLION = 1000000
        TRAIN_STEPS = args.n
        if TRAIN_STEPS is None:
            TRAIN_STEPS = 60 * MILLION

        N_ENVS = args.envs

        if args.debug:
            env = DummyVecEnv([lambda: RosnavNavRepEnv(encoder=roboter, silent=True, scenario='train', collect_statistics=True)]*N_ENVS)
        else:
            env = SubprocVecEnv([lambda: RosnavNavRepEnv(encoder=roboter, silent=True, scenario='train', collect_statistics=True)]*N_ENVS,
                                start_method='spawn')
        eval_env = RosnavNavRepEnv(encoder=roboter, silent=True, scenario='train', collect_statistics=True)
        def test_env_fn():  # noqa
            return RosnavNavRepEnv(encoder=roboter, silent=True, scenario='test', collect_statistics=True)
        cb = NavrepEvalCallback(eval_env, test_env_fn=test_env_fn,
                                logpath=LOGPATH, savepath=MODELPATH, verbose=1)
        model = PPO2(MlpPolicy, env, verbose=0)
        model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
        obs = env.reset()

        model.save(MODELPATH)
        model.save(MODELPATH2)
        print("Model '{}' saved".format(MODELPATH))

        del model

    #     model = PPO2.load(MODELPATH)

    #     env = RosnavNavRepEnv(encoder=roboter, silent=True, scenario='train')
    #     obs = env.reset()
    #     for i in range(512):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, _, done, _ = env.step(action)
    #         if done:
    #             env.reset()
    # #         env.render()
