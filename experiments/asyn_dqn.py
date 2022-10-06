from itertools import chain
import numpy as np
from tqdm import tqdm

import wandb
from gym.wrappers import TimeLimit, RecordEpisodeStatistics
from ray.rllib.models.catalog import ModelCatalog
from ray.tune import register_env

from copy import deepcopy
from pirl._networks import ShallowUQnet
from pirl.envs.reticolo_env import ReticoloEnv
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from utils import load_pretrained_apex

NUM_GPUS = 1
ENV_ID = "deflector-v0"
BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 16


wandb.init(project='PIRL')

from ray.rllib.algorithms.dqn import DQNConfig

config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=ENV_ID,
    env_config={
        'n_cells': 256,
        'wavelength': 1100,
        'desired_angle': 70
    }
).training(
    model={
        'custom_model': 'model'
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=BATCH_SIZE
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=128,
    num_rollout_workers=NUM_ROLLOUT_WORKERS,
    num_envs_per_worker=4,
    # rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
).exploration(
    explore=True,
    exploration_config={
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        # 'epsilon_timesteps': 100000,
    }
).evaluation(
    evaluation_duration_unit='episodes',
    evaluation_duration=10,
)


# try:
#     episode.custom_metrics['efficiency'] = max(effs)
#     plt.imshow(env_lst[_i].struct.tolist())
# except:
#     pass

# g_max_eff = -1.
# g_best_struct = np.ones(shape=(1,256), dtype=int)


def find_best_struct(envs):
    # global g_max_eff, g_best_struct
    effs, structs = [], []
    _e, _i = -1., 0
    for i, env, in enumerate(envs):
        effs.append(env.eff)
        structs.append(env.struct)
        if env.eff > _e:
            _e, _i = env.eff, i

    s = structs[_i].tolist()
    s = [0 if s[int(i)] == -1 else 1 for i in s]

    return _e, s


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            **kwargs,
    ) -> None:
        find_best_struct(base_env.get_sub_environments())

    def on_episode_step(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            **kwargs,
    ) -> None:
        find_best_struct(base_env.get_sub_environments())

    # def on_episode_end(
    #         self,
    #         *,
    #         worker,
    #         base_env,
    #         policies,
    #         episode,
    #         **kwargs,
    # ) -> None:
    #     episode.custom_metrics['eff'] = max([i.eff for i in base_env.get_unwrapped()])

    # wandb.log({'max_eff': max_eff})




def make_env(**env_config):
    env = ReticoloEnv(**env_config)
    env = TimeLimit(env, max_episode_steps=128)
    env = RecordEpisodeStatistics(env)

    return env


def main(
        pretrained_dir='/mnt/8tb/Sep16_ShallowUNet_v2_256_2man_1100_70_0.00347449_0.00411770_stateDict.pt'
):
    # TODO: how to pass polyak tau
    register_env(ENV_ID, lambda c: make_env(**config.env_config))
    ModelCatalog.register_custom_model("model", ShallowUQnet)
    # ModelCatalog.register_custom_model("model", UNet)

    # stop = {
    #     # "training_iteration": stop_iters,
    #     "timesteps_total": 10000000  # 134217728,  # 8192
    #     # "episode_reward_mean": stop_reward,
    # }
    trainer = DQN(config=config)

    policy_ids = trainer.get_weights().keys()
    pretrained = load_pretrained_apex(pretrained_dir)
    trainer.set_weights({
        i: deepcopy(pretrained) for i in policy_ids
    })

    def train_fn(config=None):
        for i in tqdm(range(500)):
            metric = trainer.train()

            def _f(e):
                return e.max_eff, e.best_struct, e.episode_returns.mean()

            r = trainer.workers.foreach_env(_f)
            r = [i for i in r if len(i) > 0]
            r = list(chain(*r))

            mean_eff = np.array([i[2] for i in r]).mean()
            if len(r) > 0:
                r.sort(key=lambda t: t[0], reverse=True)
                eff, st, _ = r[0]
                st = np.array([0 if i == -1 else 1 for i in st], dtype=int)[np.newaxis, :]
                img = wandb.Image(st, caption=f"efficiency {eff:.6f}")
                wandb.log({
                    'max_eff': eff,
                    'mean_eff': mean_eff,
                    "best_structure": img,
                }, step=i*BATCH_SIZE)
            # s = trainer.get_state()
            # print(f"timesteps_total: {s['timesteps_total']}, time_total: {s['time_total']}")
            # pprint(metric)

    train_fn()
    # results = tune.Tuner(
    #     train_fn,
    #     # param_space=config.to_dict(),
    #     run_config=air.RunConfig(
    #         local_dir='/mnt/8tb/anthony/ray-result',
    #         stop={'training_iteration': 10000},
    #         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
    #     )
    # ).fit()


if __name__ == "__main__":
    main()
