from itertools import chain
from typing import Dict, Union, Optional

import numpy as np
from ray import tune, air
from ray.rllib import Policy, BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from tqdm import tqdm

import wandb
from gym.wrappers import TimeLimit, RecordEpisodeStatistics, NormalizeReward, NormalizeObservation
from ray.rllib.models.catalog import ModelCatalog
from ray.tune import register_env

from copy import deepcopy

from experiments.utils import load_pretrained
from pirl._networks import ShallowUQnet
from pirl.envs.reticolo_env import ReticoloEnv
from pirl.envs.meent_env import MeentEnv
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.callbacks import DefaultCallbacks

NUM_GPUS = 1
ENV_ID = "deflector-v0"
BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 32

# wandb.init(project='PIRL')

from ray.rllib.algorithms.dqn import DQNConfig


class Callbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs,
    ) -> None:
        episode.custom_metrics['eff'] = [e.max_eff for e in base_env.get_sub_environments()]


config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=ENV_ID,
    env_config={
        'n_cells': 256,
        'wavelength': 1100,
        'desired_angle': 70
    },
    # disable_env_checking=True
).training(
    # hiddens=[256, 256, 256],
    model={
        'custom_model': 'model'
    },
    # target_network_update_freq=2000,
    # replay_buffer_config={
    #     "_enable_replay_buffer_api": True,
    #     "type": "MultiAgentReplayBuffer",
    #     "learning_starts": 1000,
    #     "capacity": 100000,
    #     "replay_sequence_length": 1,
    # },
    dueling=False,
    lr=0.0001,
#     gamma=0.99,
#     train_batch_size=BATCH_SIZE
# ).resources(
#     num_gpus=4
).rollouts(
#     horizon=128,
    num_rollout_workers=32,
    num_envs_per_worker=3,
#     # rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
).callbacks(
    Callbacks
)
# ).exploration(
#     explore=True,
#     exploration_config={
#         'initial_epsilon': 0.99,
#         'final_epsilon': 0.01,
#         # 'epsilon_timesteps': 100000,
#     }
# ).evaluation(
#     evaluation_duration_unit='episodes',
#     evaluation_duration=10,
# )


def make_env(**env_config):
    # env = ReticoloEnv(**env_config)
    env = MeentEnv(**env_config)
    env = TimeLimit(env, max_episode_steps=128)
    env = RecordEpisodeStatistics(env)
    env = NormalizeReward(env)
    # env = NormalizeObservation(env)

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
    # trainer = DQN(config=config).restore('/mnt/8tb/anthony/ckpt/dqn/checkpoint_000000')
    # trainer
    # policy_ids = trainer.get_weights().keys()
    # pretrained = load_pretrained(pretrained_dir)
    # trainer.set_weights({
    #     i: deepcopy(pretrained) for i in policy_ids
    # })
    # ckpt_dir = trainer.save('/mnt/8tb/anthony/ckpt/dqn')
    # new_trainer = DQN(config=config)
    # new_trainer.restore(ckpt_dir)
    # print(new_trainer)

    # def train_fn(config=None):
    #     for i in tqdm(range(500)):
    #         metric = trainer.train()
    #
    #         def _f(e):
    #             return e.max_eff, e.best_struct, e.episode_returns.mean()
    #
    #         r = trainer.workers.foreach_env(_f)
    #         r = [i for i in r if len(i) > 0]
    #         r = list(chain(*r))
    #
    #         mean_eff = np.array([i[2] for i in r]).mean()
    #         if len(r) > 0:
    #             r.sort(key=lambda t: t[0], reverse=True)
    #             eff, st, _ = r[0]
    #             st = np.array([0 if i == -1 else 1 for i in st], dtype=int)[np.newaxis, :]
    #             img = wandb.Image(st, caption=f"efficiency {eff:.6f}")
    #             wandb.log({
    #                 'max_eff': eff,
    #                 'mean_eff': mean_eff,
    #                 "best_structure": img,
    #             }, step=i*BATCH_SIZE)
            # s = trainer.get_state()
            # print(f"timesteps_total: {s['timesteps_total']}, time_total: {s['time_total']}")
            # pprint(metric)

    # train_fn()
    # trainer = DQN(config=config)
    # trainer.restore('/mnt/8tb/anthony/ckpt/dqn/checkpoint_000000')
    # #
    # def train_fn(config=None):
    #     while True:
    #         trainer.train()

    # results = tune.Tuner(
    #     DQN,
    #     param_space=config.to_dict(),
    #     tune_config=tune.TuneConfig(reuse_actors=True),
    #     run_config=air.RunConfig(
    #         local_dir='/mnt/8tb/anthony/ray-result',
    #         stop={'training_iteration': 1000},
    #         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
    #     ),
    # ).fit()

    tune.run(
        DQN,
        config=config.to_dict(),
        reuse_actors=True,
        restore='/mnt/8tb/anthony/ckpt/dqn/checkpoint_000000',
        local_dir='/mnt/8tb/anthony/ray-result',
        # verbose=0,
    )

from ray.tune.logger import TBXLoggerCallback
if __name__ == "__main__":
    main()
