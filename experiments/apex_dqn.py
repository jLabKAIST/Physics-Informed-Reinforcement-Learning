from copy import deepcopy
from itertools import chain

import numpy as np

import wandb
from gym.wrappers import TimeLimit, RecordEpisodeStatistics

from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.models.catalog import ModelCatalog
from ray import tune, air
from ray.tune import register_env
from tqdm import tqdm

from experiments.utils import Callbacks
from experiments.utils import load_pretrained
from pirl._networks import ShallowUQnet
from pirl.envs.reticolo_env import ReticoloEnv
from pirl.envs.meent_env import MeentEnv

NUM_GPUS = 1
ENV_ID = "deflector-v0"
BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 16

# TODO: how to pass polyak tau
config = ApexDQNConfig()
config.framework(
    framework='torch'
).environment(
    env=ENV_ID,
    env_config={
        'n_cells': 256,
        'wavelength': 1100,
        'desired_angle': 70
    },
    # normalize_actions=False,
).callbacks(
    Callbacks
)
    # .training(
    # model={
    #     'custom_model': 'model'
    # },
    # dueling=,
    # target_network_update_freq=2000,
# ).rollouts(
    # num_rollout_workers=1,
    # num_envs_per_worker=1,
# )
#     replay_buffer_config={
#         "_enable_replay_buffer_api": True,
#         "type": "MultiAgentReplayBuffer",
#         "learning_starts": 1000,
#         "capacity": 100000,
#         "replay_sequence_length": 1,
#     },
#     dueling=True,
#     lr=0.001,
#     gamma=0.99,
#     train_batch_size=BATCH_SIZE
# ).resources(
#     num_gpus=NUM_GPUS
# ).rollouts(
#     horizon=128,
#     num_rollout_workers=NUM_ROLLOUT_WORKERS,
#     num_envs_per_worker=1,
#     rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
#     # rollout_fragment_length: Divide episodes into fragments of this many steps
#     # each during rollouts. Sample batches of this size are collected from
#     # rollout workers and combined into a larger batch of `train_batch_size`
#     # for learning. For example, given rollout_fragment_length=100 and
#     # train_batch_size=1000: 1. RLlib collects 10 fragments of 100 steps each
#     # from rollout workers. 2. These fragments are concatenated and we
#     # perform an epoch of SGD. When using multiple envs per worker,
#     # the fragment size is multiplied by `num_envs_per_worker`. This is since
#     # we are collecting steps from multiple envs in parallel. For example,
#     # if num_envs_per_worker=5, then rollout workers will return experiences
#     # in chunks of 5*100 = 500 steps. The dataflow here can vary per
#     # algorithm. For example, PPO further divides the train batch into
#     # minibatches for multi-epoch SGD.
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
# ).callbacks(
#     MyCallbacks
# )

from pirl.envs.meent_env import DirectionEnv
def make_env(**env_config):
    env = DirectionEnv(**env_config)
    # env = MeentEnv(**env_config)
    env = TimeLimit(env, max_episode_steps=1024)
    env = RecordEpisodeStatistics(env)

    return env


#register_env(ENV_ID, lambda c: make_env(**config.env_config))
#ModelCatalog.register_custom_model("model", ShallowUQnet)

# wandb.init(project='PIRL', config=config)
# trainer = ApexDQN(config=config)

# pretrained_dir='/home/anthony/Sep16_ShallowUNet_v2_256_2man_1100_70_0.00347449_0.00411770_stateDict.pt'
# policy_ids = trainer.get_weights().keys()
# pretrained = load_pretrained(pretrained_dir)
# trainer.set_weights({
#     i: deepcopy(pretrained) for i in policy_ids
# })
# trainer.save('/home/anthony/apex-dqn')


def main(
        pretrained_dir='/mnt/8tb/anthony/ckpt/apex-dqn/'
):
    # TODO: how to pass polyak tau
    register_env(ENV_ID, lambda c: make_env(**config.env_config))
    # ModelCatalog.register_custom_model("model", ShallowUQnet)
    # ModelCatalog.register_custom_model("model", UNet)
    tune.run(
        ApexDQN,
        config=config.to_dict(),
#        reuse_actors=True,
        # restore='/home/anthony/apex-dqn/checkpoint_000000',
        local_dir='/home/anthony/ray-result',
        # verbose=0,
    )

if __name__ == "__main__":
    main()


# we can save trainer here and load


# def train_fn(config=None):
#     for i in tqdm(range(50000)):
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
#             }, step=i * BATCH_SIZE)
#
# train_fn()
