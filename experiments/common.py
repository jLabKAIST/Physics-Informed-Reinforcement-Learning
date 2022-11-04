"""
these configs and methods are instantly changed, but used across experiments
"""
import itertools
from netrc import netrc
from operator import itemgetter

from torch.utils.tensorboard import SummaryWriter

PROJECT = 'PIRL-FINAL'

from typing import Dict, Union, Optional

from ray.rllib.models import ModelCatalog
from ray.tune import register_env
import numpy as np
import wandb
import deflector_gym
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

# logging and data path
DATA_DIR = '/mnt/8tb/anthony'
LOG_DIR = f'{DATA_DIR}/ray-result'
# ENV_ID = 'MeentIndex-v0'  # 'MeentDirection-v0'
WANDB = {
    'project': PROJECT,
    'api_key': netrc().authenticators('api.wandb.ai')[-1]
}
APEX_MEENTINDEX_PRETRAIN = f'{DATA_DIR}/ckpt/apex-dqn/checkpoint_000000'
DQN_MEENTINDEX_PRETRAIN = f'{DATA_DIR}/ckpt/dqn/checkpoint_000000'

# algorithm
ENV_ID = 'MeentIndex-v0'
# ENV_ID = 'MeentIndex-v0'
ENV_CONFIG = {
    'n_cells': 256,
    'wavelength': 1100,
    'desired_angle': 70,
    'order': 40,
}
MAX_TIMESTEPS = int(2e+7)
import gym
# helper methods
def make_env(wrapper_clses=None, **env_config):
    env = deflector_gym.make(ENV_ID, **env_config)
    for w in wrapper_clses:
        env = w(env)
    # env = TimeLimit(env, max_episode_steps=128)
    # env = RecordEpisodeStatistics(env)
    # env = gym.wrappers.FlattenObservation(env)
    # env = deflector_gym.wrappers.BestRecorder(env)
    # env = deflector_gym.wrappers.ExpandObservation(env)

    return env


def register_all(config, wrapper_clses=None, model_cls=None):
    register_env(ENV_ID, lambda c: make_env(wrapper_clses, **config.env_config))
    if model_cls:
        ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

class Callbacks(DefaultCallbacks):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        pass

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        pass

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        pass

    # @wandb_mixin
    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            **kwargs,
    ) -> None:
        # best (eff, struct)
        bests = [e.best for e in base_env.get_sub_environments()]

        best = max(bests, key=itemgetter(0))

        max_eff = best[0]
        img = best[1][np.newaxis, np.newaxis, :].repeat(32, axis=1)
        mean_eff = np.array([i[0] for i in bests]).mean()

        episode.custom_metrics['best_efficiency'] = max_eff
        episode.custom_metrics['mean_efficiency'] = mean_eff

        episode.media['best_structure'] = img


def process_result(algo):
    bests = algo.workers.foreach_env(lambda env: env.best)

    bests.pop(0)
    bests = list(itertools.chain(*bests))

    # bests.sort(key=lambda x: x[0]) # ascending order
    # best = bests[-1]
    best = max(bests, key=itemgetter(0))

    max_eff = best[0]
    img = best[1][np.newaxis, np.newaxis, :].repeat(32, axis=1)
    mean_eff = np.array([i[0] for i in bests]).mean()

    return {
        'scalar': {
            f'best_efficiency': max_eff,
            f'mean_efficiency': mean_eff,
        },
        'img': {
            f'best_structure': img,
        }
    }