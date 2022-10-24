"""
these configs and methods are instantly changed, but used across experiments
"""
import itertools
from netrc import netrc

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
ENV_ID = 'MeentDirectionChoice-v0'
# ENV_ID = 'MeentIndex-v0'
ENV_CONFIG = {
    'n_cells': 256,
    'wavelength': 1100,
    'desired_angle': 70
}
MAX_TIMESTEPS = int(2e+7)
import gym
# helper methods
def make_env(wrapper_clses, **env_config):
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
        # wandb.require('service')
        # for w in algorithm.workers:
        #     wandb.init(project='PIRL-FINAL', group='test-group', name=w.__name__)
        pass

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        # wandb.init(project='PIRL-FINAL', group='test-group')
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
        bests = [e.best for e in base_env.get_sub_environments()]
        bests.sort(key=lambda x: x[0])
        episode.media['best'] = bests[0]
        # eff = bests[0][0]
        # img = wandb.Image(bests[0][1][np.newaxis, :], caption=f"efficiency {eff:.6f}")
        #
        #
        # wandb.log({
        #     f'best efficiency': eff,
        #     f'best structure': img,
        # }, step=episode.episode_id)


def process_result(algo):
    bests = algo.workers.foreach_env(lambda env: env.best)
    bests.pop(0)
    bests = list(itertools.chain(*bests))
    bests.sort(key=lambda x: x[0]) # ascending order
    max_eff = bests[-1][0]
    img = wandb.Image(bests[-1][1][np.newaxis, :], caption=f"efficiency {max_eff:.6f}")
    mean_eff = np.array([i[0] for i in bests]).mean()

    return {
        f'best efficiency': max_eff,
        f'best structure': img,
        f'mean efficiency': mean_eff,
    }
