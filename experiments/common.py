"""
these configs and methods are instantly changed, but used across experiments
"""
PROJECT = 'PIRL'

from typing import Dict, Union

from ray.rllib.models import ModelCatalog
from ray.tune import register_env

import wandb
from gym.wrappers import TimeLimit, RecordEpisodeStatistics
import deflector_gym
from deflector_gym.wrappers import BestRecorder
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.tune.integration.wandb import wandb_mixin
from pathlib import Path

ENV_ID = 'MeentIndex-v0'  # 'MeentDirection-v0'
LOG_DIR = f'{Path().home()}/ray-result'
WANDB = {
    'project': PROJECT,
    'api_key_file': '/home/anthony/.netrc'  # confidential
}
from pirl._networks import ShallowUQnet

MODEL_CLS = ShallowUQnet
MODEL_NAME = MODEL_CLS.__name__

APEX_PRETRAIN = f'/home/anthony/physics-informed-metasurface/ckpt/apex-dqn/checkpoint_000000'
DQN_PRETRAIN = f'/home/anthony/physics-informed-metasurface/ckpt/dqn/checkpoint_000000'
MAX_TIMESTEPS = int(2e+7)

ENV_CONFIG = {
    'n_cells': 256,
    'wavelength': 1100,
    'desired_angle': 70
}


def make_env(**env_config):
    env = deflector_gym.make(ENV_ID, **env_config)
    # env = TimeLimit(env, max_episode_steps=128)
    # env = RecordEpisodeStatistics(env)
    env = BestRecorder(env)

    return env


def register_all(config, model_name=MODEL_NAME, model_cls=MODEL_CLS):
    register_env(ENV_ID, lambda c: make_env(**config.env_config))
    ModelCatalog.register_custom_model(model_name, model_cls)


class Callbacks(DefaultCallbacks):
    @wandb_mixin
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

        eff = bests[0][0]
        img = wandb.Image(bests[0][1], caption=f"efficiency {eff:.6f}")

        wandb.log({
            f'best efficiency': eff,
            f'best structure': img,
        }, step=episode.total_env_steps)
