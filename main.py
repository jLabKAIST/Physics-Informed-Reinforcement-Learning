import os
from operator import itemgetter

import numpy as np

import torch

import ray
from ray import air, tune
from ray.tune import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog

import deflector_gym
from deflector_gym.wrappers import BestRecorder, ExpandObservation

from model import ShallowUQNet
from utils import StructureWriter, seed_all

ROOT_DIR = '/mnt/8tb'
PRETRAINED_MODEL_PATH = os.path.join(
    ROOT_DIR,
    'Sep16_ShallowUNet_v2_256_2man_1100_70_0.00347449_0.00411770_stateDict.pt',
)
"""
seeding needs to be taken care when multiple workers are used,
that is, you need to set seed for each worker
"""
seed_all(42)

class Callbacks(DefaultCallbacks):
    """
    logging class for rllib
    the method's name itself stands for the logging timing

    e.g. 
    `if step % train_interval == 0:` is equivalent to `on_learn_on_batch`

    you may feel uncomfortable with this logging procedure, 
    but when distributed training is used, it's inevitable
    """
    def on_create_policy(self, *, policy_id, policy) -> None:
        state_dict = torch.load(
            PRETRAINED_MODEL_PATH, 
            map_location=torch.device('cpu')
        )
        policy.set_weights(state_dict)

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)

    def _get_max(self, base_env):
        # retrieve `env.best`, where env is wrapped with BestWrapper to record the best structure
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))

        return best[0], best[1]

    def _tb_image(self, structure):
        # transform sttructure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_learn_on_batch(self, *, policy, train_batch: SampleBatch, result, **kwargs) -> None:
        pass
        
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        pass

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)

        episode.custom_metrics['initial_efficiency'] = eff

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs,) -> None:
        eff, struct = self._get_max(base_env)
        
        episode.custom_metrics['max_efficiency'] = eff
        

if __name__ == '__main__':
    stop = {
        "timesteps_total": 200000,
    }
    env_id = 'MeentIndex-v0'
    env_config = {}
    model_cls = ShallowUQNet

    def make_env(config):
        e = deflector_gym.make(env_id, **config)
        e = BestRecorder(e)
        e = ExpandObservation(e)
        e = StructureWriter(e)

        return e


    register_env(env_id, make_env)
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

    from configs.simple_q import multiple_worker as config
    config.framework(
        framework='torch'
    ).environment(
        env=env_id,
        env_config=env_config,
        normalize_actions=False
    ).callbacks(
        Callbacks # register logging
    ).training(
        model={'custom_model': model_cls}
    ).debugging( 
        # seed=tune.grid_search([1, 2, 3, 4, 5]) # if you want to run experiments with multiple seeds
    )

    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir=ROOT_DIR,
            name='pirl',
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
            ),
        ),
    )

    ray.init(local_mode=False)
    results = tuner.fit()
    ray.shutdown()