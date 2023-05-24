import argparse
from datetime import datetime
import os
from operator import itemgetter

import numpy as np

import torch

import ray
from ray import air, tune
from ray.tune import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog

from gym.wrappers import TimeLimit
import deflector_gym
from deflector_gym.wrappers import BestRecorder, ExpandObservation

from model import ShallowUQNet
from utils import StructureWriter, seed_all

DATA_DIR = None
PRETRAINED_CKPT = None
LOG_DIR = None

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
            PRETRAINED_CKPT,
            map_location=torch.device('cpu')
        )
        policy.set_weights(state_dict)

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)
        # seed_all(42)

    def _get_max(self, base_env):
        # retrieve `env.best`, where env is wrapped with BestWrapper to record the best structure
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))

        return best[0], best[1]

    def _tb_image(self, structure):
        # transform structure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)

        episode.custom_metrics['initial_efficiency'] = eff

    def _j(self, a, b):
        return os.path.join(a, b)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs, ) -> None:
        eff, struct = self._get_max(base_env)
        episode.custom_metrics['max_efficiency'] = eff
        filename = 'w' + str(worker.worker_index) + f'_{eff * 100:.6f}'.replace('.', '-')
        filename = self._j(LOG_DIR, filename)
        np.save(filename, struct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, default='run',
        help='absolute path to data directory'
    )
    parser.add_argument(
        '--transfer_ckpt', type=str, default=None,
        help='absolute path to checkpoint file to do transfer learning'
    )
    parser.add_argument(
        '--pretrained_ckpt', type=str, default='pretrained/pretrained_1100nm_60degree.pt',
        help='absolute path to checkpoint file of pretrained model'
    )
    parser.add_argument(
        '--wavelength', type=int, default=1100,
        help='wavelength of the incident light'
    )
    parser.add_argument(
        '--angle', type=int, default=60,
        help='target deflection angle condition'
    )
    parser.add_argument(
        '--thickness', type=int, default=325,
        help='thickness of the pillar'
    )
    parser.add_argument(
        '--train_steps', type=int, default=200000,
        help='number of training steps'
    )

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    LOG_DIR = f"{args.data_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    PRETRAINED_CKPT = args.pretrained_ckpt

    os.makedirs(LOG_DIR, exist_ok=True)

    ray.init(local_mode=False)

    env_id = 'MeentIndex-v0'
    env_config = {'wavelength': args.wavelength, 'desired_angle': args.angle, 'thickness': args.thickness}
    model_cls = ShallowUQNet  # model_cls = ShallowUQNet / FCNQNet / FCNQNet_heavy

    def make_env(config):
        env = deflector_gym.make(env_id, **config)
        env = BestRecorder(env)
        env = ExpandObservation(env)
        env = StructureWriter(env, DATA_DIR)
        env = TimeLimit(env, max_episode_steps=128)

        return env

    register_env(env_id, lambda c: make_env(env_config))
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

    from configs.simple_q import multiple_worker as config

    config.framework(
        framework='torch'
    ).environment(
        env=env_id,
        env_config=env_config,
        normalize_actions=False
    ).callbacks(
        Callbacks  # register logging
    ).training(
        model={'custom_model': model_cls}
    ).debugging(
        # seed=tune.grid_search([1, 2, 3, 4, 5]) # if you want to run experiments with multiple seeds
    )

    algo = config.build()
    if args.transfer_ckpt:
        algo.load_checkpoint(args.transfer_ckpt)
    stop = {
        "timesteps_total": args.train_steps,
    }
    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir=DATA_DIR,
            name='pirl',
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute='episode_reward_max',
                checkpoint_score_order='max',
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
        ),
    )

    results = tuner.fit()
    ray.shutdown()
