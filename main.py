from datetime import datetime
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

from model import ShallowUQNet, FCNQNet, FCNQNet_heavy
from utils import StructureWriter, seed_all

ROOT_DIR = '/mnt/8tb'
TIMEFOLDERNAME = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = '/mnt/8tb/np_struct'+'/'+TIMEFOLDERNAME
ckpt_folder = 'pirl/SimpleQ_MeentIndex-v0_9f4ca_00000_0_2023-02-01_03-46-50/checkpoint_000196'

os.makedirs(LOG_DIR, exist_ok=True)

FINAL_Q_PATH = os.path.join(
    ROOT_DIR,
    ckpt_folder,
)

PRETRAINED_MODEL_PATH = os.path.join(
    ROOT_DIR,
    'May13_normalized_UNet_256_2man_900_50_0.002_0.014_stateDict.pt',
)

# seed_all(42)
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
        # pass

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

    def on_learn_on_batch(self, *, policy, train_batch: SampleBatch, result, **kwargs) -> None:
        pass
        
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        pass

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)

        episode.custom_metrics['initial_efficiency'] = eff

    def _j(self, a, b):
        return os.path.join(a, b)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs,) -> None:
        eff, struct = self._get_max(base_env)
        episode.custom_metrics['max_efficiency'] = eff
        filename = 'w'+ str(worker.worker_index) + f'_{eff*100:.6f}'.replace('.', '-')
        filename = self._j(LOG_DIR, filename)
        np.save(filename, struct)

if __name__ == '__main__':
    ray.init(local_mode=False) 
    stop = {
        "timesteps_total": 200000,
    }
    env_id = 'MeentIndex-v0'
    env_config = {'wavelength': 900, 'desired_angle': 50, 'thickness': 325}
    model_cls = ShallowUQNet  #model_cls = ShallowUQNet / FCNQNet / FCNQNet_heavy


    def make_env(config):
        e = deflector_gym.make(env_id, **config)
        e = BestRecorder(e)
        e = ExpandObservation(e)
        # e = StructureWriter(e)
        return e

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
        Callbacks # register logging
    ).training(
        model={'custom_model': model_cls}
    ).debugging( 
        # seed=tune.grid_search([1, 2, 3, 4, 5]) # if you want to run experiments with multiple seeds
    )
    # algo = config.build() # transfer learning with fully trained agent network
    # algo.load_checkpoint(FINAL_Q_PATH) # transfer learning with fully trained agent network
    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir=ROOT_DIR,
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