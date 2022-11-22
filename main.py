import random
from operator import itemgetter

import numpy as np

import gym
from gym.wrappers import NormalizeReward
import deflector_gym
from deflector_gym.wrappers import BestRecorder, ExpandObservation

import torch
from torch import nn

import ray
from ray import air, tune
from ray.tune import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.simple_q import SimpleQConfig
from ray.rllib.algorithms.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from model import ShallowUQNet

class Callbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)

    def _get_max(self, base_env):
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))
        return best[0], best[1]

    def _tb_image(self, structure):
        # transform sttructure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_learn_on_batch(self, *, policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        pass
        
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        pass

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)
        episode.custom_metrics['initial_efficiency'] = eff

    def on_episode_end(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            **kwargs,
    ) -> None:
        eff, struct = self._get_max(base_env)
        
        episode.custom_metrics['max_efficiency'] = eff
        

if __name__ == '__main__':
    stop = {
        "timesteps_total": 200000,
    }
    env_id = 'MeentIndex-v0'

    def make_env(config):
        e = deflector_gym.make(env_id)
        e = BestRecorder(e)
        e = ExpandObservation(e)

        return e


    register_env(env_id, make_env)
    ModelCatalog.register_custom_model(ShallowUQNet.__name__, ShallowUQNet)

    config = SimpleQConfig()
    config.framework(
        framework='torch'
    ).environment(
        env=env_id,
        normalize_actions=False
    ).callbacks(
        Callbacks # logging
    ).training(
        model={
            'custom_model': ShallowUQNet.__name__,
            'no_final_linear': True,
            'vf_share_layers': False,
        },
        target_network_update_freq=2000,
        replay_buffer_config={
            "_enable_replay_buffer_api": True,
            "type": "ReplayBuffer",
            # "type": "MultiAgentReplayBuffer", # when num_workers > 0
            "learning_starts": 1000,
            "capacity": 100000,
            "replay_sequence_length": 1,
        },
        # dueling=False,
        lr=0.001,
        gamma=0.99,
        train_batch_size=512,
        tau=0.1,
    ).resources(
        num_gpus=1
    ).rollouts(
        horizon=128,
        num_rollout_workers=0, # important!! each accounts for process
        num_envs_per_worker=1, # each accounts for process
        rollout_fragment_length=2,
    ).exploration(
        explore=True,
        exploration_config={
            "type": "EpsilonGreedy",
            'initial_epsilon': 0.99,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 100000,
        }
    )

    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir='/mnt/8tb/anthony/pirl',
            name='debug-simpleq',
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
            ),
        ),
    )

    ray.init(local_mode=False)
    results = tuner.fit()