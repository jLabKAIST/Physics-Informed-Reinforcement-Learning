from ray import tune
from ray.rllib.agents.dqn import apex
from ray.rllib.algorithms import apex_dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

from pirl.networks import ShallowUQnet
from pirl.envs.reticolo_env import ReticoloEnv
from pirl.networks import UNet

torch, nn = try_import_torch()

if __name__ == '__main__':
    ModelCatalog.register_custom_model("model", ShallowUQnet)

    config = {
        'framework': 'torch',
        'horizon': 512,
        # 'num_gpus': 4
        'replay_buffer_config': {
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'capacity': 5000000
        },
        'model': {
            'custom_model': 'model'
        }
    }

    config = apex.ApexTrainer.merge_trainer_configs(
        apex.APEX_DEFAULT_CONFIG,
        config
    )

    env_id = "deflector-v0"
    tune.register_env(env_id, ReticoloEnv)
    config['env'] = env_id

    stop = {
        # "training_iteration": stop_iters,
        "timesteps_total": 100000000  # 134217728,  # 8192
        # "episode_reward_mean": stop_reward,
    }

    tune.run(
        apex.ApexTrainer,
        name='pirl-apex',
        config=config,
        reuse_actors=True,
        # checkpoint_freq=1024,
        checkpoint_at_end=True,
        # max_failures=3,
        local_dir='log',
        stop=stop
    )
