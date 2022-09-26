from ray import tune
from ray.rllib.agents.ppo import ddppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

from pirl.env import ReticoloDeflector
from pirl.networks import UNet

torch, nn = try_import_torch()


if __name__ == '__main__':
    config = {
        # "disable_env_checking": True,
        # "_disable_preprocessor_api": False,
        # "num_workers": 4,
        # "num_cpus_per_worker": 15,
        # "observation_filter": "NoFilter",
        # "num_envs_per_worker": 1,
        # "sgd_minibatch_size": 512,
        # "train_batch_size": 32,
        # "rollout_fragment_length": 8,
        # "sgd_minibatch_size": 32,
        # "num_sgd_iter": 8,
        # "model": {
        #     "custom_model": "model",
        #     "vf_share_layers": False,
        # },
        # "log_level": "DEBUG",
        # "ignore_worker_failures": True,
        # "vf_loss_coeff": 0.5,
        # "grad_clip": 10,
        # "clip_param": 0.1,
        "horizon": 1024,
    }
    config = ddppo.DDPPOTrainer.merge_trainer_configs(
        ddppo.DEFAULT_CONFIG,
        config
    )
    ModelCatalog.register_custom_model("model", UNet)
    config = ddppo.DEFAULT_CONFIG
    env_id = "deflector-v0"
    tune.register_env(env_id, ReticoloDeflector)
    config['env'] = env_id

    stop = {
        # "training_iteration": stop_iters,
        "timesteps_total": 10000000 # 134217728,  # 8192
        # "episode_reward_mean": stop_reward,
    }
    tune.run(
        ddppo.DDPPOTrainer,
        name='pirl-ddppo',
        config=config,
        reuse_actors=True,
        checkpoint_freq=1024,
        checkpoint_at_end=True,
        # max_failures=3,
        local_dir='log',
        stop = stop
    )
