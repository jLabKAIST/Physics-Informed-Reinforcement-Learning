from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn import DQNConfig

import common
from experiments.common import Callbacks

NUM_GPUS = 1

BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 16

# wandb.init(project='PIRL')


config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).training(
    model={
        'custom_model': common.MODEL_NAME
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=BATCH_SIZE,
    training_intensity=2 * BATCH_SIZE,
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=128,
    num_rollout_workers=32,
    num_envs_per_worker=4,
    # rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
).exploration(
    explore=True,
    exploration_config={
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        # 'epsilon_timesteps': 100000,
    }
).callbacks(
    Callbacks
)

# ).evaluation(
#     evaluation_duration_unit='episodes',
#     evaluation_duration=10,

# TODO: how to pass polyak tau
common.register_all(config=config)

tune.run(
    DQN,
    config=config.to_dict(),
    reuse_actors=True,
    restore=common.DQN_PRETRAIN,
    local_dir=common.LOG_DIR,
    stop={'timesteps_total': common.MAX_TIMESTEPS},
    callbacks=[WandbLoggerCallback(**common.WANDB)],
    verbose=0,
)