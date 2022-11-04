from deflector_gym.wrappers import BestRecorder
from ray.rllib.algorithms.dqn import DQNConfig

import common
from pirl._networks import ShallowUQnet

NUM_GPUS = 1
BATCH_SIZE = 512

MODEL_CLS = ShallowUQnet

"""
define configs
"""
config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).training(
    model={
        'custom_model': MODEL_CLS.__name__,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        # "type": "ReplayBuffer",
        "type": "MultiAgentReplayBuffer",
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=512,
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=128,
    num_rollout_workers=1,# important!! each accounts for process
    num_envs_per_worker=1, # each accounts for process
    rollout_fragment_length=2,
).exploration(
    explore=True,
    exploration_config={
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 100000,
    }
).callbacks(
    common.Callbacks # logging
)

from deflector_gym.wrappers import ExpandObservation

# TODO: how to pass polyak tau
common.register_all(
    config=config,
    wrapper_clses=[BestRecorder, ExpandObservation],
    model_cls=MODEL_CLS
)

"""
build algorithm
"""
algo = config.build()

"""
load pretrained model
"""
algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

"""
main logic
"""
step = 0
while step < common.MAX_TIMESTEPS:
    result = algo.train()
    step = result['agent_timesteps_total']
