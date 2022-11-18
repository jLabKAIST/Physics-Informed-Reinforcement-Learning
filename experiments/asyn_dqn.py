from pprint import pprint

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
    env='MeentIndex-v0',
    normalize_actions=False
).training(
    double_q=False,
    model={
        'custom_model': MODEL_CLS.__name__,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "ReplayBuffer",
        # "type": "MultiAgentReplayBuffer",
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=512,
    grad_clip=9999, # TODO check
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=128,
    num_rollout_workers=0, # important!! each accounts for process
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
from gym.wrappers import TimeLimit

# TODO: how to pass polyak tau
common.register_all(
    config=config,
    wrapper_clses=[
        BestRecorder, 
        ExpandObservation, 
        # lambda x: TimeLimit(x, max_episode_steps=128)
    ],
    model_cls=MODEL_CLS
)

"""
build algorithm
"""
print('config build')
algo = config.build()

"""
load pretrained model
"""
# print('restore')
# algo.restore(common.DQN_MEENTINDEX_PRETRAIN)
# dummy_layers = [
#     "advantage_module.dueling_A_0._model.0.weight",
#     "advantage_module.dueling_A_0._model.0.bias",
#     "advantage_module.A._model.0.weight",
#     "advantage_module.A._model.0.bias"
# ]

"""
main logic
"""
step = 0
print('*'*100, algo.logdir)
while step < common.MAX_TIMESTEPS:
    result = algo.train()
    pprint(result)
    step = result['agent_timesteps_total']
