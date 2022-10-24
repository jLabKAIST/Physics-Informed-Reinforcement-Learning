import wandb
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from experiments.common import process_result
from pirl._networks import ShallowUQnet

import common

NUM_GPUS = 4
BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 16

MODEL_CLS = ShallowUQnet
# MODEL_CLS = RecurrentNetwork
# MODEL_CLS = FullyConnectedNetwork

config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).training(
    model={
        'custom_model': MODEL_CLS.__name__,
        # 'fcnet_hiddens': [256, 256, 256]
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
#    training_intensity=2 * BATCH_SIZE,
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=1024,
    num_rollout_workers=32,
    num_envs_per_worker=8,
    # rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
).exploration(
    explore=True,
    exploration_config={
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        # 'epsilon_timesteps': 100000,
    }
).callbacks(
    common.Callbacks
)

# TODO: how to pass polyak tau
common.register_all(config=config, model_cls=MODEL_CLS)
algo = config.build()
algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

wandb.init(project='PIRL-FINAL', group='test-group', config=config.to_dict())
# wandb.require('service')


step = 0
while step < 20000000:
    result = algo.train()

    # logging
    result_dict = process_result(algo)

    wandb.log(result_dict, step=step)

    step = result['agent_timesteps_total']