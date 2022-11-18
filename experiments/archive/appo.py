from pprint import pprint

from deflector_gym.wrappers import BestRecorder
from ray.rllib.algorithms.ppo import PPOConfig


import common
import wandb
# from pirl._networks import ShallowUQnet
# MODEL_CLS = ShallowUQnet

config = PPOConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).callbacks(
    common.Callbacks
)

# TODO: how to pass polyak tau
common.register_all(wrapper_clses=[BestRecorder], config=config)

algo = config.build()
# algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

wandb.init(project='PIRL-FINAL', group='test-group', config=config.to_dict())
# wandb.require('service')


step = 0
while step < common.MAX_TIMESTEPS:
    result = algo.train()
    pprint(result)
    # logging
    result_dict = common.process_result(algo)

    wandb.log(result_dict, step=step)

    step = result['agent_timesteps_total']