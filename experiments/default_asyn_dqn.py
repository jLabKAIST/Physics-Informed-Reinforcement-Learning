from pprint import pprint

from ray.rllib.algorithms.dqn import DQNConfig

import common
import wandb
from pirl._networks import ShallowUQnet

# MODEL_CLS = ShallowUQnet

config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).training(
    dueling=True,
    # model={
    #     'custom_model': MODEL_CLS.__name__
    # },
).resources(
    num_gpus=4
).rollouts(
    num_rollout_workers=32
).exploration(
    explore=True
)


# TODO: how to pass polyak tau
common.register_all(config=config)

algo = config.build()
# algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

wandb.init(project='PIRL-FINAL', group='test-group', config=config.to_dict())
# wandb.require('service')


step = 0
while step < 20000000:
    result = algo.train()
    pprint(result)
    # logging
    result_dict = common.process_result(algo)

    wandb.log(result_dict, step=step)

    step = result['agent_timesteps_total']