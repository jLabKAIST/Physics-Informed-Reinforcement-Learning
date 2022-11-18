from pprint import pprint

from ray.rllib.algorithms.bandit import BanditLinUCBConfig

import common
import wandb
from pirl._networks import ShallowUQnet

config = BanditLinUCBConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).callbacks(
    common.Callbacks
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
    eff, img = common.process_result(algo)

    wandb.log({
        f'best efficiency': eff,
        f'best structure': img,
    }, step=step)

    step = result['agent_timesteps_total']