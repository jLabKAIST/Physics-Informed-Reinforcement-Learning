from pprint import pprint

from ray.rllib.algorithms.sac import SACConfig

import common
import wandb

config = SACConfig()
config.framework(
    framework='torch',
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).callbacks(
    common.Callbacks
).training(
    twin_q=True,
    q_model_config={
        'fcnet_hiddens': [512, 512, 512]
    },
    policy_model_config={
        'fcnet_hiddens': [512, 512, 512],
    },
    n_step=3,
)

# TODO: how to pass polyak tau
common.register_all(config=config)

algo = config.build()
# algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

wandb.init(
    project='PIRL-FINAL',
    group='test-group',
    config=config.to_dict(),
    name=f"{common.ENV_ID}"
)
# wandb.require('service')


step = 0
while step < 20000000:
    result = algo.train()
    pprint(result)
    # logging
    result_dict = common.process_result(algo)

    wandb.log(result_dict, step=step)

    step = result['agent_timesteps_total']