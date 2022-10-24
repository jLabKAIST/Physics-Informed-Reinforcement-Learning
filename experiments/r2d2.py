from pprint import pprint

from tqdm import tqdm

import wandb
from ray.rllib.algorithms.r2d2 import R2D2Config

import common

# TODO: how to pass polyak tau
config = R2D2Config()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG
).training(
    model=dict(use_lstm=True)
).callbacks(
    common.Callbacks
)
common.register_all(config=config)
algo = config.build()

wandb.init(project='PIRL-FINAL', group='test-group')

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