import random
from operator import itemgetter

import deflector_gym
import gym
import numpy as np
import torch
from deflector_gym.wrappers import BestRecorder, ExpandObservation
from gym.wrappers import NormalizeReward
from ray import air, tune
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchModel
from ray.rllib.algorithms.simple_q import SimpleQConfig
from ray.rllib.algorithms.simple_q.simple_q_torch_policy import \
    SimpleQTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune import register_env
from torch import nn

# seeding needs to be taken care when multiple workers are used
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) 

def init_params(net, val=np.sqrt(2)):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, val)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, val)
            if module.bias is not None:
                module.bias.data.zero_()

class convrelu(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.convrelu = nn.Sequential(
            nn.Conv1d(nin, nout, 3, padding='same', padding_mode='circular'),
            nn.BatchNorm1d(nout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convrelu(x)


class ShallowUQnet(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = 256
        init_params(self)


        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2)  # non-Uniform

        self.conv2_1 = nn.Conv1d(16, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv2_2 = convrelu(32, 32)
        self.conv2_3 = convrelu(32, 32)
        self.pool_2 = nn.MaxPool1d(2)  # non-Uniform

        self.conv3_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv3_2 = convrelu(32, 32)
        self.conv3_3 = convrelu(32, 32)
        self.pool_3 = nn.MaxPool1d(2)  # Uniform (X

        self.conv4_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv4_2 = convrelu(32, 32)
        self.conv4_3 = convrelu(32, 32)
        self.pool_4 = nn.MaxPool1d(2)  # Uniform (X

        self.conv6_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv6_2 = convrelu(32, 32)
        self.conv6_3 = convrelu(32, 32)
        self.upsam_6 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv8_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv8_2 = convrelu(32, 32)
        self.conv8_3 = convrelu(32, 32)
        self.upsam_8 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv9_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv9_2 = convrelu(32, 32)
        self.conv9_3 = convrelu(32, 32)
        self.upsam_9 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv10_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv10_2 = convrelu(32, 32)
        self.conv10_3 = convrelu(32, 32)
        self.upsam_10 = nn.Upsample(scale_factor=2)  # non-Uniform

        self.conv11_1 = nn.Conv1d(16 + 32, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same', bias=True, padding_mode='circular')

    def forward(self, input_dict, state, seq_lens):
        img = input_dict['obs']

        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1
        shortcut1 = temp
        temp = self.pool_1(shortcut1)

        res2_1 = self.conv2_1(temp)
        temp = self.conv2_2(res2_1)
        temp = self.conv2_3(temp) + res2_1
        shortcut2 = temp
        temp = self.pool_2(shortcut2)

        res3_1 = self.conv3_1(temp)
        temp = self.conv3_2(res3_1)
        temp = self.conv3_3(temp) + res3_1
        shortcut3 = temp
        temp = self.pool_3(shortcut3)

        res4_1 = self.conv4_1(temp)
        temp = self.conv4_2(res4_1)
        temp = self.conv4_3(temp) + res4_1
        shortcut4 = temp
        temp = self.pool_4(shortcut4)

        res6_1 = self.conv6_1(temp)
        temp = self.conv6_2(res6_1)
        temp = self.conv6_3(temp) + res6_1
        temp = self.upsam_6(temp)
        temp = torch.cat([temp, shortcut4], dim=1)  ######

        res8_1 = self.conv8_1(temp)
        temp = self.conv8_2(res8_1)
        temp = self.conv8_3(temp) + res8_1
        temp = self.upsam_8(temp)
        temp = torch.cat([temp, shortcut3], dim=1)  ######

        res9_1 = self.conv9_1(temp)
        temp = self.conv9_2(res9_1)
        temp = self.conv9_3(temp) + res9_1
        temp = self.upsam_9(temp)
        temp = torch.cat([temp, shortcut2], dim=1)  ######

        res10_1 = self.conv10_1(temp)
        temp = self.conv10_2(res10_1)
        temp = self.conv10_3(temp) + res10_1
        temp = self.upsam_10(temp)
        temp = torch.cat([temp, shortcut1], dim=1)  ######

        res11_1 = self.conv11_1(temp)
        temp = self.conv11_2(res11_1)
        temp = self.conv11_3(temp) + res11_1
        temp = self.conv11_fin(temp)
        
        temp = temp.flatten(1)

        return temp, []

class OneHot(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0., high=1.,
            shape=(256, ), #### TODO fix shape
            dtype=np.float64
        )
        
    def observation(self, obs):
        obs[obs == -1] = 0

        return obs
    

class Callbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)

    def _get_max(self, base_env):
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))
        return best[0], best[1]

    def on_learn_on_batch(self, *, policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        pass
        
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        pass

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)
        
        # print(f'initialized {best}')
        
        episode.custom_metrics['initial_efficiency'] = eff

    def on_episode_end(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            **kwargs,
    ) -> None:
        eff, struct = self._get_max(base_env)
        # img = best[1][np.newaxis, np.newaxis, :].repeat(32, axis=1)
        # mean_eff = np.array([i[0] for i in bests]).mean()

        episode.custom_metrics['max_efficiency'] = eff
        # episode.custom_metrics['mean_efficiency'] = mean_eff

        # episode.media['best_structure'] = img

stop = {
    # "training_iteration": args.stop_iters,
    "timesteps_total": 20000000,
    # "episode_reward_mean": args.stop_reward,
}
env_id = 'MeentIndex-v0'
def make_env(config):
    e = deflector_gym.make(env_id)
    e = BestRecorder(e)
    
    # e = NormalizeReward(e)
    # e = OneHot(e)
    e = ExpandObservation(e)
    return e


register_env(env_id, make_env)
ModelCatalog.register_custom_model(ShallowUQnet.__name__, ShallowUQnet)

config = SimpleQConfig()
config.framework(
    framework='torch'
).environment(
    env=env_id,
    normalize_actions=False
).callbacks(
    Callbacks # logging
).training(
    model={
        'custom_model': ShallowUQnet.__name__,
        'no_final_linear': True,
        'vf_share_layers': False,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "ReplayBuffer",
        # "type": "MultiAgentReplayBuffer", # when num_workers > 0
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    # dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=512,
    tau=0.1,
).resources(
    num_gpus=1
).rollouts(
    horizon=128,
    num_rollout_workers=0, # important!! each accounts for process
    num_envs_per_worker=1, # each accounts for process
    rollout_fragment_length=2,
).exploration(
    explore=True,
    exploration_config={
        "type": "EpsilonGreedy",
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 100000,
    }
)

tuner = tune.Tuner(
    'SimpleQ',
    param_space=config.to_dict(),
    # tune_config=tune.TuneConfig(), # for hparam search
    run_config=air.RunConfig(
        stop=stop,
        local_dir='/mnt/8tb/anthony/pirl',
        name='debug-simpleq',
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True,
        ),
    ),
)

import ray

ray.init(local_mode=False)
results = tuner.fit()