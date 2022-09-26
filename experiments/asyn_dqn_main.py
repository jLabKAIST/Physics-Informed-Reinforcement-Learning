from ray.rllib import TorchPolicy
from ray.rllib.algorithms.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.algorithms.dqn.dqn_torch_policy import (
    build_q_losses, build_q_model_and_distribution, build_q_stats,
    get_distribution_inputs_and_class, extra_action_out_fn,
    adam_optimizer, before_loss_init,
    grad_process_and_td_error_fn, ComputeTDErrorMixin,
    setup_early_mixins,
)
from ray.rllib.algorithms.dqn.dqn import DEFAULT_CONFIG # deprecated
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import build_policy_class
from ray.rllib.policy.tf_mixins import LearningRateSchedule
from ray.rllib.utils import override
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors
from ray.train.trainer import tune
from ray.tune import register_env
from ray.rllib.models.catalog import ModelCatalog

from pirl.env import ReticoloDeflector, DummyEnv
from pirl.networks import UNet, UQNet

NUM_GPUS = 1
ENV_ID = "deflector-v0"
BATCH_SIZE = 512
NUM_ROLLOUT_WORKERS = 32

from ray.rllib.algorithms.dqn import DQN, DQNConfig, DQNTorchPolicy


# class TargetNetworkMixin:
#     """Assign the `update_target` method to the SimpleQTorchPolicy
#
#     The function is called every `target_network_update_freq` steps by the
#     master learner.
#     """
#
#     def __init__(self, tau=0.1):
#         # Hard initial update from Q-net(s) to target Q-net(s).
#         self.update_target()
#         self.tau = tau
#
#     def update_target(self):
#         # Update_target will be called periodically to copy Q network to
#         # target Q network, using (soft) tau-synching.
#
#         state_dict = self.model.state_dict()
#         # Support partial (soft) synching.
#         # If tau == 1.0: Full sync from Q-model to target Q-model.
#
#         for target in self.target_models.values():
#             target_state_dict = target.state_dict()
#             partial_state_dict = {
#                 k: self.tau * state_dict[k] + (1 - self.tau) * v
#                 for k, v in target_state_dict.items()
#             }
#             target.load_state_dict(partial_state_dict)
#
#     @override(TorchPolicy)
#     def set_weights(self, weights):
#         # Makes sure that whenever we restore weights for this policy's
#         # model, we sync the target network (from the main model)
#         # at the same time.
#         TorchPolicy.set_weights(self, weights)
#         self.update_target()
#
#
# DQNTorchPolicy = build_policy_class(
#     name="DQNTorchPolicy",
#     framework="torch",
#     loss_fn=build_q_losses,
#     get_default_config=lambda: DEFAULT_CONFIG,
#     make_model_and_action_dist=build_q_model_and_distribution,
#     action_distribution_fn=get_distribution_inputs_and_class,
#     stats_fn=build_q_stats,
#     postprocess_fn=postprocess_nstep_and_prio,
#     optimizer_fn=adam_optimizer,
#     extra_grad_process_fn=grad_process_and_td_error_fn,
#     extra_learn_fetches_fn=concat_multi_gpu_td_errors,
#     extra_action_out_fn=extra_action_out_fn,
#     before_init=setup_early_mixins,
#     before_loss_init=before_loss_init,
#     mixins=[
#         TargetNetworkMixin,
#         ComputeTDErrorMixin,
#         LearningRateSchedule,
#     ],
# )

# TODO: how to pass polyak tau
config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=ENV_ID,
    env_config={
        'initial_eff': 0,
        'n_cells': 256,
        'wavelength': 1100,
        'desired_angle': 70
    }
).training(
    model={
        'custom_model': 'model'
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    dueling=True,
    lr=0.001,
    gamma=0.99,
    train_batch_size=BATCH_SIZE
).resources(
    num_gpus=NUM_GPUS
).rollouts(
    horizon=128,
    num_rollout_workers=NUM_ROLLOUT_WORKERS,
    num_envs_per_worker=1,
    rollout_fragment_length=int(BATCH_SIZE / NUM_ROLLOUT_WORKERS),
    # rollout_fragment_length: Divide episodes into fragments of this many steps
    # each during rollouts. Sample batches of this size are collected from
    # rollout workers and combined into a larger batch of `train_batch_size`
    # for learning. For example, given rollout_fragment_length=100 and
    # train_batch_size=1000: 1. RLlib collects 10 fragments of 100 steps each
    # from rollout workers. 2. These fragments are concatenated and we
    # perform an epoch of SGD. When using multiple envs per worker,
    # the fragment size is multiplied by `num_envs_per_worker`. This is since
    # we are collecting steps from multiple envs in parallel. For example,
    # if num_envs_per_worker=5, then rollout workers will return experiences
    # in chunks of 5*100 = 500 steps. The dataflow here can vary per
    # algorithm. For example, PPO further divides the train batch into
    # minibatches for multi-epoch SGD.
).exploration(
    explore=True,
    exploration_config={
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 100000,
    }
).evaluation(
    evaluation_duration=10
)

# t

register_env(ENV_ID, lambda c: DummyEnv(**config.env_config))
ModelCatalog.register_custom_model("model", UNet)

# stop = {
#     # "training_iteration": stop_iters,
#     "timesteps_total": 10000000  # 134217728,  # 8192
#     # "episode_reward_mean": stop_reward,
# }

trainer = DQN(config=config)
for i in range(200000):
    trainer.train()
# tune.run(
#     ddppo.DDPPOTrainer,
#     name='pirl-ddppo',
#     config=config,
#     reuse_actors=True,
#     checkpoint_freq=1024,
#     checkpoint_at_end=True,
#     # max_failures=3,
#     local_dir='log',
#     stop=stop
# tune.run(
#     'Model',
# )
