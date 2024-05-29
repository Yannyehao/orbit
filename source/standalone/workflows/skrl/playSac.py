# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg


def main():
    """Play with skrl agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/modules/skrl.utils.model_instantiators.html
    class StochasticActor(GaussianMixin, Model):
            def __init__(self, observation_space, action_space, device, clip_actions=False,
                        clip_log_std=True, min_log_std=-5, max_log_std=2):
                Model.__init__(self, observation_space, action_space, device)
                GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

                self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.num_actions),
                                        nn.Tanh())
                self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

            def compute(self, inputs, role):
                return self.net(inputs["states"]), self.log_std_parameter, {}

    class Critic(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 1))

        def compute(self, inputs, role):
            return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

    # define the environment action and observation space
    observation_space = env.observation_space
    action_space = env.action_space
    device = env.device

    models = {}
    models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

       # instantiate a ReplayMemory as rollout buffer (any memory can be used for this)
    memory_size = experiment_cfg["agent"]["rollouts"]  
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))


    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = SAC(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # initialize agent
    agent.init()
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            # env stepping
            obs, _, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()