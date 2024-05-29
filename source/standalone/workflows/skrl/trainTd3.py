# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
from datetime import datetime
import torch
import torch.nn as nn

from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer, SkrlVecEnvWrapper, process_skrl_cfg


def main():
    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

 
    # define models (deterministic models) using mixins
    class DeterministicActor(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, self.num_actions),
                                    nn.Tanh())

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

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
    models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
    models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
 

    # instantiate a ReplayMemory as rollout buffer (any memory can be used for this)
    memory_size = experiment_cfg["agent"]["rollouts"]  
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)


    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = TD3_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = TD3(models=models,
                memory=memory,
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
    trainer_cfg = experiment_cfg["trainer"]
    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # train the agent
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()