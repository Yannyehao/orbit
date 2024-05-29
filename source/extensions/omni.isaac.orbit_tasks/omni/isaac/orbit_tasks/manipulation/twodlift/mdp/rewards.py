# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def object_is_lifted(
    env: RLTaskEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: RLTaskEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: RLTaskEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    # return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
    return 1 - torch.tanh(distance / std)

def drop_success(
    env: RLTaskEnv,
    tolerance: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward the agent for successfully dropping the object at the target pose within a certain tolerance.
    The target pose is dynamically retrieved from a command associated with the environment.

    Parameters:
        env (RLTaskEnv): The reinforcement learning task environment.
        tolerance (float): The tolerance within which the object must be dropped near the target pose.
        minimal_height (float): The minimum height the object must be above to consider it successfully dropped.
        command_name (str): The name of the command to retrieve the target pose.
        object_cfg (SceneEntityCfg): Configuration for the object, defaulted to an object named "object".

    Returns:
        torch.Tensor: A tensor of rewards, where 1.0 indicates success and 0.0 indicates failure.
    """
    # Extract the object
    object: RigidObject = env.scene[object_cfg.name]

    # Retrieve the target pose from the command manager
    command = env.command_manager.get_command(command_name)
    target_pose = command[:, :3]  

    # Compute the distance between the object's current position and the target pose
    distance = torch.norm(object.data.root_pos_w[:, :3] - target_pose, dim=1)

    # Check if the object is within the tolerance range and under the minimal height
    is_within_tolerance = (distance <= tolerance) & (object.data.root_pos_w[:, 2] < minimal_height)

    # Return reward: 1.0 if within tolerance and under minimal height, otherwise 0.0
    return torch.where(is_within_tolerance, torch.tensor(1.0, device=env.device), torch.tensor(0.0, device=env.device))

def object_release_penalty(
    env: RLTaskEnv,
    release_threshold: float,
    distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    
) -> torch.Tensor:
    """Penalize the agent for not increasing the distance to the object when it should release."""
    # Calculate the current distance to the goal and to the object
    current_goal_distance = object_goal_distance(
        env,
        std=release_threshold,
        minimal_height=0.0,  # Assume minimal_height is not important for release
        command_name="object_pose",  # The name should match the command used for the goal
        object_cfg=object_cfg,
    )
    current_ee_distance = object_ee_distance(
        env,
        std=distance_threshold,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    
    # Check if the object is within the release threshold
    is_within_release_area = current_goal_distance < release_threshold
    # Check if the end-effector is too close to the object when it should release
    is_too_close = current_ee_distance < distance_threshold
    # Penalize if both conditions are met
    penalty = torch.where(is_within_release_area & is_too_close, -1.0, 0.0)
    
    # # Check if the object is within the release threshold
    # if current_goal_distance < release_threshold:
    #     # Check if the end-effector is too close to the object when it should release
    #     if current_ee_distance < distance_threshold:
    #         penalty = -1.0
    #     else:
    #         penalty = 0.0
    # # Check if the end-effector is too close to the object when it should release
    # is_too_close = current_ee_distance < distance_threshold
    # # Penalize if both conditions are met
    # penalty = torch.where(is_within_release_area & is_too_close, -1.0, 0.0)
    
    return penalty
