# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the aic task (UR5e assembly with task board).

Includes:
- Command-tracking rewards with exponential / tanh kernels (inspired by the
  gear-assembly deploy environment).
- A sparse reaching bonus.
- Smoothness and safety penalties (torques, joint acceleration, action rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Command-pose tracking (position)
# ---------------------------------------------------------------------------


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward position tracking using a Gaussian (exponential) kernel.

    Unlike tanh, this kernel drops off very steeply beyond *sigma*, providing
    almost no gradient far from the target while giving a strong signal
    close-in — ideal for fine insertion tasks.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    dist_sq = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


# ---------------------------------------------------------------------------
# Command-pose tracking (orientation)
# ---------------------------------------------------------------------------


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize orientation error (shortest-path angular distance in rad)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward orientation tracking using the tanh kernel.

    Maps the angular error through ``1 - tanh(error / std)`` so that perfectly
    aligned orientations yield 1.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ang_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


# ---------------------------------------------------------------------------
# Sparse reaching bonus
# ---------------------------------------------------------------------------


def ee_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse +1 bonus when the EE is within *threshold* (m) of the command position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


# ---------------------------------------------------------------------------
# Smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques (L2 squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear acceleration of selected bodies (encourages gentle motion)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1
    )


# ---------------------------------------------------------------------------
# Port-targeting rewards for cable-insertion task
# ---------------------------------------------------------------------------

# Cache to avoid calling find_bodies every step.
_ee_body_idx: dict[str, int] = {}

# sfp_port_0_link_entrance offset in NIC card local frame.
# Derived from USD prim hierarchy: entrance world pos minus card root, then
# rotated back through the card's fixed init rotation R=[[-1,0,0],[0,0,-1],[0,-1,0]].
_SFP_PORT_0_LOCAL_OFFSET = torch.tensor([0.01295, -0.07737, 0.00556])


def _get_ee_pos(robot: Articulation, body_name: str) -> torch.Tensor:
    """Return EE world position; caches the body index on first call."""
    if body_name not in _ee_body_idx:
        idx, _ = robot.find_bodies(body_name)
        _ee_body_idx[body_name] = idx[0]
    return robot.data.body_pos_w[:, _ee_body_idx[body_name], :]


def _get_sfp_entrance_pos(port: RigidObject) -> torch.Tensor:
    """World position of sfp_port_0_link_entrance, robust to NIC card pose changes."""
    offset = _SFP_PORT_0_LOCAL_OFFSET.to(port.data.root_pos_w.device)
    offset_b = offset.unsqueeze(0).expand(port.data.root_pos_w.shape[0], -1)
    return port.data.root_pos_w[:, :3] + quat_apply(port.data.root_quat_w, offset_b)


def dist_to_port(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    ee_body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """L2 distance from EE to port origin. Use with a negative weight."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_pos = _get_ee_pos(robot, ee_body_name)
    port_pos = _get_sfp_entrance_pos(port)
    return torch.norm(ee_pos - port_pos, dim=1)


def dist_to_port_tanh(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    std: float = 0.05,
    ee_body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Tanh reward in [0, 1] that peaks at 1.0 when EE is on the port."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_pos = _get_ee_pos(robot, ee_body_name)
    port_pos = _get_sfp_entrance_pos(port)
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def dist_to_port_exp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    ee_body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Exp-shaped reward: exp(-dist/0.15). Active within ~0.3m, peaks at 1.0 on port."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_pos = _get_ee_pos(robot, ee_body_name)
    port_pos = _get_sfp_entrance_pos(port)
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return torch.exp(-dist / 0.15)


def dist_to_port_exp_fine(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    ee_body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Tight exp reward: exp(-dist/0.02). Active within ~6cm, near 1.0 at 1cm."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_pos = _get_ee_pos(robot, ee_body_name)
    port_pos = _get_sfp_entrance_pos(port)
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return torch.exp(-dist / 0.02)


def insertion_success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    threshold: float = 0.01,
    ee_body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Sparse +1 when EE is within *threshold* metres of port origin."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_pos = _get_ee_pos(robot, ee_body_name)
    port_pos = _get_sfp_entrance_pos(port)
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return (dist < threshold).float()
