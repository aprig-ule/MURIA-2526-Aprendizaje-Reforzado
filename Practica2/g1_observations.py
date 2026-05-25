# g1_tray_observations.py
#
# Observaciones adicionales para tray carrying

from __future__ import annotations

import torch


def arm_joint_positions(robot):

    joint_pos = robot.data.joint_pos

    return torch.stack(
        [
            joint_pos[:, robot.left_shoulder_pitch_joint_id],
            joint_pos[:, robot.left_elbow_joint_id],
            joint_pos[:, robot.left_wrist_pitch_joint_id],
            joint_pos[:, robot.right_shoulder_pitch_joint_id],
            joint_pos[:, robot.right_elbow_joint_id],
            joint_pos[:, robot.right_wrist_pitch_joint_id],
        ],
        dim=-1,
    )


def arm_joint_velocities(robot):

    joint_vel = robot.data.joint_vel

    return torch.stack(
        [
            joint_vel[:, robot.left_shoulder_pitch_joint_id],
            joint_vel[:, robot.left_elbow_joint_id],
            joint_vel[:, robot.left_wrist_pitch_joint_id],
            joint_vel[:, robot.right_shoulder_pitch_joint_id],
            joint_vel[:, robot.right_elbow_joint_id],
            joint_vel[:, robot.right_wrist_pitch_joint_id],
        ],
        dim=-1,
    )


def elbow_heights(robot):

    left_elbow = robot.data.body_pos_w[:, robot.left_elbow_body_id]
    right_elbow = robot.data.body_pos_w[:, robot.right_elbow_body_id]

    return torch.stack(
        [
            left_elbow[:, 2],
            right_elbow[:, 2],
        ],
        dim=-1,
    )