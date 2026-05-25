# g1_tray_env_cfg.py
#
# Configuración del entorno RL para G1 tray carrying
#
# Basado en locomotion environment de IsaacLab / Unitree RL Lab

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

import tray_rewards as tray_rew


# ============================================================
# Reward Configuration
# ============================================================

class TrayRewardsCfg:

    # --------------------------------------------------------
    # Existing locomotion rewards
    # --------------------------------------------------------

    tracking_lin_vel = RewTerm(
        func="track_lin_vel_xy_exp",
        weight=1.0,
    )

    tracking_ang_vel = RewTerm(
        func="track_ang_vel_z_exp",
        weight=0.5,
    )

    feet_air_time = RewTerm(
        func="feet_air_time",
        weight=0.25,
    )

    # --------------------------------------------------------
    # Tray posture rewards
    # --------------------------------------------------------

    elbow_height = RewTerm(
        func=tray_rew.elbow_height_reward,
        weight=0.4,
        params={
            "target_height": 1.10,
            "sigma": 0.03,
        },
    )

    wrist_extension = RewTerm(
        func=tray_rew.wrist_extension_reward,
        weight=0.25,
        params={
            "target_left_wrist": 0.3,
            "target_right_wrist": 0.3,
            "sigma": 0.05,
        },
    )

    forearm_horizontal = RewTerm(
        func=tray_rew.forearm_horizontal_reward,
        weight=0.35,
        params={
            "sigma": 0.04,
        },
    )

    arm_symmetry = RewTerm(
        func=tray_rew.arm_symmetry_reward,
        weight=0.15,
        params={
            "sigma": 0.03,
        },
    )

    # --------------------------------------------------------
    # Penalties
    # --------------------------------------------------------

    arm_joint_velocity = RewTerm(
        func=tray_rew.arm_joint_velocity_penalty,
        weight=-0.05,
    )

    arm_pose_deviation = RewTerm(
        func=tray_rew.arm_deviation_penalty,
        weight=-0.1,
        params={
            "target_pose": {
                "left_shoulder_pitch_joint_id": 0.4,
                "left_elbow_joint_id": 1.1,
                "left_wrist_pitch_joint_id": 0.3,
                "right_shoulder_pitch_joint_id": 0.4,
                "right_elbow_joint_id": 1.1,
                "right_wrist_pitch_joint_id": 0.3,
            }
        },
    )


# ============================================================
# Environment cfg
# ============================================================

class G1TrayEnvCfg:

    rewards: TrayRewardsCfg = TrayRewardsCfg()

    episode_length_s = 20.0

    control_frequency = 50

    decimation = 4

    enable_debug_vis = True