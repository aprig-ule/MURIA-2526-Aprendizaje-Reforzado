# train_g1_tray.py
#
# Script de entrenamiento

from __future__ import annotations

from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from g1_tray_env_cfg import G1TrayEnvCfg
from g1_ppo_config import G1TrayPPORunnerCfg


def main():

    env_cfg = G1TrayEnvCfg()

    agent_cfg = G1TrayPPORunnerCfg()

    print("Starting G1 tray carrying training...")

    # Aquí iría integración exacta con tu repo:
    #
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    #
    # runner = OnPolicyRunner(
    #     env,
    #     agent_cfg,
    # )
    #
    # runner.learn()


if __name__ == "__main__":
    main()