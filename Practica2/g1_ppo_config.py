# g1_tray_ppo_cfg.py
#
# PPO fine-tuning config para tray carrying

from __future__ import annotations


class G1TrayPPORunnerCfg:

    experiment_name = "g1_tray_carrying"

    max_iterations = 2000

    save_interval = 250

    empirical_normalization = True

    # ========================================================
    # Resume from locomotion checkpoint
    # ========================================================

    resume = True

    load_run = "g1_walking_policy"

    load_checkpoint = "model_10000.pt"

    # ========================================================
    # PPO
    # ========================================================

    class policy:

        init_noise_std = 0.5

        actor_hidden_dims = [512, 256, 128]

        critic_hidden_dims = [512, 256, 128]

        activation = "elu"

    class algorithm:

        value_loss_coef = 1.0

        use_clipped_value_loss = True

        clip_param = 0.2

        entropy_coef = 0.005

        num_learning_epochs = 5

        num_mini_batches = 4

        learning_rate = 1e-4

        schedule = "adaptive"

        gamma = 0.99

        lam = 0.95

        desired_kl = 0.01

        max_grad_norm = 1.0