"""Standalone sanity check -- no hardware, no other files needed.

Loads the trained gear-mesh policy using rl_games' own Runner +
create_player() path (same architecture-loading approach as Isaac Lab's
play.py), using the AGENT_PARAMS below -- a direct transcription of your
actual training yaml. Feeds it random observations and prints the actions.

This only proves the checkpoint loads and the network runs end to end
(correct architecture, correct obs/action dims). It does NOT validate that
the actions are meaningful -- random input can't tell you that.

Install:
    pip install rl-games gym torch --break-system-packages

Run:
    python3 test_random_inference.py
"""

import copy

import numpy as np
import torch
import gym

from rl_games.torch_runner import Runner


# ----------------------------------------------------------------------
# EDIT THESE THREE
# ----------------------------------------------------------------------
CHECKPOINT_PATH = "Factory.pth"   # TODO: path to your .pth checkpoint

OBS_DIM = 19     # matches your GearMesh cfg.observation_space (13 obs terms + 6 prev_actions)
ACTION_DIM = 6   # matches cfg.action_space

DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"

# Your yaml has player.deterministic: False (useful during training eval).
# For an inference sanity check, force the deterministic mean action so
# results are repeatable run to run.
FORCE_DETERMINISTIC = True

# From CtrlCfg in factory_env_cfg.py -- the raw [-1,1] action is scaled by
# these to get a physical per-step delta (meters / radians). This does NOT
# include pos_action_bounds, since that clip is relative to the fixed-asset
# frame and needs a real fingertip/gear pose -- not available with random
# synthetic observations, so it's intentionally left out here. See
# _apply_action() in factory_env.py for the full chain if you need it exactly.
POS_ACTION_THRESHOLD = np.array([0.02, 0.02, 0.02])     # meters
ROT_ACTION_THRESHOLD = np.array([0.097, 0.097, 0.097])  # radians (axis-angle)
EMA_FACTOR = 0.2  # cfg.ctrl.ema_factor


# ----------------------------------------------------------------------
# Your training yaml, transcribed directly -- unmodified aside from
# device/num_actors, which don't matter for inference-only loading.
# ----------------------------------------------------------------------
AGENT_PARAMS = {
    "seed": 0,
    "algo": {"name": "a2c_continuous"},
    "env": {"clip_actions": 1.0},
    "model": {"name": "continuous_a2c_logstd"},
    "network": {
        "name": "actor_critic",
        "separate": False,
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": False,
            }
        },
        "mlp": {
            "units": [512, 128, 64],
            "activation": "elu",
            "d2rl": False,
            "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
        "rnn": {
            "name": "lstm",
            "units": 1024,
            "layers": 2,
            "before_mlp": True,
            "concat_input": True,
            "layer_norm": True,
        },
    },
    "load_checkpoint": False,
    "load_path": "",
    "config": {
        "name": "Factory",
        "device": DEVICE_NAME,
        "device_name": DEVICE_NAME,   # PpoPlayerContinuous reads this key specifically
        "full_experiment_name": "New",
        "env_name": "rlgpu",          # unused -- env_info injected below, no sim env created
        "multi_gpu": False,
        "ppo": True,
        "mixed_precision": True,
        "normalize_input": True,
        "normalize_value": True,
        "value_bootstrap": True,
        "num_actors": 1,              # 1 for inference, regardless of the 128 used in training
        "reward_shaper": {"scale_value": 1.0},
        "normalize_advantage": True,
        "gamma": 0.995,
        "tau": 0.95,
        "learning_rate": 1.0e-4,
        "lr_schedule": "adaptive",
        "schedule_type": "standard",
        "kl_threshold": 0.008,
        "score_to_win": 20000,
        "max_epochs": 200,
        "save_best_after": 10,
        "save_frequency": 100,
        "print_stats": True,
        "grad_norm": 1.0,
        "entropy_coef": 0.0,
        "truncate_grads": True,
        "e_clip": 0.2,
        "horizon_length": 128,
        "minibatch_size": 512,
        "mini_epochs": 4,
        "critic_coef": 2,
        "clip_value": True,
        "seq_length": 128,
        "bounds_loss_coef": 0.0001,
        "central_value_config": {
            "minibatch_size": 512,
            "mini_epochs": 4,
            "learning_rate": 1e-4,
            "lr_schedule": "adaptive",
            "kl_threshold": 0.008,
            "clip_value": True,
            "normalize_input": True,
            "truncate_grads": True,
            "network": {
                "name": "actor_critic",
                "central_value": True,
                "mlp": {
                    "units": [512, 128, 64],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"},
                },
                "rnn": {
                    "name": "lstm",
                    "units": 1024,
                    "layers": 2,
                    "before_mlp": True,
                    "concat_input": True,
                    "layer_norm": True,
                },
            },
        },
        "player": {"deterministic": False},
    },
}


def build_env_info():
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    return {
        "observation_space": obs_space,
        "action_space": action_space,
        "agents": 1,
        "value_size": 1,
    }


def build_agent(checkpoint_path: str):
    """Mirrors play.py's Runner().load(...) / create_player() / restore() / reset()."""
    agent_cfg = {"params": copy.deepcopy(AGENT_PARAMS)}
    agent_cfg["params"]["config"]["env_info"] = build_env_info()

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()

    if FORCE_DETERMINISTIC:
        agent.is_deterministic = True

    return agent


def test_random_inference(agent, num_steps: int = 400000000, done_every: int = 7):
    obs = np.random.uniform(-1.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
    obs = agent.obs_to_torch(obs)

    _ = agent.get_batch_size(obs, 1)  # mirrors play.py -- sets has_batch_dimension correctly
    if agent.is_rnn:
        agent.init_rnn()

    print(f"Loaded policy. is_rnn={agent.is_rnn}, is_deterministic={agent.is_deterministic}, "
          f"device={DEVICE_NAME}")
    print(f"Feeding {num_steps} random observations of dim {OBS_DIM}, "
          f"expecting actions of dim {ACTION_DIM}")
    print(f"Printing physical-unit actions: pos delta in meters, rot delta as axis-angle in radians "
          f"(EMA factor={EMA_FACTOR}, matching _apply_action's scaling chain minus pos_action_bounds,\n"
          f"which needs a real fingertip/gear pose not available with random observations)\n")

    smoothed_action = np.zeros(ACTION_DIM)  # self.actions in the training code

    for step in range(num_steps):
        with torch.inference_mode():
            action = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            action_np = action.detach().cpu().numpy()

            # Exercise the dones-indexed LSTM state reset path too, not just
            # get_action(). This MUST stay inside inference_mode() -- doing
            # it outside raises "Inplace update to inference tensor" since
            # agent.states tensors are created under inference_mode.
            done = (step + 1) % done_every == 0
            if agent.is_rnn and agent.states is not None:
                dones = torch.tensor([done], dtype=torch.bool, device=DEVICE_NAME)
                for s in agent.states:
                    s[:, dones, :] = 0.0

        assert action_np.shape[-1] == ACTION_DIM, (
            f"action dim mismatch: got {action_np.shape[-1]}, expected {ACTION_DIM}"
        )

        # EMA smoothing: self.actions = ema*action + (1-ema)*prev_actions
        smoothed_action = EMA_FACTOR * action_np + (1 - EMA_FACTOR) * smoothed_action

        # Scale to physical units, same split as _apply_action.
        pos_delta_m = smoothed_action[0:3] * POS_ACTION_THRESHOLD          # meters
        rot_delta_rad = smoothed_action[3:6] * ROT_ACTION_THRESHOLD        # radians, axis-angle vector
        rot_angle_deg = np.linalg.norm(rot_delta_rad) * 180.0 / np.pi      # magnitude, for readability

        print(f"[step {step:02d}] raw action (clipped [-1,1]): "
              f"{np.array2string(action_np, precision=4, suppress_small=True)}")
        print(f"           pos delta (m):   {np.array2string(pos_delta_m, precision=4, suppress_small=True)}")
        print(f"           rot delta (rad): {np.array2string(rot_delta_rad, precision=4, suppress_small=True)}"
              f"   (magnitude ~{rot_angle_deg:.2f} deg)")
        if done:
            print("          -- simulated done, LSTM state zeroed --")

        obs = np.random.uniform(-1.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
        obs = agent.obs_to_torch(obs)

    print("\nDone. No shape/loading errors -- plumbing is sound.")
    print("(Random observations only validate loading, not correctness of behavior.)")


if __name__ == "__main__":
    agent = build_agent(CHECKPOINT_PATH)
    test_random_inference(agent)