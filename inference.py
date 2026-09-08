"""
Hardware inference for an Isaac Lab "Factory" (gear-meshing) task,
trained with rl_games PPO using a recurrent (LSTM) continuous policy.

This mirrors the control flow of Isaac Lab's official
`scripts/reinforcement_learning/rl_games/play.py` as closely as possible,
just swapping the simulated env for real hardware I/O:

    runner = Runner(); runner.load(cfg); agent = runner.create_player()
    agent.restore(checkpoint); agent.reset()
    obs = agent.obs_to_torch(obs)
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn: agent.init_rnn()
    ... loop ...
        actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
        # on episode end, per-env dones-indexed zeroing (not a full reset()):
        for s in agent.states:
            s[:, dones, :] = 0.0

WHY Runner + create_player() INSTEAD OF HAND-BUILDING THE NETWORK
  Your policy is a recurrent (LSTM) network with a shared actor/critic
  trunk, LSTM placed *before* the MLP, raw input concatenated into the
  LSTM output, and layer norm on the LSTM output. Reimplementing that
  by hand is easy to get subtly wrong in a way that loads without
  error but produces wrong actions. Runner/create_player() is the same
  code path rl_games itself uses, so the architecture is guaranteed to
  match the checkpoint -- your full training YAML is passed through
  unmodified below.

INSTALL (no Isaac Sim needed for inference)
  pip install rl-games gym torch --break-system-packages

WHAT YOU MUST EDIT (search "TODO")
  - OBS_DIM / ACTION_DIM to match numObservations / numActions
  - CHECKPOINT_PATH
  - get_observation() / send_action() / episode_is_done() for your robot
  - DEVICE_NAME if you don't have a GPU on the control PC
"""

import copy
import time

import numpy as np
import torch
import gym

from rl_games.torch_runner import Runner


# ----------------------------------------------------------------------
# 1. CONFIG -- edit these
# ----------------------------------------------------------------------
CHECKPOINT_PATH = "gear_meshing.pth"   # TODO: path to your .pth checkpoint

OBS_DIM = 19      # TODO: must match cfg.env.numObservations used in training
ACTION_DIM = 6    # TODO: must match cfg.env.numActions used in training

DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"   # TODO: force "cpu" if your control PC has no GPU

CONTROL_HZ = 60.0   # TODO: match the control frequency your policy expects

# Your training yaml sets player.deterministic: False (useful for stochastic
# eval during training). For hardware deployment you almost always want the
# deterministic mean action instead -- override it here rather than trusting
# the training-time default.
FORCE_DETERMINISTIC = True


# ----------------------------------------------------------------------
# 2. YOUR TRAINING YAML, VERBATIM -- this is exactly the `params:` block
#    you provided, unmodified, so there is no risk of a hand-transcription
#    error changing the architecture.
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
        "env_name": "rlgpu",          # unused -- we inject env_info below to skip env creation
        "multi_gpu": False,
        "ppo": True,
        "mixed_precision": True,
        "normalize_input": True,
        "normalize_value": True,
        "value_bootstrap": True,
        "num_actors": 1,
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


# ----------------------------------------------------------------------
# 3. BUILD A FAKE "env_info" SO THE PLAYER SKIPS CREATING A SIM ENV
# ----------------------------------------------------------------------
def build_env_info():
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    # TODO: Isaac Lab Factory tasks normalize actions to [-1, 1]; adjust
    # bounds here if your task's action_space differs.
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


# ----------------------------------------------------------------------
# 4. HARDWARE INTERFACE -- REPLACE THESE FUNCTIONS
# ----------------------------------------------------------------------
def get_observation() -> np.ndarray:
    """
    TODO: Read real sensor / robot state and assemble the observation
    vector in the EXACT same order/units/frame as your sim task's
    compute_observations() -- e.g. joint pos, joint vel, gear pose
    relative to target, fingertip pose, force/torque readings, etc.
    Must return shape (OBS_DIM,), dtype float32.
    """
    raise NotImplementedError("Wire this up to your robot's state/sensor API")


def send_action(action: np.ndarray) -> None:
    """
    TODO: Send the action to the real robot, converting from the
    policy's normalized [-1, 1] output to whatever your controller
    expects. Enforce safety/joint limits here -- the policy itself has
    no built-in safety clamp beyond the [-1, 1] range.
    """
    raise NotImplementedError("Wire this up to your robot's control API")


def episode_is_done() -> bool:
    """
    TODO: return True when the current gear-meshing attempt ends
    (success, failure, or timeout). This drives the dones-indexed LSTM
    state reset below -- get it right or the recurrent state will carry
    "memory" from the previous attempt into the next one.
    """
    return False


# ----------------------------------------------------------------------
# 5. CONTROL LOOP -- mirrors play.py's simulate loop
# ----------------------------------------------------------------------
def run_inference_loop(agent, duration_s: float = None):
    dt = 1.0 / CONTROL_HZ

    obs = get_observation()
    assert obs.shape[-1] == OBS_DIM, f"obs dim mismatch: got {obs.shape[-1]}, expected {OBS_DIM}"
    obs = agent.obs_to_torch(obs)

    # required: enables the has_batch_dimension flag correctly (mirrors play.py)
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print(f"Starting inference loop at {CONTROL_HZ} Hz on device={DEVICE_NAME}, is_rnn={agent.is_rnn}")

    start = time.time()
    step = 0
    try:
        while True:
            loop_start = time.time()

            with torch.inference_mode():
                action = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                action_np = action.detach().cpu().numpy()
                send_action(action_np)

                done = episode_is_done()
                if agent.is_rnn and agent.states is not None:
                    dones = torch.tensor([done], dtype=torch.bool, device=DEVICE_NAME)
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                    if done:
                        print(f"[step {step}] episode boundary -- LSTM state zeroed")

            step += 1
            if duration_s is not None and (time.time() - start) > duration_s:
                break

            obs = get_observation()
            assert obs.shape[-1] == OBS_DIM, f"obs dim mismatch: got {obs.shape[-1]}, expected {OBS_DIM}"
            obs = agent.obs_to_torch(obs)

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif step % 100 == 0:
                print(f"[warn] loop overran by {-sleep_time * 1000:.1f} ms at step {step}")

    except KeyboardInterrupt:
        print("Inference loop stopped by user.")


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    agent = build_agent(CHECKPOINT_PATH)
    run_inference_loop(agent)