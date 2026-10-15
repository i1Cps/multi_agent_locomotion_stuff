import os
import warp as wp
import jax
import json
import datetime
import functools
from pathlib import Path
from orbax import checkpoint as ocp
from mujoco_playground import registry
from main.multi_agent_algorithm.main import train

# Import registry file to initialise all environments
from main import registry as registry_init

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
wp.config.quiet = True


def save_inference_model(train_state, model_path: Path, *, overwrite: bool = False):
    """Save both actors and their observation normalisers for playback"""
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    inference_checkpoint = {
        "actor_params": train_state.params["actors"],
        "actor_normaliser_params": train_state.normaliser_params["actor"],
        "action_log_std": train_state.params["action_log_std"],
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(model_path, inference_checkpoint, force=overwrite)
        checkpointer.wait_until_finished()
    print(f"Saved inference model to {model_path}")


def main():
    """Run training and evaluation for the specified environment"""

    gpu = jax.devices("gpu")[0]
    print(f"JAX using GPU: {gpu.device_kind} (platform={gpu.platform}, id={gpu.id})")

    env_name = "sumo"

    # Load the options for specific environment
    with open(f"main/environment/{env_name}/options.json") as f:
        options = json.load(f)

    ppo_options = options["ppo"]
    mjx_options = options["mjx"]

    # Prepare PPO arguments
    training_params = dict(ppo_options)
    seed = training_params.pop("seed")

    train_fn = functools.partial(train, **training_params)

    # Apply sim options
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = mjx_options["impl"]
    env_cfg["ctrl_dt"] = mjx_options["ctrl_dt"]
    env_cfg["sim_dt"] = mjx_options["sim_dt"]
    env_cfg["episode_length"] = ppo_options["episode_length"]

    env = registry.load(env_name, config=env_cfg)

    print(f"\033[1mmjx options:\033[0m {mjx_options}")
    print(f"\033[1mEnvironment Config:\033[0m\n{env_cfg}")

    print("\033[1mPPO Training Parameters:\033[0m")
    for name, value in sorted(training_params.items()):
        print(f"{name}: {value}")
    print()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"sumo-{timestamp}"
    print(f"Experiment name: {exp_name}\n")

    models_dir = Path(__file__).resolve().parent / "models" / env_name
    run_models_dir = models_dir / exp_name

    # A separate run folder keeps earlier evaluation snapshots intact
    run_models_dir.mkdir(parents=True, exist_ok=False)

    def save_eval_model(eval_idx, environment_steps, train_state):
        model_path = run_models_dir / f"eval_{eval_idx:03d}"
        save_inference_model(train_state, model_path)
        print(f"Checkpoint: eval {eval_idx}, environment steps {environment_steps}")

    # Call main RL training function
    final_trained_state = train_fn(environment=env, rng=seed, checkpoint_fn=save_eval_model)

    save_inference_model(final_trained_state, models_dir / "model", overwrite=True)

    print("Done training.")

if __name__ == "__main__":
    main()
