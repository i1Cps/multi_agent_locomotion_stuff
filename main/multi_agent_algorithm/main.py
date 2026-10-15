from typing import Callable, Sequence

import jax
import time
import functools

from jax import numpy as jp
from mujoco_playground import wrapper, State

from .import normalise
from .actor import evaluate_policy, unroll
from .custom_types import CustomTrainState
from .logger import log_eval
from .loss_function import ppo_loss_fn
from .network import ActorNetwork, CriticNetwork
from .policy import create_train_state, learn


def train(
    environment,
    rng: int,
    *,

    # NOTE: PPO hyperparameters
    num_timesteps:     int = 50_000_000,
    num_envs:          int = 4096,
    unroll_length:     int = 10,
    batch_size:        int = 512,         # Number of trajectories in each batch
    num_minibatches:   int = 32,
    updates_per_batch: int = 8,
    learning_rate:     float = 0.0003,
    discount:          float = 0.99,
    entropy_cost:      float = 0.005,
    gae_lambda:        float = 0.95,
    clipping_epsilon:  float = 0.2,
    vf_coefficient:    float = 0.5,
    num_evals:         int = 10,
    episode_length:    int = 1000,
    actor_network_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    critic_network_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    checkpoint_fn: Callable[[int, int, CustomTrainState], None] | None = None,
  ):

    steps_per_eval = num_timesteps // num_evals
    steps_per_train_step = batch_size * num_minibatches * unroll_length
    steps_per_epoch = steps_per_eval // steps_per_train_step



    print(f"Welcome to jax rl, the functional way! \n\n \
    We're about to train our agent on {type(environment).__name__} using PPO \n")

    print(
        f"""Training Params: \n
        Training steps                        : {num_timesteps}
        Number of parallel environments       : {num_envs}
        Number of steps per unroll            : {unroll_length}
        Number of trajectories in each batch  : {batch_size}
        Number of mini-batches                : {num_minibatches}
        Number of updates per batch           : {updates_per_batch}
        Number of steps per epoch             : {steps_per_epoch}
        """
    )

    # Measure compilation time
    jit_start_time = time.time()

    # Prepare the parallel training environments.
    env = wrapper.wrap_for_brax_training(
        environment,
        episode_length   = episode_length,
        action_repeat    = 1,
        randomization_fn = None,
        full_reset       = False,
    )
    reset_fn = jax.jit(env.reset)

    # Prepare keys
    key = jax.random.PRNGKey(rng)
    key, env_key, train_state_key, epoch_key = jax.random.split(key, 4)
    env_keys = jax.random.split(env_key, num_envs)
    env_states = reset_fn(env_keys)

    agent_action_size = env.action_size//2

    # Create networks
    actor_network = ActorNetwork(tuple(actor_network_layer_sizes), agent_action_size)
    critic_network = CriticNetwork(tuple(critic_network_layer_sizes))

    # Use one train state for simplicity
    print(env.observation_size)
    train_state = create_train_state(
        actor_network,
        critic_network,
        env.observation_size["actor"],
        env.observation_size["critic"],
        learning_rate,
        train_state_key,
    )

    loss_fn = functools.partial(
        ppo_loss_fn,
        entropy_cost     = entropy_cost,
        discount         = discount,
        gae_lambda       = gae_lambda,
        clipping_epsilon = clipping_epsilon,
        vf_coefficient   = vf_coefficient
    )

    def train_step(carry, tmp):
        train_state, env_states, key = carry
        key, learn_key, rollout_key = jax.random.split(key, 3)

        def get_trajectories(carry,_):
            env_states, rollout_key = carry

            trajectories, next_states, rollout_key, = unroll(
                train_state,
                env_states,
                rollout_key,
                env,
                unroll_length
            )

            return (next_states, rollout_key), trajectories


        # First: Collect our desired amount of trajectories (batch_size * num_minibatches // num_envs)
        (next_states, _), batched_trajectories = jax.lax.scan(
            f      = get_trajectories, 
            xs     = None, 
            init   = (env_states, rollout_key), 
            length = (batch_size * num_minibatches) // num_envs
        )

        # NOTE: `batched_trajectories` is of shape ~ [(batch_size * num_minibatches) // num_envs, unroll_length, num_envs, data]

        # Swap "unroll_length" and "num_envs" dims around -> [(batch_size * num_minibatches) // num_envs, num_envs, unroll_length, data]
        batched_trajectories = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 1, 2), batched_trajectories)

        # Now squash together the first two dims -> [(batch_size * num_minibatches)], unroll_length, data]
        batched_trajectories = jax.tree_util.tree_map(lambda x: jp.reshape(x, (-1,) + x.shape[2:]), batched_trajectories)

        # After collecting all our data, update the normalisation statistics
        # Red agent actor observation
        red_actor_normaliser_params = normalise.update_running_mean_std(
            train_state.normaliser_params["actor"]["red"],
            batched_trajectories.agents["red"].observation
        )

        blue_actor_normaliser_params = normalise.update_running_mean_std(
            train_state.normaliser_params["actor"]["blue"],
            batched_trajectories.agents["blue"].observation
        )

        critic_normaliser_params = normalise.update_running_mean_std(
            train_state.normaliser_params["critic"],
            batched_trajectories.critic_observation
        )

        new_normaliser_params = {
            "actor": {
                "red": red_actor_normaliser_params,
                "blue": blue_actor_normaliser_params,
            },
            "critic": critic_normaliser_params
        }
        train_state = train_state.replace(normaliser_params=new_normaliser_params)

        train_state, _, _ = learn(
            train_state,
            batched_trajectories,
            num_minibatches,
            updates_per_batch,
            loss_fn,
            learn_key,
        )

        return (train_state, next_states, key), None


    # For every eval we compute a training epoch , each training epoch is made up of x steps, where x == `steps_per_epoch` 
    def training_epoch(
        train_state: CustomTrainState,
        env_states   : State,
        key          : jax.Array
    ):

        result, _ = jax.lax.scan(
            f      = train_step,
            init   = (train_state, env_states, key),
            length = steps_per_epoch,
            xs     = None
        )

        train_state, env_states, key = result
        return train_state, env_states, key


    training_epoch = jax.jit(training_epoch)

    jit_end_time = time.time()
    print(f"Time taken to JIT {jit_end_time - jit_start_time}\n")
    train_start_time = time.time()


    environment_steps = 0


    for eval in range(num_evals):
        # After every training epoch, we evaluate the current policy across 128 seeds
        train_state, env_states, epoch_key = training_epoch(
            train_state,
            env_states,
            epoch_key
        )

        environment_steps += steps_per_epoch * steps_per_train_step

        red_mean_reward, blue_mean_reward, mean_episode_length, eval_outcomes = evaluate_policy(
            train_state = train_state,
            env         = env,
            key         = key,
            max_steps   = episode_length
        )

        eval_idx = eval + 1
        log_eval(
            eval_idx=eval_idx,
            num_evals=num_evals,
            environment_steps=environment_steps,
            training_steps=num_timesteps,
            red_mean_reward=red_mean_reward,
            blue_mean_reward=blue_mean_reward,
            mean_episode_length=mean_episode_length,
            mean_episode_duration=mean_episode_length * env.dt,
            eval_outcomes=eval_outcomes,
            train_start_time=train_start_time,
        )
        # Save from the host loop, outside JIT, after each completed evaluation.
        if checkpoint_fn is not None:
            checkpoint_fn(eval_idx, environment_steps, train_state)


    print(f"Final environment step count: {environment_steps}")
    return train_state
