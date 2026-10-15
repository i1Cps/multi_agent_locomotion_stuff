
import jax
from jax import numpy as jp
from brax import envs
from .custom_types import AgentTransition, CustomTrainState, MultiAgentTransition
from .policy import choose_actions
from functools import partial
import mediapy as media

# Step the parallel environments one time
def parallel_env_step(
    train_state: CustomTrainState,
    env_states  : envs.State,
    key         : jax.Array,
    env         : envs.Env
):

    next_key, current_key = jax.random.split(key, 2)

    # actions
    agent_outputs = choose_actions(
        raw_observations = env_states.obs,
        train_state  = train_state,
        key          = current_key,
    )

    red_agent_outputs = agent_outputs["red"]
    blue_agent_outputs = agent_outputs["blue"]

    actions = {
        "red": red_agent_outputs["actions"],
        "blue": blue_agent_outputs["actions"]
    }
    raw_actions = {
        "red": red_agent_outputs["raw_actions"],
        "blue": blue_agent_outputs["raw_actions"]
    }
    log_probs = {
        "red": red_agent_outputs["log_probs"],
        "blue": blue_agent_outputs["log_probs"],
    }

    next_env_states = env.step(env_states, actions)

    # Prep the agent transitions
    red_agent_transtion = AgentTransition(
        observation=env_states.obs["actor"]["red"],
        action= actions["red"],
        raw_actions = raw_actions["red"],
        log_probs = log_probs["red"],
        # For now we are using teh env.info to store rewards, we'll update this soon using our custom env wrapper
        reward = next_env_states.info["agent_rewards"]["red"],
        next_observation=next_env_states.obs["actor"]["red"]
    )

    blue_agent_transtion = AgentTransition(
        observation=env_states.obs["actor"]["blue"],
        action= actions["blue"],
        raw_actions = raw_actions["blue"],
        log_probs = log_probs["blue"],
        # For now we are using teh env.info to store rewards, we'll update this soon using our custom env wrapper
        reward = next_env_states.info["agent_rewards"]["blue"],
        next_observation=next_env_states.obs["actor"]["blue"]
    )


    return (
        MultiAgentTransition(
            agents = {
                "red": red_agent_transtion,
                "blue": blue_agent_transtion,
            },
            done = next_env_states.done,
            truncation = next_env_states.info["truncation"],
            critic_observation=env_states.obs["critic"],
            next_critic_observation=next_env_states.obs["critic"]

        ),
        next_env_states,
        next_key
    )


# Returns a batch of trajectories of shape ~ (unroll_length, num_envs, data)
def unroll(train_state, env_states, key, env, unroll_length):
    # Scan function that follows haskell like signature
    def step_func(carry, _):
        env_states, key = carry
        transition, next_env_states, next_key = parallel_env_step(train_state, env_states, key, env)
        return (next_env_states, next_key), transition

    (next_states, next_key), generated_transitions = jax.lax.scan(
        f      = step_func,
        xs     = None,
        init   = (env_states, key),
        length = unroll_length
    )
    return generated_transitions, next_states, next_key 

# Return first-episode rewards, mean length, and outcome counts for evaluation.
@partial(jax.jit, static_argnames=("env", "num_envs", "max_steps"))
def evaluate_policy(
    train_state : CustomTrainState,
    env         : envs.Env,
    key         : jax.Array,
    num_envs    : int  = 128,
    max_steps   : int  = 500
):

    env_keys   = jax.random.split(key, num_envs)
    env_states = env.reset(env_keys)
    finished   = jp.zeros(num_envs, dtype=jp.bool)
    contact = jp.zeros(num_envs, dtype=jp.bool)
    episode_lengths = jp.zeros(num_envs, dtype=jp.int32)
    outcomes = {
        name: jp.array(0, dtype=jp.int32)
        for name in ("red", "blue", "both_failed", "timeout")
    }

    def step_fn(carry, _):
        env_states, finished, episode_lengths, outcomes, contact = carry

        agent_outputs = choose_actions(
            raw_observations = env_states.obs,
            train_state  = train_state,
            key          = key,
            deterministic = True,
        )

        actions = {
            "red": agent_outputs["red"]["actions"],
            "blue": agent_outputs["blue"]["actions"],
        }

        next_env_states = env.step(env_states, actions)
        # Latch contact only within each environment's first evaluation episode.
        contact = contact | (~finished & next_env_states.info["contact"])

        all_rewards = jax.tree_util.tree_map(
            lambda reward: jp.where(finished, 0.0, reward),
            next_env_states.info["agent_rewards"],
        )
        # Count the terminal step, but not steps after the first episode ends.
        episode_lengths = episode_lengths + (~finished).astype(jp.int32)
        newly_finished = jp.logical_and(~finished, next_env_states.done)
        red_failed = next_env_states.info["agent_terminated"]["red"]
        blue_failed = next_env_states.info["agent_terminated"]["blue"]
        events = {
            "red": newly_finished & blue_failed & ~red_failed,
            "blue": newly_finished & red_failed & ~blue_failed,
            "both_failed": newly_finished & red_failed & blue_failed,
            "timeout": newly_finished & ~(red_failed | blue_failed),
        }
        outcomes = {
            name: count + jp.sum(events[name]) for name, count in outcomes.items()
        }
        finished    = jp.logical_or(next_env_states.done, finished)

        return (next_env_states, finished, episode_lengths, outcomes, contact), all_rewards

    (_, finished, episode_lengths, outcomes, contact), rewards = jax.lax.scan(
        f      = step_fn,
        init   = (env_states, finished, episode_lengths, outcomes, contact),
        xs     = None,
        length = max_steps,
    )

    red_mean_reward = jp.mean(jp.sum(rewards["red"], axis=0))
    blue_mean_reward = jp.mean(jp.sum(rewards["blue"], axis=0))
    mean_episode_length = jp.mean(episode_lengths)
    # Episodes still running at the evaluation horizon count as timeouts too.
    outcomes["timeout"] = outcomes["timeout"] + jp.sum(~finished)
    outcomes["contact"] = jp.sum(contact)
    return red_mean_reward, blue_mean_reward, mean_episode_length, outcomes


def visualise_policy(
    actor_state : CustomTrainState, 
    env         : envs.Env, 
    key         : jax.Array
):

    rollout_length = 1000

    def do_rollout(state):
        empty_data = state.data.__class__(
            **{field: None for field in state.data.__annotations__}
        )
        empty_trajectory_state = state.__class__(
            **{field: None for field in state.__annotations__}
        ).replace(data=empty_data)

        def step_fn(state, _):
            agent_outputs = choose_actions(
                train_state  = actor_state,
                raw_observations = state.obs,
                key          = key,
                deterministic = True,
            )
            actions = {
                "red": agent_outputs["red"]["actions"],
                "blue": agent_outputs["blue"]["actions"],
            }
            next_state = env.step(state, actions)
            render_state = empty_trajectory_state.tree_replace({
                "data.qpos": next_state.data.qpos,
                "data.qvel": next_state.data.qvel,
                "data.time": next_state.data.time,
                "data.ctrl": next_state.data.ctrl,
                "data.mocap_pos": next_state.data.mocap_pos,
                "data.mocap_quat": next_state.data.mocap_quat,
                "data.xfrc_applied": next_state.data.xfrc_applied,
            })
            return next_state, render_state

        _, traj = jax.lax.scan(
            f      = step_fn,
            init   = state,
            length = rollout_length,
            xs     = None
        )
        return traj

    reset_state = jax.jit(env.reset)(key)
    rollout     = jax.device_get(jax.jit(do_rollout)(reset_state))

    # Extract the PyTree to a list
    rollout_list = [jax.tree.map(lambda x, j=j: x[j], rollout) for j in range(rollout_length)]

    # Render and save the rollout.
    render_every = 2
    fps          = 1.0 / env.dt / render_every
    print(f"FPS for rendering: {fps}")

    final_rollout = rollout_list[::render_every]

    # Render the frames from each collected state in the rollout
    frames       = env.render(
        final_rollout,
        width        = 640,
        height       = 480,
        camera       = "red_humanoid_camera",
    )
    media.write_video("visual.mp4", frames, fps=fps)
    print("Saved rollout to visual.mp4")
