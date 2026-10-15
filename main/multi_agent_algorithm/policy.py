import jax
import optax
from jax import numpy as jp

from .custom_types import CustomTrainState
from .normalise import init_running_mean_std, normalize_obs
import distrax

# This creates a train state containing network parameters, optimizer state, and the network apply functions
# for both agents
def create_train_state(
    actor_network,
    critic_network, 
    actor_input_dims, 
    critic_input_dims,
    learning_rate,
    key
):
    actor_key1, actor_key2, critic_key = jax.random.split(key, 3)

    actor_dummy_obs = jp.zeros((1, actor_input_dims))
    critic_dummy_obs = jp.zeros((1, critic_input_dims))

    actor1_variables  = actor_network.init(actor_key1, actor_dummy_obs)
    actor2_variables  = actor_network.init(actor_key2, actor_dummy_obs)
    critic_variables  = critic_network.init(critic_key, critic_dummy_obs)

    # We will model the params object as a dict containing both the actor and critic params, 
    # and use their respective apply function manually
    params = {
        "actors": {
            "red": actor1_variables,
            "blue": actor2_variables,
        },
        "critic" : critic_variables,
        "action_log_std" : {
            "red": jp.zeros((actor_network.n_actions,)),
            "blue": jp.zeros((actor_network.n_actions,)),
        }
    }

    # TODO: Add configurable max_grad_norm clipping before the Adam optimizer.
    ppo_train_state = CustomTrainState.create(
        apply_fn  = {"actor": actor_network.apply, "critic": critic_network.apply},
        params    = params,
        tx        = optax.adam(learning_rate),
        normaliser_params = {
            "actor": {
                "red": init_running_mean_std(actor_dummy_obs.shape[1:]),
                "blue": init_running_mean_std(actor_dummy_obs.shape[1:]),
            },
            "critic": init_running_mean_std(critic_dummy_obs.shape[1:]),
        },
    )

    return ppo_train_state

# Calculate actions for both Red and Blue agents
def choose_actions(
    train_state: CustomTrainState,
    raw_observations: jp.ndarray,
    key: jax.Array,
    deterministic: bool = False,
) -> dict[str, dict[str, jax.Array]]:

    # Normalize observations before the policy forward pass using running stats.

    # Quick key split
    red_key, blue_key = jax.random.split(key)
    
    # Red agent
    red_actor_raw_obs = raw_observations["actor"]["red"]
    red_actor_obs = normalize_obs(red_actor_raw_obs, train_state.normaliser_params["actor"]["red"])

    red_action_mean = train_state.apply_fn["actor"](train_state.params["actors"]["red"], red_actor_obs)
    red_action_std = jp.broadcast_to(jp.exp(train_state.params["action_log_std"]["red"]), red_action_mean.shape)
    red_action_dist = distrax.MultivariateNormalDiag(loc=red_action_mean, scale_diag=red_action_std)

    # Sample the actions from the distribution, sample in pre-tanh space to avoid atanh instability
    red_raw_actions = red_action_dist.sample(seed=red_key) if not deterministic else red_action_mean 

    # Calculate the log probabilities in raw space then use a jacobian correction to transfer them to tanh space
    red_action_log_probs_raw = red_action_dist.log_prob(red_raw_actions)
    red_action_log_det = 2.0 * (jp.log(2.0) - red_raw_actions - jax.nn.softplus(-2.0 * red_raw_actions))
    red_action_log_prob = red_action_log_probs_raw - jp.sum(red_action_log_det, axis=-1)

    # Postprocess into env actions in [-1, 1].
    red_actions = jp.tanh(red_raw_actions)


    # Blue agent
    blue_actor_raw_obs = raw_observations["actor"]["blue"]
    blue_actor_obs = normalize_obs(blue_actor_raw_obs, train_state.normaliser_params["actor"]["blue"])

    blue_action_mean = train_state.apply_fn["actor"](train_state.params["actors"]["blue"], blue_actor_obs)
    blue_action_std = jp.broadcast_to(jp.exp(train_state.params["action_log_std"]["blue"]), blue_action_mean.shape)
    blue_action_dist = distrax.MultivariateNormalDiag(loc=blue_action_mean, scale_diag=blue_action_std)

    # Sample the actions from the distribution, sample in pre-tanh space to avoid atanh instability
    blue_raw_actions = blue_action_dist.sample(seed=blue_key) if not deterministic else blue_action_mean 

    # Calculate the log probabilities in raw space then use a jacobian correction to transfer them to tanh space
    blue_action_log_probs_raw = blue_action_dist.log_prob(blue_raw_actions)
    blue_action_log_det = 2.0 * (jp.log(2.0) - blue_raw_actions - jax.nn.softplus(-2.0 * blue_raw_actions))
    blue_action_log_prob = blue_action_log_probs_raw - jp.sum(blue_action_log_det, axis=-1)

    # Postprocess into env actions in [-1, 1].
    blue_actions = jp.tanh(blue_raw_actions)

    return {
        "red": {
            "actions": red_actions,
            "log_probs": red_action_log_prob,
            "raw_actions": red_raw_actions,
        },
        "blue": {
            "actions": blue_actions,
            "log_probs": blue_action_log_prob,
            "raw_actions": blue_raw_actions,
        }
    }


def learn(
    train_state       : CustomTrainState,
    trajectories,
    num_minibatches   : int,
    updates_per_batch : int,
    loss_fn,
    rng               : jax.Array
):

    def minibatch_step(carry, mini_batch):
        # Take a minibatch of trajectories and send it off to the loss function
        train_state, minibatch_rng = carry

        minibatch_rng, loss_rng = jax.random.split(minibatch_rng)

        loss, gradients = jax.value_and_grad(loss_fn, argnums=0)(
            train_state.params,
            mini_batch,
            train_state,
            rng = loss_rng
        )

        # Update the network weights
        updated_train_state = train_state.apply_gradients(grads=gradients)
        return (updated_train_state, minibatch_rng), loss

    # Learning epoch
    def run_epoch(carry, _):
        train_state, epoch_rng = carry

        epoch_rng, shuffle_rng, minibatch_rng= jax.random.split(epoch_rng, 3)

        # Shuffle and batch trajectories into mini batches
        def shuffle_and_batch_data(x: jp.ndarray):
            x = jax.random.permutation(shuffle_rng, x)
            x = jp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        minibatches = jax.tree_util.tree_map(shuffle_and_batch_data, trajectories)

        (new_train_state, _), minibatch_losses = jax.lax.scan(
            f      = minibatch_step,
            xs     = minibatches,
            init   = (train_state, minibatch_rng),
            length = num_minibatches,
        )

        return (new_train_state, epoch_rng), minibatch_losses

    # Perform a learning epoch x times on this batch of data
    (train_state, rng), epoch_losses = jax.lax.scan(
        f      = run_epoch,
        xs     = None,
        init   = (train_state, rng),
        length = updates_per_batch
    )

    return train_state, rng, epoch_losses
