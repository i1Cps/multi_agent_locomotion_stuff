import jax
from jax import numpy as jp
from .normalise import normalize_obs
import distrax


def gae_fn(
    rewards,
    values,
    bootstrap_value,
    termination,
    truncation,
    lambda_,
    discount
):
    truncation_mask = 1 - truncation
    termination_mask = 1 - termination

    next_values = jp.concatenate([values[1:],  jp.expand_dims(bootstrap_value, 0)], axis=0)

    deltas = rewards + discount * (1 - termination) * next_values - values
    deltas *= truncation_mask

    advantage = jp.zeros_like(bootstrap_value)

    def advnatage_calc_fn(advantage_carry, xs):
        delta, trunc_mask, termi_mask = xs
        advantage = delta + discount * termi_mask * lambda_ * trunc_mask * advantage_carry
        return advantage, advantage

    _, advantage = jax.lax.scan(
        f       = advnatage_calc_fn,
        xs      = (deltas, truncation_mask, termination_mask),
        init    = advantage,
        reverse = True
    )

    vs = advantage + values

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantage)

def actor_loss_fn(
    train_state,
    params,
    obs,
    raw_actions,
    log_probs,
    action_log_std,
    advantage,
    rng,
    entropy_cost,
    clipping_epsilon
):
    # Sample a distribution using the current policy
    mean = train_state.apply_fn["actor"](params, obs)
    std = jp.broadcast_to(jp.exp(action_log_std), mean.shape)
    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    # Calculate the log probabilities in raw space then use a jacobian correction to transfer them to tanh space
    new_log_probs_raw = dist.log_prob(raw_actions)
    log_det = 2.0 * (jp.log(2.0) - raw_actions - jax.nn.softplus(-2.0 * raw_actions))
    new_log_probs = new_log_probs_raw - jp.sum(log_det, axis=-1)

    # Old log probs are from the collected data 
    old_log_probs = log_probs

    # Prob ratio calculation, basic PPO formulation
    prob_ratio = jp.exp(new_log_probs - old_log_probs)

    surragote_loss1 = advantage * prob_ratio
    surragote_loss2 = advantage * (jp.clip(prob_ratio, 1 - clipping_epsilon, 1 + clipping_epsilon))
    policy_loss     = -jp.mean(jp.minimum(surragote_loss1, surragote_loss2))
    # Entropy cost encourages exploration
    # Calculate the entropy in raw space then use a jacobian correction to transfer it to tanh space
    entropy_raw = dist.entropy()
    raw_actions = dist.sample(seed=rng)
    log_det = 2.0 * (jp.log(2.0) - raw_actions - jax.nn.softplus(-2.0 * raw_actions))
    entropy = jp.mean(entropy_raw + jp.sum(log_det, axis=-1))
    entropy_loss = entropy_cost * -entropy

    return policy_loss, entropy_loss


def ppo_loss_fn(
    params, # Directly pass params as the first arg, jax.value_and_grad differnetiates respective to the first argument of the function you pass it (in def learn)
    data,
    train_state,
    entropy_cost,
    clipping_epsilon,
    gae_lambda,
    vf_coefficient,
    discount,
    rng,
):
    # NOTE: TEST DIM SHAPES FOR EVERYTHING HERE!!!!!!!!
    # At this point `data` is a single mini-batch in the shape: [Individual batch, Time]

    # Use tree map since we are not working with single jp.ndarrays
    data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x,0,1), data)

    # Normalize observations for all three network passes
    red_obs = normalize_obs(data.agents["red"].observation, train_state.normaliser_params["actor"]["red"])
    blue_obs = normalize_obs(data.agents["blue"].observation, train_state.normaliser_params["actor"]["blue"])
    critic_obs = normalize_obs(data.critic_observation, train_state.normaliser_params["critic"])

    # Remember our critic network has two heads one for each agent
    # values = jp.squeeze(train_state.apply_fn["critic"](params["critic"], critic_obs), axis=-1)
    values = jp.squeeze(train_state.apply_fn["critic"](params["critic"], critic_obs))

    # Get the last observations for the critic in the trajectory
    final_critic_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_critic_observation)

    # normalise that too
    final_critic_obs = normalize_obs(final_critic_obs, train_state.normaliser_params["critic"])
    
    # Remember our critic network has two heads one for each agent
    # bootstrap_value = jp.squeeze(train_state.apply_fn["critic"](params["critic"], final_critic_obs), axis=-1)
    bootstrap_value = jp.squeeze(train_state.apply_fn["critic"](params["critic"], final_critic_obs))

    termination = data.done * (1 - data.truncation)

    red_rewards = data.agents["red"].reward
    blue_rewards = data.agents["blue"].reward # [10,512] [unroll, batchsize]

    rewards = jp.stack([red_rewards, blue_rewards], axis=-1)

    termination = jp.expand_dims(termination, axis=-1) # [10,512,1]
    truncation = jp.expand_dims(data.truncation, axis=-1) # [10,512,1]

    red_action_log_std = params["action_log_std"]["red"]
    blue_action_log_std = params["action_log_std"]["blue"]


    returns, advantage = gae_fn(
        rewards, # [unroll, batch size, num_agents]
        values, # [unroll, batch size, num_agents]
        bootstrap_value, # [batch size, num_agents]
        termination,
        truncation,
        gae_lambda,
        discount,
    )

    # Normalise advantages for a more stable policy gradient
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    #NOTE: Calculate the policy loss
    red_policy_loss, red_entropy_loss = actor_loss_fn(
        train_state, 
        params["actors"]["red"], 
        red_obs, 
        data.agents["red"].raw_actions, 
        data.agents["red"].log_probs, 
        red_action_log_std,
        advantage[...,0], 
        rng, 
        entropy_cost, 
        clipping_epsilon 
    )

    blue_policy_loss, blue_entropy_loss = actor_loss_fn(
        train_state, 
        params["actors"]["blue"], 
        blue_obs, 
        data.agents["blue"].raw_actions, 
        data.agents["blue"].log_probs, 
        blue_action_log_std,
        advantage[...,1], 
        rng, 
        entropy_cost, 
        clipping_epsilon 
    )

    red_actor_loss = red_policy_loss + red_entropy_loss
    blue_actor_loss = blue_policy_loss + blue_entropy_loss

    # NOTE: Calculate the value loss
    value_error = returns - values
    value_loss = jp.mean(value_error * value_error) * 0.5 * vf_coefficient 

    # Sum the loss for jax_and_grad
    return red_actor_loss + blue_actor_loss+ value_loss 
