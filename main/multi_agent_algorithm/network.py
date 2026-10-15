from jax import numpy as jp
from flax import linen as nn
from typing import Callable, Sequence


# This network corresponds to our value function, it estimates the expected return for a state
class CriticNetwork(nn.Module):
    h_dims      : Sequence[int]
    activate_fn : Callable = nn.relu

    @nn.compact
    def __call__(self, state):
        for h in self.h_dims:
            state = nn.Dense(h)(state)
            state = self.activate_fn(state)

        # Final output layer
        state = nn.Dense(2)(state)
        return state


# This network corresponds to our policy, it returns parameters for an action distribution 
class ActorNetwork(nn.Module):
    h_dims       : Sequence[int]
    n_actions    : int
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, state):
        for h in self.h_dims:
            state = nn.Dense(h)(state)
            state = self.activation_fn(state)

        # Output the mean action for each action dimension.
        state = nn.Dense(self.n_actions)(state)
        return state
