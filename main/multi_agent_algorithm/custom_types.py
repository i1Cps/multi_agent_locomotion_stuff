from flax.training.train_state import TrainState
from flax import struct
from jax import numpy as jp
from typing import Any, Dict

@struct.dataclass
class AgentTransition:
    observation      : jp.ndarray
    action           : jp.ndarray
    raw_actions      : jp.ndarray
    log_probs        : jp.ndarray
    reward           : jp.ndarray
    next_observation : jp.ndarray

@struct.dataclass
class MultiAgentTransition:
    agents                  : Dict[str, AgentTransition]
    done                    : jp.ndarray
    truncation              : jp.ndarray
    critic_observation      : jp.ndarray
    next_critic_observation : jp.ndarray

@struct.dataclass
class RunningMeanStd:
    mean  : jp.ndarray
    var   : jp.ndarray
    count : jp.ndarray

@struct.dataclass
class CustomTrainState(TrainState):
    apply_fn: Dict[str, Any] = struct.field(pytree_node=False)
    normaliser_params: Dict[str, Any]
