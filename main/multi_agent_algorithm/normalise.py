
from jax import numpy as jp
from .custom_types import RunningMeanStd


def init_running_mean_std(obs_shape, eps: float = 1e-4) -> RunningMeanStd:
    return RunningMeanStd(
        var   = jp.ones(obs_shape),
        mean  = jp.zeros(obs_shape),
        count = jp.array(eps, dtype=jp.float32),
    )


def update_running_mean_std(
    rms: RunningMeanStd,
    batch: jp.ndarray,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
) -> RunningMeanStd:
    flat_batch = batch.reshape((-1, batch.shape[-1]))
    batch_var   = jp.var(flat_batch, axis=0)
    batch_mean  = jp.mean(flat_batch, axis=0)
    batch_count = flat_batch.shape[0]

    delta = batch_mean - rms.mean
    total_count = rms.count + batch_count

    # Quick stats, do a refresher on variance, mean, std's
    new_mean = rms.mean + delta * batch_count / total_count

    # Use the parallel variance formula
    # Ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    old_sum_squared   = rms.var * rms.count
    new_sum_squared   = batch_var * batch_count
    total_sum_squared = old_sum_squared + new_sum_squared + (delta * delta) * rms.count * batch_count / total_count
    new_var = total_sum_squared / total_count

    # Clip std to avoid outlier effects
    new_std = jp.sqrt(new_var + 1e-8)
    new_std = jp.clip(new_std, std_min_value, std_max_value)
    new_var = new_std * new_std

    return rms.replace(mean=new_mean, var=new_var, count=total_count)


def normalize_obs(
    obs: jp.ndarray, 
    rms: RunningMeanStd, 
    eps: float = 1e-8, 
) -> jp.ndarray:
    obs = (obs - rms.mean) / jp.sqrt(rms.var + eps)
    return obs
