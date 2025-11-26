import jax
import jax.numpy as jnp
from functools import partial

def crec_sweep(param_grid, schedule, init_state, observable, num_samples=100):
    """
    Sweeps CREC across a grid of parameters.
    
    Args:
      param_grid: [num_points, ...] array of parameter PyTrees
      schedule: sampling schedule (fixed)
      init_state: initial global state (could be batched or reset each run)
      observable: as above
      num_samples: samples per parameter setting
    Returns:
      crec_means: [num_points] array, CREC average at each grid point
      crec_vars: [num_points] array, sample variances
    """
    @partial(jax.jit, static_argnums=(3, 4))
    def single_crec_est(param, key, schedule, observable, num_samples):
        observ_vals, _ = run_block_sampler_with_observable(
            init_state, param, schedule, key, observable, num_samples
        )
        mean = jnp.mean(observ_vals)
        var = jnp.var(observ_vals)
        return mean, var

    # Vectorized over param_grid
    keys = jax.random.split(jax.random.PRNGKey(0), len(param_grid))
    means, vars_ = jax.vmap(single_crec_est, in_axes=(0, 0, None, None, None))(param_grid, keys, schedule, observable, num_samples)
    return means, vars_