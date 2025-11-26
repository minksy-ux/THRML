import jax
from typing import Callable

def run_block_sampler_with_observable(init_state, params, schedule, key, observable: Callable, num_samples=100):
    """
    Run a block Gibbs sampler, collecting observable measurements.

    Args:
      init_state, params, schedule, key: Standard THRML/JAX MCMC arguments.
      observable: function(global_state, params) -> float
      num_samples: number of MC samples to collect
    Returns:
      observ_vals: [num_samples] array of observable values
      states: [num_samples] PyTree, all visited global states
    """
    states = []
    observ_vals = []

    state = init_state
    rng = key
    for i in range(num_samples):
        # One Gibbs (or block) step
        state, rng = sample_one_step(state, params, schedule, rng)  # Replace with THRML primitive
        obs_val = observable(state, params)
        observ_vals.append(obs_val)
        states.append(state)

    return jnp.stack(observ_vals), states

# in practice, consider using JAX.lax.scan for JIT/vectorization!