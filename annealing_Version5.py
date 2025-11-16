import jax.numpy as jnp

def geometric_annealing(beta_start=0.1, beta_end=5.0, n_steps=1000):
    """Geometric temperature annealing schedule for MCMC."""
    betas = jnp.geomspace(beta_start, beta_end, n_steps)
    return betas

def adaptive_annealing(energy_history):
    """Adapt beta (inverse temp) based on recent energy variation."""
    recent = energy_history[-10:]
    delta = jnp.mean(jnp.abs(jnp.diff(recent)))
    current_beta = 1.0 / (delta + 1e-6)
    return jnp.clip(current_beta, 0.1, 10.0)