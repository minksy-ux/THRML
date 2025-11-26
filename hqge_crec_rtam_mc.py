import jax
import jax.numpy as jnp

### --- Reuse all HQGE array setup from previous reply! --- ###

# Chaos resonance: Add random temporal fields at each step
def chaos_resonance_field(step, n_qubits, key):
    # Per-step random field (can be more sophisticated, e.g. correlated)
    subkey = jax.random.fold_in(key, step)
    return jax.random.normal(subkey, (n_qubits,)) * 0.3  # scale as needed

def run_crec_block_gibbs(key, state, n_steps, J, src, tgt, h, beta, blocks):
    def step_fn(carry, i):
        key, state = carry
        key, *sub = jax.random.split(key, len(blocks) + 2)
        # Add chaos field to h each step (echo chamber)
        h_dyn = h + chaos_resonance_field(i, len(h), sub[0])
        for bi, block in zip(range(len(blocks)), blocks):
            state = gibbs_update(sub[bi+1], state, J, src, tgt, h_dyn, beta, block)
        return (key, state), state
    (final_key, final_state), states = jax.lax.scan(
        step_fn, (key, state), jnp.arange(n_steps)
    )
    return states

# Reverse-Time Adversarial Mirror: use -J, run backward in time
def reverse_time_adversarial_mirror(key, state, n_steps, J, src, tgt, h, beta, blocks):
    return run_block_gibbs(key, state, n_steps, -J, src, tgt, h, beta, blocks)

# Self-Rejective Monte Carlo + Weighted Fusion
def self_rejective_mc(key, n_chains, n_steps, J, src, tgt, h, beta, blocks, thresh=1.0):
    # Run a very wide MC with rejection and weighted (importance) fusion
    keys = jax.random.split(key, n_chains)
    init_states = jax.vmap(lambda k: hinton_init(k, len(h)))(keys)
    final_states = jax.vmap(lambda state, k: run_block_gibbs(k, state, n_steps, J, src, tgt, h, beta, blocks))(
        init_states, keys)
    # final_states: (n_chains, n_steps, n_qubits)
    # For each chain, score energy of last MC step
    energies = jax.vmap(lambda st: ising_energy({'spins': st[-1]}, J, src, tgt, h))(final_states)
    # Reject too-typical samples (self-rejection)
    mask = jnp.abs(energies - jnp.mean(energies)) > thresh * jnp.std(energies)
    survivors = final_states[mask][:, -1, :]
    # Weighted fusion (e.g., by exp(-energy))
    surv_energies = energies[mask]
    weights = jax.nn.softmax(-surv_energies)
    fused = jnp.average(survivors, axis=0, weights=weights)  # shape (n_qubits,)
    return survivors, weights, fused

# Example Usage:
key = jax.random.key(123)
# Run CREC
crec_states = run_crec_block_gibbs(key, hinton_init(key, n_qubits), 200, J, src, tgt, h, beta, blocks)

# RTAM from a final sample
rtam_states = reverse_time_adversarial_mirror(key, cread_states['spins'][-1], 100, J, src, tgt, h, beta, blocks)

# Self-Rejective MC (with moderate sample count for demo—use 10**8+ for full effect)
survivors, weights, fused = self_rejective_mc(key, 10000, 50, J, src, tgt, h, beta, blocks)
print("Self-rejective MC: #survivors:", survivors.shape[0])
print("Weighted-fused survivor state:\n", fused)