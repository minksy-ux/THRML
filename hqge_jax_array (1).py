import jax
import jax.numpy as jnp

# -----------------------------
# Hyper-Quantum Gap Entanglement (HQGE) on 10 qubits
# Fully global-state and JAX array level (CREC pattern)
# -----------------------------
n_qubits = 10
skip = 10
gap_coupling_strength = 5.0
beta = 8.0

# Sparse adjacency for skip connection: [i, (i+skip) % n]
src = jnp.arange(n_qubits)
tgt = (src + skip) % n_qubits

# J-couplings, one per skip-edge
J = jnp.full((n_qubits,), gap_coupling_strength)

# Local biases
h = jnp.zeros((n_qubits,))

# Block indices: even/odd for checkerboard Gibbs
even_idx = jnp.arange(0, n_qubits, 2)
odd_idx = jnp.arange(1, n_qubits, 2)
blocks = [even_idx, odd_idx]

# ---- Global Ising State (PyTree) ----
def make_state(spins):
    # spins: shape (n_qubits,), values {-1, +1}
    return {'spins': spins}

# Hinton initialization (for JAX, initialize {-1,+1} random spins)
def hinton_init(key, n_qubits):
    return make_state(jax.random.choice(key, jnp.array([-1,1]), (n_qubits,)))

# ---- Energy function (array version) ----
def ising_energy(state, J, src, tgt, h):
    spins = state['spins']  # shape (n_qubits,)
    edge_sum = jnp.sum(J * spins[src] * spins[tgt])
    bias_sum = jnp.sum(h * spins)
    return -edge_sum - bias_sum

# ---- Blocked Gibbs kernel ----
def gibbs_update(key, state, J, src, tgt, h, beta, block_idx):
    # Update just the spins in block_idx
    spins = state['spins']
    def flip_prob(i, s):
        # For site i, calculate the local field (mean field for just this site)
        connected = (src == i) | (tgt == i)
        partner = jnp.where(src[connected] == i, tgt[connected], src[connected])
        field = jnp.sum(J[connected] * spins[partner]) + h[i]
        logit = 2 * beta * field * s  # -h if spin is +1 (since flip)
        prob = jax.nn.sigmoid(-logit)
        return prob
    keys = jax.random.split(key, len(block_idx))
    def flip_site(i, k):
        prob = flip_prob(i, spins[i])
        flip = jax.random.bernoulli(k, prob)
        new_spin = jnp.where(flip, -spins[i], spins[i])
        return new_spin
    new_spins = spins.at[block_idx].set(
        jax.vmap(flip_site)(block_idx, keys)
    )
    return make_state(new_spins)

# ---- Block Gibbs Sampling ----
def run_block_gibbs(key, state, n_steps, J, src, tgt, h, beta, blocks):
    def step_fn(carry, i):
        key, state = carry
        key, *sub = jax.random.split(key, len(blocks)+1)
        for bi, block in zip(range(len(blocks)), blocks):
            state = gibbs_update(sub[bi], state, J, src, tgt, h, beta, block)
        return (key, state), state
    (final_key, final_state), states = jax.lax.scan(
        step_fn,
        (key, state),
        jnp.arange(n_steps)
    )
    return states

# ---- Full Sampling Pipeline ----
key = jax.random.key(42)
k_init, k_sample = jax.random.split(key)
init_state = hinton_init(k_init, n_qubits)

# Warmup
n_warmup = 500
states = run_block_gibbs(k_sample, init_state, n_warmup, J, src, tgt, h, beta, blocks)
warm_state = states['spins'][-1]

# Main sampling (take 10,000 samples, 1 sweeps/sample)
n_samples = 10_000
def sample_fn(carry, i):
    key, state = carry
    key, state = jax.random.split(key), state
    states = run_block_gibbs(key[0], state, 1, J, src, tgt, h, beta, blocks)
    final_state = states['spins'][-1]
    return (key[1], make_state(final_state)), final_state
_, samples = jax.lax.scan(sample_fn, (k_init, make_state(warm_state)), jnp.arange(n_samples))

bit_samples = (samples + 1) // 2  # {0,1}
print("First 10 HQGE samples (10-skip-gap hyper-states):")
print(bit_samples[:10])

corr_0_5 = jnp.mean(bit_samples[:, 0] * bit_samples[:, 5])
print(f"\nHyper-correlation <q0 · q5> = {corr_0_5:.4f}  (should be near ±1.0 due to strong 10-skip cycle)")

def vmap_energy(samples):
    # batched energy
    return jax.vmap(ising_energy, in_axes=(0, None, None, None, None))(samples, J, src, tgt, h)

energy = vmap_energy({'spins': samples})
print(f"Mean energy: {jnp.mean(energy):.2f} (lower = more coherent hyper-entangled state)")