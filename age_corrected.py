import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from thrml import (
    SpinNode,
    CategoricalNode,
    Block,
    SamplingSchedule,
    sample_states,
    hinton_init
)
from thrml.models import FactorGraphEBM, CustomSamplingProgram

# ================================================================
# 1. DEFINE SIZES FIRST
# ================================================================
N_ANNIHILATION = 1024
N_PBR = 256
N_GVTS = 128
N_PEDL = 64
N_MDEE = 32
N_DLP = 8

# ================================================================
# 2. Node types
# ================================================================

# Core annihilation spins (±1)
annihilation_nodes = [SpinNode(name=f"A_{i}") for i in range(N_ANNIHILATION)]

# Positional bias recalibrator nodes (continuous-ish but discretized to 8 levels)
PBR_nodes = [CategoricalNode(8, name=f"PBR_{i}") for i in range(N_PBR)]

# Gap volatility tail states (high-dynamic-range tail events)
GVTS_nodes = [CategoricalNode(16, name=f"GVTS_{i}") for i in range(N_GVTS)]

# Echo-distance locks (discrete distance bins in log-scale)
PEDL_nodes = [CategoricalNode(12, name=f"PEDL_{i}") for i in range(N_PEDL)]

# Multi-decade echo expanders (very long-range temporal memory)
MDEE_nodes = [CategoricalNode(10, name=f"MDEE_{i}") for i in range(N_MDEE)]

# Dynamic law pruner (meta-variables that can turn entire factor classes on/off)
DLP_nodes = [SpinNode(name=f"DLP_{i}") for i in range(N_DLP)]

all_nodes = annihilation_nodes + PBR_nodes + GVTS_nodes + PEDL_nodes + MDEE_nodes + DLP_nodes

# ================================================================
# 3. Custom factors for the Annihilation Gate Ensemble
# ================================================================

def age_factors(state):
    """Returns total energy contribution."""
    energy = 0.0

    # Extract sub-states
    A = state[:N_ANNIHILATION]  # ±1
    PBR = state[N_ANNIHILATION:N_ANNIHILATION+N_PBR]  # 0..7
    GVTS = state[N_ANNIHILATION+N_PBR:N_ANNIHILATION+N_PBR+N_GVTS]  # 0..15
    PEDL = state[N_ANNIHILATION+N_PBR+N_GVTS:N_ANNIHILATION+N_PBR+N_GVTS+N_PEDL]  # 0..11
    MDEE = state[-N_DLP-N_MDEE:-N_DLP]  # 0..9
    DLP = state[-N_DLP:]  # ±1 (dynamic law pruner states)

    # Convert DLP to mask: -1 → 0, +1 → 1
    dlp_mask = (DLP + 1.0) / 2.0

    # 1. Core annihilation gate (encourages pairwise annihilation → 0 energy)
    # Only active if DLP[0] = +1
    core_energy = 0.0
    for i in range(N_ANNIHILATION-1):
        core_energy += 5.0 * (1 - A[i] * A[i+1])
    energy += core_energy * dlp_mask[0]

    # 2. Positional Bias Recalibrator → gently pushes spins toward learned biases
    # Only active if DLP[1] = +1
    learned_bias = (PBR - 3.5) / 4.0  # map 0..7 → roughly -0.9..+0.9
    pbr_energy = jnp.sum(-3.0 * A[:N_PBR] * learned_bias)
    energy += pbr_energy * dlp_mask[1]

    # 3. Gap Volatility Tail Sampler → heavy tails create rare "explosions"
    # Only active if DLP[2] = +1
    tail_strength = GVTS / 15.0 * 20.0  # up to ±20 energy swings
    gvts_energy = jnp.sum(tail_strength * A[:N_GVTS] ** 2)
    energy += gvts_energy * dlp_mask[2]

    # 4. Echo-Distance Lock → enforces consistency across distances
    # Only active if DLP[3] = +1
    log_dist = PEDL / 11.0 * 6.0  # 0 → 6 "decades"
    distance_penalty = jnp.exp(2.0 - log_dist)
    pedl_energy = 0.0
    for i in range(min(20, N_ANNIHILATION-8)):
        pedl_energy += distance_penalty[i//8] * (1 - A[i] * A[i+8])
    energy += pedl_energy * dlp_mask[3]

    # 5. Multi-Decade Echo Expander → extremely long-range ferromagnetic chains
    # Only active if DLP[4] = +1
    mdee_coupling = (MDEE - 4.5) / 5.0 * 0.8  # coupling ∈ [-0.8, 0.8]
    mdee_energy = 0.0
    for k in range(N_MDEE):
        i, j = k*37 % N_ANNIHILATION, (k*59 + 13) % N_ANNIHILATION
        mdee_energy += -mdee_coupling[k] * A[i] * A[j]
    energy += mdee_energy * dlp_mask[4]

    return energy

# ================================================================
# 4. Build the model
# ================================================================

model = FactorGraphEBM(
    nodes=all_nodes,
    factors=[age_factors]  # Single energy function
)

# ================================================================
# 5. Block structure optimized for parallel updates
# ================================================================

# Highly non-local blocks to maximize parallel conditional updates
blocks = [
    Block(annihilation_nodes[::8] + PBR_nodes[::4]),   # Block 0
    Block(annihilation_nodes[1::8] + GVTS_nodes[::3]),  # Block 1
    Block(annihilation_nodes[2::8] + PEDL_nodes),      # Block 2
    Block(annihilation_nodes[3::8] + MDEE_nodes),      # Block 3
    Block(annihilation_nodes[4::8]),                   # Block 4
    Block(annihilation_nodes[5::8]),                   # Block 5
    Block(annihilation_nodes[6::8]),                   # Block 6
    Block(annihilation_nodes[7::8]),                   # Block 7
    Block(DLP_nodes),                                  # Meta-block
]

# All blocks are free during sampling
program = CustomSamplingProgram(model, free_blocks=blocks, clamped_blocks=[])

# ================================================================
# 6. Sampling loop with occasional DLP updates ("law evolution")
# ================================================================

@partial(jit, static_argnums=(2, 3))
def sample_age(key, init_state, n_samples=10_000, dlp_update_every=500):
    """Sample from AGE model with periodic DLP updates."""
    
    schedule = SamplingSchedule(
        n_warmup=2000,
        n_samples=n_samples,
        steps_per_sample=4
    )

    def body(carry, _):
        k, state, step = carry
        k1, k2 = jax.random.split(k)

        # Normal AGE sampling (one sweep through all blocks)
        new_state = state
        for block_idx in range(len(blocks)):
            k1, k_block = jax.random.split(k1)
            # Sample this block conditionally
            new_state = sample_states(
                k_block, 
                program, 
                SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=1),
                new_state, 
                [],
                [blocks[block_idx]]
            )[0]

        # Occasionally flip DLP nodes (dynamic law pruning)
        def update_dlp(s):
            dlp_part = s[-N_DLP:]
            prob_flip = jnp.full(N_DLP, 0.1)  # 10% flip probability per DLP spin
            mask = jax.random.bernoulli(k2, prob_flip, shape=(N_DLP,))
            flipped = jnp.where(mask, -dlp_part, dlp_part)
            return s.at[-N_DLP:].set(flipped)

        new_state = lax.cond(
            (step % dlp_update_every == 0) & (step > 0),
            update_dlp,
            lambda s: s,
            new_state
        )
        
        return (k2, new_state, step + 1), new_state

    _, samples = lax.scan(body, (key, init_state, 0), None, length=n_samples)
    return samples

# ================================================================
# 7. Run it
# ================================================================

if __name__ == "__main__":
    key = jax.random.key(42)
    k_init, k_sample = jax.random.split(key)

    # Initialize state
    init_state = hinton_init(k_init, model, blocks, ())

    # Sample
    print("Starting AGE sampling...")
    samples = sample_age(k_sample, init_state, n_samples=1000, dlp_update_every=100)

    print("AGE samples shape:", samples.shape)
    print("Final DLP configuration:", samples[-1, -N_DLP:])
    
    # Analyze results
    final_annihilation = samples[-1, :N_ANNIHILATION]
    print(f"Final annihilation state mean: {jnp.mean(final_annihilation):.3f}")
    print(f"Final annihilation state std: {jnp.std(final_annihilation):.3f}")
