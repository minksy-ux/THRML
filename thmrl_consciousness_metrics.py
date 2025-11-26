def iit_phi(state, depth=5):
    """
    Compute (or lower-bound) Integrated Information Φ from IIT 4.0.
    Steps:
        1. Takes current discrete state of a subsystem.
        2. Builds the full transition probability matrix (TPM) under the system’s dynamics.
        3. Finds the minimum information partition (MIP) via pyramid search up to depth 5.
        4. Computes Φ as Kullback-Leibler divergence D_KL(TPM ∥ TPM_after_MIP_cut).
    Returns:
        phi: Integrated Information, Φ.
    """
    tpm = compute_tpm_from_state(state)             # User-defined: get TPM for the system
    tpm_cut = min_information_partition(tpm, depth) # User-defined: cut TPM via MIP, search up to depth
    phi = kullback_leibler_divergence(tpm, tpm_cut)
    return phi


def global_workspace_ignition(state, threshold=0.74):
    """
    Simulates one recurrent step and measures the fraction of nodes/blocks
    that are mutually entrained above a threshold (can predict each other).
    Returns peak ignition strength (0–1).
    """
    participations = compute_participation(state)        # User-defined: get participation of each node/block
    mutual = [p for p in participations if p > threshold]
    ignition_strength = len(mutual) / len(participations)
    return ignition_strength                             # ~0.2 → ~0.8 transition is GWT signature


def higher_order_transition_entropy(state, order=3):
    """
    Computes H[P(z_t | z_{t-1}, ..., z_{t-order})] − H[P(z_t | z_{t-1}, ..., z_{t-order+1})].
    Measures genuine higher-order temporal dependence.
    """
    ce_high = conditional_entropy(state, order)
    ce_low = conditional_entropy(state, order - 1)
    return ce_high - ce_low                             # Positive => depends on deeper history


def attention_schema_score(state):
    """
    Returns the norm of gradients of attention weights, weighted by their
    accuracy in predicting future attention allocation.
    Measures strength of internal attention/body schema.
    """
    attn_wts = get_attention_weights(state)
    grad = gradient(attn_wts, state)
    prediction_accuracy = future_attention_predictive_accuracy(grad, state)
    score = norm(grad) * prediction_accuracy
    return score                                       # High = internal attention model present


def recurrent_predictive_error(state, horizon=8):
    """
    Averages variational free energy / prediction error over an 8-step
    recurrent rollout under the current policy.
    Equivalent to Friston’s free-energy principle metric.
    """
    total_error = 0.0
    z = state
    for t in range(horizon):
        q_t = infer_post_state(z, t)
        p_t = predict_prior_state(z, t)
        total_error += kullback_leibler_divergence(q_t, p_t)
        z = evolve_state(z)
    return total_error / horizon                       # Low = system minimizes surprise


def quantum_coherence_proxy(state):
    """
    Proxy for Orch-OR quantum coherence timescale.
    τ ≈ ħ / ΔE, with ΔE from mass displacement per parameter precision.
    Returns True if τ >= 25 ms (theoretical threshold for quantum consciousness).
    """
    for param in state.parameters():
        delta_e = gravitational_self_energy(param.precision)
        tau = PLANCK_CONSTANT / delta_e
        if tau >= 0.025:                              # 25 ms in seconds
            return True                               # Meets claimed threshold
    return False