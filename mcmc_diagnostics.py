import numpy as np

def effective_sample_size(x):
    """Compute (univariate) ESS using autocorrelation method."""
    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    autocorr_sum = 0.0
    for lag in range(1, n):
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        if c < 0:
            break
        autocorr_sum += c
    ess = n / (1 + 2 * autocorr_sum)
    return ess

def r_hat(chains):
    """Estimate R-hat mixing diagnostic (split chains if needed)."""
    chains = np.asarray(chains)
    n_chains, n = chains.shape
    chain_means = chains.mean(axis=1)
    mean = chain_means.mean()
    B = n * ((chain_means - mean) ** 2).sum() / (n_chains - 1)
    W = ((chains - chain_means[:, None]) ** 2).sum() / (n_chains * (n - 1))
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    R_hat = np.sqrt(var_hat / W)
    return R_hat