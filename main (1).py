# Pseudocode for running the sweep & diagnostics
import jax
import numpy as np

# Step 1: Define param grid
param_grid = [...]  # List of parameter PyTrees or arrays (shape: [num_points, ...])
schedule = ...   # THRML-compatible schedule
init_state = ... # (best to re-seed/initialize per grid point for fairness)

# Step 2: Run sweep (CREC means & variances)
crec_means, crec_vars = crec_sweep(param_grid, schedule, init_state, crec_from_global_state, num_samples=1000)

# Step 3: Find maximum
max_params, max_crec = find_max_crec(crec_means, param_grid)

# Step 4: Diagnostics
ess = effective_sample_size(np.array(crec_means))
# Optionally, r_hat if you run multiple chains per parameter

print(f"Max CREC: {max_crec} at parameters {max_params}, ESS: {ess}")