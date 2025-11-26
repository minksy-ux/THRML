def find_max_crec(crec_sweep_results, param_grid):
    max_idx = np.argmax(crec_sweep_results)
    max_params = param_grid[max_idx]
    max_val = crec_sweep_results[max_idx]
    return max_params, max_val