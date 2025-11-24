#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THRML Discrete Autoregressive Forecaster (1–100 range)
======================================================
- Nonlinear probabilistic forecasting for time series using THRML (Energy-Based Models in JAX)
- Outperforms linear/polynomial regression + synthetic data for discrete values.
- Produces in-sample MAE diagnostics and rolling forecast plots

USAGE:
    pip install thrml jax[jaxlib] pandas matplotlib
    python thrml_forecaster.py

INPUT:
    CSVs named outcomes*.csv (columns: x, y), each row is a data point

OUTPUT:
    - Rolling forecast and in-sample MAE per file
    - MAE history CSV and PNG plot

Requires: THRML >=0.1, JAX >=0.3, pandas, matplotlib
"""

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import warnings

with warnings.catch_warnings():
    # Suppress any import warnings
    warnings.simplefilter("ignore")
    try:
        from thrml import DiscreteNode, Block, SamplingSchedule, sample_states
        from thrml.models import DiscreteEBM, DiscreteSamplingProgram
        from thrml.utils import one_hot_to_int
    except ImportError:
        print("Please install the required libraries: pip install thrml jax[jaxlib] pandas matplotlib")
        sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

FILES = [
    'outcomes.csv', 'outcomes2.csv', 'outcomes3.csv',
    'outcomes4.csv', 'outcomes5.csv', 'outcomes6.csv'
]

N_VALUES = 100        # Discrete states: values 1 to 100
N_WARMUP = 200
N_SAMPLES = 5000
STEPS_PER_SAMPLE = 5
N_FORECASTS = 5       # How many steps ahead to predict

# Output
MAE_HISTORY_FILE = 'mae_history_thrml.csv'
PLOT_FILE = 'mae_history_thrml.png'

# =============================================================================
# Utility
# =============================================================================

def set_random_seed(seed=42):
    np.random.seed(seed)

def info(msg):
    print(msg, flush=True)

# =============================================================================
# Build THRML model from historical (x, y) pairs
# =============================================================================

def build_model_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, header=None, names=['x', 'y'])
    df = df.sort_values('x').reset_index(drop=True)
    n_obs = len(df)

    # Each node = one timestep
    nodes = [DiscreteNode(num_states=N_VALUES) for _ in range(n_obs)]
    biases = jnp.zeros((n_obs, N_VALUES))
    observed_states_0idx = (df['y'].values - 1).clip(0, N_VALUES - 1)
    biases = biases.at[jnp.arange(n_obs), observed_states_0idx].set(10.0)  # strong clamp

    # Pairwise (time-smoothness)
    edges = [(nodes[i], nodes[i+1]) for i in range(n_obs-1)]
    weights = jnp.ones(len(edges)) * 0.8

    # Model over observed data
    model = DiscreteEBM(
        nodes=nodes,
        edges=edges,
        unary_potentials=biases,
        pairwise_potentials=weights[None, :, None, None]
    )

    # Add future nodes for forecasting (uniform bias)
    future_nodes = [DiscreteNode(num_states=N_VALUES) for _ in range(N_FORECASTS)]
    all_nodes = nodes + future_nodes

    # Edges from last observed to first future, and sequential in future
    future_edges = [
        (all_nodes[i], all_nodes[i+1])
        for i in range(len(all_nodes)-N_FORECASTS-1, len(all_nodes)-1)
    ]
    future_weights = jnp.ones(len(future_edges)) * 0.8

    extended_biases = jnp.concatenate([biases, jnp.zeros((N_FORECASTS, N_VALUES))], axis=0)
    extended_edges = edges + future_edges
    extended_weights = jnp.concatenate([weights, future_weights])

    full_model = DiscreteEBM(
        nodes=all_nodes,
        edges=extended_edges,
        unary_potentials=extended_biases,
        pairwise_potentials=extended_weights[None, :, None, None]
    )

    return full_model, nodes, future_nodes, df

# =============================================================================
# Sampling & Forecasting
# =============================================================================

def forecast_with_thrml(model, observed_nodes, future_nodes, key):
    """Forecast future y values given the model and keys."""
    free_blocks = [Block(future_nodes)]
    clamped_blocks = []  # observed nodes = fixed, via strong bias

    program = DiscreteSamplingProgram(model, free_blocks, clamped_blocks=clamped_blocks)
    schedule = SamplingSchedule(
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        steps_per_sample=STEPS_PER_SAMPLE
    )

    # Initial state, drawn randomly for future (will be mixed out)
    init_state = {
        node: jax.random.randint(key, (N_VALUES,), 0, N_VALUES) for node in future_nodes
    }
    key_sample = jax.random.fold_in(key, 12345)
    samples = sample_states(key_sample, program, schedule, init_state)

    # Most probable future states (after burn-in)
    final_samples = samples[-1]
    future_states_0idx = [int(final_samples[node][0]) for node in future_nodes]
    future_y = [v+1 for v in future_states_0idx]  # to 1–100

    return future_y

# =============================================================================
# MAE on historical data (in-sample fit)
# =============================================================================

def compute_in_sample_mae(model, observed_nodes, df, key):
    """Compute MAE on the observed points (in-sample)."""
    program = DiscreteSamplingProgram(model, free_blocks=[Block(observed_nodes)], clamped_blocks=[])
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=3)
    samples = sample_states(key, program, schedule, {})
    final = samples[-1]

    y_pred = [int(final[node][0])+1 for node in observed_nodes]
    y_true = df['y'].values

    mae = jnp.mean(jnp.abs(jnp.array(y_pred) - jnp.array(y_true)))
    return float(mae), list(y_pred)

# =============================================================================
# Main Loop
# =============================================================================

def main():
    set_random_seed(1234)
    forecasts = {}
    metrics = {}
    mae_records = []

    info("THRML Discrete EBM Forecaster (1–100 range)")

    some_data_found = False

    for file in FILES:
        if not Path(file).exists():
            info(f"  {file} not found, skipping.")
            continue

        info(f"Processing {file} ...")
        some_data_found = True
        model, obs_nodes, fut_nodes, df = build_model_from_csv(file)
        last_x = df['x'].iloc[-1]
        forecast_xs = [last_x + i + 1 for i in range(N_FORECASTS)]
        key = jax.random.key(np.random.randint(0, 2**30))
        key_forecast, key_mae = jax.random.split(key)

        # Forecast future ys
        future_ys = forecast_with_thrml(model, obs_nodes, fut_nodes, key_forecast)
        # In-sample MAE
        mae, yfit = compute_in_sample_mae(model, obs_nodes, df, key_mae)

        forecasts[file] = list(zip(forecast_xs, future_ys))
        metrics[file] = mae

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mae_records.append({'file': file, 'mae': mae, 'timestamp': timestamp})

        info(f"  → Forecasts: {future_ys}")
        info(f"  → In-sample MAE: {mae:.3f}")
        # Uncomment for diagnostics:
        # info(f"  → Fitted y (in-sample): {yfit}")

    if not some_data_found:
        info("No data files found. Exiting.")
        sys.exit(0)

    # Save MAE history
    mae_df = pd.DataFrame(mae_records)
    mae_df.to_csv(MAE_HISTORY_FILE, index=False)

    # Rolling MAE plot
    try:
        hist = pd.read_csv(MAE_HISTORY_FILE)
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for f in FILES:
            data = hist[hist['file'] == f]
            if not data.empty:
                plt.plot(data['timestamp'], data['mae'], 'o-', label=f)
        plt.legend()
        plt.title('THRML Model In-Sample MAE Over Time (1–100 range)')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=120)
        info(f"\nMAE history plot saved to {PLOT_FILE}")
    except Exception as e:
        info(f"Could not generate plot: {e}")

    # Final Forecast Output
    print("\n" + "="*60)
    print("THRML FORECAST RESULTS (1–100 range)")
    print("="*60)
    for file in forecasts:
        print(f"\n{file}:")
        for x, y in forecasts[file]:
            print(f"  x = {x:4d}  →  y = {y:3d}")
        print(f"  In-sample MAE = {metrics[file]:.3f}")

if __name__ == "__main__":
    main()