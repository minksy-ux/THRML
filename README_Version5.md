# THRML TSP Synchronization Demo

**Traveling Salesman Problem (TSP)** solved by physics-inspired block Gibbs synchronization via Potts model, all in JAX with THRML. Visualize optimal tours and sample energy landscapes. Extensible for robotics, constraints, hybrid classical solvers, or hardware-in-the-loop.

---

## Install Dependencies

```bash
pip install thrml networkx matplotlib jax[cpu] panel plotly
```

Or use `jax[cuda]` for GPU acceleration.

---

## Run Standalone Script

```bash
python run_tsp_demo.py
```

---

## Run Real-Time Dashboard

```bash
panel serve dashboard.py --show --autoreload
```

---

## Extending & Research

- Add custom constraints or blocks to `tsp_model.py`.
- Try adaptive schedules in `annealing.py`.
- Integrate hardware or classical solvers for hybrid initialization.
- Visualize/benchmark energy landscapes, solutions, and convergence.

---

**Built with THRML, JAX, and open-source tooling for combinatorial optimization and physics-inspired modeling.**