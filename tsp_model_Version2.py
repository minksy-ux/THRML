import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block
from thrml.models import PottsEBM, PottsSamplingProgram, uniform_init
import numpy as np

def generate_tsp_cities(n_cities: int, seed: int = 42) -> jnp.ndarray:
    rng = jax.random.PRNGKey(seed)
    return jax.random.uniform(rng, (n_cities, 2)) * 100

def tsp_distance_matrix(cities: jnp.ndarray) -> jnp.ndarray:
    diff = cities[:, None, :] - cities[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))

def build_tsp_potts_model(n_cities: int, dist_matrix: jnp.ndarray, beta: float = 1.0):
    N = n_cities
    nodes = [[CategoricalNode(N) for _ in range(N)] for _ in range(N)]  # position x city
    nodes_flat = [node for row in nodes for node in row]
    edges = []
    weights = []
    for pos in range(N):
        next_pos = (pos + 1) % N
        for city in range(N):
            for next_city in range(N):
                if city != next_city:
                    edges.append((nodes[pos][city], nodes[next_pos][next_city]))
                    weights.append(-dist_matrix[city, next_city])
    for pos in range(N):
        for c1 in range(N):
            for c2 in range(c1 + 1, N):
                edges.append((nodes[pos][c1], nodes[pos][c2]))
                weights.append(10.0)  # hard constraint
    for city in range(N):
        for p1 in range(N):
            for p2 in range(p1 + 1, N):
                edges.append((nodes[p1][city], nodes[p2][city]))
                weights.append(10.0)
    model = PottsEBM(nodes_flat, edges, jnp.array(weights), jnp.array(beta))
    return model, nodes

def decode_tour(sample: jnp.ndarray, N: int) -> list:
    return [int(jnp.argmax(sample[pos * N : (pos + 1) * N])) for pos in range(N)]

def run_tsp_sync(cities: jnp.ndarray, beta: float = 2.0, n_samples=500, n_warmup=200, steps_per_sample=3, seed=42):
    N = len(cities)
    dist_matrix = tsp_distance_matrix(cities)
    model, nodes = build_tsp_potts_model(N, dist_matrix, beta)
    free_blocks = [Block([nodes[pos][city] for city in range(N)]) for pos in range(N)]
    program = PottsSamplingProgram(model, free_blocks, clamped_blocks=[])
    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key)
    init_state = uniform_init(k_init, model, free_blocks, ())
    from thrml import SamplingSchedule, sample_states
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block([n for row in nodes for n in row])])
    return samples[0], model, cities