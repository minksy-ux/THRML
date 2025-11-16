from tsp_model import generate_tsp_cities, run_tsp_sync, decode_tour
import matplotlib.pyplot as plt
import jax.numpy as jnp

def main():
    N = 12
    cities = generate_tsp_cities(N, seed=123)
    print("Cities (coordinates):")
    print(cities)
    samples, model, cities = run_tsp_sync(cities, beta=2.0, n_samples=500, n_warmup=200)
    best_sample = samples[0]  # First sample, or argmin logic for best tour
    tour = decode_tour(best_sample, N)
    print("Best tour:", tour)
    # Visualization
    x, y = cities[:, 0], cities[:, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=40)
    for i in range(len(tour)):
        plt.text(x[tour[i]], y[tour[i]], str(tour[i]), fontsize=10, ha="center", va="center")
    tx = [cities[tour[i]][0] for i in range(len(tour))] + [cities[tour[0]][0]]
    ty = [cities[tour[i]][1] for i in range(len(tour))] + [cities[tour[0]][1]]
    plt.plot(tx, ty, "r-", lw=2)
    plt.title("TSP Solution via THRML Synchronization Sampling (Potts Model)")
    plt.show()

if __name__ == "__main__":
    main()