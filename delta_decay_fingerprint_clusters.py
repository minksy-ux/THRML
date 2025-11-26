import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist

def delta_decay_fingerprint(X, decay=0.95):
    """
    Calculates a 'fingerprint' over a time-series matrix X using exponential decay over deltas.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        decay: float, decay rate for deltas

    Returns:
        fingerprints: np.ndarray, decay-weighted deltas
    """
    deltas = np.diff(X, axis=0)
    decay_weights = np.power(decay, np.arange(deltas.shape[0])[::-1])[:,None]
    fingerprints = (deltas * decay_weights).sum(axis=0)
    return fingerprints

def cluster_by_fingerprint(X, decay=0.95, eps=0.5, min_samples=5):
    """
    Cluster samples using Delta-Decay Fingerprint.
    """
    fingerprints = np.array([delta_decay_fingerprint(X[max(0,i-20):i+1], decay) for i in range(20, len(X))])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(fingerprints)
    return labels

class MicroRegimeAttractor:
    """
    Splits time series into short-term recurrence regimes.
    """
    def __init__(self, window=30, attractor_radius=1.0):
        self.window = window
        self.radius = attractor_radius
        self.centers = []
        self.regimes = []

    def fit(self, X):
        """
        Build short-term attractor regimes.
        """
        for i in range(len(X) - self.window):
            segment = X[i:i+self.window]
            center = segment.mean(axis=0)
            self.centers.append(center)
            # Recurrence: return indices/segments within radius
            dists = np.linalg.norm(segment - center, axis=1)
            recur_idx = np.where(dists < self.radius)[0]
            self.regimes.append((i, recur_idx))

def gap_entropy(x):
    """
    Measure entropy of the gaps in sorted data.
    """
    sorted_x = np.sort(x)
    gaps = np.diff(sorted_x)
    # Normalize gaps to sum to 1
    P = gaps / gaps.sum() if gaps.sum() != 0 else np.ones_like(gaps)/len(gaps)
    entropy = -np.sum(P * np.log(P + 1e-8))
    return entropy

def gap_entropy_rejection(X, threshold=0.5):
    """
    Reject samples with low gap-entropy (i.e., too regular/gappy).
    """
    entropies = np.apply_along_axis(gap_entropy, 1, X)
    mask = entropies > threshold
    return X[mask]

def local_density_veto(X, k=5, thresh=0.2):
    """
    Veto samples in low-density regions (outliers).
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    densities = 1.0 / (dists[:,1:].mean(axis=1) + 1e-8)
    mask = densities > thresh
    return X[mask]

def reverse_engineered_simulation_leak(X_train, y_train, X_sim, y_sim):
    """
    Detect 'leak' — e.g., simulation patterns mimicking training data.
    (Simple example: cosine similarity between distributions)
    """
    sim_fingerprint = X_sim.mean(axis=0)
    train_fingerprint = X_train.mean(axis=0)
    cosine_sim = np.dot(sim_fingerprint, train_fingerprint) / (np.linalg.norm(sim_fingerprint) * np.linalg.norm(train_fingerprint) + 1e-8)
    leak_detected = cosine_sim > 0.99  # or another high threshold
    return leak_detected, cosine_sim

class DynamicWeightedFusion:
    """
    Combines predictions using weights from Ridge regression.
    """
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, X_preds, y):
        """
        X_preds: ensemble predictions shape (n_samples, n_models)
        y: ground truth, shape (n_samples,)
        """
        self.model.fit(X_preds, y)
        self.fitted = True

    def predict(self, X_preds):
        return self.model.predict(X_preds)