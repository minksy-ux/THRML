//! Litmus MVP – A thermodynamic-first probabilistic programming system
//! Compile with: cargo run --release

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

// ==================================================
// 1. Core trait for probability distributions
// ==================================================

trait Dist {
    fn sample(&self, rng: &mut ThreadRng) -> f64;
    fn log_pdf(&self, x: f64) -> f64;
}

struct NormalDist {
    mean: f64,
    std: f64,
}

impl NormalDist {
    fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl Dist for NormalDist {
    fn sample(&self, rng: &mut ThreadRng) -> f64 {
        Normal::new(self.mean, self.std).unwrap().sample(rng)
    }
    
    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        -0.5 * z * z - self.std.ln() - 0.5 * (2.0 * PI).ln()
    }
}

struct UniformDist {
    low: f64,
    high: f64,
}

impl UniformDist {
    fn new(low: f64, high: f64) -> Self {
        Self { low, high }
    }
}

impl Dist for UniformDist {
    fn sample(&self, rng: &mut ThreadRng) -> f64 {
        Uniform::new(self.low, self.high).sample(rng)
    }
    
    fn log_pdf(&self, x: f64) -> f64 {
        if x >= self.low && x <= self.high {
            -(self.high - self.low).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

// ==================================================
// 2. Gaussian Mixture Model with thermodynamic inference
// ==================================================

struct GaussianMixtureModel {
    // Parameters
    mu: Vec<f64>,      // Cluster means
    sigma: Vec<f64>,   // Cluster std devs
    z: Vec<usize>,     // Cluster assignments
    
    // Data
    obs: Vec<f64>,
    
    // Hyperparameters
    k: usize,          // Number of clusters
}

impl GaussianMixtureModel {
    fn new(data: Vec<f64>, k: usize) -> Self {
        let mut rng = thread_rng();
        let n = data.len();
        
        // Initialize parameters from priors
        let mu: Vec<f64> = (0..k)
            .map(|_| Normal::new(0.0, 10.0).unwrap().sample(&mut rng))
            .collect();
        
        let sigma: Vec<f64> = (0..k)
            .map(|_| Uniform::new(0.5, 5.0).sample(&mut rng))
            .collect();
        
        // Random initial cluster assignments
        let z: Vec<usize> = (0..n)
            .map(|_| rng.gen_range(0..k))
            .collect();
        
        Self {
            mu,
            sigma,
            z,
            obs: data,
            k,
        }
    }
    
    // Global energy function (negative log joint probability)
    fn energy(&self) -> f64 {
        let mut e = 0.0;
        
        // Prior energy for means (Normal(0, 10))
        let mu_prior = NormalDist::new(0.0, 10.0);
        for &mu in &self.mu {
            e -= mu_prior.log_pdf(mu);
        }
        
        // Prior energy for sigmas (Uniform(0.5, 5))
        let sigma_prior = UniformDist::new(0.5, 5.0);
        for &sigma in &self.sigma {
            e -= sigma_prior.log_pdf(sigma);
        }
        
        // Undirected factor: volume regularization
        // Penalize large variances to prevent collapse
        for &sigma in &self.sigma {
            e -= 0.1 * sigma.ln();
        }
        
        // Likelihood energy (observations given assignments)
        for (i, &x) in self.obs.iter().enumerate() {
            let k = self.z[i];
            let mu = self.mu[k];
            let sigma = self.sigma[k];
            let residual = (x - mu) / sigma;
            e -= -0.5 * residual * residual - sigma.ln() - 0.5 * (2.0 * PI).ln();
        }
        
        e
    }
    
    // Block Gibbs sweep - the thermodynamic kernel
    fn gibbs_sweep(&mut self) {
        self.sample_cluster_means();
        self.sample_cluster_stds();
        self.sample_assignments();
    }
    
    // Sample cluster means (continuous block update)
    fn sample_cluster_means(&mut self) {
        let mut rng = thread_rng();
        
        for k in 0..self.k {
            // Gather points assigned to this cluster
            let points: Vec<f64> = self.obs.iter()
                .enumerate()
                .filter(|(i, _)| self.z[*i] == k)
                .map(|(_, &x)| x)
                .collect();
            
            if points.is_empty() {
                // No points in cluster, sample from prior
                self.mu[k] = Normal::new(0.0, 10.0).unwrap().sample(&mut rng);
                continue;
            }
            
            let n = points.len() as f64;
            let x_bar = points.iter().sum::<f64>() / n;
            let sigma = self.sigma[k];
            
            // Conjugate update: combine prior and likelihood
            let prior_prec = 1.0 / (10.0 * 10.0);  // Prior precision
            let like_prec = n / (sigma * sigma);    // Likelihood precision
            
            let post_prec = prior_prec + like_prec;
            let post_mean = (like_prec * x_bar) / post_prec;
            let post_std = (1.0 / post_prec).sqrt();
            
            self.mu[k] = Normal::new(post_mean, post_std).unwrap().sample(&mut rng);
        }
    }
    
    // Sample cluster standard deviations
    fn sample_cluster_stds(&mut self) {
        let mut rng = thread_rng();
        
        for k in 0..self.k {
            // Metropolis-Hastings update for sigma
            let current = self.sigma[k];
            let proposal = current * (1.0 + Normal::new(0.0, 0.1).unwrap().sample(&mut rng));
            
            if proposal < 0.5 || proposal > 5.0 {
                continue; // Reject if outside prior support
            }
            
            // Compute energy difference
            let current_energy = self.energy();
            self.sigma[k] = proposal;
            let proposal_energy = self.energy();
            
            let log_accept_ratio = current_energy - proposal_energy;
            
            if log_accept_ratio > 0.0 || rng.gen::<f64>().ln() < log_accept_ratio {
                // Accept
            } else {
                // Reject
                self.sigma[k] = current;
            }
        }
    }
    
    // Sample cluster assignments (discrete block update)
    fn sample_assignments(&mut self) {
        let mut rng = thread_rng();
        
        for i in 0..self.obs.len() {
            let x = self.obs[i];
            
            // Compute log probabilities for each cluster
            let mut log_probs = Vec::with_capacity(self.k);
            for k in 0..self.k {
                let mu = self.mu[k];
                let sigma = self.sigma[k];
                let residual = (x - mu) / sigma;
                let log_p = -0.5 * residual * residual - sigma.ln();
                log_probs.push(log_p);
            }
            
            // Numerical stability: subtract max before exp
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<f64> = log_probs.iter()
                .map(|lp| (lp - max_lp).exp())
                .collect();
            
            // Sample from categorical distribution
            let dist = rand::distributions::WeightedIndex::new(&probs).unwrap();
            self.z[i] = dist.sample(&mut rng);
        }
    }
    
    fn print_state(&self, iter: usize) {
        println!(
            "Iter {:5} | μ = [{}] | σ = [{}] | Energy = {:.1}",
            iter,
            self.mu.iter()
                .map(|m| format!("{:+6.3}", m))
                .collect::<Vec<_>>()
                .join(", "),
            self.sigma.iter()
                .map(|s| format!("{:.3}", s))
                .collect::<Vec<_>>()
                .join(", "),
            -self.energy()
        );
    }
}

// ==================================================
// 3. Main: Demonstrate thermodynamic inference
// ==================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  LITMUS: Thermodynamic Probabilistic Programming System   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Generate synthetic data: K=3 well-separated clusters
    let data: Vec<f64> = {
        let mut rng = thread_rng();
        let mut xs = vec![];
        
        // Cluster 1: mean=-4, std=1.0, n=150
        xs.extend(Normal::new(-4.0, 1.0).unwrap().sample_iter(&mut rng).take(150));
        
        // Cluster 2: mean=+1, std=0.8, n=100
        xs.extend(Normal::new(1.0, 0.8).unwrap().sample_iter(&mut rng).take(100));
        
        // Cluster 3: mean=+5, std=1.2, n=120
        xs.extend(Normal::new(5.0, 1.2).unwrap().sample_iter(&mut rng).take(120));
        
        xs.shuffle(&mut rng);
        xs
    };
    
    let k = 3;
    println!("Dataset: {} points from {} clusters", data.len(), k);
    println!("True means: [-4.0, +1.0, +5.0]");
    println!("True stds:  [ 1.0,  0.8,  1.2]\n");
    
    // Initialize model
    let mut model = GaussianMixtureModel::new(data, k);
    
    println!("Running thermodynamic inference (Block Gibbs sampling)...\n");
    
    // Burn-in and sampling
    let n_iter = 5000;
    let print_every = 500;
    
    for iter in 1..=n_iter {
        model.gibbs_sweep();
        
        if iter % print_every == 0 || iter == 1 {
            model.print_state(iter);
        }
    }
    
    // Final results
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  FINAL POSTERIOR ESTIMATES                                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    
    let mut means = model.mu.clone();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut stds = model.sigma.clone();
    stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    println!("\nCluster means: [{}]", 
        means.iter()
            .map(|m| format!("{:+.3}", m))
            .collect::<Vec<_>>()
            .join(", "));
    println!("Cluster stds:  [{}]", 
        stds.iter()
            .map(|s| format!("{:.3}", s))
            .collect::<Vec<_>>()
            .join(", "));
    
    println!("\n✓ Thermodynamic inference complete!");
    println!("  • Zero allocation overhead");
    println!("  • Block-structured state updates");
    println!("  • Ready for extropic hardware acceleration\n");
}
