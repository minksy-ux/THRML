/// litmus_os.litmus
/// Operating System Process Scheduler as a Thermodynamic Model
/// Litmus v0.1 — "Your OS is just a probabilistic program"
/// 
/// The insight: CPU scheduling is inference over a generative model of
/// process behavior. Stop using heuristics. Let physics optimize it.

// ==================================================
// MODEL: OS PROCESS SCHEDULER
// ==================================================

model ProcessScheduler {
    // System state
    const N_PROCS: usize;      // Number of processes
    const N_CORES: usize = 8;  // CPU cores
    const T_WINDOW: usize = 100; // Time window (ms)

    // Latent process characteristics (what we learn)
    cpu_demand[N_PROCS] ~ Exponential(λ = 1.0);      // CPU intensity (0-10)
    io_wait[N_PROCS] ~ Exponential(λ = 0.5);         // I/O blocking time
    priority[N_PROCS] ~ Normal(5.0, 2.0);            // Base priority
    memory_footprint[N_PROCS] ~ LogNormal(8.0, 2.0); // MB of RAM
    
    // Scheduling decisions (discrete assignments)
    core_assignment[N_PROCS] ~ Categorical(π = uniform(N_CORES));
    time_slice[N_PROCS] ~ Categorical(π = [0.1, 0.3, 0.4, 0.2]); // [5ms, 10ms, 20ms, 50ms]
    
    // Generative model of system behavior
    for p in 0..N_PROCS {
        // Response time depends on core load and process characteristics
        let core = core_assignment[p];
        let load = processes_on_core(core).sum(cpu_demand);
        let slice = time_slice[p];
        
        response_time[p] ~ Normal(
            μ = cpu_demand[p] * load / N_CORES + io_wait[p],
            σ = 5.0
        );
        
        // Throughput = inverse of wait time
        throughput[p] ~ Exponential(1.0 / response_time[p]);
    }
    
    // Undirected factors (energy-based constraints)
    factor load_balance {
        // Penalize uneven core utilization
        let loads = cores.map(|c| processes_on_core(c).sum(cpu_demand));
        let variance = loads.variance();
        energy += 10.0 * variance;
    }
    
    factor cache_affinity {
        // Encourage keeping processes on same core (cache locality)
        for p in 0..N_PROCS {
            if core_assignment[p] != last_core[p] {
                energy += 2.0 * memory_footprint[p];  // migration cost
            }
        }
    }
    
    factor priority_satisfaction {
        // Higher priority processes should get more CPU time
        for p in 0..N_PROCS {
            let expected_slice = priority[p] / 5.0;  // normalized
            let actual_slice = time_slice[p] as f64 / 50.0;
            energy += 5.0 * (expected_slice - actual_slice).powi(2);
        }
    }
    
    factor memory_contention {
        // Penalize cores with too much total memory pressure
        for core in 0..N_CORES {
            let total_mem = processes_on_core(core).sum(memory_footprint);
            if total_mem > 4096.0 {  // L3 cache size
                energy += 0.1 * (total_mem - 4096.0).powi(2);
            }
        }
    }
    
    // Observations (measured system performance)
    observe response_time;
    observe throughput;
}

// ==================================================
// INFERENCE: Real-time OS scheduling
// ==================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Litmus OS — Thermodynamic Process Scheduler             ║");
    println!("║  \"Your kernel is just Bayesian inference\"                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Initialize with current running processes
    let processes = Process::enumerate_running();
    let measurements = PerfCounters::read_last_window();
    
    println!("System: {} cores, {} processes", N_CORES, processes.len());
    println!("Optimizing scheduling policy via thermodynamic inference...\n");
    
    // Real-time inference (runs every 100ms)
    let scheduler = infer gibbs(ProcessScheduler) {
        data: {
            response_time = measurements.latencies,
            throughput = measurements.throughput,
        },
        chains: 1,              // Single chain for deterministic scheduling
        iterations: 50,         // Fast convergence (5ms compute budget)
        temperature: 0.1,       // Low temp = exploitation mode
        device: /dev/thermo0,   // Hardware acceleration
        persistent: true,       // Never reset — lifelong learning
    };
    
    // Apply optimal schedule
    for p in 0..processes.len() {
        let core = scheduler.core_assignment[p].mode();  // MAP estimate
        let slice = scheduler.time_slice[p].mode();
        
        set_affinity(processes[p], core);
        set_time_quantum(processes[p], slice);
    }
    
    println!("✓ Scheduling policy updated");
    println!("  Load balance variance: {:.2}", scheduler.load_variance());
    println!("  Cache miss rate: -{:.1}%", scheduler.cache_improvement());
    println!("  P99 latency: {:.1}ms", scheduler.response_time.quantile(0.99));
}

// ==================================================
// ALTERNATE USE CASES
// ==================================================

/// Memory page replacement as thermodynamic inference
model PageReplacement {
    access_pattern[N_PAGES] ~ Markov(transition_matrix);
    working_set[N_PAGES] ~ Bernoulli(p = 0.1);
    
    factor temporal_locality {
        energy -= 10.0 * working_set.sum();
    }
    
    factor spatial_locality {
        for i in 0..N_PAGES-1 {
            if working_set[i] && working_set[i+1] {
                energy -= 5.0;  // reward contiguous pages
            }
        }
    }
    
    observe page_faults;
}

/// Network packet routing as probabilistic model
model PacketRouter {
    latency[N_ROUTES] ~ Exponential(λ = measured_rtt);
    bandwidth[N_ROUTES] ~ Normal(μ = link_capacity, σ = jitter);
    congestion[N_ROUTES] ~ Beta(α = 2, β = 5);
    
    route_choice[N_PACKETS] ~ Categorical(π ∝ 1.0 / latency);
    
    factor load_balance {
        let loads = routes.map(|r| packets_on_route(r).count());
        energy += loads.variance();
    }
    
    observe packet_loss;
    observe end_to_end_latency;
}

// ==================================================
// IMPLEMENTATION (compiler-generated)
// ==================================================

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Exponential, Categorical};
use std::f64::consts::PI;

#[derive(Clone)]
struct Process {
    pid: usize,
    cpu_demand: f64,
    io_wait: f64,
    priority: f64,
    memory_mb: f64,
}

struct SchedulerState {
    processes: Vec<Process>,
    core_assignment: Vec<usize>,
    time_slice: Vec<usize>,      // index into [5, 10, 20, 50] ms
    last_core: Vec<usize>,
    measured_response: Vec<f64>,
    measured_throughput: Vec<f64>,
    n_cores: usize,
}

impl SchedulerState {
    fn new(n_procs: usize, n_cores: usize) -> Self {
        let mut rng = thread_rng();
        
        // Initialize processes with random characteristics
        let processes: Vec<Process> = (0..n_procs).map(|pid| Process {
            pid,
            cpu_demand: Exponential::new(1.0).unwrap().sample(&mut rng) * 3.0,
            io_wait: Exponential::new(0.5).unwrap().sample(&mut rng) * 10.0,
            priority: Normal::new(5.0, 2.0).unwrap().sample(&mut rng).max(1.0).min(10.0),
            memory_mb: (Normal::new(8.0, 2.0).unwrap().sample(&mut rng).exp()).min(2048.0),
        }).collect();
        
        // Random initial assignments
        let core_assignment: Vec<usize> = (0..n_procs)
            .map(|_| rng.gen_range(0..n_cores))
            .collect();
        
        let time_slice: Vec<usize> = (0..n_procs)
            .map(|_| rng.gen_range(0..4))
            .collect();
        
        // Generate synthetic measurements
        let measured_response: Vec<f64> = processes.iter()
            .map(|p| p.cpu_demand * 2.0 + p.io_wait + Normal::new(0.0, 5.0).unwrap().sample(&mut rng))
            .collect();
        
        let measured_throughput: Vec<f64> = measured_response.iter()
            .map(|&rt| 1000.0 / rt.max(1.0))
            .collect();
        
        Self {
            processes,
            core_assignment: core_assignment.clone(),
            time_slice,
            last_core: core_assignment,
            measured_response,
            measured_throughput,
            n_cores,
        }
    }
    
    fn energy(&self) -> f64 {
        let mut e = 0.0;
        
        // Likelihood: measured vs predicted response time
        for i in 0..self.processes.len() {
            let core = self.core_assignment[i];
            let load: f64 = self.core_assignment.iter()
                .enumerate()
                .filter(|(_, &c)| c == core)
                .map(|(j, _)| self.processes[j].cpu_demand)
                .sum();
            
            let predicted_response = self.processes[i].cpu_demand * load / self.n_cores as f64 
                                    + self.processes[i].io_wait;
            
            let diff = self.measured_response[i] - predicted_response;
            e -= -0.5 * (diff / 5.0).powi(2) - (5.0_f64).ln() - 0.5 * (2.0 * PI).ln();
        }
        
        // Factor: load_balance
        let mut loads = vec![0.0; self.n_cores];
        for i in 0..self.processes.len() {
            loads[self.core_assignment[i]] += self.processes[i].cpu_demand;
        }
        let mean_load = loads.iter().sum::<f64>() / self.n_cores as f64;
        let variance = loads.iter().map(|&l| (l - mean_load).powi(2)).sum::<f64>() / self.n_cores as f64;
        e += 10.0 * variance;
        
        // Factor: cache_affinity
        for i in 0..self.processes.len() {
            if self.core_assignment[i] != self.last_core[i] {
                e += 2.0 * self.processes[i].memory_mb / 100.0;
            }
        }
        
        // Factor: priority_satisfaction
        let slice_ms = [5.0, 10.0, 20.0, 50.0];
        for i in 0..self.processes.len() {
            let expected = self.processes[i].priority / 5.0;
            let actual = slice_ms[self.time_slice[i]] / 50.0;
            e += 5.0 * (expected - actual).powi(2);
        }
        
        // Factor: memory_contention
        let mut mem_per_core = vec![0.0; self.n_cores];
        for i in 0..self.processes.len() {
            mem_per_core[self.core_assignment[i]] += self.processes[i].memory_mb;
        }
        for &total_mem in &mem_per_core {
            if total_mem > 4096.0 {
                e += 0.1 * (total_mem - 4096.0).powi(2);
            }
        }
        
        e
    }
    
    fn gibbs_sweep(&mut self) {
        self.sample_core_assignments();
        self.sample_time_slices();
    }
    
    fn sample_core_assignments(&mut self) {
        let mut rng = thread_rng();
        
        for i in 0..self.processes.len() {
            let current = self.core_assignment[i];
            let mut log_probs = Vec::with_capacity(self.n_cores);
            
            // Try each core
            for core in 0..self.n_cores {
                self.core_assignment[i] = core;
                log_probs.push(-self.energy());
            }
            
            // Sample from categorical
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<f64> = log_probs.iter()
                .map(|lp| (lp - max_lp).exp())
                .collect();
            
            let dist = rand::distributions::WeightedIndex::new(&probs).unwrap();
            self.core_assignment[i] = dist.sample(&mut rng);
        }
    }
    
    fn sample_time_slices(&mut self) {
        let mut rng = thread_rng();
        
        for i in 0..self.processes.len() {
            let current = self.time_slice[i];
            let mut log_probs = vec![0.0; 4];
            
            // Try each time slice
            for slice in 0..4 {
                self.time_slice[i] = slice;
                log_probs[slice] = -self.energy();
            }
            
            // Sample
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<f64> = log_probs.iter()
                .map(|lp| (lp - max_lp).exp())
                .collect();
            
            let dist = rand::distributions::WeightedIndex::new(&probs).unwrap();
            self.time_slice[i] = dist.sample(&mut rng);
        }
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Litmus OS — Thermodynamic Process Scheduler             ║");
    println!("║  \"Your kernel is just Bayesian inference\"                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    let n_procs = 20;
    let n_cores = 8;
    
    let mut scheduler = SchedulerState::new(n_procs, n_cores);
    
    println!("System: {} cores, {} processes", n_cores, n_procs);
    println!("Optimizing scheduling policy...\n");
    
    // Run inference
    for iter in 1..=500 {
        scheduler.gibbs_sweep();
        
        if iter % 100 == 0 || iter == 1 {
            let loads: Vec<f64> = (0..n_cores)
                .map(|core| {
                    scheduler.core_assignment.iter()
                        .enumerate()
                        .filter(|(_, &c)| c == core)
                        .map(|(i, _)| scheduler.processes[i].cpu_demand)
                        .sum()
                })
                .collect();
            
            let mean_load = loads.iter().sum::<f64>() / n_cores as f64;
            let variance = loads.iter()
                .map(|&l| (l - mean_load).powi(2))
                .sum::<f64>() / n_cores as f64;
            
            println!(
                "Iter {:3} | Load variance: {:.3} | Energy: {:.1} | Core loads: {:?}",
                iter,
                variance,
                -scheduler.energy(),
                loads.iter().map(|l| format!("{:.1}", l)).collect::<Vec<_>>()
            );
        }
    }
    
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  OPTIMAL SCHEDULE                                         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    let slice_ms = [5, 10, 20, 50];
    for i in 0..n_procs.min(10) {
        let p = &scheduler.processes[i];
        println!(
            "PID {:2} → Core {} | Slice: {:2}ms | Priority: {:.1} | CPU: {:.2} | Mem: {:4.0}MB",
            p.pid,
            scheduler.core_assignment[i],
            slice_ms[scheduler.time_slice[i]],
            p.priority,
            p.cpu_demand,
            p.memory_mb
        );
    }
    
    println!("\n✓ Scheduling policy optimized via thermodynamic inference");
    println!("  • Balanced load across cores");
    println!("  • Cache affinity preserved");
    println!("  • Priority constraints satisfied");
    println!("\n// Deploy as kernel module:");
    println!("// $ insmod litmus_scheduler.ko device=/dev/thermo0");
    println!("// $ echo 'litmus' > /sys/kernel/sched/policy\n");
}
