import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

# --- Configuration ---
# Scale up problem
N_WORKERS = 100 # Number of workers
N_FEATURES = 100 # Dimension of optimimzation variable (x)
N_SAMPLES = 100 # m_i: Number of samples per worker
RHO = 1.0 # Penalty parameter of ADMM
MAX_TIME = 10.0 # Max time of simulation
MAX_ITER = 100 # Max number of iterations

# Straggler config
# 90% workers: Normal (0.01s ~ 0.05s)
# 10% workers: Straggler (0.2s ~ 0.5s) -> 10x slower
STRAGGLER_RATIO = 0.1


def get_random_delay(rng):
    if rng.random() < STRAGGLER_RATIO: # 각 agent는 시작할 때 난수를 하나 뽑읍 (random) + 만약 그것이 STRAGGLER_RATIO보다 작으면,
        return rng.uniform(0.2, 0.5) # 그 agent는 0.2 ~ 0.5초의 delay를 가짐 (Stragger)
    else:
        return rng.uniform(0.01, 0.05) # 그 외의 agent는 0.01 ~ 0.05초의 delay를 가짐 (Normal worker)

class Worker:
    def __init__(self, node_id, A, b, rho, n_features, learning_rate=0.01, local_iters=10):
        self.id = node_id
        self.A = A
        self.b = b
        self.rho = rho
        self.n_features = n_features
        self.lr = learning_rate
        self.local_iters = local_iters
        
        self.x = np.zeros(n_features)
        self.y = np.zeros(n_features)
        self.last_z = np.zeros(n_features) # Last z received
        self.current_k = 0
        
        # Precompute for analytical solution (optional, for speed)
        # Using iterative GD as per previous code to act like "work"
        
    def local_step(self, z_global, k_global):
        # Update internal state if new k received
        if k_global > self.current_k:
            self.last_z = np.copy(z_global)
            self.current_k = k_global
            # Standard ADMM dual update happens at the END of previous round
            # But here we do simplified update:
            # y = y + rho * (x - z)
            # In SR-ADMM paper, y update helps memory.
            # We follow the same logic as previous simple SR-ADMM code
            self.y = self.y + self.rho * (self.x - self.last_z)

        # Primal Update: x = argmin ...
        # (A'A + rho I) x = A'b + rho*z - y
        # We simulate this cost.
        # For simulation accuracy, we compute closed form here to be robust
        # H = A.T @ A + rho * I
        # target = A.T @ b + rho * z - y
        # but to keep it simple and stable:
        H = self.A.T @ self.A + self.rho * np.eye(self.n_features)
        target = self.A.T @ self.b + self.rho * self.last_z - self.y
        self.x = np.linalg.solve(H, target)
        
        # Compute s
        s = self.x + (self.y / self.rho)
        return s, self.current_k

class SimulationEnv:
    def __init__(self, mode, n_workers, n_min, inputs):
        self.mode = mode # 'SR-ADMM' or 'STD-ADMM'
        self.n_workers = n_workers
        self.n_min = n_min
        self.workers = []
        self.rng = np.random.default_rng(42)
        
        # Initialize Workers
        true_x, As, bs = inputs
        self.true_x = true_x
        
        for i in range(n_workers):
            self.workers.append(Worker(i, As[i], bs[i], RHO, N_FEATURES))
            
        # Central State
        self.global_k = 0
        self.global_z = np.zeros(N_FEATURES)
        self.collected_s = {} # map: worker_id -> s
        self.contributors_current_k = set()
        
        # Event Queue: (time, event_type, worker_id, data)
        self.pq = []
        self.current_time = 0.0
        
        # History
        self.history_time = []
        self.history_error = []
        self.history_k = []

    def push_event(self, timestamp, entropy, evt_type, worker_id, data):
        # entropy is just a tie-breaker
        heapq.heappush(self.pq, (timestamp, entropy, evt_type, worker_id, data))

    def run(self):
        # Start Iteration 1
        self.start_new_iteration()
        
        event_counter = 0
        
        while self.current_time < MAX_TIME and self.global_k < MAX_ITER:
            if not self.pq:
                break
                
            t, _, type, wid, data = heapq.heappop(self.pq)
            self.current_time = t
            
            if type == 'WORKER_DONE':
                # Process worker result
                s_val, k_w = data
                
                # Only accept results for current global_k (or older? No, sync model)
                # In SR-ADMM, we only care about current k
                if k_w == self.global_k:
                    if wid not in self.contributors_current_k:
                        self.collected_s[wid] = s_val
                        self.contributors_current_k.add(wid)
                        
                        # Check termination condition
                        if self.check_round_completion():
                            self.update_global_model()
                            self.record_metrics()
                            self.start_new_iteration()
            
            event_counter += 1
            
        return self.history_time, self.history_error

    def start_new_iteration(self):
        self.global_k += 1
        self.contributors_current_k = set()
        # In Standard ADMM, we clear collected_s? 
        # Actually SR-ADMM keeps history. 
        # Standard ADMM must clear to be "Synchronous"
        if self.mode == 'STD-ADMM':
            self.collected_s = {}
            
        # Trigger all workers
        stragglers = []
        for i in range(self.n_workers):
            # Calculate delay
            delay = get_random_delay(self.rng)
            
            # Identify straggler (Delay > 0.1s implies straggler logic 0.2~0.5s)
            if delay > 0.1:
                stragglers.append(i)
                
            finish_time = self.current_time + delay
            
            # Worker computation (instant in virtual time logic, result applied at finish_time)
            # We pass current state to worker
            s_val, k_out = self.workers[i].local_step(self.global_z, self.global_k)
            
            # Push completion event
            self.push_event(finish_time, i, 'WORKER_DONE', i, (s_val, k_out))
            
        # Log stragglers
        if self.mode == 'SR-ADMM':
            print(f"[{self.mode} Iter {self.global_k}] Stragglers ({len(stragglers)}): {stragglers}")

    def check_round_completion(self):
        count = len(self.contributors_current_k)
        if self.mode == 'SR-ADMM':
            return count >= self.n_min
        else: # STD-ADMM
            return count >= self.n_workers

    def update_global_model(self):
        # Aggregation
        if self.mode == 'SR-ADMM':
            # Average of all historically collected s
            # (In this impl, collected_s tracks latest s from each worker)
            # For unvisited workers, it might be 0 (if never visited) or old value
            # Only average over workers who have contributed at least once?
            # SR-ADMM paper: sum over N^{1:k} (all who ever contributed)
            valid_s = [self.collected_s[i] for i in self.collected_s if np.any(self.collected_s[i])]
            if valid_s:
                self.global_z = np.mean(valid_s, axis=0)
        else:
            # Std ADMM: Average of current round contributors (which is ALL)
            s_values = list(self.collected_s.values())
            self.global_z = np.mean(s_values, axis=0)
            
    def record_metrics(self):
        err = np.linalg.norm(self.global_z - self.true_x)
        self.history_time.append(self.current_time)
        self.history_error.append(err)
        self.history_k.append(self.global_k)
        
def generate_data():
    rng = np.random.default_rng(42)
    true_x = rng.standard_normal(N_FEATURES)
    As = []
    bs = []
    for _ in range(N_WORKERS):
        A = rng.standard_normal((N_SAMPLES, N_FEATURES))
        noise = 0.1 * rng.standard_normal(N_SAMPLES)
        b = A @ true_x + noise
        As.append(A)
        bs.append(b)
    return true_x, As, bs

def main():
    print(f"Generating Data (N={N_WORKERS}, D={N_FEATURES})...")
    data = generate_data()
    
    # 1. Run SR-ADMM (N_min = 60% of workers? Paper uses smaller fraction sometimes, let's say 24/30 -> 80%)
    # Let's try N_min = 80 for 100 workers (Aggressive straggler resilience)
    n_min_sr = int(N_WORKERS * 0.8)
    print(f"Running SR-ADMM (N_min={n_min_sr})...")
    env_sr = SimulationEnv('SR-ADMM', N_WORKERS, n_min_sr, data)
    t_sr, err_sr = env_sr.run()
    print(f"SR-ADMM finished in {t_sr[-1]:.2f}s (virtual), Final Error: {err_sr[-1]:.4f}")
    
    # 2. Run Standard ADMM
    print(f"Running Standard ADMM (Waiting for all {N_WORKERS})...")
    env_std = SimulationEnv('STD-ADMM', N_WORKERS, N_WORKERS, data)
    t_std, err_std = env_std.run()
    print(f"Std-ADMM finished in {t_std[-1]:.2f}s (virtual), Final Error: {err_std[-1]:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_sr, err_sr, 'b-o', label=f'SR-ADMM (N_min={n_min_sr})', linewidth=2, markersize=4)
    plt.plot(t_std, err_std, 'r--s', label='Standard ADMM (Sync)', linewidth=2, markersize=4)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('L2 Error ||z - x*||')
    plt.title(f'Convergence Comparison (Straggler Ratio={STRAGGLER_RATIO})')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    output_path = '../results/SR-ADMM.png'
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == '__main__':
    main()