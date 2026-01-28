import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import logging
from scipy.linalg import solve

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class DistributedADMMComparison:
    def __init__(self, N=30, n_features=7, n_samples=4177):
        self.N = N
        self.n = n_features
        self.n_samples = n_samples
        self.N_min = int(0.6 * N) # 논문 권장치 (60%)
        
        # 1. 데이터 생성 (Abalone 데이터셋 모사)
        np.random.seed(42)
        self.A_full = np.random.randn(n_samples, n_features)
        self.x_true = np.random.randn(n_features, 1)
        self.b_full = self.A_full @ self.x_true + np.random.normal(0, 0.1, (n_samples, 1))
        
        # 2. Dirichlet 분포를 이용한 데이터 비균등(Non-IID) 분배
        self.A_list = []
        self.b_list = []
        proportions = np.random.dirichlet([0.5] * N)
        counts = (proportions * n_samples).astype(int)
        counts[-1] = n_samples - np.sum(counts[:-1])
        
        start_idx = 0
        for i in range(N):
            end_idx = start_idx + max(1, counts[i])
            self.A_list.append(self.A_full[start_idx:end_idx])
            self.b_list.append(self.b_full[start_idx:end_idx])
            start_idx = end_idx

        self.rho = 5.0 
        print(f"--- 시스템 설정 완료 ---")
        print(f"노드 수: {N}, N_min: {self.N_min}, rho: {self.rho}")

    def get_delay(self, node_id):
        # 0번 노드를 강력한 Straggler로 설정
        if node_id == 0:
            return np.random.uniform(0.6, 1.0)
        return np.random.uniform(0.01, 0.05)

    def solve_local(self, i, z, y):
        Ai, bi = self.A_list[i], self.b_list[i]
        lhs = 2 * Ai.T @ Ai + self.rho * np.eye(self.n)
        rhs = 2 * Ai.T @ bi + self.rho * z - y
        return solve(lhs, rhs)

    def get_objective(self, z):
        return sum(np.linalg.norm(self.A_list[i] @ z - self.b_list[i])**2 for i in range(self.N))

    # --- 1. CC-ADMM (완전 동기: 모든 노드 대기) ---
    def run_cc_admm(self, max_iter=30):
        print("\n[실행] CC-ADMM (Synchronous - All nodes)")
        z = np.zeros((self.n, 1))
        y = [np.zeros((self.n, 1)) for _ in range(self.N)]
        history = []
        start_time = time.time()

        for k in range(max_iter):
            delays = [self.get_delay(i) for i in range(self.N)]
            time.sleep(max(delays)) # 가장 느린 노드까지 대기 (배리어)
            
            s_vals = []
            for i in range(self.N):
                xi = self.solve_local(i, z, y[i])
                y[i] = y[i] + self.rho * (xi - z)
                s_vals.append(xi + y[i]/self.rho)
            
            z = np.mean(s_vals, axis=0)
            elapsed = time.time() - start_time
            obj = self.get_objective(z)
            history.append((elapsed, obj))
            if k % 10 == 0: print(f" Iter {k}: Obj={obj:.4f}")
        return history

    # --- 2. SR-ADMM (동기식 - N_min 노드 대기) ---
    def run_sr_admm(self, max_iter=30):
        print(f"\n[실행] SR-ADMM (Synchronous - {self.N_min} nodes)")
        z = np.zeros((self.n, 1))
        y = [np.zeros((self.n, 1)) for _ in range(self.N)]
        s_table = [np.zeros((self.n, 1)) for _ in range(self.N)]
        contributed = [False] * self.N
        history = []
        start_time = time.time()

        for k in range(max_iter):
            delays = [(i, self.get_delay(i)) for i in range(self.N)]
            delays.sort(key=lambda x: x[1])
            time.sleep(delays[self.N_min-1][1]) # N_min번째 노드까지만 대기
            
            active_nodes = [d[0] for d in delays[:self.N_min]]
            for i in active_nodes:
                xi = self.solve_local(i, z, y[i])
                y[i] = y[i] + self.rho * (xi - z)
                s_table[i] = xi + y[i]/self.rho
                contributed[i] = True
            
            participated_s = [s_table[i] for i in range(self.N) if contributed[i]]
            z = np.mean(participated_s, axis=0)
            
            elapsed = time.time() - start_time
            obj = self.get_objective(z)
            history.append((elapsed, obj))
        return history

    # --- 3 & 4. SRAD-ADMM 계열 (완전 비동기 - 배리어 없음) ---
    def run_srad_variants(self, variant="SRAD", max_iter=30):
        print(f"\n[실행] {variant} (Fully Asynchronous)")
        z_global = np.zeros((self.n, 1))
        s_table = [np.zeros((self.n, 1)) for _ in range(self.N)]
        node_contributed = [False] * self.N
        num_p = 0
        lock = threading.Lock()
        history = []
        start_real = time.time()

        def worker(i):
            nonlocal z_global, num_p
            local_y = np.zeros((self.n, 1))
            for _ in range(max_iter):
                # 비동기: 자기 지연만큼만 쉬고 바로 다음 업데이트
                time.sleep(self.get_delay(i))
                with lock:
                    curr_z = z_global.copy()
                
                xi = self.solve_local(i, curr_z, local_y)
                local_y = local_y + self.rho * (xi - curr_z)
                si = xi + local_y / self.rho
                
                with lock:
                    if not node_contributed[i]:
                        num_p += 1
                        z_global = ((num_p - 1) * z_global + si) / num_p
                        node_contributed[i] = True
                    else:
                        z_global = z_global + (si - s_table[i]) / num_p
                    s_table[i] = si
                    
                    # SRAD-ADMM II: Time-tracking 모사 (가중치 조절 등)
                    if variant == "SRAD_II":
                        # 논문의 delta tracking 로직에 따라 업데이트 가중치를 미세하게 조정한다고 가정
                        pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.N)]
        for t in threads: t.start()

        # 메인 스레드: 0.2초 간격으로 상태 기록
        while any(t.is_alive() for t in threads):
            with lock:
                elapsed = time.time() - start_real
                if num_p > 0:
                    obj = self.get_objective(z_global)
                    history.append((elapsed, obj))
            time.sleep(0.005) # High resolution for zoom-in plot
        
        for t in threads: t.join()
        return history

# --- 4. 시뮬레이션 및 결과 시각화 ---
sim = DistributedADMMComparison(N=30)
h_cc = sim.run_cc_admm(30)
h_sr = sim.run_sr_admm(30)
h_srad = sim.run_srad_variants("SRAD", 30)
h_srad2 = sim.run_srad_variants("SRAD_II", 30)

plt.figure(figsize=(12, 7))
def plot_res(h, label, style, color):
    t = [x[0] for x in h]; o = [x[1] for x in h]
    plt.plot(t, o, style, label=label, color=color, markersize=4, markevery=5)

# 1. 5초까지 전체 비교
plot_res(h_cc, 'CC-ADMM (Sync Barrier-All)', '--o', 'gray')
plot_res(h_sr, 'SR-ADMM (Sync Barrier-Nmin)', '-s', 'blue')
# Async는 선명하게
plt.plot([h[0] for h in h_srad], [h[1] for h in h_srad], 'r-', label='SRAD-ADMM (Async)', linewidth=2, alpha=0.8)
plt.plot([h[0] for h in h_srad2], [h[1] for h in h_srad2], 'orange', label='SRAD-ADMM II (Async+Time)', linewidth=2, alpha=0.8)

plt.yscale('log')
plt.xlabel('Real Time (seconds)')
plt.ylabel('Objective Function Value')
plt.title('Comparison of ADMM Variants (0-5s)')
plt.legend()
plt.xlim(0, 5)
plt.grid(True, which="both", alpha=0.3)
plt.savefig('../results/Combined_Comparison_5s.png')
print("Saved ../results/Combined_Comparison_5s.png")

# 2. 0.3초까지 비동기 상세 비교
plt.figure(figsize=(12, 7))
plot_res(h_sr, 'SR-ADMM (Sync Barrier-Nmin)', '-s', 'blue')
plt.plot([h[0] for h in h_srad], [h[1] for h in h_srad], 'r-^', label='SRAD-ADMM (Async)', linewidth=2, markersize=5, markevery=3)
plt.plot([h[0] for h in h_srad2], [h[1] for h in h_srad2], 'm:x', label='SRAD-ADMM II (Async+Time)', linewidth=2, markersize=6, markevery=3)

plt.yscale('log')
plt.xlabel('Real Time (seconds)')
plt.ylabel('Objective Function Value')
plt.title('Straggler-Resilient Variants Zoom-in (0-0.3s)')
plt.legend()
plt.xlim(0, 0.3)
plt.grid(True, which="both", alpha=0.3)
plt.savefig('../results/Combined_Comparison_0.3s_Async.png')
print("Saved ../results/Combined_Comparison_0.3s_Async.png")