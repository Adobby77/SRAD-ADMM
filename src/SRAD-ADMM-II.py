import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve

# 1. 시뮬레이션 파라미터 설정 (논문 설정 반영)
N = 30           # 총 노드 수 [cite: 541]
N_min = 20        # 최소 참여 노드 수 (0.6N) [cite: 525]
total_samples = 4177
n_features = 7
K = 100          # 반복 횟수 [cite: 516]
np.random.seed(42)

# 2. 데이터 생성 (Abalone 데이터셋 모사) [cite: 517]
A_full = np.random.randn(total_samples, n_features)
x_true = np.random.randn(n_features)
b_full = A_full @ x_true + np.random.normal(0, 0.1, total_samples)

# Dirichlet 분포를 이용한 데이터 비균등 분배 (beta=0.5) [cite: 518]
def dirichlet_partition(data_size, n_parties, beta=0.5):
    proportions = np.random.dirichlet([beta] * n_parties)
    proportions = (proportions * data_size).astype(int)
    diff = data_size - proportions.sum()
    for i in range(abs(diff)):
        proportions[i % n_parties] += 1 if diff > 0 else -1
    
    indices = np.random.permutation(data_size)
    party_indices = []
    curr = 0
    for p in proportions:
        party_indices.append(indices[curr:curr+p])
        curr += p
    return party_indices

party_indices = dirichlet_partition(total_samples, N, beta=0.5)
node_data = []
for idx in party_indices:
    node_data.append((A_full[idx], b_full[idx]))

# 3. 립시츠 상수 및 Step-size(rho) 계산 [cite: 374, 540]
Ls = []
for Ai, bi in node_data:
    if len(Ai) == 0:
        Ls.append(0)
        continue
    L_i = np.linalg.eigvalsh(2 * Ai.T @ Ai).max()
    Ls.append(L_i)

L = max(Ls)
rho = 2 * L + 2  # 논문의 수렴 조건 반영 [cite: 374]

def objective(z, data):
    total_f = 0
    for Ai, bi in data:
        if len(Ai) == 0: continue
        total_f += np.linalg.norm(Ai @ z - bi)**2
    return total_f

# 지연 노드(Straggler) 시뮬레이션을 위한 시간 생성 함수 [cite: 6]
def get_comp_time():
    return np.random.exponential(scale=1.0) + 0.01

# --- 알고리즘 1: CC-ADMM (동기 방식) --- [cite: 37]
def run_cc_admm(node_data, rho, K):
    n = n_features
    z = np.random.uniform(-0.5, 0.5, n)
    x = [z.copy() for _ in range(N)]
    y = [np.zeros(n) for _ in range(N)]
    history = []
    current_time = 0
    
    for k in range(K):
        comp_times = [get_comp_time() for _ in range(N)]
        for i in range(N):
            Ai, bi = node_data[i]
            if len(Ai) == 0: continue
            x[i] = solve(2 * Ai.T @ Ai + rho * np.eye(n), 2 * Ai.T @ bi + rho * z - y[i])
            y[i] = y[i] + rho * (x[i] - z)
        
        z = np.mean([xi + yi/rho for xi, yi in zip(x, y)], axis=0)
        current_time += max(comp_times) # 모든 노드를 기다림 (동기)
        history.append({'time': current_time, 'obj': objective(z, node_data)})
    return history

# --- 알고리즘 2: SRAD-ADMM-II (비동기 방식) --- [cite: 260]
def run_srad_admm_ii(node_data, rho, K, N_min):
    n = n_features
    z_global = np.random.uniform(-0.5, 0.5, n)
    x = [z_global.copy() for _ in range(N)]
    y = [-rho * x[i] for i in range(N)] # 논문 초기화 방식 [cite: 143]
    s = [xi + yi/rho for xi, yi in zip(x, y)]
    
    node_status = [{'time_ready': 0} for _ in range(N)]
    history = []
    
    for k_iter in range(K):
        finishing_times = []
        for i in range(N):
            comp_time = get_comp_time()
            arrival = node_status[i]['time_ready'] + comp_time
            finishing_times.append((arrival, i))
        
        finishing_times.sort()
        # N_min번째 노드가 도착했을 때 즉시 업데이트 (비동기 탄력성) [cite: 69]
        advance_time, _ = finishing_times[N_min-1]
        participating = [idx for t, idx in finishing_times if t <= advance_time]
        
        # 증분 업데이트 로직 반영 [cite: 158]
        z_global = np.mean([s[idx] for idx in participating], axis=0)
        
        for idx in participating:
            Ai, bi = node_data[idx]
            if len(Ai) == 0: continue
            x[idx] = solve(2 * Ai.T @ Ai + rho * np.eye(n), 2 * Ai.T @ bi + rho * z_global - y[idx])
            y[idx] = y[idx] + rho * (x[idx] - z_global)
            s[idx] = x[idx] + y[idx]/rho
            node_status[idx]['time_ready'] = advance_time
            
        history.append({'time': advance_time, 'obj': objective(z_global, node_data)})
    return history

# 4. 실행 및 그래프 생성
print("시뮬레이션 실행 중...")
hist_cc = run_cc_admm(node_data, rho, K)
hist_srad = run_srad_admm_ii(node_data, rho, K, N_min)

df_cc = pd.DataFrame(hist_cc)
df_srad = pd.DataFrame(hist_srad)

plt.figure(figsize=(10, 6))
plt.plot(df_cc['time'], df_cc['obj'], label='CC-ADMM (Sync)', color='blue', linestyle='--')
plt.plot(df_srad['time'], df_srad['obj'], label='SRAD-ADMM-II (Async)', color='red')
plt.yscale('log')
plt.xlabel('Time (seconds)')
plt.ylabel('Objective Value F^k')
plt.title('Convergence: CC-ADMM vs SRAD-ADMM-II')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# 결과 저장 (이 코드를 실행하면 파일이 생깁니다)
save_dir = '../results/'
plt.savefig(save_dir + 'SRAD-ADMM2(with comparison).png')
df_results = pd.concat([df_cc.add_prefix('cc_'), df_srad.add_prefix('srad_')], axis=1)
df_results.to_csv(save_dir + 'admm_simulation_results.csv', index=False)
print(f"완료! {save_dir}SRAD-ADMM2(with comparison).png 파일을 확인하세요.")