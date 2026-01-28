import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
import os

# 1. 시뮬레이션 파라미터 설정
N = 30 
N_min = 20 # 0.6N
n_features = 10
total_samples = 3000
K = 100
rho = 10.0 # 수렴 안정성을 위해 조정
np.random.seed(0)

# 2. 데이터 생성 및 Dirichlet 분배 (비균등 환경 모사)
A_full = np.random.randn(total_samples, n_features)
x_true = np.random.randn(n_features)
b_full = A_full @ x_true + np.random.normal(0, 0.1, total_samples)

def dirichlet_partition(data_size, n_parties, beta=0.5):
    proportions = np.random.dirichlet([beta] * n_parties)
    proportions = (proportions * data_size).astype(int)
    indices = np.random.permutation(data_size)
    party_indices = []
    curr = 0
    for p in proportions:
        party_indices.append(indices[curr:curr+p])
        curr += p
    return party_indices

party_indices = dirichlet_partition(total_samples, N, beta=0.5)
node_data = [(A_full[idx], b_full[idx]) for idx in party_indices]

def get_comp_time():
    # 지연 노드(Straggler) 효과를 극대화하기 위한 지수 분포 사용
    return np.random.exponential(scale=1.0) + 0.1

def objective(z):
    # 전역 목적 함수 F(z) = sum ||Ai*z - bi||^2
    err = 0
    for Ai, bi in node_data:
        if len(Ai) > 0: err += np.linalg.norm(Ai @ z - bi)**2
    return err

# --- 알고리즘 1: Standard ADMM (Sync) ---
def run_std_admm(K):
    z = np.zeros(n_features)
    x = [z.copy() for _ in range(N)]
    y = [np.zeros(n_features) for _ in range(N)]
    history = [{'time': 0, 'obj': objective(z)}]
    current_time = 0
    
    for k in range(K):
        delays = [get_comp_time() for _ in range(N)]
        for i in range(N):
            Ai, bi = node_data[i]
            if len(Ai) == 0: continue
            # x-update
            x[i] = solve(2 * Ai.T @ Ai + rho * np.eye(n_features), 2 * Ai.T @ bi + rho * z - y[i])
            # y-update
            y[i] = y[i] + rho * (x[i] - z)
        
        # z-update (Full Average)
        z = np.mean([xi + yi/rho for xi, yi in zip(x, y)], axis=0)
        current_time += max(delays) # 모든 노드 대기
        history.append({'time': current_time, 'obj': objective(z)})
    return history

# --- 알고리즘 2: SRAD-ADMM (Async Original) ---
def run_srad_admm(K, N_min):
    z = np.zeros(n_features)
    x = [z.copy() for _ in range(N)]
    y = [-rho * x[i] for i in range(N)] # SRAD 논문 초기화
    s = [xi + yi/rho for xi, yi in zip(x, y)]
    
    node_ready_time = np.zeros(N)
    history = [{'time': 0, 'obj': objective(z)}]
    
    for k in range(K):
        # 각 노드가 다음 연산을 마칠 시간을 계산
        finish_times = []
        for i in range(N):
            finish_times.append((node_ready_time[i] + get_comp_time(), i))
        
        finish_times.sort()
        # N_min번째 빠른 노드의 시간으로 전진 (비동기 핵심)
        advance_time, _ = finish_times[N_min-1]
        winners = [idx for t, idx in finish_times if t <= advance_time]
        
        # 증분 업데이트 (부분 평균)
        for idx in winners:
            Ai, bi = node_data[idx]
            if len(Ai) == 0: continue
            # x, y 업데이트
            x[idx] = solve(2 * Ai.T @ Ai + rho * np.eye(n_features), 2 * Ai.T @ bi + rho * z - y[idx])
            y[idx] = y[idx] + rho * (x[idx] - z)
            s[idx] = x[idx] + y[idx]/rho
            node_ready_time[idx] = advance_time
        
        # 최신 s값들을 모아 z 업데이트
        z = np.mean(s, axis=0)
        history.append({'time': advance_time, 'obj': objective(z)})
    return history

# 실행 및 시각화
print("Standard ADMM vs SRAD-ADMM 시뮬레이션 중...")
h_std = run_std_admm(K)
h_srad = run_srad_admm(K, N_min)

df_std = pd.DataFrame(h_std)
df_srad = pd.DataFrame(h_srad)

plt.figure(figsize=(10, 6))
plt.plot(df_std['time'], df_std['obj'], 'b--o', label='Standard ADMM (Sync)', markersize=4)
plt.plot(df_srad['time'], df_srad['obj'], 'r-s', label='SRAD-ADMM (Async)', markersize=4)
plt.yscale('log'); plt.xlabel('Virtual Time (seconds)'); plt.ylabel('Global Objective F(z)')
plt.title('Comparison: Straggler Resilience in Distributed ADMM')
plt.legend(); plt.grid(True, alpha=0.3); plt.xlim(0, 50)

# 결과 저장
os.makedirs('../results/', exist_ok=True)
plt.savefig('../results/SRAD-ADMM(50).png')
print("완료! ../results/SRAD-ADMM(50).png 파일을 확인하세요.")