"""
=============================================================================
SRAD-ADMM Simulation for Distributed Least Squares — FINAL VERSION
=============================================================================
Based on: He et al., "Straggler-Resilient Asynchronous ADMM for
          Distributed Consensus Optimization", IEEE TSP, 2025.

Algorithms:
  1. CC-ADMM      — Classical Centralized ADMM (Eq. 2)
  2. SR-ADMM      — Straggler-Resilient ADMM (Eq. 15, Algorithms 1-2)
  3. SRAD-ADMM    — SR Async Decentralized ADMM (Eq. 32, Algorithms 3-4)
  4. SRAD-ADMM-II — Extension with time tracking (Algorithm 5)

Wall-time simulation model:
  Each node i has per-iteration compute time t_i^k drawn from:
    - Normal:    t_i^k ~ U(0.8, 1.2)
    - Straggler: t_i^k ~ U(3.0, 8.0)   with probability p_straggler=0.3

  CC-ADMM:     All N nodes must finish → wall_time += max_i(t_i^k)
  SR-ADMM:     Wait for Nmin-th fastest → wall_time += sort(t)[Nmin-1]
  SRAD-ADMM:   Decentralized async — nodes process concurrently,
               z is updated incrementally as nodes finish.
               Wall time = time when Nmin-th node finishes AND
               z has been propagated.
  SRAD-ADMM-II: Same as SRAD with dynamic termination using δ.

Message dropout (Section VI, Fig 4-5):
  Each node's message is dropped with probability p_drop (default 0.01).
=============================================================================
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA GENERATION & DISTRIBUTION
# =============================================================================

def generate_data(n_samples=4177, n_features=7):
    """Generate Abalone-like synthetic data (논문 Section VI)."""
    rng = np.random.RandomState(2025)
    A = rng.randn(n_samples, n_features)
    scales = np.array([0.1, 0.08, 0.1, 0.5, 0.2, 0.1, 0.15])
    A = A * scales + np.array([0.5, 0.4, 0.13, 0.83, 0.37, 0.18, 0.27])
    x_true = rng.randn(n_features) * 2
    b = A @ x_true + rng.randn(n_samples) * 0.5
    return A, b

def split_random(A, b, N):
    idx = np.random.permutation(A.shape[0])
    parts = np.array_split(idx, N)
    return [(A[p], b[p]) for p in parts]

def split_dirichlet(A, b, N, beta=0.5):
    order = np.argsort(b)
    chunks = np.array_split(order, N * 5)
    props = np.random.dirichlet(np.ones(N) * beta, len(chunks))
    bins = [[] for _ in range(N)]
    for ci, ch in enumerate(chunks):
        alloc = (props[ci] * len(ch)).astype(int)
        alloc[np.argmax(alloc)] += len(ch) - alloc.sum()
        s = 0
        for ni in range(N):
            bins[ni].extend(ch[s:s+alloc[ni]].tolist())
            s += alloc[ni]
    out = []
    for idx_list in bins:
        idx = np.array(idx_list) if len(idx_list) > 0 else np.array([0])
        out.append((A[idx], b[idx]))
    return out

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_rho(data):
    """ρ = 2·max_i{L_i} + 2 — Proposition V.5"""
    Lmax = max(float(np.max(np.linalg.eigvalsh(Ai.T @ Ai))) for Ai, _ in data)
    return 2.0 * Lmax + 2.0

def obj_val(data, z):
    """F(z) = Σ_i (1/2)||A_i z - b_i||²"""
    return sum(0.5 * np.dot(Ai @ z - bi, Ai @ z - bi) for Ai, bi in data)

def optimal_solution(data):
    """x* = (Σ A_i^T A_i)^{-1}(Σ A_i^T b_i)"""
    n = data[0][0].shape[1]
    H, g = np.zeros((n, n)), np.zeros(n)
    for Ai, bi in data:
        H += Ai.T @ Ai
        g += Ai.T @ bi
    return np.linalg.solve(H, g)

def rel_diff(z, z_star):
    """ε^k = ||z^k - z*|| / ||z*||"""
    return np.linalg.norm(z - z_star) / max(np.linalg.norm(z_star), 1e-12)

def sim_times(N, straggler_prob=0.3):
    """Per-node compute time for one iteration."""
    times = np.random.uniform(0.8, 1.2, N)
    for i in range(N):
        if np.random.random() < straggler_prob:
            times[i] = np.random.uniform(3.0, 8.0)
    return times

# =============================================================================
# 1. CC-ADMM — Eq. (2)
#
# Fully synchronous. All N nodes compute every iteration.
# Wall time: must wait for the slowest node → max(t_i^k)
#
# Updates:
#   x_i^k = (A_i^T A_i + ρI)^{-1}(A_i^T b_i - y_i^{k-1} + ρ z^{k-1})
#   y_i^k = y_i^{k-1} + ρ(x_i^k - z^{k-1})
#   z^k   = (1/N) Σ_i (x_i^k + y_i^k / ρ)
# =============================================================================

def cc_admm(data, rho, K, x_init):
    N, n = len(data), data[0][0].shape[1]
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    x = [xi.copy() for xi in x_init]
    y = [-rho * xi.copy() for xi in x_init]
    z = np.zeros(n)

    z_list, t_list = [z.copy()], [0.0]
    cum_time = 0.0

    for k in range(1, K + 1):
        times = sim_times(N, 0.3)
        cum_time += np.max(times)  # 가장 느린 노드 대기

        for i in range(N):
            x[i] = Q[i] @ (g[i] - y[i] + rho * z)
            y[i] = y[i] + rho * (x[i] - z)
        z = np.mean([x[i] + y[i] / rho for i in range(N)], axis=0)

        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list

# =============================================================================
# 2. SR-ADMM — Eq. (15), Algorithms 1-2
#
# Centralized, straggler-resilient. Server waits for Nmin fastest nodes.
# Remaining nodes' contributions from previous iterations are reused.
#
# Wall time: Nmin-th fastest node → sort(t)[Nmin-1]
#
# z^k = (1/|N^{1:k}|) Σ_{i∈N^{1:k}} s_i^{k_i}    — Eq. (13)
#
# Dual update:
#   Returning (i contributed before): y_i = y_i + ρ(x_i - z)
#   New (first contribution):         y_i = ρ(x_i - x_i^0)  — Eq. (10)
#
# Message dropout: node's update lost with prob p_drop
# =============================================================================

def sr_admm(data, rho, K, x_init, Nmin=2, drop_prob=0.0):
    N, n = len(data), data[0][0].shape[1]
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    x = [xi.copy() for xi in x_init]
    x0 = [xi.copy() for xi in x_init]
    y = [-rho * xi.copy() for xi in x_init]
    z = np.zeros(n)
    s = [np.zeros(n) for _ in range(N)]
    contribs = [0] * N
    ever = set()

    z_list, t_list = [z.copy()], [0.0]
    cum_time = 0.0

    for k in range(1, K + 1):
        times = sim_times(N, 0.3)
        order = np.argsort(times)
        sorted_times = times[order]

        # 서버는 Nmin개 결과가 도착할 때까지 기다림
        # Nmin번째 도착 시각이 deadline의 기준
        deadline_base = sorted_times[min(Nmin - 1, N - 1)]
        # 서버가 z를 계산하는 동안(~z 계산 시간) 추가 노드가 도착할 수 있음
        # bimodal 분포에서 빠른 그룹(U(0.8,1.2))은 거의 동시에 끝남
        # → deadline = 빠른 그룹 내 최대 시간 or Nmin번째 중 큰 값
        # 간단한 모델: straggler가 아닌 노드(t < 2.0)는 모두 참여
        fast_cutoff = 2.0  # 빠른 그룹과 straggler 경계
        deadline = max(deadline_base, min(fast_cutoff, sorted_times[-1]))
        cum_time += deadline

        # deadline 이전에 완료된 모든 노드가 참여
        Nk = [int(i) for i in order if times[i] <= deadline]
        if len(Nk) < Nmin:
            Nk = [int(order[j]) for j in range(Nmin)]

        # Message dropout (논문 Section VI, Fig 4-5)
        if drop_prob > 0:
            survived = [i for i in Nk if np.random.random() >= drop_prob]
            if len(survived) >= 1:
                Nk = survived

        for i in Nk:
            x[i] = Q[i] @ (g[i] - y[i] + rho * z)
            if contribs[i] > 0:
                y[i] = y[i] + rho * (x[i] - z)   # Returning
            else:
                y[i] = rho * (x[i] - x0[i])       # New — Eq. (10)
            s[i] = x[i] + y[i] / rho
            ever.add(i)
            contribs[i] += 1

        # z = (1/|N^{1:k}|) Σ_{i∈ever} s_i — Eq. (13)
        z = np.mean([s[i] for i in ever], axis=0) if ever else z
        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list

# =============================================================================
# 3. SRAD-ADMM — Eq. (32), Algorithms 3-4
#
# Decentralized, asynchronous. Key differences from SR-ADMM:
#   (a) z is updated INCREMENTALLY as each node finishes (not batch)
#   (b) No central server — each node computes its own z
#   (c) Conflict resolution: lower ID has priority (Rule 1, Eq. 34)
#
# Eq. (32) — incremental z-update:
#   Returning (i ∈ N^{1:k-1}):
#     z = z_prev + (1/N_j)(s_i^k − s_i^{prev})     [replacement, N_j same]
#   New (i ∉ N^{1:k-1}):
#     z = (1/N_j)((N_j−1)·z_prev + s_i^k)           [additive, N_j++]
#
# Wall-time model for async:
#   - 각 노드의 완료 시각 = cum_time_start + t_i (parallel computation)
#   - 완료 순서대로 z를 증분 업데이트
#   - Nmin번째 노드가 z를 업데이트한 시각이 iteration wall time
#   - 핵심: 다음 iteration의 시작은 현재 iteration의 완료와 overlap 가능
#     → cum_time += min(Nmin번째 시간, 이전 iteration 잔여 시간)
#
# x-update에서 z^{k-1} 사용:
#   논문 Eq. (22): x_i^k = argmin f_i(x) + (ρ/2)||x - z^{k-1}||²
#   모든 노드는 iteration 시작 시점의 z를 사용 (병렬 계산)
#   z_base는 merge 중간값이지 broadcast된 z가 아님
# =============================================================================

def srad_admm(data, rho, K, x_init, Nmin=2, drop_prob=0.0):
    N, n = len(data), data[0][0].shape[1]
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    x = [xi.copy() for xi in x_init]
    x0 = [xi.copy() for xi in x_init]
    y = [-rho * xi.copy() for xi in x_init]
    z = np.zeros(n)
    s_prev = [np.zeros(n) for _ in range(N)]
    contribs = [0] * N
    ever = set()
    Ncount = 0

    z_list, t_list = [z.copy()], [0.0]
    cum_time = 0.0

    # 비동기 overlap 모델을 위한 상태
    prev_iter_remaining = 0.0  # 이전 iteration에서 아직 계산 중인 느린 노드의 잔여 시간

    for k in range(1, K + 1):
        times = sim_times(N, 0.3)
        order = np.argsort(times)
        sorted_times = times[order]

        # --- 참여 노드 결정 ---
        # Nmin개 이상의 결과가 도착할 때까지 기다림
        # bimodal 분포에서 빠른 그룹(straggler가 아닌 노드)은 거의 동시에 끝남
        deadline_base = sorted_times[min(Nmin - 1, N - 1)]
        fast_cutoff = 2.0
        deadline = max(deadline_base, min(fast_cutoff, sorted_times[-1]))
        
        Nk = [int(order[j]) for j in range(N) if sorted_times[j] <= deadline]
        if len(Nk) < Nmin:
            Nk = [int(order[j]) for j in range(Nmin)]

        # Conflict resolution: lower ID first — Rule 1, Eq. (34)
        Nk.sort()

        # Message dropout (논문 Section VI, Fig 4-5)
        if drop_prob > 0:
            survived = [i for i in Nk if np.random.random() >= drop_prob]
            if len(survived) >= 1:
                Nk = survived

        # --- Wall time ---
        # 비동기 모델: deadline까지의 시간
        cum_time += deadline
        prev_iter_remaining = max(sorted_times[-1] - deadline, 0.0)

        # --- 증분 z-update — Eq. (32) ---
        z_base = z.copy()
        Nj = Ncount

        for nid in Nk:
            # x-update: z^{k-1} 사용 (이전 iteration의 최종 z)
            x_new = Q[nid] @ (g[nid] - y[nid] + rho * z)

            if contribs[nid] > 0:
                y_new = y[nid] + rho * (x_new - z)    # Returning
            else:
                y_new = rho * (x_new - x0[nid])        # New — Eq. (10)
            s_new = x_new + y_new / rho

            is_ret = (nid in ever)
            if Nj == 0:
                Nj = 1
                z_base = s_new.copy()
            elif is_ret:
                # Returning: replacement — Eq. (32) case 1
                z_base = z_base + (1.0 / Nj) * (s_new - s_prev[nid])
            else:
                # New: additive — Eq. (32) case 2
                Nj += 1
                z_base = (1.0 / Nj) * ((Nj - 1) * z_base + s_new)

            s_prev[nid] = s_new.copy()
            x[nid] = x_new.copy()
            y[nid] = y_new.copy()
            contribs[nid] += 1
            ever.add(nid)

        Ncount = len(ever)
        z = z_base.copy()
        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list

# =============================================================================
# 4. SRAD-ADMM-II — Algorithm 5
#
# Extension with dynamic termination (Section IV-D).
# Proceeds to next iteration when EITHER:
#   Condition 1: |N^k| ≥ Nmin                    (enough nodes)
#   Condition 2: |N^k| ≥ Nmin* AND |t - t_last| ≥ δ  (timeout)
#
# δ = time gap between the last two ITERATION COMPLETIONS
#     (inter-iteration gap, not intra-iteration)
# This prevents infinite waiting when Nmin = N and stragglers exist.
#
# Uses Nmin = N (논문 Section VI 설정) to maximize per-iteration info,
# but δ-based timeout allows early termination when stragglers delay.
# =============================================================================

def srad_admm_ii(data, rho, K, x_init, Nmin=None, Nmin_star=2, drop_prob=0.0):
    N, n = len(data), data[0][0].shape[1]
    if Nmin is None:
        Nmin = N
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    x = [xi.copy() for xi in x_init]
    x0 = [xi.copy() for xi in x_init]
    y = [-rho * xi.copy() for xi in x_init]
    z = np.zeros(n)
    s_prev = [np.zeros(n) for _ in range(N)]
    contribs = [0] * N
    ever = set()
    Ncount = 0

    z_list, t_list = [z.copy()], [0.0]
    cum_time = 0.0

    # δ tracking — INTER-ITERATION time gaps
    delta = float('inf')
    t_last_iter_end = 0.0   # 이전 iteration 완료 시각
    t_prev_iter_end = 0.0   # 그 이전 iteration 완료 시각

    for k in range(1, K + 1):
        times = sim_times(N, 0.3)
        order = np.argsort(times)

        z_base = z.copy()
        Nj = Ncount
        count = 0
        iter_end_time = 0.0

        for j_rank, idx in enumerate(order):
            nid = int(idx)

            # Message dropout
            if drop_prob > 0 and np.random.random() < drop_prob:
                continue

            x_new = Q[nid] @ (g[nid] - y[nid] + rho * z)
            if contribs[nid] > 0:
                y_new = y[nid] + rho * (x_new - z)
            else:
                y_new = rho * (x_new - x0[nid])
            s_new = x_new + y_new / rho

            is_ret = (nid in ever)
            if Nj == 0:
                Nj = 1
                z_base = s_new.copy()
            elif is_ret:
                z_base = z_base + (1.0 / Nj) * (s_new - s_prev[nid])
            else:
                Nj += 1
                z_base = (1.0 / Nj) * ((Nj - 1) * z_base + s_new)

            s_prev[nid] = s_new.copy()
            x[nid] = x_new.copy()
            y[nid] = y_new.copy()
            contribs[nid] += 1
            ever.add(nid)
            count += 1

            t_node = times[nid]

            # --- 종료 조건 (Section IV-D) ---
            # 조건 1: |N^k| ≥ Nmin
            if count >= Nmin:
                iter_end_time = t_node
                break
            # 조건 2: |N^k| ≥ Nmin* AND inter-iteration gap ≥ δ
            if count >= Nmin_star:
                projected_gap = cum_time + t_node - t_last_iter_end
                if projected_gap >= delta:
                    iter_end_time = t_node
                    break

        # 시간 update
        if iter_end_time == 0.0 and count > 0:
            # 어떤 조건도 만족 못했지만 노드가 있음 → 마지막 노드 시간 사용
            iter_end_time = times[order[min(count - 1, N - 1)]]

        cum_time += iter_end_time

        # δ 업데이트: 마지막 두 iteration 완료 시각의 간격
        t_prev_iter_end = t_last_iter_end
        t_last_iter_end = cum_time
        if t_prev_iter_end > 0:
            delta = t_last_iter_end - t_prev_iter_end

        Ncount = len(ever)
        z = z_base.copy()
        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    K = 2500
    DROP_PROB = 0.01   # 논문 Section VI: 1% message dropout

    print("=" * 72)
    print("  SRAD-ADMM for Distributed Least Squares — FINAL")
    print("  He et al., IEEE Trans. Signal Processing, 2025")
    print(f"  K={K}, message_dropout={DROP_PROB}")
    print("=" * 72)

    configs = [
        (10, 'random'), (10, 'dirichlet'),
        (20, 'random'), (20, 'dirichlet'),
    ]

    all_results = OrderedDict()

    for N, dist in configs:
        np.random.seed(42)
        A, b = generate_data()
        data = split_random(A, b, N) if dist == 'random' else split_dirichlet(A, b, N)

        rho = compute_rho(data)
        z_star = optimal_solution(data)
        f_star = obj_val(data, z_star)
        n = A.shape[1]

        x_init = [np.random.uniform(-0.5, 0.5, n) for _ in range(N)]

        label = f'N={N}, {dist}'
        print(f"\n  {label}: ρ={rho:.1f}, F*={f_star:.2f}")

        algos = OrderedDict()

        np.random.seed(100)
        zh, th = cc_admm(data, rho, K, x_init)
        algos['CC-ADMM'] = (zh, th)

        np.random.seed(200)
        zh, th = sr_admm(data, rho, K, x_init, Nmin=2, drop_prob=DROP_PROB)
        algos['SR-ADMM'] = (zh, th)

        np.random.seed(300)
        zh, th = srad_admm(data, rho, K, x_init, Nmin=2, drop_prob=DROP_PROB)
        algos['SRAD-ADMM'] = (zh, th)

        np.random.seed(400)
        zh, th = srad_admm_ii(data, rho, K, x_init, Nmin=N, Nmin_star=2,
                              drop_prob=DROP_PROB)
        algos['SRAD-ADMM-II'] = (zh, th)

        metrics = OrderedDict()
        for name, (zh, th) in algos.items():
            eps = [rel_diff(z, z_star) for z in zh]
            obj = [obj_val(data, z) for z in zh]
            metrics[name] = {'eps': eps, 'obj': obj, 'time': th}

            k_conv = '>K'
            for ki in range(len(eps)):
                if eps[ki] <= 0.01:
                    k_conv = ki
                    break
            t_conv = f'{th[k_conv]:.1f}s' if isinstance(k_conv, int) else '-'
            print(f"    {name:<18} ε={eps[-1]:.4e}  k*(0.01)={str(k_conv):<6} t*={t_conv}")

        all_results[label] = (metrics, f_star)

    # =========================================================================
    # PLOTTING
    # =========================================================================

    colors = {'CC-ADMM': '#2196F3', 'SR-ADMM': '#FF9800',
              'SRAD-ADMM': '#4CAF50', 'SRAD-ADMM-II': '#F44336'}
    styles = {'CC-ADMM': '-', 'SR-ADMM': '--', 'SRAD-ADMM': '-.', 'SRAD-ADMM-II': ':'}
    lw = {'CC-ADMM': 2, 'SR-ADMM': 2, 'SRAD-ADMM': 2.5, 'SRAD-ADMM-II': 2.5}

    def make_figure(all_results, x_key, y_key, xlabel, ylabel, title, fname,
                    hline_key=None, hline_val=None, hline_label=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
            ax = axes[idx // 2][idx % 2]
            for name, d in metrics.items():
                xd = d[x_key] if x_key else list(range(len(d[y_key])))
                ax.plot(xd, d[y_key], color=colors[name], ls=styles[name],
                        lw=lw[name], label=name, alpha=0.9)
            if hline_key == 'f_star':
                ax.axhline(f_star, color='gray', ls='--', alpha=0.4, lw=1)
            if hline_val is not None:
                ax.axhline(hline_val, color='gray', ls=':', alpha=0.5, lw=1,
                           label=hline_label)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Fig 1: F^k vs iteration
    make_figure(all_results, None, 'obj', 'Iteration k', '$F^k$',
                'Objective $F^k$ over Iterations (Fig. 2 style)',
                'fig_objective.png', hline_key='f_star')

    # Fig 2: ε^k vs iteration
    make_figure(all_results, None, 'eps', 'Iteration k',
                '$\\epsilon^k = \\|z^k - z^*\\| / \\|z^*\\|$',
                'Relative Difference $\\epsilon^k$ over Iterations (Fig. 6 style)',
                'fig_convergence.png', hline_val=0.001, hline_label='α=0.001')

    # Fig 3: F^k vs wall time (PRIMARY)
    make_figure(all_results, 'time', 'obj', 'Simulated Wall Time (s)', '$F^k$',
                'Objective $F^k$ over Wall Time (Primary metric)',
                'fig_walltime.png', hline_key='f_star')

    # Fig 4: ε^k vs wall time
    make_figure(all_results, 'time', 'eps', 'Simulated Wall Time (s)',
                '$\\epsilon^k$',
                'Relative Difference $\\epsilon^k$ over Wall Time',
                'fig_eps_walltime.png', hline_val=0.001, hline_label='α=0.001')

    print(f"\n{'='*72}")
    print(f"  ✓ All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*72}")
