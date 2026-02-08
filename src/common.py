"""
common.py — 공통 유틸리티 (데이터 생성, 분배, 지표 계산)
He et al., IEEE TSP 2025
"""

import numpy as np


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_samples=4177, n_features=7):
    """
    Abalone-like synthetic data (논문 Section VI).
    4177 samples, 7 features, linear regression target + noise.
    """
    rng = np.random.RandomState(2025)
    A = rng.randn(n_samples, n_features)
    scales = np.array([0.1, 0.08, 0.1, 0.5, 0.2, 0.1, 0.15])
    A = A * scales + np.array([0.5, 0.4, 0.13, 0.83, 0.37, 0.18, 0.27])
    x_true = rng.randn(n_features) * 2
    b = A @ x_true + rng.randn(n_samples) * 0.5
    return A, b


# =============================================================================
# DATA DISTRIBUTION
# =============================================================================

def split_random(A, b, N):
    """Randomly distribute data to N nodes (roughly equal partitions)."""
    idx = np.random.permutation(A.shape[0])
    parts = np.array_split(idx, N)
    return [(A[p], b[p]) for p in parts]


def split_dirichlet(A, b, N, beta=0.5):
    """
    Non-IID distribution using Dirichlet (Li et al., ICDE 2022 [31]).
    Lower beta → more heterogeneity.
    """
    order = np.argsort(b)
    chunks = np.array_split(order, N * 5)
    props = np.random.dirichlet(np.ones(N) * beta, len(chunks))
    bins = [[] for _ in range(N)]
    for ci, ch in enumerate(chunks):
        alloc = (props[ci] * len(ch)).astype(int)
        alloc[np.argmax(alloc)] += len(ch) - alloc.sum()
        s = 0
        for ni in range(N):
            bins[ni].extend(ch[s:s + alloc[ni]].tolist())
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
    """ρ = 2·max_i{L_i} + 2 — Proposition V.5, Eq. (49)"""
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
    """ε^k = ||z^k - z*|| / ||z*|| — Eq. (79)"""
    return np.linalg.norm(z - z_star) / max(np.linalg.norm(z_star), 1e-12)


# =============================================================================
# SIMULATED WALL-CLOCK TIMING & MESSAGE DROPOUT
# =============================================================================

def sim_times(N, straggler_prob=0.3):
    """
    Per-node compute time for one iteration.
      Normal:    t ~ U(0.8, 1.2)
      Straggler: t ~ U(3.0, 8.0) with probability straggler_prob
    """
    times = np.random.uniform(0.8, 1.2, N)
    for i in range(N):
        if np.random.random() < straggler_prob:
            times[i] = np.random.uniform(3.0, 8.0)
    return times


def get_participating_nodes(times, Nmin, drop_prob=0.0):
    """
    Determine participating nodes for an iteration.

    논문의 straggler 모델 (bimodal: fast U(0.8,1.2) vs slow U(3,8)):
      - 서버가 Nmin개 결과를 기다림
      - 빠른 그룹은 거의 동시에 도착 → 모두 참여
      - Straggler(t >= 2.0)는 제외

    Returns: (participating_node_ids, deadline_time)
    """
    N = len(times)
    order = np.argsort(times)
    sorted_times = times[order]

    # Nmin번째 도착이 최소 기다림
    deadline_base = sorted_times[min(Nmin - 1, N - 1)]
    # 빠른 그룹(t < 2.0)이 모두 도착할 때까지 포함
    fast_cutoff = 2.0
    deadline = max(deadline_base, min(fast_cutoff, sorted_times[-1]))

    Nk = [int(order[j]) for j in range(N) if sorted_times[j] <= deadline]
    if len(Nk) < Nmin:
        Nk = [int(order[j]) for j in range(Nmin)]

    # Message dropout (논문 Section VI, Fig 4-5)
    if drop_prob > 0:
        survived = [i for i in Nk if np.random.random() >= drop_prob]
        if len(survived) >= 1:
            Nk = survived

    return Nk, deadline
