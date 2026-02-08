"""
cc_admm.py — Classical Centralized ADMM (Eq. 2)
He et al., IEEE TSP 2025

Fully synchronous. All N nodes compute every iteration.
Wall time: must wait for the slowest node → max(t_i^k)

Updates:
  x_i^k = (A_i^T A_i + ρI)^{-1}(A_i^T b_i - y_i^{k-1} + ρ z^{k-1})
  y_i^k = y_i^{k-1} + ρ(x_i^k - z^{k-1})
  z^k   = (1/N) Σ_i (x_i^k + y_i^k / ρ)
"""

import numpy as np
from common import sim_times


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
