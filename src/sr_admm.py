"""
sr_admm.py — Straggler-Resilient ADMM (Eq. 15, Algorithms 1-2)
He et al., IEEE TSP 2025

Centralized, straggler-resilient.
Server waits for Nmin fastest nodes; all fast nodes participate.
Stale contributions from non-participating nodes are reused in z.

Wall time: time until fast group finishes (straggler cutoff at t=2.0)

z^k = (1/|N^{1:k}|) Σ_{i∈N^{1:k}} s_i^{k_i}  — Eq. (13)

Dual update:
  Returning (contributed before): y_i = y_i + ρ(x_i - z)
  New (first contribution):       y_i = ρ(x_i - x_i^0)  — Eq. (10)
"""

import numpy as np
from common import sim_times, get_participating_nodes


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
        Nk, deadline = get_participating_nodes(times, Nmin, drop_prob)
        cum_time += deadline

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
