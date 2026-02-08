"""
srad_admm.py — Straggler-Resilient Asynchronous Decentralized ADMM
               (Eq. 32, Algorithms 3-4)
He et al., IEEE TSP 2025

Decentralized, asynchronous. Key differences from SR-ADMM:
  (a) z updated INCREMENTALLY as each node finishes (not batch average)
  (b) No central server — each node computes its own z
  (c) Conflict resolution: lower ID has priority (Rule 1, Eq. 34)

Eq. (32) — incremental z-update:
  Returning (i ∈ N^{1:k-1}):
    z = z_prev + (1/N_j)(s_i^k − s_i^{prev})     [replacement, N_j same]
  New (i ∉ N^{1:k-1}):
    z = (1/N_j)((N_j−1)·z_prev + s_i^k)           [additive, N_j++]

x-update uses z^{k-1} (previous iteration's z):
  논문 Eq. (22): x_i^k = argmin f_i(x) + (ρ/2)||x - z^{k-1}||²
  All nodes use the same z^{k-1} for computation (parallel model).
  z_base is the merge intermediate, not the broadcasted z.

Wall time: same as SR-ADMM (fast group deadline).
"""

import numpy as np
from common import sim_times, get_participating_nodes


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

    for k in range(1, K + 1):
        times = sim_times(N, 0.3)
        Nk, deadline = get_participating_nodes(times, Nmin, drop_prob)
        cum_time += deadline

        # Conflict resolution: lower ID first — Rule 1, Eq. (34)
        Nk.sort()

        # --- 증분 z-update — Eq. (32) ---
        z_base = z.copy()
        Nj = Ncount

        for nid in Nk:
            # x-update: z^{k-1} (이전 iteration z) 사용
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
