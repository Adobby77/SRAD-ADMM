"""
srad_admm_ii.py — SRAD-ADMM-II (Algorithm 5)
He et al., IEEE TSP 2025

Extension with dynamic termination (Section IV-D).
Proceeds to next iteration when EITHER:
  Condition 1: |N^k| ≥ Nmin                        (enough nodes)
  Condition 2: |N^k| ≥ Nmin* AND |t - t_last| ≥ δ  (timeout)

δ = time gap between the last two ITERATION COMPLETIONS
    (inter-iteration gap, not intra-iteration gap)

Uses Nmin = N (논문 Section VI) to maximize per-iteration info,
but δ-based timeout allows early termination when stragglers delay.
"""

import numpy as np
from common import sim_times


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
    t_last_iter_end = 0.0
    t_prev_iter_end = 0.0

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
            # 조건 1: |N^k| ≥ Nmin (enough nodes)
            if count >= Nmin:
                iter_end_time = t_node
                break
            # 조건 2: |N^k| ≥ Nmin* AND inter-iteration gap ≥ δ
            if count >= Nmin_star:
                projected_gap = cum_time + t_node - t_last_iter_end
                if projected_gap >= delta:
                    iter_end_time = t_node
                    break

        if iter_end_time == 0.0 and count > 0:
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
