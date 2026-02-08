"""
run_all.py — 4개 알고리즘 비교 실험 실행
He et al., IEEE TSP 2025

실행: python run_all.py
결과: ./results/ 디렉토리에 4개 PNG 파일 생성
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from common import (generate_data, split_random, split_dirichlet,
                     compute_rho, obj_val, optimal_solution, rel_diff)
from cc_admm import cc_admm
from sr_admm import sr_admm
from srad_admm import srad_admm
from srad_admm_ii import srad_admm_ii

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {'CC-ADMM': '#2196F3', 'SR-ADMM': '#FF9800',
           'SRAD-ADMM': '#4CAF50', 'SRAD-ADMM-II': '#F44336'}
STYLES = {'CC-ADMM': '-', 'SR-ADMM': '--',
           'SRAD-ADMM': '-.', 'SRAD-ADMM-II': ':'}
LW = {'CC-ADMM': 2, 'SR-ADMM': 2, 'SRAD-ADMM': 2.5, 'SRAD-ADMM-II': 2.5}


def make_figure(all_results, x_key, y_key, xlabel, ylabel, title, fname,
                hline_key=None, hline_val=None, hline_label=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
        ax = axes[idx // 2][idx % 2]
        for name, d in metrics.items():
            if x_key:
                xd = d[x_key]
            else:
                xd = list(range(len(d[y_key])))
            ax.plot(xd, d[y_key], color=COLORS[name], ls=STYLES[name],
                    lw=LW[name], label=name, alpha=0.9)
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
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    K = 2500
    DROP_PROB = 0.01

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
        data = (split_random(A, b, N) if dist == 'random'
                else split_dirichlet(A, b, N))

        rho = compute_rho(data)
        z_star = optimal_solution(data)
        f_star = obj_val(data, z_star)
        n = A.shape[1]
        x_init = [np.random.uniform(-0.5, 0.5, n) for _ in range(N)]

        label = f'N={N}, {dist}'
        print(f"\n  {label}: rho={rho:.1f}, F*={f_star:.2f}")

        algos = OrderedDict()

        np.random.seed(100)
        algos['CC-ADMM'] = cc_admm(data, rho, K, x_init)

        np.random.seed(200)
        algos['SR-ADMM'] = sr_admm(data, rho, K, x_init,
                                    Nmin=2, drop_prob=DROP_PROB)

        np.random.seed(300)
        algos['SRAD-ADMM'] = srad_admm(data, rho, K, x_init,
                                        Nmin=2, drop_prob=DROP_PROB)

        np.random.seed(400)
        algos['SRAD-ADMM-II'] = srad_admm_ii(data, rho, K, x_init,
                                              Nmin=N, Nmin_star=2,
                                              drop_prob=DROP_PROB)

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
            t_conv = (f'{th[k_conv]:.1f}s' if isinstance(k_conv, int)
                      else '-')
            print(f"    {name:<18} eps={eps[-1]:.4e}  "
                  f"k*(0.01)={str(k_conv):<6} t*={t_conv}")

        all_results[label] = (metrics, f_star)

    # --- Plotting ---
    print("\n  Generating figures...")

    make_figure(all_results, None, 'obj',
                'Iteration k', '$F^k$',
                'Objective $F^k$ over Iterations (Fig. 2 style)',
                'fig_objective.png', hline_key='f_star')

    make_figure(all_results, None, 'eps',
                'Iteration k',
                r'$\epsilon^k = \|z^k - z^*\| / \|z^*\|$',
                r'Relative Difference $\epsilon^k$ over Iterations (Fig. 6 style)',
                'fig_convergence.png',
                hline_val=0.001, hline_label='alpha=0.001')

    make_figure(all_results, 'time', 'obj',
                'Simulated Wall Time (s)', '$F^k$',
                'Objective $F^k$ over Wall Time (Primary metric)',
                'fig_walltime.png', hline_key='f_star')

    make_figure(all_results, 'time', 'eps',
                'Simulated Wall Time (s)', r'$\epsilon^k$',
                r'Relative Difference $\epsilon^k$ over Wall Time',
                'fig_eps_walltime.png',
                hline_val=0.001, hline_label='alpha=0.001')

    print(f"\n{'=' * 72}")
    print(f"  Done! All figures in: {OUTPUT_DIR}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
