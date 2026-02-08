# SRAD-ADMM: Straggler-Resilient Asynchronous Distributed ADMM

This repository contains simulations for various **Distributed ADMM (Alternating Direction Method of Multipliers)** algorithms, focusing on their resilience to **stragglers** (slow computing nodes) in a distributed learning environment.

**Reference:** Straggler-Resilient Asynchronous ADMM for Distributed Consensus Optimization, J He, M Xiao, M Skoglund, HV Poor — *IEEE Transactions on Signal Processing, 2025*

## 📂 Repository Structure

- **`common.py`**: Data generation, utilities, and `get_participating_nodes()` function.
- **`cc_admm.py`**: Implementation of Eq. (2) - Standard/CC-ADMM.
- **`sr_admm.py`**: Implementation of Eq. (15), Alg 1-2 - SR-ADMM.
- **`srad_admm.py`**: Implementation of Eq. (32), Alg 3-4 - SRAD-ADMM.
- **`srad_admm_ii.py`**: Implementation of Alg 5 - SRAD-ADMM-II.
- **`run_all.py`**: Main script for running experiments and generating comparisons.
- **`srad_admm_simulation_final.py`**: Single file version of the simulation.

## 🔄 Changes from Previous Version

| Feature | Previous Version | Final Version |
| :--- | :--- | :--- |
| **Participating Nodes** | $N_{min}$ only (exactly 2) | $N_{min}$ is minimum waiting count; fast group whole participation (~7) |
| **Random Constants** | * 0.7 overlap, * 1.3 window | All removed |
| **Message Dropout** | None | `drop_prob=0.01` (Paper Fig 4-5) |
| **SRAD-ADMM-II $\delta$** | Intra-iteration gap | Inter-iteration gap (Paper Section IV-D) |
| **Wall Time Calculation** | CC-ADMM: max, Others: $N_{min}$-th | CC-ADMM: max, Others: fast group deadline |

## 🧪 Implemented Algorithms

1.  **CC-ADMM (Coded Computation ADMM) / Standard ADMM**
    *   **Synchronous**: The central server waits for **all** $N$ worker nodes.
    *   **Wall Time**: Determined by the slowest node (max time).

2.  **SR-ADMM (Straggler-Resilient ADMM)**
    *   **Relaxed Synchronous**: The server waits for a subset of nodes.
    *   Uses **Fast Group Deadline** for wall time calculation.

3.  **SRAD-ADMM (Straggler-Resilient Asynchronous Distributed ADMM)**
    *   **Asynchronous**: Workers update at their own pace.
    *   **Robustness**: Handles heterogeneous and time-varying delays.

4.  **SRAD-ADMM II**
    *   **Enhanced Asynchronous**: Incorporates Inter-iteration gap ($\delta$) as per Paper Section IV-D.

## 🚀 How to Run

1.  **Navigate to the source directory:**
    ```bash
    cd src
    ```

2.  **Run the full comparison simulation:**
    ```bash
    python run_all.py
    ```
    This will execute the simulations and save the resulting plots to the `src/results/` folder.
