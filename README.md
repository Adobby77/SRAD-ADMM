# Async ADMM Research: Straggler-Resilient Algorithms

This repository contains simulations for various **Distributed ADMM (Alternating Direction Method of Multipliers)** algorithms, focusing on their resilience to **stragglers** (slow computing nodes) in a distributed learning environment.

Reference: Straggler-Resilient Asynchronous ADMM for Distributed Consensus Optimization, J He, M Xiao, M Skoglund, HV Poor - IEEE Transactions on Signal Processing, 2025

## 🧪 Implemented Algorithms

1.  **CC-ADMM (Coded Computation ADMM) / Standard ADMM**
    *   **Synchronous**: The central server waits for **all** $N$ worker nodes to finish their local updates before proceeding to the global update.
    *   **Straggler Sensitivity**: Highly sensitive. The speed is bottlenecked by the slowest node (straggler).

2.  **SR-ADMM (Straggler-Resilient ADMM)**
    *   **Synchronous (Relaxed)**: The central server waits only for the fastest $N_{min}$ nodes (e.g., 20 out of 30) and ignores the stragglers for the current iteration.
    *   **Pros**: Significantly faster per-iteration time than CC-ADMM.
    *   **Cons**: May discard valuable data from slow nodes if not handled carefully.

3.  **SRAD-ADMM (Straggler-Resilient Asynchronous Distributed ADMM)**
    *   **Asynchronous**: Completely removes the synchronization barrier.
    *   **Mechanism**: Worker nodes perform local updates at their own pace. The central server updates the global model incrementally as soon as it receives an update from any worker.
    *   **Pros**: Maximizes resource utilization; fast convergence in time-varying heterogeneous environments.

4.  **SRAD-ADMM II**
    *   **Asynchronous + Time Tracking**: An advanced version of SRAD-ADMM that incorporates time-tracking mechanisms or adaptive weights to further improve convergence stability (simulated).

## 📂 Repository Structure

*   **`src/`**: Source code for simulations.
    *   `Combined.py`: **[Main]** Runs all 4 variants together and generates comparison plots.
    *   `SRAD-ADMM.py`: Standalone simulation for SRAD-ADMM vs Standard ADMM.
    *   `SRAD-ADMM-II.py`: Standalone simulation for SRAD-ADMM II.
    *   `SR-ADMM.py`: Standalone simulation for SR-ADMM.
*   **`results/`**: Simulation outputs and plots.
    *   `Combined_Comparison_5s.png`: Comparison of all algorithms (0-5s).
    *   `Combined_Comparison_0.3s_Async.png`: Zoomed-in comparison of asynchronous variants (0-0.3s).

## 🚀 How to Run

1.  **Run the combined comparison:**
    ```bash
    cd src
    python3 Combined.py
    ```
    This will generate the comparison plots in the `results/` directory.

2.  **Run individual algorithms:**
    ```bash
    cd src
    python3 SRAD-ADMM.py
    # or
    python3 SRAD-ADMM-II.py
    ```

## 📈 Visual Results

### 1. Overall Comparison (0-5s)
This plot allows you to compare the convergence speed of all algorithms, including the synchronous CC-ADMM.
![Overall Comparison](results/Combined_Comparison_5s.png)

### 2. Async Variants Zoom-in (0-0.3s)
A detailed look at the initial convergence of the Straggler-Resilient algorithms. Note the rapid updates of the SRAD variants compared to the step-wise updates of SR-ADMM.
![Async Zoom-in](results/Combined_Comparison_0.3s_Async.png)

## 📊 Results Summary

*   **CC-ADMM** suffers significantly from stragglers, showing very slow progress in wall-clock time.
*   **SR-ADMM** improves speed by ignoring stragglers but is still bound by the synchronous barrier of the $N_{min}$-th node.
*   **SRAD-ADMM (Async)** algorithms show the fastest convergence by fully utilizing the computational power of all available nodes without waiting.

---
*Created for Async ADMM Research.*
