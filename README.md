# UWB-TDOA: Closing the Gap to the Ziv-Zakai Bound

This repository contains the simulation framework and tracking pipeline for a sub-metre **Ultra-Wideband (UWB)** indoor localization system utilizing **Time Difference of Arrival (TDoA)**. The project serves as a comprehensive study on the discrepancy between practical signal processing pipelines and the theoretical **Ziv-Zakai Bound (ZZB)** floor.

##  Project Overview
Indoor environments present significant challenges for RF-based localization, including multi-path interference, clock synchronization errors, and Non-Line-of-Sight (NLOS) conditions. This project implements a full 1GHz-sampled PHY simulation to evaluate how specific engineering choices—such as super-resolution and hybrid filtering—approach the fundamental limits of positioning accuracy.

##  Key Contributions
* **Theoretical Benchmarking**: Numerical derivation of the 3-D Bayesian CRLB and ZZB. The ZZB (1.04 m) was found to be **3.4x tighter** than the CRLB (3.58 m) at practical SNR levels, successfully capturing threshold effects.
* **Super-Resolution (SR) Pipeline**: Implementation of coherent averaging (**N = 8, M = 2**) which reduced static localization failure rates from 25% to 9%.
* **Hybrid UKF Tracker**: A dynamic tracking architecture fusing a warm-started Gauss-Newton solver with **Unscented Kalman Filter (UKF)** predictions. This achieved a **0.52 m mean error** with 0% failure.
* **NLOS Classification**: A gradient-boosting classifier using 10 CIR features (**AUC = 0.9975**) to reduce localization error by 27% through link discarding.
* **Anchor Optimization**: Identified the "GDOP Trap" in pure ZZB-minimization and implemented a height-varied corner layout that recovered **+24%** tracking improvement.

##  System Architecture
The pipeline is modeled after high-performance UWB hardware (e.g., DW3000) and includes:
1. **PHY Simulation**: Gaussian monocycle pulses with 1 ns resolution.
2. **Estimation**: Two-step TOA estimator (coarse matched-filter + 90%-threshold crossing).
3. **Synchronization**: Asynchronous reference-tag clock correction architecture.
4. **Dynamic Tracking**: Constant-velocity model with dimensional stability (processing measurements in metres to avoid scaling instability).

##  Performance Summary

### Static Localization Results
| Mode | Mean Error (m) | Failure Rate (%) | Gap to ZZB |
| :--- | :---: | :---: | :---: |
| **Sync (Two-Step)** | 1.423 | 25.3% | 37% Above |
| **Sync + Super-Res** | 1.198 | 9.3% | 16% Above |
| **Async + Super-Res** | 1.312 | 10.7% | 26% Above |
| **ZZB Target** | 1.036 | - | - |

### Dynamic Tracking Performance
| Method | Mean Error (m) | Jitter (σ) | Failure Rate (%) |
| :--- | :---: | :---: | :---: |
| **Static Solver** | 1.064 | 0.565 m | 40% |
| **UKF Only** | 1.880 | 0.127 m | 0% |
| **Hybrid UKF** | **0.615** | **0.031 m** | **0%** |

