# UWB-TDOA: Closing the Gap to the Ziv-Zakai Bound

[cite_start]This repository contains the simulation framework and tracking pipeline for a sub-metre **Ultra-Wideband (UWB)** indoor localization system utilizing **Time Difference of Arrival (TDoA)**[cite: 5, 13, 17]. [cite_start]The project serves as a comprehensive study on the discrepancy between practical signal processing pipelines and the theoretical **Ziv-Zakai Bound (ZZB)** floor[cite: 5, 18, 19].

## Project Overview
[cite_start]Indoor environments present significant challenges for RF-based localization, including multi-path interference, clock synchronization errors, and Non-Line-of-Sight (NLOS) conditions[cite: 15, 16, 21, 22]. [cite_start]This project implements a full 1GHz-sampled PHY simulation to evaluate how specific engineering choices—such as super-resolution and hybrid filtering—approach the fundamental limits of positioning accuracy[cite: 6, 18, 19].

## Key Contributions
* [cite_start]**Theoretical Benchmarking**: Numerical derivation of the 3-D Bayesian CRLB and ZZB[cite: 26]. [cite_start]The ZZB (1.04 m) was found to be **3.4x tighter** than the CRLB (3.58 m) at practical SNR levels, successfully capturing threshold effects[cite: 27, 314].
* [cite_start]**Super-Resolution (SR) Pipeline**: Implementation of coherent averaging ($N=8, M=2$) which reduced static localization failure rates from 25% to 9%[cite: 8, 313].
* [cite_start]**Hybrid UKF Tracker**: A dynamic tracking architecture fusing a warm-started Gauss-Newton solver with **Unscented Kalman Filter (UKF)** predictions[cite: 9, 234]. [cite_start]This achieved a **0.52 m mean error** with 0% failure[cite: 9, 28].
* [cite_start]**NLOS Classification**: A gradient-boosting classifier using 10 CIR features ($AUC = 0.9975$) to reduce localization error by 27% through link discarding[cite: 10, 316].
* [cite_start]**Anchor Optimization**: Identified the "GDOP Trap" in pure ZZB-minimization and implemented a height-varied corner layout that recovered **+24%** tracking improvement[cite: 11, 12, 291, 317].

## System Architecture
[cite_start]The pipeline is modeled after high-performance UWB hardware (e.g., DW3000) and includes[cite: 31, 318]:
1. [cite_start]**PHY Simulation**: Gaussian monocycle pulses with 1 ns resolution[cite: 44, 46, 102].
2. [cite_start]**Estimation**: Two-step TOA estimator (coarse matched-filter + 90%-threshold crossing)[cite: 6, 101, 102, 103].
3. [cite_start]**Synchronization**: Asynchronous reference-tag clock correction architecture[cite: 6, 100, 134, 136].
4. [cite_start]**Dynamic Tracking**: Constant-velocity model with dimensional stability (processing measurements in metres to avoid scaling instability)[cite: 242, 245, 247, 248].

## Performance Summary

### Static Localization Results
| Mode | Mean Error (m) | Failure Rate (%) | Gap to ZZB |
| :--- | :---: | :---: | :---: |
| **Sync (Two-Step)** | [cite_start]1.423 [cite: 111] | [cite_start]25.3% [cite: 111] | [cite_start]37% Above [cite: 294] |
| **Sync + Super-Res** | [cite_start]1.198 [cite: 111] | [cite_start]9.3% [cite: 111] | [cite_start]16% Above [cite: 294] |
| **Async + Super-Res** | [cite_start]1.312 [cite: 111] | [cite_start]10.7% [cite: 111] | [cite_start]26% Above [cite: 294] |
| **ZZB Target** | [cite_start]1.036 [cite: 111] | - | - |

### Dynamic Tracking Performance
| Method | Mean Error (m) | Jitter ($\sigma$) | Failure Rate (%) |
| :--- | :---: | :---: | :---: |
| **Static Solver** | [cite_start]1.064 [cite: 258] | [cite_start]0.565 m [cite: 258] | [cite_start]40% [cite: 258] |
| **UKF Only** | [cite_start]1.880 [cite: 258] | [cite_start]0.127 m [cite: 258] | [cite_start]0% [cite: 258] |
| **Hybrid UKF** | [cite_start]**0.615 [cite: 258]** | [cite_start]**0.031 m [cite: 258]** | [cite_start]**0% [cite: 258]** |



