# FED_IIDS: Technical Report (Unified, Fully Updated)

## Version: 1.0

This document is the **unified and fully corrected technical report** for the FED_IIDS project. It merges the previous technical report with the updated corrections you provided (including all the comment-flagged modifications), and reflects the **final, working implementation** of the system.

This is a standalone technical report separate from the project README.

---

# 1. Introduction

**FED_IIDS (Federated Intrusion Detection System)** is a distributed, privacy-preserving intrusion detection system designed to detect malicious network traffic using **Federated Learning (FL)**. The system emulates a real-world scenario where multiple organizations—such as a hospital and a factory—collaborate without sharing raw network logs.

This report presents the core architecture, algorithms, federated learning workflow, data engineering pipeline, experimental results, and the final implementation details used in the project.

---

# 2. System Motivation

Modern intrusion detection requires large, diverse datasets, but organizations cannot share raw logs due to:
- Privacy regulations (GDPR, HIPAA)
- Business confidentiality
- Operational security risks

Federated Learning enables **collaborative training without data sharing**, making it suitable for distributed defense systems.

**Motivations:**
- Enable collaborative IDS training without compromising privacy.
- Capture domain-specific attack patterns via **non-IID client datasets**.
- Improve model robustness across heterogeneous environments.
- Evaluate FL performance under realistic distributed security constraints.

---

# 3. System Architecture Overview

FED_IIDS uses a modular client–server architecture based on the **Flower (flwr)** FL framework.

## 3.1 Server
- Acts as the FL coordinator.
- Initializes the global model.
- Sends configuration (e.g., local epochs) to clients.
- Aggregates updates using **FedAvg**.
- Evaluates on `global_test_set.npz`.

## 3.2 FL Clients
Each client:
- Represents an independent organization.
- Loads its private `.npz` dataset.
- Trains the model locally using **DP-SGD**.
- Sends back **only model weight updates**, never data.

## 3.3 Communication Protocol
- Built on Flower’s **gRPC**-based orchestration.
- Model weights exchanged as NumPy arrays.
- Server-side config stored in `server_config.py`.
- Default server bind address: `0.0.0.0:8080`.
- Default client target address: `127.0.0.1:8080` for local testing.

---

# 4. Data Engineering Pipeline

Data preparation was performed in the `Fed_IIDS.ipynb` preprocessing notebook.

## 4.1 Source Data
- Derived from **IoT-DIAD** and similar datasets.
- >1.6 million network flow samples.
- 23 CSV files aggregated.

## 4.2 Cleaning & Sanitization
- Removed NaN and Inf values.
- Removed identifier fields: `flow_id`, IP addresses, timestamps.
- Applied **Min-Max Scaling (0 to 1)** to all numeric features.

## 4.3 Feature Reduction
A two-stage feature reduction pipeline:
1. **Correlation filtering** removed highly redundant features.
2. **LightGBM feature importance** selected the strongest predictors.

Result:
- Feature count reduced from **74 → 30**.

## 4.4 Non-IID Partitioning (Final Implementation)
Clients receive disjoint sets of attack categories:

### Client 1 (Hospital)
- Benign
- Spoofing

### Client 2 (Factory)
- Benign
- DoS
- DDoS
- Mirai
- Recon

This **intentionally introduces heterogeneity**, enabling evaluation of catastrophic forgetting and cross-domain generalization.

---

# 5. Machine Learning Model

## 5.1 Architecture (Final Implementation)
A **binary classifier** (Benign vs Attack) defined in `client/model.py`.

- Input: 30 features
- Hidden Layers: Dense(64) → Dense(32) → Dense(16), all ReLU
- Dropout applied after hidden layers
- Output: **Sigmoid neuron** (`0 = Benign`, `1 = Attack`)

## 5.2 Training Parameters
- Optimizer: `DPKerasAdamOptimizer` (clients), `Adam` (server evaluation)
- Loss: **BinaryCrossentropy**
- Batch Size: **256** (required for DP microbatch divisibility)
- Local Epochs: **2**

## 5.3 Differential Privacy (DP-SGD)
DP-SGD ensures privacy via:
- **Gradient clipping** (`l2_norm_clip`)
- **Gaussian noise injection** (`noise_multiplier`)

A major bug was identified:
- `DPOptimizerKeras` is incompatible with `validation_split`.
- **Fix:** Use only `validation_data`.

---

# 6. Federated Learning Workflow

## Round 0 (Baseline Evaluation)
- Server initializes global model.
- Evaluates on global test set.

## Per-Round Procedure
1. Server selects all clients.
2. Server sends global weights and config to clients.
3. Clients train locally using DP-SGD.
4. Clients return updated weight deltas.
5. Server aggregates using:
```
GlobalWeights = Σ (n_k / N) * LocalWeights_k
```
6. Server evaluates on:
   - Global test set
   - Client-side local test sets (weighted)

---

# 7. Implementation Details

## 7.1 Project Layout
```
Fed_IIDS/
│
├── server.py               # Main server logic
├── server_config.py        # Server-side runtime configuration
│
└── client/
    ├── run_client.py       # Client entry script
    ├── nids_client.py      # Federated client (fit/evaluate)
    ├── model.py            # Shared DNN model
    ├── data_loader.py      # Loads and trims NPZ datasets
    ├── config.py           # Client configuration (DP, FL)
    ├── standalone_test.py  # Local DP training debug script
    └── data/               # Partitioned non-IID datasets
```

## 7.2 Network & Communication
- Server listens on: `0.0.0.0:8080`
- Clients connect to: `127.0.0.1:8080` or LAN IP
- Flower handles all RPC/grpc communication

---

# 8. Evaluation Metrics

Server tracks:
- **Accuracy** (global)
- **Binary Cross-Entropy Loss**
- **F1-Score** (critical for class imbalance)
- **Weighted client-side accuracy** (local evaluations)

---

# 9. Experimental Observations

### 9.1 Catastrophic Forgetting
Training on a single non-IID client (e.g., only hospital data) causes collapse in global test performance (e.g., accuracy dropping from 0.75 → 0.26). The model forgets unseen attack categories.

### 9.2 Federated Learning Benefit
Using **both non-IID clients** resolves forgetting and produces a generalized model capable of detecting all attack categories.

### 9.3 DP-SGD Operational Issues
`tensorflow-privacy` introduces strict constraints:
- No `validation_split`
- Batch-size/divisibility rules

Fixes were incorporated into the final implementation.

---

# 10. Limitations
- FedAvg struggles with extreme non-IID skew.
- DP-SGD introduces accuracy and speed penalties.
- Synchronous aggregation may not scale to very large numbers of clients.

---

# 11. Future Work
- Investigate **FedProx, FedAdam, FedNova** for better non-IID convergence.
- Add secure aggregation (HE, TEEs, or masking).
- Extend the model to temporal architectures (LSTM, GRU, Transformers).
- Support more than two clients.

---

# 12. Conclusion

FED_IIDS demonstrates that a high-performance, privacy-preserving intrusion detection system can be collaboratively trained across organizations without exposing sensitive network logs. The system integrates federated learning, differential privacy, non-IID data simulation, and deep learning into a unified, deployable architecture.

This technical report reflects the **final, verified implementation and results** of the project.

---