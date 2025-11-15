# **FED_IIDS: Technical Report**
**Version:** 1.0  
**Status:** Final

This document is the complete and finalized technical report for the **FED_IIDS** project. It reflects the working implementation, including the decoupled **client**, **server**, and **shared** modules, the finalized data pipeline, and the final model architecture with Differential Privacy (DP-SGD).

---
# **1. Introduction**

**FED_IIDS (Federated Intrusion Detection System)** is a distributed, privacy-preserving intrusion detection framework designed to detect malicious network traffic using **Federated Learning (FL)**. It simulates real-world collaboration between organizations—such as hospitals and manufacturing facilities—without requiring them to share any raw traffic logs.

This report documents the complete system architecture, federated workflow, data engineering steps, machine learning model, and all verified implementation details.

---
# **2. System Motivation**

Modern intrusion detection requires large, diverse datasets, but organizations are often unable or unwilling to share logs due to:

- **Privacy Regulations:** (GDPR, HIPAA) restrict movement of sensitive logs.
- **Business Confidentiality:** Internal activity patterns must remain private.
- **Operational Security:** Raw logs may reveal infrastructure or vulnerabilities.

Federated Learning enables decentralized model training without transferring sensitive data.

### **Key Motivations**
- Enable collaborative IDS training without compromising privacy.
- Capture domain-specific attack patterns via **non-IID** datasets.
- Improve model robustness across heterogeneous environments.
- Evaluate FL performance under realistic distributed constraints.

---
# **3. System Architecture Overview**

FED_IIDS adopts a **client–server** architecture using the **Flower (flwr)** framework. All components—client, server, and shared—are fully decoupled.

---
## **3.1 Server (server/)**
- Acts as the FL coordinator.
- Loads configuration from `server_config.py`.
- Initializes the global model from `shared/model.py`.
- Sends global weights and round configuration to clients.
- Aggregates updates via **FedAvg**.
- Evaluates performance on `server/data/global_test_set.npz`.

---
## **3.2 FL Clients (client/)**
Each client:
- Represents an independent organization.
- Loads its own private dataset from `client/data/`.
- Imports the model architecture from the `shared/` folder.
- Trains locally using **DP-SGD**.
- Sends back only model weight updates—**never raw data**.

---
## **3.3 Shared API Contract (shared/)**
Defines the model specification shared between all components:
- `shared/model.py` → Model architecture
- `shared/model_config.py` → `NUM_FEATURES = 30`

This contract ensures consistent architecture across all clients and the server.

---
# **4. Data Engineering Pipeline**

The full preprocessing pipeline is implemented in the `Fed_IIDS.ipynb` notebook.

## **4.1 Source Data**
- Based on the **IoT-DIAD** dataset
- Contains **1.6 million** network flow records
- Aggregates **23 CSV files**

## **4.2 Cleaning & Sanitization**
- Removed samples with NaN or Inf values
- Dropped non-predictive identifier fields (flow_id, IPs, timestamps)
- Applied **Min-Max Scaling (0–1)** to all numeric features

## **4.3 Feature Reduction**
A two-stage feature selection pipeline:
1. **Correlation Filtering** → Removes redundant features
2. **LightGBM Feature Importance** → Selects top predictors

Final feature count: **74 → 30**

## **4.4 Non-IID Partitioning**
Clients receive disjoint sets of attack categories:

### **Client 1 (Hospital)**
- Benign
- Spoofing

### **Client 2 (Factory)**
- Benign
- DoS
- DDoS
- Mirai
- Recon

This heterogeneity enables realistic evaluation of catastrophic forgetting.

---
# **5. Machine Learning Model**

## **5.1 Architecture (Final)**
Defined in `shared/model.py`.  
A **binary classifier** for *Benign vs Attack*.

- Input: **30-dimensional** vector
- Dense(64, activation='relu')
- Dropout(0.2)
- Dense(32, activation='relu')
- Dropout(0.2)
- Output: Dense(1, activation='sigmoid')

This section matches the final implemented model.

## **5.2 Training Parameters**
- Optimizer (clients): **DPKerasAdamOptimizer**
- Optimizer (server evaluation): Adam
- Loss: **BinaryCrossentropy**
- Batch size: **256** (fixed due to DP microbatch requirement)
- Local epochs: **2**

## **5.3 Differential Privacy (DP-SGD)**
DP-SGD provides record-level privacy guarantees via:
- **Gradient clipping** (l2_norm_clip)
- **Gaussian noise injection** (noise_multiplier)

### Important Fix
`DPOptimizerKeras` is incompatible with `validation_split`.
- Solution: Use **validation_data** instead.

---
# **6. Federated Learning Workflow**

## **Round 0 — Baseline Evaluation**
- Server initializes the global model
- Evaluates on `global_test_set.npz`

## **Per-Round Workflow (Rounds 1…N)**
1. **Send Configuration:**  
   Server sends global weights + `{local_epochs: 2}`

2. **Local Training:**  
   Clients train using DP-SGD on private data

3. **Return Updates:**  
   Clients return updated model weights

4. **Aggregation (FedAvg):**
```
GlobalWeights = Σ (n_k / N) * LocalWeights_k
```

5. **Evaluation:**  
   - Clients evaluate updated model on local test sets  
   - Server evaluates on global test set

6. Repeat until `NUM_ROUNDS` is reached

---
# **7. Implementation Details**

## **7.1 Final Project Structure**
```
Fed_IIDS/
│
├── client/
│   ├── run_client.py
│   ├── nids_client.py
│   ├── config.py
│   ├── data_loader.py
│   ├── requirements.txt
│   ├── check_config.py
│   ├── standalone_test.py
│   └── data/
│       └── client_hospital_train.npz
│       └── ...
│
├── server/
│   ├── server.py
│   ├── server_config.py
│   └── data/
│       └── global_test_set.npz
│
├── shared/
│   ├── model.py
│   └── model_config.py
│
├── outputs/
│   ├── server_terminal_output.txt
│   └── client_terminal_output.txt
│
├── Fed_IIDS.ipynb
├── README.md
└── TECHNICAL_REPORT.md
```

## **7.2 Network & Communication**
- Server listens on: **0.0.0.0:8080**
- Clients connect to: **127.0.0.1:8080** (local testing)
- Communication is via **Flower gRPC protocol**

## **7.3 How to Run**
Commands must be executed from the project root.

### **Start Server:**
```
python -m server.server
```

### **Start Client 1:**
```
python -m client.run_client --client-id hospital
```

### **Start Client 2:**
```
python -m client.run_client --client-id factory
```

---
# **8. Evaluation Metrics**
- **Accuracy**
- **Binary Cross-Entropy Loss**
- **F1-Score** (important for imbalanced datasets)

The server also logs weighted client-side accuracy.

---
# **9. Experimental Observations**

## **9.1 Catastrophic Forgetting**
Training on one non-IID client leads to severe forgetting of unseen attacks.

## **9.2 Federated Generalization**
FedAvg merges client-specific knowledge into a generalized IDS.

## **9.3 DP-SGD Constraints**
- Batch size must equal microbatch count
- `validation_split` cannot be used

---
# **10. Limitations**
- FedAvg struggles with extreme non-IID skew
- DP-SGD reduces accuracy slightly
- Synchronous aggregation doesn't scale to thousands of clients

---
# **11. Future Work**
- Implement FedProx, FedAdam, FedNova
- Add secure aggregation protocols
- Extend to temporal models (LSTM/Transformer)
- Support real-world multi-client deployment

---
# **12. Conclusion**

FED_IIDS demonstrates that a high-accuracy intrusion detection system can be collaboratively trained across organizations **without sharing raw data**, protected by Differential Privacy and supported by a realistic non-IID FL environment.

This document serves as the final, verified technical report for the project.

