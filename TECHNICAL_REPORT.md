# **FED_IIDS: Technical Report**
**Version:** 1.0  

This document is the complete technical report for the **FED_IIDS** project. It reflects the final implementation, including the decoupled client, server, and shared modules, the finalized dataset preparation pipeline, and the implementation of Differential Privacy in a federated setting. All datasets required to run the system are **already included inside this repository**.

---
# **1. Introduction**

**FED_IIDS (Federated Intrusion Detection System)** is a distributed, privacy-preserving intrusion detection framework designed to classify malicious network traffic using **Federated Learning (FL)**. It simulates collaboration between multiple organizations—such as a hospital and a factory—without sharing any private network logs.

This report provides a detailed description of the system architecture, data engineering pipeline, federated learning workflow, model design, experimental observations, and implementation details.

---
# **2. System Motivation**

Modern intrusion detection depends on large, diverse datasets, yet organizations cannot share raw traffic logs due to:

- **Privacy regulations** (GDPR, HIPAA)
- **Business confidentiality**
- **Operational security** concerns

Federated Learning solves this by enabling collaborative model training without exposing private data.

### **Motivations**
- Enable collaborative IDS training without compromising privacy.
- Capture domain-specific attack patterns using **non-IID** client datasets.
- Improve robustness of IDS models across heterogeneous environments.
- Evaluate FL performance under realistic distributed security constraints.

---
# **3. System Architecture Overview**

FED_IIDS uses a modular **client–server architecture** implemented using the **Flower (flwr)** framework. The system is cleanly separated into:

- **server/** – coordination, aggregation, and global evaluation
- **client/** – local training using DP-SGD
- **shared/** – shared model and configuration specification

---
## **3.1 Server**
- Acts as the central FL coordinator
- Loads configuration from `server_config.py`
- Initializes the global model from `shared/model.py`
- Sends model parameters & configuration to clients
- Aggregates updates using **FedAvg**
- Evaluates globally using `server/data/global_test_set.npz`

---
## **3.2 FL Clients**
Each client:
- Represents an independent organization
- Loads its private dataset from `client/data/`
- Imports model definition from `shared/`
- Trains locally using **DP-SGD**
- Sends back **only model weights**, not data

---
## **3.3 Communication Protocol**
- Based on Flower’s **gRPC** communication layer
- Serializes model parameters as NumPy arrays
- Default addresses:
  - Server bind: **0.0.0.0:8080**
  - Client target: **127.0.0.1:8080** (local testing)

---
# **4. Data Engineering Pipeline**

All datasets used by the system are **already included in the repository** under:
- `client/data/`
- `server/data/`

The following describes the preprocessing approach used to generate these datasets.

## **4.1 Source Data**
- Derived from the IoT-DIAD dataset
- Over **1.6 million** network flow samples
- Aggregated from **23 CSV files**

## **4.2 Cleaning & Sanitization**
- Removed rows with NaN and Inf values
- Removed identifiers (flow_id, IP addresses, timestamps)
- Applied **Min-Max Scaling** to all numeric features (0 to 1)

## **4.3 Feature Reduction**
Two-stage reduction:
1. **Correlation filtering** to remove redundant features
2. **LightGBM feature ranking** to select most predictive features

Final feature set reduced from **74 → 30**.

## **4.4 Non-IID Partitioning (Final Implementation)**
To simulate real-world heterogeneity, client datasets contain different attack distributions:

### **Client 1: Hospital**
- Benign
- Spoofing

### **Client 2: Factory**
- Benign
- DoS
- DDoS
- Mirai
- Recon

These partitions were used to produce the `.npz` files bundled with the repository.

---
# **5. Machine Learning Model**

## **5.1 Architecture (Final)**
Defined in `shared/model.py`, the system uses a lightweight binary classifier:

- Input: 30 features
- Dense(64, ReLU)
- Dropout(0.2)
- Dense(32, ReLU)
- Dropout(0.2)
- Output: Dense(1, Sigmoid)

## **5.2 Training Parameters**
- Optimizer (clients): **DPKerasAdamOptimizer**
- Optimizer (server): Adam
- Loss: **BinaryCrossentropy**
- Batch size: **256** (required for DP microbatch equality)
- Local epochs: **2** (sent from server)

## **5.3 Differential Privacy (DP-SGD)**
- Uses **gradient clipping** and **Gaussian noise injection**
- Ensures record-level privacy guarantees

### **Important Implementation Detail**
`validation_split` is incompatible with DP-SGD.  
Solution: use `validation_data` instead.

---
# **6. Federated Learning Workflow**

## **Round 0: Baseline**
- Server initializes the global model
- Evaluates on `global_test_set.npz`

## **Each Round (1…N)**
1. **Send global model + config** to clients
2. **Local DP-SGD training** on each client
3. **Clients return updated weights**
4. **Server aggregates updates using FedAvg**
5. **Server evaluates model globally**
6. Process repeats for `NUM_ROUNDS`

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
│   ├── standalone_test.py
│   └── data/
│       ├── client_hospital_train.npz
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
- Server binds to: **0.0.0.0:8080**
- Clients connect to: **127.0.0.1:8080** (local setups)
- Communication handled via Flower's gRPC protocol

## **7.3 Execution Commands**
Run from project root:

### **Start Server**
```
python -m server.server
```

### **Start Client 1**
```
python -m client.run_client --client-id hospital
```

### **Start Client 2**
```
python -m client.run_client --client-id factory
```

---
# **8. Evaluation Metrics**
- Accuracy
- Binary Cross-Entropy Loss
- F1-Score (important for imbalanced attack data)
- Weighted mean of client-side evaluation metrics

---
# **9. Experimental Observations**

## **9.1 Catastrophic Forgetting**
Single-client training fails on unseen attack families.

## **9.2 Federated Generalization**
FedAvg averages knowledge across heterogeneous clients, achieving broad attack coverage.

## **9.3 DP-SGD Side Effects**
- Noise reduces accuracy marginally
- DP introduces strict batch size constraints
- `validation_split` incompatibility fixed

---
# **10. Limitations**
- FedAvg struggles with extreme non-IID skew
- DP-SGD reduces accuracy modestly
- Synchronous aggregation does not scale to very large client counts

---
# **11. Future Work**
- Implement advanced optimizers (FedProx, FedAdam, FedNova)
- Add secure aggregation (HE-based or enclave-based)
- Introduce sequence models (LSTMs, Transformers)
- Deploy multi-client real-time FL

---
# **12. Conclusion**

FED_IIDS demonstrates the feasibility of collaboratively training a high-quality intrusion detection model **without sharing raw data**, combining federated learning with differential privacy and realistic non-IID conditions.

All datasets required for the system are included within this repository, and this report reflects the final implementatio