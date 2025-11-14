# **FED_IIDS: A Federated Intrusion Detection System**

### *Privacy-Preserving Federated Learning for Network Intrusion Detection*

GitHub Repository: **[https://github.com/letsbegincode/Fed_IIDS](https://github.com/letsbegincode/Fed_IIDS)**

---

## **Overview**

**FED_IIDS** is a **privacy-preserving Federated Learning (FL) system** for building a **Network Intrusion Detection System (NIDS)** using deep learning.
It simulates a real-world scenario where multiple organizations—such as a **hospital** and a **factory**—collaboratively train a **global IDS model** **without sharing private network traffic data**.

The system is built using:

* **Flower Federated Learning Framework** (FedAvg)
* **TensorFlow / Keras Deep Learning Model**
* **TensorFlow Privacy (DP-SGD)** for Differential Privacy
* **Custom Non-IID Data Partitions** (core research contribution)

---

## **Key Features**

### **Federated Learning (FL)**

* Built using **Flower** to coordinate distributed training.
* Implements the **FedAvg** aggregation algorithm.
* Global model updated after each communication round.

### **Privacy-Preserving Training**

* Uses **DP-SGD** (Differentially Private SGD) via `tensorflow-privacy`.
* Ensures individual client records cannot be reconstructed from model updates.

### **Deep Learning-Based NIDS**

* Modular **Keras DNN** defined in `model.py`.
* Acts as the **API contract** shared across all clients and the server.

### **Non-IID Data Simulation (Core Research Component)**

Realistic partitioning to simulate domain-specific traffic:

**Client 1 (Hospital)**

* Benign
* Spoofing
* Web-Based attacks

**Client 2 (Factory)**

* Benign
* DoS
* DDoS
* Mirai

### **Advanced Data Preprocessing Pipeline**

Includes:

* Aggregation of **1.6M+ network flows**
* Cleaning, sanitization, and NaN/Inf removal
* Two-stage feature reduction:

  * **Correlation-based filtering**
  * **LightGBM tree-based feature selection**
* Final feature set reduced **from 79 → Top 30** predictive features

---

## **Project Architecture**

### 📌 **Root Folder**

#### **server.py**

* Acts as the **central coordinator**
* Initializes global model
* Waits for at least **2 clients**
* Shares global weights, collects updates
* Runs **FedAvg aggregation**
* Evaluates on `global_test_set.npz` (never seen by clients)

---

###  **client/**

Self-contained FL client module.

| File             | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `run_client.py`  | Entry point, takes `--client-id`                                 |
| `nids_client.py` | Flower NumPyClient: controls `fit()` and `evaluate()`            |
| `config.py`      | Central configuration: server IP, DP parameters, training params |
| `model.py`       | Shared Keras model                                               |
| `data_loader.py` | Loads client's local non-IID datasets                            |
| `data/`          | Contains `.npz` files for each client                            |

---

## **Installation & Setup**

### **1. Prerequisites**

* Python **3.9+**
* Git
* 3 separate machines (or same machine with multiple terminals):

  * **Server**
  * **Client 1 (Hospital)**
  * **Client 2 (Factory)**

---

## **2. Installation**

### Clone the repository

```bash
git clone https://github.com/letsbegincode/Fed_IIDS.git
cd Fed_IIDS
```

### Create a virtual environment

```bash
python -m venv venv
```

#### Windows:

```bash
.\venv\Scripts\activate
```

#### Mac/Linux:

```bash
source venv/bin/activate
```

### Install requirements (all team members)

```bash
cd client
pip install -r requirements.txt
```

---

## **3. Data Setup (Most Important Step)**

> **This project does NOT ship with data.**
> Download the `.npz` files using the Google Drive link provided in `client/data/README.md`.

### Required files:

* `client_hospital_train.npz`
* `client_hospital_test.npz`
* `client_factory_train.npz`
* `client_factory_test.npz`
* `global_test_set.npz`

Place all files into:

```
FED_IIDS/client/data/
```

### Now partition them:

#### **Server Machine**

Keep only:

```
global_test_set.npz
```

#### **Client 1 (Hospital)**

Keep:

```
client_hospital_train.npz
client_hospital_test.npz
```

#### **Client 2 (Factory)**

Keep:

```
client_factory_train.npz
client_factory_test.npz
```

---

# ▶️ **How to Run the Simulation**

---

## 🎛️ **Step 1: Start the Server**

### Find your IP address:

Windows:

```bash
ipconfig
```

Linux/Mac:

```bash
ifconfig
```

Example: `192.168.1.10`

### Run the server:

From the root folder:

```bash
python server.py
```

Allow firewall access when prompted.

The server now waits for 2 clients.

---

## 🖥️ **Step 2: Start Clients**

### Set the server IP in:

`FED_IIDS/client/config.py`

```python
SERVER_ADDRESS = "192.168.1.10:8080"
```

### Run Client 1 (Hospital)

```bash
python run_client.py --client-id hospital
```

### Run Client 2 (Factory)

```bash
python run_client.py --client-id factory
```

Once both connect, training begins automatically.

Monitor global metrics (accuracy, F1-score) in the server terminal.

---

# 📊 **Output**

After each federated round:

* Server displays global model performance on unseen dataset
* Clients display local training loss and accuracy
* FedAvg aggregated model improves over rounds

---

# 📘 **Folder Structure**

```
Fed_IIDS/
│
├── server.py
├── client/
│   ├── run_client.py
│   ├── nids_client.py
│   ├── config.py
│   ├── model.py
│   ├── data_loader.py
│   └── data/
│       ├── client_hospital_train.npz
│       ├── client_hospital_test.npz
│       ├── client_factory_train.npz
│       ├── client_factory_test.npz
│       └── global_test_set.npz
└── Fed_IIDS.ipynb
```

---

# 🧑‍💻 **Contributors**


---
