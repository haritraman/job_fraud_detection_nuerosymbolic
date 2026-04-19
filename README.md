 Neuro-Symbolic Framework for Fake Job Detection using DistilBERT with Explainability 🕵️‍♂️🤖


The growth of online job portals has made hiring easier but has also created a space for highly sophisticated recruitment scams. While traditional deep learning models can detect these scams, they act as "black boxes" that fail to explain their decisions. 

This project proposes a **Neuro-Symbolic Framework** that bridges the gap between neural performance and logical transparency. By combining the deep semantic text understanding of **DistilBERT** with a strict **15-rule Symbolic Logic Engine**, this system achieves state-of-the-art accuracy while providing visual, human-readable explanations for every flagged job post.

Key Features
* **Dual-Processing Engine:** Combines deep contextual understanding (Neural Tower) with explicit safety checks (Symbolic Tower).
* **Multiple Fusion Architectures:** Implements and compares *Baseline DistilBERT*, *Early Fusion*, and *Late Fusion* architectures.
* **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** to generate interactive waterfall plots, proving exactly which words and rules pushed the model toward a fraud prediction.
* **Imbalance Handling:** Utilizes PyTorch's `BCEWithLogitsLoss` with dynamic class weighting to heavily penalize minority class (fraud) misclassifications.

📊 Dataset
This project uses the **Employment Scam Aegean Dataset (EMSCAD)**. 
* **Total Records:** 17,880 job postings
* **Legitimate Jobs:** 17,014
* **Fraudulent Jobs:** 866
* **Link:** [EMSCAD via Kaggle](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam)

* System Architectures

### 1. Neuro-Symbolic Early Fusion
The text embedding extracted from DistilBERT is mathematically concatenated with the 15-dimensional binary rule vector before being passed through a single classification layer. This allows the model to learn hidden relationships between professional vocabulary and hard-coded safety rules.

### 2. Neuro-Symbolic Late Fusion (Recommended for High Security)
A strict two-tower approach. Text and rules are processed completely independently, and their individual logits are merged right before the final Sigmoid activation. This ensures that a highly polished, professional scam description can *never* override a critical symbolic rule violation.

 📈 Performance & Results

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline DistilBERT** | 98.29% | 97.00% | 93.00% | 94.00% |
| **Neuro-Symbolic Early Fusion** | 98.05% | 98.00% | 98.00% | 98.00% |
| **Neuro-Symbolic Late Fusion** | 97.26% | 98.00% | 97.00% | 97.00% |

*Note: While Early Fusion achieved the highest F1-Score, the Late Fusion architecture provides superior logical determinism for real-world deployment.*

# Explainability (SHAP)
This system does not just output a probability score; it explains *why*. Using a custom SHAP wrapper around the DistilBERT tokenizer, the system outputs waterfall plots for predictions:
* 🟥 **Red Highlights:** Words/symbols pushing the prediction toward **Fraud** (e.g., suspicious financial terminology like "₹" or "fee").
* 🟦 **Blue Highlights:** Words pushing the prediction toward **Legitimate**.

👥 Contributors
Raman Haritash (Conceptualization, Methodology, Original Draft)

Kriti Tiwari (Data Curation, Software, Visualization)

Lakshay Kalra (Validation, Formal Analysis, Review & Editing)

Prof. Saira Banu J (Supervision, Project Administration)

Institution: School of Computer Science & Engineering, Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India.
