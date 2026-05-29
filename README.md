# EXPOSE-Fake-News-Detection

EXPOSE-Fake-News-Detection is a research-grade machine learning system for fake news detection, designed to benchmark classical NLP methods against transformer-based models under controlled, leakage-aware experimental conditions. The project emphasizes reproducibility, interpretability, and robustness analysis.

---

## Abstract

This project investigates the problem of fake news detection using a structured experimental pipeline spanning classical machine learning, transformer-based models, and post-hoc explainability techniques. We perform rigorous dataset analysis, leakage detection, model benchmarking, and robustness evaluation to understand the limitations and strengths of different NLP approaches.

---

## Key Contributions

- 🔬 Leakage-aware evaluation pipeline for NLP classification  
- 📊 Benchmarking of classical ML vs transformer models  
- 🧠 Interpretability using SHAP-based explanations  
- ⚖️ Robustness and error analysis under distributional shifts  
- 📈 Structured research reports with reproducible figures  

---

## Problem Definition

Given a news article x, predict its label:

y ∈ {0, 1}

where:
- 0 → Real news  
- 1 → Fake news  

The objective is to learn a function:

f(x) → y

that generalizes well across unseen samples while remaining robust to dataset artifacts and leakage.

---

## Methodology Overview

The system follows a structured research pipeline:

### 1. Exploratory Data Analysis & Leakage Investigation
- Class distribution analysis  
- Subject/domain distribution study  
- Article length statistics  
- Leakage detection and validation  

📓 `01_EDA_and_Leakage_Investigation.ipynb`

---

### 2. Classical Machine Learning Baselines
- Logistic Regression  
- Support Vector Machines (SVM)  
- TF-IDF vectorization  
- Performance benchmarking  

📓 `02_Baseline_Models.ipynb`

---

### 3. Transformer-Based Models
- Fine-tuned transformer architectures  
- Context-aware embeddings  
- Comparative evaluation with baselines  

📓 `03_Transformer_Models.ipynb`

---

### 4. Explainability (SHAP Analysis)
- Feature importance estimation  
- Local and global explanations  
- Model interpretability analysis  

📓 `04_Explainability_SHAP.ipynb`

---

### 5. Robustness & Error Analysis
- Misclassification analysis  
- Failure mode categorization  
- Stress-testing under distribution shifts  

📓 `05_Robustness_and_Error_Analysis.ipynb`

---

## Repository Structure

```bash
EXPOSE-Fake-News-Detection/
│
├── notebooks/ # Research notebooks
│ ├── 01_EDA_and_Leakage_Investigation.ipynb
│ ├── 02_Baseline_Models.ipynb
│ ├── 03_Transformer_Models.ipynb
│ ├── 04_Explainability_SHAP.ipynb
│ └── 05_Robustness_and_Error_Analysis.ipynb
│
├── reports/ # Research artifacts
│ ├── figures/ # Visual analysis outputs
│ │ ├── class_distribution.png
│ │ ├── subject_distribution.png
│ │ ├── article_length_boxplot.png
│ │ ├── leakage_results.png
│ │ ├── confusion_matrix_lr.png
│ │ ├── confusion_matrix_svm.png
│ │ └── shap_explanation.png
│ │
│ └── research_report.pdf # Final compiled report
│
├── src/ # Core pipeline (modular code)
│ ├── preprocess.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── app/ # Deployment layer
│ └── streamlit_app.py
│
├── models/ # Trained artifacts
│ ├── logistic_regression.pkl
│ ├── svm.pkl
│ └── vectorizer.pkl
│
├── data/
│ └── README.md # Dataset documentation
│
├── results/ # Benchmark outputs
│ ├── baseline_results.csv
│ ├── transformer_results.csv
│ └── robustness_results.csv
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 📊 Experimental Summary

### Baseline Models
- TF-IDF + Logistic Regression  
- TF-IDF + SVM  

### Transformer Models
- Fine-tuned transformer encoder models for sequence classification  

---

## 📏 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix analysis  

---

## 🧠 Explainability

We use SHAP-based analysis to interpret model predictions:

- Global feature importance  
- Local instance-level explanations  
- Model behavior consistency checks  

📌 Output example:  
`reports/figures/shap_explanation.png`

---

## 🛡️ Robustness Analysis

We evaluate model stability under:

- Class imbalance sensitivity  
- Domain variation in news topics  
- Misclassification patterns  

📌 Outputs:
- `robustness_results.csv`
- `leakage_results.png`

---

## 📈 Results Summary

| Model                | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | XX.X%    | XX.X%    |
| SVM                 | XX.X%    | XX.X%    |
| Transformer         | XX.X%    | XX.X%    |

---

## 🚀 Streamlit Demo

Run the interactive inference system:

```bash
streamlit run app/streamlit_app.py
```

## 🔁 Reproducibility

Clone and run:

```bash
git clone https://github.com/<your-username>/EXPOSE-Fake-News-Research.git
cd EXPOSE-Fake-News-Research
pip install -r requirements.txt
```

Run Notebooks Sequentially

1. EDA & Leakage Analysis  
2. Baseline Models  
3. Transformer Models  
4. Explainability  
5. Robustness Analysis  

---

## 🧭 Design Philosophy

This project follows research lab principles:

- Reproducibility-first design  
- Leakage-aware evaluation  
- Separation of experimentation and deployment  
- Interpretability over black-box metrics  
- Robustness over raw accuracy  

---

## ⚖️ Ethical Considerations

Fake news detection systems must be deployed carefully due to:

- Risk of political bias amplification  
- False positive classification of legitimate content  
- Dataset and domain shift limitations  

This system is intended for **research and educational purposes only**.

---

## 📜 License

MIT License © 2026  

---

## 👤 Author

**Saonee Nandi**  
Focus: Machine Learning Theory • Numerical Methods • Computational Intelligence
