# Machine Learning — MSc Data Science, University of Bath

Applied machine learning coursework from the MSc Data Science programme at the University of Bath (2023–2025). Covers supervised learning, probabilistic modelling, and model evaluation — from first-principles implementation through to practical application on real datasets.

---

## Assignments Overview

### Lab 1 — Exploratory Data Analysis & Preprocessing
**Dataset:** Loan Approval Dataset  
**Objective:** Profile a real financial dataset, identify data quality issues, and prepare features for downstream modelling.

Key work:
- Structured EDA methodology: schema profiling → missing value analysis → distribution inspection → correlation mapping
- Categorical and numerical feature analysis with business interpretation
- Visualisation of class imbalance and its implications for model evaluation
- Missing value strategy selection (imputation vs removal) with justification

`ML1_Lab_1_Data_Exploration.ipynb`

---

### Lab 2 — Decision Trees & Feature Importance
**Dataset:** Loan Approval Dataset  
**Objective:** Build and evaluate a decision tree classifier; interpret feature importance for business explainability.

Key work:
- CART algorithm implementation with Gini impurity and entropy splitting criteria
- Tree depth tuning to balance bias-variance tradeoff — visualised with validation curves
- Feature importance extraction and business interpretation (which variables drive loan decisions)
- Comparison of pre-pruning vs post-pruning on generalisation performance
- Confusion matrix analysis with precision, recall, and F1-score breakdown

`ML1_Lab_2_Decision_Trees.ipynb`

---

### Lab 3 — Linear Regression & Model Evaluation
**Dataset:** House Prices Dataset  
**Objective:** Predict continuous outcomes using linear regression; evaluate model assumptions and diagnostic metrics.

Key work:
- OLS regression from scratch vs scikit-learn implementation
- Feature scaling (StandardScaler, MinMaxScaler) and its effect on coefficient interpretation
- Residual analysis to validate regression assumptions (linearity, homoscedasticity, normality)
- Model evaluation using RMSE, MAE, and R² — with explanation of when each metric matters
- Feature selection using correlation analysis and variance inflation factor (VIF) for multicollinearity

`ML1_Lab_3_Linear_Regression.ipynb`

---

### Lab 4 — Bayesian Networks & Probabilistic Fault Prediction
**Dataset:** Simulated industrial sensor data (coffee machine fault prediction)  
**Objective:** Model conditional dependencies between system components using Bayesian networks; apply belief propagation for probabilistic inference.

Key work:
- Construction of a directed acyclic graph (DAG) representing causal relationships between sensor readings and fault states
- Conditional probability table (CPT) specification and parameter learning
- Belief propagation for exact inference — computing posterior fault probability given observed sensor states
- Sensitivity analysis: which sensor inputs most change the posterior fault probability
- Application discussion: predictive maintenance use cases in manufacturing and IoT

`ML1_Lab_4_Bayesian_Network_Fault_Prediction.ipynb`

> **Note on filename:** File renamed from original submission filename to follow consistent naming convention.

---

## Skills Demonstrated

| Skill | Labs |
|-------|------|
| Structured EDA and data profiling | Lab 1, Lab 2 |
| Supervised classification (tree-based models) | Lab 2 |
| Supervised regression with diagnostic evaluation | Lab 3 |
| Probabilistic graphical models | Lab 4 |
| Model evaluation and performance interpretation | Lab 2, Lab 3 |
| Feature importance and business explainability | Lab 2 |
| Bayesian reasoning and inference | Lab 4 |

---

## Technical Environment

```
Python 3.10+
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.1
matplotlib >= 3.6
seaborn >= 0.12
pgmpy >= 0.1.19    # Bayesian networks (Lab 4)
```

---

## Programme Context

These labs were completed as part of the MSc Data Science at the University of Bath (2023–2025). The programme combined statistical theory, applied machine learning, and data engineering — completed alongside a Data Analyst internship at Predictea Digital.

MSc awarded with **Dean's Award for Academic Excellence** — presented to top-performing postgraduate students for distinction-level achievement.

---

## Related Portfolio Projects

The methods covered in these labs are applied at scale in production-style projects:

- **[Retail KPI Analytics System](https://github.com/anandi-mahure/retail-kpi-analytics)** — anomaly detection using statistical methods (Z-score on time-series data), SQL + Power BI
- **[Customer Churn & Retention Analytics](https://github.com/anandi-mahure/retail-kpi-analytics)** — classification models (Logistic Regression, XGBoost) on 150K+ customer records, Tableau
- **[MSc Dissertation — Sentiment Analysis](https://github.com/anandi-mahure/sentiment-analysis-dissertation)** — NLP classification advancing from traditional ML (SVM: F1 0.66) to fine-tuned DistilBERT (F1 0.78)

---

*Anandi Mahure | MSc Data Science, University of Bath, 2025*  
*[linkedin.com/in/anandirm](https://linkedin.com/in/anandirm)*
