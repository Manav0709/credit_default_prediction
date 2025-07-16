# 💳 Credit Default Prediction using XGBoost

This project predicts whether a customer will default on their credit card payment next month, using demographic and historical payment data. It uses an XGBoost classifier with hyperparameter tuning to optimize performance, especially recall — a key metric in real-world credit risk assessment.

---

## 📌 Problem Statement

**Goal:** Build a machine learning model to predict credit card defaults and minimize the risk of issuing credit to potentially unreliable customers.

**Use Case:** Banks and financial institutions like **American Express** can use this model to improve credit decision-making, reduce financial losses, and flag high-risk accounts.

---

## 📂 Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Type: Tabular data with 30,000 customer records
- Target column: `default payment next month` (1 = Default, 0 = No Default)

---

## 🔍 Features Used

| Category         | Features |
|------------------|----------|
| **Demographics** | `SEX`, `AGE`, `EDUCATION`, `MARRIAGE` |
| **Credit Info**  | `LIMIT_BAL`, `BILL_AMT1-6`, `PAY_AMT1-6` |
| **Repayment**    | `PAY_0` to `PAY_6` (Past 6 months repayment status) |

---

## 🧠 Model Overview

| Model           | Method                 |
|-----------------|------------------------|
| Algorithm       | XGBoost Classifier     |
| Tuning          | GridSearchCV (Recall optimized) |
| Evaluation      | Classification Report, Confusion Matrix |
| Threshold       | Adjusted to 0.4 to increase recall |

---

## 📈 Model Performance (Final Tuned XGBoost)

| Metric            | Value      |
|-------------------|------------|
| **Recall (Default)** | `0.75` ✅ |
| **Precision (Default)** | `0.37` |
| **F1-score (Default)** | `0.50` |
| **False Negatives** | Significantly reduced |
| **True Positives** | High, crucial for catching defaulters |

> 🎯 **Why Recall?**  
> In risk modeling, it's more important to catch potential defaulters (even at the cost of false alarms), as the financial cost of missing one can be huge.

---

## 🚀 Usage

1. Run `credit_default_prediction.ipynb` in Google Colab.
2. Load the UCI dataset and preprocess.
3. Train XGBoost classifier.
4. Tune hyperparameters using GridSearchCV.
5. Evaluate model using `classification_report()` and `confusion_matrix()`.

---

## ✅ Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Jupyter/Colab

---

## 💼 Real-World Relevance

- This project aligns with **American Express**’s real-world use of ML for **fraud detection**, **risk scoring**, and **credit decisioning**.
- Financial institutions need **high-recall models** to flag risky customers and reduce losses.

---

## ❓ Common Interview Questions

### 🧠 Conceptual
- Why did you choose XGBoost over other models?
- What metric did you optimize and why?
- Why is recall more important than accuracy in credit scoring?
- How does XGBoost handle missing data or multicollinearity?

### 🛠️ Technical
- How does GridSearchCV work?
- How did you handle class imbalance?
- Why did you choose 0.4 as a threshold instead of 0.5?
- Did you try other models (e.g., Logistic Regression, Random Forest)?

### 🔎 Explainability
- Which features were most important in your model?
- How can this model be used in production?
- How would you monitor model drift or update it over time?

---

## 📈 Possible Extensions

- Add SHAP explainability for feature insights
- Build a Streamlit front-end for business users
- Deploy the model using FastAPI or Streamlit Cloud
- Retrain with monthly/quarterly refreshed data

---

## 👨‍💻 Author

**Manav Sharma**  
Aspiring Data Scientist | FinTech & ML Enthusiast  
LinkedIn: https://www.linkedin.com/in/manav-sharma-481023311/
GitHub: https://github.com/Manav0709

---

## 🏁 Final Words

This project is a practical, resume-ready example of how machine learning can be applied to credit scoring — highly relevant to companies like **American Express**.

