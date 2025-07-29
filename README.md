# 🚨 End-to-End Fraud Detection Model

This repository contains a comprehensive machine learning project aimed at detecting fraudulent financial transactions. It demonstrates a full pipeline — from initial data exploration and feature engineering to building a high-performance predictive model and deriving actionable business insights.

---

## 📌 Project Overview

The primary goal is to build a robust and accurate fraud detection model using advanced machine learning techniques. The model is designed to assist in real-time fraud prevention, helping minimize financial losses and protect customers.

### 🔑 Key Features

- **Large-Scale Dataset:** Trained and evaluated on a dataset of over **6.3 million transactions**.
- **Advanced Feature Engineering:** Created behavioral features to improve model performance significantly.
- **Imbalance Handling:** Combined **SMOTE** and **RandomUnderSampler** to tackle extreme class imbalance.
- **High-Performance Model:** Tuned **XGBoost classifier**, achieving near-perfect test results.
- **Actionable Insights:** Identified key fraud drivers for potential business rule integration.

---

## 📈 Model Performance

| Metric             | Score   |
|--------------------|---------|
| **Test AUC**       | 0.9997  |
| **Average Precision** | 0.9985  |

These results indicate a **near-perfect ability** to distinguish between fraudulent and legitimate transactions.

---

## 🧠 Methodology

### 1. 🔍 Exploratory Data Analysis (EDA)

- Analyzed patterns in transaction types and user behaviors.
- Compared distributions of features for fraud vs. non-fraud transactions.

### 2. 🧪 Feature Engineering

Created new features that capture transactional anomalies and behavior:

- **`errorBalanceOrig`:** Measures balance inconsistencies post-transaction.
- **`exactBalance`:** Flags transactions that drain the entire balance.
- **Time-based features** like `hourOfDay` to capture temporal fraud patterns.

### 3. 🧹 Data Preprocessing & Imbalance Handling

- Stratified train-test split to maintain fraud ratio.
- Applied **SMOTE** to oversample the minority class.
- Applied **RandomUnderSampler** to downsample the majority class.
- Scaled features using `StandardScaler`.

### 4. ⚙️ Model Training & Optimization

- Evaluated multiple models — **XGBoost** gave the best results.
- Tuned hyperparameters with `RandomizedSearchCV`, optimizing for **Average Precision Score**.

### 5. 🧾 Evaluation & Interpretation

- Used **ROC Curve**, **Precision-Recall Curve**, and **Confusion Matrix** for evaluation.
- Identified top predictive features using XGBoost’s feature importance scores.

---

## 🚨 Key Fraud Indicators

The model found several strong predictors of fraud:

- **Balance Errors:** Post-transaction discrepancies in sender’s balance.
- **Transaction Types:** `TRANSFER` and `CASH_OUT` are highly associated with fraud.
- **High Transaction Amounts:** Larger sums are inherently riskier.
- **Account Draining:** Entire balance being transferred is highly suspicious.

---

## 🛠️ How to Use This Project

This repository includes all files needed to replicate the model or make new fraud predictions.

### 📁 Files Included

- `fraud_detection_notebook.ipynb` – Full analysis, training, and evaluation code.
- `fraud_detection_model.pkl` – Saved trained XGBoost model.
- `fraud_detection_scaler.pkl` – Scaler object used in preprocessing.
- `feature_names.pkl` – Ordered list of features expected by the model.

### 📊 Making a Prediction

To predict the fraud risk of a new transaction:

1. Load the model, scaler, and feature list using `joblib`.
2. Format new transaction data as a DataFrame with correct columns.
3. Scale the features using the loaded scaler.
4. Use `.predict_proba()` method to get the fraud risk score.

---

## 🧰 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing, model selection
- **XGBoost** – Final classification model
- **Imbalanced-learn** – SMOTE & under-sampling strategies

---

## 📬 Contact

For questions, contributions, or feedback, feel free to open an issue or pull request!

