# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning techniques. The dataset is highly imbalanced, with fraud cases being rare compared to legitimate ones. We use sampling strategies and ensemble learning techniques to build effective classifiers.

## 📊 Data

The dataset used for this project is from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains **284,807 transactions**, among which only **394 are fraudulent**. Each transaction is anonymized and contains numerical input features resulting from a PCA transformation, except for `Time` and `Amount`.

### Key Data Insights

- **Transaction Amounts**:
  - Highly **right-skewed** with a sharp peak near zero.
  - Most legitimate transactions are of **small amounts**.
  - Outliers go as high as **$25,000+**, but these are rare.

- **Fraud Timing**:
  - Fraudulent activity **spikes around 2 AM**, suggesting potential bot-driven fraud.
  - Another minor peak around **11 AM**.
  - Activity is **moderate between 12 PM - 6 PM**, possibly due to higher transaction volumes.

## 🧠 Models

The project compares multiple models to detect fraudulent transactions, using metrics appropriate for imbalanced classification such as **precision**, **recall**, **F1-score**, **AUPRC**, and **ROC-AUC**.

### Classifiers Used

- **XGBoost Classifier** (with and without resampling)
- **Logistic Regression**
- **Random Forest**

### Sampling Techniques

- The original dataset is highly imbalanced:
  - Legitimate: 227,451
  - Fraudulent: 394

- **Resampled dataset** used for training:
  - Legitimate: 227,451
  - Fraudulent: 227,451 (via oversampling)

### Key Metrics

| Model               | Recall (Fraud) | Precision (Fraud) | F1-score (Fraud) | AUPRC  | ROC-AUC |
|---------------------|----------------|-------------------|------------------|--------|---------|
| XGBoost (raw)       | 0.88           | 0.38              | 0.53             | 0.8412 | 0.9799  |
| XGBoost (tuned)     | 0.85           | 0.81              | 0.83             | 0.8771 | —       |
| Logistic Regression | 0.85           | 0.81              | 0.83             | 0.8771 | —       |
| Random Forest       | 0.85           | 0.81              | 0.83             | 0.8771 | —       |

> Note: The models show a high level of **accuracy**, but recall and AUPRC are prioritized due to the imbalanced nature of the dataset.

## 📁 Notebooks

The project is implemented in Python in the following Jupyter notebook:

- [`Credit Card Fraud Detection.ipynb`](./Credit%20Card%20Fraud%20Detection.ipynb): Complete end-to-end workflow including:
  - Data preprocessing and EDA
  - Feature insights
  - Handling class imbalance
  - Model training and evaluation

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
