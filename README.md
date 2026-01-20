# Client Churn Guard: Proactive Prediction & Analysis

> **An end-to-end, production-style Machine Learning project that predicts customer churn and supports data-driven retention strategies.**

This project is built to **showcase real-world Machine Learning skills** that recruiters look for: business understanding, feature engineering, model selection, and deployment readiness.

---

## Business Problem

**Key Question:**  
> Which customers are at high risk of leaving, and how confident is the prediction?

**Business Value:**
- Reduce revenue loss by identifying at-risk customers.
- Improve customer lifetime value (CLV) by implementing targeted retention campaigns.
- Enable proactive customer support and engagement.

---

## Features

- **Interactive UI:** A user-friendly web interface built with Streamlit.
- **Real-time Predictions:** Predict churn risk for a single customer based on their information.
- **Probability Score:** Get the probability of churn to gauge the confidence of the prediction.
- **Dynamic & Responsive:** The UI is designed to be responsive and easy to use on different screen sizes.

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/customer-churn-predictor.git
    cd customer-churn-predictor
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file. See the section below.)*

### Creating `requirements.txt`

The following libraries are required to run the application. You can create a `requirements.txt` file with the following content:

```
streamlit
pandas
xgboost
scikit-learn
numpy
```

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser and navigate to the provided URL (usually `http://localhost:8501`).**

---

## Machine Learning Approach

### Model
- **XGBoost Classifier**

**Why XGBoost?**
- High predictive performance due to its gradient boosting algorithm.
- Efficiently handles missing values and non-linear relationships.
- Widely adopted in the industry for its robustness and speed.

### Problem Type
- **Binary Classification**
  - `0` → Customer will **NOT churn**
  - `1` → Customer **WILL churn**

### Features Used
The model is trained on a set of 20 features, including:
- **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Financials:** `MonthlyCharges`, `TotalCharges`
- **Engineered Feature:** `AvgMonthlySpend` (TotalCharges / tenure)

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset (available on Kaggle).
- **Records:** ~7,000 customers.
- **Target Variable:** `Churn`

---

## 📂 Project Structure

```
.
├── app.py                  # The Streamlit web application script.
├── cus_churn.pkl           # The trained XGBoost model file.
├── model.py                # The script for training the model.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # The dataset file.
└── README.md               # This file.
```

**Ammar Ahmed**
- [LinkedIn](https://www.linkedin.com/in/ammarahmeddeveloper/)
- [GitHub](https://github.com/Ammarprogrammer)
