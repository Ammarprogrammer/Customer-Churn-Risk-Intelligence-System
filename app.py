import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM THEME ----------------
st.markdown("""
<style>
    body {
        background-color: #0f172a;
    }

    .main {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
    }

    h1 {
        font-weight: 800;
        font-size: clamp(1.6rem, 2.5vw, 2.4rem);
        color: #0f172a;
    }

    h2, h3 {
        color: #1e40af;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white;
        border-radius: 10px;
        padding: 0.7rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
    }

    .stButton>button:hover {
        transform: scale(1.01);
    }

    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
model = xgb.Booster()
model.load_model("cus_churn.pkl")

# ---------------- LOAD & PREP DATA ----------------
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)

label_encoders = {}
categorical_cols = data.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(["Churn", "customerID"], axis=1)
X["AvgMonthlySpend"] = (data["TotalCharges"] / data["tenure"].replace(0, 1)).round(2)

scaler = StandardScaler()
scaler.fit(X)

# ---------------- TITLE ----------------
st.title("🛡️ Client Churn Guard: Proactive Prediction & Analysis")

st.markdown("""
AI-powered system that predicts **customer churn risk** to help businesses:

- 🔍 Identify high-risk customers  
- 💰 Reduce revenue loss  
- 📈 Improve retention strategy  
""")

# ---------------- KPI CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "XGBoost Classifier")
with col2:
    st.metric("Domain", "Telecom / SaaS")
with col3:
    st.metric("Goal", "Customer Retention")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧾 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0
)

TotalCharges = st.sidebar.number_input(
    "Total Charges ($)", min_value=0.0, max_value=20000.0, value=2000.0, step=50.0
)

# ---------------- USER DATAFRAME ----------------
user_input = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

user_df = pd.DataFrame([user_input])
user_df["AvgMonthlySpend"] = (user_df["TotalCharges"] / max(1, user_df["tenure"][0])).round(2)

# Encode categorical
for col in categorical_cols:
    if col in user_df.columns:
        user_df[col] = label_encoders[col].transform(user_df[col])

# Align columns
user_df = user_df[X.columns]

# Scale
user_scaled = scaler.transform(user_df)
dmatrix = xgb.DMatrix(user_scaled, feature_names=X.columns.tolist())

# ---------------- PREDICTION ----------------
if st.sidebar.button("🚀 Predict Churn"):
    proba = model.predict(dmatrix)[0]
    prediction = 1 if proba > 0.424 else 0

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Churn Risk Detected")
        st.markdown("""
        **Recommended Actions:**
        - Offer loyalty discounts  
        - Improve customer engagement  
        - Proactive support outreach  
        """)
    else:
        st.success("✅ Low Churn Risk")
        st.markdown("""
        **Customer Status:**
        - Stable relationship  
        - No immediate action required  
        """)

    st.subheader("📈 Churn Probability")
    st.progress(float(proba))
    st.caption(f"Predicted churn probability: **{proba*100:.2f}%**")

# ---------------- FOOTER ----------------
st.markdown("""
---
👨‍💻 **Developed by Ammar Ahmed**  
📂 Project: Real-World Customer Churn Prediction using XGBoost  
🔗 [LinkedIn](https://www.linkedin.com/in/ammarahmeddeveloper)
""")
