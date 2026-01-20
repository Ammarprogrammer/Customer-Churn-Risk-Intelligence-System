import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report , confusion_matrix,  roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert target column to categorical (R: as.factor)
data["Churn"] = data["Churn"].astype("category")

# Check class distribution (R: table)
print(data["Churn"].value_counts())

# Baseline accuracy (R: max(prop.table(table())))
baseline_accuracy = data["Churn"].value_counts(normalize=True).max()

print("Baseline Accuracy:", round(baseline_accuracy, 4))

# drop ID
data.drop("customerID", axis=1, inplace=True)

#convert outcome object to num 
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to Numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)

#labelecode all objects cols
label_encoders = {}

categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data["AvgMonthlySpend"] = (data["TotalCharges"] / data["tenure"].replace(0, 1)).round(2)

X = data.drop("Churn",axis=1)
Y = data["Churn"]
feature_names = X.columns

Scaler = StandardScaler()
X = Scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2 , random_state=42)

scale_pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, Y_train)
y_probs = model.predict_proba(X_test)[:,1]
y_pred = (y_probs >= 0.424).astype(int)
model.save_model("cus_churn.pkl")

print('Classification report')
print(classification_report(Y_test,y_pred))

conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=Y.unique(), yticklabels=Y.unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.tight_layout()
plt.show()

importance = model.feature_importances_

# Make a DataFrame for plotting
lab = pd.DataFrame({
    "Feature": feature_names,
    "Dependency (%)": importance * 100
}).sort_values(by="Dependency (%)", ascending=False)

# Plot
plt.figure()
sns.barplot(data=lab, x="Dependency (%)", y="Feature")
plt.title("Feature Dependency on Prediction (%)")
plt.show()

print("---- Customer Churn Prediction ----")

try:

    # ----------- User Inputs -----------

        gender = str(input("Gender (Male/Female): ")).capitalize()
        SeniorCitizen = int(input("Senior Citizen? (0 = No, 1 = Yes): "))
        Partner = str(input("Partner (Yes/No): ")).capitalize()
        Dependents = str(input("Dependents (Yes/No): ")).capitalize()
        tenure = int(input("Tenure (months): "))
        PhoneService = str(input("Phone Service (Yes/No): ")).capitalize()
        MultipleLines = str(input("Multiple Lines (Yes/No): ")).capitalize()
        InternetService = str(input("Internet Service (DSL/Fiber optic/No): ")).capitalize()
        OnlineSecurity = str(input("Online Security (Yes/No): ")).capitalize()
        OnlineBackup = str(input("Online Backup (Yes/No): ")).capitalize()
        DeviceProtection = str(input("Device Protection (Yes/No): ")).capitalize()
        TechSupport = str(input("Tech Support (Yes/No): ")).capitalize()
        StreamingTV = str(input("Streaming TV (Yes/No): ")).capitalize()
        StreamingMovies = str(input("Streaming Movies (Yes/No): ")).capitalize()
        Contract = str(input("Contract (Month-to-month/One year/Two year): ")).capitalize()
        PaperlessBilling = str(input("Paperless Billing (Yes/No): ")).capitalize()
        PaymentMethod = str(input("Payment Method (Electronic check/Mailed check/Bank transfer/Credit card): ")).capitalize()
        MonthlyCharges = float(input("Monthly Charges: "))
        TotalCharges = float(input("Total Charges: "))

        user_input_df = pd.DataFrame([{
          'gender': gender,
          'SeniorCitizen': SeniorCitizen,
          'Partner': Partner,
          'Dependents': Dependents,
          'tenure': tenure,
          'PhoneService': PhoneService,
          'MultipleLines': MultipleLines,
          'InternetService': InternetService,
          'OnlineSecurity': OnlineSecurity,
          'OnlineBackup': OnlineBackup,
          'DeviceProtection': DeviceProtection,
          'TechSupport': TechSupport,
          'StreamingTV': StreamingTV,
          'StreamingMovies': StreamingMovies,
          'Contract': Contract,
          'PaperlessBilling': PaperlessBilling,
          'PaymentMethod': PaymentMethod,
          'MonthlyCharges': MonthlyCharges,
          'TotalCharges': TotalCharges,
          'AvgMonthlySpend': TotalCharges/tenure,
        
        }])

    # ----------- Encode Categorical Columns -----------
    # categorical_cols = all string/object type
        categorical_cols = user_input_df.select_dtypes(include='object').columns

    # label_encoders is a dict containing fitted LabelEncoders from training
    # scaler is the fitted StandardScaler from training
        for col in categorical_cols:
          le = LabelEncoder()
          user_input_df[col] = le.fit_transform(user_input_df[col])
          label_encoders[col] = le

    # ----------- Scale Numeric Features -----------
        user_input_scaled = Scaler.transform(user_input_df)

    # ----------- Make Prediction -----------
        prediction = model.predict(user_input_scaled)[0]
        probability = model.predict_proba(user_input_scaled)[0][1]  # probability of churn

    # ----------- Display Result -----------
        if prediction == 0:
          print("\nPrediction: Customer will NOT churn ✅")
        else:
          print("\nPrediction: Customer is likely to churn ⚠️")

        print(f"Churn Probability: {round(probability*100, 2)}%")

except Exception as e:
    print("Error:", e)