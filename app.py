import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload Telco Customer Churn CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing
    data.drop("customerID", axis=1, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = pd.get_dummies(data)

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("🔍 Model Evaluation")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    st.subheader("📈 Top 10 Important Features")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False)[:10]

    fig, ax = plt.subplots()
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    st.pyplot(fig)

    # Sample prediction
    st.subheader("🔮 Sample Prediction")
    sample = X_test.iloc[0:1]
    pred = model.predict(sample)
    st.write("Prediction:", "Churn" if pred[0] == 1 else "No Churn")

else:
    st.info("Upload a CSV file to start")
