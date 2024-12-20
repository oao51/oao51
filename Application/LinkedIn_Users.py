import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

#Application title w small desc.
st.title("Predict LinkedIn Users")
st.write("This app predicts whether someone is a LinkedIn user based on demographic attributes. Provide the inputs in the sidebar to see the prediction.")

# Sidebar for User Inputs
st.sidebar.header("Input User Details")

# User inputs
income = st.sidebar.number_input("Income (1-9)", min_value=1, max_value=9, value=5)
education = st.sidebar.number_input("Education (1-8)", min_value=1, max_value=8, value=5)
age = st.sidebar.number_input("Age (1-98)", min_value=1, max_value=98, value=30)
parent = st.sidebar.selectbox("Parent (1: Yes, 0: No)", options=[0, 1])
married = st.sidebar.selectbox("Married (1: Yes, 0: No)", options=[0, 1])
female = st.sidebar.selectbox("Female (1: Yes, 0: No)", options=[0, 1])

# File uploader for training data
uploaded_file = st.file_uploader("Upload your CSV file for training", type="csv")

if uploaded_file is not None:
    # Load and preprocess the data
    data = pd.read_csv(uploaded_file)

    # Data preprocessing
    data['income'] = np.where(data['income'] <= 9, data['income'], np.nan)
    data['education'] = np.where(data['educ2'] <= 8, data['educ2'], np.nan)
    data['age'] = np.where(data['age'] <= 98, data['age'], np.nan)
    data['parent'] = np.where(data['par'] == 1, 1, 0)
    data['married'] = np.where(data['marital'] == 1, 1, 0)
    data['female'] = np.where(data['gender'] == 2, 1, 0)
    data['sm_li'] = np.where(data['clean_sm'] == 1, 1, 0)

    # Drop missing values
    data = data.dropna()

    # Define target and features
    X = data[['income', 'education', 'age', 'parent', 'married', 'female']]
    y = data['sm_li']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic regression model
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    st.write("Model Evaluation:")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    #prediction in real time
    user_data = pd.DataFrame({
        'income': [income],
        'education': [education],
        'age': [age],
        'parent': [parent],
        'married': [married],
        'female': [female]
    })

    prediction = lr.predict(user_data)[0]
    probability = lr.predict_proba(user_data)[0][1]

    # show prediction
    st.subheader("Prediction Results")
    st.write(f"Predicted LinkedIn User: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of LinkedIn Usage: {probability:.2f}")

else:
    st.write("Please upload a CSV file to train the model.")


