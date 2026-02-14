import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Page configuration
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

st.title("🩺 M.Tech Assignment: Heart Disease Classification")
st.markdown("""
This application allows you to evaluate 6 different Machine Learning models 
on the Heart Disease (Cleveland) dataset.
""")

# Sidebar for Inputs
st.sidebar.header("User Input & Settings")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Select ML Model:", 
    ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload 'Heart_Disease_Cleaned.csv'", type=["csv"])

# Main Logic
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview (First 5 Rows)")
    st.dataframe(df.head())

    # Check if target exists
    if 'target' in df.columns:
        X = df.drop(['num', 'target'], axis=1, errors='ignore')
        y = df['target']

        # Load the selected model
        model_path = f"model/{model_choice}.pkl"
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Make Predictions
            preds = model.predict(X)
            
            # 3. Display Metrics
            st.subheader(f"Results for {model_choice.replace('_', ' ').upper()}")
            
            acc = accuracy_score(y, preds)
            st.metric("Model Accuracy", f"{acc:.2%}")

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y, preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

            with col2:
                st.write("#### Classification Report")
                report = classification_report(y, preds, output_dict=False)
                st.text(report)
        else:
            st.error(f"Error: Model file '{model_path}' not found in the 'model' folder.")
    else:
        st.error("Error: The uploaded CSV must contain a 'target' column.")
else:
    st.info("Please upload your cleaned dataset via the sidebar to see the model evaluation.")