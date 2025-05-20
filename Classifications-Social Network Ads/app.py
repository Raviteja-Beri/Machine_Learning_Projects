import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

st.set_page_config(layout="wide")

st.title("SVM Classifier on Social Network Ads Dataset")

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    dataset = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    # Extract features and target
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)  # Use transform here, not fit_transform again

    # Train SVM model
    svc_classifier = SVC()
    svc_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = svc_classifier.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    bias = svc_classifier.score(X_train, y_train)
    variance = svc_classifier.score(X_test, y_test)
    class_report = classification_report(y_test, y_pred, output_dict=False)

    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Bias (Training Score):** {bias:.4f}")
    st.write(f"**Variance (Test Score):** {variance:.4f}")
    st.text("Classification Report")
    st.code(class_report)

    def plot_decision_boundary(X_set, y_set, title):
        X1, X2 = np.meshgrid(
            np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
            np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
        )
        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2,
                     svc_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(title)
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()

        return plt

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Set Decision Boundary")
        fig_train = plot_decision_boundary(X_train, y_train, "SVM (Training set)")
        st.pyplot(fig_train)

    with col2:
        st.subheader("Test Set Decision Boundary")
        fig_test = plot_decision_boundary(X_test, y_test, "SVM (Test set)")
        st.pyplot(fig_test)
else:
    st.info("Please upload the `Social_Network_Ads.csv` file to proceed.")
