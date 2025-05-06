# 🧠 Machine Learning Projects

Welcome to my collection of machine learning projects! This repository contains end-to-end implementations of various ML algorithms and web apps to demonstrate practical use cases using real-world datasets.

---

## 📌 Projects

### 🏠 House Price Prediction using Simple Linear Regression

#### 🔍 Objective
Predict the price of a house based on its square footage using a Simple Linear Regression model.

#### ⚙️ Project Structure
- **`main.py`**: 
  - Loads and preprocesses the dataset (`House_data.csv`).
  - Trains a Simple Linear Regression model using `scikit-learn`.
  - Visualizes training and test results using Matplotlib.
  - Serializes the trained model using `pickle`.

- **`app.py`**:
  - Interactive **Streamlit** web application.
  - Accepts user input (square footage) and predicts house price.
  - Loads the trained model and performs inference in real-time.

#### 📊 Features
- End-to-end pipeline from training to deployment.
- Clean data visualizations for better understanding of model performance.
- Interactive user interface using Streamlit.
- Model saved and reused using Python’s `pickle`.

#### 🧰 Technologies Used
- Python
- NumPy, Pandas, Matplotlib
- scikit-learn
- Streamlit

#### 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raviteja-Beri/Machine_Learning_Projects.git
   cd Machine_Learning_Projects/House_Price_Prediction


