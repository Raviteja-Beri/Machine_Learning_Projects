# 🧠 Machine Learning Projects

Welcome to my collection of machine learning projects! This repository contains end-to-end implementations of regression models using real-world data, along with interactive web applications powered by Streamlit.

---

## 📌 Projects

---

### 🏠 House Price Prediction using Simple Linear Regression

#### 🔍 Objective  
Predict the price of a house based on its square footage using a Simple Linear Regression model.

#### ⚙️ Project Structure
- **`main.py`**:
  - Loads housing dataset (`House_data.csv`).
  - Trains a linear regression model to learn price vs. square footage.
  - Visualizes training and test set performance.
  - Saves the trained model to a `.pkl` file using `pickle`.

- **`app.py`**:
  - Streamlit web app that allows users to input square footage.
  - Loads the saved model and provides real-time house price prediction.

#### 📊 Features
- Clean visualizations of model fit.
- Interactive Streamlit frontend.
- End-to-end model deployment using `pickle`.

#### 🧰 Technologies Used
- Python, NumPy, Pandas, Matplotlib
- scikit-learn
- Streamlit

#### 📝 Sample Output
> ✅ The predicted price for a house with **1000** square feet is: **$245,000.00**

---

### 💼 Salary Prediction using Simple Linear Regression

#### 🔍 Objective  
Predict the salary of an individual based on their years of experience using a Simple Linear Regression model.

#### ⚙️ Project Structure
- **`main.py`**:
  - Loads dataset (`Salary_Data.csv`).
  - Trains and evaluates a linear regression model.
  - Displays model performance (R², MSE) in the console.
  - Saves the trained model to disk using `pickle`.

- **`app.py`**:
  - Streamlit application for real-time salary prediction.
  - User inputs years of experience, receives salary prediction instantly.

#### 📊 Features
- Data visualization and evaluation metrics.
- Interactive UI via Streamlit.
- Real-world salary prediction model based on experience.

#### 🧰 Technologies Used
- Python, NumPy, Pandas, Matplotlib
- scikit-learn
- Streamlit

#### 📝 Sample Output
> ✅ The predicted salary for **12 years** of experience is: **$120,000.00**

---

## 🚀 How to Run the Projects Locally

### 📦 Prerequisites
Install the required Python libraries:
```bash
pip install numpy pandas matplotlib scikit-learn streamlit
```

### ⬇️ Step 1: Clone the Repository

```bash
git clone https://github.com/Raviteja-Beri/Machine_Learning_Projects.git
cd Machine_Learning_Projects
```

### 🏠 Run: House Price Prediction
```bash
cd House_Price_Prediction
python main.py         # Optional: Train the model
streamlit run app.py   # Run the web app
```

### 💼 Run: Salary Prediction
```bash
cd Salary_Prediction
python main.py         # Optional: Train the model
streamlit run app.py   # Run the web app
```

#### 💡 After launching, the Streamlit app will open in your browser. You can input values and get real-time predictions using the trained models.

