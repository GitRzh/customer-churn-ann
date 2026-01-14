# 📊 Customer Churn Prediction using Artificial Neural Network (ANN)

Predict whether a bank customer is likely to **leave (churn)** or **stay**, using an **Artificial Neural Network (ANN)**.  
This project covers the **complete ML lifecycle** — training, evaluation, and deployment with **Streamlit**.

---

## 📖 Project Overview

Customer churn is a major challenge for banks and financial institutions.  
This project uses customer demographic and financial data to predict churn and help businesses take **proactive retention decisions**.

### Key Highlights
- End-to-end Machine Learning project
- ANN model built using TensorFlow / Keras
- Interactive web app using Streamlit
- Clean, production-ready project structure

---

## 🧠 Problem Statement

Build a machine learning model that predicts customer churn based on:
- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Active Membership
- Estimated Salary

**Target Variable**
- `1` → Customer Exited  
- `0` → Customer Stayed  

---

## 🏗️ Project Structure

```text
Customer-Churn-ANN/
│
├── data/
│   └── churn_dataset.csv
│
├── model/
│   └── ann_model.h5
│
├── notebooks/
│   └── ann_training.ipynb #not used
│
├── app.py
├── requirements.txt
└── README.md
```
---

## ⚙️ Part 1: ANN Model Training
- Data preprocessing (encoding, scaling, feature selection)
- Splitting data into training and testing sets
- Building an ANN with:
  - Input layer
  - Hidden layers
  - Output layer (Binary Classification)
- Model evaluation using accuracy and loss
- Saving the trained model for deployment

## 🔮 Part 2: Customer Churn Prediction
- Takes customer input data
- Applies the same preprocessing used during training
- Predicts whether the customer will:
  - **Stay Loyal**
  - **Exit (Churn)**

## 🚀 Part 3: Model Deployment using Streamlit
- Interactive web interface for real-time prediction
- Users can input customer details using sliders and dropdowns
- Displays churn prediction instantly
- Makes the ML model accessible without technical knowledge

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
- **Model Type:** Artificial Neural Network (ANN)
- **Web Framework:** Streamlit
- **IDE & Tools:** VS Code
- **Version Control:** Git & GitHub

---

## 📊 Dataset
- Publicly available **Bank Customer Churn Dataset**
- Contains customer demographic and financial information
- Binary target variable:
  - `1` → Customer Exited  
  - `0` → Customer Stayed

---

## ▶️ How to Run the Project
- Step 1: Clone the Repository
```bash
git clone https://github.com/GitRzh/imdb-sentiment-analysis-rnn.git
cd imdb-sentiment-analysis-rnn
```
- Step 2: Create Virtual Enviroment
```bash
python -m venv venv
```
```bash
source venv/bin/activate        #linux/mac
```
```bash
venv\Scripts\activate           #windows
```
- Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
- Step 4: Run the Application Locally
```bash
streamlit run app.py
```

---

## 📌 Future Enhancements

- Hyperparameter tuning for better accuracy

- Add model performance metrics in UI

- Deploy the app on cloud platforms (Heroku / AWS / Streamlit Cloud)

- Compare ANN with other ML models

---

## 👤 Author

**Raz**

Python | AI & ML Enthusiast

---

## ⭐ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.

Connect with Me!

**GitHub:** https://github.com/GitRzh

**E-mail:** GitRzh@users.noreply.github.com
