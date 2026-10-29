# 🚢 Titanic Survival Prediction Web App

An interactive Machine Learning web application that predicts passenger survival probability on the RMS Titanic based on demographic, ticket, and family attributes. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 📌 Features

* **Real-time Predictions:** Calculates survival probability and displays model confidence percentage dynamically.
* **Domain Feature Engineering:** Uses key predictors including title extractions, family dynamics (`FamilySize`, `IsAlone`), cabin indicators, and ticket-group adjusted fare (`FarePerPerson`).
* **Regularized Random Forest Model:** Trained and validated using 5-Fold Cross-Validation on Kaggle's Titanic dataset to prevent overfitting.
* **Interactive UI:** Clean user interface with sliders, dropdowns, and numeric inputs built using Streamlit.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn, Joblib
* **Data Processing:** Pandas, NumPy
* **Frontend / Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud / GitHub

---

## 📂 Project Structure

```text
titanic-survival-streamlit/
│
├── app.py                 # Streamlit web application script
├── titanic_model.pkl      # Pre-trained Random Forest model
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation
