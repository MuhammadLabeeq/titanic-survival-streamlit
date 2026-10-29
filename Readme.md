# 🚢 Titanic Survival Prediction Web App

An interactive Machine Learning web application that predicts passenger survival probability on the RMS Titanic based on demographic, ticket, and family attributes. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌐 Live Demo

👉 **[Launch Interactive Web App](https://titanic-survival-app-d22fyd9byexejjq37k9drq.streamlit.app/)**

---

## 📊 Model Performance & Metrics

The prediction model is a regularized **Random Forest Classifier** trained and cross-validated on the [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data):

* **5-Fold Cross-Validation Accuracy:** **82.72%** (± 1.45%)
* **Kaggle Public Leaderboard Score:** **0.77990** (~78.0% Accuracy)
* **Generalization Strategy:** Hyperparameters are constrained (`max_depth=4`, `min_samples_leaf=2`) combined with group-adjusted feature engineering (`FarePerPerson`, title extraction) to reduce overfitting on unseen test data.

---

## 📌 Key Features

* **Real-time Predictions:** Calculates survival status along with probability confidence percentages dynamically.
* **Domain Feature Engineering:** Integrates domain-specific variables:
  * **Title Extraction:** Maps social status derived from passenger names (*Mr*, *Miss*, *Mrs*, *Master*, *Rare*).
  * **Group-Adjusted Fare (`FarePerPerson`):** Calculates individual ticket cost by dividing total ticket fare by ticket group size.
  * **Family Dynamics:** Computes `FamilySize`, `IsAlone`, and `IsChild` indicators.
  * **Cabin Status (`HasCabin`):** Tracks cabin allocation.
* **Interactive Interface:** Clean Streamlit UI equipped with input sliders, dropdowns, and numeric fields.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn, Joblib
* **Data Processing:** Pandas, NumPy
* **Frontend Framework:** Streamlit
* **Deployment & Hosting:** Streamlit Community Cloud, GitHub

---

## 📂 Project Structure

```text
titanic-survival-streamlit/
│
├── app.py                 # Streamlit web application script
├── titanic_model.pkl      # Pre-trained Random Forest model
├── requirements.txt       # Python environment dependencies
└── Readme.md              # Project documentation
