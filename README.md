# 🛥️ Ferry Ticket Demand Forecasting System

A **Streamlit-based predictive analytics web application** that forecasts short-term ferry ticket demand using historical time-series data and machine learning models.

---

## 🔗 Project Links

* 📁 **GitHub Repository:**
  https://github.com/sudip758582/ferry-demand-forecasting

* 💻 **Local Dashboard (runs on your system only):**
  http://localhost:8503/

---

## 📌 Project Overview

This project is a **Predictive Decision Support System** designed to forecast ferry ticket demand at short intervals (15 minutes to 2 hours).
It helps improve operational planning, reduce congestion, and support data-driven decisions.

---

## 🚀 Key Features

* 📈 Exploratory Data Analysis (EDA)
* 🔮 Short-term demand forecasting (15 min → 2 hours)
* 🤖 Multiple ML models:

  * Moving Average
  * Linear Regression
  * Random Forest
  * Gradient Boosting
  * ARIMA
* 📊 Model comparison across horizons
* 📉 Feature engineering (lags, rolling stats, time features)
* 📋 Interactive dashboard with filtering & download

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* Statsmodels
* Streamlit

---

## 📂 Project Structure

```
ferry-demand-forecasting/
│── ferry_forecast_app.py
│── Toronto_Island_Ferry_Tickets.csv
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sudip758582/ferry-demand-forecasting.git
cd ferry-demand-forecasting
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run ferry_forecast_app.py
```

Then open in browser:

```
http://localhost:8503/
```

---

## 📊 Dataset

* Source: Toronto Parks, Forestry & Recreation
* Time Range: 2015 – 2025
* Frequency: 15-minute intervals

---

## 🎯 Business Use Case

* 🚢 Optimize ferry scheduling
* 👥 Manage passenger demand
* 📉 Reduce congestion
* 📊 Enable data-driven planning

---

## 🌐 Future Improvements

* Deploy on Streamlit Cloud (public link)
* Add real-time data pipeline
* Advanced models (LSTM / Prophet)
* Hyperparameter tuning

---

## 👨‍💻 Author

**Sudip Senapati**
Aspiring Data Analyst

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
