# 🛥️ Ferry Ticket Demand Forecasting System

A **Streamlit-based predictive analytics web application** that forecasts short-term ferry ticket demand using historical time-series data and machine learning models.

---

## 📌 Project Overview

This project aims to build a **Predictive Decision Support System** for ferry operations.
It helps estimate future ticket demand at short intervals (15 minutes to 2 hours), enabling better planning, crowd management, and operational efficiency.

---

## 🚀 Key Features

* 📈 **Exploratory Data Analysis (EDA)**

  * Daily trends
  * Hourly patterns
  * Weekly & monthly seasonality
  * Distribution analysis

* 🔮 **Demand Forecasting**

  * Predicts ticket demand for:

    * 15 minutes
    * 30 minutes
    * 1 hour
    * 2 hours

* 🤖 **Machine Learning Models**

  * Moving Average (Baseline)
  * Linear Regression
  * Random Forest
  * Gradient Boosting
  * ARIMA (Time Series)

* 📊 **Model Evaluation**

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * MAPE (Mean Absolute Percentage Error)

* 📉 **Feature Engineering**

  * Lag features (1, 2, 4, 8 intervals)
  * Rolling mean & standard deviation
  * Time-based features (hour, weekday, month)

* 📊 **Model Comparison Dashboard**

  * Compare performance across different horizons

* 📋 **Interactive Data Table**

  * Filter and download dataset

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * pandas, numpy
  * matplotlib
  * scikit-learn
  * statsmodels
* **Framework:** Streamlit

---

## 📂 Project Structure

```id="e6k2fd"
ferry-demand-forecasting/
│── ferry_forecast_app.py        # Main Streamlit application
│── Toronto_Island_Ferry_Tickets.csv   # Dataset
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash id="o8z6z1"
git clone https://github.com/your-username/ferry-demand-forecasting.git
cd ferry-demand-forecasting
```

### 2️⃣ Install dependencies

```bash id="d1n0c4"
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash id="q6ks2c"
streamlit run ferry_forecast_app.py
```

---

## 📊 Dataset

* Source: Toronto Parks, Forestry & Recreation
* Time Range: **2015 – 2025**
* Frequency: **15-minute intervals**
* Key Columns:

  * Timestamp
  * Sales Count
  * Redemption Count

---

## 🎯 Business Use Case

This system can help:

* 🚢 Optimize ferry scheduling
* 👥 Manage passenger crowd effectively
* 📉 Reduce waiting time and congestion
* 📊 Support data-driven decision making

---

## 📸 Application Preview

*(You can add screenshots here after uploading images to GitHub)*

---

## 🌐 Future Improvements

* Deploy using Streamlit Cloud / AWS
* Add real-time data integration
* Use advanced models (LSTM, Prophet)
* Improve hyperparameter tuning
* Add alert system for peak demand

---

## 👨‍💻 Author

**Sudip Senapati**
Aspiring Data Analyst

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
