# 🚗 Car Price Prediction Dashboard

This project is developed to predict automobile market prices by utilizing various machine learning algorithms. It provides a comprehensive analysis of how different features affect vehicle valuation.

## 🛠️ Tech Stack & Architecture
- **Python & Pandas:** For data manipulation and feature analysis.
- **Scikit-learn Pipeline:** An end-to-end professional architecture that combines preprocessing and modeling into a single object.
- **ColumnTransformer:** Used for automated data transformation; `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.
- **Algorithms:** Linear Regression, Random Forest, Decision Tree, and XGBoost.

## 🚀 Key Features
- **Pipeline Integration:** A robust architecture designed to prevent **Data Leakage** and ensure the model is production-ready.
- **Multi-Model Prediction:** Enables simultaneous price predictions from all trained models for a single input, allowing for cross-model comparison.
- **Dynamic User Input:** An interactive application that accepts real-time data from the user to generate instant price estimates.

## 📈 Performance Metrics
The models were evaluated based on **MAE, MSE, RMSE, and R2** scores. During the evaluation phase, the **Linear Regression** model demonstrated the most stable performance for this specific dataset.

## 💻 Installation & Usage
1. Clone this repository.
2. Install the required dependencies: `pip install pandas scikit-learn xgboost`
3. Run the application: `python car_prediction.py`
