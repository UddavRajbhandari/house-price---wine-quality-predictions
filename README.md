# 🏠🍷 House Price & Wine Quality Predictions

This repository contains two machine learning projects:
- **House Price Prediction** (Regression Task)
- **Wine Quality Prediction** (Classification Task)

Each project covers data preprocessing, model building, evaluation, and model serialization for production use.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Datasets](#datasets)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🧠 Overview

This project is focused on:
- Predicting **house prices** based on features like area, bedrooms, stories, etc.
- Classifying **wine quality** based on physicochemical tests like acidity, alcohol content, pH, and sulphates.

Both problems are tackled using Machine Learning algorithms, and the models are saved for easy deployment.

---


## Installation

git clone https://github.com/UddavRajbhandari/house-price---wine-quality-predictions.git
cd house-price---wine-quality-predictions

    
# 🚀 How to Use

### Train models:
- Open and run **`house_price.ipynb`** for house price prediction.
- Open and run **`wine_quality.ipynb`** for wine quality classification.

### Predict using saved models:
- Use **`house_price_pred.ipynb`** or **`wine_quality_pred.ipynb`** to make predictions using `.pkl` files.

### Evaluate models:
- Run **`evaluation.py`** to access custom evaluation functions.
- Use **`test_model.py`** or **`test_model_wine.py`** to test performance of serialized models.

---

# 🤖 Models Used

### House Price Prediction:
- Linear Regression
- Ridge Regression

### Wine Quality Classification:
- Gaussian Naive Bayes
- Random Forest Classifier
- Gradient Boosting Classifier
- Stacking Classifier

> Feature scaling is performed using **StandardScaler**.

---

# 📈 Evaluation Metrics

### Regression (House Price):
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification (Wine Quality):
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 📊 Datasets

- **House Price Dataset**: A custom dataset (`house.csv`) with features like area, number of bedrooms, bathrooms, etc.
- **Wine Quality Dataset**: Based on the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

---

# 🔮 Future Work

- Add hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Develop a real-time prediction web app using Streamlit or Flask.
- Perform deeper feature engineering to improve model performance.
- Apply cross-validation techniques for model robustness.

---

# 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

# 🙏 Acknowledgments

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
