# ❤️ Predicting Heart Disease with Machine Learning

This project applies supervised machine learning techniques to predict the risk of cardiovascular disease using clinical data. It includes a full pipeline from data loading and exploratory analysis to model training, evaluation, and feature importance interpretation.

---

## 📁 Project Structure

### 1. 📊 Exploratory Data Analysis (EDA)
- **Dataset**: `cardio_train.csv`
- **EDA Tools**: `pandas`, `seaborn`, and interactive `plotly` charts.
- **Visualizations**:
  - Boxplots for numerical features such as age, weight, systolic and diastolic blood pressure.
  - Bar charts for categorical variables like gender, cholesterol, glucose, smoking, alcohol consumption, and physical activity.
  - Class distribution for the target variable (`cardio`).

### 2. 🤖 Machine Learning Pipeline
- **Target Variable**: `cardio` (binary classification)
- **Features**: All columns except the target.
- **Train-Test Split**: 67% training, 33% testing.
- **Model Used**: `RandomForestClassifier` from `scikit-learn`.
  - Parameters: `n_estimators=20`, `max_depth=4`, `n_jobs=4`

### 3. 📈 Model Evaluation
- **Metrics**:
  - `confusion_matrix`
  - `classification_report` (precision, recall, F1-score)
- **Interpretability**:
  - `permutation_importance` from `sklearn.inspection` to assess the impact of each feature.
  - `SHAP` (`shap.TreeExplainer`) for global interpretability using SHAP values.

---

## 📚 Libraries Used
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `plotly`
- `scikit-learn`
- `shap`

