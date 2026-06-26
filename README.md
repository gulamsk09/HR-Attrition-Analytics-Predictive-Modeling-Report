# HR Employee Attrition Prediction Pipeline

Developed by **Mohd Gulam Sheikh**, this repository contains an end-to-end data science and machine learning pipeline designed to analyze, visualize, and predict employee attrition. 

Because standard evaluation metrics can be highly misleading when dealing with imbalanced HR data, this project focuses heavily on **F1-Score** and **ROC-AUC** to identify actual leavers while minimizing false alarms.

---


##  Key Features

* **Synthetic Data Generation:** Generates a structured dataset of 1,470 employees with diverse categorical and numerical features mimicking realistic HR scenarios.
* **Comprehensive Data Preprocessing:** Includes categorical tracking with One-Hot Encoding, scaling via `StandardScaler`, and dropping non-predictive high-cardinality features.
* **Exploratory Data Analysis (EDA):** Generates multi-panel visualizations comparing attrition against metrics like monthly income, department roles, work-life balance, and tenure.
* **Comparative ML Pipeline:** Evaluates three core machine learning models:
    * Logistic Regression (with class balancing)
    * Random Forest Classifier (with class balancing)
    * Gradient Boosting Classifier
* **Feature Importance & Metrics:** Generates ROC Curves, Confusion Matrices, and a Top 10 Feature Importance breakdown for the optimal model.

---

##  Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

##  Pipeline & Workflow

### 1. Data Prep & Cleaning
* Generates or loads `HR_Attrition_Data.csv`.
* Drops irrelevant identifiers (`EmployeeNumber`, `Over18`, `StandardHours`).
* Maps target feature `Attrition` to binary format (`1` for Yes, `0` for No).

### 2. Model Training & Evaluation
The dataset is stratified and split into an 80/20 train-test ratio. Models are trained and compared across standard classification metrics:

```text
Precision | Recall | F1-Score | ROC-AUC Score

