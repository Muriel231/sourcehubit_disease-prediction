# sourcehubit_disease-prediction

## Internship Task 4 Report Title:
Disease Prediction from Medical Data Intern Name: Chrizel Muriel Cardoza Domain: Machine Learning Platform: Jupyter Notebook Tools Used: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Objective:
The aim of this project is to build a machine learning model that predicts whether a patient has a disease based on medical data. The dataset used includes features like blood pressure, glucose level, BMI, and other attributes.

## Technologies Used:
Python – Programming language

Pandas – Data loading and processing

Matplotlib & Seaborn – Data visualization

Scikit-learn – Model training and evaluation

Jupyter Notebook – Development environment

## Dataset Description:
The dataset used is the PIMA Indian Diabetes Dataset, which contains the following features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0: Non-diabetic, 1: Diabetic)

Steps Performed Data Loading

Loaded CSV dataset using Pandas.

Data Exploration

Checked shape, null values, and descriptive statistics.

Visualized class distribution using count plots.

Preprocessing

Separated features (X) and labels (y).

Split the data into training and testing sets (80% train, 20% test).

Model Building

Used RandomForestClassifier from scikit-learn.

Trained the model on training data.

Model Evaluation

Used accuracy score, confusion matrix, and classification report to evaluate the model.

Results Accuracy: 88.04%

Confusion Matrix: [[66 11] [11 96]]

## Classification Report:

          precision    recall  f1-score   support
 0           0.86       0.86      0.86         77
 1           0.90       0.90      0.90        107
 
## Conclusion:
The machine learning model successfully predicts the presence of diabetes with an accuracy of around 88%. This shows that Random Forest is a good classifier for this type of medical data. The model can be improved further by hyperparameter tuning or using ensemble methods.
