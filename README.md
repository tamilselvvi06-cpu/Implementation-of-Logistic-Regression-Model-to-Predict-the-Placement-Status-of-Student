# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset using the Pandas library.

2.Create a copy of the dataset and remove unnecessary columns like serial number and salary.

3.Check the dataset for missing values and duplicate records.

4.Convert all categorical attributes into numerical form using Label Encoding.

5.Separate the dataset into independent features (X) and target variable (status).

6.Split the dataset into training and testing sets using an 80:20 ratio.

7.Initialize the Logistic Regression model with a suitable solver.

8.Train the model using the training dataset.

9.Predict the placement status using the test dataset.

10.Evaluate the model performance using accuracy score and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: STEFFI J
RegisterNumber:  212224220107
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())
print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]
le = LabelEncoder()
for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])
print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]
print("\nFeatures (X) sample:")
print(X.head())
print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("\nPredicted values (y_pred):")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
new_prediction = lr.predict(new_student)
print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

```

## Output:

<img width="1096" height="842" alt="image" src="https://github.com/user-attachments/assets/f7db83dc-b7b3-4adb-a761-85a6c377c6fa" />

<img width="1322" height="847" alt="image" src="https://github.com/user-attachments/assets/1830e7ed-82ca-478e-b2fe-85c78abaf8d5" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
