import pandas as pd
import matplotlib.pyplot as mlt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ""
df = pd.read_csv("data/students.csv")

# mlt.scatter(df['Study_Hours'], df['Final_Marks'])
# mlt.title('Study Hours vs Final Marks')
# mlt.xlabel('Study Hours')
# mlt.ylabel('Final Marks')
# mlt.show()

correlation = df.corr()

mlt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

mlt.title("Student Performance Correlation Heatmap")
mlt.show()

print("\nCorrelation with Final Marks:")

print(df.corr()["Final_Marks"].drop("Final_Marks").sort_values(ascending=False).round(4))

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments_Completed"
    ]
]

y = df["Final_Marks"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("\nTesting Data Shape:", X_test.shape)
print("\nTraining Target Shape:", y_train.shape)
print("\nTraining Target Shape:", y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)
print(" Model Training Completed!")

y_pred = model.predict(X_test)

print("\nPredicted Marks:")
print(y_pred)

print('\nActual Marks:')
print(y_test.values)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 SCORE:", r2)

new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [85],
    "Previous_Marks": [70],
    "Assignments_Completed": [9]
})

new_prediction = model.predict(new_student)

print("\nNew Student Prediction:")
print("Predicted Final Marks:", round(new_prediction[0], 2))

import joblib

joblib.dump(model, "student_model.pkl")
print("Model saved successfully!")