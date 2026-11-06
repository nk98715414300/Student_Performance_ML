import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as mlt

df = pd.read_csv("data/students.csv")

X = df[[
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments_Completed"
]]

y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("student_model.pkl")

predictions = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual_Marks": y_test.values,
    "Predicted_Marks": np.round(predictions, 2),
    "Error": np.round(y_test.values - predictions, 2)
})

print("\n===== ACTUAL VS PREDICTED =====")
print(comparison.to_string(index=False))

comparison.to_csv(
    "model_evaluation_results.csv", index=False
)
print("\nEvaluation results saved succussfully!")

mlt.figure(figsize=(8, 5))
mlt.scatter(y_test, predictions)

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

mlt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)
mlt.xlabel("Actual Marks")
mlt.ylabel("Predicted Marks")
mlt.title("Actual vs Predicted Marks")
mlt.grid(True)
mlt.show()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE REPORT =====")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))