import pandas as pd
import os
import matplotlib.pyplot as mlt

file_name = "prediction_history.csv"

if os.path.exists(file_name):
    history = pd.read_csv(file_name)

    if history.empty:
        print("No prediction records found.")

    else:
        print("\n===== STUDENT PREDICTION HISTORY =====")
        print(history)
        print("\n===== SUMMARY =====")

        mlt.figure(figsize=(8, 5))
        mlt.plot(range(1, len(history) + 1), history["Predicted_Marks"], marker="o", color="green")
        mlt.xlabel("Student Number")
        mlt.ylabel("Predicted Marks")
        mlt.title("Student Performance Prediction History")
        mlt.grid(True)
        mlt.show()

        print("Total Predictions:", len(history))
        print("Average Predicted Marks", round(history["Predicted_Marks"].mean(), 2))
        print("Highest Predicted Marks:", history["Predicted_Marks"].max())
        print("Lowest Predicted Marks:", history["Predicted_Marks"].min())

else:
    print("Prediction history file not found!")