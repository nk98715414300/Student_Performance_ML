import os
import joblib
import pandas as pd

model = joblib.load("student_model.pkl")
prediction_history = []

while True:

    try:

        study_hours = float(input("Enter Study Hours (0-24): "))
        attendance = float(input("Enter Attendance (0-100): "))
        previous_marks = float(input("Enter Previous Marks (0-100): "))
        assignments = int(input("Enter Completed Assignments (0-10): "))

        if not 0 <= study_hours <= 24:
            raise ValueError("Study Hours must be between 0 and 24")
        if not 0 <= attendance <= 100:
            raise ValueError("Attendance must be between 0 and 100")
        if not 0 <= previous_marks <= 100:
            raise ValueError("Previous marks must be between 0 and 100")
        if not 0 <= assignments <= 10:
            raise ValueError("Assignments must be between 0 and 10")
    
        student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks],
            "Assignments_Completed": [assignments]
        })

        prediction = model.predict(student)
        print("Predicted Final Marks:", round(prediction[0], 2))

        prediction_history.append({
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": previous_marks,
            "Assignments_Completed": assignments,
            "Predicted_Marks": round(prediction[0], 2)          
        })

    except ValueError as e:
        print("\nInvalid Input:", e)

    choice = input("\nPredict another student? (yes/no): ").lower()

    if choice != "yes":
        print("Program ended.")
        break

if prediction_history:
    history_df = pd.DataFrame(prediction_history)

    file_exists = os.path.exists("prediction_history.csv")

    history_df.to_csv(
        "prediction_history.csv",
        mode="a",
        header=not file_exists,
        index=False
    )
    print("\nPrediction history saved successfully!")