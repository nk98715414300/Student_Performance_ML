Student Performance Prediction using Machine Learning

Project Overview

This project uses Machine Learning to predict student performance based on academic and study-related factors.

A Linear Regression model is trained using student data to predict their performance.

Features

- Data loading and preprocessing using Pandas
- Exploratory Data Analysis (EDA)
- Data visualization using Matplotlib and Seaborn
- Linear Regression model training
- Model evaluation using ML metrics
- Student performance prediction through user input
- Prediction history management
- Actual vs Predicted performance visualization

Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Project Structure

Student_Performance_ML/
│
├── data/
│   └── students.csv
├── main.py
├── predict.py
├── history.py
├── evaluate.py
├── requirements.txt
├── student_model.pkl
├── model_evaluation_results.csv
├── README.md
└── .gitignore

Installation

1. Clone the repository:

git clone https://github.com/nk98715414300/Student_Performance_ML.git

2. Open the project folder:

cd Student_Performance_ML

3. Create a virtual environment:

python -m venv .venv

4. Activate it on Windows:

.venv\Scripts\activate

5. Install dependencies:

pip install -r requirements.txt

How to Run

Train the model and perform data analysis:

python main.py

Make predictions:

python predict.py

View prediction history:

python history.py

Evaluate the model:

python evaluate.py

Machine Learning Algorithm

Linear Regression

The model learns the relationship between input features and student performance to make predictions.

Project Learning Outcomes

- Understanding the Machine Learning workflow
- Data preprocessing and visualization
- Training and testing ML models
- Model evaluation
- Saving and loading trained models
- Making predictions using a trained model

Disclaimer

This project is for educational purposes. Predictions depend on the dataset and model performance and should not be treated as definitive academic assessments.

Author

Created as a Python and Machine Learning learning project.

GitHub: https://github.com/nk98715414300