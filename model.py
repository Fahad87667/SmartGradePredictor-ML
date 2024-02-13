import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv(r'C:\Users\HP\Desktop\Satudents Marks Prediction\Data\Student_Marks.csv')
df = df.isnull().sum()
# Assuming 'df' is your DataFrame containing the dataset
df.mean()
df=df.fillna(df.mean())
df=df.isnull().sum()
df=df.copy()

# Independent features (X)
x = df[['number_courses', 'time_study']]

# Dependent variable (y)
y = df[['Marks']]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=30)

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

# Training Multiple Models
models = {
    'LinearRegresssion': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet()
}

trained_model_list = []
model_list = []
r2_list = []

# Initialize variables to track the best-performing model
best_model = None
best_r2_score = -float('inf')

for model_name, model in models.items():
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)
    mae, rmse, r2_square = evaluate_model(y_test, y_pred)
    print(model_name)
    model_list.append(model_name)
    r2_list.append(r2_square)

    # Save the best-performing model
    if r2_square > best_r2_score:
        best_model = model
        best_r2_score = r2_square

# Save the best model using joblib
joblib.dump(best_model, 'students_marks_prediction_best_model.pkl')

# Load the best model
loaded_model = joblib.load('students_marks_prediction_best_model.pkl')
