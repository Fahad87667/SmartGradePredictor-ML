from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('students_marks_prediction_best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text='Predicted Marks: 0')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        number_courses = float(request.form['number_courses'])
        time_study = float(request.form['time_study'])

        # Reshape the input features into a 2D array
        feature_value = np.array([[number_courses, time_study]])

        # Convert the feature array to a 2D array for prediction
        output = model.predict(feature_value.reshape(1, -1))[0].round(2)

        return render_template('index.html', prediction_text=f'Predicted Marks: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text='Error occurred. Please check your input.')


if __name__ == '__main__':
    app.run(debug=True)
