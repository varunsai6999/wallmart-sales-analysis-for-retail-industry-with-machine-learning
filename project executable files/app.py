import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import datetime as dt
import calendar

app = Flask(__name__)

# Load the trained Random Forest model
loaded_model = pickle.load(open('rf_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch input data from the form
        store = request.form.get('store')
        dept = request.form.get('dept')
        isHoliday = request.form.get('isHoliday')
        size = request.form.get('size')
        temp = request.form.get('temp')
        unemployment = request.form.get('unemployment')
        is_weekend = request.form.get('is_weekend')
        cpi = request.form.get('cpi')
        type_b = request.form.get('type_b')
        type_c = request.form.get('type_c')
        month = request.form.get('month')
        year = request.form.get('year')

        # Debugging print statements
        print(f"Store: {store}, Dept: {dept}, IsHoliday: {isHoliday}, Size: {size}, Temp: {temp}, Unemployment: {unemployment}, IsWeekend: {is_weekend}, CPI: {cpi}, Type_B: {type_b}, Type_C: {type_c}, Month: {month}, Year: {year}")

        # Ensure all form fields are provided
        if not all([store, dept, isHoliday, size, temp, unemployment, is_weekend, cpi, type_b, type_c, month, year]):
            raise ValueError("All form fields are required.")

        month_name = calendar.month_name[int(month)]

        # Create input DataFrame for prediction with consistent feature names and order
        X_test = pd.DataFrame({
            'Store': [int(store)],
            'Dept': [int(dept)],
            'Month': [int(month)],
            'Year': [int(year)],
            'IsHoliday': [int(isHoliday)],
            'Type_B': [int(type_b)],
            'Type_C': [int(type_c)],
            'Temperature': [float(temp)],
            'CPI': [float(cpi)],
            'Size': [int(size)],
            'Unemployment': [float(unemployment)],
            'is_weekend': [int(is_weekend)]
        })

        print(f"Input DataFrame: \n{X_test}")

        # Perform prediction using the loaded model
        y_pred = loaded_model.predict(X_test)
        output = round(y_pred[0], 2)

        # Debugging print statement
        print(f"Predicted output: {output}")

        # Render the result template with prediction result
        return render_template('result.html', output=output, store=store, dept=dept, month_name=month_name, year=year)
    except Exception as e:
        error_message = f"Error occurred: {e}"
        print(error_message)
        return render_template('index.html', error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
