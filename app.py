from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
data = pd.read_csv('cleaned_car_prices.csv')

# Get unique values for dropdowns
names = sorted(data['name'].unique())
years = sorted(data['year'].unique(), reverse=True)
fuel_types = sorted(data['fuel'].unique())
mileages = sorted([str(m) for m in data['mileage(km/ltr/kg)'].unique()])
engines = sorted([str(e) for e in data['engine'].unique()])
powers = sorted([str(p) for p in data['max_power'].unique()])
seats = sorted(data['seats'].dropna().astype(int).unique())

@app.route('/')
def index():
    return render_template('index.html', names=names, years=years, fuel_types=fuel_types, mileages=mileages, engines=engines, powers=powers, seats=seats)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    name = request.form['name']
    year = int(request.form['year'])
    km_driven = int(request.form['km_driven'])
    fuel = request.form['fuel']
    mileage = float(request.form['mileage'])
    engine = float(request.form['engine'].split()[0])  # In case 'cc' is present in data
    max_power = float(request.form['max_power'].split()[0])  # In case 'bhp' is present in data
    seats = int(request.form['seats'])

    # Create a dataframe with the input values
    input_data = pd.DataFrame([[name, year, km_driven, fuel, mileage, engine, max_power, seats]],
                              columns=['name', 'year', 'km_driven', 'fuel', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return jsonify({'prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)