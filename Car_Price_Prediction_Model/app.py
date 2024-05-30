from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    car = pd.read_csv('car_data.csv')
    
    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)
    
    car = car[car['Price'] != 'Ask For Price']
    car['Price'] = car['Price'].str.replace(',', '').astype(int)
    
    car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)
    
    car = car[~car['fuel_type'].isna()]
    
    car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
    car = car.reset_index(drop=True)
    
    return car

# Train the model
def train_model():
    car = load_and_preprocess_data()
    
    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X[['name', 'company', 'fuel_type']])
    
    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_, handle_unknown='ignore'), ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )
    
    pipe = Pipeline([
        ('preprocessor', column_trans),
        ('regressor', XGBRegressor(random_state=42))
    ])
    
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
        'regressor__reg_alpha': [0, 0.1, 0.5],
        'regressor__reg_lambda': [1, 1.5, 2]
    }
    
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    pickle.dump(best_model, open('BestXGBoostModelWithRegularization.pkl', 'wb'))
    return best_model

# Load the trained model
model = train_model()

# Define HTML templates
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background: #120035;
        }
        .container {
            margin-top: 50px;
            max-width: 600px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background: #008D09;
            border: none;
        }
        .btn-primary:hover {
            background: #00CF0E;
        }
        .form-group label {
            font-weight: bold;
        }
        .btn{
            margin-top:10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mt-4">Car Price Predictor</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="name">Car Name</label>
                <input type="text" class="form-control" id="name" name="name" required>
            </div>
            <div class="form-group">
                <label for="company">Company</label>
                <input type="text" class="form-control" id="company" name="company" required>
            </div>
            <div class="form-group">
                <label for="year">Year</label>
                <input type="number" class="form-control" id="year" name="year" required>
            </div>
            <div class="form-group">
                <label for="kms_driven">Kilometers Driven</label>
                <input type="number" class="form-control" id="kms_driven" name="kms_driven" required>
            </div>
            <div class="form-group">
                <label for="fuel_type">Fuel Type</label>
                <select class="form-control" id="fuel_type" name="fuel_type" required>
                    <option value="Petrol">Petrol</option>
                    <option value="Diesel">Diesel</option>
                    <option value="CNG">CNG</option>
                    <option value="LPG">LPG</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary btn-block">Predict Price</button>
        </form>
    </div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Prediction Result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background: #200069;
        }
        .container {
            margin-top: 50px;
            max-width: 600px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .alert-info {
            font-size: 1.5rem;
        }
        .btn-secondary {
            background: #A90014;
            border: none;
        }
        .btn-secondary:hover {
            background: #B80000;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mt-4">Car Price Predictor</h2>
        <div class="alert alert-info text-center">
            {{ prediction_text }}
        </div>
        <div class="text-center">
            <a href="/" class="btn btn-secondary">Go Back</a>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                  data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5))
        
        prediction = model.predict(input_data)
        
        return render_template_string(result_html, prediction_text=f'Estimated Price: ₹{int(prediction[0])}')

if __name__ == "__main__":
    app.run(debug=True)
