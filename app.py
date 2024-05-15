from flask import Flask, render_template, request
import joblib
import datetime

app = Flask(__name__)

# Load the trained XGBoost regression model
model = joblib.load('car_price_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        car_name = request.form['car_name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kilometers_driven = int(request.form['kilometers_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        # Calculate car age
        current_year = datetime.datetime.now().year
        age = current_year - year

        # Map categorical variables to numeric
        fuel_type_encoded = 0 if fuel_type == 'Petrol' else (1 if fuel_type == 'Diesel' else 2)
        seller_type_encoded = 0 if seller_type == 'Dealer' else 1
        transmission_encoded = 0 if transmission == 'Manual' else 1

        # Predict price using trained model
        predicted_price = model.predict([[age, present_price, kilometers_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner]])[0]

        return render_template('result.html', predicted_price=predicted_price)
    else:
        # Handle GET request
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
