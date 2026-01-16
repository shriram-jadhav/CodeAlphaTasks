from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/car_price_model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form.get('year'))
    present_price = float(request.form.get('present_price'))
    kms_driven = int(request.form.get('kms_driven'))
    owner = int(request.form.get('owner'))

    fuel_type = request.form.get('fuel_type')
    seller_type = request.form.get('seller_type')
    transmission = request.form.get('transmission')

    # Fuel encoding
    fuel_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_petrol = 1 if fuel_type == 'Petrol' else 0

    # Seller encoding
    seller_individual = 1 if seller_type == 'Individual' else 0

    # Transmission encoding
    transmission_manual = 1 if transmission == 'Manual' else 0

    # Final input (MUST MATCH TRAINING ORDER)
    input_data = np.array([[ 
        present_price,
        kms_driven,
        owner,
        year,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Car Price: ₹ {round(prediction, 2)} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)
