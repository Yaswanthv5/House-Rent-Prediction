from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='./templates')

# Load the trained model
model_path = "./best_house_price_model.pkl"
model = joblib.load(model_path)

# Define a function to make predictions
def predict_house_price(input_data):
    """
    Predicts house price using trained XGBoost model.
    """
    input_array = np.array(input_data).reshape(1, -1)  # Ensure input is reshaped correctly
    predicted_price = model.predict(input_array)[0]
    return predicted_price  # Convert log-transformed price back to normal

# API Endpoint for predictions (For API requests)
@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()
    
    try:
        # Extract basic features
        features = [
            data['MedInc'], data['HouseAge'], data['AveRooms'], 
            data['AveBedrms'], data['Latitude'], data['Longitude']
        ]

        # Compute extra features
        BedrmsPerRoom = data['AveBedrms'] / data['AveRooms']
        RoomsPerHouse = data['AveRooms'] / data['AveOccup']
        PopPerHouse = data['Population'] / data['AveOccup']

        # Append computed features
        features.extend([BedrmsPerRoom, RoomsPerHouse, PopPerHouse])

        # Make prediction
        price = predict_house_price(features)
        return jsonify({'Predicted Price': round(price, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# UI for Predictions (For Frontend)
@app.route('/', methods=['GET', 'POST'])
def index():
    price = None
    if request.method == 'POST':
        try:
            # Print form data for debugging
            # print("Received Form Data:", request.form)

            # Get user inputs
            MedInc = float(request.form['MedInc'])
            HouseAge = float(request.form['HouseAge'])
            AveRooms = float(request.form['AveRooms'])
            AveBedrms = float(request.form['AveBedrms'])
            Latitude = float(request.form['Latitude'])
            Longitude = float(request.form['Longitude'])
            AveOccup = float(request.form['AveOccup'])  # Needed for extra features
            Population = float(request.form['Population'])

            # Compute additional features
            BedrmsPerRoom = AveBedrms / AveRooms
            RoomsPerHouse = AveRooms / AveOccup
            PopPerHouse = Population / AveOccup

            # Create input feature array
            input_features = [MedInc, HouseAge, AveRooms, AveBedrms, Latitude, Longitude, BedrmsPerRoom, RoomsPerHouse, PopPerHouse]

            # Make prediction
            price = predict_house_price(input_features)
            price = round(price, 2)

            # print("Final Prediction:", price)  # Debugging output

        except Exception as e:
            # print("Error:", str(e))
            price = "Invalid input. Please enter numeric values."
    
    return render_template('index.html', price=price)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
