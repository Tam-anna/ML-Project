from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Load the pre-trained scaler
scaler = joblib.load('scaler.pkl')

# Route to display the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_data = []
        fields = [
            'customer','age', 'income', 'age_youngest_child', 'debt_equity', 'gender',
            'bad_payment', 'gold_card', 'pension_plan', 'household_debt_to_equity_ratio', 
            'members_in_household', 'months_current_account', 'months_customer', 'call_center_contacts',
            'loan_accounts', 'number_products', 'number_transactions', 'non_worker_percentage',
            'white_collar_percentage', 'mortgage', 'pension', 'savings'
        ]
        
        # Loop through the fields to collect data
        for field in fields:
            value = request.form.get(field)
            if value:
                input_data.append(float(value))
            else:
                return f"Missing value for {field}."

        # Scale the features before making prediction
        input_data_scaled = scaler.transform([input_data])

        # Make prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Return the result to the user
        return render_template('index.html', prediction_text=f'Predicted RFM Score: {prediction[0]:.2f}')

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
