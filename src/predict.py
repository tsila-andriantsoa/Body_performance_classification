from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask('Body_performance_classification')

# Load the model and utilities 
loaded_pipeline = joblib.load('model/pipeline_baseline.pkl')



# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Create dataframe based on JSON object
        df = pd.DataFrame(input_data, index = [0])       

        # rename data columns
        new_col_names = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_%', 'diastolic',
           'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit_ups_counts', 'broad_jump_cm', ]
        
        df.columns = new_col_names

        # Make a prediction
        prediction = loaded_pipeline.predict(df)[0]
        result = {
            'prediction': str(prediction),
        }
        
        # Return the prediction as JSON
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define a home check endpoint
@app.route('/home', methods=['GET'])
def home():
    return jsonify({'status': 'ok'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)