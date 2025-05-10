from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

MODEL_PATHS = {
    'north': 'models/ndvi_predictor_north.h5',
    'south': 'models/ndvi_predictor_south.h5',
    'east': 'models/ndvi_predictor_east.h5',
    'west': 'models/ndvi_predictor_west.h5'
}

# Check all models exist
for region, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model for {region} not found at {path}")

# Load models
models = {}
for region, path in MODEL_PATHS.items():
    print(f"Loading model for {region} from {path}")
    models[region] = load_model(path, compile=False)
    print(f"Loaded {region} region model: LSTM neural network")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get region from form
        region = request.form['region'].lower()
        
        if region not in models:
            return jsonify({'error': f'Invalid region: {region}'})
        
        # Get NDVI values from form
        ndvi_values = [
            float(request.form['ndvi1']),
            float(request.form['ndvi2']),
            float(request.form['ndvi3']),
            float(request.form['ndvi4']),
            float(request.form['ndvi5'])
        ]
        
        # Validate NDVI values
        for val in ndvi_values:
            if val < -1 or val > 1:
                return jsonify({'error': 'NDVI values must be between -1 and 1'})
                
        # Make prediction using the loaded model
        predictions = model_prediction(ndvi_values, region)
        
        # Format prediction results
        results = {
            'year1': round(float(predictions[0]), 4),
            'year2': round(float(predictions[1]), 4),
            'year3': round(float(predictions[2]), 4)
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

def model_prediction(ndvi_values, region):
    """
    Make a prediction using the pre-built ML model for the selected region
    """
    # Get the model for this region
    model = models[region]
    
    # Reshape input for prediction (LSTM expects 3D shape: [samples, timesteps, features])
    X = np.array(ndvi_values).reshape(1, 5, 1)
    
    # Use TensorFlow eager execution to get predictions
    predictions = model.predict(X)
    
    # Convert to NumPy array if it's a tensor (safe handling)
    predictions = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
    
    # Flatten and clip predictions to valid NDVI range
    predictions = np.clip(predictions.flatten(), -1.0, 1.0)
    
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
