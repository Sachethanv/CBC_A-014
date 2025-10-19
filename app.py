"""
NDVI Predictor Flask Application
=================================
This application provides a web interface for predicting future NDVI (Normalized Difference Vegetation Index) 
values using pre-trained LSTM neural network models for different geographical regions (North, South, East, West).

NDVI is a vegetation health indicator that ranges from -1 to 1, where higher values indicate healthier vegetation.
The application takes 5 consecutive historical NDVI values and predicts the next 3 future values.

Author: CBC_A-014
Date: 2024
"""

# Import required libraries
from flask import Flask, render_template, request, jsonify  # Flask web framework and utilities
import numpy as np  # NumPy for numerical operations and array handling
import tensorflow  # TensorFlow for deep learning operations
from tensorflow.keras.models import load_model  # Keras model loading functionality
import os  # Operating system interface for file operations

# Initialize Flask application
app = Flask(__name__)

# Dictionary mapping regions to their corresponding pre-trained model file paths
# Each region has a specialized LSTM model trained on regional NDVI data
MODEL_PATHS = {
    'north': 'models/ndvi_predictor_north.h5',
    'south': 'models/ndvi_predictor_south.h5',
    'east': 'models/ndvi_predictor_east.h5',
    'west': 'models/ndvi_predictor_west.h5'
}

# Validation: Check that all model files exist before attempting to load them
# This prevents runtime errors if models are missing from the models directory
for region, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model for {region} not found at {path}")

# Initialize empty dictionary to store loaded models
models = {}

# Load all LSTM models into memory at application startup
# Loading at startup ensures faster prediction response times during runtime
for region, path in MODEL_PATHS.items():
    print(f"Loading model for {region} from {path}")
    # compile=False: Skip model compilation since we only need prediction functionality
    models[region] = load_model(path, compile=False)
    print(f"Loaded {region} region model: LSTM neural network")


@app.route('/')
def index():
    """
    Route handler for the home page.
    
    Returns:
        Rendered HTML template for the main application interface
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for NDVI prediction.
    
    Accepts POST requests with form data containing:
    - region: Geographical region (north/south/east/west)
    - ndvi1-ndvi5: Five consecutive historical NDVI values
    
    Returns:
        JSON response containing either:
        - Success: Predicted NDVI values for the next 3 time periods
        - Error: Error message if validation fails or prediction encounters issues
    
    Example successful response:
        {
            'year1': 0.7234,
            'year2': 0.7156,
            'year3': 0.7089
        }
    """
    try:
        # Extract and normalize region name from form data (convert to lowercase)
        region = request.form['region'].lower()
        
        # Validate that the requested region has a loaded model
        if region not in models:
            return jsonify({'error': f'Invalid region: {region}'})
        
        # Extract the 5 historical NDVI values from form data and convert to float
        # These values represent consecutive time periods (e.g., 5 consecutive years)
        ndvi_values = [
            float(request.form['ndvi1']),
            float(request.form['ndvi2']),
            float(request.form['ndvi3']),
            float(request.form['ndvi4']),
            float(request.form['ndvi5'])
        ]
        
        # Validate that all NDVI values are within the valid range [-1, 1]
        # NDVI values outside this range are physically impossible
        for val in ndvi_values:
            if val < -1 or val > 1:
                return jsonify({'error': 'NDVI values must be between -1 and 1'})
                
        # Call the prediction function with validated input values
        predictions = model_prediction(ndvi_values, region)
        
        # Format prediction results as JSON-serializable dictionary
        # Round to 4 decimal places for cleaner output
        results = {
            'year1': round(float(predictions[0]), 4),  # First future prediction
            'year2': round(float(predictions[1]), 4),  # Second future prediction
            'year3': round(float(predictions[2]), 4)   # Third future prediction
        }
        
        return jsonify(results)
        
    except Exception as e:
        # Catch any unexpected errors and return them as JSON
        # This prevents server crashes and provides user-friendly error messages
        return jsonify({'error': str(e)})


def model_prediction(ndvi_values, region):
    """
    Generate NDVI predictions using the pre-trained LSTM model for a specific region.
    
    The function performs the following steps:
    1. Retrieves the appropriate regional model
    2. Reshapes input data to match LSTM input requirements
    3. Generates predictions using the model
    4. Post-processes predictions to ensure valid NDVI range
    
    Args:
        ndvi_values (list): List of 5 consecutive historical NDVI values
        region (str): Region identifier ('north', 'south', 'east', or 'west')
    
    Returns:
        numpy.ndarray: Array of 3 predicted NDVI values for future time periods
    
    Technical details:
        - LSTM models require 3D input shape: [samples, timesteps, features]
        - Our input shape: [1, 5, 1] = 1 sample, 5 timesteps, 1 feature per timestep
        - Output is clipped to valid NDVI range [-1.0, 1.0]
    """
    # Retrieve the pre-loaded model for the specified region
    model = models[region]
    
    # Reshape input data to match LSTM model expectations
    # LSTM requires 3D input: [batch_size, sequence_length, features]
    # Here: [1 sample, 5 timesteps, 1 feature]
    X = np.array(ndvi_values).reshape(1, 5, 1)
    
    # Generate predictions using the trained LSTM model
    # The model outputs predictions for the next 3 time periods
    predictions = model.predict(X)
    
    # Convert TensorFlow tensor to NumPy array if necessary
    # Some TensorFlow versions return tensors that need conversion
    predictions = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
    
    # Flatten the predictions array (remove extra dimensions)
    # and clip values to valid NDVI range [-1.0, 1.0]
    # Clipping ensures predictions don't exceed physically possible values
    predictions = np.clip(predictions.flatten(), -1.0, 1.0)
    
    return predictions


# Application entry point
if __name__ == '__main__':
    # Run Flask development server
    # debug=True enables:
    #   - Auto-reload on code changes
    #   - Detailed error pages
    #   - Interactive debugger
    # Note: Set debug=False in production for security
    app.run(debug=True)
