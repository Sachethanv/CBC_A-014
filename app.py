from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Region-specific parameters
region_factors = {
    'north': 0.95,  # Slower growth in northern regions
    'south': 1.05,  # Faster growth in southern regions
    'east': 0.98,   # Slightly slower growth in eastern regions
    'west': 1.02    # Slightly faster growth in western regions
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get region from form
        region = request.form['region'].lower()
        
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
                
        # Make prediction
        predictions = statistical_prediction(ndvi_values, region)
        
        # Format prediction results
        results = {
            'year1': round(float(predictions[0]), 4),
            'year2': round(float(predictions[1]), 4),
            'year3': round(float(predictions[2]), 4)
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)})

def statistical_prediction(ndvi_values, region):
    """
    Make a prediction based on statistical analysis of the historical NDVI values
    """
    # Calculate differences between consecutive values
    diffs = [ndvi_values[i+1] - ndvi_values[i] for i in range(len(ndvi_values)-1)]
    
    # Apply weights to favor more recent trends
    weights = [0.1, 0.2, 0.3, 0.4]  # More weight to recent changes
    weighted_avg = sum(d * w for d, w in zip(diffs, weights)) / sum(weights)
    
    # Get region factor
    factor = region_factors.get(region, 1.0)
    
    # Apply dampening for future predictions
    dampening = [1.0, 0.9, 0.8]  # Less confidence in distant future predictions
    
    # Make predictions
    base = ndvi_values[-1]
    predictions = []
    
    for i, damp in enumerate(dampening):
        next_val = base + (weighted_avg * (i+1) * damp * factor)
        # Ensure prediction stays within valid NDVI range
        next_val = max(-1.0, min(1.0, next_val))
        predictions.append(next_val)
    
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
