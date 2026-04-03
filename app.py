from flask import Flask, render_template, request, jsonify
import os
from predict import predict_toxicity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming JSON or Form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            # Convert form string values to float
            for k in data:
                data[k] = float(data[k])
        
        result = predict_toxicity(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Ensure images folder exists
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True, port=5000)
