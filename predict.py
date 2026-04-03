import pickle
import pandas as pd

model_path = 'model/toxicity_model.pkl'

def load_model():
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['features']

def predict_toxicity(input_data):
    """
    input_data: dict containing keys for molecular descriptors
    """
    model, scaler, features = load_model()
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure columns match
    df = df[features]
    
    # Scale
    scaled_data = scaler.transform(df)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    
    prediction_prob = 0.0
    if hasattr(model, "predict_proba"):
        prediction_prob = model.predict_proba(scaled_data)[0][1]
        
    result = "Toxic" if prediction == 1 else "Non-Toxic"
    
    return {
        "prediction": result,
        "probability": float(prediction_prob)
    }

if __name__ == "__main__":
    test_data = {
        'MolecularWeight': 350.5,
        'LogP': 3.2,
        'TPSA': 65.0,
        'NumHDonors': 1,
        'NumHAcceptors': 3,
        'NumRotatableBonds': 4,
        'AromaticRings': 2
    }
    print(predict_toxicity(test_data))
