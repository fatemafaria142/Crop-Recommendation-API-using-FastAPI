from fastapi import FastAPI
import pickle 
import numpy as np

app = FastAPI(debug=True)

# Your classification dictionary
classifications = {'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5, 'cotton': 6,
                   'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11, 'mango': 12, 'mothbeans': 13,
                   'mungbean': 14, 'muskmelon': 15, 'orange': 16, 'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19,
                   'rice': 20, 'watermelon': 21}

@app.get('/')
def home():
    return {'text': 'Crop Recommend'}

@app.get('/predict')
def predict(N: int, P: int, K: int, temperature: float, humidity: float, ph: float, rainfall: float):
    # Load the model and scaler
    model = pickle.load(open('E:/FastAPI-Crop-Recommendation/model.pkl', 'rb'))
    scaler = pickle.load(open('E:/FastAPI-Crop-Recommendation/scaler.pkl', 'rb'))
    
    # Scale the input values
    input_values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_values_scaled = scaler.transform(input_values)

    # Make a prediction
    make_prediction = model.predict(input_values_scaled)
    predicted_crop_index = make_prediction[0]  # Assuming make_prediction is a 1D array
    
    # Map the predicted index to the crop name
    predicted_crop_name = next((crop for crop, index in classifications.items() if index == predicted_crop_index), None)

    return {'Crop Name': predicted_crop_name}
