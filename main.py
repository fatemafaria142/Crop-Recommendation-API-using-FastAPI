from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI(debug=True)


# classification dictionary
classifications = {'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5, 'cotton': 6,
                   'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11, 'mango': 12, 'mothbeans': 13,
                   'mungbean': 14, 'muskmelon': 15, 'orange': 16, 'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19,
                   'rice': 20, 'watermelon': 21}

class CropPredictionInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post('/')
def predict(input_data: CropPredictionInput):
    # Load the model and scaler
    model = pickle.load(open('E:/FastAPI-Crop-Recommendation/model.pkl', 'rb'))
    scaler = pickle.load(open('E:/FastAPI-Crop-Recommendation/scaler.pkl', 'rb'))

    # Scale the input values
    input_values = np.array([[input_data.N, input_data.P, input_data.K,
                              input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall]])
    input_values_scaled = scaler.transform(input_values)

    # Make a prediction
    make_prediction = model.predict(input_values_scaled)
    predicted_crop_index = make_prediction[0]  # Assuming make_prediction is a 1D array

    # Map the predicted index to the crop name
    predicted_crop_name = next((crop for crop, index in classifications.items() if index == predicted_crop_index), None)

    return {'Crop Name': predicted_crop_name}
