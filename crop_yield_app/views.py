import numpy as np
import joblib
from django.shortcuts import render
from .forms import CropPredictionForm

# Load the trained model
MODEL_PATH = "C:\\Users\\Yash\\OneDrive\\Desktop\\Django\\MpProject\\crop_yield_app\\crop_model.pkl"
soil_model = joblib.load(MODEL_PATH)

def predict_soil_type(request):
    if request.method == 'POST':
        form = CropPredictionForm(request.POST)
        if form.is_valid():
            # Extract user inputs
            temperature = form.cleaned_data['temperature']
            humidity = form.cleaned_data['humidity']
            moisture = form.cleaned_data['moisture']
            nitrogen = form.cleaned_data['nitrogen']
            phosphorus = form.cleaned_data['phosphorus']
            potassium = form.cleaned_data['potassium']
            fertilizer_name = form.cleaned_data['fertilizer_name']  # If needed

            # Convert input to numpy array (ensure it matches model training format)
            input_data = np.array([[temperature, humidity, moisture, nitrogen, phosphorus, potassium]])

            # Predict soil type
            predicted_soil = soil_model.predict(input_data)[0]  # Assuming model outputs a label

            return render(request, 'crop_yield_app/results.html', {'predicted_soil': predicted_soil})

    else:
        form = CropPredictionForm()

    return render(request, 'crop_yield_app/form.html', {'form': form})
