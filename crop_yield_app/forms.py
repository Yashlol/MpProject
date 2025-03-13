from django import forms

class CropPredictionForm(forms.Form):
    temperature = forms.FloatField(
        label="Temperature (°C)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter temperature in °C'})
    )
    humidity = forms.FloatField(
        label="Humidity (%)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter humidity in %'})
    )
    moisture = forms.FloatField(
        label="Soil Moisture (%)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter soil moisture in %'})
    )
    nitrogen = forms.FloatField(
        label="Nitrogen Level",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter nitrogen level'})
    )
    phosphorus = forms.FloatField(
        label="Phosphorus Level",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter phosphorus level'})
    )
    potassium = forms.FloatField(
        label="Potassium Level",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter potassium level'})
    )
    fertilizer_name = forms.CharField(
        label="Fertilizer Name",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter fertilizer name'})
    )
