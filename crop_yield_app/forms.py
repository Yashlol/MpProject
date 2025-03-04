from django import forms
from .models import Crop

class CropForm(forms.Form):
    soil_type = forms.CharField(label="Soil Type", max_length=100)
    water_available = forms.FloatField(label="Water Available (Liters per acre)")
    budget = forms.FloatField(label="Budget per acre")


SOIL_CHOICES = [
    ('loamy', 'Loamy'),
    ('clayey', 'Clayey'),
    ('sandy', 'Sandy'),
    ('alluvial', 'Alluvial'),
    ('black', 'Black'),
    ('arid', 'Arid'),
]

class CropRecommendationForm(forms.Form):
    soil_type = forms.ChoiceField(choices=SOIL_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    water_available = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Water in liters'}))
    budget = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Budget in currency'}))
