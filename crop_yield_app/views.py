import cvxpy as cp
import numpy as np
from django.shortcuts import render
from .forms import CropForm
from .models import Crop



def recommend_crops(request):
    if request.method == 'POST':
        form = CropForm(request.POST)
        if form.is_valid():
            soil = form.cleaned_data['soil_type']
            water = form.cleaned_data['water_available']
            budget = form.cleaned_data['budget']

            # Get crop data from database
            crops = Crop.objects.filter(soil_type=soil)

            # Convert to lists
            costs = np.array([crop.cost for crop in crops])
            water_needs = np.array([crop.water_needed for crop in crops])
            yields = np.array([crop.expected_yield for crop in crops])

            # Define variables (Binary: 0 or 1 for each crop)
            x = cp.Variable(len(crops), boolean=True)

            # Objective: Maximize total yield
            objective = cp.Maximize(cp.sum(cp.multiply(yields, x)))

            # Constraints
            constraints = [
                cp.sum(cp.multiply(costs, x)) <= budget,  # Budget limit
                cp.sum(cp.multiply(water_needs, x)) <= water  # Water limit
            ]

            # Solve the optimization problem
            prob = cp.Problem(objective, constraints)
            prob.solve()

            # Get recommended crops
            selected_crops = [crops[i].name for i in range(len(crops)) if x.value[i] > 0.5]

            return render(request, 'crop_yield_app/results.html', {'crops': selected_crops})

    else:
        form = CropForm()

    return render(request, 'crop_yield_app/form.html', {'form': form})
