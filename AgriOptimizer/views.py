from django.shortcuts import render
from lol import views
from django.views.decorators.csrf import csrf_exempt

def agri_home(request):
    return render(request, 'agri_home.html')

@csrf_exempt
def multi_objective_optimize(request):
    if request.method == 'POST':
        # Get priorities from sliders
        profit = int(request.POST.get('profit', 0))
        water = int(request.POST.get('water', 0))
        soil = int(request.POST.get('soil', 0))

        total = profit + water + soil
        if total == 0: total = 1  # Avoid division by zero

        # Normalize weights
        weight_profit = profit / total
        weight_water = water / total
        weight_soil = soil / total

        # Sample crop dataset
        crops = [
            {"name": "Wheat", "profit": 80, "water": 60, "soil": 70},
            {"name": "Rice", "profit": 60, "water": 30, "soil": 50},
            {"name": "Sugarcane", "profit": 90, "water": 20, "soil": 40},
            {"name": "Millets", "profit": 50, "water": 80, "soil": 90},
            {"name": "Maize", "profit": 70, "water": 60, "soil": 60},
        ]

        # Compute scores
        for crop in crops:
            crop["score"] = (
                crop["profit"] * weight_profit +
                crop["water"] * weight_water +
                crop["soil"] * weight_soil
            )

        # Sort by score (descending)
        sorted_crops = sorted(crops, key=lambda x: x["score"], reverse=True)

        return render(request, 'agri_multi_result.html', {
            'sorted_crops': sorted_crops,
            'weights': {
                'profit': round(weight_profit * 100),
                'water': round(weight_water * 100),
                'soil': round(weight_soil * 100),
            }
        })

    return render(request, 'agri_multi_objective.html')
