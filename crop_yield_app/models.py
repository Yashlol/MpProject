from django.db import models

class Crop(models.Model):
    name = models.CharField(max_length=100)
    soil_type = models.CharField(max_length=100)
    water_needed = models.FloatField()  # Liters per acre
    expected_yield = models.FloatField()  # Yield per acre
    cost = models.FloatField()  # Cost per acre

    def __str__(self):
        return self.name
