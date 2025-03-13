from django.db import models

class Crop(models.Model):
    name = models.CharField(max_length=100)
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    moisture = models.FloatField()
    fertilizer_name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
