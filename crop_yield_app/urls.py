from django.urls import path
from .views import recommend_crops

urlpatterns = [
    path('', recommend_crops, name='recommend_crops'),
]
