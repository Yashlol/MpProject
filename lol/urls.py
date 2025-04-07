from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Index page
    path('graphical_method/', views.graphical_method_view, name='graphical_method'), 
    path('simplex/', views.simplex_method_view, name='simplex_method'),
    path('transportation/', views.transportation_method_view, name='transportation_method'),
    path('knapsack/', views.knapsack_solver, name='knapsack'),
     path('genetic-algorithm/', views.genetic_algorithm, name='genetic_algorithm'),
]
