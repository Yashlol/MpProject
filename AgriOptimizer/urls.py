from django.urls import path
from . import views

urlpatterns = [
    path('', views.agri_home, name='agri_home'),
    path('multi-objective/', views.multi_objective_optimize, name='multi_objective_optimize')
]
