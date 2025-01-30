from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Index page
    path('graphical_method/', views.graphical_method_view, name='graphical_method'),  # Graphical method page
]
