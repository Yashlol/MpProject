from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin site
    path('agri/', include('AgriOptimizer.urls')),  # Include URLs from AgriOptimizer app
    path('', include('lol.urls')),  # Replace "math_methods" with your app's name
]
            