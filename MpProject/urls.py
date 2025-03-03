from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin site
    path('', include('lol.urls')),  # Replace "math_methods" with your app's name
    path('crop-yield-app', include('crop_yield_app.urls')),  # Replace "math_methods" with your app's name
]
            