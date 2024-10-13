from django.urls import path
from .views import predict_profile

urlpatterns = [
    path('predict/', predict_profile, name='predict_profile'),
]
