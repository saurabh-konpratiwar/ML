# MLService/urls.py
from django.contrib import admin
from django.urls import path, include  # include is needed

urlpatterns = [
    path('admin/', admin.site.urls),
    path('MLApi/', include('MLApi.urls')),  # Make sure to include this line
]
