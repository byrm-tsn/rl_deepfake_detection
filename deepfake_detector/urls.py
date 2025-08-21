"""
Main URL configuration for deepfake detection project.
Routes admin and app URLs, handles media serving in development.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),    # Django admin interface
    path('', include('detector.urls')), # Main application URLs
]

# Serve media files in development mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)