"""
Django application configuration for the deepfake detector app.
"""

from django.apps import AppConfig


class DetectorConfig(AppConfig):
    """Configuration for the deepfake detection application."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
