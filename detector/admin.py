"""
Django admin configuration for deepfake detection models.
Provides admin interface for managing videos and results.
"""

from django.contrib import admin
from .models import VideoUpload, DetectionResult


@admin.register(VideoUpload)
class VideoUploadAdmin(admin.ModelAdmin):
    """Admin interface for uploaded videos."""
    list_display = ['id', 'uploaded_at', 'processed']
    list_filter = ['processed', 'uploaded_at']
    search_fields = ['id']
    date_hierarchy = 'uploaded_at'


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    """Admin interface for detection results."""
    list_display = ['video', 'is_deepfake', 'confidence_score', 'created_at']
    list_filter = ['is_deepfake', 'created_at']
    search_fields = ['video__id']
    readonly_fields = ['detailed_analysis'] 