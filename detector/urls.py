"""
URL routing configuration for the deepfake detection application.
Maps URLs to their corresponding view functions.
"""

from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),                                    # Landing page
    path('upload/', views.upload_video, name='upload'),                   # Video upload form
    path('history/', views.history, name='history'),                     # Analysis history
    path('about/', views.about, name='about'),                          # Project information
    
    # Video processing and results
    path('result/<uuid:video_id>/', views.result, name='result'),        # Show analysis results
    path('process_ajax/', views.process_video_ajax, name='process_ajax'), # AJAX ML processing
    path('check_status/<uuid:video_id>/', views.check_status, name='check_status'), # Processing status
    
    # Video management
    path('delete/<uuid:video_id>/', views.delete_video, name='delete'),  # Delete single video
    path('bulk_delete/', views.bulk_delete_videos, name='bulk_delete'),  # Delete multiple videos
]