"""
Django forms for video upload handling.
"""

from django import forms
from .models import VideoUpload


class VideoUploadForm(forms.ModelForm):
    """Form for uploading video files for deepfake analysis."""
    
    class Meta:
        model = VideoUpload
        fields = ['video_file']
        widgets = {
            'video_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*',  
                'id': 'video-input'
            })
        }