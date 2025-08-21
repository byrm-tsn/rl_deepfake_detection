"""
Database models for the deepfake detection application.
Handles video uploads and their analysis results.
"""

from django.db import models
import uuid


class VideoUpload(models.Model):
    """Model to store uploaded video files and their metadata."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_file = models.FileField(upload_to='uploads/videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False) 
    
    class Meta:
        ordering = ['-uploaded_at'] 
    
    def __str__(self):
        return f"Video {self.id} - {self.uploaded_at}"


class DetectionResult(models.Model):
    """Model to store AI analysis results for each video."""
    
    video = models.OneToOneField(VideoUpload, on_delete=models.CASCADE, related_name='result')
    is_deepfake = models.BooleanField(default=False)  # Main detection result
    confidence_score = models.FloatField(default=0.0)  # AI confidence (0-1)
    processing_time = models.FloatField(default=0.0)  # Analysis duration in seconds
    detailed_analysis = models.JSONField(default=dict, blank=True)  # Store heatmaps/metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Result for {self.video.id} - Deepfake: {self.is_deepfake}"