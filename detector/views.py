"""
Django views for the deepfake detection web application.
Handles video upload, processing, results display, and history management.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import VideoUpload, DetectionResult
from .forms import VideoUploadForm
import time
import json
import os

# Import ML detector only when needed to avoid initialization delays
ML_AVAILABLE = True

def home(request):
    """Display the main landing page."""
    return render(request, 'detector/home.html')

def upload_video(request):
    """Handle video file uploads and redirect to processing page."""
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            
            # Mark as unprocessed for async ML analysis
            video.processed = False
            video.save()
            
            messages.success(request, 'Video uploaded successfully! Processing will begin shortly.')
            return redirect('detector:result', video_id=video.id)
    else:
        form = VideoUploadForm()
    
    return render(request, 'detector/upload.html', {'form': form})



def result(request, video_id):
    """Display video analysis results or processing status."""
    video = get_object_or_404(VideoUpload, id=video_id)
    
    # Check if ML analysis is complete
    try:
        result = DetectionResult.objects.get(video=video)
        # Analysis complete - show results with heatmaps
        heatmap_images = []
        if result.detailed_analysis:
            heatmap_images = result.detailed_analysis.get('heatmap_images', [])
        
        context = {
            'video': video,
            'result': result,
            'heatmap_images': heatmap_images,
            'result_text': 'Deepfake' if result.is_deepfake else 'Authentic',
            'processing': False
        }
    except DetectionResult.DoesNotExist:
        # Analysis not complete - show processing animation
        context = {
            'video': video,
            'result': None,
            'processing': True
        }
    
    return render(request, 'detector/result.html', context)

@csrf_exempt
def process_video_ajax(request):
    """Run ML analysis on uploaded video via AJAX request."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            video_id = data.get('video_id')
            
            if not video_id:
                return JsonResponse({'status': 'error', 'message': 'Video ID required'}, status=400)
            
            video = get_object_or_404(VideoUpload, id=video_id)
            
            # Check if already processed
            try:
                result = DetectionResult.objects.get(video=video)
                return JsonResponse({
                    'status': 'success',
                    'message': 'Video is FAKE' if result.is_deepfake else 'Video is REAL',
                    'is_deepfake': result.is_deepfake
                })
            except DetectionResult.DoesNotExist:
                pass
            
            # Initialize and run RL-based deepfake detector
            from .ml_integration.detector import get_detector
            
            start_time = time.time()
            detector = get_detector()
            if detector is None:
                raise Exception("Failed to initialize detector")
            
            # Run ML analysis and get result with attention heatmaps
            detection_result = detector.process_video(video.video_file.path)
            
            # Store analysis results in database
            detailed_analysis = {
                'heatmap_images': detection_result.get('heatmap_images', []),
                'error': detection_result.get('error', None)
            }
            
            result = DetectionResult.objects.create(
                video=video,
                is_deepfake=detection_result['is_deepfake'],
                confidence_score=0.99 if detection_result['is_deepfake'] else 0.01,
                processing_time=time.time() - start_time,
                detailed_analysis=detailed_analysis
            )
            
            video.processed = True
            video.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Video is FAKE' if result.is_deepfake else 'Video is REAL',
                'is_deepfake': result.is_deepfake
            })
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def check_status(request, video_id):
    """Check if video analysis is complete via AJAX."""
    video = get_object_or_404(VideoUpload, id=video_id)
    
    try:
        result = DetectionResult.objects.get(video=video)
        return JsonResponse({
            'status': 'completed',
            'is_deepfake': result.is_deepfake,
            'has_error': 'error' in result.detailed_analysis if result.detailed_analysis else False
        })
    except DetectionResult.DoesNotExist:
        return JsonResponse({'status': 'processing'})


def history(request):
    """Display recent video analysis history with filtering options."""
    # Show last 10 processed videos
    videos = VideoUpload.objects.filter(processed=True).order_by('-uploaded_at')[:10]
    
    # Calculate authentic vs deepfake statistics
    authentic_count = sum(1 for video in videos if video.result and not video.result.is_deepfake)
    deepfake_count = sum(1 for video in videos if video.result and video.result.is_deepfake)
    
    context = {
        'videos': videos,
        'authentic_count': authentic_count,
        'deepfake_count': deepfake_count,
    }
    return render(request, 'detector/history.html', context)

def about(request):
    """Display information about the project and technology."""
    return render(request, 'detector/about.html')

@require_http_methods(["POST", "DELETE"])
def delete_video(request, video_id):
    """Delete a single video and its analysis results from database and filesystem."""
    video = get_object_or_404(VideoUpload, id=video_id)
    
    try:
        # Get file path before database deletion
        video_file_path = video.video_file.path if video.video_file else None
        
        # Delete database record (DetectionResult auto-deleted via CASCADE)
        video.delete()
        
        # Clean up physical video file
        if video_file_path and os.path.exists(video_file_path):
            try:
                os.remove(video_file_path)
                print(f"[DELETE] Successfully deleted video file: {video_file_path}")
            except OSError as e:
                print(f"[DELETE] Warning: Could not delete video file {video_file_path}: {e}")
        
        messages.success(request, 'Video deleted successfully!')
        
        # Return appropriate response based on request type
        if request.content_type == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True, 
                'message': 'Video deleted successfully!',
                'redirect': '/'
            })
        else:
            return redirect('detector:home')
            
    except Exception as e:
        error_msg = f'Failed to delete video: {str(e)}'
        messages.error(request, error_msg)
        
        if request.content_type == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': error_msg}, status=500)
        else:
            return redirect('detector:result', video_id=video_id)


@require_http_methods(["POST"])
def bulk_delete_videos(request):
    """Delete multiple videos and their analysis results in batch operation."""
    try:
        import json
        data = json.loads(request.body)
        video_ids = data.get('video_ids', [])
        
        if not video_ids:
            return JsonResponse({'success': False, 'error': 'No videos selected'}, status=400)
        
        deleted_count = 0
        failed_deletions = []
        
        # Process each video deletion
        for video_id in video_ids:
            try:
                video = VideoUpload.objects.get(id=video_id)
                
                # Get file path before database deletion
                video_file_path = video.video_file.path if video.video_file else None
                
                # Delete database record
                video.delete()
                
                # Clean up physical file
                if video_file_path and os.path.exists(video_file_path):
                    try:
                        os.remove(video_file_path)
                        print(f"[BULK_DELETE] Successfully deleted video file: {video_file_path}")
                    except OSError as e:
                        print(f"[BULK_DELETE] Warning: Could not delete video file {video_file_path}: {e}")
                
                deleted_count += 1
                
            except VideoUpload.DoesNotExist:
                failed_deletions.append(f"Video {video_id} not found")
            except Exception as e:
                failed_deletions.append(f"Failed to delete video {video_id}: {str(e)}")
        
        if deleted_count > 0:
            messages.success(request, f'Successfully deleted {deleted_count} video{"s" if deleted_count > 1 else ""}!')
        
        if failed_deletions:
            error_msg = f"{len(failed_deletions)} deletion(s) failed: {'; '.join(failed_deletions[:3])}"
            if len(failed_deletions) > 3:
                error_msg += f" and {len(failed_deletions) - 3} more..."
            messages.warning(request, error_msg)
        
        return JsonResponse({
            'success': True,
            'deleted_count': deleted_count,
            'failed_count': len(failed_deletions),
            'message': f'Successfully deleted {deleted_count} video{"s" if deleted_count > 1 else ""}'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        error_msg = f'Bulk deletion failed: {str(e)}'
        return JsonResponse({'success': False, 'error': error_msg}, status=500)