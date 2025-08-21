#!/usr/bin/env python3
"""
Video Frame Extraction and Face Detection Pipeline
Extracts 50 frames evenly throughout videos with face detection and tracking
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis
from retinaface.pre_trained_models import get_model
from facenet_pytorch import MTCNN
import torch
import platform
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device detection for GPU acceleration
def get_device():
    """Detect and return the best available device (MPS for M-series Macs, CUDA for NVIDIA, CPU otherwise)"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Apple Silicon GPU (MPS) detected")
    else:
        device = 'cpu'
        logger.info("Using CPU for processing")
    return device

# Get ONNX providers based on platform
def get_onnx_providers():
    """Get optimal ONNX execution providers for the current platform"""
    if torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logger.info("Using CUDA execution provider for ONNX")
    elif platform.system() == 'Darwin':
        # For macOS, try CoreML provider first
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        logger.info("Using CoreML execution provider for ONNX")
    else:
        providers = ['CPUExecutionProvider']
        logger.info("Using CPU execution provider for ONNX")
    return providers


class FrameExtractor:
    """Handles video frame extraction with temporal consistency"""
    
    def __init__(self, target_frames: int = 50, extraction_multiplier: float = 3.0):
        self.target_frames = target_frames
        self.extraction_multiplier = extraction_multiplier  # Extract more frames to account for failed detections
    
    def extract_all_frames(self, video_path: str) -> List[Tuple[np.ndarray, int]]:
        """
        Extract frames throughout the video, more than needed to account for detection failures
        Returns: List of (frame, frame_index) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract more frames than needed
        num_to_extract = min(int(self.target_frames * self.extraction_multiplier), total_frames)
        
        if total_frames < self.target_frames:
            logger.warning(f"Video has only {total_frames} frames, less than required {self.target_frames}")
            frame_indices = list(range(total_frames))
        else:
            # Calculate evenly spaced frame indices
            frame_indices = np.linspace(0, total_frames - 1, num_to_extract, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((frame, idx))
            else:
                logger.warning(f"Failed to read frame {idx} from {video_path}")
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video for processing")
        
        return frames


class FaceDetector:
    """Handles face detection using GPU-accelerated RetinaFace, MTCNN, with InsightFace fallback"""
    
    def __init__(self, margin: float = 0.3, target_size: int = 224):
        self.margin = margin
        self.target_size = target_size
        
        # Get optimal device and providers
        self.device = get_device()
        self.providers = get_onnx_providers()
        
        # Initialize RetinaFace first (most robust, GPU-accelerated)
        try:
            # RetinaFace doesn't support MPS, so use CUDA if available, otherwise CPU
            retinaface_device = 'cuda' if self.device == 'cuda' else 'cpu'
            self.retinaface_model = get_model("resnet50_2020-07-20", max_size=2048, device=retinaface_device)
            self.retinaface_model.eval()
            logger.info(f"RetinaFace initialized as PRIMARY detector with {retinaface_device}")
        except Exception as e:
            logger.warning(f"Failed to initialize RetinaFace: {e}")
            self.retinaface_model = None
        
        # Initialize MTCNN as secondary detector with GPU support
        # MTCNN supports both CUDA and MPS
        self.mtcnn = MTCNN(min_face_size=20, device=self.device, select_largest=True)
        logger.info(f"MTCNN initialized as SECONDARY detector with {self.device}")
        
        # Initialize InsightFace as final fallback (CPU-based)
        try:
            self.app = FaceAnalysis(providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace initialized as FALLBACK detector")
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace: {e}")
            self.app = None
        
        # Tracking state for difficult videos
        self.last_valid_bbox = None
        self.frames_since_detection = 0
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face using cascading approach: RetinaFace -> MTCNN -> InsightFace
        Returns: Dict with bbox, confidence, and detection_method
        """
        # Try RetinaFace first (most robust, GPU-accelerated)
        if self.retinaface_model is not None:
            logger.debug("Trying RetinaFace (PRIMARY)...")
            try:
                # Convert BGR to RGB for RetinaFace
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Predict faces
                predictions = self.retinaface_model.predict_jsons(image_rgb)
                
                if predictions and len(predictions) > 0:
                    # Get the face with highest confidence
                    best_face = max(predictions, key=lambda x: x.get('score', 0))
                    
                    if best_face.get('score', 0) >= 0.5:  # Confidence threshold
                        # Extract bounding box
                        bbox = np.array([
                            int(best_face['x1']),
                            int(best_face['y1']),
                            int(best_face['x2']),
                            int(best_face['y2'])
                        ])
                        confidence = float(best_face.get('score', 0.9))
                        
                        self.last_valid_bbox = bbox
                        self.frames_since_detection = 0
                        
                        return {
                            'bbox': bbox,
                            'confidence': confidence,
                            'detection_method': 'retinaface'
                        }
            except Exception as e:
                logger.debug(f"RetinaFace detection failed: {e}")
        
        # If RetinaFace fails, try MTCNN (GPU-accelerated, good for clear faces)
        logger.debug("RetinaFace failed/unavailable, trying MTCNN (SECONDARY)...")
        try:
            # MTCNN expects RGB PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect faces - returns boxes and probabilities
            boxes, probs = self.mtcnn.detect(image_rgb)
            
            if boxes is not None and len(boxes) > 0:
                # Get box with highest probability
                best_idx = np.argmax(probs)
                
                if probs[best_idx] >= 0.5:  # Lower MTCNN threshold
                    box = boxes[best_idx]
                    bbox = np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                    
                    self.last_valid_bbox = bbox
                    self.frames_since_detection = 0
                    
                    return {
                        'bbox': bbox,
                        'confidence': float(probs[best_idx]),
                        'detection_method': 'mtcnn'
                    }
        except Exception as e:
            logger.debug(f"MTCNN detection failed: {e}")
        
        # If MTCNN fails, try InsightFace as final fallback
        if self.app is not None:
            logger.debug("MTCNN failed, trying InsightFace (FALLBACK)...")
            try:
                faces = self.app.get(image)
                
                # Filter faces by confidence (much lower threshold)
                valid_faces = [f for f in faces if f.det_score >= 0.3]  # Very low threshold
                
                if len(valid_faces) > 0:
                    # Get the largest face (most likely the main subject)
                    face = max(valid_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    
                    bbox = face.bbox.astype(int)
                    confidence = float(face.det_score)
                    
                    self.last_valid_bbox = bbox
                    self.frames_since_detection = 0
                    
                    return {
                        'bbox': bbox,
                        'confidence': confidence,
                        'detection_method': 'insightface'
                    }
            except Exception as e:
                logger.debug(f"InsightFace detection failed: {e}")
        
        # All methods failed - try tracking from previous frame if available
        self.frames_since_detection += 1
        
        # If we had a recent detection, use interpolation/tracking
        if self.last_valid_bbox is not None and self.frames_since_detection < 5:
            logger.debug("Using previous bbox as fallback")
            return {
                'bbox': self.last_valid_bbox,
                'confidence': 0.3,  # Low confidence for tracked
                'detection_method': 'tracked'
            }
        
        return None
    
    def enhance_frame(self, image: np.ndarray) -> np.ndarray:
        """Enhance frame for better face detection in poor quality videos"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def add_margin(self, bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Add margin to bounding box while keeping it within image bounds"""
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        
        width = x2 - x1
        height = y2 - y1
        
        # Add margin
        margin_x = int(width * self.margin)
        margin_y = int(height * self.margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        return np.array([x1, y1, x2, y2])
    
    
    def resize_frame(self, image: np.ndarray) -> np.ndarray:
        """Resize frame to target size (224x224) using center crop to avoid black padding"""
        h, w = image.shape[:2]
        
        # Calculate scale to fill the target size (max instead of min)
        scale = max(self.target_size / w, self.target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image (it will be larger than target_size in at least one dimension)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center crop to target size
        y_start = (new_h - self.target_size) // 2
        x_start = (new_w - self.target_size) // 2
        
        cropped = resized[y_start:y_start + self.target_size, 
                         x_start:x_start + self.target_size]
        
        return cropped
    
    def reset_tracking(self):
        """Reset tracking state for new video"""
        self.last_valid_bbox = None
        self.frames_since_detection = 0


class FaceTracker:
    """Tracks faces across frames using IoU threshold"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.track_id_counter = 0
        self.active_tracks = {}  # track_id -> last_bbox
    
    def calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def assign_track_id(self, bbox: np.ndarray) -> int:
        """Assign track ID based on IoU with existing tracks"""
        best_iou = 0.0
        best_track_id = None
        
        for track_id, last_bbox in self.active_tracks.items():
            iou = self.calculate_iou(bbox, last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        if best_iou >= self.iou_threshold and best_track_id is not None:
            # Update existing track
            self.active_tracks[best_track_id] = bbox
            return best_track_id
        else:
            # Create new track
            new_track_id = self.track_id_counter
            self.track_id_counter += 1
            self.active_tracks[new_track_id] = bbox
            return new_track_id
    
    def reset(self):
        """Reset tracking for new video"""
        self.track_id_counter = 0
        self.active_tracks = {}


class VideoPreprocessor:
    """Main preprocessing pipeline"""
    
    def __init__(self, 
                 dataset_path: str,
                 output_base_path: str = None,
                 num_frames: int = 50,
                 face_margin: float = 0.3,
                 iou_threshold: float = 0.5):
        
        self.dataset_path = Path(dataset_path)
        # Default output path adds '_processed' suffix
        if output_base_path is None:
            dataset_name = self.dataset_path.name
            output_base_path = self.dataset_path.parent / f"{dataset_name}_processed"
        self.output_base_path = Path(output_base_path)
        self.frame_extractor = FrameExtractor(num_frames)
        self.face_detector = FaceDetector(face_margin)
        self.face_tracker = FaceTracker(iou_threshold)
        self.metadata_records = []
        self.processing_stats = {
            'total_frames_processed': 0,
            'total_frames_skipped': 0,
            'videos_with_insufficient_faces': []
        }
    
    def process_video(self, video_path: Path, split: str, label: str) -> List[Dict]:
        """Process single video and return metadata records"""
        video_id = video_path.stem
        logger.info(f"Processing video: {video_id}")
        
        # Extract frames
        frames = self.frame_extractor.extract_all_frames(str(video_path))
        if not frames:
            logger.error(f"No frames extracted from {video_path}")
            return []
        
        # Reset face tracker and detector for new video
        self.face_tracker.reset()
        self.face_detector.reset_tracking()
        
        records = []
        successful_detections = 0
        skipped_frames = 0
        
        # Create output directory
        output_dir = self.output_base_path / split / label / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame, frame_idx in frames:
            # Stop if we've reached target number of successful detections
            if successful_detections >= self.frame_extractor.target_frames:
                break
                
            # Detect face
            face_result = self.face_detector.detect_face(frame)
            
            # If detection fails, try with enhanced frame
            if not face_result:
                enhanced_frame = self.face_detector.enhance_frame(frame)
                face_result = self.face_detector.detect_face(enhanced_frame)
                if face_result:
                    frame = enhanced_frame  # Use enhanced frame if it worked
                    face_result['detection_method'] += '_enhanced'
            
            if not face_result:
                # Skip this frame if no face detected
                skipped_frames += 1
                logger.debug(f"No face detected in {video_id} frame {frame_idx}, skipping")
                continue
            
            # Face detected (either InsightFace or RetinaFace)
            bbox = self.face_detector.add_margin(face_result['bbox'], frame.shape)
            track_id = self.face_tracker.assign_track_id(bbox)
            confidence = face_result['confidence']
            detection_method = face_result['detection_method']
            
            # Crop frame
            x1, y1, x2, y2 = bbox
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if cropped_frame.shape[0] < 10 or cropped_frame.shape[1] < 10:
                logger.warning(f"Crop too small: {cropped_frame.shape}, skipping frame {frame_idx}")
                skipped_frames += 1
                continue
            
            # Resize to 224x224
            resized_frame = self.face_detector.resize_frame(cropped_frame)
            
            # Verify shape before saving
            if resized_frame.shape != (self.face_detector.target_size, self.face_detector.target_size, 3):
                logger.warning(f"Unexpected shape after resize: {resized_frame.shape}, skipping")
                skipped_frames += 1
                continue
            
            # Save frame as PNG
            frame_filename = f"frame_{successful_detections:04d}_idx_{frame_idx:06d}.png"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), resized_frame)
            
            # Create metadata record
            record = {
                'video_id': video_id,
                'frame_number': successful_detections,
                'frame_index': frame_idx,
                'frame_path': str(frame_path.relative_to(self.output_base_path)),
                'label': label,
                'split': split,
                'original_width': frame.shape[1],
                'original_height': frame.shape[0],
                'face_x': int(bbox[0]),
                'face_y': int(bbox[1]),
                'face_width': int(bbox[2] - bbox[0]),
                'face_height': int(bbox[3] - bbox[1]),
                'detection_confidence': confidence,
                'face_tracking_id': track_id,
                'detection_method': detection_method
            }
            records.append(record)
            successful_detections += 1
        
        # Update statistics
        self.processing_stats['total_frames_processed'] += len(frames)
        self.processing_stats['total_frames_skipped'] += skipped_frames
        
        logger.info(f"Video {video_id}: {successful_detections} faces detected, {skipped_frames} frames skipped")
        
        if successful_detections < self.frame_extractor.target_frames:
            logger.warning(f"Only {successful_detections} faces detected in {video_id}, target was {self.frame_extractor.target_frames}")
            self.processing_stats['videos_with_insufficient_faces'].append({
                'video_id': video_id,
                'detected': successful_detections,
                'target': self.frame_extractor.target_frames
            })
        
        return records
    
    def process_dataset(self):
        """Process entire dataset"""
        logger.info(f"Starting preprocessing of dataset: {self.dataset_path}")
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                logger.warning(f"Split {split} not found, skipping")
                continue
            
            # Process each label
            for label in ['real', 'fake']:
                label_path = split_path / label
                if not label_path.exists():
                    logger.warning(f"Label {label} not found in {split}, skipping")
                    continue
                
                # Find all video files
                video_files = list(label_path.glob('*.mp4')) + list(label_path.glob('*.avi'))
                logger.info(f"Found {len(video_files)} videos in {split}/{label}")
                
                # Process each video
                for video_path in tqdm(video_files, desc=f"{split}/{label}"):
                    try:
                        records = self.process_video(video_path, split, label)
                        self.metadata_records.extend(records)
                    except Exception as e:
                        logger.error(f"Error processing {video_path}: {str(e)}")
                        continue
        
        # Save metadata
        self.save_metadata()
        logger.info("Preprocessing completed!")
    
    def save_metadata(self):
        """Save metadata to CSV"""
        if not self.metadata_records:
            logger.warning("No metadata to save")
            return
        
        df = pd.DataFrame(self.metadata_records)
        metadata_path = self.output_base_path / 'metadata.csv'
        df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Total frames processed: {len(df)}")
        
        # Print summary statistics
        print("\n=== Processing Summary ===")
        print(f"Total videos processed: {df['video_id'].nunique()}")
        print(f"Total frames successfully extracted: {len(df)}")
        print(f"\nFrames per split:")
        print(df.groupby('split').size())
        print(f"\nDetection methods used:")
        print(df['detection_method'].value_counts())
        print(f"\nAverage detection confidence: {df['detection_confidence'].mean():.3f}")
        
        # Print skipped frames statistics
        print(f"\n=== Detection Statistics ===")
        print(f"Total frames processed: {self.processing_stats['total_frames_processed']}")
        print(f"Total frames with successful face detection: {len(df)}")
        print(f"Total frames skipped (no face detected): {self.processing_stats['total_frames_skipped']}")
        
        if self.processing_stats['total_frames_processed'] > 0:
            success_rate = (len(df) / self.processing_stats['total_frames_processed']) * 100
            skip_rate = (self.processing_stats['total_frames_skipped'] / self.processing_stats['total_frames_processed']) * 100
            print(f"Face detection success rate: {success_rate:.1f}%")
            print(f"Frame skip rate: {skip_rate:.1f}%")
        
        # Print videos with insufficient faces
        if self.processing_stats['videos_with_insufficient_faces']:
            print(f"\n=== Videos with Insufficient Faces ===")
            print(f"Total: {len(self.processing_stats['videos_with_insufficient_faces'])} videos")
            for video_info in self.processing_stats['videos_with_insufficient_faces']:
                print(f"  - {video_info['video_id']}: {video_info['detected']}/{video_info['target']} faces detected")


def main():
    """Main entry point"""
    # Show GPU acceleration status
    logger.info("=== GPU Acceleration Status ===")
    device = get_device()
    logger.info(f"Primary device: {device}")
    
    if device == 'cuda':
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device == 'mps':
        logger.info("Apple Silicon GPU acceleration enabled")
    
    # Initialize and run preprocessor with default settings
    preprocessor = VideoPreprocessor(
        dataset_path='../data/balanced_dataset',
        output_base_path='../data/processed_data',
        num_frames=50,
        face_margin=0.3,
        iou_threshold=0.5
    )
    
    preprocessor.process_dataset()


if __name__ == '__main__':
    main()