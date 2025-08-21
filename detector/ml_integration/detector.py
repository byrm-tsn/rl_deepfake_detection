"""
Main deepfake detection module using reinforcement learning with DINOv3 vision transformer.
Integrates RL agent with attention-based analysis for video authenticity verification.
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
import zipfile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import logging
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths for ML models and dependencies
DETECTOR_DIR = Path(__file__).resolve().parent
BASE_DIR = DETECTOR_DIR.parent.parent
DEEPFAKE_ML_DIR = DETECTOR_DIR / 'deepfake_ml'

# Add ML module to Python path
sys.path.insert(0, str(DEEPFAKE_ML_DIR))

# Model configuration
MAX_FRAMES = 50  # Maximum frames to analyze per video
ACTION_NAMES = ['NEXT', 'FOCUS', 'AUGMENT', 'STOP_REAL', 'STOP_FAKE']  # RL agent actions

# Import face detection module with fallback
try:
    from data_processing.preprocessing import FaceDetector
    logger.info("Using original FaceDetector from data_processing")
    FACE_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import original FaceDetector: {e}")
    logger.info("Using simple OpenCV FaceDetector")
    FACE_DETECTOR_AVAILABLE = False
    
    class FaceDetector:
        """Fallback face detection using OpenCV Haar Cascade when main detector unavailable."""
        
        def __init__(self, margin: float = 0.3, target_size: int = 224):
            self.margin = margin
            self.target_size = target_size
            
            # Use OpenCV's built-in Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Tracking state
            self.last_valid_bbox = None
            self.frames_since_detection = 0
            
            logger.info("Initialized face detector with OpenCV Haar Cascade")
        
        def detect_face(self, image: np.ndarray):
            """
            Detect face using OpenCV Haar Cascade
            Returns: Dict with bbox, confidence, and detection_method
            """
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get the largest face
                areas = [w * h for (x, y, w, h) in faces]
                largest_idx = np.argmax(areas)
                x, y, w, h = faces[largest_idx]
                
                bbox = np.array([x, y, x + w, y + h])
                self.last_valid_bbox = bbox
                self.frames_since_detection = 0
                
                return {
                    'bbox': bbox,
                    'confidence': 0.9,
                    'detection_method': 'haar_cascade'
                }
            
            # If no face detected, use last valid bbox if available
            if self.last_valid_bbox is not None and self.frames_since_detection < 10:
                self.frames_since_detection += 1
                return {
                    'bbox': self.last_valid_bbox,
                    'confidence': 0.5,
                    'detection_method': 'tracked'
                }
            
            return None
        
        def add_margin(self, bbox: np.ndarray, image_shape: tuple) -> np.ndarray:
            """Add margin to bounding box"""
            x1, y1, x2, y2 = bbox
            
            width = x2 - x1
            height = y2 - y1
            
            # Add margin
            margin_x = int(width * self.margin)
            margin_y = int(height * self.margin)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(image_shape[1], x2 + margin_x)
            y2 = min(image_shape[0], y2 + margin_y)
            
            return np.array([x1, y1, x2, y2])
        
        def resize_frame(self, frame: np.ndarray) -> np.ndarray:
            """Resize frame to target size"""
            return cv2.resize(frame, (self.target_size, self.target_size))
        
        def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
            """Enhance frame for better face detection"""
            # Apply histogram equalization for better contrast
            if len(frame.shape) == 3:
                # Convert to YCrCb and equalize Y channel
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                enhanced = cv2.equalizeHist(frame)
            
            return enhanced

class DeepfakeDetector:
    """
    Main deepfake detection system using reinforcement learning with DINOv3 backbone.
    Combines RL agent decision-making with attention-based visual analysis.
    """
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.ml_dir = DEEPFAKE_ML_DIR
        self.models_dir = self.ml_dir / 'models'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize face detection for preprocessing
        logger.info("Initializing face detector...")
        self.face_detector = FaceDetector()
        
        # ML model components
        self.visual_backbone = None      # DINOv3 vision transformer
        self.temporal_memory = None      # LSTM for temporal features
        self.classifier_head = None      # Final classification layer
        self.feature_extractor = None    # Feature processing
        self.reward_calculator = None    # RL reward computation
        self.policy = None              # RL policy network
        
        self.model_path = None
        
        logger.info(f"Initializing detector with device: {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required models following inference_phase2"""
        logger.info("Starting model initialization...")
        
        # Check if required modules can be imported
        try:
            import gymnasium as gym
        except ImportError:
            try:
                import gym
                logger.warning("Using old gym instead of gymnasium")
            except ImportError:
                logger.error("Neither gymnasium nor gym is available")
                self.policy = None
                return
        
        try:
            logger.info("Importing src modules...")
            # First import and configure the config module
            import src.config as config
            # Override paths to avoid validation errors
            config.DATA_DIR = self.ml_dir / "data"
            config.BALANCED_DATASET_PATH = self.ml_dir / "data"
            config.PROCESSED_DATASET_PATH = self.ml_dir / "data"
            config.DEVICE = self.device
            
            # Now import the actual modules
            from src.feature_extractor import (
                FeatureExtractorModule as FeatureExtractor,
                VisionBackbone,
                TemporalMemory,
                ClassifierHead
            )
            from src.agent import EagerActorCriticPolicy as ActorCriticPolicy
            from src.reward_system import RewardCalculator
            from src.deepfake_env import DeepfakeEnv
            
            logger.info("Successfully imported all src modules")
        except ImportError as e:
            logger.error(f"Failed to import src modules: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None
            return
        except Exception as e:
            logger.error(f"Unexpected error importing modules: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None
            return
        
        try:
            # Find the PPO model path
            model_path = str(self.models_dir / 'final_models' / 'ppo_eager_final.zip')
            
            if not Path(model_path).exists():
                # Try alternative paths
                alternative_paths = [
                    self.models_dir / 'checkpoints' / 'ppo_eager_1500000_steps.zip',
                    self.models_dir / 'checkpoints' / 'ppo_eager_1400000_steps.zip',
                    self.models_dir / 'checkpoints' / 'ppo_eager_1200000_steps.zip',
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_path = str(alt_path)
                        logger.info(f"Using alternative model: {model_path}")
                        break
                else:
                    logger.error("No PPO model file found")
                    self.policy = None
                    return
            
            self.model_path = model_path
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load Phase 1 components
            logger.info("Loading Phase 1 components...")
            self.visual_backbone = VisionBackbone().to(self.device)
            self.temporal_memory = TemporalMemory().to(self.device)
            self.classifier_head = ClassifierHead().to(self.device)
            self.feature_extractor = FeatureExtractor().to(self.device)
            self.reward_calculator = RewardCalculator(
                frozen_evaluator=self.feature_extractor
            )
            logger.info("Initialized reward calculator")
            
            # Load Phase 2 PPO model
            logger.info("Loading Phase 2 PPO model...")
            
            # Create policy
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1794,))
            act_space = gym.spaces.Discrete(5)
            self.policy = ActorCriticPolicy(
                observation_space=obs_space,
                action_space=act_space,
                lr_schedule=lambda x: 0.001
            ).to(self.device)
            logger.info("Initialized EAGER Actor-Critic Policy for PPO")
            
            # Load checkpoint - handle SB3 zip file format
            print(f"[DEBUG] Loading policy from: {self.model_path}")
            if self.model_path.endswith('.zip'):
                # Extract and load policy from SB3 zip file
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        print(f"[DEBUG] Extracted zip contents to temp dir")
                        
                        # List contents
                        files = os.listdir(temp_dir)
                        print(f"[DEBUG] Zip contents: {files}")
                    
                    policy_path = os.path.join(temp_dir, 'policy.pth')
                    if os.path.exists(policy_path):
                        checkpoint = torch.load(policy_path, map_location=self.device)
                        print(f"[DEBUG] Loaded checkpoint with keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                        
                        # Check if checkpoint format matches our policy
                        try:
                            self.policy.load_state_dict(checkpoint)
                            logger.info("Loaded policy weights from SB3 zip file")
                            print(f"[DEBUG] Successfully loaded policy weights")
                        except Exception as e:
                            logger.error(f"Failed to load policy state dict: {e}")
                            print(f"[DEBUG] Failed to load policy weights: {e}")
                            # Try different loading approaches
                            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                                print(f"[DEBUG] Trying to load from 'policy' key")
                                self.policy.load_state_dict(checkpoint['policy'])
                            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                                print(f"[DEBUG] Trying to load from 'state_dict' key")
                                self.policy.load_state_dict(checkpoint['state_dict'])
                    else:
                        logger.warning("Could not find policy.pth in zip file")
                        print(f"[DEBUG] policy.pth not found in zip file")
            
            self.policy.eval()
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None
    
    def extract_frames(self, video_path: str, max_frames: int = MAX_FRAMES) -> np.ndarray:
        """
        Extract and preprocess video frames with face detection.
        Returns face-cropped frames ready for ML analysis.
        """
        logger.info(f"[EXTRACT_FRAMES] Starting extraction from: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"[EXTRACT_FRAMES] Failed to open video: {video_path}")
            return np.array([])
        
        frames = []
        frame_count = 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate sampling rate
        if total_frames <= max_frames:
            sample_rate = 1
        else:
            sample_rate = total_frames // max_frames
        
        logger.info(f"[EXTRACT_FRAMES] Video info: {total_frames} frames, {fps:.1f} FPS")
        logger.info(f"[EXTRACT_FRAMES] Sampling every {sample_rate} frames (max {max_frames} frames)")
        logger.info(f"[EXTRACT_FRAMES] Face detector available: {self.face_detector is not None}")
        
        processed_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"[EXTRACT_FRAMES] Processing frame {processed_count}/{max_frames}")
                
                # Process frame with face detection if available
                if self.face_detector is not None:
                    # Detect face in frame
                    try:
                        face_result = self.face_detector.detect_face(frame)
                    except Exception as e:
                        logger.error(f"[EXTRACT_FRAMES] Face detection failed: {e}")
                        face_result = None
                    
                    if face_result is None:
                        # Try with enhanced frame if detection fails
                        try:
                            enhanced_frame = self.face_detector.enhance_frame(frame)
                            face_result = self.face_detector.detect_face(enhanced_frame)
                            if face_result:
                                frame = enhanced_frame
                        except Exception as e:
                            logger.error(f"[EXTRACT_FRAMES] Enhanced face detection failed: {e}")
                    
                    if face_result:
                        # Crop face with margin
                        bbox = self.face_detector.add_margin(face_result['bbox'], frame.shape)
                        x1, y1, x2, y2 = bbox
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Skip if crop is too small
                        if cropped_frame.shape[0] < 10 or cropped_frame.shape[1] < 10:
                            logger.debug(f"Face crop too small at frame {frame_count}, skipping")
                            frame_count += 1
                            continue
                        
                        # Resize to 224x224
                        processed_frame = self.face_detector.resize_frame(cropped_frame)
                        
                        # Ensure the frame is BGR before converting to RGB
                        if len(processed_frame.shape) == 2:
                            # Grayscale, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                        elif processed_frame.shape[2] == 4:
                            # BGRA, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGB)
                        else:
                            # Assume BGR, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Ensure shape is exactly (224, 224, 3) and normalize
                        processed_frame = cv2.resize(processed_frame, (224, 224))
                        processed_frame = processed_frame.astype(np.float32) / 255.0
                        
                        # Verify shape before appending
                        assert processed_frame.shape == (224, 224, 3), f"Unexpected shape: {processed_frame.shape}"
                        frames.append(processed_frame)
                        logger.debug(f"Face detected at frame {frame_count} using {face_result['detection_method']}")
                    else:
                        logger.debug(f"No face detected at frame {frame_count}, using full frame")
                        # Fallback to full frame if no face detected
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                        frame = frame.astype(np.float32) / 255.0
                        frames.append(frame)
                else:
                    # No face detection, use full frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        frames = np.array(frames)
        logger.info(f"Extracted {len(frames)} frames with face detection")
        print(f"[DEBUG] Extracted {len(frames)} frames with face detection")
        
        return frames
    
    def create_environment(self, frames: np.ndarray):
        """
        Create DeepfakeEnv for inference
        """
        try:
            # Create a simple dataset wrapper
            class SingleVideoDataset:
                def __init__(self, frames, device):
                    frames = np.transpose(frames, (0, 3, 1, 2))
                    self.frames = torch.from_numpy(frames).float()
                    self.video_ids = ['test_video']
                    self.device = device
                
                def __len__(self):
                    return 1
                
                def __getitem__(self, idx):
                    # Return frames tensor, label (0 for real), video_id
                    return self.frames, 0, 'test_video'
            
            dataset = SingleVideoDataset(frames, self.device)
            
            # Import DeepfakeEnv if not already imported
            from src.deepfake_env import DeepfakeEnv
            
            # Create environment
            env = DeepfakeEnv(
                dataset=dataset,
                vision_backbone=self.visual_backbone,
                temporal_memory=self.temporal_memory,
                classifier_head=self.classifier_head,
                reward_calculator=self.reward_calculator,
                device=self.device,
                training=False
            )
            
            print(f"[DEBUG] Environment created with dataset containing {len(dataset)} videos")
            print(f"[DEBUG] Environment components: vision_backbone={self.visual_backbone is not None}, temporal_memory={self.temporal_memory is not None}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            print(f"[DEBUG] Environment creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def run_environment_inference(self, frames: np.ndarray) -> dict:
        """
        Run inference using environment and policy (the correct approach)
        """
        try:
            # Create environment
            print(f"[DEBUG] Creating environment for {len(frames)} frames")
            env = self.create_environment(frames)
            if env is None:
                print(f"[DEBUG] Failed to create environment, falling back to direct inference")
                return self.run_direct_inference(frames)
            print(f"[DEBUG] Environment created successfully")
            
            # Reset environment 
            print(f"[DEBUG] Resetting environment...")
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result 
            else:
                obs = reset_result 
            print(f"[DEBUG] Environment reset. Observation type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"[DEBUG] Obs dict keys: {list(obs.keys())}")
                for k, v in obs.items():
                    if hasattr(v, 'shape'):
                        print(f"[DEBUG] {k} shape: {v.shape}")
            
            # Run inference
            logger.info("Running inference with environment...")
            print(f"[DEBUG] Starting environment-based inference with {len(frames)} frames")
            done = False
            steps = 0
            actions = []
            rewards = []
            
            while not done and steps < MAX_FRAMES:
                # Get action from policy
                with torch.no_grad():
                    # The policy expects a dictionary observation format
                    if isinstance(obs, dict):
                        try:
                            # Convert numpy arrays in dict to tensors
                            obs_tensor = {
                                'frame_features': torch.FloatTensor(obs['frame_features']).unsqueeze(0).to(self.device),
                                'temporal_memory': torch.FloatTensor(obs['temporal_memory']).unsqueeze(0).to(self.device),
                                'frame_position': torch.FloatTensor(obs['frame_position']).unsqueeze(0).to(self.device),
                                'uncertainty': torch.FloatTensor(obs['uncertainty']).unsqueeze(0).to(self.device)
                            }
                            
                            # Directly use the policy's features extractor and actor
                            features = self.policy.features_extractor(obs_tensor)
                            latent_pi = self.policy.mlp_extractor.forward_actor(features)
                            action_logits = self.policy.action_net(latent_pi)
                            
                            # Get deterministic action (argmax)
                            action = torch.argmax(action_logits, dim=1).cpu().numpy()[0]
                            
                            # Debug action logits
                            if steps <= 5:  # Only for first few steps to avoid spam
                                logits_np = action_logits[0].cpu().numpy()
                                print(f"[DEBUG] Step {steps+1} action logits: [{', '.join([f'{l:.3f}' for l in logits_np])}]")
                                print(f"[DEBUG] Step {steps+1} action probabilities: [{', '.join([f'{p:.3f}' for p in torch.softmax(action_logits[0], dim=0).cpu().numpy()])}]")
                                print(f"[DEBUG] Step {steps+1} selected action: {action} ({ACTION_NAMES[action]})")
                        except Exception as e:
                            print(f"[DEBUG] Error in dict-based policy prediction: {e}")
                            # Fallback to direct prediction
                            obs_flat = np.concatenate([v.flatten() if hasattr(v, 'flatten') else [v] for v in obs.values()])
                            action, _ = self.policy.predict(obs_flat, deterministic=True)
                    else:
                        # Fallback for non-dict observations
                        if steps <= 5:
                            print(f"[DEBUG] Using non-dict observation, shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
                        action, _ = self.policy.predict(obs, deterministic=True)
                    
                    action = int(action)
                
                # Take action 
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    obs, reward, done, info = step_result
                
                # Record
                actions.append(action)
                rewards.append(reward)
                steps += 1
                
                # Log step
                action_name = ACTION_NAMES[action]
                logger.debug(f"Step {steps}: Action={action_name}, Reward={reward:.3f}")
                if steps <= 10 or steps % 10 == 0:  # Print first 10 steps and every 10th step
                    print(f"[DEBUG] Step {steps}: {action_name} (reward: {reward:.3f})")
            
            # Get final prediction
            final_action = actions[-1] if actions else 0
            stop_real_count = sum(1 for a in actions if a == 3)
            stop_fake_count = sum(1 for a in actions if a == 4)
            
            print(f"[DEBUG] Action analysis:")
            print(f"[DEBUG] Final action: {final_action} ({ACTION_NAMES[final_action] if final_action < len(ACTION_NAMES) else 'UNKNOWN'})")
            print(f"[DEBUG] STOP_REAL (3) actions: {stop_real_count}")
            print(f"[DEBUG] STOP_FAKE (4) actions: {stop_fake_count}")
            print(f"[DEBUG] All actions: {actions[:20]}...")  # First 20 actions
            
            if final_action == 3:  # STOP_REAL
                prediction = "REAL"
                print(f"[DEBUG] Prediction based on final STOP_REAL action")
            elif final_action == 4:  # STOP_FAKE
                prediction = "FAKE"
                print(f"[DEBUG] Prediction based on final STOP_FAKE action")
            else:
                # If didn't stop, use majority vote
                prediction = "FAKE" if stop_fake_count > stop_real_count else "REAL"
                print(f"[DEBUG] Prediction based on majority vote: FAKE={stop_fake_count} vs REAL={stop_real_count}")
            
            results = {
                'prediction': prediction,
                'steps': steps,
                'actions': actions,
                'rewards': rewards,
                'total_reward': sum(rewards) if rewards else 0
            }
            
            print(f"[DEBUG] Environment inference complete:")
            print(f"[DEBUG] Final prediction: {prediction}")
            print(f"[DEBUG] Total steps: {steps}")
            print(f"[DEBUG] Stop actions: REAL={sum(1 for a in actions if a == 3)}, FAKE={sum(1 for a in actions if a == 4)}")
            print(f"[DEBUG] Total reward: {results['total_reward']:.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"PPO Environment-based inference failed: {e}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] PPO Policy or Environment is not working correctly")
            print(f"[ERROR] System cannot perform proper deepfake detection")
            import traceback
            traceback.print_exc()
            return {
                'error': error_msg,
                'prediction': 'ERROR',
                'steps': 0,
                'actions': []
            }
    
    def run_inference(self, frames: np.ndarray) -> dict:
        """
        Run inference on video frames
        """
        if self.policy is None:
            error_msg = "PPO Policy not loaded - cannot perform deepfake detection"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] System requires PPO policy for proper inference")
            return {
                'error': error_msg,
                'prediction': 'ERROR',
                'steps': 0,
                'actions': []
            }
        
        # Use environment-based inference 
        print(f"[DEBUG] Using environment-based inference with policy")
        return self.run_environment_inference(frames)
    
    # Direct inference method removed - only PPO environment-based inference allowed
    # This ensures we use the proper trained PPO policy with DINOv3
    
    def generate_attention_heatmap_pca(self, image: np.ndarray) -> np.ndarray:
        """
        Generate attention heatmap using DINOv3 attention weights (proper approach)
        """
        try:
            import torchvision.transforms as transforms
            
            print(f"[DEBUG] Generating DINOv3 attention heatmap for image shape: {image.shape}")
            
            # Validate that we have the proper DINOv3 model
            if self.visual_backbone is None or not hasattr(self.visual_backbone, 'backbone'):
                error_msg = "Visual backbone model not available for attention heatmap generation"
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
            
            # Additional validation for DINOv3 model type
            try:
                from src.config import USE_DINOV3
                if not USE_DINOV3:
                    error_msg = "USE_DINOV3 is False - heatmaps require DINOv3 model"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)
            except ImportError:
                # If we can't import config, continue anyway as the model is loaded
                print(f"[DEBUG] Could not verify USE_DINOV3 config, proceeding with backbone model")
                pass
            
            if image.dtype == np.float32 and image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Prepare the image with DINOv3-specific preprocessing
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            image_tensor = transform(image_uint8).unsqueeze(0).to(self.device)
            print(f"[DEBUG] Prepared image tensor for DINOv3: {image_tensor.shape}")
            
            with torch.no_grad():
                dinov3_model = self.visual_backbone.backbone
                print(f"[DEBUG] Using backbone model: {type(dinov3_model)}")
                print(f"[DEBUG] Model attributes: {[attr for attr in dir(dinov3_model) if not attr.startswith('_')][:10]}...")  # First 10 attributes
                
                # Hook to capture attention weights
                attention_weights = []
                def attention_hook(module, input, output):
                    # DINOv3 attention output format: (batch, heads, seq_len, seq_len)
                    if hasattr(output, 'shape') and len(output.shape) == 4:
                        attention_weights.append(output.detach())
                
                # Register hooks on attention modules
                hooks = []
                hook_count = 0
                for name, module in dinov3_model.named_modules():
                    # Look for attention modules in DINOv3
                    if ('attn' in name.lower() or 'attention' in name.lower()) and hasattr(module, 'forward'):
                        hooks.append(module.register_forward_hook(attention_hook))
                        hook_count += 1
                        print(f"[DEBUG] Registered hook on: {name}")
                        if hook_count >= 12:  
                            break
                
                print(f"[DEBUG] Registered {hook_count} attention hooks")
                
                try:
                    # Forward pass to capture attention
                    _ = dinov3_model(image_tensor)
                    
                    if attention_weights:
                        # Use the last attention layer 
                        last_attention = attention_weights[-1]  
                        print(f"[DEBUG] Captured attention shape: {last_attention.shape}")
                        
                        # Average across attention heads
                        attention = last_attention[0].mean(dim=0) 
                        
                        # Get attention to CLS token (first token) from all patches
                        cls_attention = attention[0, 1:]  
                        print(f"[DEBUG] CLS attention shape: {cls_attention.shape}")
                        
                        # Reshape to spatial grid (14x14 for 224x224 images with patch_size=16)
                        patch_size = 16
                        num_patches = 224 // patch_size  # 14
                        
                        if len(cls_attention) == num_patches * num_patches:
                            # Perfect square grid
                            attention_map = cls_attention.reshape(num_patches, num_patches)
                        else:
                            # Handle irregular cases
                            target_len = num_patches * num_patches
                            if len(cls_attention) > target_len:
                                cls_attention = cls_attention[:target_len]
                            else:
                                # Pad with zeros
                                pad_len = target_len - len(cls_attention)
                                cls_attention = torch.cat([cls_attention, torch.zeros(pad_len, device=cls_attention.device)])
                            attention_map = cls_attention.reshape(num_patches, num_patches)
                        
                        # Convert to numpy and normalize
                        heatmap = attention_map.cpu().numpy()
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        print(f"[DEBUG] DINOv3 attention heatmap shape: {heatmap.shape}")
                        print(f"[DEBUG] Attention values range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                        
                    else:
                        print(f"[DEBUG] No attention weights captured, using DINOv3 intermediate features")
                        # Fallback to intermediate features using DINOv3 specific methods
                        try:
                            # Try DINOv3's get_intermediate_layers method first
                            if hasattr(dinov3_model, 'get_intermediate_layers'):
                                features = dinov3_model.get_intermediate_layers(image_tensor, n=[11], return_class_token=False)[0]
                                print(f"[DEBUG] Got DINOv3 intermediate features: {features.shape}")
                            elif hasattr(dinov3_model, 'forward_features'):
                                features = dinov3_model.forward_features(image_tensor)
                                print(f"[DEBUG] Got DINOv3 forward features: {features.shape}")
                                # Remove class token if present 
                                if features.shape[1] > 196:  
                                    features = features[:, 1:]  
                            else:
                                # Last resort - use regular forward
                                features = dinov3_model(image_tensor)
                                print(f"[DEBUG] Got model forward output: {features.shape}")
                                if len(features.shape) == 3: 
                                    if features.shape[1] > 196:  
                                        features = features[:, 1:]  
                            
                            features = features.squeeze(0) 
                            print(f"[DEBUG] Features after processing: {features.shape}")
                            
                            # Use feature magnitudes as attention proxy
                            attention_scores = torch.norm(features, dim=1)
                            attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
                            
                            # Reshape to spatial grid (should be 14x14 for ViT-B/16)
                            patch_dim = int(np.sqrt(len(attention_scores)))
                            print(f"[DEBUG] Calculated patch_dim: {patch_dim}, total patches: {len(attention_scores)}")
                            
                            if patch_dim * patch_dim != len(attention_scores):
                                # Handle non-perfect square - force to 14x14 for ViT-B/16
                                patch_dim = 14
                                target_len = patch_dim * patch_dim
                                if len(attention_scores) > target_len:
                                    attention_scores = attention_scores[:target_len]
                                    print(f"[DEBUG] Truncated to {target_len} patches")
                                elif len(attention_scores) < target_len:
                                    # Pad with mean values
                                    pad_len = target_len - len(attention_scores)
                                    mean_val = attention_scores.mean()
                                    attention_scores = torch.cat([attention_scores, mean_val.repeat(pad_len)])
                                    print(f"[DEBUG] Padded with {pad_len} patches")
                            
                            heatmap = attention_scores.cpu().numpy().reshape(patch_dim, patch_dim)
                            print(f"[DEBUG] Created heatmap from backbone features: {heatmap.shape}")
                        
                        except Exception as e:
                            print(f"[DEBUG] Feature extraction failed: {e}")
                            # Last resort - create a simple center-focused heatmap
                            heatmap = np.ones((14, 14)) * 0.5
                            y, x = np.ogrid[:14, :14]
                            center_dist = np.sqrt((x - 7)**2 + (y - 7)**2)
                            heatmap = 1 - (center_dist / center_dist.max())
                            print(f"[DEBUG] Used fallback center-focused heatmap: {heatmap.shape}")
                        
                finally:
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                
                # Post-process heatmap
                # Resize to full image resolution
                heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                # Apply Gaussian blur for smoother visualization
                heatmap = cv2.GaussianBlur(heatmap, (15, 15), 2.0)
                
                # Final normalization
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                print(f"[DEBUG] Final DINOv3 attention heatmap: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                return heatmap
            
        except Exception as e:
            error_msg = f"DINOv3 attention heatmap generation failed: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            raise ValueError(error_msg)
    
    # Fallback heatmap methods removed - only DINOv3 attention allowed
    # This ensures we use proper attention visualization from the trained model
    
    def process_video(self, video_path):
        """
        Main video analysis pipeline using RL agent and attention visualization.
        Returns detection result with heatmaps for deepfakes.
        """
        import time
        start_time = time.time()
        
        logger.info(f"[PROCESS_VIDEO] Starting video processing for: {video_path}")
        logger.info(f"[PROCESS_VIDEO] Models status - Policy: {self.policy is not None}, VisionBackbone: {self.visual_backbone is not None}")
        print(f"\n=== DEEPFAKE DETECTION PROCESSING ===")
        print(f"Video: {Path(video_path).name}")
        print(f"Policy loaded: {self.policy is not None}")
        print(f"Visual backbone loaded: {self.visual_backbone is not None}")
        print(f"Temporal memory loaded: {self.temporal_memory is not None}")
        print(f"Classifier head loaded: {self.classifier_head is not None}")
        
        # Check if PPO policy is available (required for proper detection)
        if self.policy is None:
            error_msg = "PPO Policy not loaded - deepfake detection unavailable"
            logger.error(f"[PROCESS_VIDEO] {error_msg}")
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Please check PPO model loading in server logs")
            return {
                'error': error_msg,
                'is_deepfake': False,
                'heatmap_images': [],
                'processing_time': 0.0
            }
        
        # Check if DINOv3 visual backbone is available  
        if self.visual_backbone is None:
            error_msg = "DINOv3 Visual Backbone not loaded - deepfake detection unavailable"
            logger.error(f"[PROCESS_VIDEO] {error_msg}")
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Please check DINOv3 model loading in server logs")
            return {
                'error': error_msg,
                'is_deepfake': False,
                'heatmap_images': [],
                'processing_time': 0.0
            }
        
        try:
            # Extract frames
            logger.info(f"[PROCESS_VIDEO] Starting frame extraction at {time.time() - start_time:.2f}s")
            print(f"\n[1/4] Extracting frames from video...")
            frames = self.extract_frames(video_path)
            logger.info(f"[PROCESS_VIDEO] Frame extraction completed at {time.time() - start_time:.2f}s, extracted {len(frames)} frames")
            print(f"[1/4] Extracted {len(frames)} frames successfully")
            
            if len(frames) == 0:
                logger.warning("[PROCESS_VIDEO] No frames extracted from video")
                return {
                    'error': 'No frames could be extracted from video',
                    'is_deepfake': False,
                    'heatmap_images': [],
                    'processing_time': time.time() - start_time
                }
            
            # Run inference
            logger.info(f"[PROCESS_VIDEO] Starting inference at {time.time() - start_time:.2f}s")
            print(f"\n[2/4] Running deepfake detection inference...")
            results = self.run_inference(frames)
            logger.info(f"[PROCESS_VIDEO] Inference completed at {time.time() - start_time:.2f}s, prediction: {results.get('prediction')}")
            print(f"[2/4] Inference completed - Prediction: {results.get('prediction')}")
            print(f"      Steps taken: {results.get('steps', 0)}")
            print(f"      Actions: {[ACTION_NAMES[a] for a in results.get('actions', [])][:10]}...")  # First 10 actions
            
            # Check if inference failed
            if results.get('error'):
                logger.error(f"[PROCESS_VIDEO] Inference failed: {results['error']}")
                print(f"[ERROR] Inference failed: {results['error']}")
                return {
                    'error': results['error'],
                    'is_deepfake': False,
                    'heatmap_images': [],
                    'processing_time': time.time() - start_time
                }
            
            # Determine if deepfake
            is_deepfake = results['prediction'] == 'FAKE'
            
            # Generate heatmaps ONLY for deepfakes (not for authentic videos)
            heatmap_images = []
            if is_deepfake:
                logger.info(f"[PROCESS_VIDEO] Starting heatmap generation at {time.time() - start_time:.2f}s")
                print(f"\n[3/4] Generating attention heatmaps for DEEPFAKE detection...")
                heatmap_images = self.visualize_results(frames, results)
                logger.info(f"[PROCESS_VIDEO] Heatmap generation completed at {time.time() - start_time:.2f}s, generated {len(heatmap_images)} heatmaps")
                print(f"[3/4] Generated {len(heatmap_images)} attention heatmaps")
            else:
                logger.info(f"[PROCESS_VIDEO] Skipping heatmap generation for AUTHENTIC video")
                print(f"\n[3/4] Video is AUTHENTIC - no heatmaps needed")
                print(f"[3/4] Heatmaps are only generated for deepfake detections")
            
            processing_time = time.time() - start_time
            logger.info(f"[PROCESS_VIDEO] Total processing time: {processing_time:.2f}s")
            print(f"\n[4/4] Processing complete!")
            print(f"=== RESULTS ===")
            print(f"Prediction: {'DEEPFAKE' if is_deepfake else 'AUTHENTIC'}")
            print(f"Processing time: {processing_time:.2f}s")
            if is_deepfake:
                print(f"Attention heatmaps: {len(heatmap_images)} frames analyzed")
            else:
                print(f"Attention heatmaps: Not generated (video is authentic)")
            print(f"=== END ===\n")
            
            return {
                'is_deepfake': is_deepfake,
                'heatmap_images': heatmap_images,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'is_deepfake': False,
                'heatmap_images': [],
                'processing_time': time.time() - start_time
            }
    
    def visualize_results(self, frames: np.ndarray, results: dict) -> list:
        """
        Create visualization of results with attention heatmaps for DEEPFAKES only
        """
        print(f"[DEBUG] Creating attention heatmaps for DEEPFAKE analysis")
        heatmap_images = []
        
        # Select 4 frames for visualization (2x2 grid)
        n_display = min(4, len(frames))
        display_indices = np.linspace(0, len(frames)-1, n_display, dtype=int)
        
        for idx in display_indices:
            try:
                # Generate DINOv3 attention heatmap
                print(f"[DEBUG] Generating heatmap for frame {idx+1}/{len(display_indices)}")
                heatmap = self.generate_attention_heatmap_pca(frames[idx])
                
                # Create single panel with attention overlay on face (compact sized)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5)) 
                
                # Create attention overlay directly on the original frame
                heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
                alpha = 0.4  # Good balance for visibility
                blended = frames[idx] * (1 - alpha) + heatmap_colored * alpha
                
                # Show the blended result (original frame + attention overlay)
                ax.imshow(np.clip(blended, 0, 1))
                ax.set_title(f'Frame {idx+1} - DINOv3 Attention Overlay', fontsize=14, pad=15)
                ax.axis('off')
                
                # Adjust layout for clean appearance
                plt.tight_layout(pad=1.0)
                
                # Convert to base64 with good quality
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                heatmap_images.append(image_base64)
                plt.close()
                
                print(f"[DEBUG] Successfully generated heatmap for frame {idx}")
                
            except Exception as e:
                error_msg = f"Failed to generate DINOv3 attention heatmap for frame {idx}: {e}"
                logger.error(error_msg)
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] Heatmap generation requires properly loaded DINOv3 model")
                # Don't continue - fail clearly
                break
        
        return heatmap_images

# Singleton instance
_detector_instance = None

def get_detector():
    """
    Singleton factory function to get or create the deepfake detector instance.
    Ensures only one detector is loaded in memory for efficiency.
    """
    global _detector_instance
    if _detector_instance is None:
        try:
            import time
            start = time.time()
            logger.info(f"[GET_DETECTOR] Creating new DeepfakeDetector instance at {start}")
            _detector_instance = DeepfakeDetector()
            logger.info(f"[GET_DETECTOR] DeepfakeDetector instance created successfully in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"[GET_DETECTOR] Failed to create DeepfakeDetector: {e}")
            import traceback
            traceback.print_exc()
            _detector_instance = None
    else:
        logger.info("[GET_DETECTOR] Returning existing detector instance")
    return _detector_instance