"""
EAGER Training Pipeline Entry Point
Main script to run the complete multi-phase EAGER training.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')


sys.path.append(str(Path(__file__).parent / 'src'))


from src.config import *
from src.data_loader import create_data_loaders
from src.feature_extractor import FeatureExtractorModule, FrozenEvaluator
from src.train import run_complete_training, Phase1WarmStartTrainer, Phase2RLTrainer
from src.evaluate import EagerEvaluator
from src.deepfake_env import DeepfakeEnv
from src.utils import set_random_seeds, get_system_info, ExperimentTracker


def setup_logging(log_level: str = 'INFO'):
    """
    Setup logging configuration with proper flushing.
    
    Args:
        log_level: Logging level
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'training_{timestamp}.log'
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler with immediate flush
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Create handlers
    file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial message
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    return log_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EAGER (Evidence-Acquiring Recurrent) Training Pipeline'
    )
    
    # Training phases
    parser.add_argument(
        '--phase',
        type=str,
        choices=['all', 'warmstart', 'rl', 'grpo'],
        default='all',
        help='Which training phase(s) to run'
    )
    
    # Data settings
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(DATA_DIR),
        help='Path to processed video dataset'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=WARMSTART_BATCH_SIZE,
        help='Batch size for supervised training'
    )
    
    # Model settings
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--pretrained-vit',
        type=str,
        default=VIT_MODEL_NAME,
        help='Pretrained ViT model name'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--warmstart-epochs',
        type=int,
        default=WARMSTART_EPOCHS,
        help='Number of epochs for warm-start phase'
    )
    
    parser.add_argument(
        '--rl-timesteps',
        type=int,
        default=TOTAL_TIMESTEPS,
        help='Total timesteps for RL training'
    )
    
    # Phase 3 GRPO arguments
    parser.add_argument(
        '--grpo-iterations',
        type=int,
        default=GRPO_ITERATIONS,  # Use config.py value (30)
        help='Number of GRPO iterations'
    )
    
    parser.add_argument(
        '--grpo-group-size',
        type=int,
        default=GRPO_GROUP_SIZE,  # Use config.py value (16)
        help='Group size for relative reward computation'
    )
    
    parser.add_argument(
        '--grpo-lr',
        type=float,
        default=GRPO_LEARNING_RATE,  # Use config.py value (5e-6)
        help='Learning rate for GRPO fine-tuning'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=PPO_LEARNING_RATE,
        help='Learning rate for RL training'
    )
    
    # Hardware settings
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Experiment settings
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=EXPERIMENT_NAME,
        help='Name of the experiment'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help='Random seed for reproducibility'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=EVAL_FREQ,
        help='Evaluation frequency in timesteps'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=SAVE_FREQ,
        help='Checkpoint save frequency'
    )
    
    # Logging settings
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )
    
    # Debug settings
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced dataset'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actual training'
    )
    
    return parser.parse_args()


def run_phase1_warmstart(args, experiment_tracker):
    """
    Run Phase 1: Supervised warm-start training.
    
    Args:
        args: Command line arguments
        experiment_tracker: Experiment tracker
        
    Returns:
        frozen_evaluator: Frozen evaluator for reward calculation
    """
    logging.info("\n" + "=" * 80)
    logging.info("🚀 PHASE 1: Supervised Warm-Start Training")
    logging.info("=" * 80)
    logging.info(f"  Epochs: {args.warmstart_epochs}")
    logging.info(f"  Learning Rate: {args.learning_rate}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  Device: {args.device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        validate=True
    )
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractorModule().to(args.device)
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Phase1WarmStartTrainer(
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Train
    if not args.dry_run:
        frozen_evaluator = trainer.train(num_epochs=args.warmstart_epochs)
        
        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / 'phase1_final.pth'
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Log metrics
        experiment_tracker.log_metrics({
            'phase1_best_acc': trainer.best_val_acc,
            'phase1_final_loss': trainer.train_losses[-1] if trainer.train_losses else 0
        }, step=1)
    else:
        logging.info("Dry run - skipping actual training")
        frozen_evaluator = None
    
    logging.info("Phase 1 completed successfully!")
    return frozen_evaluator, feature_extractor, train_loader.dataset, val_loader.dataset


def run_phase2_rl(args, feature_extractor, train_dataset, val_dataset, 
                  frozen_evaluator, experiment_tracker):
    """
    Run Phase 2: PPO-LSTM Reinforcement Learning.
    
    Args:
        args: Command line arguments
        feature_extractor: Trained feature extractor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        frozen_evaluator: Frozen evaluator for rewards
        experiment_tracker: Experiment tracker
        
    Returns:
        ppo_model: Trained PPO model
    """
    logging.info("\n" + "=" * 80)
    logging.info("🤖 PHASE 2: PPO-LSTM Reinforcement Learning")
    logging.info("=" * 80)
    logging.info(f"  Timesteps: {args.rl_timesteps}")
    logging.info(f"  Learning Rate: {PPO_LEARNING_RATE}")
    logging.info(f"  Batch Size: {PPO_BATCH_SIZE}")
    logging.info(f"  Actions: {', '.join(ACTION_NAMES)}")
    # Log uncertainty configuration
    if USE_BAYESIAN_UNCERTAINTY:
        logging.info(f"  Uncertainty: Bayesian (MCDO, {MC_SAMPLES} samples)")
    else:
        logging.info(f"  Uncertainty: Deterministic (1 - Confidence)")

    
    # Extract components
    vision_backbone = feature_extractor.vision_backbone
    temporal_memory = feature_extractor.temporal_memory
    classifier_head = feature_extractor.classifier_head
    
    
    # Freeze vision backbone
    vision_backbone.eval()
    for param in vision_backbone.parameters():
        param.requires_grad = False

    # Freeze Temporal Memory (LSTM)
    temporal_memory.eval()
    for param in temporal_memory.parameters():
        param.requires_grad = False

    # Freeze Classifier Head
    classifier_head.eval()
    for param in classifier_head.parameters():
        param.requires_grad = False

    logging.info("Phase 1 components (Vision, LSTM, Classifier) frozen and set to eval mode for Phase 2.")
    
    # Create trainer
    trainer = Phase2RLTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vision_backbone=vision_backbone,
        temporal_memory=temporal_memory,
        classifier_head=classifier_head,
        frozen_evaluator=frozen_evaluator,
        device=args.device
    )
    
    # Train
    if not args.dry_run:
        ppo_model = trainer.train(
            total_timesteps=args.rl_timesteps,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            experiment_tracker=experiment_tracker
        )
        
        # Log final summary metrics
        experiment_tracker.log_metrics({
            'phase2_timesteps': args.rl_timesteps
        }, step=args.rl_timesteps)
    else:
        logging.info("Dry run - skipping actual training")
        ppo_model = None
    
    logging.info("Phase 2 completed successfully!")
    return ppo_model, trainer.train_env


def run_phase3_grpo(args, ppo_model, train_env, val_env, experiment_tracker):
    """
    Run Phase 3: GRPO Fine-tuning.
    
    Args:
        args: Command line arguments
        ppo_model: Trained PPO model from Phase 2
        train_env: Training environment
        val_env: Validation environment
        experiment_tracker: Experiment tracker
        
    Returns:
        refined_model: GRPO fine-tuned model
    """
    logging.info("\n" + "=" * 80)
    logging.info("🎯 PHASE 3: GRPO Fine-tuning")
    logging.info("=" * 80)
    logging.info(f"  Iterations: {args.grpo_iterations}")
    logging.info(f"  Group Size: {args.grpo_group_size}")
    logging.info(f"  Learning Rate: {args.grpo_lr}")
    logging.info(f"  Device: {args.device}")
    
    # Import GRPO trainer and evaluator
    from src.grpo_trainer import Phase3GRPOTrainer
    from src.grpo_evaluation import GRPOEvaluator
    
    # Create GRPO trainer
    grpo_trainer = Phase3GRPOTrainer(
        ppo_model=ppo_model,
        train_env=train_env,
        val_env=val_env,
        device=args.device,
        learning_rate=args.grpo_lr,
        group_size=args.grpo_group_size,
        buffer_size=10000,
        gradient_clip=0.5
    )
    
    # Load checkpoint if resuming GRPO
    if args.phase == 'grpo' and args.checkpoint:
        grpo_checkpoint_path = Path(args.checkpoint)
        if grpo_checkpoint_path.exists():
            logging.info(f"Loading GRPO checkpoint from {grpo_checkpoint_path}")
            start_iteration = grpo_trainer.load_checkpoint(grpo_checkpoint_path)
        else:
            start_iteration = 0
    else:
        start_iteration = 0
    
    # Train with GRPO
    if not args.dry_run:
        refined_model = grpo_trainer.train(
            num_iterations=args.grpo_iterations - start_iteration,
            episodes_per_iteration=GRPO_EPISODES_PER_ITER,  # Use config value (32)
            updates_per_iteration=GRPO_UPDATES_PER_ITER,  # Use config value (2)
            eval_freq=args.eval_freq // 100,  # Adjust for iterations instead of timesteps
            save_freq=args.save_freq // 100,
            checkpoint_dir=CHECKPOINT_DIR / 'phase3_grpo'
        )
        
        # Log metrics
        experiment_tracker.log_metrics({
            'phase3_iterations': args.grpo_iterations,
            'phase3_group_size': args.grpo_group_size,
            'phase3_lr': args.grpo_lr
        }, step=args.rl_timesteps + args.grpo_iterations)
        
        # Run comprehensive GRPO evaluation
        logging.info("\n" + "=" * 80)
        logging.info("📊 Running comprehensive GRPO evaluation...")
        logging.info("=" * 80)
        
        evaluator = GRPOEvaluator(
            grpo_model=refined_model,
            phase2_model=ppo_model,
            env=val_env,
            device=args.device,
            output_dir=RESULTS_DIR / 'phase3_grpo'
        )
        
        # Generate all evaluation visualizations
        evaluator.run_full_evaluation(
            num_episodes=50,
            training_history=grpo_trainer.training_history if hasattr(grpo_trainer, 'training_history') else None
        )
        
        logging.info("✅ GRPO evaluation completed!")
    else:
        logging.info("Dry run - skipping actual training")
        refined_model = ppo_model
    
    logging.info("Phase 3 GRPO completed successfully!")
    return refined_model

def evaluate_model(args, ppo_model, test_env):
    """
    Evaluate the trained model.
    
    Args:
        args: Command line arguments
        ppo_model: Trained PPO model (or GRPO refined model)
        test_env: Test environment
    """
    logging.info("\n" + "=" * 80)
    logging.info("📊 FINAL EVALUATION")
    logging.info("=" * 80)
    logging.info(f"  Test Episodes: 100")
    logging.info(f"  Deterministic: True")
    logging.info(f"  Device: {args.device}")
    
    # Create evaluator
    evaluator = EagerEvaluator(
        model=ppo_model,
        test_env=test_env,
        explanation_head=None,
        saliency_head=None,
        device=args.device
    )
    
    # Evaluate
    if not args.dry_run:
        results = evaluator.evaluate_dataset(num_episodes=100, deterministic=True)
        
        # Generate report
        report_path = RESULTS_DIR / 'evaluation_report.txt'
        report = evaluator.generate_report(results, save_path=report_path)
        logging.info("\n" + report)
        
        # Generate plots
        plot_dir = RESULTS_DIR / 'plots'
        plot_dir.mkdir(exist_ok=True)
        evaluator.plot_results(results, save_dir=plot_dir)
        
        # Save results
        results_path = RESULTS_DIR / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"Results saved to {RESULTS_DIR}")
    else:
        logging.info("Dry run - skipping evaluation")


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_file = setup_logging(args.log_level)
    logging.info(f"Logging to {log_file}")
    
    # Log system info
    system_info = get_system_info()
    logging.info("System Information:")
    for key, value in system_info.items():
        logging.info(f"  {key}: {value}")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker(Path('experiments'))
    experiment_tracker.log_config(vars(args))
    
    # Print configuration
    logging.info("Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    
    try:
        # Run training phases based on arguments
        if args.phase in ['all', 'warmstart']:
            frozen_evaluator, feature_extractor, train_dataset, val_dataset = \
                run_phase1_warmstart(args, experiment_tracker)
        
        if args.phase in ['all', 'rl']:
            if args.phase == 'rl':
                # Load from checkpoint for standalone RL
                logging.info("Loading Phase 1 checkpoint for RL training...")
                checkpoint_path = CHECKPOINT_DIR / 'phase1_final.pth'
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Phase 1 checkpoint not found at {checkpoint_path}")
                
                # Load components
                feature_extractor = FeatureExtractorModule().to(args.device)
                checkpoint = torch.load(checkpoint_path, map_location=args.device)
                feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                
                # Create data loaders
                train_loader, val_loader, _ = create_data_loaders(
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    validate=True
                )
                train_dataset = train_loader.dataset
                val_dataset = val_loader.dataset
                
                # Create frozen evaluator
                frozen_evaluator = FrozenEvaluator(feature_extractor.classifier_head)
            
            ppo_model, train_env = run_phase2_rl(
                args, feature_extractor, train_dataset, val_dataset,
                frozen_evaluator, experiment_tracker
            )
        
        # Phase 3: GRPO fine-tuning
        if args.phase in ['all', 'grpo']:
            if args.phase == 'grpo':
                # Load from Phase 2 checkpoint for standalone GRPO
                logging.info("Loading Phase 2 checkpoint for GRPO training...")
                # Try to find the best PPO model
                ppo_checkpoint_path = FINAL_MODEL_DIR / 'ppo_eager_final.zip'
                if not ppo_checkpoint_path.exists():
                    # Try without .zip extension (SB3 adds it automatically)
                    ppo_checkpoint_path = FINAL_MODEL_DIR / 'ppo_eager_final'
                    if not (ppo_checkpoint_path.with_suffix('.zip')).exists():
                        # Try in checkpoints folder
                        ppo_checkpoint_path = CHECKPOINT_DIR / 'ppo_eager_2000000_steps.zip'
                        if not ppo_checkpoint_path.exists():
                            # List available models for debugging
                            available_models = list(CHECKPOINT_DIR.glob("*.zip")) + list(FINAL_MODEL_DIR.glob("*.zip"))
                            logging.error(f"Available models: {available_models}")
                            raise FileNotFoundError(f"Phase 2 checkpoint not found. Looked in: {CHECKPOINT_DIR} and {FINAL_MODEL_DIR}")
                
                # Load PPO model
                from stable_baselines3 import PPO
                ppo_model = PPO.load(ppo_checkpoint_path)
                
                # Recreate environments
                # Load components for environments
                feature_extractor = FeatureExtractorModule().to(args.device)
                phase1_checkpoint = CHECKPOINT_DIR / 'phase1_final.pth'
                checkpoint = torch.load(phase1_checkpoint, map_location=args.device)
                feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                
                # Create data loaders
                train_loader, val_loader, _ = create_data_loaders(
                    batch_size=1,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    validate=True
                )
                
                from src.reward_system import RewardCalculator
                frozen_evaluator = FrozenEvaluator(feature_extractor.classifier_head)
                reward_calculator = RewardCalculator(frozen_evaluator)
                
                # Create environments for GRPO
                train_env = DeepfakeEnv(
                    dataset=train_loader.dataset,
                    vision_backbone=feature_extractor.vision_backbone,
                    temporal_memory=feature_extractor.temporal_memory,
                    classifier_head=feature_extractor.classifier_head,
                    reward_calculator=reward_calculator,
                    device=args.device,
                    training=True
                )
                
                val_env = DeepfakeEnv(
                    dataset=val_loader.dataset,
                    vision_backbone=feature_extractor.vision_backbone,
                    temporal_memory=feature_extractor.temporal_memory,
                    classifier_head=feature_extractor.classifier_head,
                    reward_calculator=reward_calculator,
                    device=args.device,
                    training=False
                )
            
            # Create validation environment if continuing from Phase 2
            if args.phase == 'all':
                # Use training environment for validation in sequential training
                val_env = train_env
            
            # Run GRPO fine-tuning
            refined_model = run_phase3_grpo(args, ppo_model, train_env, val_env, experiment_tracker)
            ppo_model = refined_model  # Update model reference for evaluation
        
        # Final evaluation
        if args.phase == 'all' and not args.dry_run:
            # Create test environment
            _, _, test_loader = create_data_loaders(
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=True,
                validate=True
            )
            
            # Ensure feature extractor components are in eval mode for the test env
            feature_extractor.eval()

            from src.reward_system import RewardCalculator
            reward_calculator = RewardCalculator(frozen_evaluator)
            
            test_env = DeepfakeEnv(
                dataset=test_loader.dataset,
                vision_backbone=feature_extractor.vision_backbone,
                temporal_memory=feature_extractor.temporal_memory,
                classifier_head=feature_extractor.classifier_head,
                reward_calculator=reward_calculator,
                device=args.device,
                training=False # Set environment to evaluation mode
            )
            
            evaluate_model(args, ppo_model, test_env)
        
        # Finalize experiment
        experiment_tracker.finalize()
        
        logging.info("\n" + "=" * 80)
        logging.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"  Experiment: {args.experiment_name}")
        logging.info(f"  Total Time: Check log timestamps")
        logging.info(f"  Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if args.tensorboard:
            logging.info("TensorBoard logs saved to tensorboard/")


if __name__ == "__main__":
    main()