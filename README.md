# EAGER - Efficient Adaptive Gated Evidence Retrieval for Deepfake Detection using Reinforcement Learning

## Overview

EAGER is an advanced deepfake detection system that leverages reinforcement learning with DINOv3 vision transformer and custom Group Relative Policy Optimization (GRPO). The framework achieves high accuracy while processing only 20% of video frames, deployed through a Django web interface for real-time detection.


## Demo

Watch the demo video below to see the application in action:

https://github.com/user-attachments/assets/12063873-db85-49fc-8e0b-eb6f60f00c14

## Model Architecture

- **Feature Extraction**: DINOv3 Vision Transformer (ViT-B/16)
- **Temporal Processing**: Bidirectional LSTM (3 layers, 512 hidden units)
- **Decision Making**: Reinforcement Learning Agent with PPO and GRPO
- **Uncertainty Estimation**: Bayesian inference via Monte Carlo Dropout
  
## Features

- **Intelligent Frame Selection**: Reinforcement learning agent analyzes only essential frames, reducing processing time to 10 seconds per video
- **User-Friendly Interface**: User-friendly Django application for video upload and analysis
- **Three-Phase Training**: CProgressive optimization through supervised warm-start, PPO-LSTM, and GRPO fine-tuning
- **Datasets Used**: FaceForensics++, Celeb-DF v2, Celeb-DF, DeeperForensics 1.0 and DFD (Google/Jigsaw) 
- **Attention Visualization**: DINOv3 heatmaps showing facial regions contributing to detection decisions
- **GPU Acceleration**: Utilizes PyTorch-GPU for faster processing
  

## Documentation

If you want to read my report, you can find my dissertation paper [here](https://github.com/byrm-tsn/rl_deepfake_detection/blob/main/DOCUMENTATION/Dissertation_Paper.pdf) and the reflective essay [here](https://github.com/byrm-tsn/rl_deepfake_detection/blob/main/DOCUMENTATION/Reflective%20Essay.pdf).

## Usage

- **Upload a video file through the web interface**
- **Wait for processing (approximately 10-15 seconds)**
- **View detection results**
- **Examine DINOv3 attention heatmaps showing detection reasoning**


## Installation

### Prerequisites

- **Python 3.10+**
- **Django 4.2+**
- **PyTorch 2.0+**
- **TorchRL**
- **CUDA 11.8+**
- **cuDNN 8.6+**

### Technologies Used

[![My Skills](https://skillicons.dev/icons?i=vscode,github,django,js,html,css,git,opencv,py,sqlite,pytorch,sklearn)](https://skillicons.dev)

## Clone the Repository
### You should train the model with the dataset you have, the model and dataset are not avaibale in the repository!)
```bash
git clone https://https://github.com/byrm-tsn/rl_deepfake_detection
cd rl_deepdake_detection
pip install -r requirements.txt
python manage.py runserver

