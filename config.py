#!/usr/bin/env python3
"""
Configuration file for face detection application
"""

import os

# Database Configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'face_detection.db')

# Camera Configuration
CAMERA_INDEX = 0  # 0 for default camera, 1 for external camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Face Detection Configuration
# OpenCV DNN model (Res10 SSD) configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DNN_PROTOTXT_PATH = os.path.join(MODEL_DIR, 'deploy.prototxt')
DNN_MODEL_PATH = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Official OpenCV model sources
DNN_PROTOTXT_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
DNN_MODEL_URL = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205/res10_300x300_ssd_iter_140000_fp16.caffemodel'

# Auto-download model files if missing
AUTO_DOWNLOAD_DNN_MODELS = True

# DNN inference tuning
DNN_INPUT_SIZE = (300, 300)
DNN_MEAN_VALUES = (104.0, 177.0, 123.0)
DNN_CONFIDENCE_THRESHOLD = 0.6
DNN_MIN_FACE_SIZE = (40, 40)

# Backward-compatible minimum size key used elsewhere in the app
MIN_SIZE = DNN_MIN_FACE_SIZE

DETECTION_COLOR = (0, 255, 0)  # Green color for bounding boxes (BGR format)
BOX_THICKNESS = 2

# Storage Configuration
SAVE_DETECTIONS = True  # Save detected faces to database
SAVE_IMAGES = True  # Save face images to disk
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'detected_faces')

# Create output directory if it doesn't exist
if SAVE_IMAGES and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Display Configuration
WINDOW_NAME = "Face Detection - Press 'q' to quit, 's' to save, 'SPACE' to capture"
SHOW_FPS = True
SHOW_FACE_COUNT = True

# Database Schema
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    num_faces INTEGER,
    image_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    x INTEGER,
    y INTEGER,
    width INTEGER,
    height INTEGER,
    confidence REAL,
    FOREIGN KEY (detection_id) REFERENCES detections(id)
);
"""
