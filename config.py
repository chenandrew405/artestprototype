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
# OpenCV Haar Cascade will be loaded in face_detection.py
SCALE_FACTOR = 1.1  # How much the image size is reduced at each image scale
MIN_NEIGHBORS = 5  # How many neighbors each candidate rectangle should have
MIN_SIZE = (30, 30)  # Minimum possible object size
DETECTION_COLOR = (0, 255, 0)  # Green color for bounding boxes (BGR format)
BOX_THICKNESS = 2

# Storage Configuration
SAVE_DETECTIONS = True  # Save detected faces to database
SAVE_IMAGES = True  # Save face images to disk
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'detected_faces')

# Create output directory if it doesn't exist
if SAVE_IMAGES and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
