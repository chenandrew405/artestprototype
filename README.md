# AR Design Prototype - Camera Capture

This project contains two camera capture prototypes for accessing a device's camera, taking photos, and saving them.

## Setup

### 1. Virtual Environment

Create and activate the virtual environment:

```bash
# The venv is already created, just activate it:
source venv/bin/activate

# To deactivate when done:
deactivate
```

### 2. Install Dependencies

```bash
# If starting fresh:
pip install -r requirements.txt
```

## Usage

### Option 1: Web-based Camera (index.html)

Browser-based solution that works on desktop and mobile devices.

**Run:**
```bash
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

**Features:**
- Live camera preview
- Click "Take Photo" to capture
- Photos automatically download with timestamps
- Shows preview of captured photos
- No installation required (uses browser APIs)

### Option 2: Python Face Detection with Database (face_detection.py)

Real-time face detection using OpenCV with SQLite database integration.

**Run:**
```bash
source venv/bin/activate
python face_detection.py
```

**Controls:**
- **q** or **ESC** - Quit the application
- **s** or **SPACE** - Manually save current detection to database with image

**Features:**
- Real-time face detection with green bounding boxes
- Automatic detection logging to SQLite database
- Saves face coordinates, timestamps, and metadata
- FPS counter and face count overlay
- Manual image capture of detections
- Database viewer to see all detections
- Total face count tracker

**View Database:**
```bash
python view_database.py
```

**Configuration:**
Edit `config.py` to customize:
- Camera settings (resolution, index)
- Detection sensitivity
- Auto-save behavior
- Output directories
- Database location

### Option 3: Python Desktop App (main.py)

Simple desktop application using OpenCV for direct camera access.

**Run:**
```bash
source venv/bin/activate
python main.py
```

**Controls:**
- **SPACE** - Take a photo
- **ESC** - Exit the application

**Features:**
- Live camera feed in window
- Photos saved to `photos/` directory
- Automatic timestamps
- Visual flash effect when capturing

## Files

- `index.html` - Web-based camera capture with TensorFlow.js face detection
- `face_detection.py` - Python face detection with database integration
- `config.py` - Configuration settings for face detection
- `view_database.py` - Database viewer utility
- `main.py` - Simple Python OpenCV camera capture
- `requirements.txt` - Python dependencies
- `venv/` - Python virtual environment
- `photos/` - Directory where main.py saves photos (created automatically)
- `detected_faces/` - Directory where face_detection.py saves images (created automatically)
- `face_detection.db` - SQLite database (created automatically)
