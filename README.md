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

### Option 2: Python Desktop App (main.py)

Desktop application using OpenCV for direct camera access.

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

- `index.html` - Web-based camera capture
- `main.py` - Python OpenCV camera capture
- `requirements.txt` - Python dependencies
- `venv/` - Python virtual environment
- `photos/` - Directory where Python script saves photos (created automatically)
