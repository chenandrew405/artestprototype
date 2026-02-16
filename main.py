#!/usr/bin/env python3
"""
Python-based camera capture script using OpenCV
Alternative to the web-based approach
"""

import cv2
import os
from datetime import datetime


def capture_photo():
    """Access camera, take a photo, and save it."""

    # Initialize camera (0 is usually the default camera)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera")
        return False

    print("Camera accessed successfully!")
    print("Press SPACE to take a photo, or ESC to exit")

    # Create photos directory if it doesn't exist
    photos_dir = "photos"
    os.makedirs(photos_dir, exist_ok=True)

    photo_count = 0

    while True:
        # Read frame from camera
        ret, frame = camera.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display the frame
        cv2.imshow('Camera Feed - Press SPACE to capture, ESC to exit', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # SPACE key - capture photo
        if key == 32:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(photos_dir, f"photo_{timestamp}.png")

            # Save the photo
            cv2.imwrite(filename, frame)
            photo_count += 1

            print(f"Photo saved: {filename}")

            # Flash effect
            flash = frame.copy()
            flash.fill(255)
            cv2.imshow('Camera Feed - Press SPACE to capture, ESC to exit', flash)
            cv2.waitKey(100)

        # ESC key - exit
        elif key == 27:
            break

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

    print(f"\nTotal photos captured: {photo_count}")
    return True


if __name__ == "__main__":
    print("Python Camera Capture Tool")
    print("=" * 40)
    capture_photo()
