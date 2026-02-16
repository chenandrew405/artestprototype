#!/usr/bin/env python3
"""
Face Detection with Database Integration
Uses OpenCV's Haar Cascade for real-time face detection
Stores detection events and face data in SQLite database
"""

import cv2
import sqlite3
import os
from datetime import datetime
import time
import config


class FaceDetectionDB:
    """Handle database operations for face detection"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript(config.DB_SCHEMA)
        conn.commit()
        conn.close()
        print(f"Database initialized at: {self.db_path}")

    def save_detection(self, num_faces, image_path=None, notes=None):
        """Save a detection event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (num_faces, image_path, notes) VALUES (?, ?, ?)",
            (num_faces, image_path, notes)
        )
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return detection_id

    def save_face(self, detection_id, x, y, width, height, confidence=1.0):
        """Save individual face coordinates to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO faces (detection_id, x, y, width, height, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (detection_id, x, y, width, height, confidence)
        )
        conn.commit()
        conn.close()

    def get_recent_detections(self, limit=10):
        """Get recent detection events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        results = cursor.fetchall()
        conn.close()
        return results

    def get_total_faces_detected(self):
        """Get total number of faces detected across all sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        conn.close()
        return count


class FaceDetector:
    """Real-time face detection with camera"""

    def __init__(self):
        self.db = FaceDetectionDB(config.DATABASE_PATH)
        self.camera = None
        self.face_cascade = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0

        self.load_face_cascade()

    def load_face_cascade(self):
        """Load OpenCV Haar Cascade for face detection"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise Exception("Failed to load Haar Cascade classifier")

        print("Face detection model loaded successfully")

    def init_camera(self):
        """Initialize camera"""
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)

        if not self.camera.isOpened():
            raise Exception("Could not open camera")

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        print(f"Camera initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")

    def detect_faces(self, frame):
        """Detect faces in frame"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_SIZE
        )

        return faces

    def draw_faces(self, frame, faces):
        """Draw bounding boxes around detected faces"""
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                config.DETECTION_COLOR,
                config.BOX_THICKNESS
            )

            # Draw label
            label = "Face"
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.DETECTION_COLOR,
                2
            )

        return frame

    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1

        if self.frame_count >= 30:
            elapsed = time.time() - self.fps_start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

        return self.fps

    def draw_info(self, frame, num_faces):
        """Draw information overlay on frame"""
        info_y = 30

        if config.SHOW_FPS:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30

        if config.SHOW_FACE_COUNT:
            faces_text = f"Faces: {num_faces}"
            cv2.putText(frame, faces_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30

        # Total faces detected
        total_faces = self.db.get_total_faces_detected()
        total_text = f"Total Detected: {total_faces}"
        cv2.putText(frame, total_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def save_detection_to_db(self, frame, faces, save_image=False):
        """Save detection event and faces to database"""
        image_path = None

        if save_image and config.SAVE_IMAGES:
            # Save image to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(config.OUTPUT_DIR, filename)
            cv2.imwrite(image_path, frame)

        # Save to database
        detection_id = self.db.save_detection(len(faces), image_path)

        # Save individual face coordinates
        for (x, y, w, h) in faces:
            self.db.save_face(detection_id, int(x), int(y), int(w), int(h))

        print(f"Saved detection: {len(faces)} face(s) - ID: {detection_id}")
        if image_path:
            print(f"Image saved: {image_path}")

    def run(self):
        """Main detection loop"""
        try:
            self.init_camera()

            print("\n" + "=" * 60)
            print("Face Detection Started")
            print("=" * 60)
            print("Controls:")
            print("  'q' or ESC - Quit")
            print("  's' or SPACE - Save current detection to database")
            print("=" * 60 + "\n")

            while True:
                # Read frame
                ret, frame = self.camera.read()

                if not ret:
                    print("Failed to capture frame")
                    break

                # Detect faces
                faces = self.detect_faces(frame)

                # Calculate FPS
                self.calculate_fps()

                # Draw faces and info
                frame = self.draw_faces(frame, faces)
                frame = self.draw_info(frame, len(faces))

                # Auto-save to database if enabled and faces detected
                if config.SAVE_DETECTIONS and len(faces) > 0 and self.frame_count % 30 == 0:
                    self.save_detection_to_db(frame, faces, save_image=False)

                # Display frame
                cv2.imshow(config.WINDOW_NAME, frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

                # Save detection manually
                elif key == ord('s') or key == 32:  # 's' or SPACE
                    if len(faces) > 0:
                        self.save_detection_to_db(frame, faces, save_image=True)
                    else:
                        print("No faces detected to save")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()

            # Show summary
            total_faces = self.db.get_total_faces_detected()
            print(f"\nTotal faces detected in database: {total_faces}")
            print("Face detection stopped")


def main():
    """Main entry point"""
    print("Face Detection with Database Integration")
    print("=" * 60)

    detector = FaceDetector()
    detector.run()


if __name__ == "__main__":
    main()
