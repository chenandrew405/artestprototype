#!/usr/bin/env python3
"""
Face Detection with Database Integration
Uses OpenCV DNN face detector (Res10 SSD) for real-time face detection
Stores detection events and face data in SQLite database
"""

import cv2
import sqlite3
import os
from datetime import datetime
import time
import urllib.request
import numpy as np
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
        self.face_net = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0

        self.load_face_detector()

    def download_model_file(self, url, output_path, label):
        """Download a DNN model file if not present locally"""
        print(f"Downloading {label}...")
        urllib.request.urlretrieve(url, output_path)
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"Downloaded {label}: {output_path} ({file_size_kb:.1f} KB)")

    def ensure_dnn_models(self):
        """Ensure required DNN model files are available"""
        required_files = [
            (config.DNN_PROTOTXT_PATH, config.DNN_PROTOTXT_URL, "DNN prototxt"),
            (config.DNN_MODEL_PATH, config.DNN_MODEL_URL, "DNN model"),
        ]

        missing_files = [item for item in required_files if not os.path.exists(item[0])]

        if missing_files and not config.AUTO_DOWNLOAD_DNN_MODELS:
            missing_paths = ", ".join(path for (path, _, _) in missing_files)
            raise FileNotFoundError(
                f"Missing DNN model file(s): {missing_paths}. "
                "Enable AUTO_DOWNLOAD_DNN_MODELS or place files manually in models/."
            )

        for output_path, url, label in missing_files:
            self.download_model_file(url, output_path, label)

    def load_face_detector(self):
        """Load OpenCV DNN face detector"""
        self.ensure_dnn_models()
        self.face_net = cv2.dnn.readNetFromCaffe(config.DNN_PROTOTXT_PATH, config.DNN_MODEL_PATH)

        if self.face_net.empty():
            raise Exception("Failed to load OpenCV DNN face detector")

        print("DNN face detection model loaded successfully")

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
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return []

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=config.DNN_INPUT_SIZE,
            mean=config.DNN_MEAN_VALUES,
            swapRB=False,
            crop=False
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < config.DNN_CONFIDENCE_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height], dtype="float32")
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            face_width = x2 - x1
            face_height = y2 - y1
            if face_width <= 0 or face_height <= 0:
                continue

            if face_width < config.DNN_MIN_FACE_SIZE[0] or face_height < config.DNN_MIN_FACE_SIZE[1]:
                continue

            faces.append((x1, y1, face_width, face_height, confidence))

        return faces

    def draw_faces(self, frame, faces):
        """Draw bounding boxes around detected faces"""
        for (x, y, w, h, confidence) in faces:
            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                config.DETECTION_COLOR,
                config.BOX_THICKNESS
            )

            # Draw label
            label = f"Face {confidence * 100:.0f}%"
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
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
        for (x, y, w, h, confidence) in faces:
            self.db.save_face(detection_id, int(x), int(y), int(w), int(h), float(confidence))

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
    print("Face Detection with OpenCV DNN + Database Integration")
    print("=" * 60)

    detector = FaceDetector()
    detector.run()


if __name__ == "__main__":
    main()
