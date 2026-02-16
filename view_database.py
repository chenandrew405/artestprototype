#!/usr/bin/env python3
"""
View face detection database contents
"""

import sqlite3
import config
from datetime import datetime


def view_detections():
    """Display all detection events"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("DETECTION EVENTS")
    print("=" * 80)

    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    detections = cursor.fetchall()

    if not detections:
        print("No detections found in database")
    else:
        for det in detections:
            det_id, timestamp, num_faces, image_path, notes = det
            print(f"\nID: {det_id}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Faces: {num_faces}")
            print(f"  Image: {image_path if image_path else 'Not saved'}")
            if notes:
                print(f"  Notes: {notes}")

    conn.close()


def view_faces():
    """Display all individual face detections"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("INDIVIDUAL FACES")
    print("=" * 80)

    cursor.execute("""
        SELECT f.id, f.detection_id, f.x, f.y, f.width, f.height, f.confidence, d.timestamp
        FROM faces f
        JOIN detections d ON f.detection_id = d.id
        ORDER BY d.timestamp DESC
        LIMIT 20
    """)
    faces = cursor.fetchall()

    if not faces:
        print("No faces found in database")
    else:
        for face in faces:
            face_id, det_id, x, y, w, h, conf, timestamp = face
            print(f"\nFace ID: {face_id} (Detection: {det_id})")
            print(f"  Time: {timestamp}")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {w}x{h}px")
            print(f"  Confidence: {conf:.2f}")

    conn.close()


def view_statistics():
    """Display database statistics"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    # Total detections
    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]

    # Total faces
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_faces = cursor.fetchone()[0]

    # Average faces per detection
    cursor.execute("SELECT AVG(num_faces) FROM detections")
    avg_faces = cursor.fetchone()[0] or 0

    # First and last detection
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM detections")
    first, last = cursor.fetchone()

    print(f"Total Detection Events: {total_detections}")
    print(f"Total Faces Detected: {total_faces}")
    print(f"Average Faces per Detection: {avg_faces:.2f}")
    print(f"First Detection: {first if first else 'N/A'}")
    print(f"Last Detection: {last if last else 'N/A'}")

    conn.close()


def main():
    """Main entry point"""
    print("\nFace Detection Database Viewer")

    try:
        view_statistics()
        view_detections()
        view_faces()
        print("\n" + "=" * 80 + "\n")
    except sqlite3.OperationalError:
        print(f"\nError: Database not found at {config.DATABASE_PATH}")
        print("Run face_detection.py first to create the database.\n")


if __name__ == "__main__":
    main()
