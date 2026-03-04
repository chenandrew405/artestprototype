from __future__ import annotations

import argparse
import logging
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlazeFace Camera</title>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: #111;
      color: #f2f2f2;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }
    main {
      width: min(92vw, 980px);
    }
    h1 {
      font-size: clamp(1rem, 2vw, 1.35rem);
      margin: 0 0 0.75rem;
      font-weight: 600;
    }
    .frame {
      border: 2px solid #2f2f2f;
      border-radius: 10px;
      overflow: hidden;
      background: #000;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
    }
    img {
      display: block;
      width: 100%;
      height: auto;
    }
    p {
      color: #b7b7b7;
      font-size: 0.9rem;
      margin: 0.6rem 0 0;
    }
  </style>
</head>
<body>
  <main>
    <h1>Web Camera Face Detection (BlazeFace)</h1>
    <div class="frame">
      <img src="/video_feed" alt="Camera stream with face detection boxes">
    </div>
    <p>Green boxes are rendered server-side on each frame before streaming.</p>
  </main>
</body>
</html>
"""


class FaceStream:
    def __init__(self, model_path: Path, camera_index: int) -> None:
        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._lock = Lock()

    def next_jpeg_frame(self) -> bytes:
        with self._lock:
            ok, frame = self._camera.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)

            for detection in result.detections:
                bbox = detection.bounding_box
                x1 = max(0, int(bbox.origin_x))
                y1 = max(0, int(bbox.origin_y))
                x2 = max(x1 + 1, int(bbox.origin_x + bbox.width))
                y2 = max(y1 + 1, int(bbox.origin_y + bbox.height))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            encoded_ok, jpg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not encoded_ok:
                raise RuntimeError("Failed to encode JPEG frame")
            return jpg.tobytes()

    def close(self) -> None:
        with self._lock:
            self._camera.release()
            self._detector.close()


class RequestHandler(BaseHTTPRequestHandler):
    face_stream: FaceStream | None = None

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        if self.path == "/video_feed":
            self._serve_video_feed()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _serve_index(self) -> None:
        page = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def _serve_video_feed(self) -> None:
        if self.face_stream is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Face stream unavailable")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.face_stream.next_jpeg_frame()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            # Browser disconnected; no action required.
            return

    def log_message(self, fmt: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_model = project_root / "blaze_face_short_range.tflite"

    parser = argparse.ArgumentParser(
        description="Web-based camera face detection using MediaPipe BlazeFace."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default=str(default_model),
        help=f"Path to BlazeFace model (default: {default_model})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stream = FaceStream(model_path=model_path, camera_index=args.camera_index)
    RequestHandler.face_stream = stream

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    logging.info("Serving face detection app at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server")
    finally:
        server.shutdown()
        server.server_close()
        stream.close()


if __name__ == "__main__":
    main()
