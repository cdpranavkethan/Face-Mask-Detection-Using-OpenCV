import os
import cv2
import glob
import time
import imutils
import argparse
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Default paths (can be overridden via CLI)
FACE_DET_PROTO = "deploy.prototxt"
FACE_DET_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = "mask_detector_pretrained.h5"

def try_set_dnn_acceleration(net: cv2.dnn_Net, accelerate: str) -> None:
    try:
        if accelerate == "opencl":
            cv2.ocl.setUseOpenCL(True)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass


class FaceDetector:
    """Abstraction over different face detection backends."""

    def __init__(
        self,
        detector: str,
        conf_threshold: float,
        dnn_prototxt: str,
        dnn_weights: str,
        accelerate: str = "cpu",
        yunet_model_path: str = "",
    ) -> None:
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.accelerate = accelerate
        self._dnn_net = None
        self._yunet = None

        if self.detector == "yunet":
            # Create YuNet detector if available
            self._create_yunet(yunet_model_path)
            if self._yunet is None:
                print("[WARN] YuNet not available or model missing. Falling back to DNN face detector.")
                self.detector = "dnn"

        if self.detector == "dnn":
            self._dnn_net = cv2.dnn.readNet(dnn_prototxt, dnn_weights)
            try_set_dnn_acceleration(self._dnn_net, accelerate)

    def _create_yunet(self, model_path: str) -> None:
        try:
            # YuNet API requires OpenCV 4.6+
            if not model_path or not os.path.isfile(model_path):
                self._yunet = None
                return
            # Initial input size will be set per frame in detect()
            self._yunet = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(320, 240),
                score_threshold=self.conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception:
            self._yunet = None

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:

        if self.detector == "yunet" and self._yunet is not None:
            h, w = frame.shape[:2]
            try:
                self._yunet.setInputSize((w, h))
            except Exception:
                pass
            _, faces = self._yunet.detect(frame)
            boxes: List[Tuple[int, int, int, int, float]] = []
            if faces is not None:
                for f in faces:
                    x, y, bw, bh, score = f[:5]
                    if score >= self.conf_threshold:
                        startX = max(0, int(x))
                        startY = max(0, int(y))
                        endX = min(w - 1, int(x + bw))
                        endY = min(h - 1, int(y + bh))
                        boxes.append((startX, startY, endX, endY, float(score)))
            return boxes

        # Default DNN SSD face detector
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()
        results: List[Tuple[int, int, int, int, float]] = []
        for i in range(0, detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                results.append((startX, startY, endX, endY, confidence))
        return results


def batch_predict_masks(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    mask_model: tf.keras.Model,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """Crop, preprocess all faces and run one batched prediction."""
    locs: List[Tuple[int, int, int, int]] = []
    preprocessed: List[np.ndarray] = []
    for (startX, startY, endX, endY, _) in boxes:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        face_arr = img_to_array(face_rgb)
        face_arr = preprocess_input(face_arr)
        preprocessed.append(face_arr)
        locs.append((startX, startY, endX, endY))

    if not preprocessed:
        return [], []

    batch = np.stack(preprocessed, axis=0)
    preds = mask_model.predict(batch, verbose=0)
    return locs, list(preds)


def draw_overlays(
    frame: np.ndarray,
    locs: List[Tuple[int, int, int, int]],
    preds: List[np.ndarray],
) -> Tuple[np.ndarray, bool, int, int]:
    """Draw boxes/labels; return frame, alert_needed, mask_count, no_mask_count."""
    alert_needed = False
    num_mask = 0
    num_no_mask = 0
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        mask_prob = float(pred[0])
        no_mask_prob = float(pred[1])
        if mask_prob >= no_mask_prob:
            label = "Mask"
            color = (0, 255, 0)
            num_mask += 1
        else:
            label = "No Mask"
            color = (0, 0, 255)
            num_no_mask += 1
        confidence = max(mask_prob, no_mask_prob) * 100.0
        label_text = f"{label}: {confidence:.1f}%"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        if label == "No Mask":
            alert_needed = True
    return frame, alert_needed, num_mask, num_no_mask


# Sound alerts removed


def process_frame(
    frame: np.ndarray,
    face_detector: FaceDetector,
    mask_model: tf.keras.Model,
    alert_threshold: float,
    enable_visual_alert: bool,
    fps_text: str,
) -> Tuple[np.ndarray, int, int, bool]:
    """Run detection+classification and add overlays and optional alerts."""
    frame = imutils.resize(frame, width=800)

    boxes = face_detector.detect(frame)
    locs, preds = batch_predict_masks(frame, boxes, mask_model)
    frame, alert_needed, num_mask, num_no_mask = draw_overlays(frame, locs, preds)

    # Alert only if there are no masked faces AND at least one high-confidence no-mask
    any_no_mask_high = any(float(pred[1]) >= alert_threshold for pred in preds)
    should_alert = (num_mask == 0) and any_no_mask_high

    if should_alert and enable_visual_alert:
        # Bottom-left, smaller text
        h = frame.shape[0]
        cv2.putText(frame, "ALERT: NO MASK", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Overlays: instructions and FPS
    instructions = [
        f"FPS: {fps_text}",
        f"Mask: {num_mask}  No Mask: {num_no_mask}",
    ]
    y_offset = 30
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    return frame, num_mask, num_no_mask, should_alert


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face mask detection demo")
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--video", type=str, default="", help="Path to input video file")
    src_group.add_argument("--images", type=str, default="", help="Path to input images folder")

    parser.add_argument("--output", type=str, default="", help="Optional output path (video file or images folder)")
    parser.add_argument("--detector", type=str, choices=["dnn", "yunet"], default="dnn", help="Face detector backend")
    parser.add_argument("--yunet-model", type=str, default="", help="Path to YuNet ONNX model (if using --detector yunet)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Face detection confidence threshold")
    parser.add_argument("--accelerate", type=str, choices=["cpu", "opencl"], default="cpu", help="Try to accelerate DNN with OpenCL")

    parser.add_argument("--mask-model", type=str, default=MASK_MODEL_PATH, help="Path to mask classifier .h5 model")
    parser.add_argument("--face-prototxt", type=str, default=FACE_DET_PROTO, help="Path to face detector prototxt")
    parser.add_argument("--face-weights", type=str, default=FACE_DET_MODEL, help="Path to face detector caffemodel")

    parser.add_argument("--alert-threshold", type=float, default=0.8, help="No-mask probability to trigger alert")
    parser.add_argument("--no-alert-visual", action="store_true", help="Disable visual alert text")
    return parser


def run_webcam(args: argparse.Namespace, face_detector: FaceDetector, mask_model: tf.keras.Model) -> None:
    print("\nStarting mask detection (webcam)...")

    # Try different camera indices
    camera_index = 0
    cap = None
    while camera_index < 3:
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera {camera_index} opened successfully!")
                break
            else:
                cap.release()
                camera_index += 1
        except Exception:
            camera_index += 1

    if cap is None or not cap.isOpened():
        print("Could not open camera. Please check camera permissions.")
        print("On macOS, you may need to grant camera access to your terminal/IDE.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_alert_time = 0.0
    fps_smoothed = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        now = time.time()
        dt = max(1e-6, now - prev_time)
        prev_time = now
        fps_inst = 1.0 / dt
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps_inst if fps_smoothed > 0 else fps_inst
        fps_text = f"{fps_smoothed:.1f}"

        frame_out, _, _, _ = process_frame(
            frame,
            face_detector,
            mask_model,
            args.alert_threshold,
            enable_visual_alert=(not args.no_alert_visual),
            fps_text=fps_text,
        )

        cv2.imshow("Mask Detection - Press 'q' to quit", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo finished!")


def run_video(args: argparse.Namespace, face_detector: FaceDetector, mask_model: tf.keras.Model) -> None:
    path = args.video
    if not os.path.isfile(path):
        print(f"Video not found: {path}")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open video: {path}")
        return

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Our processing resizes to width=800; writer size will follow first processed frame
        writer = cv2.VideoWriter(args.output, fourcc, fps, (800, int(h * (800 / max(1, w)))))

    last_alert_time = 0.0
    fps_smoothed = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-6, now - prev_time)
        prev_time = now
        fps_inst = 1.0 / dt
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps_inst if fps_smoothed > 0 else fps_inst
        fps_text = f"{fps_smoothed:.1f}"

        frame_out, _, _, _ = process_frame(
            frame,
            face_detector,
            mask_model,
            args.alert_threshold,
            enable_visual_alert=(not args.no_alert_visual),
            fps_text=fps_text,
        )

        cv2.imshow("Mask Detection - Press 'q' to quit", frame_out)
        if writer is not None:
            writer.write(frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Video processing finished!")


def run_images(args: argparse.Namespace, face_detector: FaceDetector, mask_model: tf.keras.Model) -> None:
    folder = args.images
    if not os.path.isdir(folder):
        print(f"Images folder not found: {folder}")
        return
    output_dir = args.output if args.output else os.path.join(folder, "output")
    os.makedirs(output_dir, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    if not files:
        print("No images found in the folder.")
        return

    last_alert_time = 0.0
    for path in files:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping unreadable image: {path}")
            continue
        start = time.time()
        frame_out, _, _, _ = process_frame(
            frame,
            face_detector,
            mask_model,
            args.alert_threshold,
            enable_visual_alert=(not args.no_alert_visual),
            fps_text=f"{1.0 / max(1e-6, (time.time() - start)):.1f}",
        )
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(output_dir, f"{name}_out{ext}")
        cv2.imwrite(out_path, frame_out)
        cv2.imshow("Mask Detection - Images (press 'q' to stop)", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(f"Saved results to: {output_dir}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load face detector
    print("Loading face detection backend...")
    face_detector = FaceDetector(
        detector=args.detector,
        conf_threshold=args.threshold,
        dnn_prototxt=args.face_prototxt,
        dnn_weights=args.face_weights,
        accelerate=args.accelerate,
        yunet_model_path=args.yunet_model,
    )

    # Load mask model
    print("Loading mask detection model...")
    try:
        mask_model = tf.keras.models.load_model(args.mask_model)
        print("Successfully loaded pre-trained mask model!")
    except Exception as e:
        print(f"Error loading mask model: {e}")
        print("Please ensure the model file exists.")
        return

    # Route to the requested mode
    if args.video:
        run_video(args, face_detector, mask_model)
    elif args.images:
        run_images(args, face_detector, mask_model)
    else:
        run_webcam(args, face_detector, mask_model)


if __name__ == "__main__":
    main()
