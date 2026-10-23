"""
marker_bearing_demo.py - ArUco marker tracking: live 3D bearing, range, and velocity.

This runs the bearing pipeline from the drone detection notes on a webcam:
  Step 1  YOLO model (best.pt) detects ArUco markers (e.g. classes 41, 32, 43, 44)
  Step 2  marker pixels -> unit bearings b^C   (undistort, K^-1, normalise)
  Step 5  range rho from marker angular width & height (known marker size)
  Step 6  r^C = rho * b^C
  Step 3  r^B = R_BC r^C  -> printed as forward / left / up, like on the drone
  Step 4  g_LT = R_WB b^B -> measured bearing in the world frame (unit vector)
          R_WB comes from --roll --pitch --yaw (degrees). Default 0, 0, 0 = level,
          world x = straight out of camera, so g_LT = b^B.
  Target velocity  v_T(k) = (r_LT(k) - r_LT(k-1)) / dt + v_L
  Desired bearing  g* = v_T / |v_T|   (only when |v_T| > --vmin, otherwise held)

Directions are from the CAMERA's point of view.

Keys:  c = calibrate range: hold marker at exactly --cal-dist metres, press c
       q = quit
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CALIB_FILE = "webcam_calib.json"
PREFERRED_MODELS = [
    "best.pt",
    "best (7).pt",
    "best(7).pt",
    "best (8).pt",
    "best (6).pt",
    "best (4).pt",
    "best (3).pt",
    "best (2).pt",
    "best_prev.pt",
    "best_30epochs.pt",
    "best_backup_before_colab.pt",
    "best_old.pt",
    "best_old_20260623.pt",
]
CAM_INDEX = 0
FALLBACK_SIZE = (640, 480)
FALLBACK_DFOV_DEG = 75.0      # rough diagonal field of view of a laptop webcam
SMOOTHING = 0.3               # position low-pass, 0..1, lower = smoother
VEL_SMOOTHING = 0.15          # velocity low-pass
DEFAULT_MARKER_SIZE = 0.10    # default marker side length in metres (100 mm / 10 cm)

# Camera frame (x right, y down, z forward) -> body frame (x forward, y left, z up)
R_BC = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]], dtype=np.float64)


# ---------------- Attitude ----------------
def R_WB_from_attitude(roll_deg, pitch_deg, yaw_deg):
    """Body -> world rotation, R_WB = Rz(yaw) Ry(pitch) Rx(roll) (ZYX).
    Body: x forward, y left, z up. Positive roll = right side down,
    positive pitch = nose down, positive yaw = nose turns from world x toward world y."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], dtype=np.float64)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], dtype=np.float64)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def attitude_from_R_WB(R):
    """Inverse of R_WB_from_attitude, used as a check."""
    pitch = math.asin(-float(np.clip(R[2, 0], -1.0, 1.0)))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return tuple(math.degrees(a) + 0.0 for a in (roll, pitch, yaw))


def clean(M, tol=5e-5):
    """Set tiny entries to exactly 0, so they print as +0.000 instead of -0.000."""
    return np.where(np.abs(M) < tol, 0.0, M)


def print_attitude(roll, pitch, yaw, R_WB):
    """Terminal printout of the attitude and R_WB once at start-up."""
    R = clean(R_WB)
    print("\nCamera attitude (from --roll --pitch --yaw):")
    print(f"  roll {roll:+.1f} deg    pitch {pitch:+.1f} deg    yaw {yaw:+.1f} deg")
    print("R_WB = Rz(yaw) Ry(pitch) Rx(roll)    (body -> world)")
    print(" " * 14 + "fwd" + " " * 6 + "left" + " " * 6 + "up")
    for name, row in zip(("world x", "world y", "world z"), R):
        print(f"  {name}  [ " + "  ".join(f"{v:+.4f}" for v in row) + " ]")
    r, p, y = attitude_from_R_WB(R_WB)
    print(f"  check: angles read back from R_WB -> roll {r:+.1f}, pitch {p:+.1f}, yaw {y:+.1f} deg")
    print("  (each column = one body axis written in world axes)\n")


# ---------------- Camera model ----------------
def load_camera_model(frame_size):
    """Use webcam_calib.json if it exists, otherwise a rough guess from a typical FOV."""
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                c = json.load(f)
            K = np.array([[c["fx"], 0.0, c["cx"]],
                          [0.0, c["fy"], c["cy"]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            dist = np.array(c.get("dist_opencv", np.zeros(5)), dtype=np.float64)
            size = (int(c["image_width"]), int(c["image_height"]))
            return K, dist, size, True
        except Exception as e:
            print(f"Warning: Failed to parse {CALIB_FILE} ({e}). Using estimated camera model.")

    W, H = frame_size
    f = (math.hypot(W, H) / 2.0) / math.tan(math.radians(FALLBACK_DFOV_DEG) / 2.0)
    K = np.array([[f, 0.0, (W - 1) / 2.0],
                  [0.0, f, (H - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print(f"NOTE: {CALIB_FILE} not found. Using estimated fx = fy = {f:.0f} px, "
          f"centre principal point, no distortion. Angles and ranges are approximate.")
    return K, np.zeros(5, dtype=np.float64), (W, H), False


def pixels_to_bearings(pts, K, dist):
    """N pixels (N x 2) -> N unit bearings (N x 3) in the camera frame."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    xn = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    q = np.hstack([xn, np.ones((len(xn), 1), dtype=np.float64)])
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return q / norm


# ---------------- Marker Measurement ----------------
def measure_marker(box, K, dist, marker_size_m, corners=None):
    """Compute bearing and range for a detected marker.
    
    If corner detections are available, uses the 4 corner coordinates.
    Otherwise uses the bounding box horizontal and vertical spans.
    The subtended angle theta gives range:
        rho = marker_size / (2 * tan(theta / 2))
    """
    marker_size_m = max(float(marker_size_m), 1e-4)

    if corners is not None and len(corners) == 4:
        c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        center = c.mean(axis=0)
        # Horizontal bearings (top and bottom edges)
        b0, b1 = pixels_to_bearings([c[0], c[1]], K, dist)
        b3, b2 = pixels_to_bearings([c[3], c[2]], K, dist)
        th_w1 = math.acos(float(np.clip(b0 @ b1, -1.0, 1.0)))
        th_w2 = math.acos(float(np.clip(b3 @ b2, -1.0, 1.0)))
        theta_w = max(th_w1, th_w2)

        # Vertical bearings (left and right edges)
        b0, b3 = pixels_to_bearings([c[0], c[3]], K, dist)
        b1, b2 = pixels_to_bearings([c[1], c[2]], K, dist)
        th_h1 = math.acos(float(np.clip(b0 @ b3, -1.0, 1.0)))
        th_h2 = math.acos(float(np.clip(b1 @ b2, -1.0, 1.0)))
        theta_h = max(th_h1, th_h2)
    else:
        x1, y1, x2, y2 = box
        cx_m = (x1 + x2) / 2.0
        cy_m = (y1 + y2) / 2.0
        center = np.array([cx_m, cy_m], dtype=np.float64)

        p_left = np.array([x1, cy_m])
        p_right = np.array([x2, cy_m])
        b_left, b_right = pixels_to_bearings([p_left, p_right], K, dist)
        theta_w = math.acos(float(np.clip(b_left @ b_right, -1.0, 1.0)))

        p_top = np.array([cx_m, y1])
        p_bot = np.array([cx_m, y2])
        b_top, b_bot = pixels_to_bearings([p_top, p_bot], K, dist)
        theta_h = math.acos(float(np.clip(b_top @ b_bot, -1.0, 1.0)))

    # Take maximum dimension to avoid foreshortening errors when tilted
    theta = max(theta_w, theta_h)
    theta = min(max(theta, 1e-4), math.pi - 1e-4)   # prevent division by zero or negative

    rho = marker_size_m / (2.0 * math.tan(theta / 2.0))
    b = pixels_to_bearings([center], K, dist)[0]
    r_C = rho * b
    r_B = R_BC @ r_C

    return {
        "theta": theta,
        "rho": rho,
        "b_C": b,
        "r_C": r_C,
        "r_B": r_B,
        "center": center,
        "corners": corners
    }


def velocity_from_positions(r_now, r_prev, dt, v_L):
    """Target velocity from position change: v_T = (r_LT(k) - r_LT(k-1)) / dt + v_L"""
    return (r_now - r_prev) / dt + v_L


def desired_bearing(v_T, v_min):
    """g* = v_T / |v_T|. Returns None when the target is still."""
    speed = float(np.linalg.norm(v_T))
    return v_T / speed if speed > v_min else None


def angle_deg(a, b):
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if math.isnan(dot):
        return 0.0
    return math.degrees(math.acos(dot))


def fmt(v, nd=3):
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in v) + "]"


def side(value, pos, neg):
    return f"{abs(value):.1f} deg {pos if value >= 0 else neg}"


# ---------------- Model & Detector Setup ----------------
def list_available_models():
    """List all available .pt models with their training metrics."""
    models = sorted(glob.glob("*.pt"))
    if not models:
        print("No .pt models found in the workspace.")
        return
    print("\nAvailable YOLO Models in Workspace:")
    print("%-28s | %-6s | %-9s | %-6s | %-6s | %-8s" %
          ("Model File", "Epochs", "Precision", "Recall", "mAP50", "mAP50-95"))
    print("-" * 75)
    for f in models:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            m = ckpt.get("train_metrics", {})
            p = m.get("metrics/precision(B)", 0.0)
            r = m.get("metrics/recall(B)", 0.0)
            map50 = m.get("metrics/mAP50(B)", 0.0)
            map5095 = m.get("metrics/mAP50-95(B)", 0.0)
            t_args = ckpt.get("train_args", {})
            epochs = t_args.get("epochs", "?") if isinstance(t_args, dict) else "?"
            print("%-28s | %-6s | %-9.4f | %-6.4f | %-6.4f | %-8.4f" %
                  (f, str(epochs), p, r, map50, map5095))
        except Exception as e:
            print("%-28s | (could not read metadata: %s)" % (f, e))
    print()


def resolve_model_file(requested_path=None):
    """Finds requested model, or preferred best.pt, or falls back to best available."""
    if requested_path:
        req = requested_path.strip()
        if req.lower() == "list":
            list_available_models()
            sys.exit(0)
        # Check exact path
        if os.path.exists(req):
            return req
        # Check if .pt extension was omitted
        if os.path.exists(req + ".pt"):
            return req + ".pt"
        # Check numeric alias (e.g. "8" -> "best (8).pt")
        for candidate in [f"best ({req}).pt", f"best({req}).pt", f"best_{req}.pt"]:
            if os.path.exists(candidate):
                return candidate
        # If requested explicitly but not found:
        print(f"Warning: Requested model '{requested_path}' not found. Searching for alternatives...")

    # Look for preferred models
    for candidate in PREFERRED_MODELS:
        if os.path.exists(candidate):
            return candidate

    # Look for any .pt file in the directory
    all_pt = sorted(glob.glob("*.pt"))
    if all_pt:
        print(f"Using available model: {all_pt[0]}")
        return all_pt[0]

    sys.exit("Error: No YOLO weights (.pt) found in current directory. Please ensure 'best.pt' is present.")


def init_aruco_detectors():
    """Initializes OpenCV ArUco detectors for corner refinement across all OpenCV versions."""
    detectors = []
    if not hasattr(cv2, "aruco"):
        return detectors
    dict_ids = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
    ]
    try:
        params = cv2.aruco.DetectorParameters()
        has_detector_class = hasattr(cv2.aruco, "ArucoDetector")
        for d_id in dict_ids:
            d = cv2.aruco.getPredefinedDictionary(d_id)
            if has_detector_class:
                det = cv2.aruco.ArucoDetector(d, params)
                detectors.append(("new", det))
            else:
                detectors.append(("legacy", (d, params)))
    except Exception as e:
        print(f"Note: ArUco subpixel refinement initialization warning: {e}")
    return detectors


def refine_with_aruco(frame, box, aruco_detectors):
    """Try to extract precise subpixel corners from the ArUco marker."""
    if not aruco_detectors:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 12
    h_img, w_img = frame.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for mode, item in aruco_detectors:
        try:
            if mode == "new":
                corners, ids, _ = item.detectMarkers(gray)
            else:
                d, params = item
                corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
            if ids is not None and len(ids) > 0:
                c = corners[0][0].copy().astype(np.float64)
                c[:, 0] += rx1
                c[:, 1] += ry1
                return c
        except Exception:
            continue
    return None


def open_video_source(source, width=None, height=None):
    """Robust camera or video/image source opener with multi-backend fallback."""
    is_cam = False
    try:
        cam_idx = int(source)
        is_cam = True
    except (ValueError, TypeError):
        cam_idx = None

    if is_cam:
        backends = []
        if sys.platform.startswith("win"):
            backends = [
                ("DirectShow", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
                ("Default", cv2.CAP_ANY),
            ]
        else:
            backends = [("Default", cv2.CAP_ANY)]

        for b_name, b_flag in backends:
            cap = cv2.VideoCapture(cam_idx, b_flag)
            if cap.isOpened():
                if width and height:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Connected to camera index {cam_idx} using {b_name} backend ({frame.shape[1]}x{frame.shape[0]}).")
                    return cap, frame, True
                cap.release()

        # If selected camera index failed, probe available cameras 0..3
        avail = []
        for idx in range(4):
            c_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
            if c_test.isOpened():
                r, _ = c_test.read()
                if r:
                    avail.append(idx)
                c_test.release()

        err_msg = f"Could not read from camera index {cam_idx}."
        if avail:
            err_msg += f" Detected active camera index(es): {avail}. Please run with --cam {avail[0]}."
        else:
            err_msg += " No active webcam detected. Please verify camera connection or test with a video/image using --source path."
        sys.exit(f"Error: {err_msg}")
    else:
        src_path = str(source)
        if not os.path.exists(src_path):
            sys.exit(f"Error: Source file '{src_path}' not found.")
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            sys.exit(f"Error: Could not open media file '{src_path}'.")
        ret, frame = cap.read()
        if not ret or frame is None:
            sys.exit(f"Error: Could not read frame from '{src_path}'.")
        print(f"Opened media source '{src_path}' ({frame.shape[1]}x{frame.shape[0]}).")
        return cap, frame, False


# ---------------- Display: White side panel ----------------
PANEL_W = 540
PANEL_MIN_H = 780
MAT_COL_RIGHT = (230, 340, 450)
FONT = cv2.FONT_HERSHEY_DUPLEX
STYLES = {
    "head": ((140, 70, 0), 0.70, 34),      # dark blue section title
    "text": ((20, 20, 20), 0.65, 28),      # main values, black
    "note": ((110, 110, 110), 0.55, 24),   # grey explanation
    "warn": ((0, 0, 210), 0.65, 28),       # red
    "held": ((0, 120, 230), 0.65, 28),     # orange: held value
    "gap":  (None, 0.0, 10),               # empty space
}


def get_required_panel_height(rows):
    """Compute required height to avoid text clipping."""
    return sum(STYLES[style][2] for _, style in rows) + 30


def make_panel(rows, height):
    """Render the white data panel next to the video."""
    panel = np.full((height, PANEL_W, 3), 255, np.uint8)
    y = 6
    for text, style in rows:
        color, scale, step = STYLES[style]
        y += step
        if color is None:
            continue
        if isinstance(text, str):
            cv2.putText(panel, text, (16, y - 6), FONT, scale, color, 1, cv2.LINE_AA)
        else:
            label, *cells = text
            cv2.putText(panel, label, (16, y - 6), FONT, scale, color, 1, cv2.LINE_AA)
            for cell, x_right in zip(cells, MAT_COL_RIGHT):
                (w, _), _ = cv2.getTextSize(cell, FONT, scale, 1)
                cv2.putText(panel, cell, (x_right - w, y - 6), FONT, scale, color, 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (0, height - 1), (180, 180, 180), 2)
    return panel


def attitude_rows(roll, pitch, yaw, R_WB):
    """Panel lines for roll, pitch, yaw and R_WB."""
    R = clean(R_WB)
    rows = [("", "gap"),
            ("ATTITUDE   R_WB  (body -> world)", "head"),
            (f"Roll {roll:+.1f}    Pitch {pitch:+.1f}    Yaw {yaw:+.1f}   deg", "text"),
            (["", "fwd", "left", "up"], "note")]
    for name, row in zip(("world x", "world y", "world z"), R):
        rows.append(([name] + [f"{v:+.3f}" for v in row], "text"))
    return rows


def label_on_video(img, text, org, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Small label readable on any video background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ---------------- Main Execution ----------------
def main():
    p = argparse.ArgumentParser(description="Drone Target: ArUco marker live bearing and range tracking.")
    p.add_argument("--model", type=str, default="best.pt",
                   help="path to YOLO weights (.pt), or 'list' to show all models and their metrics")
    p.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                   help=f"marker side length in metres (default {DEFAULT_MARKER_SIZE:.2f} m = {DEFAULT_MARKER_SIZE*1000:.0f} mm)")
    p.add_argument("--ipd", type=float, default=None, help="alias for --marker-size")
    p.add_argument("--cal-dist", type=float, default=1.0, help="distance in metres when pressing c to calibrate")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size (default 640)")
    p.add_argument("--target-id", type=str, default=None, help="specific marker class to track (e.g. '41')")
    p.add_argument("--cam", type=int, default=CAM_INDEX, help="camera device index (default 0)")
    p.add_argument("--source", type=str, default=None, help="input source: camera index or video/image file path")
    p.add_argument("--device", type=str, default="", help="inference device: '' (auto), 'cuda', 'cpu', or device id")
    p.add_argument("--roll", type=float, default=0.0, help="camera/body roll in degrees")
    p.add_argument("--pitch", type=float, default=0.0, help="camera/body pitch in degrees")
    p.add_argument("--yaw", type=float, default=0.0, help="camera/body yaw in degrees")
    p.add_argument("--vmin", type=float, default=0.08, help="speed in m/s below which g* is held")

    args = p.parse_args()

    marker_size = args.ipd if args.ipd is not None else args.marker_size

    # Resolve and load model
    model_file = resolve_model_file(args.model)
    print("\n" + "=" * 60)
    print(f"Loading YOLO model from: {model_file}...")
    model = YOLO(model_file)
    print("Model loaded successfully.")
    print(f"Available marker classes: {model.names}")
    dev = args.device if args.device else ("CUDA GPU" if torch.cuda.is_available() else "CPU")
    print(f"Inference device: {dev} | Image size: {args.imgsz}px | Confidence threshold: {args.conf}")
    print("=" * 60 + "\n")

    aruco_detectors = init_aruco_detectors()

    # Determine camera or video source
    source = args.source if args.source is not None else args.cam
    want = FALLBACK_SIZE
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                c = json.load(f)
            want = (int(c["image_width"]), int(c["image_height"]))
        except Exception:
            pass

    cap, initial_frame, is_webcam = open_video_source(source, want[0], want[1])
    H, W = initial_frame.shape[:2]
    K, dist, size, calibrated = load_camera_model((W, H))
    if calibrated and size != (W, H):
        print(f"Warning: Camera frame is {W}x{H}, but calibration is for {size[0]}x{size[1]}. Using fallback.")
        K, dist, size, calibrated = load_camera_model((W, H))

    cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))
    win = "Drone Detection - ArUco Marker Bearing Demo (c = calibrate, q = quit)"

    try:
        run_loop(cap, model, aruco_detectors, K, dist, marker_size, args, cx, cy, W, H, calibrated, win,
                 is_webcam=is_webcam, model_file=model_file)
    except KeyboardInterrupt:
        print("\nStopped with Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_loop(cap, model, aruco_detectors, K, dist, marker_size, args, cx, cy, W, H, calibrated, win,
             is_webcam=True, model_file="best.pt"):
    r_B_smooth = None
    last = None
    frame_no = 0
    R_WB = R_WB_from_attitude(args.roll, args.pitch, args.yaw)
    print_attitude(args.roll, args.pitch, args.yaw, R_WB)
    att_rows = attitude_rows(args.roll, args.pitch, args.yaw, R_WB)
    v_L = np.zeros(3, dtype=np.float64)
    r_W_prev = None
    t_prev = None
    v_T = None
    g_star = None

    fps_smooth = None
    t_last_frame = None

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W + PANEL_W, max(H, PANEL_MIN_H))

    while True:
        ok, frame = cap.read()
        if not ok:
            if not is_webcam:
                # Video file: loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                print("End of stream or disconnected.")
                break

        t_now = time.perf_counter()
        frame_no += 1

        # FPS calculation
        dt_frame = t_now - (t_last_frame if t_last_frame is not None else t_now)
        t_last_frame = t_now
        fps = (1.0 / dt_frame) if dt_frame > 0 else 0.0
        fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)

        # Run YOLO inference
        t_inf_start = time.perf_counter()
        predict_kwargs = dict(conf=args.conf, imgsz=args.imgsz, verbose=False)
        if args.device:
            predict_kwargs["device"] = args.device
        results = model.predict(frame, **predict_kwargs)[0]
        infer_ms = (time.perf_counter() - t_inf_start) * 1000.0
        boxes = results.boxes

        # ---------- Video side ----------
        view = frame.copy()
        cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        label_on_video(view, "straight ahead", (cx + 10, cy - 10))
        label_on_video(view, f"FPS: {fps_smooth:.1f} ({infer_ms:.0f}ms)", (12, 24), color=(0, 255, 120))

        rows = [("WHERE THE MARKER IS   (camera = origin)", "head")]

        detected_candidates = []
        if len(boxes) > 0:
            for b_idx in range(len(boxes)):
                xyxy = boxes.xyxy[b_idx].cpu().numpy()
                conf = float(boxes.conf[b_idx].item())
                cls_id = int(boxes.cls[b_idx].item())
                cls_name = model.names.get(cls_id, str(cls_id))

                if args.target_id is not None and str(cls_name) != str(args.target_id):
                    continue

                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                area = w * h
                detected_candidates.append({
                    "box": xyxy,
                    "conf": conf,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "area": area
                })

        if not detected_candidates:
            r_W_prev = None
            t_prev = None
            rows += [("No marker detected - show ArUco marker to camera", "warn")]
        else:
            # Pick candidate with largest area (closest marker)
            target = max(detected_candidates, key=lambda c: c["area"])
            target_box = target["box"]
            cls_name = target["cls_name"]
            conf = target["conf"]

            # Draw non-primary markers lightly
            for c in detected_candidates:
                if c is not target:
                    x1, y1, x2, y2 = [int(v) for v in c["box"]]
                    cv2.rectangle(view, (x1, y1), (x2, y2), (200, 160, 0), 1)
                    label_on_video(view, f"Marker {c['cls_name']}", (x1, max(15, y1 - 6)))

            # Attempt ArUco subpixel corner refinement
            corners = refine_with_aruco(frame, target_box, aruco_detectors)

            m = measure_marker(target_box, K, dist, marker_size, corners=corners)
            last = m

            r_B_smooth = (m["r_B"] if r_B_smooth is None
                          else SMOOTHING * m["r_B"] + (1 - SMOOTHING) * r_B_smooth)

            # Draw primary marker
            x1, y1, x2, y2 = [int(v) for v in target_box]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 255), 2)
            if corners is not None:
                cv2.polylines(view, [corners.astype(np.int32)], True, (0, 255, 0), 2)

            tcx, tcy = int(round(m["center"][0])), int(round(m["center"][1]))
            cv2.circle(view, (tcx, tcy), 4, (0, 0, 255), -1)
            cv2.line(view, (cx, cy), (tcx, tcy), (0, 255, 0), 1)

            # ---------- Measured position and bearing ----------
            fwd, left, up = r_B_smooth
            rho = float(np.linalg.norm(r_B_smooth))
            az = math.degrees(math.atan2(-left, fwd))
            el = math.degrees(math.atan2(up, math.hypot(fwd, left)))

            b_B = r_B_smooth / max(rho, 1e-6)
            g_LT = R_WB @ b_B

            label_on_video(view, f"Marker {cls_name} ({conf:.2f}) | {rho:.2f} m",
                           (x1, max(18, y1 - 8)), color=(0, 255, 255))

            # ---------- Target velocity in world axes ----------
            r_W = R_WB @ r_B_smooth
            dt = None
            if r_W_prev is not None and t_prev is not None:
                dt = t_now - t_prev
                if dt > 0:
                    v_raw = velocity_from_positions(r_W, r_W_prev, dt, v_L)
                    v_T = v_raw if v_T is None else VEL_SMOOTHING * v_raw + (1 - VEL_SMOOTHING) * v_T
            r_W_prev, t_prev = r_W, t_now

            # ---------- Desired bearing g* ----------
            g_new = desired_bearing(v_T, args.vmin) if v_T is not None else None
            held = g_new is None
            if not held:
                g_star = g_new

            # ---------- Panel text ----------
            rows += [
                (f"Target: Marker {cls_name}   (conf {conf:.2f})", "text"),
                (f"Distance             {rho:.2f} m", "text"),
                (f"Direction            {side(az, 'right', 'left')},  {side(el, 'up', 'down')}", "text"),
                (f"Fwd / Left / Up      {fwd:+.2f} / {left:+.2f} / {up:+.2f} m", "text"),
                ("", "gap"),
                ("BEARINGS   (unit vectors, world axes x, y, z)", "head"),
                (f"g_LT  measured    {fmt(g_LT)}", "text"),
                ("          direction from the camera to the marker", "note"),
            ]
            if g_star is None:
                rows += [("g*    desired     not defined yet", "held"),
                         ("          move the marker so velocity can be measured", "note")]
            else:
                rows += [(f"g*    desired     {fmt(g_star)}", "held" if held else "text"),
                         ("          direction marker is moving"
                          + ("   (held: target is still)" if held else ""), "note"),
                         (f"Angle between g_LT and g*     {angle_deg(g_LT, g_star):.1f} deg", "text"),
                         ("          0 deg = camera is right behind the marker", "note")]

            rows += [("", "gap"), ("TARGET VELOCITY  v_T   (world axes)", "head")]
            if v_T is None:
                rows += [("measuring...", "note")]
            else:
                speed = float(np.linalg.norm(v_T))
                state = "moving" if speed > args.vmin else "still"
                rows += [(f"v_T      {fmt(v_T, 2)} m/s", "text"),
                         (f"Speed    {speed:.2f} m/s   -> {state}  (threshold {args.vmin:.2f})", "text")]

            if frame_no % 30 == 0:
                print(f"[Marker {cls_name}] rho {rho:.2f} m | r_B = [fwd {fwd:+.2f}, left {left:+.2f}, up {up:+.2f}] m"
                      f" | marker angle {math.degrees(m['theta']):.2f} deg"
                      + (f" | dt {dt * 1000:.0f} ms" if dt else ""))
                line = f"   g_LT {fmt(g_LT)}"
                if v_T is not None:
                    line += f" | v_T {fmt(v_T, 2)} m/s, speed {np.linalg.norm(v_T):.2f}"
                if g_star is not None:
                    line += (f" | g* {fmt(g_star)}{' (held)' if held else ''}"
                             f" | angle {angle_deg(g_LT, g_star):.1f} deg")
                else:
                    line += " | g* undefined"
                print(line)

        # ---------- Attitude and R_WB ----------
        rows += att_rows

        # ---------- Settings & Status ----------
        rows += [
            ("", "gap"),
            ("SETTINGS & STATUS", "head"),
            (f"Marker size: {marker_size * 1000:.1f} mm", "note"),
            (f"Model: {os.path.basename(model_file)} ({args.imgsz}px)", "note"),
            (f"Perf: {fps_smooth:.1f} FPS (inf {infer_ms:.1f} ms)", "note"),
        ]
        if not calibrated:
            rows.append(("Camera NOT CALIBRATED - values approximate", "warn"))
        rows.append(("Keys:   c = calibrate range      q = quit", "note"))

        # Put video and panel side by side with dynamic panel height
        req_h = get_required_panel_height(rows)
        panel_h = max(H, PANEL_MIN_H, req_h)

        if H < panel_h:
            view = np.vstack([view, np.zeros((panel_h - H, W, 3), np.uint8)])
        elif H > panel_h:
            panel_h = H

        panel_img = make_panel(rows, panel_h)
        combined = np.hstack([view, panel_img])
        cv2.imshow(win, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
            print("\nExiting on user request.")
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("\nWindow closed.")
            break
        if key in (ord("c"), ord("C")):
            if last is None:
                print("No marker visible - cannot calibrate.")
            else:
                marker_size = 2.0 * args.cal_dist * math.tan(last["theta"] / 2.0)
                r_B_smooth = None
                r_W_prev = None
                t_prev = None
                v_T = None
                print(f"Calibrated at {args.cal_dist:.2f} m: effective marker size = "
                      f"{marker_size * 1000:.1f} mm. Restart with --marker-size {marker_size:.4f} to reuse it.")


if __name__ == "__main__":
    main()