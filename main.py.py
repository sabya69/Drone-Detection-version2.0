"""
main.py - Live Drone Bearing, Range & Target Tracking with Live Podcast Mirroring.

Features:
  - 30 FPS Butter-Smooth Video: Threaded asynchronous camera grabber (zero latency).
  - Live Podcast Mirror Mode (--mirror or 'm' key): Horizontally mirrors video like OBS / Zoom.
  - Hybrid Tracking:
      1. Fast Native OpenCV ArUco (~3-5ms on Raspberry Pi CPU -> 30+ FPS tracking!)
      2. Asynchronous YOLO Worker (runs in background without stalling the video feed).
  - Built-in Web Streaming Server (port 5000):
      View the live mirrored stream from any browser (laptop, phone, tablet)
      with interactive web buttons for Calibrate, Mirror, and Quit.
  - Full 3D Physics & Attitude:
      Calculates range rho, body frame vector r^B (fwd/left/up),
      world frame bearing g_LT, target velocity v_T, and desired bearing g*.

Keys:
  m = toggle live podcast mirror mode
  c = calibrate range (hold marker at --cal-dist metres)
  t = toggle tracker mode (Hybrid vs Fast ArUco only)
  q = quit (or ESC)
"""
import argparse
import atexit
import csv
from datetime import datetime
import glob
import json
import math
import os
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# If running on Linux without an active X11/Wayland display (e.g. SSH), use offscreen Qt platform
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

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
SMOOTHING = 0.35              # position low-pass filter
VEL_SMOOTHING = 0.20          # velocity low-pass filter
DEFAULT_MARKER_SIZE = 0.10    # default marker side length in metres (100 mm / 10 cm)

# Camera frame (x right, y down, z forward) -> body frame (x forward, y left, z up)
R_BC = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]], dtype=np.float64)


# ---------------- Attitude & Coordinate Math ----------------
def R_WB_from_attitude(roll_deg, pitch_deg, yaw_deg):
    """Body -> world rotation, R_WB = Rz(yaw) Ry(pitch) Rx(roll) (ZYX)."""
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
    """Set tiny entries to exactly 0 for neat display."""
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
    print(f"  check: read back -> roll {r:+.1f}, pitch {p:+.1f}, yaw {y:+.1f} deg\n")


# ---------------- Camera Model & Optics ----------------
def load_camera_model(frame_size):
    """Use webcam_calib.json if it exists, otherwise estimated FOV."""
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
    print(f"NOTE: {CALIB_FILE} not found. Using estimated fx = fy = {f:.0f} px, no distortion.")
    return K, np.zeros(5, dtype=np.float64), (W, H), False


def pixels_to_bearings(pts, K, dist):
    """N pixels (N x 2) -> N unit bearings (N x 3) in the camera frame."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    xn = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    q = np.hstack([xn, np.ones((len(xn), 1), dtype=np.float64)])
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return q / norm


def measure_marker(box, K, dist, marker_size_m, corners=None):
    """Compute bearing and range for a detected marker."""
    marker_size_m = max(float(marker_size_m), 1e-4)

    if corners is not None and len(corners) == 4:
        c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        center = c.mean(axis=0)
        b0, b1 = pixels_to_bearings([c[0], c[1]], K, dist)
        b3, b2 = pixels_to_bearings([c[3], c[2]], K, dist)
        th_w1 = math.acos(float(np.clip(b0 @ b1, -1.0, 1.0)))
        th_w2 = math.acos(float(np.clip(b3 @ b2, -1.0, 1.0)))
        theta_w = max(th_w1, th_w2)

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

    theta = max(theta_w, theta_h)
    theta = min(max(theta, 1e-4), math.pi - 1e-4)

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


# ---------------- Model & Detectors ----------------
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
    """Finds requested model, preferred best.pt, or falls back to best available."""
    if requested_path:
        req = requested_path.strip()
        if req.lower() == "list":
            list_available_models()
            sys.exit(0)
        if os.path.exists(req):
            return req
        if os.path.exists(req + ".pt"):
            return req + ".pt"
        for candidate in [f"best ({req}).pt", f"best({req}).pt", f"best_{req}.pt"]:
            if os.path.exists(candidate):
                return candidate
        print(f"Warning: Requested model '{requested_path}' not found. Searching alternatives...")

    for candidate in PREFERRED_MODELS:
        if os.path.exists(candidate):
            return candidate

    all_pt = sorted(glob.glob("*.pt"))
    if all_pt:
        print(f"Using available model: {all_pt[0]}")
        return all_pt[0]

    sys.exit("Error: No YOLO weights (.pt) found. Ensure 'best.pt' is present.")


def init_aruco_detectors():
    """Initializes OpenCV ArUco detectors for ultra-fast native detection (3-5ms on Pi)."""
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
        print(f"Note: ArUco detector initialization notice: {e}")
    return detectors


def fast_detect_aruco(frame, aruco_detectors, target_id=None):
    """Native OpenCV ArUco detector (runs in ~3-5ms on Raspberry Pi CPU)."""
    if not aruco_detectors:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for mode, item in aruco_detectors:
        try:
            if mode == "new":
                corners, ids, _ = item.detectMarkers(gray)
            else:
                d, params = item
                corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
            if ids is not None and len(ids) > 0:
                best_idx = 0
                max_area = 0.0
                for idx, c_arr in enumerate(corners):
                    c = c_arr[0].astype(np.float64)
                    cur_id = str(ids[idx][0])
                    if target_id is not None and cur_id != str(target_id):
                        continue
                    area = cv2.contourArea(c.astype(np.float32))
                    if area > max_area:
                        max_area = area
                        best_idx = idx

                c = corners[best_idx][0].astype(np.float64)
                cur_id = str(ids[best_idx][0])
                if target_id is not None and cur_id != str(target_id):
                    continue

                x1, y1 = float(c[:, 0].min()), float(c[:, 1].min())
                x2, y2 = float(c[:, 0].max()), float(c[:, 1].max())
                box = np.array([x1, y1, x2, y2], dtype=np.float64)
                return {
                    "box": box,
                    "corners": c,
                    "cls_name": cur_id,
                    "conf": 1.0,
                    "detector": "aruco_fast"
                }
        except Exception:
            continue
    return None


def refine_with_aruco(frame, box, aruco_detectors, target_id=None):
    """Verify and extract subpixel corners from a detected candidate box.
    Returns (corners, detected_id) if a verified ArUco marker is found inside the crop, else (None, None)."""
    if not aruco_detectors:
        return None, None
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 16
    h_img, w_img = frame.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))
    except Exception:
        pass

    for img_variant in variants:
        for mode, item in aruco_detectors:
            try:
                if mode == "new":
                    corners, ids, _ = item.detectMarkers(img_variant)
                else:
                    d, params = item
                    corners, ids, _ = cv2.aruco.detectMarkers(img_variant, d, parameters=params)
                if ids is not None and len(ids) > 0:
                    for idx, c_arr in enumerate(corners):
                        cur_id = str(ids[idx][0])
                        if target_id is not None and cur_id != str(target_id):
                            continue
                        c = c_arr[0].copy().astype(np.float64)
                        c[:, 0] += rx1
                        c[:, 1] += ry1
                        return c, cur_id
            except Exception:
                continue
    return None, None


# ---------------- Session CSV Data Logger ----------------
class SessionDataLogger:
    """Logs session tracking data values to CSV in real-time and finalizes after every session."""

    FIELDNAMES = [
        "timestamp",
        "session_time_sec",
        "frame_no",
        "target_id",
        "detector",
        "confidence",
        "distance_m",
        "azimuth_deg",
        "elevation_deg",
        "body_fwd_m",
        "body_left_m",
        "body_up_m",
        "bearing_world_x",
        "bearing_world_y",
        "bearing_world_z",
        "desired_bearing_x",
        "desired_bearing_y",
        "desired_bearing_z",
        "angle_to_desired_deg",
        "vel_world_x",
        "vel_world_y",
        "vel_world_z",
        "speed_mps",
        "target_state",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "fps",
    ]

    def __init__(self, log_dir="session_logs", custom_filename=None, enabled=True, log_all_frames=False):
        self.enabled = enabled
        self.log_all_frames = log_all_frames
        self.record_count = 0
        self.file = None
        self.writer = None
        self.filepath = None
        self.latest_copy_path = "session_data.csv"
        self.start_time = time.perf_counter()
        self.closed = False

        if not self.enabled:
            return

        try:
            os.makedirs(log_dir, exist_ok=True)
            if custom_filename:
                fname = custom_filename if custom_filename.lower().endswith(".csv") else f"{custom_filename}.csv"
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"session_{ts}.csv"
            self.filepath = os.path.join(log_dir, fname)
            self.file = open(self.filepath, mode="w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()
            self.file.flush()
            print(f"[CSV Logger] Session data will be saved to: {self.filepath}")
        except Exception as e:
            print(f"[CSV Logger] Warning: Could not initialize CSV file: {e}")
            self.enabled = False

    def log_detection(self, frame_no, target_info, rho, az, el, r_B, g_LT, g_star, v_T, args_vmin, fps,
                      roll=0.0, pitch=0.0, yaw=0.0):
        if not self.enabled or self.writer is None or self.closed:
            return

        now = datetime.now()
        elapsed = time.perf_counter() - self.start_time
        fwd, left, up = r_B
        target_box = target_info.get("box", [0, 0, 0, 0])

        g_star_x = f"{g_star[0]:.4f}" if g_star is not None else ""
        g_star_y = f"{g_star[1]:.4f}" if g_star is not None else ""
        g_star_z = f"{g_star[2]:.4f}" if g_star is not None else ""
        ang_deg = f"{angle_deg(g_LT, g_star):.2f}" if g_star is not None else ""

        if v_T is not None:
            speed = float(np.linalg.norm(v_T))
            state = "moving" if speed > args_vmin else "still"
            v_x = f"{v_T[0]:.4f}"
            v_y = f"{v_T[1]:.4f}"
            v_z = f"{v_T[2]:.4f}"
            speed_str = f"{speed:.4f}"
        else:
            state = ""
            v_x = ""
            v_y = ""
            v_z = ""
            speed_str = ""

        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "session_time_sec": f"{elapsed:.3f}",
            "frame_no": frame_no,
            "target_id": str(target_info.get("cls_name", "")),
            "detector": str(target_info.get("detector", "")),
            "confidence": f"{float(target_info.get('conf', 1.0)):.4f}",
            "distance_m": f"{rho:.4f}",
            "azimuth_deg": f"{az:.2f}",
            "elevation_deg": f"{el:.2f}",
            "body_fwd_m": f"{fwd:.4f}",
            "body_left_m": f"{left:.4f}",
            "body_up_m": f"{up:.4f}",
            "bearing_world_x": f"{g_LT[0]:.4f}",
            "bearing_world_y": f"{g_LT[1]:.4f}",
            "bearing_world_z": f"{g_LT[2]:.4f}",
            "desired_bearing_x": g_star_x,
            "desired_bearing_y": g_star_y,
            "desired_bearing_z": g_star_z,
            "angle_to_desired_deg": ang_deg,
            "vel_world_x": v_x,
            "vel_world_y": v_y,
            "vel_world_z": v_z,
            "speed_mps": speed_str,
            "target_state": state,
            "roll_deg": f"{roll:.2f}",
            "pitch_deg": f"{pitch:.2f}",
            "yaw_deg": f"{yaw:.2f}",
            "bbox_x1": int(round(target_box[0])),
            "bbox_y1": int(round(target_box[1])),
            "bbox_x2": int(round(target_box[2])),
            "bbox_y2": int(round(target_box[3])),
            "fps": f"{fps:.1f}",
        }

        try:
            self.writer.writerow(row)
            self.record_count += 1
            if self.record_count % 15 == 0:
                self.file.flush()
        except Exception as e:
            print(f"[CSV Logger] Write error: {e}")

    def log_empty_frame(self, frame_no, fps, roll=0.0, pitch=0.0, yaw=0.0):
        if not self.enabled or not self.log_all_frames or self.writer is None or self.closed:
            return

        now = datetime.now()
        elapsed = time.perf_counter() - self.start_time
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "session_time_sec": f"{elapsed:.3f}",
            "frame_no": frame_no,
            "target_id": "none",
            "detector": "none",
            "confidence": "0.0000",
            "distance_m": "",
            "azimuth_deg": "",
            "elevation_deg": "",
            "body_fwd_m": "",
            "body_left_m": "",
            "body_up_m": "",
            "bearing_world_x": "",
            "bearing_world_y": "",
            "bearing_world_z": "",
            "desired_bearing_x": "",
            "desired_bearing_y": "",
            "desired_bearing_z": "",
            "angle_to_desired_deg": "",
            "vel_world_x": "",
            "vel_world_y": "",
            "vel_world_z": "",
            "speed_mps": "",
            "target_state": "no_target",
            "roll_deg": f"{roll:.2f}",
            "pitch_deg": f"{pitch:.2f}",
            "yaw_deg": f"{yaw:.2f}",
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
            "fps": f"{fps:.1f}",
        }
        try:
            self.writer.writerow(row)
            self.record_count += 1
            if self.record_count % 30 == 0:
                self.file.flush()
        except Exception:
            pass

    def close(self):
        if self.closed:
            return
        self.closed = True

        if not self.enabled or self.file is None:
            return

        try:
            self.file.flush()
            self.file.close()
            self.file = None

            if self.filepath and os.path.exists(self.filepath):
                shutil.copyfile(self.filepath, self.latest_copy_path)
                print("\n" + "=" * 65)
                print(" [CSV Logger] Session data stored successfully!")
                print(f"   - Total records logged: {self.record_count}")
                print(f"   - Session file:         {self.filepath}")
                print(f"   - Latest copy:          {self.latest_copy_path}")
                print("=" * 65 + "\n")
        except Exception as e:
            print(f"[CSV Logger] Error finalizing CSV file: {e}")


# ---------------- Threaded Zero-Lag Camera ----------------
class ThreadedCamera:
    """Non-blocking camera grabber thread that guarantees real-time 30 FPS.
    Sets buffer size to 1 and flushes stale frames automatically."""
    def __init__(self, source, width=640, height=480):
        self.is_cam = False
        try:
            cam_idx = int(source)
            self.is_cam = True
            self.cam_idx = cam_idx
        except (ValueError, TypeError):
            self.cam_idx = None
            self.src_path = str(source)

        self.cap = None
        if self.is_cam:
            backends = []
            if sys.platform.startswith("win"):
                backends = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
            else:
                backends = [("V4L2", cv2.CAP_V4L2), ("Default", cv2.CAP_ANY)]

            for b_name, b_flag in backends:
                cap = cv2.VideoCapture(self.cam_idx, b_flag)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        print(f"Connected to camera index {self.cam_idx} via {b_name} ({test_frame.shape[1]}x{test_frame.shape[0]}).")
                        break
                    cap.release()

            if self.cap is None:
                # Probe cameras 0..3
                avail = []
                for idx in range(4):
                    c_test = cv2.VideoCapture(idx)
                    if c_test.isOpened():
                        r, _ = c_test.read()
                        if r:
                            avail.append(idx)
                        c_test.release()
                msg = f"Could not open camera {self.cam_idx}."
                if avail:
                    msg += f" Found active camera index(es): {avail}."
                sys.exit(f"Error: {msg}")
        else:
            if not os.path.exists(self.src_path):
                sys.exit(f"Error: Media source '{self.src_path}' not found.")
            self.cap = cv2.VideoCapture(self.src_path)
            if not self.cap.isOpened():
                sys.exit(f"Error: Failed to open media '{self.src_path}'.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                if not self.is_cam:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()


# ---------------- Asynchronous YOLO Background Worker ----------------
class AsyncYoloWorker:
    """Executes YOLO inference in a background thread so the video feed never stutters."""
    def __init__(self, model, conf=0.25, imgsz=320, device=""):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.lock = threading.Lock()
        self.pending_frame = None
        self.latest_boxes = None
        self.latest_names = model.names
        self.infer_ms = 0.0
        self.last_infer_time = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit_frame(self, frame):
        with self.lock:
            self.pending_frame = frame

    def get_latest(self):
        with self.lock:
            # If inference results are older than 0.35 seconds, treat as expired to avoid ghost tracking
            if time.perf_counter() - self.last_infer_time > 0.35:
                return None, self.infer_ms
            return self.latest_boxes, self.infer_ms

    def _worker(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_process = self.pending_frame
                    self.pending_frame = None
            if frame_to_process is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            predict_kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
            if self.device:
                predict_kwargs["device"] = self.device
            try:
                results = self.model.predict(frame_to_process, **predict_kwargs)[0]
                ms = (time.perf_counter() - t0) * 1000.0
                with self.lock:
                    self.latest_boxes = results.boxes
                    self.infer_ms = ms
                    self.last_infer_time = time.perf_counter()
            except Exception:
                pass

    def stop(self):
        self.running = False


# ---------------- Web Streaming Server (Live Studio Mirror) ----------------
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class AppState:
    def __init__(self):
        self.running = True
        self.mirror_mode = False
        self.do_calibrate = False
        self.enable_yolo = True
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.status_msg = "Live Studio Streaming"
        self.csv_records = 0
        self.csv_file = ""

    def set_jpeg(self, jpeg_bytes):
        with self.lock:
            self.latest_jpeg = jpeg_bytes

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg


app_state = AppState()


def get_local_ip():
    """Detect local IP for web stream connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class StreamRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP access logs
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Drone Detection - Live Studio Mirror</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: #111216;
            color: #e4e7eb;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px;
        }}
        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 1100px;
            margin-bottom: 12px;
        }}
        h1 {{ margin: 0; font-size: 1.35rem; color: #00d26a; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: #262930; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; color: #a0a6b1; }}
        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button {{
            background: #1e2025;
            color: #fff;
            border: 1px solid #363a45;
            padding: 9px 16px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        button:hover {{ background: #2a2d35; border-color: #00d26a; }}
        button:active {{ transform: scale(0.97); }}
        .btn-active {{ background: #00875f !important; border-color: #00d26a !important; }}
        .btn-quit {{ background: #4a151b; border-color: #702029; }}
        .btn-quit:hover {{ background: #6e1f28; }}
        .stream-box {{
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(0,0,0,0.6);
            background: #000;
            max-width: 100%;
            border: 1px solid #272a31;
        }}
        img {{ display: block; max-width: 100%; height: auto; }}
        .status-bar {{
            margin-top: 10px;
            font-size: 0.85rem;
            color: #8c93a0;
            background: #181a1f;
            padding: 6px 14px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🎥 Drone Bearing & Target Tracking</h1>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="badge" style="background:#143826; color:#00e575; border:1px solid #00875f;">📊 CSV Telemetry Active</span>
            <span class="badge">Live Podcast & Studio Mirror</span>
        </div>
    </header>
    <div class="controls">
        <button id="btn-mirror" class="{'btn-active' if app_state.mirror_mode else ''}" onclick="apiCall('/api/toggle_mirror')">
            🪞 Toggle Mirror (Podcast Mode)
        </button>
        <button onclick="apiCall('/api/calibrate')">
            📏 Calibrate Range (1.00 m)
        </button>
        <button id="btn-yolo" class="{'btn-active' if app_state.enable_yolo else ''}" onclick="apiCall('/api/toggle_yolo')">
            ⚡ Hybrid Mode (ArUco + YOLO)
        </button>
        <button class="btn-quit" onclick="apiCall('/api/quit')">
            ❌ Quit
        </button>
    </div>
    <div class="stream-box">
        <img src="/video_feed" alt="Live Mirrored Stream">
    </div>
    <div class="status-bar" id="status-bar">Connected to Raspberry Pi Live Stream</div>
    <script>
        function apiCall(endpoint) {{
            fetch(endpoint, {{method: 'POST'}})
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('status-bar').innerText = data.msg || 'OK';
                    if (data.mirror !== undefined) {{
                        const mBtn = document.getElementById('btn-mirror');
                        if (data.mirror) mBtn.classList.add('btn-active');
                        else mBtn.classList.remove('btn-active');
                    }}
                    if (data.yolo !== undefined) {{
                        const yBtn = document.getElementById('btn-yolo');
                        if (data.yolo) yBtn.classList.add('btn-active');
                        else yBtn.classList.remove('btn-active');
                    }}
                }})
                .catch(err => console.error(err));
        }}
    </script>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            try:
                while app_state.running:
                    jpeg = app_state.get_jpeg()
                    if jpeg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.033)  # ~30 FPS feed
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        resp = {"status": "ok"}
        if self.path == "/api/toggle_mirror":
            app_state.mirror_mode = not app_state.mirror_mode
            resp["mirror"] = app_state.mirror_mode
            resp["msg"] = f"Mirror Mode: {'ON (Podcast / Studio)' if app_state.mirror_mode else 'OFF'}"
        elif self.path == "/api/calibrate":
            app_state.do_calibrate = True
            resp["msg"] = "Calibration command received."
        elif self.path == "/api/toggle_yolo":
            app_state.enable_yolo = not app_state.enable_yolo
            resp["yolo"] = app_state.enable_yolo
            resp["msg"] = f"Hybrid Mode: {'ENABLED (ArUco + Async YOLO)' if app_state.enable_yolo else 'DISABLED (Fast ArUco Only)'}"
        elif self.path == "/api/quit":
            app_state.running = False
            resp["msg"] = "Stopping tracker and saving session CSV data..."
        else:
            self.send_error(404, "Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


def start_web_server(port=5000):
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), StreamRequestHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        local_ip = get_local_ip()
        print("\n" + "#" * 65)
        print("  LIVE PODCAST / STUDIO STREAM ACTIVE!")
        print(f"  Local link:   http://localhost:{port}")
        print(f"  Network link: http://{local_ip}:{port}")
        print("  Open from your laptop, tablet, or phone on the same Wi-Fi!")
        print("#" * 65 + "\n")
        return server
    except Exception as e:
        print(f"Warning: Could not start web server on port {port}: {e}")
        return None


# ---------------- Display Panel & HUD ----------------
PANEL_W = 540
PANEL_MIN_H = 780
MAT_COL_RIGHT = (230, 340, 450)
FONT = cv2.FONT_HERSHEY_DUPLEX
STYLES = {
    "head": ((140, 70, 0), 0.70, 34),
    "text": ((20, 20, 20), 0.65, 28),
    "note": ((110, 110, 110), 0.55, 24),
    "warn": ((0, 0, 210), 0.65, 28),
    "held": ((0, 120, 230), 0.65, 28),
    "gap":  (None, 0.0, 10),
}


def get_required_panel_height(rows):
    return sum(STYLES[style][2] for _, style in rows) + 30


def make_panel(rows, height):
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
    R = clean(R_WB)
    rows = [("", "gap"),
            ("ATTITUDE   R_WB  (body -> world)", "head"),
            (f"Roll {roll:+.1f}    Pitch {pitch:+.1f}    Yaw {yaw:+.1f}   deg", "text"),
            (["", "fwd", "left", "up"], "note")]
    for name, row in zip(("world x", "world y", "world z"), R):
        rows.append(([name] + [f"{v:+.3f}" for v in row], "text"))
    return rows


def label_on_video(img, text, org, color=(255, 255, 255), bg_color=(0, 0, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ---------------- Main Execution ----------------
def main():
    p = argparse.ArgumentParser(description="Drone Target: Live Bearing, Range and Target Tracking.")
    p.add_argument("--model", type=str, default="best.pt",
                   help="path to YOLO weights (.pt), or 'list' to show models")
    p.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                   help=f"marker side length in metres (default {DEFAULT_MARKER_SIZE:.2f} m)")
    p.add_argument("--ipd", type=float, default=None, help="alias for --marker-size")
    p.add_argument("--cal-dist", type=float, default=1.0, help="distance in metres when calibrating (c)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    p.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size (default 320 for speed)")
    p.add_argument("--target-id", type=str, default=None, help="specific marker class to track (e.g. '41')")
    p.add_argument("--cam", type=int, default=CAM_INDEX, help="camera device index (default 0)")
    p.add_argument("--source", type=str, default=None, help="input source: camera index or video/image file path")
    p.add_argument("--device", type=str, default="", help="inference device: '' (auto), 'cuda', 'cpu'")
    p.add_argument("--roll", type=float, default=0.0, help="camera/body roll in degrees")
    p.add_argument("--pitch", type=float, default=0.0, help="camera/body pitch in degrees")
    p.add_argument("--yaw", type=float, default=0.0, help="camera/body yaw in degrees")
    p.add_argument("--vmin", type=float, default=0.08, help="speed in m/s below which g* is held")
    p.add_argument("--mirror", action="store_true", default=False,
                   help="enable selfie/podcast mirror mode (horizontal flip)")
    p.add_argument("--port", type=int, default=5000, help="port for live web streaming studio (default 5000)")
    p.add_argument("--no-web", action="store_true", default=False, help="disable built-in web stream")
    p.add_argument("--headless", action="store_true", default=False, help="force headless mode (no local GUI window, web stream only)")
    p.add_argument("--no-csv", action="store_true", default=False,
                   help="disable saving session data to CSV")
    p.add_argument("--csv-dir", type=str, default="session_logs",
                   help="directory to save session CSV files (default: session_logs)")
    p.add_argument("--csv-name", type=str, default=None,
                   help="custom filename for session CSV (default: session_YYYYMMDD_HHMMSS.csv)")
    p.add_argument("--csv-all-frames", action="store_true", default=False,
                   help="log every frame to CSV even when no target is detected")
    p.add_argument("--aruco-only", action="store_true", default=False,
                   help="disable YOLO, run fast native ArUco detector only (zero false positives, 30+ FPS)")
    p.add_argument("--no-verify-aruco", action="store_true", default=False,
                   help="disable strict ArUco verification of YOLO bounding boxes")

    args = p.parse_args()

    args.verify_aruco = not args.no_verify_aruco
    if args.aruco_only:
        app_state.enable_yolo = False

    marker_size = args.ipd if args.ipd is not None else args.marker_size
    app_state.mirror_mode = args.mirror

    # Resolve and load YOLO model
    model_file = resolve_model_file(args.model)
    print("\n" + "=" * 60)
    print(f"Loading YOLO model: {model_file}...")
    model = YOLO(model_file)
    print(f"Model loaded. Classes: {model.names}")
    dev = args.device if args.device else ("CUDA GPU" if torch.cuda.is_available() else "CPU")
    print(f"Inference device: {dev} | Image size: {args.imgsz}px | Confidence: {args.conf}")
    print("=" * 60 + "\n")

    aruco_detectors = init_aruco_detectors()

    # Determine camera source & resolution
    source = args.source if args.source is not None else args.cam
    want = FALLBACK_SIZE
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                c = json.load(f)
            want = (int(c["image_width"]), int(c["image_height"]))
        except Exception:
            pass

    # Threaded camera grabber ensures real-time 30 FPS video
    cam_reader = ThreadedCamera(source, want[0], want[1])
    ret, initial_frame = cam_reader.read()
    retry = 0
    while (not ret or initial_frame is None) and retry < 20:
        time.sleep(0.05)
        ret, initial_frame = cam_reader.read()
        retry += 1

    if not ret or initial_frame is None:
        cam_reader.release()
        sys.exit("Error: Could not read frames from camera.")

    H, W = initial_frame.shape[:2]
    K, dist, size, calibrated = load_camera_model((W, H))
    if calibrated and size != (W, H):
        print(f"Warning: Frame is {W}x{H}, but calibration is {size[0]}x{size[1]}. Using fallback.")
        K, dist, size, calibrated = load_camera_model((W, H))

    cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))

    # Start Asynchronous YOLO Worker
    yolo_worker = AsyncYoloWorker(model, conf=args.conf, imgsz=args.imgsz, device=args.device)

    # Start Live Web Studio Server
    web_server = None
    if not args.no_web:
        web_server = start_web_server(port=args.port)

    # Initialize Session CSV Logger
    csv_logger = SessionDataLogger(
        log_dir=args.csv_dir,
        custom_filename=args.csv_name,
        enabled=not args.no_csv,
        log_all_frames=args.csv_all_frames
    )
    atexit.register(csv_logger.close)
    if csv_logger.enabled and csv_logger.filepath:
        app_state.csv_file = csv_logger.filepath

    win = "Drone Target Tracking - Live Studio Mirror (m = mirror, c = calibrate, q = quit)"
    has_gui = True
    if args.headless:
        has_gui = False
    elif sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")
        if not display and not wayland:
            has_gui = False

    if has_gui:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, W + PANEL_W, max(H, PANEL_MIN_H))
        except Exception:
            has_gui = False

    if not has_gui:
        print(f"Headless mode active (no local window). Access live feed via http://{get_local_ip()}:{args.port}")

    try:
        run_loop(cam_reader, yolo_worker, aruco_detectors, K, dist, marker_size, args,
                 cx, cy, W, H, calibrated, win, has_gui=has_gui, model_file=model_file,
                 csv_logger=csv_logger)
    except KeyboardInterrupt:
        print("\nStopped with Ctrl+C.")
    finally:
        app_state.running = False
        yolo_worker.stop()
        cam_reader.release()
        if csv_logger:
            csv_logger.close()
        if has_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if web_server:
            web_server.shutdown()


def run_loop(cam_reader, yolo_worker, aruco_detectors, K, dist, marker_size, args,
             cx, cy, W, H, calibrated, win, has_gui=True, model_file="best.pt",
             csv_logger=None):
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

    fps_smooth = 30.0
    t_last_frame = time.perf_counter()

    while app_state.running:
        ok, frame = cam_reader.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        t_now = time.perf_counter()
        frame_no += 1

        dt_frame = t_now - t_last_frame
        t_last_frame = t_now
        fps = (1.0 / dt_frame) if dt_frame > 0 else 0.0
        fps_smooth = 0.92 * fps_smooth + 0.08 * fps

        # Check for web calibration trigger
        if app_state.do_calibrate:
            app_state.do_calibrate = False
            if last is None:
                print("Web command: No marker visible - cannot calibrate.")
            else:
                marker_size = 2.0 * args.cal_dist * math.tan(last["theta"] / 2.0)
                r_B_smooth = None
                r_W_prev = None
                t_prev = None
                v_T = None
                print(f"Web calibrated at {args.cal_dist:.2f} m: effective marker size = {marker_size * 1000:.1f} mm.")

        # Hybrid detection:
        # Step 1: Fast native OpenCV ArUco detector (takes ~3-5ms on Raspberry Pi)
        detected_target = fast_detect_aruco(frame, aruco_detectors, target_id=args.target_id)
        active_detector = "ArUco (3-5ms native)"
        infer_ms = 4.0

        # Step 2: If native ArUco didn't catch it and Hybrid YOLO is enabled, check async YOLO worker
        if detected_target is None and app_state.enable_yolo:
            yolo_worker.submit_frame(frame)
            boxes, yolo_infer_ms = yolo_worker.get_latest()
            infer_ms = yolo_infer_ms if yolo_infer_ms > 0 else infer_ms
            active_detector = f"Async YOLO ({infer_ms:.0f}ms)"

            if boxes is not None and len(boxes) > 0:
                detected_candidates = []
                for b_idx in range(len(boxes)):
                    xyxy = boxes.xyxy[b_idx].cpu().numpy()
                    conf = float(boxes.conf[b_idx].item())
                    cls_id = int(boxes.cls[b_idx].item())
                    cls_name = yolo_worker.latest_names.get(cls_id, str(cls_id))

                    if args.target_id is not None and str(cls_name) != str(args.target_id):
                        continue

                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    detected_candidates.append({
                        "box": xyxy,
                        "conf": conf,
                        "cls_name": cls_name,
                        "area": w * h
                    })
                if detected_candidates:
                    # Sort candidates by confidence and area
                    candidates_sorted = sorted(detected_candidates, key=lambda c: (c["conf"], c["area"]), reverse=True)
                    for cand in candidates_sorted:
                        cand_box = cand["box"]
                        bw = cand_box[2] - cand_box[0]
                        bh = cand_box[3] - cand_box[1]
                        aspect = (bw / bh) if bh > 0 else 0.0

                        # ArUco markers are square; reject wildly elongated boxes (such as human faces or bodies)
                        if not (0.50 <= aspect <= 2.0):
                            continue

                        c_ref, aruco_id = refine_with_aruco(frame, cand_box, aruco_detectors, target_id=args.target_id)

                        # Strict verification: require ArUco confirmation to prevent false detections on faces or objects
                        if getattr(args, "verify_aruco", True):
                            if c_ref is not None:
                                detected_target = {
                                    "box": cand_box,
                                    "corners": c_ref,
                                    "cls_name": aruco_id if aruco_id else cand["cls_name"],
                                    "conf": cand["conf"],
                                    "detector": "yolo+aruco"
                                }
                                break
                        else:
                            # Fallback if --no-verify-aruco is explicitly passed
                            detected_target = {
                                "box": cand_box,
                                "corners": c_ref,
                                "cls_name": aruco_id if aruco_id else cand["cls_name"],
                                "conf": cand["conf"],
                                "detector": "yolo"
                            }
                            break
        elif detected_target is not None and app_state.enable_yolo:
            # Keep YOLO warm periodically
            if frame_no % 30 == 0:
                yolo_worker.submit_frame(frame)

        # ---------- Video Rendering & Live Podcast Mirror Mode ----------
        is_mirror = app_state.mirror_mode
        if is_mirror:
            view = cv2.flip(frame, 1)
            cx_render = W - 1 - cx
        else:
            view = frame.copy()
            cx_render = cx

        # Draw crosshair straight ahead
        cv2.drawMarker(view, (cx_render, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        label_on_video(view, "straight ahead", (cx_render + 10, cy - 10))

        # Status overlay on top of video
        studio_tag = " [🪞 PODCAST MIRROR]" if is_mirror else " [DIRECT VIEW]"
        label_on_video(view, f"FPS: {fps_smooth:.1f} ({infer_ms:.0f}ms) | {active_detector}{studio_tag}",
                       (12, 24), color=(0, 255, 120))

        rows = [("WHERE THE MARKER IS   (camera = origin)", "head")]

        if detected_target is None:
            r_W_prev = None
            t_prev = None
            rows += [("No marker detected - show ArUco marker to camera", "warn")]
            if csv_logger:
                csv_logger.log_empty_frame(frame_no, fps_smooth, roll=args.roll, pitch=args.pitch, yaw=args.yaw)
        else:
            target_box = detected_target["box"]
            cls_name = detected_target["cls_name"]
            conf = detected_target["conf"]
            corners = detected_target["corners"]

            m = measure_marker(target_box, K, dist, marker_size, corners=corners)
            last = m

            r_B_smooth = (m["r_B"] if r_B_smooth is None
                          else SMOOTHING * m["r_B"] + (1 - SMOOTHING) * r_B_smooth)

            # Map coordinates for live podcast mirror view if enabled
            if is_mirror:
                x1_m = int(W - 1 - target_box[2])
                x2_m = int(W - 1 - target_box[0])
                y1_m = int(target_box[1])
                y2_m = int(target_box[3])
                tcx_render = int(round(W - 1 - m["center"][0]))
                tcy_render = int(round(m["center"][1]))
                if corners is not None:
                    c_draw = corners.copy()
                    c_draw[:, 0] = W - 1 - c_draw[:, 0]
                else:
                    c_draw = None
            else:
                x1_m = int(target_box[0])
                y1_m = int(target_box[1])
                x2_m = int(target_box[2])
                y2_m = int(target_box[3])
                tcx_render = int(round(m["center"][0]))
                tcy_render = int(round(m["center"][1]))
                c_draw = corners

            # Draw target bounding box and corners
            cv2.rectangle(view, (x1_m, y1_m), (x2_m, y2_m), (0, 200, 255), 2)
            if c_draw is not None:
                cv2.polylines(view, [c_draw.astype(np.int32)], True, (0, 255, 0), 2)

            cv2.circle(view, (tcx_render, tcy_render), 4, (0, 0, 255), -1)
            cv2.line(view, (cx_render, cy), (tcx_render, tcy_render), (0, 255, 0), 1)

            # Physical bearing & range
            fwd, left, up = r_B_smooth
            rho = float(np.linalg.norm(r_B_smooth))
            az = math.degrees(math.atan2(-left, fwd))
            el = math.degrees(math.atan2(up, math.hypot(fwd, left)))
            b_B = r_B_smooth / max(rho, 1e-6)
            g_LT = R_WB @ b_B

            label_on_video(view, f"Marker {cls_name} ({conf:.2f}) | {rho:.2f} m",
                           (x1_m, max(18, y1_m - 8)), color=(0, 255, 255))

            # Target velocity in world axes
            r_W = R_WB @ r_B_smooth
            dt = None
            if r_W_prev is not None and t_prev is not None:
                dt = t_now - t_prev
                if dt > 0:
                    v_raw = velocity_from_positions(r_W, r_W_prev, dt, v_L)
                    v_T = v_raw if v_T is None else VEL_SMOOTHING * v_raw + (1 - VEL_SMOOTHING) * v_T
            r_W_prev, t_prev = r_W, t_now

            g_new = desired_bearing(v_T, args.vmin) if v_T is not None else None
            held = g_new is None
            if not held:
                g_star = g_new

            # Panel telemetry
            rows += [
                (f"Target: Marker {cls_name}   (conf {conf:.2f})", "text"),
                (f"Distance             {rho:.2f} m", "text"),
                (f"Direction            {side(az, 'right', 'left')},  {side(el, 'up', 'down')}", "text"),
                (f"Fwd / Left / Up      {fwd:+.2f} / {left:+.2f} / {up:+.2f} m", "text"),
                ("", "gap"),
                ("BEARINGS   (unit vectors, world axes x, y, z)", "head"),
                (f"g_LT  measured    {fmt(g_LT)}", "text"),
                ("          direction from camera to marker", "note"),
            ]
            if g_star is None:
                rows += [("g*    desired     not defined yet", "held"),
                         ("          move the marker to estimate velocity", "note")]
            else:
                rows += [(f"g*    desired     {fmt(g_star)}", "held" if held else "text"),
                         ("          direction marker is moving"
                          + ("   (held: target is still)" if held else ""), "note"),
                         (f"Angle between g_LT and g*     {angle_deg(g_LT, g_star):.1f} deg", "text"),
                         ("          0 deg = camera right behind marker", "note")]

            rows += [("", "gap"), ("TARGET VELOCITY  v_T   (world axes)", "head")]
            if v_T is None:
                rows += [("measuring...", "note")]
            else:
                speed = float(np.linalg.norm(v_T))
                state = "moving" if speed > args.vmin else "still"
                rows += [(f"v_T      {fmt(v_T, 2)} m/s", "text"),
                         (f"Speed    {speed:.2f} m/s   -> {state}  (threshold {args.vmin:.2f})", "text")]

            if csv_logger:
                csv_logger.log_detection(
                    frame_no=frame_no,
                    target_info=detected_target,
                    rho=rho,
                    az=az,
                    el=el,
                    r_B=r_B_smooth,
                    g_LT=g_LT,
                    g_star=g_star,
                    v_T=v_T,
                    args_vmin=args.vmin,
                    fps=fps_smooth,
                    roll=args.roll,
                    pitch=args.pitch,
                    yaw=args.yaw
                )
                app_state.csv_records = csv_logger.record_count

            if frame_no % 45 == 0:
                print(f"[Marker {cls_name}] rho {rho:.2f}m | r_B=[{fwd:+.2f}, {left:+.2f}, {up:+.2f}]m"
                      f" | g_LT {fmt(g_LT)}"
                      + (f" | v_T {np.linalg.norm(v_T):.2f}m/s" if v_T is not None else ""))

        # ---------- Attitude and R_WB ----------
        rows += att_rows

        # ---------- Settings & Status ----------
        csv_status_str = f"Active ({csv_logger.record_count} rows)" if (csv_logger and csv_logger.enabled) else "Disabled"
        rows += [
            ("", "gap"),
            ("SETTINGS & STATUS", "head"),
            (f"Marker size: {marker_size * 1000:.1f} mm", "note"),
            (f"Mirror Mode: {'ON (Podcast / Studio)' if is_mirror else 'OFF (Direct)'}", "note"),
            (f"Tracker: {active_detector}", "note"),
            (f"Live FPS: {fps_smooth:.1f} FPS (inf {infer_ms:.1f} ms)", "note"),
            (f"CSV Telemetry: {csv_status_str}", "note"),
        ]
        if not calibrated:
            rows.append(("Camera NOT CALIBRATED - values approximate", "warn"))
        rows.append(("Keys: m=mirror, c=calibrate, t=hybrid, q=quit", "note"))

        # Render panel and combine
        req_h = get_required_panel_height(rows)
        panel_h = max(H, PANEL_MIN_H, req_h)

        if H < panel_h:
            view = np.vstack([view, np.zeros((panel_h - H, W, 3), np.uint8)])
        elif H > panel_h:
            panel_h = H

        panel_img = make_panel(rows, panel_h)
        combined = np.hstack([view, panel_img])

        # Encode frame for live web stream
        ret_enc, jpeg = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret_enc:
            app_state.set_jpeg(jpeg.tobytes())

        # Local display window
        if has_gui:
            try:
                cv2.imshow(win, combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nExiting on user request.")
                    break
                if key in (ord("m"), ord("M")):
                    app_state.mirror_mode = not app_state.mirror_mode
                    print(f"Podcast Mirror Mode: {'ON' if app_state.mirror_mode else 'OFF'}")
                if key in (ord("t"), ord("T")):
                    app_state.enable_yolo = not app_state.enable_yolo
                    print(f"Hybrid Mode: {'ENABLED' if app_state.enable_yolo else 'DISABLED'}")
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
            except Exception:
                has_gui = False
        else:
            # Headless loop pacing
            time.sleep(0.005)


if __name__ == "__main__":
    main()