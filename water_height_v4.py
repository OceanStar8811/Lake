#!/usr/bin/env python3
"""
water_height_v4.py

Water-level measurement from a floating cylindrical target.

V4 design
----------
- Fixed, calibrated camera.
- Optional lens undistortion using the camera calibration.
- Yellow circular top -> ellipse detection.
- Target is allowed to move freely inside the camera FOV.
- Water level is assumed approximately constant during one measurement window.
- One robust water-level result is produced every MEASUREMENT_WINDOW_SEC.
- Target diameter is an explicit configurable parameter.
- Frame-level measurements are retained for diagnostics.
- Calibration learns an empirical relationship from image geometry to
  camera-to-water vertical distance.

Important:
----------
A completely free-floating target creates a monocular scale ambiguity:
horizontal target motion toward/away from the camera can resemble a water-level
change. V4 therefore does NOT claim that target diameter alone makes the
problem uniquely observable. The calibration must contain representative
target motion, and the 5-10 s window estimator reduces motion/rocking noise.

Recommended physical configuration:
- camera fixed rigidly;
- zoom/focus fixed;
- camera intrinsics/distortion known;
- calibration and operation use the same camera geometry;
- target diameter set correctly;
- target remains visible for a reasonable fraction of each measurement window.

Commands
========

1. Calibrate from a video with known water heights:

    python water_height_v4.py calibrate \
        calibration.mp4 calibration.json model_v4.npz \
        --camera camera_calibration.npz

JSON:
{
  "frames": [
    {
      "frame_index": 0,
      "camera_to_water_vertical_height_m": 12.0
    }
  ]
}

camera_calibration.npz:
    camera_matrix = 3x3
    dist_coeffs   = distortion coefficients

2. Measure:

    python water_height_v4.py measure \
        video.mp4 model_v4.npz measurements.csv \
        --camera camera_calibration.npz \
        --window 10 \
        --annotated annotated.mp4

3. Change target diameter:

    --target-diameter 0.75

The target diameter is stored in the model as metadata and can also be
overridden during measurement.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Default configuration
# ============================================================

TARGET_DIAMETER_M = 0.75

MEASUREMENT_WINDOW_SEC = 10.0
MIN_VALID_FRAMES_PER_WINDOW = 15

# Yellow HSV range.
YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA_PX = 35
MAX_TARGET_AREA_FRACTION = 0.15
MIN_AXIS_RATIO = 0.20

# Reject implausible temporal jumps in the detector.
MAX_MAJOR_JUMP_FRACTION = 0.45

# Calibration bins.
HEIGHT_BIN_M = 0.20

# Robust regression.
RIDGE_LAMBDA = 0.10

# Window outlier rejection.
MAD_SCALE = 3.5

MODEL_VERSION = "v4"


# ============================================================
# Camera calibration
# ============================================================

def load_camera_calibration(path):
    if not path:
        return None

    data = np.load(path)

    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise RuntimeError(
            "Camera calibration NPZ must contain camera_matrix and dist_coeffs."
        )

    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64)

    if K.shape != (3, 3):
        raise RuntimeError("camera_matrix must be 3x3.")

    return K, dist


def undistort_frame(frame, camera):
    if camera is None:
        return frame

    K, dist = camera
    return cv2.undistort(frame, K, dist)


# ============================================================
# Target detection
# ============================================================

def detect_yellow_top(frame):
    """
    Return the best yellow ellipse candidate.

    Returned fields:
        center_px
        major_px
        minor_px
        axis_ratio
        angle_deg
        contour_area_px
        quality
        ellipse
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    H, W = frame.shape[:2]
    max_area = H * W * MAX_TARGET_AREA_FRACTION

    best = None
    best_score = -np.inf

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < MIN_TARGET_AREA_PX or area > max_area:
            continue
        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        (u, v), (a, b), angle = ellipse

        major = float(max(a, b))
        minor = float(min(a, b))

        if major <= 1.0 or minor <= 1.0:
            continue

        ratio = minor / major

        if ratio < MIN_AXIS_RATIO:
            continue

        ellipse_area = math.pi * major * minor / 4.0
        if ellipse_area <= 1.0:
            continue

        fill = area / ellipse_area
        fill_quality = max(0.0, 1.0 - abs(fill - 1.0))

        perimeter = float(cv2.arcLength(contour, True))
        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )
        compactness = float(np.clip(compactness, 0.0, 1.0))

        quality = float(
            0.65 * fill_quality + 0.35 * compactness
        )

        score = area * (0.3 + 0.7 * quality)

        if score > best_score:
            best_score = score
            best = {
                "center_px": (float(u), float(v)),
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": float(ratio),
                "angle_deg": float(angle),
                "contour_area_px": area,
                "quality": quality,
                "ellipse": ellipse,
            }

    return best


# ============================================================
# Robust statistics
# ============================================================

def robust_median(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.median(x))


def robust_mad(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_filter(values, scale=MAD_SCALE):
    """
    Keep values inside median +/- scale * robust_sigma,
    where robust_sigma = 1.4826 * MAD.

    If MAD is zero or too small, all finite values are retained.
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)

    if np.count_nonzero(finite) < 3:
        return finite

    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med))

    if mad < 1e-9:
        return finite

    sigma = 1.4826 * mad
    return finite & (np.abs(x - med) <= scale * sigma)


# ============================================================
# Feature construction
# ============================================================

def raw_features(det, width, height):
    """
    Features intentionally use image measurements rather than assuming a
    known target horizontal location.

    The target diameter is stored separately as model metadata. It is included
    in the feature vector so changing target size requires explicit model
    compatibility.
    """
    u, v = det["center_px"]
    major = det["major_px"]
    minor = det["minor_px"]
    ratio = det["axis_ratio"]
    angle = math.radians(det["angle_deg"])

    # Normalize image coordinates to roughly [-1, 1].
    x = 2.0 * (u / max(width - 1, 1)) - 1.0
    y = 2.0 * (v / max(height - 1, 1)) - 1.0

    # Scale features are log-based so large/small targets are numerically tame.
    log_major = math.log(max(major, 1e-6))
    log_minor = math.log(max(minor, 1e-6))

    # sin/cos avoid angle discontinuity.
    sa = math.sin(2.0 * angle)
    ca = math.cos(2.0 * angle)

    return np.array([
        log_major,
        log_minor,
        ratio,
        x,
        y,
        sa,
        ca,
        x * log_major,
        y * log_major,
        x * ratio,
        y * ratio,
        x * y,
        ratio * log_major,
    ], dtype=np.float64)


def feature_matrix(detections, width, height):
    return np.vstack([
        raw_features(d, width, height)
        for d in detections
    ])


# ============================================================
# Calibration
# ============================================================

def read_known_heights(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    result = {}

    for row in meta.get("frames", []):
        if (
            "frame_index" in row
            and "camera_to_water_vertical_height_m" in row
        ):
            result[int(row["frame_index"])] = float(
                row["camera_to_water_vertical_height_m"]
            )

    if not result:
        raise RuntimeError(
            "No frame_index / camera_to_water_vertical_height_m records found."
        )

    return result


def ridge_fit(X, y, lam=RIDGE_LAMBDA):
    """
    Standardize non-intercept features, then solve ridge regression.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std < 1e-9] = 1.0

    Z = (X - mean) / std

    # Explicit intercept.
    Z1 = np.column_stack([np.ones(len(Z)), Z])

    reg = np.eye(Z1.shape[1], dtype=np.float64) * lam
    reg[0, 0] = lam * 0.01

    beta = np.linalg.solve(
        Z1.T @ Z1 + reg,
        Z1.T @ y
    )

    return mean, std, beta


def ridge_predict(X, mean, std, beta):
    X = np.asarray(X, dtype=np.float64)
    Z = (X - mean) / std
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    return Z1 @ beta


def calibrate(video_path, json_path, model_path, camera_path,
              target_diameter_m):
    if target_diameter_m <= 0:
        raise ValueError("target diameter must be > 0.")

    camera = load_camera_calibration(camera_path)
    known = read_known_heights(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    y = []
    frame_ids = []

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in known:
            frame = undistort_frame(frame, camera)
            det = detect_yellow_top(frame)

            if det is not None:
                detections.append(det)
                y.append(known[idx])
                frame_ids.append(idx)

        idx += 1

    cap.release()

    if len(detections) < 30:
        raise RuntimeError(
            f"Only {len(detections)} usable calibration samples; "
            "at least 30 are recommended."
        )

    X = feature_matrix(detections, width, height)
    y = np.asarray(y, dtype=np.float64)

    # First fit.
    mean, std, beta = ridge_fit(X, y)

    pred = ridge_predict(X, mean, std, beta)
    residual = pred - y

    # Robust second fit after rejecting gross calibration/detection outliers.
    keep = robust_filter(residual)

    if np.count_nonzero(keep) >= 20:
        mean, std, beta = ridge_fit(
            X[keep],
            y[keep]
        )
        pred = ridge_predict(X, mean, std, beta)
        residual = pred - y

    rmse = float(np.sqrt(np.mean(residual[np.isfinite(residual)] ** 2)))
    mad = robust_mad(residual)

    np.savez(
        model_path,
        model_version=np.array([MODEL_VERSION]),
        target_diameter_m=np.array([target_diameter_m], dtype=np.float64),
        image_width=np.array([width], dtype=np.int32),
        image_height=np.array([height], dtype=np.int32),
        feature_mean=mean,
        feature_std=std,
        beta=beta,
        calibration_min_height_m=np.array([np.min(y)]),
        calibration_max_height_m=np.array([np.max(y)]),
        calibration_samples=np.array([len(y)], dtype=np.int32),
        calibration_used=np.array([np.count_nonzero(keep)], dtype=np.int32),
        calibration_rmse_m=np.array([rmse]),
        calibration_mad_m=np.array([mad]),
        camera_calibration_used=np.array(
            [1 if camera is not None else 0], dtype=np.int32
        ),
    )

    print("V4 calibration complete")
    print(f"  model                  : {model_path}")
    print(f"  target diameter        : {target_diameter_m:.3f} m")
    print(f"  samples                : {len(y)}")
    print(f"  samples used           : {np.count_nonzero(keep)}")
    print(f"  height range           : {np.min(y):.3f} .. {np.max(y):.3f} m")
    print(f"  calibration RMSE       : {rmse:.4f} m")
    print(f"  calibration MAD        : {mad:.4f} m")
    print(f"  undistortion           : {'yes' if camera is not None else 'no'}")

    if rmse > 0.25:
        print(
            "  WARNING: calibration error is relatively high. "
            "Check target motion, camera stability, and calibration coverage."
        )


# ============================================================
# Model loading
# ============================================================

def load_model(path):
    d = np.load(path)

    version = str(d["model_version"][0])

    if version != MODEL_VERSION:
        raise RuntimeError(
            f"Expected model {MODEL_VERSION}, got {version}."
        )

    return {
        "target_diameter_m": float(d["target_diameter_m"][0]),
        "image_width": int(d["image_width"][0]),
        "image_height": int(d["image_height"][0]),
        "feature_mean": np.asarray(d["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(d["feature_std"], dtype=np.float64),
        "beta": np.asarray(d["beta"], dtype=np.float64),
        "height_min": float(d["calibration_min_height_m"][0]),
        "height_max": float(d["calibration_max_height_m"][0]),
        "calibration_samples": int(d["calibration_samples"][0]),
        "calibration_used": int(d["calibration_used"][0]),
        "calibration_rmse": float(d["calibration_rmse_m"][0]),
        "calibration_mad": float(d["calibration_mad_m"][0]),
        "undistortion_used": bool(d["camera_calibration_used"][0]),
    }


def predict_detection(det, model, width, height):
    X = raw_features(det, width, height)[None, :]
    h = float(
        ridge_predict(
            X,
            model["feature_mean"],
            model["feature_std"],
            model["beta"],
        )[0]
    )

    # Do not extrapolate far outside the calibration range.
    margin = max(0.10 * (model["height_max"] - model["height_min"]), 0.20)

    if h < model["height_min"] - margin:
        return np.nan, "below_calibration_range"

    if h > model["height_max"] + margin:
        return np.nan, "above_calibration_range"

    return h, ""


# ============================================================
# Temporal detection stabilization
# ============================================================

def accept_temporal_detection(det, previous):
    if det is None or previous is None:
        return det

    old = previous["major_px"]
    new = det["major_px"]

    if old <= 0:
        return det

    jump = abs(new - old) / old

    if jump > MAX_MAJOR_JUMP_FRACTION:
        return None

    return det


# ============================================================
# Measurement windows
# ============================================================

def summarize_window(rows, start_frame, end_frame, fps):
    """
    Produce one water-level result for one time window.

    Rows must already contain frame-level predicted heights.
    """
    selected = [
        r for r in rows
        if start_frame <= r["frame"] < end_frame
        and np.isfinite(r["height_m"])
    ]

    total_frames = max(0, end_frame - start_frame)
    valid = len(selected)

    if valid < MIN_VALID_FRAMES_PER_WINDOW:
        return {
            "window_start_s": start_frame / fps,
            "window_end_s": end_frame / fps,
            "water_level_m": np.nan,
            "water_level_std_m": np.nan,
            "water_level_mad_m": np.nan,
            "valid_frames": valid,
            "total_frames": total_frames,
            "valid_fraction": valid / total_frames if total_frames else 0.0,
            "target_major_median_px": np.nan,
            "target_axis_ratio_median": np.nan,
            "target_u_median_px": np.nan,
            "target_v_median_px": np.nan,
            "motion_major_std_px": np.nan,
            "motion_u_std_px": np.nan,
            "motion_v_std_px": np.nan,
            "confidence": 0.0,
            "status": "insufficient_frames",
        }

    h = np.asarray([r["height_m"] for r in selected], dtype=np.float64)

    # Robustly remove frame-level outliers.
    keep = robust_filter(h)
    h2 = h[keep]

    if h2.size < MIN_VALID_FRAMES_PER_WINDOW:
        return {
            "window_start_s": start_frame / fps,
            "window_end_s": end_frame / fps,
            "water_level_m": np.nan,
            "water_level_std_m": np.nan,
            "water_level_mad_m": np.nan,
            "valid_frames": int(h2.size),
            "total_frames": total_frames,
            "valid_fraction": float(h2.size / total_frames),
            "target_major_median_px": np.nan,
            "target_axis_ratio_median": np.nan,
            "target_u_median_px": np.nan,
            "target_v_median_px": np.nan,
            "motion_major_std_px": np.nan,
            "motion_u_std_px": np.nan,
            "motion_v_std_px": np.nan,
            "confidence": 0.0,
            "status": "insufficient_after_outlier_rejection",
        }

    selected2 = [selected[i] for i in np.where(keep)[0]]

    major = np.asarray(
        [r["major_px"] for r in selected2], dtype=np.float64
    )
    ratio = np.asarray(
        [r["axis_ratio"] for r in selected2], dtype=np.float64
    )
    u = np.asarray(
        [r["u_px"] for r in selected2], dtype=np.float64
    )
    v = np.asarray(
        [r["v_px"] for r in selected2], dtype=np.float64
    )
    quality = np.asarray(
        [r["quality"] for r in selected2], dtype=np.float64
    )

    # The window estimate is deliberately a median, not a mean.
    water = float(np.median(h2))

    std = float(np.std(h2))
    mad = robust_mad(h2)

    valid_fraction = float(h2.size / total_frames)

    # Detection quality.
    q = float(np.clip(np.median(quality), 0.0, 1.0))

    # Stability: low within-window height spread is better.
    # 0.30 m is deliberately a conservative normalization.
    stability = float(
        np.clip(1.0 - std / 0.30, 0.0, 1.0)
    )

    coverage = float(
        np.clip(valid_fraction / 0.70, 0.0, 1.0)
    )

    confidence = float(
        np.clip(
            0.45 * q +
            0.35 * stability +
            0.20 * coverage,
            0.0,
            1.0,
        )
    )

    return {
        "window_start_s": start_frame / fps,
        "window_end_s": end_frame / fps,
        "water_level_m": water,
        "water_level_std_m": std,
        "water_level_mad_m": mad,
        "valid_frames": int(h2.size),
        "total_frames": total_frames,
        "valid_fraction": valid_fraction,
        "target_major_median_px": float(np.median(major)),
        "target_axis_ratio_median": float(np.median(ratio)),
        "target_u_median_px": float(np.median(u)),
        "target_v_median_px": float(np.median(v)),
        "motion_major_std_px": float(np.std(major)),
        "motion_u_std_px": float(np.std(u)),
        "motion_v_std_px": float(np.std(v)),
        "confidence": confidence,
        "status": "ok",
    }


def measure(video_path, model_path, csv_path, camera_path=None,
            window_sec=MEASUREMENT_WINDOW_SEC,
            target_diameter_override=None,
            annotated_path=None):
    model = load_model(model_path)

    if target_diameter_override is not None:
        target_diameter = float(target_diameter_override)
        if abs(target_diameter - model["target_diameter_m"]) > 1e-9:
            print(
                "WARNING: target diameter override differs from the "
                "diameter used to build the calibration model."
            )
    else:
        target_diameter = model["target_diameter_m"]

    camera = load_camera_calibration(camera_path)

    if model["undistortion_used"] and camera is None:
        print(
            "WARNING: model was calibrated with camera undistortion, "
            "but no camera calibration was supplied for measurement."
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width != model["image_width"] or height != model["image_height"]:
        print(
            f"WARNING: video resolution {width}x{height} differs from "
            f"calibration resolution {model['image_width']}x"
            f"{model['image_height']}."
        )

    rows = []
    previous = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        original = frame.copy()
        frame = undistort_frame(frame, camera)

        det = detect_yellow_top(frame)
        det = accept_temporal_detection(det, previous)

        if det is None:
            row = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "u_px": np.nan,
                "v_px": np.nan,
                "major_px": np.nan,
                "minor_px": np.nan,
                "axis_ratio": np.nan,
                "angle_deg": np.nan,
                "quality": np.nan,
                "height_m": np.nan,
                "reason": "target_not_found_or_jump",
                "ellipse": None,
            }
        else:
            h, reason = predict_detection(
                det, model, width, height
            )

            u, v = det["center_px"]

            row = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "u_px": u,
                "v_px": v,
                "major_px": det["major_px"],
                "minor_px": det["minor_px"],
                "axis_ratio": det["axis_ratio"],
                "angle_deg": det["angle_deg"],
                "quality": det["quality"],
                "height_m": h,
                "reason": reason,
                "ellipse": det["ellipse"],
            }

            if np.isfinite(h):
                previous = det

        rows.append(row)
        frame_idx += 1

    cap.release()

    window_frames = max(1, int(round(window_sec * fps)))

    summaries = []

    start = 0
    while start < len(rows):
        end = min(len(rows), start + window_frames)

        summaries.append(
            summarize_window(
                rows,
                start,
                end,
                fps,
            )
        )

        start = end

    # --------------------------------------------------------
    # CSV: frame-level diagnostics + window-level measurements
    # --------------------------------------------------------
    frame_csv = str(Path(csv_path).with_name(
        Path(csv_path).stem + "_frames.csv"
    ))

    window_csv = csv_path

    frame_fields = [
        "frame", "time_s", "u_px", "v_px",
        "major_px", "minor_px", "axis_ratio", "angle_deg",
        "quality", "height_m", "reason",
    ]

    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()

        for r in rows:
            out = {k: r[k] for k in frame_fields}
            for k, value in out.items():
                if isinstance(value, (float, np.floating)):
                    out[k] = (
                        f"{float(value):.6f}"
                        if np.isfinite(value) else ""
                    )
            writer.writerow(out)

    window_fields = [
        "window_start_s", "window_end_s",
        "water_level_m", "water_level_std_m", "water_level_mad_m",
        "valid_frames", "total_frames", "valid_fraction",
        "target_major_median_px", "target_axis_ratio_median",
        "target_u_median_px", "target_v_median_px",
        "motion_major_std_px", "motion_u_std_px", "motion_v_std_px",
        "confidence", "status",
    ]

    with open(window_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=window_fields)
        writer.writeheader()

        for s in summaries:
            out = dict(s)
            for k, value in out.items():
                if isinstance(value, (float, np.floating)):
                    out[k] = (
                        f"{float(value):.6f}"
                        if np.isfinite(value) else ""
                    )
            writer.writerow(out)

    # --------------------------------------------------------
    # Optional annotated video
    # --------------------------------------------------------
    if annotated_path:
        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            annotated_path,
            fourcc,
            fps,
            (width, height),
        )

        frame_i = 0
        summary_i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = undistort_frame(frame, camera)

            r = rows[frame_i]

            if r["ellipse"] is not None:
                cv2.ellipse(
                    frame,
                    r["ellipse"],
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                center = (
                    int(round(r["u_px"])),
                    int(round(r["v_px"])),
                )

                cv2.circle(
                    frame,
                    center,
                    4,
                    (255, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )

            if summary_i < len(summaries):
                s = summaries[summary_i]

                if frame_i + 1 >= int(
                    round(s["window_end_s"] * fps)
                ):
                    summary_i += 1

            if summary_i < len(summaries):
                s = summaries[summary_i]
            elif summaries:
                s = summaries[-1]
            else:
                s = None

            cv2.rectangle(
                frame,
                (20, 20),
                (700, 175),
                (25, 25, 25),
                -1,
            )

            cv2.putText(
                frame,
                f"target D: {target_diameter:.3f} m",
                (40, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if np.isfinite(r["height_m"]):
                cv2.putText(
                    frame,
                    f"frame estimate: {r['height_m']:.3f} m",
                    (40, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if s is not None and np.isfinite(s["water_level_m"]):
                text = (
                    f"window: {s['water_level_m']:.3f} m "
                    f"+/- {s['water_level_mad_m']:.3f} m"
                )
            else:
                text = "window: invalid"

            cv2.putText(
                frame,
                text,
                (40, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if s is not None:
                status = (
                    f"confidence: {s['confidence']:.2f}  "
                    f"valid: {s['valid_frames']}/{s['total_frames']}"
                )
                cv2.putText(
                    frame,
                    status,
                    (40, 148),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_i += 1

        cap.release()
        writer.release()

    good = [
        s["water_level_m"]
        for s in summaries
        if np.isfinite(s["water_level_m"])
    ]

    print("\nV4 measurement complete")
    print(f"  target diameter       : {target_diameter:.3f} m")
    print(f"  window                : {window_sec:.2f} s")
    print(f"  windows               : {len(summaries)}")
    print(f"  valid windows         : {len(good)}")
    print(f"  frame CSV             : {frame_csv}")
    print(f"  window CSV            : {window_csv}")

    if good:
        print(f"  median water level    : {np.median(good):.4f} m")
        print(f"  range                 : "
              f"{np.min(good):.4f} .. {np.max(good):.4f} m")

    if annotated_path:
        print(f"  annotated video       : {annotated_path}")


# ============================================================
# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="V4 floating-target water-level measurement."
    )
    parser.add_argument(
        "video",
        help="input video file",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # All configuration is intentionally kept in the source code.
    #
    # Edit these constants near the top of this file:
    #   TARGET_DIAMETER_M
    #   MEASUREMENT_WINDOW_SEC
    #
    # Existing camera calibration, if used, should also be configured
    # in the source rather than supplied as a CLI argument.

    model_path = "model_v4.npz"
    output_csv = "water_levels.csv"
    annotated_path = "annotated_v4.mp4"

    measure(
        video_path=args.video,
        model_path=model_path,
        csv_path=output_csv,
        camera_path=None,
        window_sec=MEASUREMENT_WINDOW_SEC,
        target_diameter_override=None,
        annotated_path=annotated_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
