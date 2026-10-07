#!/usr/bin/env python3
"""
water_height_cv.py

Monocular lake-water height estimation from a yellow circular floating target.

Dependencies:
    pip install opencv-python numpy

Important:
    If camera FoV / focal length and camera pose are unknown, absolute metric height
    cannot be recovered from a single monocular video from the known target diameter
    alone. This script therefore uses a short metric calibration.

Calibration assumption used by the default model:
    - camera is fixed;
    - float nominal horizontal distance from camera is approximately constant;
    - top target diameter is fixed;
    - target tilts only moderately;
    - at least 2 known water-height samples are available (3+ recommended).

The script fits:

    d_px ~= K / sqrt(R^2 + H^2)

so:

    1 / d_px^2 = a * H^2 + b

This removes the need to know FoV explicitly.

Usage
-----
1) Create calibration.csv:

    frame,height_m
    0,5.0
    150,10.0
    300,15.0
    450,20.0
    600,25.0

2) Calibrate:

    python water_height_cv.py calibrate input.mp4 calibration.csv calibration.npz

3) Measure:

    python water_height_cv.py measure input.mp4 calibration.npz heights.csv annotated.mp4

Outputs:
    heights.csv      frame-by-frame raw and filtered height
    annotated.mp4    optional annotated result video
"""

import sys
import csv
import math
import cv2
import numpy as np


# -----------------------------
# Yellow-target detection
# -----------------------------

# OpenCV HSV ranges:
# H: 0..179, S: 0..255, V: 0..255
YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_AREA_PX = 80
MAX_AREA_FRAC = 0.20

# Frames whose ellipse axis ratio changes too much from calibration can be rejected.
# Set to None to disable.
MAX_RATIO_REL_ERROR = 0.30


def detect_yellow_ellipse(frame):
    """
    Detect yellow top and fit ellipse.

    Returns dict or None:
      center: (u, v)
      major_px
      minor_px
      ratio = minor/major
      area_px
      contour_area_px
      mask
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

    # Clean mask
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    H, W = frame.shape[:2]
    max_area = H * W * MAX_AREA_FRAC

    best = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX or area > max_area:
            continue
        if len(c) < 5:
            continue

        ellipse = cv2.fitEllipse(c)
        (u, v), (a, b), angle = ellipse

        major = max(a, b)
        minor = min(a, b)
        if major <= 1 or minor <= 1:
            continue

        ratio = minor / major

        # Compare contour area with fitted ellipse area.
        ellipse_area = math.pi * (major * 0.5) * (minor * 0.5)
        if ellipse_area <= 1:
            continue

        fill = area / ellipse_area

        # Soft quality score:
        # prefer larger regions with reasonable fitted-ellipse occupancy.
        fill_score = max(0.0, 1.0 - abs(fill - 1.0))
        score = area * (0.5 + 0.5 * fill_score)

        if score > best_score:
            best_score = score
            best = {
                "center": (float(u), float(v)),
                "major_px": float(major),
                "minor_px": float(minor),
                "ratio": float(ratio),
                "area_px": float(ellipse_area),
                "contour_area_px": float(area),
                "ellipse": ellipse,
                "mask": mask,
            }

    return best


# -----------------------------
# Video utilities
# -----------------------------

def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def robust_target_scale(det):
    """
    Scale cue used for range/height estimation.

    Use major ellipse axis. For moderate tilting of a circular disk, the major axis
    is usually more stable than the minor axis or ellipse area.
    """
    return det["major_px"]


# -----------------------------
# Calibration
# -----------------------------

def load_calibration_csv(path):
    samples = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "height_m"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Calibration CSV must have columns: frame,height_m")

        for row in reader:
            samples.append((int(row["frame"]), float(row["height_m"])))

    if len(samples) < 2:
        raise ValueError("Need at least 2 calibration samples; 3+ are recommended.")

    return samples


def calibrate(video_path, cal_csv, output_npz):
    samples = load_calibration_csv(cal_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    heights = []
    scales = []
    ratios = []
    centers = []

    print("Calibration detections:")
    for frame_idx, Hm in samples:
        frame = read_frame(cap, frame_idx)
        if frame is None:
            print(f"  frame {frame_idx}: READ FAILED")
            continue

        det = detect_yellow_ellipse(frame)
        if det is None:
            print(f"  frame {frame_idx}: TARGET NOT FOUND")
            continue

        d = robust_target_scale(det)

        heights.append(Hm)
        scales.append(d)
        ratios.append(det["ratio"])
        centers.append(det["center"])

        print(
            f"  frame={frame_idx:6d}  H={Hm:8.3f} m  "
            f"major={d:8.2f}px  ratio={det['ratio']:.3f}  "
            f"center=({det['center'][0]:.1f},{det['center'][1]:.1f})"
        )

    cap.release()

    heights = np.asarray(heights, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    if len(heights) < 2:
        raise RuntimeError("Too few valid calibration detections.")

    # Physical model:
    # 1/d^2 = a*H^2 + b
    X = np.column_stack([heights ** 2, np.ones_like(heights)])
    y = 1.0 / (scales ** 2)

    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeff

    if a <= 0:
        raise RuntimeError(
            "Calibration produced non-positive coefficient a. "
            "Check that target scale decreases as camera-to-water height increases, "
            "and that calibration heights are correct."
        )

    pred_y = X @ coeff
    rmse_inv_d2 = float(np.sqrt(np.mean((pred_y - y) ** 2)))

    baseline_ratio = float(np.median(ratios))

    np.savez(
        output_npz,
        model=np.array(["inverse_square"], dtype="<U32"),
        a=np.array([a], dtype=np.float64),
        b=np.array([b], dtype=np.float64),
        baseline_ratio=np.array([baseline_ratio], dtype=np.float64),
        heights=heights,
        scales=scales,
        ratios=ratios,
        centers=centers,
    )

    # Derived values from model:
    # a = 1/K^2
    # b = R^2/K^2
    K = 1.0 / math.sqrt(a)
    R2 = b / a
    R_est = math.sqrt(R2) if R2 > 0 else float("nan")

    print("\nCalibration complete.")
    print(f"  a = {a:.12e}")
    print(f"  b = {b:.12e}")
    print(f"  baseline ellipse ratio = {baseline_ratio:.4f}")
    print(f"  model residual RMSE in 1/d^2 = {rmse_inv_d2:.6e}")
    print(f"  effective K = {K:.3f} pixel*m")
    print(f"  inferred nominal horizontal range R ~= {R_est:.3f} m")
    print(f"  saved: {output_npz}")


# -----------------------------
# Height estimation
# -----------------------------

def height_from_scale(d_px, a, b):
    val = (1.0 / (d_px * d_px) - b) / a
    if val < 0:
        return float("nan")
    return math.sqrt(val)


def rolling_median(values, window=31):
    """
    Pure NumPy rolling median.
    NaNs are ignored.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    half = window // 2

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        chunk = values[i0:i1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[i] = np.median(finite)

    return out


def measure(video_path, calibration_npz, output_csv, annotated_path=None):
    data = np.load(calibration_npz)

    a = float(data["a"][0])
    b = float(data["b"][0])
    baseline_ratio = float(data["baseline_ratio"][0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    raw_heights = []

    writer = None
    if annotated_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (W, H))

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_ellipse(frame)

        raw_h = float("nan")
        valid = False
        reject_reason = ""

        if det is None:
            reject_reason = "target_not_found"
            u = v = major = minor = ratio = float("nan")
        else:
            u, v = det["center"]
            major = det["major_px"]
            minor = det["minor_px"]
            ratio = det["ratio"]

            if (
                MAX_RATIO_REL_ERROR is not None
                and baseline_ratio > 0
                and abs(ratio - baseline_ratio) / baseline_ratio > MAX_RATIO_REL_ERROR
            ):
                reject_reason = "ellipse_ratio_outlier"
            else:
                raw_h = height_from_scale(major, a, b)
                valid = np.isfinite(raw_h)
                if not valid:
                    reject_reason = "model_out_of_range"

        raw_heights.append(raw_h)

        rows.append({
            "frame": idx,
            "time_s": idx / fps,
            "raw_height_m": raw_h,
            "filtered_height_m": float("nan"),   # filled later
            "u_px": u,
            "v_px": v,
            "major_px": major,
            "minor_px": minor,
            "axis_ratio": ratio,
            "valid": int(valid),
            "reject_reason": reject_reason,
        })

        idx += 1

    cap.release()

    raw_heights = np.asarray(raw_heights, dtype=np.float64)

    # About 1 second median filter by default.
    med_window = max(3, int(round(fps)))
    if med_window % 2 == 0:
        med_window += 1
    filtered = rolling_median(raw_heights, window=med_window)

    for i, h in enumerate(filtered):
        rows[i]["filtered_height_m"] = h

    # Write CSV
    fieldnames = [
        "frame", "time_s",
        "raw_height_m", "filtered_height_m",
        "u_px", "v_px",
        "major_px", "minor_px", "axis_ratio",
        "valid", "reject_reason",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()

        for row in rows:
            row2 = dict(row)
            for key in [
                "time_s", "raw_height_m", "filtered_height_m",
                "u_px", "v_px", "major_px", "minor_px", "axis_ratio"
            ]:
                val = row2[key]
                if isinstance(val, (float, np.floating)) and not np.isfinite(val):
                    row2[key] = ""
            writer_csv.writerow(row2)

    # Optional annotated pass
    if annotated_path:
        cap = cv2.VideoCapture(video_path)
        i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            det = detect_yellow_ellipse(frame)

            if det is not None:
                cv2.ellipse(frame, det["ellipse"], (0, 0, 255), 2, cv2.LINE_AA)
                u, v = det["center"]
                cv2.circle(frame, (int(round(u)), int(round(v))), 4, (255, 0, 255), -1)

            h_raw = raw_heights[i]
            h_f = filtered[i]

            text1 = (
                f"raw H: {h_raw:.3f} m"
                if np.isfinite(h_raw)
                else "raw H: invalid"
            )
            text2 = (
                f"filtered H: {h_f:.3f} m"
                if np.isfinite(h_f)
                else "filtered H: invalid"
            )

            cv2.rectangle(frame, (20, 20), (470, 115), (20, 20, 20), -1)
            cv2.putText(
                frame, text1, (40, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, text2, (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

            writer.write(frame)
            i += 1

        cap.release()
        writer.release()

    finite = filtered[np.isfinite(filtered)]
    print(f"Processed {len(rows)} frames.")
    if finite.size:
        print(
            f"Filtered height: median={np.median(finite):.3f} m, "
            f"min={np.min(finite):.3f} m, max={np.max(finite):.3f} m"
        )
    print(f"CSV saved: {output_csv}")
    if annotated_path:
        print(f"Annotated video saved: {annotated_path}")


# -----------------------------
# CLI
# -----------------------------

def usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        usage()
        return 1

    mode = sys.argv[1].lower()

    if mode == "calibrate":
        if len(sys.argv) != 5:
            usage()
            return 1
        _, _, video_path, cal_csv, output_npz = sys.argv
        calibrate(video_path, cal_csv, output_npz)
        return 0

    if mode == "measure":
        if len(sys.argv) not in (5, 6):
            usage()
            return 1

        video_path = sys.argv[2]
        calibration_npz = sys.argv[3]
        output_csv = sys.argv[4]
        annotated_path = sys.argv[5] if len(sys.argv) == 6 else None

        measure(video_path, calibration_npz, output_csv, annotated_path)
        return 0

    usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
