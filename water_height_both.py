#!/usr/bin/env python3
"""
water_height_both.py

Measure camera-to-water height from the yellow circular top of a floating target.

Works with:
  A) a height-changing calibration video (preferred for creating the model)
  B) a swinging/bobbing video at fixed or changing water height

External dependencies:
    OpenCV (cv2)
    NumPy

Python standard-library modules json/csv/sys are also used.

WHY THIS WORKS WITH UNKNOWN FoV
-------------------------------
The camera FoV/focal length does NOT need to be known if the same fixed camera
and the same target are calibrated empirically.

The height-changing video supplies pairs:
    apparent yellow-disc size in pixels  <->  known water height in metres

The script builds a monotonic lookup model:
    ellipse major-axis pixels -> camera-to-water height

Then the same model can be applied to:
    - another height-changing video
    - a swinging/bobbing target video

The ellipse MAJOR AXIS is intentionally used because a circular disc that tilts
usually changes its minor axis much more strongly than its major axis.

ASSUMPTIONS
-----------
1. Camera remains fixed after calibration.
2. Camera zoom/focus geometry does not change.
3. Target physical diameter is unchanged.
4. Target's nominal horizontal distance from camera is approximately unchanged.
   If the float moves several metres toward/away from the camera, monocular scale
   alone cannot distinguish that from a change in water height.
5. The yellow top remains sufficiently visible.

COMMANDS
--------

1) Calibrate automatically from a height-changing video + its JSON sidecar:

    python water_height_both.py calibrate-json \
        lake_float_camera_simulation_v2.mp4 \
        lake_float_camera_simulation_v2.json \
        water_height_model.npz

The JSON is expected to contain:
    frames[i]["camera_to_water_vertical_height_m"]

2) Measure ANY video with the calibrated model:

    python water_height_both.py measure \
        some_video.mp4 water_height_model.npz result.csv annotated.mp4

Examples:

    # height-changing video
    python water_height_both.py measure \
        height_changing.mp4 water_height_model.npz height_result.csv height_annotated.mp4

    # fixed-height swinging video
    python water_height_both.py measure \
        swinging.mp4 water_height_model.npz swing_result.csv swing_annotated.mp4

Optional calibration tuning constants are near the top of this file.
"""

import sys
import json
import csv
import math
import cv2
import numpy as np


# ============================================================
# Detection configuration
# ============================================================

# OpenCV HSV: H=0..179, S=0..255, V=0..255
YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA_PX = 40
MAX_TARGET_AREA_FRACTION = 0.15

# Calibration bin width in metres.
# Several detected frames are collapsed to one robust median point per bin.
HEIGHT_BIN_M = 0.25

# Runtime filtering:
MEDIAN_FILTER_SECONDS = 1.0

# Swing / tilt quality check.
# We do NOT directly use the minor axis to estimate height, but very extreme
# ellipse shapes may indicate occlusion, bad segmentation, or excessive tilt.
MIN_AXIS_RATIO = 0.25


# ============================================================
# Yellow target detector
# ============================================================

def detect_yellow_top(frame):
    """
    Return the best yellow elliptical target detection or None.

    Output keys:
        center_px = (u, v)
        major_px
        minor_px
        axis_ratio = minor / major
        contour_area_px
        ellipse
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    H, W = frame.shape[:2]
    max_area = H * W * MAX_TARGET_AREA_FRACTION

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_TARGET_AREA_PX or area > max_area:
            continue

        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        (u, v), (w, h), angle = ellipse

        major = float(max(w, h))
        minor = float(min(w, h))

        if major <= 1.0 or minor <= 1.0:
            continue

        ratio = minor / major

        # Area consistency between contour and fitted ellipse.
        ellipse_area = math.pi * 0.5 * major * 0.5 * minor
        if ellipse_area <= 0:
            continue

        fill_ratio = area / ellipse_area

        # Prefer a large yellow region that is reasonably ellipse-like.
        ellipse_quality = max(0.0, 1.0 - abs(fill_ratio - 1.0))
        score = area * (0.4 + 0.6 * ellipse_quality)

        if score > best_score:
            best_score = score
            best = {
                "center_px": (float(u), float(v)),
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": float(ratio),
                "contour_area_px": float(area),
                "ellipse": ellipse,
            }

    return best


# ============================================================
# Robust calibration model
# ============================================================

def _bin_calibration_points(heights, majors, ratios):
    """
    Bin samples by true height and take robust medians.
    """
    heights = np.asarray(heights, dtype=np.float64)
    majors = np.asarray(majors, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)

    h0 = math.floor(np.min(heights) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    indices = np.floor((heights - h0) / HEIGHT_BIN_M).astype(int)

    out_h = []
    out_d = []
    out_r = []
    out_n = []

    for k in np.unique(indices):
        mask = indices == k

        if np.sum(mask) < 2:
            continue

        out_h.append(float(np.median(heights[mask])))
        out_d.append(float(np.median(majors[mask])))
        out_r.append(float(np.median(ratios[mask])))
        out_n.append(int(np.sum(mask)))

    return (
        np.asarray(out_h, dtype=np.float64),
        np.asarray(out_d, dtype=np.float64),
        np.asarray(out_r, dtype=np.float64),
        np.asarray(out_n, dtype=np.int32),
    )


def _make_monotonic_lookup(heights, majors):
    """
    Construct robust monotonic lookup arrays.

    For a fixed camera + approximately fixed horizontal float position:
        greater water distance -> smaller apparent target diameter.

    np.interp requires ascending x, so returned scale_px is ascending.
    """
    order_h = np.argsort(heights)
    h = heights[order_h]
    d = majors[order_h]

    # Enforce non-increasing apparent diameter as height increases.
    # This suppresses small segmentation/noise reversals.
    d_mono = d.copy()
    for i in range(1, len(d_mono)):
        if d_mono[i] > d_mono[i - 1]:
            d_mono[i] = d_mono[i - 1]

    # Remove duplicate/non-informative scales.
    keep = np.ones(len(d_mono), dtype=bool)
    for i in range(1, len(d_mono)):
        if abs(d_mono[i] - d_mono[i - 1]) < 1e-6:
            keep[i] = False

    h = h[keep]
    d_mono = d_mono[keep]

    if len(h) < 2:
        raise RuntimeError(
            "Not enough distinct calibration points after monotonic cleanup."
        )

    # Reverse: scale ascending => height descending.
    scale_asc = d_mono[::-1].copy()
    height_for_scale = h[::-1].copy()

    return scale_asc, height_for_scale


def calibrate_from_json(video_path, json_path, model_path):
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    frame_meta = metadata.get("frames", [])
    if not frame_meta:
        raise RuntimeError("JSON contains no 'frames' array.")

    height_by_frame = {}
    for item in frame_meta:
        idx = int(item["frame_index"])
        height = float(item["camera_to_water_vertical_height_m"])
        height_by_frame[idx] = height

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    detected_heights = []
    detected_majors = []
    detected_ratios = []
    detected_frames = []

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index in height_by_frame:
            det = detect_yellow_top(frame)

            if det is not None and det["axis_ratio"] >= MIN_AXIS_RATIO:
                detected_heights.append(height_by_frame[frame_index])
                detected_majors.append(det["major_px"])
                detected_ratios.append(det["axis_ratio"])
                detected_frames.append(frame_index)

        frame_index += 1

    cap.release()

    if len(detected_heights) < 10:
        raise RuntimeError(
            f"Only {len(detected_heights)} valid target detections found. "
            "Check HSV thresholds, framing, and target visibility."
        )

    bin_h, bin_d, bin_r, bin_n = _bin_calibration_points(
        detected_heights, detected_majors, detected_ratios
    )

    if len(bin_h) < 2:
        raise RuntimeError("Need at least two usable height calibration bins.")

    scale_asc, height_for_scale = _make_monotonic_lookup(bin_h, bin_d)

    np.savez(
        model_path,
        model_type=np.array(["monotonic_major_axis_lookup"], dtype="<U64"),
        scale_px_ascending=scale_asc,
        height_m_for_scale=height_for_scale,
        calibration_height_m=bin_h,
        calibration_major_px=bin_d,
        calibration_axis_ratio=bin_r,
        calibration_count=bin_n,
        valid_detection_frames=np.asarray(detected_frames, dtype=np.int32),
        valid_detection_heights_m=np.asarray(detected_heights, dtype=np.float64),
        valid_detection_major_px=np.asarray(detected_majors, dtype=np.float64),
    )

    print("Calibration complete")
    print(f"  raw valid detections : {len(detected_heights)}")
    print(f"  robust height bins   : {len(bin_h)}")
    print(
        f"  calibrated height range: "
        f"{np.min(bin_h):.3f} .. {np.max(bin_h):.3f} m"
    )
    print(
        f"  target major-axis range: "
        f"{np.min(bin_d):.2f} .. {np.max(bin_d):.2f} px"
    )
    print(f"  model saved: {model_path}")


def estimate_height_from_major(major_px, scale_asc, height_for_scale):
    """
    Interpolate inside calibrated range only.
    Returns NaN outside range rather than unsafe extrapolation.
    """
    if not np.isfinite(major_px):
        return float("nan")

    lo = scale_asc[0]
    hi = scale_asc[-1]

    if major_px < lo or major_px > hi:
        return float("nan")

    return float(np.interp(major_px, scale_asc, height_for_scale))


# ============================================================
# Temporal filtering
# ============================================================

def rolling_nanmedian(values, window):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)

    half = window // 2

    for i in range(len(values)):
        a = max(0, i - half)
        b = min(len(values), i + half + 1)

        chunk = values[a:b]
        chunk = chunk[np.isfinite(chunk)]

        if len(chunk):
            out[i] = np.median(chunk)

    return out


# ============================================================
# Measurement
# ============================================================

def measure_video(video_path, model_path, csv_path, annotated_path=None):
    model = np.load(model_path)

    scale_asc = np.asarray(model["scale_px_ascending"], dtype=np.float64)
    height_for_scale = np.asarray(
        model["height_m_for_scale"], dtype=np.float64
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    raw_heights = []

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_top(frame)

        if det is None:
            row = {
                "frame": frame_index,
                "time_s": frame_index / fps,
                "u_px": np.nan,
                "v_px": np.nan,
                "major_px": np.nan,
                "minor_px": np.nan,
                "axis_ratio": np.nan,
                "raw_height_m": np.nan,
                "valid": 0,
                "reason": "target_not_found",
                "ellipse": None,
            }

        else:
            u, v = det["center_px"]
            ratio = det["axis_ratio"]

            if ratio < MIN_AXIS_RATIO:
                h = np.nan
                valid = 0
                reason = "excessive_tilt_or_bad_segmentation"
            else:
                h = estimate_height_from_major(
                    det["major_px"], scale_asc, height_for_scale
                )

                if np.isfinite(h):
                    valid = 1
                    reason = ""
                else:
                    valid = 0
                    reason = "outside_calibration_range"

            row = {
                "frame": frame_index,
                "time_s": frame_index / fps,
                "u_px": u,
                "v_px": v,
                "major_px": det["major_px"],
                "minor_px": det["minor_px"],
                "axis_ratio": ratio,
                "raw_height_m": h,
                "valid": valid,
                "reason": reason,
                "ellipse": det["ellipse"],
            }

        detections.append(row)
        raw_heights.append(row["raw_height_m"])
        frame_index += 1

    cap.release()

    # Median filtering is particularly helpful for the swinging float video.
    window = max(3, int(round(MEDIAN_FILTER_SECONDS * fps)))
    if window % 2 == 0:
        window += 1

    filtered = rolling_nanmedian(raw_heights, window)

    for i in range(len(detections)):
        detections[i]["filtered_height_m"] = filtered[i]

    # CSV output
    fields = [
        "frame",
        "time_s",
        "raw_height_m",
        "filtered_height_m",
        "u_px",
        "v_px",
        "major_px",
        "minor_px",
        "axis_ratio",
        "valid",
        "reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()

        for row in detections:
            out = {key: row[key] for key in fields}

            for key in (
                "time_s",
                "raw_height_m",
                "filtered_height_m",
                "u_px",
                "v_px",
                "major_px",
                "minor_px",
                "axis_ratio",
            ):
                value = out[key]
                if isinstance(value, (float, np.floating)):
                    if np.isfinite(value):
                        out[key] = f"{float(value):.6f}"
                    else:
                        out[key] = ""

            wcsv.writerow(out)

    # Optional annotated video.
    if annotated_path is not None:
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            annotated_path, fourcc, fps, (W, H)
        )

        i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            row = detections[i]
            ellipse = row["ellipse"]

            if ellipse is not None:
                cv2.ellipse(
                    frame, ellipse,
                    (0, 0, 255), 2, cv2.LINE_AA
                )

                if np.isfinite(row["u_px"]) and np.isfinite(row["v_px"]):
                    center = (
                        int(round(row["u_px"])),
                        int(round(row["v_px"])),
                    )
                    cv2.circle(
                        frame, center, 4,
                        (255, 0, 255), -1, cv2.LINE_AA
                    )

            raw_h = row["raw_height_m"]
            filt_h = row["filtered_height_m"]

            if np.isfinite(raw_h):
                raw_text = f"raw height: {raw_h:.3f} m"
            else:
                raw_text = "raw height: invalid"

            if np.isfinite(filt_h):
                filt_text = f"filtered height: {filt_h:.3f} m"
            else:
                filt_text = "filtered height: invalid"

            cv2.rectangle(
                frame, (20, 20), (520, 130),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame, raw_text, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, filt_text, (40, 103),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            writer.write(frame)
            i += 1

        cap.release()
        writer.release()

    good = filtered[np.isfinite(filtered)]

    print(f"Processed frames: {len(detections)}")
    print(
        f"Valid raw height frames: "
        f"{np.sum(np.isfinite(np.asarray(raw_heights)))}"
    )

    if len(good):
        print(f"Filtered median: {np.median(good):.4f} m")
        print(f"Filtered min   : {np.min(good):.4f} m")
        print(f"Filtered max   : {np.max(good):.4f} m")
        print(f"Filtered std   : {np.std(good):.4f} m")

    print(f"CSV: {csv_path}")

    if annotated_path:
        print(f"Annotated video: {annotated_path}")


# ============================================================
# CLI
# ============================================================

def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1].lower()

    if command == "calibrate-json":
        if len(sys.argv) != 5:
            print_usage()
            return 1

        calibrate_from_json(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
        )
        return 0

    if command == "measure":
        if len(sys.argv) not in (5, 6):
            print_usage()
            return 1

        annotated = sys.argv[5] if len(sys.argv) == 6 else None

        measure_video(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            annotated,
        )
        return 0

    print_usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
