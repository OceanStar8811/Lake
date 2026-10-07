#!/usr/bin/env python3
"""
water_height_v3.py

Robust monocular water-height measurement from the yellow circular top of a
floating cylindrical target.

Dependencies:
    OpenCV (cv2)
    NumPy
Only Python standard-library modules are otherwise used.

Designed for TWO cases:
    1) height-changing video
    2) swinging / rocking / bobbing float video

UNKNOWN CAMERA FoV
------------------
FoV is not required. The fixed camera is calibrated empirically from a video
whose frame-by-frame camera-to-water heights are known.

The script uses the yellow top as an ellipse and stores a robust calibration
curve:
    ellipse major-axis size [px] -> water height [m]

WHY MAJOR AXIS?
---------------
For a circular top that rocks, the ellipse minor axis changes strongly with
tilt. The major axis is normally much more stable, so it is the primary
metric-height cue.

OPTIONAL SWING ANCHOR
---------------------
A short clip recorded at one known fixed height (for example 10.0 m) can be
used to adapt the model to float rocking and small scale differences.

This creates:
    - a multiplicative apparent-scale correction;
    - a learned nuisance correction based on:
        * ellipse axis ratio,
        * horizontal image displacement,
        * vertical image displacement.

This is useful for a swinging target.

IMPORTANT IDENTIFIABILITY LIMIT
-------------------------------
With a single uncalibrated monocular camera, a large target motion toward or
away from the camera can look like a water-height change. No software can
uniquely separate those effects from one circular target alone unless there is
additional calibration/constraint/reference geometry.

For best results:
    - keep camera fixed;
    - keep zoom/focus fixed;
    - constrain float horizontal drift;
    - use the SAME camera geometry for calibration and operation.

COMMANDS
========

A) Build base model from height-changing video + JSON:

    python water_height_v3.py calibrate \
        height_changing.mp4 height_changing.json base_model.npz

JSON must contain:
    frames[i]["frame_index"]
    frames[i]["camera_to_water_vertical_height_m"]

B) Measure height-changing video:

    python water_height_v3.py measure \
        height_changing.mp4 base_model.npz height.csv height_annotated.mp4

C) Adapt a COPY of the model using a known-height swinging clip:

    python water_height_v3.py swing-anchor \
        base_model.npz swinging_10m.mp4 10.0 swing_model.npz

D) Measure swinging video:

    python water_height_v3.py measure \
        swinging_10m.mp4 swing_model.npz swing.csv swing_annotated.mp4
"""

import sys
import json
import csv
import math
import cv2
import numpy as np


# ============================================================
# User-tunable detector settings
# ============================================================

# OpenCV HSV ranges
YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA_PX = 35
MAX_TARGET_AREA_FRACTION = 0.15
MIN_AXIS_RATIO = 0.20

# Robust calibration resolution
HEIGHT_BIN_M = 0.20

# Final temporal smoothing
MEDIAN_FILTER_SECONDS = 1.0

# Ridge regularization for swing-nuisance correction
RIDGE_LAMBDA = 0.03


# ============================================================
# Target detection
# ============================================================

def detect_yellow_top(frame):
    """
    Detect the yellow top and fit an ellipse.

    Returns:
        dict or None

    dict fields:
        center_px
        major_px
        minor_px
        axis_ratio
        contour_area_px
        ellipse
        quality
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
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))

        if area < MIN_TARGET_AREA_PX or area > max_area:
            continue

        if len(c) < 5:
            continue

        ellipse = cv2.fitEllipse(c)
        (u, v), (a, b), _angle = ellipse

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

        # Compactness with perimeter helps reject reflections/noisy yellow patches.
        perimeter = float(cv2.arcLength(c, True))
        compactness = 0.0
        if perimeter > 0:
            compactness = 4.0 * math.pi * area / (perimeter * perimeter)
        compactness = max(0.0, min(1.0, compactness))

        quality = 0.65 * fill_quality + 0.35 * compactness

        # Large, ellipse-like targets win.
        score = area * (0.3 + 0.7 * quality)

        if score > best_score:
            best_score = score
            best = {
                "center_px": (float(u), float(v)),
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": float(ratio),
                "contour_area_px": area,
                "ellipse": ellipse,
                "quality": float(quality),
            }

    return best


# ============================================================
# Helpers
# ============================================================

def nanmedian_filter(values, window):
    x = np.asarray(values, dtype=np.float64)
    out = np.full_like(x, np.nan)

    half = window // 2

    for i in range(len(x)):
        a = max(0, i - half)
        b = min(len(x), i + half + 1)
        z = x[a:b]
        z = z[np.isfinite(z)]

        if z.size:
            out[i] = np.median(z)

    return out


def interp_inside(x, xp, fp):
    """
    Interpolate only inside the calibrated range.
    """
    if not np.isfinite(x):
        return float("nan")

    if x < xp[0] or x > xp[-1]:
        return float("nan")

    return float(np.interp(x, xp, fp))


def interp_height_feature(height, h_asc, feature):
    if not np.isfinite(height):
        return float("nan")

    if height < h_asc[0] or height > h_asc[-1]:
        return float("nan")

    return float(np.interp(height, h_asc, feature))


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
            "No frame_index / camera_to_water_vertical_height_m data found."
        )

    return result


# ============================================================
# Calibration
# ============================================================

def robust_bin(samples):
    """
    samples rows:
        [height, major, minor, ratio, u, v, quality]
    """
    A = np.asarray(samples, dtype=np.float64)

    heights = A[:, 0]

    h0 = (
        math.floor(float(np.min(heights)) / HEIGHT_BIN_M)
        * HEIGHT_BIN_M
    )

    bin_idx = np.floor((heights - h0) / HEIGHT_BIN_M).astype(int)

    rows = []

    for k in np.unique(bin_idx):
        B = A[bin_idx == k]

        if len(B) < 2:
            continue

        # Median is deliberately used to suppress occasional segmentation errors.
        row = np.median(B, axis=0)
        rows.append(row)

    if len(rows) < 3:
        raise RuntimeError("Too few robust calibration bins.")

    return np.asarray(rows, dtype=np.float64)


def enforce_monotonic_scale(height, major):
    """
    At approximately fixed horizontal distance:
      larger camera-water height -> smaller apparent disk major axis.

    Enforce this softly through a monotonic envelope.
    """
    order = np.argsort(height)

    h = np.asarray(height, dtype=np.float64)[order]
    d = np.asarray(major, dtype=np.float64)[order].copy()

    for i in range(1, len(d)):
        if d[i] > d[i - 1]:
            d[i] = d[i - 1]

    # Drop duplicate scale points because np.interp needs usable spacing.
    keep = np.ones(len(d), dtype=bool)
    for i in range(1, len(d)):
        if abs(d[i] - d[i - 1]) < 1e-5:
            keep[i] = False

    h = h[keep]
    d = d[keep]

    if len(h) < 2:
        raise RuntimeError("Calibration scale is not sufficiently varying.")

    # x for np.interp = ascending major-axis pixels.
    scale_asc = d[::-1].copy()
    height_for_scale = h[::-1].copy()

    return scale_asc, height_for_scale


def calibrate(video_path, json_path, model_path):
    gt = read_known_heights(json_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    samples = []

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in gt:
            det = detect_yellow_top(frame)

            if det is not None:
                u, v = det["center_px"]

                samples.append([
                    gt[idx],
                    det["major_px"],
                    det["minor_px"],
                    det["axis_ratio"],
                    u,
                    v,
                    det["quality"],
                ])

        idx += 1

    cap.release()

    if len(samples) < 10:
        raise RuntimeError(
            f"Only {len(samples)} calibration detections. "
            "Check target visibility / HSV threshold."
        )

    B = robust_bin(samples)

    # Sort all calibration feature curves by true height.
    order = np.argsort(B[:, 0])
    B = B[order]

    h = B[:, 0]
    major = B[:, 1]
    minor = B[:, 2]
    ratio = B[:, 3]
    u = B[:, 4]
    v = B[:, 5]
    quality = B[:, 6]

    scale_asc, height_for_scale = enforce_monotonic_scale(h, major)

    np.savez(
        model_path,
        model_version=np.array(["v3"], dtype="<U8"),
        scale_px_ascending=scale_asc,
        height_m_for_scale=height_for_scale,
        calibration_height_m=h,
        calibration_major_px=major,
        calibration_minor_px=minor,
        calibration_ratio=ratio,
        calibration_u_px=u,
        calibration_v_px=v,
        calibration_quality=quality,
        scale_factor=np.array([1.0], dtype=np.float64),
        anchor_enabled=np.array([0], dtype=np.int32),
        nuisance_beta=np.zeros(10, dtype=np.float64),
        anchor_height_m=np.array([np.nan], dtype=np.float64),
    )

    print("Base calibration saved.")
    print(f"  detections       : {len(samples)}")
    print(f"  robust bins      : {len(h)}")
    print(
        f"  height range     : {h.min():.3f} .. {h.max():.3f} m"
    )
    print(
        f"  major-axis range : {major.min():.2f} .. {major.max():.2f} px"
    )
    print(f"  model            : {model_path}")


# ============================================================
# Base height model and nuisance features
# ============================================================

def load_model(path):
    M = np.load(path)

    return {
        "scale_px_ascending": np.asarray(
            M["scale_px_ascending"], dtype=np.float64
        ),
        "height_m_for_scale": np.asarray(
            M["height_m_for_scale"], dtype=np.float64
        ),
        "h": np.asarray(M["calibration_height_m"], dtype=np.float64),
        "major": np.asarray(
            M["calibration_major_px"], dtype=np.float64
        ),
        "minor": np.asarray(
            M["calibration_minor_px"], dtype=np.float64
        ),
        "ratio": np.asarray(
            M["calibration_ratio"], dtype=np.float64
        ),
        "u": np.asarray(M["calibration_u_px"], dtype=np.float64),
        "v": np.asarray(M["calibration_v_px"], dtype=np.float64),
        "quality": np.asarray(
            M["calibration_quality"], dtype=np.float64
        ),
        "scale_factor": float(M["scale_factor"][0]),
        "anchor_enabled": bool(int(M["anchor_enabled"][0])),
        "nuisance_beta": np.asarray(
            M["nuisance_beta"], dtype=np.float64
        ),
        "anchor_height_m": float(M["anchor_height_m"][0]),
    }


def base_height(det, M):
    normalized_major = det["major_px"] * M["scale_factor"]

    return interp_inside(
        normalized_major,
        M["scale_px_ascending"],
        M["height_m_for_scale"],
    )


def expected_features_at_height(h, M):
    hc = M["h"]

    return {
        "major": interp_height_feature(h, hc, M["major"]),
        "ratio": interp_height_feature(h, hc, M["ratio"]),
        "u": interp_height_feature(h, hc, M["u"]),
        "v": interp_height_feature(h, hc, M["v"]),
    }


def nuisance_vector(det, h_base, M):
    """
    Swing/drift features relative to the normal height-calibration trajectory.

    Vector length = 10.
    """
    exp = expected_features_at_height(h_base, M)

    major_norm = max(
        1.0, det["major_px"] * M["scale_factor"]
    )

    du = (det["center_px"][0] - exp["u"]) / major_norm
    dv = (det["center_px"][1] - exp["v"]) / major_norm
    dr = det["axis_ratio"] - exp["ratio"]

    # Polynomial nuisance basis.
    return np.array([
        1.0,
        du,
        dv,
        dr,
        du * du,
        dv * dv,
        dr * dr,
        du * dv,
        du * dr,
        dv * dr,
    ], dtype=np.float64)


# ============================================================
# Swing anchor calibration
# ============================================================

def swing_anchor(base_model_path, swing_video, known_height, output_model):
    M = load_model(base_model_path)

    # Expected target major-axis size at the known true height.
    hc = M["h"]
    expected_major = interp_height_feature(
        known_height, hc, M["major"]
    )

    if not np.isfinite(expected_major):
        raise RuntimeError(
            "Known swing height is outside base calibration height range."
        )

    cap = cv2.VideoCapture(swing_video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {swing_video}")

    detections = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_top(frame)

        if det is not None:
            detections.append(det)

    cap.release()

    if len(detections) < 20:
        raise RuntimeError("Too few detections in swing-anchor video.")

    observed_major = np.asarray(
        [d["major_px"] for d in detections],
        dtype=np.float64
    )

    median_observed_major = float(np.median(observed_major))

    # Primary transfer adaptation:
    # map this camera/clip's known-height scale to the base calibration scale.
    scale_factor = expected_major / median_observed_major

    # Warn quantitatively if the transfer is large.
    transfer_error_pct = 100.0 * abs(scale_factor - 1.0)

    # Build temporary model with adapted scale.
    M2 = dict(M)
    M2["scale_factor"] = scale_factor

    X = []
    y = []

    base_values = []

    for det in detections:
        hb = base_height(det, M2)

        if not np.isfinite(hb):
            continue

        phi = nuisance_vector(det, hb, M2)

        # Learn correction to the known fixed height.
        correction = known_height - hb

        X.append(phi)
        y.append(correction)
        base_values.append(hb)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(X) < 10:
        beta = np.zeros(10, dtype=np.float64)
    else:
        # Do not strongly regularize intercept.
        reg = np.eye(X.shape[1], dtype=np.float64) * RIDGE_LAMBDA
        reg[0, 0] = RIDGE_LAMBDA * 0.05

        beta = np.linalg.solve(
            X.T @ X + reg,
            X.T @ y
        )

    np.savez(
        output_model,
        model_version=np.array(["v3"], dtype="<U8"),
        scale_px_ascending=M["scale_px_ascending"],
        height_m_for_scale=M["height_m_for_scale"],
        calibration_height_m=M["h"],
        calibration_major_px=M["major"],
        calibration_minor_px=M["minor"],
        calibration_ratio=M["ratio"],
        calibration_u_px=M["u"],
        calibration_v_px=M["v"],
        calibration_quality=M["quality"],
        scale_factor=np.array([scale_factor], dtype=np.float64),
        anchor_enabled=np.array([1], dtype=np.int32),
        nuisance_beta=beta,
        anchor_height_m=np.array([known_height], dtype=np.float64),
    )

    before = np.asarray(base_values, dtype=np.float64)

    corrected = []

    for det in detections:
        hb = base_height(det, M2)

        if not np.isfinite(hb):
            continue

        phi = nuisance_vector(det, hb, M2)
        corrected.append(hb + float(phi @ beta))

    corrected = np.asarray(corrected, dtype=np.float64)

    print("Swing-anchor adaptation saved.")
    print(f"  known height            : {known_height:.3f} m")
    print(f"  expected major axis     : {expected_major:.3f} px")
    print(f"  observed median major   : {median_observed_major:.3f} px")
    print(f"  scale factor            : {scale_factor:.6f}")
    print(f"  transfer magnitude      : {transfer_error_pct:.2f}%")

    if transfer_error_pct > 8.0:
        print(
            "  WARNING: >8% scale transfer. Camera pose, zoom, or "
            "float distance may differ materially from base calibration."
        )

    if before.size:
        print(
            f"  pre-nuisance median/std : "
            f"{np.median(before):.3f} / {np.std(before):.3f} m"
        )

    if corrected.size:
        print(
            f"  corrected median/std    : "
            f"{np.median(corrected):.3f} / {np.std(corrected):.3f} m"
        )

    print(f"  model                   : {output_model}")


# ============================================================
# Measurement
# ============================================================

def estimate_frame(det, M):
    hb = base_height(det, M)

    if not np.isfinite(hb):
        return float("nan"), float("nan"), "outside_calibration_range"

    hc = hb

    if M["anchor_enabled"]:
        phi = nuisance_vector(det, hb, M)
        hc = hb + float(phi @ M["nuisance_beta"])

    # Protect against unstable extrapolation from nuisance correction.
    hmin = float(np.min(M["h"]))
    hmax = float(np.max(M["h"]))

    margin = 0.10 * (hmax - hmin)

    if hc < hmin - margin or hc > hmax + margin:
        return hb, float("nan"), "corrected_height_outlier"

    return hb, hc, ""


def measure(video_path, model_path, csv_path, annotated_path=None):
    M = load_model(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    corrected_values = []

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_top(frame)

        if det is None:
            row = {
                "frame": idx,
                "time_s": idx / fps,
                "u_px": np.nan,
                "v_px": np.nan,
                "major_px": np.nan,
                "minor_px": np.nan,
                "axis_ratio": np.nan,
                "quality": np.nan,
                "base_height_m": np.nan,
                "corrected_height_m": np.nan,
                "filtered_height_m": np.nan,
                "reason": "target_not_found",
                "ellipse": None,
            }
            corrected_values.append(np.nan)

        else:
            hb, hc, reason = estimate_frame(det, M)

            u, v = det["center_px"]

            row = {
                "frame": idx,
                "time_s": idx / fps,
                "u_px": u,
                "v_px": v,
                "major_px": det["major_px"],
                "minor_px": det["minor_px"],
                "axis_ratio": det["axis_ratio"],
                "quality": det["quality"],
                "base_height_m": hb,
                "corrected_height_m": hc,
                "filtered_height_m": np.nan,
                "reason": reason,
                "ellipse": det["ellipse"],
            }

            corrected_values.append(hc)

        rows.append(row)
        idx += 1

    cap.release()

    window = max(
        3, int(round(MEDIAN_FILTER_SECONDS * fps))
    )

    if window % 2 == 0:
        window += 1

    filtered = nanmedian_filter(
        np.asarray(corrected_values, dtype=np.float64),
        window
    )

    for i in range(len(rows)):
        rows[i]["filtered_height_m"] = filtered[i]

    # CSV
    fields = [
        "frame",
        "time_s",
        "u_px",
        "v_px",
        "major_px",
        "minor_px",
        "axis_ratio",
        "quality",
        "base_height_m",
        "corrected_height_m",
        "filtered_height_m",
        "reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in rows:
            out = {k: row[k] for k in fields}

            for k in fields:
                val = out[k]
                if isinstance(val, (float, np.floating)):
                    if np.isfinite(val):
                        out[k] = f"{float(val):.6f}"
                    else:
                        out[k] = ""

            w.writerow(out)

    # Annotated output
    if annotated_path:
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

            row = rows[i]

            if row["ellipse"] is not None:
                cv2.ellipse(
                    frame,
                    row["ellipse"],
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                c = (
                    int(round(row["u_px"])),
                    int(round(row["v_px"]))
                )

                cv2.circle(
                    frame, c, 4,
                    (255, 0, 255), -1,
                    cv2.LINE_AA
                )

            h = row["filtered_height_m"]

            cv2.rectangle(
                frame, (20, 20), (560, 150),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame,
                f"frame: {i}",
                (40, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if np.isfinite(h):
                text = f"water height: {h:.3f} m"
            else:
                text = "water height: invalid"

            cv2.putText(
                frame,
                text,
                (40, 99),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.88,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            mode_text = (
                "swing-corrected model"
                if M["anchor_enabled"]
                else "base calibration model"
            )

            cv2.putText(
                frame,
                mode_text,
                (40, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            i += 1

        cap.release()
        writer.release()

    good = filtered[np.isfinite(filtered)]

    print(f"Processed frames : {len(rows)}")
    print(f"Valid filtered   : {len(good)}")

    if good.size:
        print(f"Median height    : {np.median(good):.4f} m")
        print(f"Mean height      : {np.mean(good):.4f} m")
        print(f"Std deviation    : {np.std(good):.4f} m")
        print(f"Min / max        : {np.min(good):.4f} / {np.max(good):.4f} m")

    print(f"CSV              : {csv_path}")

    if annotated_path:
        print(f"Annotated video  : {annotated_path}")


# ============================================================
# Command line
# ============================================================

def usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        usage()
        return 1

    cmd = sys.argv[1].lower()

    if cmd == "calibrate":
        if len(sys.argv) != 5:
            usage()
            return 1

        calibrate(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
        )

        return 0

    if cmd == "swing-anchor":
        if len(sys.argv) != 6:
            usage()
            return 1

        swing_anchor(
            sys.argv[2],
            sys.argv[3],
            float(sys.argv[4]),
            sys.argv[5],
        )

        return 0

    if cmd == "measure":
        if len(sys.argv) not in (5, 6):
            usage()
            return 1

        annotated = (
            sys.argv[5]
            if len(sys.argv) == 6
            else None
        )

        measure(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            annotated,
        )

        return 0

    usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
