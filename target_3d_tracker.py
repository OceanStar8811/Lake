#!/usr/bin/env python3
"""
target_3d_tracker.py

Track a yellow circular floating target and estimate its 3D position relative
to a distortion-free monocular camera.

Dependencies:
    pip install opencv-python numpy

USER-SETTABLE PHYSICAL PARAMETER
================================
TARGET_DIAMETER_M = 1.0

CAMERA ASSUMPTIONS
==================
- distortion-free image
- principal point approximately at image center
- square pixels: fx = fy = focal_px
- camera is fixed
- yellow top is a physical circle
- target swing/rocking is treated as fast disturbance and reduced temporally

IMPORTANT GEOMETRIC LIMIT
=========================
The known target diameter fixes metric scale ONLY when the camera focal length
(in pixels) is known.

If focal length / FoV is completely unknown, a single monocular image cannot
uniquely recover absolute metric X,Y,Z from only one circle.

Therefore this program supports two modes:

1) Metric XYZ:
       provide --focal-px <value>

2) No focal length:
       the program still tracks the target and outputs:
           - image center
           - ellipse axes
           - normalized camera ray
           - range/focal = D / f
       but absolute metric XYZ is left blank.

JSON is NEVER used for estimation.
An optional JSON file can be supplied only for comparison.

COORDINATE SYSTEM
=================
Camera coordinates:
    X: right
    Y: down
    Z: forward

Thus XYZ is the 3D center of the yellow top relative to camera.

SWING HANDLING
==============
The target may rock/bob. We handle this using:
    1) yellow ellipse detection
    2) outlier rejection
    3) short median filtering of image measurements
    4) exponential temporal averaging of XYZ

This is appropriate for water-level monitoring because target swing is usually
faster than the true lake-level variation.

USAGE
=====

Metric XYZ, for example focal length 1350 px:

    python target_3d_tracker.py input.mp4 output.csv \
        --focal-px 1350 \
        --annotated annotated.mp4

No focal length:

    python target_3d_tracker.py input.mp4 output.csv

Optional JSON comparison only:

    python target_3d_tracker.py input.mp4 output.csv \
        --focal-px 1350 \
        --compare-json simulation.json

The JSON is not read until AFTER all video estimates have been computed.
"""

import sys
import csv
import json
import math
import cv2
import numpy as np


# ============================================================
# User-settable physical parameter
# ============================================================

TARGET_DIAMETER_M = 1.0


# ============================================================
# Detection / temporal settings
# ============================================================

YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_AREA_PX = 35
MAX_AREA_FRACTION = 0.15
MIN_AXIS_RATIO = 0.12

# Median window rejects short target-swing / segmentation disturbances.
MEDIAN_SECONDS = 0.50

# EMA time constant: larger => more stable XYZ, slower response.
EMA_TIME_CONSTANT_S = 2.0

# Hard jump rejection after median filtering.
MAX_RELATIVE_MAJOR_JUMP = 0.30


# ============================================================
# Yellow circle / ellipse detection
# ============================================================

def detect_yellow_target(frame):
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
    max_area = H * W * MAX_AREA_FRACTION

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))

        if area < MIN_AREA_PX or area > max_area or len(c) < 5:
            continue

        ellipse = cv2.fitEllipse(c)
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

        perimeter = float(cv2.arcLength(c, True))
        compactness = 0.0
        if perimeter > 0.0:
            compactness = 4.0 * math.pi * area / (perimeter * perimeter)
        compactness = max(0.0, min(1.0, compactness))

        quality = 0.65 * fill_quality + 0.35 * compactness
        score = area * (0.25 + 0.75 * quality)

        if score > best_score:
            best_score = score
            best = {
                "u_px": float(u),
                "v_px": float(v),
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": ratio,
                "angle_deg": float(angle),
                "quality": float(quality),
                "ellipse": ellipse,
            }

    return best


# ============================================================
# Temporal filtering
# ============================================================

def rolling_nanmedian(values, window):
    x = np.asarray(values, dtype=np.float64)
    y = np.full_like(x, np.nan)

    half = window // 2

    for i in range(len(x)):
        a = max(0, i - half)
        b = min(len(x), i + half + 1)

        z = x[a:b]
        z = z[np.isfinite(z)]

        if z.size:
            y[i] = np.median(z)

    return y


def ema_filter_xyz(xyz, fps, tau_s):
    """
    Exponential temporal averaging.

    Missing samples are skipped.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    out = np.full_like(xyz, np.nan)

    dt = 1.0 / fps
    alpha = 1.0 - math.exp(-dt / max(1e-6, tau_s))

    state = None

    for i in range(len(xyz)):
        p = xyz[i]

        if not np.all(np.isfinite(p)):
            if state is not None:
                out[i] = state
            continue

        if state is None:
            state = p.copy()
        else:
            state = alpha * p + (1.0 - alpha) * state

        out[i] = state

    return out


# ============================================================
# 3D geometry
# ============================================================

def estimate_xyz_from_circle(
    u_px,
    v_px,
    major_px,
    focal_px,
    cx,
    cy,
    diameter_m
):
    """
    Approximate metric 3D center from known circular target diameter.

    We use the ellipse major axis as the projected circle diameter because
    target rocking mostly compresses the minor axis.

    Perspective relation:
        major_px ~= f * diameter_m / Z

    therefore:
        Z = f * diameter_m / major_px

    and pinhole back-projection:
        X = (u-cx) * Z / f
        Y = (v-cy) * Z / f

    This approximation is highly useful when:
        - target diameter << distance
        - disk tilt is moderate
        - major axis is measured robustly

    Swing is then suppressed by temporal filtering.
    """
    if (
        not np.isfinite(u_px)
        or not np.isfinite(v_px)
        or not np.isfinite(major_px)
        or major_px <= 0.0
        or focal_px <= 0.0
    ):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    Z = focal_px * diameter_m / major_px
    X = (u_px - cx) * Z / focal_px
    Y = (v_px - cy) * Z / focal_px

    return np.array([X, Y, Z], dtype=np.float64)


def normalized_ray(u, v, cx, cy, focal_px=None):
    """
    Unit camera ray.

    If focal is unknown, return a projective ray in normalized-by-f form:
        [(u-cx), (v-cy), 1]
    normalized only in arbitrary image units.

    If focal is known:
        [(u-cx)/f, (v-cy)/f, 1], normalized to unit length.
    """
    if not np.isfinite(u) or not np.isfinite(v):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    if focal_px is None:
        r = np.array([u - cx, v - cy, 1.0], dtype=np.float64)
    else:
        r = np.array(
            [(u - cx) / focal_px, (v - cy) / focal_px, 1.0],
            dtype=np.float64
        )

    n = np.linalg.norm(r)
    if n <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    return r / n


# ============================================================
# Optional JSON comparison ONLY
# ============================================================

def load_comparison_json(path):
    """
    This function is called only after estimation has finished.

    Supported optional fields:
      target.top_center_world_m
      camera.world_position_m

    If unavailable, comparison columns remain blank.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    camera_pos = None

    if (
        isinstance(obj.get("camera"), dict)
        and "world_position_m" in obj["camera"]
    ):
        camera_pos = np.asarray(
            obj["camera"]["world_position_m"], dtype=np.float64
        )

    gt = {}

    for row in obj.get("frames", []):
        idx = int(row.get("frame_index", -1))
        target = row.get("target", {})

        if idx < 0:
            continue

        if camera_pos is not None and "top_center_world_m" in target:
            pw = np.asarray(target["top_center_world_m"], dtype=np.float64)

            # This is world delta only; unless JSON contains camera orientation,
            # it should not be interpreted as camera-axis XYZ.
            gt[idx] = {
                "world_delta_from_camera_m": pw - camera_pos
            }

    return gt


# ============================================================
# CLI
# ============================================================

def parse_args(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    args = {
        "video": argv[1],
        "csv": argv[2],
        "focal_px": None,
        "annotated": None,
        "compare_json": None,
        "median_seconds": MEDIAN_SECONDS,
        "ema_seconds": EMA_TIME_CONSTANT_S,
    }

    i = 3

    while i < len(argv):
        a = argv[i]

        if a == "--focal-px":
            args["focal_px"] = float(argv[i + 1])
            i += 2

        elif a == "--annotated":
            args["annotated"] = argv[i + 1]
            i += 2

        elif a == "--compare-json":
            args["compare_json"] = argv[i + 1]
            i += 2

        elif a == "--median-seconds":
            args["median_seconds"] = float(argv[i + 1])
            i += 2

        elif a == "--ema-seconds":
            args["ema_seconds"] = float(argv[i + 1])
            i += 2

        else:
            raise ValueError(f"Unknown option: {a}")

    return args


# ============================================================
# Main processing
# ============================================================

def process(args):
    cap = cv2.VideoCapture(args["video"])

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args['video']}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cx = W / 2.0
    cy = H / 2.0

    rows = []

    raw_u = []
    raw_v = []
    raw_major = []
    raw_minor = []
    raw_ratio = []

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_target(frame)

        if det is None:
            u = v = major = minor = ratio = angle = quality = np.nan
            ellipse = None
        else:
            u = det["u_px"]
            v = det["v_px"]
            major = det["major_px"]
            minor = det["minor_px"]
            ratio = det["axis_ratio"]
            angle = det["angle_deg"]
            quality = det["quality"]
            ellipse = det["ellipse"]

        rows.append({
            "frame": idx,
            "time_s": idx / fps,
            "u_raw_px": u,
            "v_raw_px": v,
            "major_raw_px": major,
            "minor_raw_px": minor,
            "axis_ratio_raw": ratio,
            "ellipse_angle_deg": angle,
            "quality": quality,
            "ellipse": ellipse,
        })

        raw_u.append(u)
        raw_v.append(v)
        raw_major.append(major)
        raw_minor.append(minor)
        raw_ratio.append(ratio)

        idx += 1

    cap.release()

    # --------------------------------------------------------
    # Temporal median filtering of image measurements
    # --------------------------------------------------------

    median_window = max(3, int(round(args["median_seconds"] * fps)))
    if median_window % 2 == 0:
        median_window += 1

    u_med = rolling_nanmedian(raw_u, median_window)
    v_med = rolling_nanmedian(raw_v, median_window)
    major_med = rolling_nanmedian(raw_major, median_window)
    minor_med = rolling_nanmedian(raw_minor, median_window)
    ratio_med = rolling_nanmedian(raw_ratio, median_window)

    # Reject implausible major-axis jumps relative to temporal median.
    major_raw_np = np.asarray(raw_major, dtype=np.float64)

    good = np.isfinite(major_raw_np) & np.isfinite(major_med)
    rel = np.full(len(rows), np.nan)

    rel[good] = np.abs(major_raw_np[good] - major_med[good]) / np.maximum(
        major_med[good], 1e-9
    )

    # --------------------------------------------------------
    # Per-frame projective / metric position
    # --------------------------------------------------------

    xyz_raw = []
    rays = []
    range_over_f = []

    for i in range(len(rows)):
        if (
            not np.isfinite(u_med[i])
            or not np.isfinite(v_med[i])
            or not np.isfinite(major_med[i])
        ):
            xyz_raw.append([np.nan, np.nan, np.nan])
            rays.append([np.nan, np.nan, np.nan])
            range_over_f.append(np.nan)
            continue

        if np.isfinite(rel[i]) and rel[i] > MAX_RELATIVE_MAJOR_JUMP:
            xyz_raw.append([np.nan, np.nan, np.nan])
            rays.append([np.nan, np.nan, np.nan])
            range_over_f.append(np.nan)
            continue

        ray = normalized_ray(
            u_med[i], v_med[i], cx, cy, args["focal_px"]
        )

        rays.append(ray)

        # From major diameter:
        # Z/f = D_target / major_px
        z_over_f = TARGET_DIAMETER_M / major_med[i]

        if args["focal_px"] is not None:
            p = estimate_xyz_from_circle(
                u_med[i],
                v_med[i],
                major_med[i],
                args["focal_px"],
                cx,
                cy,
                TARGET_DIAMETER_M,
            )
            xyz_raw.append(p)

            distance = np.linalg.norm(p)
            range_over_f.append(distance / args["focal_px"])
        else:
            xyz_raw.append([np.nan, np.nan, np.nan])

            # scale-free forward-distance / focal value
            range_over_f.append(z_over_f)

    xyz_raw = np.asarray(xyz_raw, dtype=np.float64)
    rays = np.asarray(rays, dtype=np.float64)
    range_over_f = np.asarray(range_over_f, dtype=np.float64)

    # --------------------------------------------------------
    # EMA smoothing for target swing
    # --------------------------------------------------------

    xyz_avg = ema_filter_xyz(
        xyz_raw,
        fps=fps,
        tau_s=args["ema_seconds"]
    )

    # Store computed values.
    for i, row in enumerate(rows):
        row["u_filtered_px"] = u_med[i]
        row["v_filtered_px"] = v_med[i]
        row["major_filtered_px"] = major_med[i]
        row["minor_filtered_px"] = minor_med[i]
        row["axis_ratio_filtered"] = ratio_med[i]

        row["ray_x"] = rays[i, 0]
        row["ray_y"] = rays[i, 1]
        row["ray_z"] = rays[i, 2]

        row["range_over_f"] = range_over_f[i]

        row["X_raw_m"] = xyz_raw[i, 0]
        row["Y_raw_m"] = xyz_raw[i, 1]
        row["Z_raw_m"] = xyz_raw[i, 2]

        row["X_avg_m"] = xyz_avg[i, 0]
        row["Y_avg_m"] = xyz_avg[i, 1]
        row["Z_avg_m"] = xyz_avg[i, 2]

        if np.all(np.isfinite(xyz_raw[i])):
            row["distance_raw_m"] = float(np.linalg.norm(xyz_raw[i]))
        else:
            row["distance_raw_m"] = np.nan

        if np.all(np.isfinite(xyz_avg[i])):
            row["distance_avg_m"] = float(np.linalg.norm(xyz_avg[i]))
        else:
            row["distance_avg_m"] = np.nan

    # --------------------------------------------------------
    # Optional JSON loaded ONLY NOW, after estimation
    # --------------------------------------------------------

    comparison = {}

    if args["compare_json"]:
        comparison = load_comparison_json(args["compare_json"])

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fields = [
        "frame",
        "time_s",
        "u_raw_px",
        "v_raw_px",
        "major_raw_px",
        "minor_raw_px",
        "axis_ratio_raw",
        "ellipse_angle_deg",
        "quality",
        "u_filtered_px",
        "v_filtered_px",
        "major_filtered_px",
        "minor_filtered_px",
        "axis_ratio_filtered",
        "ray_x",
        "ray_y",
        "ray_z",
        "range_over_f",
        "X_raw_m",
        "Y_raw_m",
        "Z_raw_m",
        "distance_raw_m",
        "X_avg_m",
        "Y_avg_m",
        "Z_avg_m",
        "distance_avg_m",
    ]

    with open(args["csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            out = {k: row[k] for k in fields}

            for k, val in list(out.items()):
                if isinstance(val, (float, np.floating)):
                    if np.isfinite(val):
                        out[k] = f"{float(val):.6f}"
                    else:
                        out[k] = ""

            writer.writerow(out)

    # --------------------------------------------------------
    # Annotated video
    # --------------------------------------------------------

    if args["annotated"]:
        cap = cv2.VideoCapture(args["video"])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args["annotated"],
            fourcc,
            fps,
            (W, H)
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

            if (
                np.isfinite(row["u_filtered_px"])
                and np.isfinite(row["v_filtered_px"])
            ):
                c = (
                    int(round(row["u_filtered_px"])),
                    int(round(row["v_filtered_px"]))
                )

                cv2.circle(
                    frame,
                    c,
                    5,
                    (255, 0, 255),
                    -1,
                    cv2.LINE_AA
                )

            cv2.rectangle(
                frame,
                (20, 20),
                (650, 205),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"target diameter: {TARGET_DIAMETER_M:.3f} m",
                (40, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if args["focal_px"] is None:
                text = "metric XYZ unavailable: focal_px unknown"

                cv2.putText(
                    frame,
                    text,
                    (40, 101),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                rof = row["range_over_f"]

                if np.isfinite(rof):
                    txt2 = f"Z/f approx: {rof:.6f} m/px"
                else:
                    txt2 = "Z/f: invalid"

                cv2.putText(
                    frame,
                    txt2,
                    (40, 144),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            else:
                x = row["X_avg_m"]
                y = row["Y_avg_m"]
                z = row["Z_avg_m"]
                d = row["distance_avg_m"]

                if np.all(np.isfinite([x, y, z, d])):
                    t1 = f"AVG X,Y,Z = {x:+.3f}, {y:+.3f}, {z:+.3f} m"
                    t2 = f"AVG distance = {d:.3f} m"
                else:
                    t1 = "AVG X,Y,Z = invalid"
                    t2 = "AVG distance = invalid"

                cv2.putText(
                    frame,
                    t1,
                    (40, 101),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    t2,
                    (40, 144),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.putText(
                frame,
                f"swing averaging tau = {args['ema_seconds']:.2f} s",
                (40, 184),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            writer.write(frame)
            i += 1

        cap.release()
        writer.release()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"Frames processed      : {len(rows)}")
    print(f"Target diameter       : {TARGET_DIAMETER_M:.4f} m")
    print(f"Median window         : {args['median_seconds']:.3f} s")
    print(f"EMA time constant     : {args['ema_seconds']:.3f} s")

    if args["focal_px"] is None:
        print("Metric XYZ            : unavailable (focal length unknown)")
        print("Scale-free tracking   : available")
    else:
        good = xyz_avg[np.all(np.isfinite(xyz_avg), axis=1)]

        print(f"Focal length          : {args['focal_px']:.3f} px")
        print(f"Valid averaged XYZ    : {len(good)}")

        if len(good):
            med = np.median(good, axis=0)
            std = np.std(good, axis=0)

            print(
                f"Median XYZ            : "
                f"X={med[0]:+.4f}, Y={med[1]:+.4f}, Z={med[2]:+.4f} m"
            )
            print(
                f"XYZ std               : "
                f"X={std[0]:.4f}, Y={std[1]:.4f}, Z={std[2]:.4f} m"
            )

    print(f"CSV                   : {args['csv']}")

    if args["annotated"]:
        print(f"Annotated video       : {args['annotated']}")

    if args["compare_json"]:
        print(
            "JSON comparison file was loaded only after estimation; "
            "it was not used by the tracker."
        )


if __name__ == "__main__":
    process(parse_args(sys.argv))
