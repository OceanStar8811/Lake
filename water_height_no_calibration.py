#!/usr/bin/env python3
"""
water_height_no_calibration.py

Calibration-free (with respect to camera FoV/focal length) estimation of the
vertical camera-to-water height from a yellow circular floating target.

External dependencies:
    - OpenCV (cv2)
    - NumPy

Python standard-library modules are used for CLI/CSV/JSON only.

CORE GEOMETRY
=============
Assume the yellow top is a circular disk and, on average, is parallel to the
horizontal water surface.

For a sufficiently small disk compared with its camera distance:

    q = minor_axis / major_axis ~= H / sqrt(R^2 + H^2)

where:
    H = vertical camera-to-water height
    R = horizontal camera-to-target distance
    q = observed ellipse minor/major axis ratio

Therefore:

    H / R = q / sqrt(1 - q^2)

and, if R is known:

    H = R * q / sqrt(1 - q^2)

IMPORTANT:
    - Camera FoV is NOT needed.
    - Camera focal length is NOT needed.
    - The JSON is NOT used for estimation.
    - JSON can be supplied optionally ONLY to compare prediction vs ground truth.

WHAT IS REQUIRED FOR ABSOLUTE METRES?
=====================================
You need one metric horizontal scale: the horizontal camera-to-float distance R.
Without R (or some equivalent metric reference), a single monocular camera with
unknown intrinsics cannot recover absolute metres uniquely. In that case this
script still outputs H/R.

SWINGING FLOAT
==============
Rocking changes the ellipse axis ratio even at fixed water height. To reduce
this, the script applies:
    1) ellipse-quality rejection;
    2) temporal median filtering of q;
    3) optional long smoothing window for swinging targets.

This works well when rocking is oscillatory around the horizontal position.
Large persistent tilt will bias the result.

USAGE
=====

Height-changing video, known horizontal target distance 20 m:

    python water_height_no_calibration.py \
        input.mp4 results.csv --range-m 20 --annotated annotated.mp4

Swinging video, known horizontal target distance 18 m:

    python water_height_no_calibration.py \
        swinging.mp4 swing.csv --range-m 18 \
        --median-seconds 2.0 --annotated swing_annotated.mp4

Optional JSON comparison ONLY:

    python water_height_no_calibration.py \
        input.mp4 results.csv --range-m 20 \
        --compare-json ground_truth.json

Expected JSON field:
    frames[i]["frame_index"]
    frames[i]["camera_to_water_vertical_height_m"]
"""

import sys
import csv
import json
import math
import cv2
import numpy as np


# ------------------------------------------------------------
# Detection defaults
# ------------------------------------------------------------

YELLOW_LO = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

MIN_AREA_PX = 35
MAX_AREA_FRACTION = 0.15
MIN_AXIS_RATIO = 0.08
MAX_AXIS_RATIO = 0.999


def detect_yellow_ellipse(frame):
    """Detect the yellow circular top and fit an ellipse."""
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

        q = minor / major

        if not (MIN_AXIS_RATIO <= q <= MAX_AXIS_RATIO):
            continue

        ellipse_area = math.pi * major * minor / 4.0
        if ellipse_area <= 1.0:
            continue

        fill = area / ellipse_area
        fill_quality = max(0.0, 1.0 - abs(fill - 1.0))

        perimeter = float(cv2.arcLength(c, True))
        compactness = 0.0
        if perimeter > 0:
            compactness = 4.0 * math.pi * area / (perimeter * perimeter)
        compactness = max(0.0, min(1.0, compactness))

        quality = 0.65 * fill_quality + 0.35 * compactness
        score = area * (0.25 + 0.75 * quality)

        if score > best_score:
            best_score = score
            best = {
                "center_px": (float(u), float(v)),
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": q,
                "angle_deg": float(angle),
                "quality": float(quality),
                "ellipse": ellipse,
            }

    return best


def rolling_nanmedian(x, window):
    x = np.asarray(x, dtype=np.float64)
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


def q_to_height_ratio(q):
    """Return H/R from ellipse aspect ratio q."""
    if not np.isfinite(q) or q <= 0.0 or q >= 1.0:
        return float("nan")

    denom = math.sqrt(max(1e-12, 1.0 - q*q))
    return q / denom


def read_gt_json(path):
    """
    Optional. Used ONLY for comparison, never estimation.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    gt = {}

    for row in obj.get("frames", []):
        if (
            "frame_index" in row
            and "camera_to_water_vertical_height_m" in row
        ):
            gt[int(row["frame_index"])] = float(
                row["camera_to_water_vertical_height_m"]
            )

    return gt


def parse_args(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    args = {
        "video": argv[1],
        "csv": argv[2],
        "range_m": None,
        "median_seconds": 1.0,
        "annotated": None,
        "compare_json": None,
    }

    i = 3
    while i < len(argv):
        a = argv[i]

        if a == "--range-m":
            args["range_m"] = float(argv[i+1])
            i += 2
        elif a == "--median-seconds":
            args["median_seconds"] = float(argv[i+1])
            i += 2
        elif a == "--annotated":
            args["annotated"] = argv[i+1]
            i += 2
        elif a == "--compare-json":
            args["compare_json"] = argv[i+1]
            i += 2
        else:
            raise ValueError(f"Unknown argument: {a}")

    return args


def run(args):
    cap = cv2.VideoCapture(args["video"])

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args['video']}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    raw_q = []

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detect_yellow_ellipse(frame)

        if det is None:
            detections.append({
                "frame": idx,
                "time_s": idx / fps,
                "u_px": np.nan,
                "v_px": np.nan,
                "major_px": np.nan,
                "minor_px": np.nan,
                "axis_ratio_raw": np.nan,
                "angle_deg": np.nan,
                "quality": np.nan,
                "ellipse": None,
            })
            raw_q.append(np.nan)
        else:
            u, v = det["center_px"]

            detections.append({
                "frame": idx,
                "time_s": idx / fps,
                "u_px": u,
                "v_px": v,
                "major_px": det["major_px"],
                "minor_px": det["minor_px"],
                "axis_ratio_raw": det["axis_ratio"],
                "angle_deg": det["angle_deg"],
                "quality": det["quality"],
                "ellipse": det["ellipse"],
            })
            raw_q.append(det["axis_ratio"])

        idx += 1

    cap.release()

    window = max(3, int(round(args["median_seconds"] * fps)))
    if window % 2 == 0:
        window += 1

    q_filtered = rolling_nanmedian(raw_q, window)

    H_over_R = np.array(
        [q_to_height_ratio(q) for q in q_filtered],
        dtype=np.float64
    )

    if args["range_m"] is not None:
        height_m = H_over_R * float(args["range_m"])
    else:
        height_m = np.full_like(H_over_R, np.nan)

    gt = {}
    if args["compare_json"]:
        gt = read_gt_json(args["compare_json"])

    errors = []

    # CSV
    fields = [
        "frame",
        "time_s",
        "u_px",
        "v_px",
        "major_px",
        "minor_px",
        "axis_ratio_raw",
        "axis_ratio_filtered",
        "ellipse_angle_deg",
        "quality",
        "height_over_range",
        "estimated_height_m",
        "ground_truth_height_m",
        "error_m",
    ]

    with open(args["csv"], "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()

        for i, row in enumerate(detections):
            gt_h = gt.get(i, np.nan)

            if np.isfinite(height_m[i]) and np.isfinite(gt_h):
                err = height_m[i] - gt_h
                errors.append(err)
            else:
                err = np.nan

            out = {
                "frame": i,
                "time_s": row["time_s"],
                "u_px": row["u_px"],
                "v_px": row["v_px"],
                "major_px": row["major_px"],
                "minor_px": row["minor_px"],
                "axis_ratio_raw": row["axis_ratio_raw"],
                "axis_ratio_filtered": q_filtered[i],
                "ellipse_angle_deg": row["angle_deg"],
                "quality": row["quality"],
                "height_over_range": H_over_R[i],
                "estimated_height_m": height_m[i],
                "ground_truth_height_m": gt_h,
                "error_m": err,
            }

            for k, v in list(out.items()):
                if isinstance(v, (float, np.floating)):
                    if np.isfinite(v):
                        out[k] = f"{float(v):.6f}"
                    else:
                        out[k] = ""

            wcsv.writerow(out)

    # Optional annotated video
    if args["annotated"]:
        cap = cv2.VideoCapture(args["video"])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args["annotated"], fourcc, fps, (W, H)
        )

        i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            row = detections[i]

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
                    (255, 0, 255),
                    -1, cv2.LINE_AA
                )

            cv2.rectangle(
                frame, (20, 20), (650, 175),
                (25, 25, 25), -1
            )

            q = q_filtered[i]
            hr = H_over_R[i]

            cv2.putText(
                frame,
                f"ellipse q (filtered): {q:.4f}" if np.isfinite(q)
                else "ellipse q: invalid",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255,255,255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"H/R: {hr:.4f}" if np.isfinite(hr)
                else "H/R: invalid",
                (40, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255,255,255),
                2,
                cv2.LINE_AA
            )

            if args["range_m"] is not None and np.isfinite(height_m[i]):
                txt = f"estimated camera-water height: {height_m[i]:.3f} m"
            else:
                txt = "absolute H needs --range-m"

            cv2.putText(
                frame,
                txt,
                (40, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255,255,255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)
            i += 1

        cap.release()
        writer.release()

    good_ratio = H_over_R[np.isfinite(H_over_R)]

    print(f"Frames processed : {len(detections)}")
    print(f"Valid H/R frames : {len(good_ratio)}")

    if good_ratio.size:
        print(
            f"H/R median       : {np.median(good_ratio):.6f}"
        )

    if args["range_m"] is not None:
        good_h = height_m[np.isfinite(height_m)]
        if good_h.size:
            print(
                f"Estimated H      : median={np.median(good_h):.3f} m, "
                f"min={np.min(good_h):.3f}, max={np.max(good_h):.3f}"
            )

    if errors:
        e = np.asarray(errors, dtype=np.float64)
        print("JSON comparison only:")
        print(f"  MAE  : {np.mean(np.abs(e)):.4f} m")
        print(f"  RMSE : {math.sqrt(np.mean(e*e)):.4f} m")
        print(f"  bias : {np.mean(e):+.4f} m")

    print(f"CSV              : {args['csv']}")

    if args["annotated"]:
        print(f"Annotated video  : {args['annotated']}")


if __name__ == "__main__":
    run(parse_args(sys.argv))
