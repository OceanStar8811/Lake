#!/usr/bin/env python3

import argparse
import math
from collections import deque

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

TARGET_DIAMETER_M = 0.75

# Measurement window
MEASUREMENT_WINDOW_SEC = 10.0
MIN_VALID_FRAMES_PER_WINDOW = 15

# Yellow HSV range
YELLOW_HSV_LOW = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([45, 255, 255], dtype=np.uint8)

# Target detection constraints
MIN_TARGET_AREA_PX = 50
MAX_TARGET_AREA_RATIO = 0.30

MIN_AXIS_RATIO = 0.55
MAX_AXIS_RATIO = 1.00

# Temporal filtering
MAX_CENTER_JUMP_PX = 150.0

# Robust statistics
MAD_SCALE = 3.0

# Smoothing
SMOOTHING_WINDOW = 15


# ============================================================
# Robust statistics
# ============================================================

def robust_median(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return float("nan")

    return float(np.median(values))


def robust_mad(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return float("nan")

    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def robust_filter(values, mad_scale=MAD_SCALE):
    """
    Remove statistical outliers using median/MAD.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 3:
        return values

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-9:
        return values

    threshold = mad_scale * 1.4826 * mad

    mask = np.abs(values - median) <= threshold

    filtered = values[mask]

    if len(filtered) == 0:
        return values

    return filtered


# ============================================================
# Yellow target detection
# ============================================================

def detect_yellow_top(frame):
    """
    Detect the yellow circular top surface of the floating target.

    Returns:
        dict or None

    The returned geometry is image-space only.
    """

    height, width = frame.shape[:2]
    frame_area = width * height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        YELLOW_HSV_LOW,
        YELLOW_HSV_HIGH,
    )

    # Clean small noise.
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_TARGET_AREA_PX:
            continue

        if area > MAX_TARGET_AREA_RATIO * frame_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter)
        )

        # Very irregular yellow regions are unlikely to be
        # the circular floating target.
        if circularity < 0.35:
            continue

        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)

        (_, _), (axis_a, axis_b), angle = ellipse

        if axis_a <= 0 or axis_b <= 0:
            continue

        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)

        axis_ratio = minor / major

        if axis_ratio < MIN_AXIS_RATIO:
            continue

        # Prefer larger and more circular candidates.
        score = (
            math.sqrt(area)
            * axis_ratio
            * (0.5 + 0.5 * min(circularity, 1.0))
        )

        candidates.append(
            {
                "center": ellipse[0],
                "major_px": float(major),
                "minor_px": float(minor),
                "axis_ratio": float(axis_ratio),
                "angle": float(angle),
                "area_px": float(area),
                "circularity": float(circularity),
                "score": float(score),
                "ellipse": ellipse,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# Temporal detection validation
# ============================================================

def accept_temporal_detection(
    detection,
    previous_detection,
    frame_width,
    frame_height,
):
    """
    Reject implausible frame-to-frame jumps.

    This is deliberately conservative because the target should
    normally move smoothly in the image.
    """

    if detection is None:
        return None

    if previous_detection is None:
        return detection

    x1, y1 = previous_detection["center"]
    x2, y2 = detection["center"]

    distance = math.hypot(
        x2 - x1,
        y2 - y1,
    )

    # Allow larger jumps for very large images.
    allowed_jump = max(
        MAX_CENTER_JUMP_PX,
        0.08 * math.hypot(frame_width, frame_height),
    )

    if distance > allowed_jump:
        return None

    return detection


# ============================================================
# Image-space measurement
# ============================================================

def target_image_measurement(detection):
    """
    Calculate image-space quantities from the detected circular top.

    The important quantity is the apparent diameter.

    For an approximately horizontal circular target, the apparent
    image size changes as the target moves farther/closer to the
    camera.

    No camera focal length is assumed here.
    """

    apparent_diameter_px = detection["major_px"]

    if apparent_diameter_px <= 0:
        return None

    # Relative scale:
    #
    #     apparent_diameter_px / real_diameter_m
    #
    # This is proportional to image scale at the target.
    #
    # We intentionally do not convert this into an absolute
    # distance because focal length and camera pose are unknown.

    image_scale_px_per_m = (
        apparent_diameter_px / TARGET_DIAMETER_M
    )

    return {
        "diameter_px": apparent_diameter_px,
        "diameter_m": TARGET_DIAMETER_M,
        "image_scale_px_per_m": image_scale_px_per_m,
        "center_x": float(detection["center"][0]),
        "center_y": float(detection["center"][1]),
        "axis_ratio": detection["axis_ratio"],
        "area_px": detection["area_px"],
        "circularity": detection["circularity"],
    }


# ============================================================
# Relative level estimation
# ============================================================

class RelativeLevelEstimator:
    """
    Estimate relative water-level changes from apparent target size.

    Because the camera focal length and absolute camera geometry
    are unknown, the first stable measurement is used as the
    reference.

    Important:
        This produces a relative level estimate, not an absolute
        elevation referenced to sea level or a surveyed datum.
    """

    def __init__(self):
        self.reference_diameter_px = None
        self.reference_distance_proxy = None

    def initialize(self, diameter_px):
        if diameter_px <= 0:
            return False

        self.reference_diameter_px = float(diameter_px)

        # Perspective depth is approximately inversely proportional
        # to apparent object size.
        #
        # We use a normalized distance proxy:
        #
        #       reference_size / current_size
        #
        # so the reference state is exactly 1.0.
        self.reference_distance_proxy = 1.0

        return True

    def estimate(self, diameter_px):
        if self.reference_diameter_px is None:
            self.initialize(diameter_px)
            return 0.0

        if diameter_px <= 0:
            return None

        current_proxy = (
            self.reference_diameter_px / diameter_px
        )

        return current_proxy - self.reference_distance_proxy


# ============================================================
# Window processing
# ============================================================

def summarize_window(
    measurements,
    reference_diameter_px,
):
    """
    Produce a robust summary of one measurement window.
    """

    if len(measurements) < MIN_VALID_FRAMES_PER_WINDOW:
        return None

    diameters = np.asarray(
        [m["diameter_px"] for m in measurements],
        dtype=np.float64,
    )

    diameters = robust_filter(diameters)

    if len(diameters) < MIN_VALID_FRAMES_PER_WINDOW:
        return None

    median_diameter_px = robust_median(diameters)

    if reference_diameter_px <= 0:
        return None

    normalized_distance = (
        reference_diameter_px / median_diameter_px
    )

    relative_change_proxy = normalized_distance - 1.0

    return {
        "median_diameter_px": median_diameter_px,
        "normalized_distance": normalized_distance,
        "relative_change_proxy": relative_change_proxy,
        "valid_frames": len(diameters),
    }


# ============================================================
# Video measurement
# ============================================================

def measure(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_sec = (
        total_frames / fps
        if total_frames > 0
        else 0.0
    )

    print()
    print("Lake Water Level Measurement")
    print("--------------------------------")
    print(f"Video       : {video_path}")
    print(f"Resolution  : {frame_width} x {frame_height}")
    print(f"FPS         : {fps:.2f}")

    if duration_sec > 0:
        print(f"Duration    : {duration_sec:.2f} s")

    print()
    print("Target")
    print(f"  Diameter  : {TARGET_DIAMETER_M:.3f} m")
    print()
    print("Mode")
    print("  No camera calibration")
    print("  No external model")
    print("  No CSV output")
    print("  No annotated video")
    print("  Relative measurement")
    print()

    previous_detection = None

    all_measurements = []

    window_measurements = []

    reference_diameter_px = None

    relative_estimator = RelativeLevelEstimator()

    smoothed_diameters = deque(
        maxlen=SMOOTHING_WINDOW
    )

    frame_index = 0

    window_start_time = None

    window_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp_sec = frame_index / fps

        if window_start_time is None:
            window_start_time = timestamp_sec

        detection = detect_yellow_top(frame)

        detection = accept_temporal_detection(
            detection,
            previous_detection,
            frame_width,
            frame_height,
        )

        if detection is not None:
            measurement = target_image_measurement(
                detection
            )

            if measurement is not None:
                diameter_px = measurement["diameter_px"]

                # Initialize reference using the first reliable
                # detection.
                if reference_diameter_px is None:
                    reference_diameter_px = diameter_px

                    relative_estimator.initialize(
                        diameter_px
                    )

                    print(
                        f"Reference initialized at "
                        f"{timestamp_sec:.2f} s: "
                        f"{diameter_px:.2f} px"
                    )

                smoothed_diameters.append(
                    diameter_px
                )

                # Use a short median filter for display/measurement.
                smoothed_diameter_px = robust_median(
                    smoothed_diameters
                )

                measurement["diameter_px"] = (
                    smoothed_diameter_px
                )

                relative_change = (
                    relative_estimator.estimate(
                        smoothed_diameter_px
                    )
                )

                measurement[
                    "relative_change_proxy"
                ] = relative_change

                measurement["timestamp_sec"] = (
                    timestamp_sec
                )

                window_measurements.append(
                    measurement
                )

                all_measurements.append(
                    measurement
                )

                previous_detection = detection

        # End of measurement window
        if (
            timestamp_sec - window_start_time
            >= MEASUREMENT_WINDOW_SEC
        ):
            window_number += 1

            summary = summarize_window(
                window_measurements,
                reference_diameter_px
                if reference_diameter_px is not None
                else 0.0,
            )

            print()

            if summary is None:
                print(
                    f"Window {window_number:03d} "
                    f"[{window_start_time:.1f} - "
                    f"{timestamp_sec:.1f} s]"
                )
                print(
                    "  Insufficient reliable "
                    "target detections"
                )
            else:
                print(
                    f"Window {window_number:03d} "
                    f"[{window_start_time:.1f} - "
                    f"{timestamp_sec:.1f} s]"
                )

                print(
                    f"  Valid frames       : "
                    f"{summary['valid_frames']}"
                )

                print(
                    f"  Median target size : "
                    f"{summary['median_diameter_px']:.2f} px"
                )

                print(
                    f"  Relative depth "
                    f"proxy              : "
                    f"{summary['relative_change_proxy']:+.5f}"
                )

                print(
                    "  Note: this is a "
                    "relative perspective "
                    "measurement, not metres."
                )

            window_measurements = []

            window_start_time = timestamp_sec

        frame_index += 1

    cap.release()

    print()
    print("--------------------------------")
    print("Final Result")
    print("--------------------------------")

    if reference_diameter_px is None:
        print("No reliable yellow target detected.")
        return

    if not all_measurements:
        print("No measurements available.")
        return

    final_diameters = np.asarray(
        [
            m["diameter_px"]
            for m in all_measurements
        ],
        dtype=np.float64,
    )

    final_diameters = robust_filter(
        final_diameters
    )

    final_median_diameter = robust_median(
        final_diameters
    )

    final_relative_proxy = (
        reference_diameter_px
        / final_median_diameter
        - 1.0
    )

    print(
        f"Reference target size : "
        f"{reference_diameter_px:.2f} px"
    )

    print(
        f"Final median size     : "
        f"{final_median_diameter:.2f} px"
    )

    print(
        f"Relative depth proxy  : "
        f"{final_relative_proxy:+.5f}"
    )

    print()
    print(
        "A positive value means the target appears "
        "smaller than at the reference state."
    )

    print(
        "An absolute water-level value in metres "
        "cannot be obtained from this video alone "
        "without an additional geometric reference."
    )


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure relative lake water-level change "
            "from a floating yellow circular target."
        )
    )

    parser.add_argument(
        "video",
        help="Input video file",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        measure(args.video)

    except KeyboardInterrupt:
        print("\nStopped.")

    except Exception as exc:
        print(f"\nError: {exc}")
        raise


if __name__ == "__main__":
    main()