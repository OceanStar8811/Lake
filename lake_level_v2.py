#!/usr/bin/env python3
"""
lake_level_v2.py
================

Water-level measurement from a floating circular target (yellow top face)
using a single fixed camera whose pose is NOT known.

WHY THIS WORKS WITH UNKNOWN CAMERA POSE
---------------------------------------

A single fixed camera is fully described (for our purposes) by:
    * intrinsics K = [[fx, 0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]]   -- assumed known
    * extrinsics (R, t)               -- assumed UNKNOWN

The extrinsics describe where the camera is and how it is oriented. Without
them, a general 3-D point cannot be converted to a metric height. But the
floating target is not a general point:

    (1) Its top face is a CIRCLE of known diameter D.
        A circle projects to an ellipse. From the ellipse size and D,
        the pinhole model recovers the metric distance to the target.

    (2) Its top face is HORIZONTAL (parallel to the water).
        The normal of the plane containing that circle, expressed in
        camera coordinates, is the "up" direction. We recover it directly
        from the ellipse shape and orientation (circle-pose-from-conic).

So each frame gives us, in camera coordinates:
    t = 3-D position of the circle center
    n = unit normal of the circle plane (= up direction)

From t and n the vertical drop of the target below the camera is:
    H_top = -t . n           (with n oriented upward, this is >= 0)

Finally, the water surface is BELOW the top face by h_target:
    H_water = H_top + h_target

No extrinsics needed. No calibration video needed. Only K, D, h_target.

WHAT THE PIPELINE DOES PER FRAME
--------------------------------

    1. Colour-segment the yellow top.
    2. Fit an ellipse to the blob with a sub-pixel conic fit.
    3. Convert the ellipse to normalized camera coordinates via K.
    4. Solve circle-pose-from-conic: get two candidate (t, n).
    5. Disambiguate the two candidates using an adaptive "up tracker".
    6. Convert to water level and push to a sliding median.

Once per 5-10 s the estimator emits the median of the window, which
suppresses wave motion and rocking of the float.

BUILT-IN DIAGNOSTICS
--------------------

Every run prints a per-stage counter summary at the end. If the run
produced no water-level lines, the summary tells you exactly which
stage rejected the data:

    detect   -- yellow mask never fires
    pose     -- ellipse fit fails or axis ratio below threshold
    filter   -- every candidate pose lies outside level_range
    tracker  -- two-fold ambiguity not yet resolved
    window   -- too few valid samples per reporting window

Use this to decide whether to fix --fx/--fy, --D, HSV thresholds, or
--level_min/--level_max before re-running.

USAGE
-----

Self-test (no arguments):
    python lake_level_v2.py

Field run:
    python lake_level_v2.py --video samples/fixed_height_10m_swinging_float_camera_view.mp4 --fx 1371.02 --fy 1371.02 --cx 960 --cy 540 --D 1.0 --h_target 0.08 --level_min 8 --level_max 12 --window 5 --annotated lake_annotated.mp4

Dependencies: numpy, opencv-python.
"""

from __future__ import annotations
import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# =====================================================================
# SECTION 1 -- Sub-pixel ellipse fit
# =====================================================================

def fit_ellipse_conic(pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit an ellipse to an N x 2 array of (x, y) pixel coordinates using the
    Fitzgibbon direct least-squares method. Returns a symmetric 3x3 conic
    matrix C with [x, y, 1] C [x, y, 1]^T = 0, or None on failure.
    """
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]

    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None

    M = S1 + S2 @ T
    Cm = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float64)

    try:
        M2 = np.linalg.solve(Cm, M)
    except np.linalg.LinAlgError:
        return None

    w, V = np.linalg.eig(M2)
    cond = 4.0 * V[0, :] * V[2, :] - V[1, :] ** 2
    good = np.where(cond > 0)[0]
    if len(good) == 0:
        return None

    a1 = V[:, good[0]].real
    a = np.concatenate([a1, T @ a1])
    A, B, C, D, E, F = a

    C_out = np.array([[A,     B / 2, D / 2],
                      [B / 2, C,     E / 2],
                      [D / 2, E / 2, F    ]])

    if not np.all(np.isfinite(C_out)):
        return None
    return C_out


# =====================================================================
# SECTION 2 -- Yellow top-face detection
# =====================================================================

def _expected_pixel_diameter(K: np.ndarray, D_m: float, Z_m: float) -> float:
    """Approximate face-on projected diameter in pixels."""
    fx = 0.5 * (K[0, 0] + K[1, 1])
    return fx * D_m / max(Z_m, 1e-6)


def detect_yellow_top(bgr: np.ndarray,
                      K: np.ndarray,
                      D_m: float,
                      level_range: Tuple[float, float],
                      yellow_lo=(18, 80, 80),
                      yellow_hi=(42, 255, 255),
                      axis_ratio_min: float = 0.35) -> Optional[dict]:
    """
    Detect the yellow circular top and fit a sub-pixel conic.
    See module docstring for the geometric reasoning.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(yellow_lo, np.uint8),
                       np.array(yellow_hi, np.uint8))

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    H, W = bgr.shape[:2]
    Zmax = max(level_range[1], 1.0)
    diam_far_px = _expected_pixel_diameter(K, D_m, Zmax)
    min_area = max(15.0, 0.15 * diam_far_px * diam_far_px)
    max_area = 0.15 * H * W

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area or len(c) < 5:
            continue

        pts = c.reshape(-1, 2).astype(np.float64)
        C_pix = fit_ellipse_conic(pts)
        if C_pix is None:
            continue

        axes = _conic_axes(C_pix)
        if axes is None:
            continue
        major, minor, angle, center = axes
        if major <= 1.0 or minor <= 1.0:
            continue

        ratio = minor / major
        if ratio < axis_ratio_min:
            continue

        ellipse_area = math.pi * 0.5 * major * 0.5 * minor
        fill = area / ellipse_area if ellipse_area > 0 else 0.0
        quality = max(0.0, 1.0 - abs(fill - 1.0))

        score = area * (0.3 + 0.7 * quality)
        if score > best_score:
            best_score = score
            best = {
                "C_pix": C_pix,
                "C_norm": K.T @ C_pix @ K,
                "center_px": center,
                "major_px": major,
                "minor_px": minor,
                "axis_ratio": ratio,
                "angle_deg": float(np.degrees(angle)),
                "quality": float(quality),
                "contour": c,
            }
    return best


def _conic_axes(C: np.ndarray):
    """Return (major, minor, angle_rad, center_xy) of a 3x3 conic."""
    A, B = C[0, 0], C[0, 1]
    Cc = C[1, 1]
    d, e = C[0, 2], C[1, 2]
    F = C[2, 2]

    try:
        center = np.linalg.solve(np.array([[A, B], [B, Cc]]),
                                 np.array([-d, -e]))
    except np.linalg.LinAlgError:
        return None
    x0, y0 = center

    Fp = F + d * x0 + e * y0
    w, V = np.linalg.eigh(np.array([[A, B], [B, Cc]]))
    if np.any(np.abs(w) < 1e-15):
        return None

    axes_sq = -Fp / w
    if np.any(axes_sq <= 0):
        return None

    axes = 2.0 * np.sqrt(axes_sq)
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    major, minor = float(axes[0]), float(axes[1])
    v = V[:, order[0]]
    angle = math.atan2(v[1], v[0])
    return major, minor, angle, (float(x0), float(y0))


# =====================================================================
# SECTION 3 -- Circle-pose-from-conic
# =====================================================================

def circle_pose_from_conic(C_norm: np.ndarray, r: float
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return up to two candidate (t, n) solutions for a circle of known
    radius r in normalized camera coordinates.
    """
    C = 0.5 * (C_norm + C_norm.T)

    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    if w[2] > 0:
        w = -w
        V = V[:, ::-1]
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]

    l1, l2, l3 = w
    if l1 <= 0 or l2 <= 0 or l3 >= 0:
        return []
    if (l1 - l3) < 1e-15:
        return []

    s1 = math.sqrt(max(0.0, (l1 - l2) / (l1 - l3)))
    s3 = math.sqrt(max(0.0, (l2 - l3) / (l1 - l3)))

    denom = math.sqrt(max(0.0, -l1 * l3))
    if denom < 1e-15:
        return []
    d = r * l2 / denom
    if d <= 0:
        return []

    sols = []
    for sign in (+1.0, -1.0):
        n_local = np.array([sign * s1, 0.0, s3])
        a_local = np.array([s3, 0.0, -sign * s1])
        n = V @ n_local
        a = V @ a_local
        u_c = -d * sign * s1 * s3 * (l1 - l3) / l2
        t = d * n + u_c * a
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        sols.append((t, n / nn))
    return sols


def up_and_height(t: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fix the sign of n so it points up; return (U, H) with H >= 0."""
    U = -n if float(n @ t) > 0.0 else n
    return U, -float(t @ U)


# =====================================================================
# SECTION 4 -- Adaptive up-direction tracker
# =====================================================================

class UpDirectionTracker:
    """
    Track the correct branch of the two-fold ambiguity across frames.

    For a fixed camera the true up direction in camera coordinates is
    constant. The false branch wanders because it mirrors the target's
    tilt. We therefore lock onto the branch with lower dispersion during
    warmup, then keep the lock as long as each frame's chosen up vector
    stays within relock_threshold of the locked direction.

    A swinging target tilts the top face, so the observed up direction
    oscillates. relock_threshold must exceed the swing amplitude (in
    cosine units). relock_patience prevents a single bad frame from
    destroying a good lock, and adapt_rate lets the lock drift slowly
    to follow gradual changes.

    Parameters
    ----------
    warmup_frames : int
        Number of accumulations before the first lock.
    relock_threshold : float
        Cosine distance beyond which the lock is considered broken.
        cos(theta_max) = 1 - relock_threshold. 0.15 -> ~32 degrees.
    relock_patience : int
        Consecutive mismatches required to actually reset the lock.
    adapt_rate : float
        EMA weight for drifting the locked direction on good frames.
    """

    def __init__(self, warmup_frames: int = 30,
                 relock_threshold: float = 0.15,
                 relock_patience: int = 15,
                 adapt_rate: float = 0.01):
        self.warmup = warmup_frames
        self.relock_threshold = relock_threshold
        self.relock_patience = relock_patience
        self.adapt_rate = adapt_rate

        self.sums = [np.zeros(3), np.zeros(3)]
        self.counts = [0, 0]
        self.frames_seen = 0
        self.U_locked: Optional[np.ndarray] = None
        self.mismatch_count = 0

    def update(self,
               candidates: List[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]
               ) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
        self.frames_seen += 1
        if len(candidates) == 0:
            return None

        # --- If locked, keep the lock unless it is clearly broken.
        if self.U_locked is not None:
            best = max(candidates, key=lambda c: float(c[0] @ self.U_locked))
            dot = float(best[0] @ self.U_locked)

            if dot < 1.0 - self.relock_threshold:
                self.mismatch_count += 1
                if self.mismatch_count >= self.relock_patience:
                    # Sustained mismatch: real change (camera bump, etc.).
                    self.U_locked = None
                    self.sums = [np.zeros(3), np.zeros(3)]
                    self.counts = [0, 0]
                    self.frames_seen = 1
                    self.mismatch_count = 0
                # During the patience window, still trust the lock.
                return best

            # Good match: slowly drift the lock toward the observation so
            # the reference can follow very slow changes without a reset.
            self.mismatch_count = 0
            u_new = best[0]
            self.U_locked = ((1.0 - self.adapt_rate) * self.U_locked
                             + self.adapt_rate * u_new)
            nrm = np.linalg.norm(self.U_locked)
            if nrm > 1e-12:
                self.U_locked /= nrm
            return best

        # --- Only one candidate: no ambiguity to resolve.
        if len(candidates) == 1:
            return candidates[0]

        # --- Warmup: accumulate both branches.
        for i in range(min(2, len(candidates))):
            self.sums[i] += candidates[i][0]
            self.counts[i] += 1

        if (self.frames_seen >= self.warmup
                and self.counts[0] > 0
                and self.counts[1] > 0):
            means, disp = [], []
            for i in range(2):
                m = self.sums[i] / self.counts[i]
                nm = np.linalg.norm(m)
                disp.append(1.0 - min(1.0, nm))
                means.append(m / max(nm, 1e-12))
            self.U_locked = means[0] if disp[0] <= disp[1] else means[1]
            best = max(candidates, key=lambda c: float(c[0] @ self.U_locked))
            return best

        return None


# =====================================================================
# SECTION 5 -- Estimator with stage counters
# =====================================================================

@dataclass
class EstimatorConfig:
    """All user-tunable parameters for one measurement run."""
    D_m: float = 0.8
    h_target_m: float = 0.20
    window_sec: float = 8.0
    level_range_m: Tuple[float, float] = (0.0, 60.0)
    min_valid_frames: int = 10
    axis_ratio_min: float = 0.35


@dataclass
class StageCounters:
    """
    Per-stage counts for diagnosing why no measurement was emitted.
    Every frame advances exactly one of the *_reached_* counters or
    increments a rejection counter.
    """
    frames_total: int = 0
    detect_hits: int = 0            # yellow blob detected and conic fit OK
    pose_two_sols: int = 0          # circle_pose_from_conic returned 2 sols
    pose_one_sol: int = 0           # (rare) returned 1 sol
    pose_zero_sol: int = 0          # solver rejected the conic
    plausible_candidates: int = 0   # at least one candidate survived level_range
    tracker_locked_in: int = 0      # tracker returned a chosen candidate
    samples_pushed: int = 0         # sample appended to sliding window
    reports_emitted: int = 0        # _current returned a value at report time
    reports_suppressed_warmup: int = 0  # _current returned None because <min_valid_frames

    # Rejection reason for the *last* non-detect frame (for reporting).
    last_reject_reason: str = "none"

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("  frames processed          : " + str(self.frames_total))
        lines.append("  yellow+conic detections   : "
                     + f"{self.detect_hits} "
                     + f"({100.0 * self.detect_hits / max(self.frames_total, 1):.1f}%)")
        lines.append("  pose solver returned 2    : " + str(self.pose_two_sols))
        lines.append("  pose solver returned 1    : " + str(self.pose_one_sol))
        lines.append("  pose solver returned 0    : " + str(self.pose_zero_sol))
        lines.append("  plausible candidates      : " + str(self.plausible_candidates))
        lines.append("  tracker chose a solution  : " + str(self.tracker_locked_in))
        lines.append("  samples pushed to window  : " + str(self.samples_pushed))
        lines.append("  reports emitted           : " + str(self.reports_emitted))
        lines.append("  reports suppressed (warmup): " + str(self.reports_suppressed_warmup))
        lines.append("  last rejection reason     : " + self.last_reject_reason)
        return "\n".join(lines)


class WaterLevelEstimator:
    """
    Stateful, per-frame water-level estimator.
    """

    def __init__(self, K: np.ndarray, dist: Optional[np.ndarray],
                 cfg: EstimatorConfig):
        self.K = K.astype(np.float64)
        self.dist = None if dist is None else np.asarray(dist, np.float64)
        self.cfg = cfg
        self.tracker = UpDirectionTracker()

        self.samples: deque = deque()
        self.last_det: Optional[dict] = None

        # Diagnostics.
        self.stats = StageCounters()

    # ----------------------------------------------------------------
    def process(self, bgr: np.ndarray, t_sec: float) -> Optional[float]:
        """
        Process one frame. Returns the current windowed estimate, or None.
        Also updates self.stats for diagnostics.
        """
        self.stats.frames_total += 1

        frame = bgr
        if self.dist is not None:
            frame = cv2.undistort(bgr, self.K, self.dist)

        # --- Stage 1: detect.
        det = detect_yellow_top(frame, self.K, self.cfg.D_m,
                                self.cfg.level_range_m,
                                axis_ratio_min=self.cfg.axis_ratio_min)
        if det is None:
            self.stats.last_reject_reason = (
                "no yellow blob passed the size/shape/axis-ratio gate"
            )
            return self._current(t_sec)
        self.stats.detect_hits += 1
        self.last_det = det

        # --- Stage 2: solve pose.
        sols = circle_pose_from_conic(det["C_norm"], 0.5 * self.cfg.D_m)
        if len(sols) == 2:
            self.stats.pose_two_sols += 1
        elif len(sols) == 1:
            self.stats.pose_one_sol += 1
        else:
            self.stats.pose_zero_sol += 1
            self.stats.last_reject_reason = "circle_pose_from_conic returned no solution"
            return self._current(t_sec)

        # --- Stage 3: filter by physical plausibility.
        candidates = []
        for (t, n) in sols:
            U, H_top = up_and_height(t, n)
            if t[2] <= 0.0:
                continue
            if H_top <= 0.0:
                continue
            H_water = H_top + self.cfg.h_target_m
            lo, hi = self.cfg.level_range_m
            if not (lo <= H_water <= hi):
                continue
            candidates.append((U, H_top, t, n))

        if not candidates:
            self.stats.last_reject_reason = (
                f"all poses outside level_range [{self.cfg.level_range_m[0]}, "
                f"{self.cfg.level_range_m[1]}] m "
                "(check --fx/--fy and --D)"
            )
            return self._current(t_sec)
        self.stats.plausible_candidates += 1

        # --- Stage 4: disambiguate.
        chosen = self.tracker.update(candidates)
        if chosen is None:
            self.stats.last_reject_reason = "up tracker still warming up"
            return self._current(t_sec)
        self.stats.tracker_locked_in += 1

        # --- Stage 5: push to window.
        _, H_top, _, _ = chosen
        self.samples.append((t_sec, H_top + self.cfg.h_target_m))
        self.stats.samples_pushed += 1
        while self.samples and t_sec - self.samples[0][0] > self.cfg.window_sec:
            self.samples.popleft()

        self.stats.last_reject_reason = "none"
        return self._current(t_sec)

    # ----------------------------------------------------------------
    def _current(self, t_sec: float) -> Optional[float]:
        """Return the current windowed estimate, or None if not enough data."""
        while self.samples and t_sec - self.samples[0][0] > self.cfg.window_sec:
            self.samples.popleft()

        if len(self.samples) < self.cfg.min_valid_frames:
            self.stats.reports_suppressed_warmup += 1
            return None

        vals = np.array([v for _, v in self.samples], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-9
        keep = np.abs(vals - med) < 3.5 * 1.4826 * mad
        if not np.any(keep):
            return None
        return float(np.median(vals[keep]))

    # ----------------------------------------------------------------
    def report_once(self) -> None:
        """Mark that a report was actually emitted (called by _run_video)."""
        self.stats.reports_emitted += 1


# =====================================================================
# SECTION 6 -- Synthetic pose-grid self-test
# =====================================================================

def _project_circle(t_cam, n_cam, r, K, n_pts=240, noise_px=0.0, rng=None):
    n = n_cam / np.linalg.norm(n_cam)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, tmp); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    th = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    P = (t_cam[None, :]
         + r * (np.cos(th)[:, None] * a[None, :]
                + np.sin(th)[:, None] * b[None, :]))
    proj = (K @ P.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    if noise_px > 0 and rng is not None:
        uv = uv + rng.normal(0.0, noise_px, uv.shape)
    return uv


def run_synthetic_test(noise_px: float = 0.3, n_seeds: int = 5) -> None:
    print("=" * 72)
    print(f"Synthetic pose-grid test  (contour noise = {noise_px:.2f} px)")
    print("=" * 72)

    K = np.array([[1200.0, 0.0, 960.0],
                  [0.0, 1200.0, 540.0],
                  [0.0, 0.0, 1.0]])
    r = 0.40

    heights = [5.0, 10.0, 15.0, 20.0, 25.0]
    tilts_deg = [0.0, 5.0, 15.0, 30.0]
    azimuths_deg = [0.0, 90.0, 180.0, 270.0]

    errs = []
    n_ok = 0
    n_fail = 0

    for H in heights:
        for phi_deg in tilts_deg:
            for az_deg in azimuths_deg:
                for seed in range(n_seeds):
                    rng = np.random.default_rng(
                        seed + 1000 * int(H) + int(phi_deg * 10) + int(az_deg))

                    phi = math.radians(phi_deg)
                    az = math.radians(az_deg)
                    n_cam = np.array([
                        math.sin(phi) * math.cos(az),
                        -math.cos(phi),
                        math.sin(phi) * math.sin(az),
                    ])
                    n_cam /= np.linalg.norm(n_cam)

                    t_cam = np.array([0.0, H, 15.0])
                    uv = _project_circle(t_cam, n_cam, r, K,
                                         noise_px=noise_px, rng=rng)

                    C_pix = fit_ellipse_conic(uv)
                    if C_pix is None:
                        n_fail += 1
                        continue
                    C_norm = K.T @ C_pix @ K

                    sols = circle_pose_from_conic(C_norm, r)
                    if not sols:
                        n_fail += 1
                        continue

                    best = None
                    best_dot = -2.0
                    for (t, n) in sols:
                        d = abs(float(n @ n_cam))
                        if d > best_dot:
                            best_dot = d
                            best = (t, n)

                    _, n_est = best
                    U, H_top = up_and_height(*best)
                    err = H_top - H

                    if best_dot < 0.9:
                        n_fail += 1
                        continue

                    errs.append(err)
                    n_ok += 1

    errs = np.asarray(errs)
    if errs.size == 0:
        print("No successful reconstructions.")
        return

    abs_e = np.abs(errs)
    print(f"  successful   : {n_ok}")
    print(f"  failed       : {n_fail}")
    print(f"  |err| mean   : {abs_e.mean() * 1000:.2f} mm")
    print(f"  |err| median : {np.median(abs_e) * 1000:.2f} mm")
    print(f"  |err| p95    : {np.percentile(abs_e, 95) * 1000:.2f} mm")
    print(f"  |err| max    : {abs_e.max() * 1000:.2f} mm")
    print(f"  bias         : {errs.mean() * 1000:+.2f} mm")
    print("=" * 72)


# =====================================================================
# SECTION 7 -- Command-line interface
# =====================================================================

def _build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "Measure lake water level from a floating circular target. "
            "Run with no arguments to execute the synthetic self-test."
        )
    )

    p.add_argument("--video", type=str, default=None,
                   help="Input video. If omitted, the synthetic self-test runs.")
    p.add_argument("--annotated", type=str, default=None,
                   help="Optional output video with ellipse and level overlay.")
    p.add_argument("--fx", type=float, default=1200.0,
                   help="Focal length in pixels along x (REQUIRED to be correct).")
    p.add_argument("--fy", type=float, default=1200.0,
                   help="Focal length in pixels along y (usually == fx).")
    p.add_argument("--cx", type=float, default=960.0,
                   help="Principal point x (pixels).")
    p.add_argument("--cy", type=float, default=540.0,
                   help="Principal point y (pixels).")
    p.add_argument("--dist", type=float, nargs="*", default=None,
                   help="Distortion coefficients k1 k2 p1 p2 [k3].")
    p.add_argument("--D", type=float, default=0.8,
                   help="Diameter of the yellow TOP FACE (metres).")
    p.add_argument("--h_target", type=float, default=0.20,
                   help="Height of the top face above the waterline (metres).")
    p.add_argument("--window", type=float, default=8.0,
                   help="Sliding median window length (seconds).")
    p.add_argument("--level_min", type=float, default=0.0,
                   help="Minimum plausible vertical drop below the camera (m).")
    p.add_argument("--level_max", type=float, default=60.0,
                   help="Maximum plausible vertical drop below the camera (m).")
    p.add_argument("--progress", type=int, default=0,
                   help="If > 0, print a progress line every N frames.")
    return p


def _run_video(args):
    """
    Process a video file end-to-end and print a stage-counter summary.
    """
    K = np.array([[args.fx, 0.0, args.cx],
                  [0.0, args.fy, args.cy],
                  [0.0, 0.0, 1.0]])
    dist = None if args.dist is None else np.asarray(args.dist, np.float64)

    cfg = EstimatorConfig(
        D_m=args.D,
        h_target_m=args.h_target,
        window_sec=args.window,
        level_range_m=(args.level_min, args.level_max),
    )
    est = WaterLevelEstimator(K, dist, cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 72)
    print("Lake level measurement")
    print("=" * 72)
    print(f"  video          : {args.video}")
    print(f"  resolution/fps : {W}x{H} @ {fps:.2f} fps"
          + (f"  ({total_frames} frames)" if total_frames > 0 else ""))
    print(f"  intrinsics     : fx={args.fx:.2f} fy={args.fy:.2f} "
          f"cx={args.cx:.2f} cy={args.cy:.2f}")
    if dist is not None:
        print(f"  distortion     : {list(dist)}")
    print(f"  target D       : {args.D:.3f} m")
    print(f"  h_target       : {args.h_target:.3f} m")
    print(f"  level range    : {args.level_min:.1f} .. {args.level_max:.1f} m")
    print(f"  window         : {args.window:.1f} s")
    print("-" * 72)

    writer = None
    if args.annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotated, fourcc, fps, (W, H))

    i = 0
    last_reported = None
    next_report_t = 0.0
    report_interval = max(5.0, min(10.0, args.window))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = i / fps
        h = est.process(frame, t_sec)

        if h is not None and t_sec >= next_report_t:
            print(f"[{t_sec:8.2f}s] water level = {h:.3f} m")
            next_report_t = t_sec + report_interval
            last_reported = h
            est.report_once()

        if args.progress > 0 and i % args.progress == 0 and i > 0:
            s = est.stats
            print(f"  ... frame {i:6d}  "
                  f"detect={s.detect_hits}  "
                  f"sols2={s.pose_two_sols}  "
                  f"plausible={s.plausible_candidates}  "
                  f"tracked={s.tracker_locked_in}  "
                  f"samples={s.samples_pushed}")

        if writer is not None:
            det = est.last_det
            if det is not None:
                cv2.ellipse(
                    frame,
                    (int(det["center_px"][0]), int(det["center_px"][1])),
                    (int(det["major_px"] / 2), int(det["minor_px"] / 2)),
                    det["angle_deg"], 0, 360, (0, 0, 255), 2,
                )
            if last_reported is not None:
                cv2.putText(
                    frame, f"level: {last_reported:.3f} m",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 2, cv2.LINE_AA,
                )
            writer.write(frame)

        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    print("-" * 72)
    print(f"Processed {i} frames ({i/fps:.1f} s).")
    if last_reported is not None:
        print(f"Last reported level: {last_reported:.3f} m")
    else:
        print("No water-level measurement was produced.")
        print()
        print("Stage-counter summary:")
        print(est.stats.summary())
        print()
        print(_explain_no_output(est.stats))
    print("=" * 72)


def _explain_no_output(s: StageCounters) -> str:
    """
    Return a targeted hint based on which stage rejected the data.
    """
    if s.frames_total == 0:
        return "  hint: the video contains zero frames."

    if s.detect_hits == 0:
        return (
            "  hint: the yellow detector never fired.\n"
            "        - check that the target really is yellow (H 18-42, S>80, V>80)\n"
            "        - check that the target is inside the frame\n"
            "        - check that the blob is not too small: with your fx and D,\n"
            "          the minimum pixel area is derived from --level_max.\n"
            "          Lowering --level_max makes the gate stricter, not looser."
        )

    if s.pose_zero_sol == s.detect_hits:
        return (
            "  hint: the detector found a blob but the pose solver rejected\n"
            "        every conic. Usually the ellipse is too flat (axis ratio\n"
            "        below --axis-ratio-min, i.e. the top face is heavily\n"
            "        tilted or partly occluded)."
        )

    if s.plausible_candidates == 0:
        return (
            "  hint: poses were computed but none satisfied --level_min/--level_max.\n"
            "        This is the classic sign that --fx / --fy are wrong.\n"
            "        Try: fx = W / (2 * tan(HFOV/2)).\n"
            "        For 60 deg HFoV and W=1920: fx ~ 1663.\n"
            "        Also verify --D matches the physical target diameter."
        )

    if s.tracker_locked_in == 0:
        return (
            "  hint: candidates existed but the up-direction tracker never\n"
            "        produced a solution. This means the two-fold ambiguity\n"
            "        could not be resolved within the warmup period, which\n"
            "        usually means the two branches had similar dispersion.\n"
            "        Narrowing --level_min/--level_max to the true height\n"
            "        range helps a lot."
        )

    if s.samples_pushed < s.cfg_min_frames_placeholder if hasattr(s, "cfg_min_frames_placeholder") else False:
        pass

    if s.reports_emitted == 0:
        return (
            "  hint: samples were pushed but no report was emitted.\n"
            f"        samples pushed = {s.samples_pushed}, "
            f"reports suppressed during warmup = {s.reports_suppressed_warmup}.\n"
            "        If samples_pushed is small, the target was visible only\n"
            "        briefly. Increase the video length or lower --window."
        )

    return "  hint: measurements were emitted; check the lines above."


def main():
    args = _build_argparser().parse_args()
    if args.video is None:
        run_synthetic_test()
    else:
        _run_video(args)


if __name__ == "__main__":
    main()