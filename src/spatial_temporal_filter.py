

import math
import statistics
from collections import namedtuple
from typing import Dict, List, Optional, Tuple


# Bone-Length Constancy result
BLCResult = namedtuple('BLCResult', ['flagged', 'ratio', 'length', 'baseline'])


class SpatialTemporalFilter:
    """Spatial-Temporal Filtering: CW-EMA smoothing + Bone-Length Constancy (BLC) + Unscorable Rep Logic.

    Smooths MediaPipe BlazePose landmark coordinates using per-landmark
    stateful EMA whose effective alpha adapts to the landmark's confidence
    score each frame. Optionally applies BLC geometric reliability checks
    and tracks per-repetition flagged-frame counts for unscorable detection.

    Usage
    -----
    >>> stf = SpatialTemporalFilter(alpha_base=0.5, c_floor=0.05, blc_beta=0.7, unscorable_frame_pct=0.20)
    >>> stf.set_blc_baseline(150.0)  # Set after calibration
    >>> filtered = stf.filter_landmarks(raw_landmarks_dict)
    >>> blc_result = stf.check_bone_length(arm_landmarks)
    >>> if stf.is_rep_unscorable:
    ...     rom_label = 'unscorable'
    >>> stf.reset_blc_rep_counters()  # Reset per-rep counts before next rep

    Parameters
    ----------
    alpha_base : float, default 0.4
        Base EMA smoothing factor.  Higher → faster response (more noise).
        Lower → heavier smoothing (more lag).  Must be in (0, 1].
    c_floor : float, default 0.05
        Minimum confidence value. Prevents α_eff from reaching zero when
        MediaPipe reports visibility = 0.0.  Must be in (0, 1].
    blc_beta : float, default 0.7
        Bone-Length Constancy threshold. Frame is flagged if L_t/L_base < β.
        Must be in (0, 1).
    unscorable_frame_pct : float, default 0.20
        Repetition is marked unscorable if > this fraction of frames are BLC-flagged.
        Must be in (0, 1).

    Attributes
    ----------
    _states : dict
        Per-landmark EMA state: {landmark_name: (x̂, ŷ)}.
    _blc_baseline : float or None
        Baseline bone-length for BLC ratio computation.
    _blc_flagged_count : int
        Count of BLC-flagged frames in current repetition.
    _blc_total_count : int
        Total frames checked in current repetition.
    """

    # ------------------------------------------------------------------
    # Default parameters (can be overridden via config.json / constructor)
    # ------------------------------------------------------------------
    DEFAULT_ALPHA_BASE: float = 0.4
    DEFAULT_C_FLOOR: float = 0.05
    DEFAULT_BLC_BETA: float = 0.7
    DEFAULT_UNSCORABLE_FRAME_PCT: float = 0.20

    def __init__(
        self,
        alpha_base: Optional[float] = None,
        c_floor: Optional[float] = None,
        blc_beta: Optional[float] = None,
        unscorable_frame_pct: Optional[float] = None,
        blc_baseline_frames: Optional[int] = None,
    ) -> None:
        # Validate and store alpha_base
        self.alpha_base = self._validate_alpha_base(
            alpha_base if alpha_base is not None else self.DEFAULT_ALPHA_BASE
        )
        # Validate and store c_floor
        self.c_floor = self._validate_c_floor(
            c_floor if c_floor is not None else self.DEFAULT_C_FLOOR
        )
        # Validate and store BLC parameters
        self.blc_beta = self._validate_blc_beta(
            blc_beta if blc_beta is not None else self.DEFAULT_BLC_BETA
        )
        self.unscorable_frame_pct = self._validate_unscorable_frame_pct(
            unscorable_frame_pct if unscorable_frame_pct is not None else self.DEFAULT_UNSCORABLE_FRAME_PCT
        )
        # Validate and store auto-baseline parameter
        if blc_baseline_frames is not None:
            self._validate_blc_baseline_frames(blc_baseline_frames)
        self.blc_baseline_frames: Optional[int] = blc_baseline_frames

        # Per-landmark filter state: {landmark_name: (x̂, ŷ)}
        self._states: Dict[str, Tuple[float, float]] = {}
        
        # BLC state (T13: Unscorable repetition logic)
        self._blc_baseline: Optional[float] = None  # Baseline bone-length
        self._blc_flagged_count: int = 0  # Flagged frames in current rep
        self._blc_total_count: int = 0  # Total frames checked in current rep

        # Auto-baseline sample accumulator
        self._blc_baseline_samples: List[float] = []

    # ==================================================================
    # Public API
    # ==================================================================

    def filter_landmarks(
        self,
        landmarks_dict: Dict[str, Tuple[float, float, float]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """Apply CW-EMA smoothing to all landmarks in a frame.

        Parameters
        ----------
        landmarks_dict : dict
            Output of ``PoseProcessor.process_frame()``.
            ``{name: (x, y, visibility)}`` where *visibility* ∈ [0, 1].
            An **empty dict** signals a detection miss — the filter holds
            its previous state and returns an empty dict (no hallucination).

        Returns
        -------
        dict
            Same structure as input, with (x, y) replaced by smoothed
            values and visibility passed through unchanged.  Landmarks
            that fail validation are omitted from the output.
        """
        if not landmarks_dict:
            # Detection miss — return empty, preserve state for next frame
            return {}

        filtered: Dict[str, Tuple[float, float, float]] = {}

        for name, values in landmarks_dict.items():
            result = self._apply_cwema(name, values)
            if result is not None:
                filtered[name] = result

        return filtered

    def reset(self) -> None:
        """Clear all per-landmark filter state and BLC state.

        Call this when starting a new session/video or when the subject
        changes, so that stale state from a prior sequence does not
        contaminate the new one.
        """
        self._states.clear()
        self.reset_blc()  # Also clear BLC baseline and counters

    def reset_landmark(self, landmark_name: str) -> None:
        """Clear filter state for a single landmark.

        Parameters
        ----------
        landmark_name : str
            Key as used in the landmarks dict (e.g. ``'TRACKED_SHOULDER'``).
        """
        self._states.pop(landmark_name, None)

    @property
    def active_landmarks(self) -> int:
        """Number of landmarks with initialised filter state."""
        return len(self._states)

    def get_state(self, landmark_name: str) -> Optional[Tuple[float, float]]:
        """Return current filtered (x̂, ŷ) for a landmark, or None."""
        return self._states.get(landmark_name)

    # ==================================================================
    # CW-EMA core
    # ==================================================================

    def _apply_cwema(
        self,
        name: str,
        values: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Apply CW-EMA to a single landmark.

        Parameters
        ----------
        name : str
            Landmark identifier (e.g. ``'TRACKED_SHOULDER'``).
        values : tuple (x, y, visibility)
            Raw coordinates and confidence from MediaPipe.

        Returns
        -------
        tuple (x_filtered, y_filtered, visibility) or None
            ``None`` if the input is invalid (NaN coords, etc.).
        """
        # --- Input validation -------------------------------------------
        if not self._is_valid_landmark(values):
            return None

        x_raw, y_raw, confidence = values[0], values[1], values[2]

        # Clamp confidence to [0, 1] defensively
        confidence = max(0.0, min(1.0, confidence))

        # --- Effective alpha (Eq. 5) ------------------------------------
        alpha_eff = self.alpha_base * max(confidence, self.c_floor)

        # --- State initialisation (first valid observation) -------------
        if name not in self._states:
            # No prior estimate → initialise to raw measurement
            self._states[name] = (x_raw, y_raw)
            return (x_raw, y_raw, confidence)

        # --- EMA update (Eq. 6) -----------------------------------------
        x_prev, y_prev = self._states[name]
        x_filtered = alpha_eff * x_raw + (1.0 - alpha_eff) * x_prev
        y_filtered = alpha_eff * y_raw + (1.0 - alpha_eff) * y_prev

        # Store updated state
        self._states[name] = (x_filtered, y_filtered)

        return (x_filtered, y_filtered, confidence)

    # ==================================================================
    # Validation helpers
    # ==================================================================

    @staticmethod
    def _is_valid_landmark(
        values: Tuple[float, float, float],
    ) -> bool:
        """Check that a landmark tuple is structurally valid.

        A landmark is invalid if:
          - It is not a 3-element sequence
          - Any coordinate is NaN or Inf
        """
        try:
            if len(values) < 3:
                return False
        except TypeError:
            return False

        x, y, vis = values[0], values[1], values[2]
        # Check for NaN / Inf in coordinates
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        # Visibility can be 0.0 — that's valid (c_floor handles it)
        if not math.isfinite(vis):
            return False
        return True

    @staticmethod
    def _validate_alpha_base(value: float) -> float:
        """Validate α_base ∈ (0, 1]."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"alpha_base must be a number, got {type(value).__name__}"
            )
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"alpha_base must be in (0, 1], got {value}"
            )
        return float(value)

    @staticmethod
    def _validate_c_floor(value: float) -> float:
        """Validate c_floor ∈ (0, 1]."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"c_floor must be a number, got {type(value).__name__}"
            )
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"c_floor must be in (0, 1], got {value}"
            )
        return float(value)

    @staticmethod
    def _validate_blc_beta(value: float) -> float:
        """Validate β ∈ (0, 1) for Bone-Length Constancy."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"blc_beta must be a number, got {type(value).__name__}"
            )
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"blc_beta must be in (0, 1], got {value}"
            )
        return float(value)

    @staticmethod
    def _validate_blc_baseline_frames(value) -> int:
        """Validate blc_baseline_frames: must be a positive integer."""
        if not isinstance(value, int):
            raise TypeError(
                f"blc_baseline_frames must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(
                f"blc_baseline_frames must be > 0, got {value}"
            )
        return value

    @staticmethod
    def _validate_unscorable_frame_pct(value: float) -> float:
        """Validate unscorable_frame_pct ∈ (0, 1)."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"unscorable_frame_pct must be a number, got {type(value).__name__}"
            )
        if not (0.0 < value < 1.0):
            raise ValueError(
                f"unscorable_frame_pct must be in (0, 1), got {value}"
            )
        return float(value)

    # ==================================================================
    # Repr / debugging
    # ==================================================================

    # ==================================================================
    # Bone-Length Constancy (BLC) Methods (T13)
    # ==================================================================

    def set_blc_baseline(self, baseline: float) -> None:
        """Set the baseline bone-length for BLC ratio computation.
        
        Typically called after calibration/tracking readiness check.
        Also clears any pending auto-baseline samples.
        
        Args:
            baseline: Baseline segment length (pixels or normalized coords).
        """
        if not isinstance(baseline, (int, float)):
            raise TypeError(f"baseline must be a number, got {type(baseline).__name__}")
        if not math.isfinite(baseline):
            raise ValueError(f"baseline must be finite, got {baseline}")
        if baseline <= 0:
            raise ValueError(f"baseline must be > 0, got {baseline}")
        self._blc_baseline = float(baseline)
        self._blc_baseline_samples.clear()

    def get_blc_baseline(self) -> Optional[float]:
        """Return current BLC baseline, or None if not set."""
        return self._blc_baseline

    @property
    def blc_baseline_ready(self) -> bool:
        """Whether BLC baseline has been set and is ready for checking."""
        return self._blc_baseline is not None

    @property
    def blc_baseline_progress(self) -> float:
        """Fraction of auto-baseline frames collected so far.

        Returns 0.0 if auto-baseline is disabled or already complete.
        """
        if self.blc_baseline_frames is None or self.blc_baseline_frames == 0:
            return 0.0
        if self._blc_baseline is not None:
            return 1.0
        return len(self._blc_baseline_samples) / self.blc_baseline_frames

    def check_bone_length(
        self,
        endpoints_dict: Dict[str, Tuple[float, float, float]],
        segment_indices: Tuple[str, str] = ('TRACKED_SHOULDER', 'TRACKED_ELBOW'),
    ) -> Optional[BLCResult]:
        """Check Bone-Length Constancy for a segment.
        
        Compares current segment length to baseline. If L_t / L_base < β,
        flags the frame as unreliable. Accumulates counts for per-rep
        unscorable detection.

        When auto-baseline is enabled (blc_baseline_frames is set) and no
        explicit baseline exists yet, this method collects valid segment
        lengths. Once enough samples are gathered, the median is used as
        the baseline. During collection, returns BLCResult(flagged=False,
        ratio=NaN, baseline=NaN) (benefit of the doubt).
        
        Args:
            endpoints_dict: Landmarks dict with at least the two segment endpoints.
            segment_indices: Tuple of two landmark names defining the segment.
                Defaults to ('TRACKED_SHOULDER', 'TRACKED_ELBOW').
        
        Returns:
            BLCResult(flagged, ratio, length, baseline) or None if baseline not set
            (and auto-baseline disabled) or endpoints unavailable.
        """
        p1_name, p2_name = segment_indices
        
        # Check endpoints exist and are visible
        if p1_name not in endpoints_dict or p2_name not in endpoints_dict:
            return None
        
        x1, y1, vis1 = endpoints_dict[p1_name]
        x2, y2, vis2 = endpoints_dict[p2_name]
        
        # Validate coordinates are finite (NaN/Inf → skip)
        if not (math.isfinite(x1) and math.isfinite(y1) and
                math.isfinite(x2) and math.isfinite(y2)):
            return None
        
        if vis1 < 0.5 or vis2 < 0.5:
            return None
        
        # Compute segment length
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # --- Auto-baseline collection ---
        if self._blc_baseline is None:
            if self.blc_baseline_frames is not None:
                self._blc_baseline_samples.append(length)
                if len(self._blc_baseline_samples) >= self.blc_baseline_frames:
                    # Enough samples — set baseline to median (outlier-robust)
                    self._blc_baseline = statistics.median(self._blc_baseline_samples)
                    self._blc_baseline_samples.clear()
                    # Fall through to normal BLC check below
                else:
                    # Still collecting — benefit of the doubt
                    return BLCResult(
                        flagged=False,
                        ratio=float('nan'),
                        length=length,
                        baseline=float('nan'),
                    )
            else:
                # No baseline set, auto-baseline disabled
                return None

        ratio = length / self._blc_baseline if self._blc_baseline > 0 else 0.0
        
        # Flag if foreshortened (ratio < β)
        flagged = ratio < self.blc_beta
        
        # Accumulate for unscorable logic
        self._blc_total_count += 1
        if flagged:
            self._blc_flagged_count += 1
        
        return BLCResult(flagged=flagged, ratio=ratio, length=length, baseline=self._blc_baseline)

    @property
    def blc_flagged_ratio(self) -> float:
        """Proportion of BLC-flagged frames in current repetition.
        
        Returns 0.0 if no frames checked yet.
        """
        if self._blc_total_count == 0:
            return 0.0
        return self._blc_flagged_count / self._blc_total_count

    @property
    def is_rep_unscorable(self) -> bool:
        """Whether current repetition should be marked unscorable.
        
        True if blc_flagged_ratio > unscorable_frame_pct.
        Uses strict > (not >=) to allow edge cases at the threshold.
        """
        return self.blc_flagged_ratio > self.unscorable_frame_pct

    def reset_blc_rep_counters(self) -> None:
        """Reset per-repetition BLC counters.
        
        Call after each repetition completes to reset frame counts
        for the next rep. Preserves baseline.
        """
        self._blc_flagged_count = 0
        self._blc_total_count = 0

    def reset_blc(self) -> None:
        """Full reset of BLC state including baseline and auto-baseline samples.
        
        Call when starting a new session or when recalibration is needed.
        """
        self._blc_baseline = None
        self._blc_baseline_samples.clear()
        self.reset_blc_rep_counters()

    # ==================================================================
    # Repr / debugging
    # ==================================================================

    def __repr__(self) -> str:
        baseline_str = f"baseline={self._blc_baseline}" if self._blc_baseline is not None else "baseline=None"
        return (
            f"SpatialTemporalFilter("
            f"alpha_base={self.alpha_base}, "
            f"c_floor={self.c_floor}, "
            f"blc_beta={self.blc_beta}, "
            f"active_landmarks={self.active_landmarks}, "
            f"{baseline_str})"
        )
