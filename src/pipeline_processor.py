"""
Three-Pipeline Processor for Frozen Shoulder Rehabilitation
===========================================================

This module supports the thesis requirement (§3.3) for within-subject
three-pipeline comparison evaluation: Raw, Simple Moving Average (SMA),
and Confidence-Weighted Exponential Moving Average (CW-EMA).

Purpose:
    All three pipelines process the same frame input simultaneously and
    produce aligned outputs for later statistical comparison against
    physiotherapist annotations and ablation analysis.

Data Flow:
    landmarks (from PoseProcessor.process_frame)
    ↓
    PipelineProcessor.process_frame()
    ├─ Raw: unchanged
    ├─ SMA: simple moving average per landmark
    ├─ CW-EMA: confidence-weighted EMA (via SpatialTemporalFilter)
    ↓
    {
        "raw": {...landmarks...},
        "sma": {...landmarks...},
        "cwema": {...landmarks...},
        "metadata": {...}
    }

Filtering Ablation Modes (§3.3.2):
    The CW-EMA pipeline supports four ablation modes to isolate the
    contribution of individual filtering components:

    - "full" (default):      CW-EMA + BLC (both active)
    - "cwema_only":          CW-EMA active, BLC inactive
    - "blc_only":            BLC active, CW-EMA inactive (placeholder)
    - "none":                Neither CW-EMA nor BLC (raw)

    The "cwema" key in output contains the result of the selected mode.
    The active mode is recorded in metadata.

Thesis Alignment Notes:
    - The three-pipeline design is specified in §3 Methodology
    - CW-EMA reuses SpatialTemporalFilter (§3.2.1)
    - SMA is a lightweight comparative baseline
    - Raw provides ground truth for comparison
    - All three receive identical input for fair evaluation
    - Output structure supports later ablation studies (§3.3.2)
    - Ablation modes are separated from geometric ablations
"""

from typing import Dict, Tuple, Optional, List
import math
import csv
import os
import time

try:
    from src.spatial_temporal_filter import SpatialTemporalFilter
except ImportError:
    from spatial_temporal_filter import SpatialTemporalFilter

# Filtering Ablation Mode Constants
ABLATION_FULL = "full"              # CW-EMA + BLC
ABLATION_CWEMA_ONLY = "cwema_only"  # CW-EMA only
ABLATION_BLC_ONLY = "blc_only"      # BLC only (placeholder)
ABLATION_NONE = "none"              # Neither

ABLATION_MODES = {ABLATION_FULL, ABLATION_CWEMA_ONLY, ABLATION_BLC_ONLY, ABLATION_NONE}

# Pipeline identifier constants (for per-pipeline CSV output)
PIPELINE_RAW = "raw"
PIPELINE_SMA = "sma"
PIPELINE_CWEMA = "cwema"

PIPELINE_NAMES = {PIPELINE_RAW, PIPELINE_SMA, PIPELINE_CWEMA}


class PipelineProcessor:
    """
    Three-pipeline landmark processor for filtering method comparison.

    Processes a single frame through three parallel pipelines:
    - **Raw**: unfiltered landmarks (ground truth)
    - **SMA**: Simple Moving Average smoothing (lightweight baseline)
    - **CW-EMA**: Confidence-Weighted EMA (thesis contribution)

    All pipelines receive the same input and produce aligned outputs
    for downstream angle computation and agreement analysis with
    physiotherapist annotations.

    Parameters
    ----------
    sma_window_size : int, default 5
        Buffer size for Simple Moving Average.
        Must be >= 1. Higher values → heavier smoothing, more lag.
        Typical range: 3-7 frames at 30 FPS = 100-233 ms lag.

    alpha_base : float, optional
        Base EMA smoothing factor for CW-EMA (0, 1].
        Defaults to SpatialTemporalFilter.DEFAULT_ALPHA_BASE.

    c_floor : float, optional
        Confidence floor for CW-EMA (0, 1].
        Defaults to SpatialTemporalFilter.DEFAULT_C_FLOOR.

    ablation_mode : str, default "full"
        Filtering ablation mode (§3.3.2):
        - "full": CW-EMA + BLC (both active)
        - "cwema_only": CW-EMA active, BLC inactive
        - "blc_only": BLC active, CW-EMA inactive (placeholder)
        - "none": Neither (raw)

    Attributes
    ----------
    _sma_buffers : dict
        Per-landmark history for SMA: {name: [(x, y, vis), ...]}
        Cleared on detection miss or reset().

    cwema_filter : SpatialTemporalFilter
        Reusable confidence-weighted EMA instance.

    ablation_mode : str
        Current filtering ablation mode.
    """

    def __init__(
        self,
        sma_window_size: int = 5,
        alpha_base: Optional[float] = None,
        c_floor: Optional[float] = None,
        ablation_mode: str = ABLATION_FULL,
    ) -> None:
        """
        Initialize three-pipeline processor with optional ablation mode.

        Raises
        ------
        ValueError
            If sma_window_size < 1 or ablation_mode not recognized
        """
        if sma_window_size < 1:
            raise ValueError(
                f"sma_window_size must be >= 1, got {sma_window_size}"
            )
        
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(
                f"ablation_mode must be one of {ABLATION_MODES}, "
                f"got '{ablation_mode}'"
            )
        
        self.sma_window_size = sma_window_size
        self.ablation_mode = ablation_mode

        # Per-landmark SMA buffers: {name: [(x, y, vis), ...]}
        self._sma_buffers: Dict[str, List[Tuple[float, float, float]]] = {}

        # CW-EMA filter (reuse existing implementation)
        self.cwema_filter = SpatialTemporalFilter(
            alpha_base=alpha_base,
            c_floor=c_floor,
        )

        # Per-pipeline rep-level row accumulation for CSV export (§16: per-pipeline output)
        # Structure: {pipeline_name: [row_dict, row_dict, ...]}
        # Used for post-session analysis and ablation evaluation
        self._row_accumulator: Dict[str, List[Dict]] = {
            PIPELINE_RAW: [],
            PIPELINE_SMA: [],
            PIPELINE_CWEMA: [],
        }

        # Per-pipeline timing accumulators for FPS/latency reporting
        # Each list stores elapsed seconds per process_frame call
        self._timing: Dict[str, List[float]] = {
            PIPELINE_RAW: [],
            PIPELINE_SMA: [],
            PIPELINE_CWEMA: [],
        }

        # Per-pipeline angle-stream jitter tracking
        # Stores previous angle and list of frame-to-frame absolute deltas
        self._jitter: Dict[str, Dict] = {
            PIPELINE_RAW: {"prev": None, "deltas": []},
            PIPELINE_SMA: {"prev": None, "deltas": []},
            PIPELINE_CWEMA: {"prev": None, "deltas": []},
        }

    def process_frame(
        self,
        landmarks: Optional[Dict[str, Tuple[float, float, float]]],
    ) -> Dict:
        """
        Process a single frame through all three pipelines in parallel.

        Each pipeline receives the same input landmarks and produces
        its own output according to its smoothing policy.

        Parameters
        ----------
        landmarks : dict or None
            Landmarks from PoseProcessor.process_frame():
            {name: (x, y, visibility)}
            where visibility ∈ [0, 1].

            Empty dict or None indicates a detection miss.
            Each pipeline handles this gracefully:
            - Raw: returns empty dict
            - SMA: clears buffers, returns empty dict
            - CW-EMA: preserves state (per design), returns empty dict

        Returns
        -------
        dict
            Three-pipeline outputs with structure:

            {
                "raw": {...landmarks...} or {},
                "sma": {...landmarks...} or {},
                "cwema": {...landmarks...} or {},
                "metadata": {
                    "sma_window_size": int,
                    "sma_active_landmarks": int,
                    "cwema_active_landmarks": int,
                    "ablation_mode": str,
                }
            }

            Each pipeline value is a dict with same structure as input:
            {name: (x_processed, y_processed, visibility)}

            Empty dict means:
            - No landmarks detected this frame, or
            - All landmarks failed validation

        Notes
        -----
        All three pipelines process identical input, enabling fair
        comparison. Visibility values are passed through unchanged in
        all pipelines (not smoothed).
        """
        # Handle detection miss or None input
        if not landmarks:
            landmarks = {}

        # Pipeline 1: Raw (no filtering) — timed
        t0 = time.perf_counter()
        raw_output = dict(landmarks) if landmarks else {}
        raw_elapsed = time.perf_counter() - t0
        self._timing[PIPELINE_RAW].append(raw_elapsed)

        # Pipeline 2: SMA (simple moving average) — timed
        t0 = time.perf_counter()
        sma_output = self._apply_sma(landmarks)
        sma_elapsed = time.perf_counter() - t0
        self._timing[PIPELINE_SMA].append(sma_elapsed)

        # Pipeline 3: CW-EMA (confidence-weighted exponential) with ablation mode — timed
        t0 = time.perf_counter()
        cwema_output = self._apply_cwema_with_mode(landmarks)
        cwema_elapsed = time.perf_counter() - t0
        self._timing[PIPELINE_CWEMA].append(cwema_elapsed)

        # Metadata for diagnostic purposes (not used in downstream)
        metadata = {
            "sma_window_size": self.sma_window_size,
            "sma_active_landmarks": len(self._sma_buffers),
            "cwema_active_landmarks": self.cwema_filter.active_landmarks,
            "ablation_mode": self.ablation_mode,
            "latency_ms": {
                PIPELINE_RAW: round(raw_elapsed * 1000, 4),
                PIPELINE_SMA: round(sma_elapsed * 1000, 4),
                PIPELINE_CWEMA: round(cwema_elapsed * 1000, 4),
            },
        }

        return {
            "raw": raw_output,
            "sma": sma_output,
            "cwema": cwema_output,
            "metadata": metadata,
        }

    def _apply_cwema_with_mode(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Apply CW-EMA filtering with active ablation mode.

        Routes filtering based on the current ablation_mode setting,
        allowing isolation of individual filtering components.

        Parameters
        ----------
        landmarks : dict
            Current frame landmarks {name: (x, y, vis)}

        Returns
        -------
        dict
            Landmarks filtered according to active ablation mode.

        Ablation Modes
        --------------
        - "full":       Apply CW-EMA + BLC (both active)
        - "cwema_only": Apply CW-EMA, BLC inactive
        - "blc_only":   Apply BLC only, CW-EMA inactive (placeholder)
        - "none":       Return unfiltered (equivalent to Raw)

        Notes
        -----
        The "cwema" pipeline key receives output from this method
        regardless of mode. Mode is tracked in metadata for ablation
        analysis and later comparison against physiotherapist annotations.

        BLC (Bone-Length Constancy) is currently a placeholder until
        implemented in SpatialTemporalFilter. "blc_only" and "full"
        modes will activate BLC filtering when available.
        """
        if not landmarks:
            # Detection miss: return empty dict
            return {}

        # Apply filtering based on ablation mode
        if self.ablation_mode == ABLATION_FULL:
            # Full: CW-EMA + BLC
            # Currently applies CW-EMA only (BLC to be added)
            return self.cwema_filter.filter_landmarks(landmarks)
        
        elif self.ablation_mode == ABLATION_CWEMA_ONLY:
            # CW-EMA active, BLC inactive
            return self.cwema_filter.filter_landmarks(landmarks)
        
        elif self.ablation_mode == ABLATION_BLC_ONLY:
            # BLC active, CW-EMA inactive
            # Placeholder: return raw landmarks
            # Future: apply BLC validation without CW-EMA smoothing
            return dict(landmarks)
        
        elif self.ablation_mode == ABLATION_NONE:
            # Neither CW-EMA nor BLC: return raw
            return dict(landmarks)
        
        # Fallback (should not reach here due to validation in __init__)
        return dict(landmarks)

    def _apply_sma(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Apply Simple Moving Average smoothing to landmarks.

        Per-landmark buffer-based averaging:
        x̂_sma = mean(x_1, ..., x_n) for n ≤ window_size

        Parameters
        ----------
        landmarks : dict
            Current frame landmarks {name: (x, y, vis)}

        Returns
        -------
        dict
            Landmarks with (x, y) coordinates replaced by window average.
            Visibility passed through unchanged.

        Notes
        -----
        - On detection miss (empty input), all buffers are cleared
        - On landmark occlusion (landmark absent from current frame),
          that landmark's buffer is cleared
        - Per-landmark buffer grows until window_size, then maintains
          size with FIFO eviction of oldest measurement
        - This is a stateless comparison point (unlike CW-EMA which
          adapts to confidence)
        """
        if not landmarks:
            # Detection miss: clear all buffers and return empty
            self._sma_buffers.clear()
            return {}

        sma_output = {}

        # Remove landmarks no longer present (e.g., occlusion)
        for name in list(self._sma_buffers.keys()):
            if name not in landmarks:
                self._sma_buffers.pop(name)

        # Process each landmark in current frame
        for name, (x, y, vis) in landmarks.items():
            # Validation: skip invalid coordinates
            if not self._is_valid_value(x) or not self._is_valid_value(y):
                continue

            # Initialize or retrieve buffer for this landmark
            if name not in self._sma_buffers:
                self._sma_buffers[name] = []

            # Add current measurement to buffer
            self._sma_buffers[name].append((x, y, vis))

            # Maintain window size (FIFO eviction)
            if len(self._sma_buffers[name]) > self.sma_window_size:
                self._sma_buffers[name].pop(0)

            # Compute moving average
            buffer = self._sma_buffers[name]
            avg_x = sum(v[0] for v in buffer) / len(buffer)
            avg_y = sum(v[1] for v in buffer) / len(buffer)
            # Use current visibility (not averaged across buffer)
            avg_vis = vis

            sma_output[name] = (avg_x, avg_y, avg_vis)

        return sma_output

    @staticmethod
    def _is_valid_value(val: float) -> bool:
        """
        Check if a coordinate is finite (not NaN or Inf).

        Parameters
        ----------
        val : float
            Coordinate value to validate

        Returns
        -------
        bool
            True if val is a finite number, False otherwise
        """
        try:
            # Check type and finiteness
            f = float(val)
            return math.isfinite(f)
        except (TypeError, ValueError):
            return False

    def reset(self) -> None:
        """
        Reset all pipeline state (both SMA and CW-EMA).

        Call this when starting a new session or video to ensure
        no stale history from the previous sequence contaminates
        the new one.

        This clears:
        - SMA per-landmark buffers
        - CW-EMA per-landmark state
        """
        self._sma_buffers.clear()
        self.cwema_filter.reset()
        self.reset_timing()
        self.reset_jitter()

    def reset_sma(self) -> None:
        """
        Reset SMA buffers only (for ablation studies).

        Preserves CW-EMA state. Used when isolating the contribution
        of CW-EMA filtering by comparing CW-EMA-only vs. Raw.
        """
        self._sma_buffers.clear()

    def reset_cwema(self) -> None:
        """
        Reset CW-EMA filter state only (for ablation studies).

        Preserves SMA buffers. Used when isolating the contribution
        of SMA filtering by comparing SMA-only vs. Raw.
        """
        self.cwema_filter.reset()

    def set_ablation_mode(self, mode: str) -> None:
        """
        Set the filtering ablation mode (for ablation studies).

        Parameters
        ----------
        mode : str
            One of "full", "cwema_only", "blc_only", "none"

        Raises
        ------
        ValueError
            If mode is not recognized
        """
        if mode not in ABLATION_MODES:
            raise ValueError(
                f"ablation_mode must be one of {ABLATION_MODES}, got '{mode}'"
            )
        self.ablation_mode = mode

    def add_rep_row(self, pipeline_name: str, row_dict: Dict) -> None:
        """
        Accumulate a rep-level data row for the specified pipeline.

        Used to buffer rep-level outputs for later CSV export. Typically called
        during post-session analysis when processing recorded session data through
        each pipeline (raw, sma, cwema) separately.

        Parameters
        ----------
        pipeline_name : str
            One of "raw", "sma", "cwema"

        row_dict : dict
            Completed rep-level data row with schema:
            {
                'exercise_block': int,
                'affected_side': str,
                'set_number': int,
                'rep_number': int,
                'timestamp': str,
                'exercise': str,
                'peak_angle': float,
                'duration_frames': int,
                'rom_label': str,
                'trunk_lean_detected': bool,
                'shoulder_hiking_detected': bool,
                'fatigue_level': str,
                'micro_break_triggered': bool,
                'stop_triggered': bool,
                'form_cue_triggered': bool,
                'mean_rom_decline_pct': float,
                'mean_dur_increase_pct': float,
                'valid_reps_in_window': int,
            }

            This schema matches the main session CSV export for fair comparison.

        Raises
        ------
        ValueError
            If pipeline_name not recognized

        Notes
        -----
        Rows are stored in order. Multiple calls append rows to the buffer.
        Call reset_row_buffers() to clear all accumulated rows.
        """
        if pipeline_name not in PIPELINE_NAMES:
            raise ValueError(
                f"pipeline_name must be one of {PIPELINE_NAMES}, got '{pipeline_name}'"
            )
        
        self._row_accumulator[pipeline_name].append(row_dict)

    def get_pipeline_rows(self, pipeline_name: str) -> List[Dict]:
        """
        Retrieve all accumulated rep-level rows for a pipeline.

        Parameters
        ----------
        pipeline_name : str
            One of "raw", "sma", "cwema"

        Returns
        -------
        list of dict
            All rows accumulated for this pipeline, in order.
            Empty list if no rows have been added.

        Raises
        ------
        ValueError
            If pipeline_name not recognized
        """
        if pipeline_name not in PIPELINE_NAMES:
            raise ValueError(
                f"pipeline_name must be one of {PIPELINE_NAMES}, got '{pipeline_name}'"
            )
        
        return self._row_accumulator[pipeline_name].copy()

    def reset_row_buffers(self) -> None:
        """
        Clear all accumulated rep-level rows across all pipelines.

        Call this before starting a new session's post-processing, or when
        switching to a different dataset.

        This does NOT affect frame-level processing state (SMA buffers,
        CW-EMA filter). Those are managed by reset().
        """
        for pipeline_name in PIPELINE_NAMES:
            self._row_accumulator[pipeline_name].clear()

    def export_aligned_csvs(
        self,
        filepath_prefix: str,
        session_metadata: Dict,
        fieldnames: List[str],
    ) -> Dict[str, str]:
        """
        Export per-pipeline rep-level data as aligned CSV files.

        Generates three separate CSV files (raw, sma, cwema) with identical
        schema for within-subject three-pipeline comparison and ablation analysis.

        This is the primary export interface for producing thesis-aligned
        per-pipeline outputs required for §3.3 three-pipeline comparison.

        Parameters
        ----------
        filepath_prefix : str
            Base path and filename prefix for CSV files.
            Example: "logs/session_S1_Abduction_20260407_132149"

            Outputs:
            - "{filepath_prefix}_raw.csv"
            - "{filepath_prefix}_sma.csv"
            - "{filepath_prefix}_cwema.csv"

        session_metadata : dict
            Session-level metadata to write as comment lines (lines starting with #).
            Typical keys:
            {
                'participant_id': 'S1',
                'exercise_type': 'Abduction',
                'session_date': '2026-04-07 13:24:02',
                'affected_side': 'Right',
                'total_sets': 3,
                'average_fps': 20.0,
                'session_notes': '(none)',
                # ... additional metadata lines
            }

        fieldnames : list of str
            CSV column headers. Typically:
            [
                'exercise_block', 'affected_side', 'set_number', 'rep_number',
                'timestamp', 'exercise', 'peak_angle', 'duration_frames',
                'rom_label', 'trunk_lean_detected', 'shoulder_hiking_detected',
                'fatigue_level', 'micro_break_triggered', 'stop_triggered',
                'form_cue_triggered', 'mean_rom_decline_pct',
                'mean_dur_increase_pct', 'valid_reps_in_window'
            ]

            This matches the main session CSV schema for consistency.

        Returns
        -------
        dict
            Mapping of pipeline names to output filepaths:
            {
                'raw': '/path/to/file_raw.csv',
                'sma': '/path/to/file_sma.csv',
                'cwema': '/path/to/file_cwema.csv',
            }

        Raises
        ------
        IOError
            If file write fails (permission, disk full, etc.)

        OSError
            If directory creation fails

        Notes
        -----
        - Creates parent directories if needed
        - Overwrites existing files with same name
        - Metadata lines are written first (as CSV comments)
        - All three pipelines write identical metadata (front matter)
        - Column headers follow metadata lines
        - Rep data rows follow in accumulated order
        - All three CSVs have identical schema for fair comparison

        Example Usage
        --------
        processor = PipelineProcessor()

        # Accumulate rows during session post-processing
        for rep in raw_reps:
            processor.add_rep_row('raw', rep)
            processor.add_rep_row('sma', rep_sma)
            processor.add_rep_row('cwema', rep_cwema)

        # Export aligned CSVs
        metadata = {
            'participant_id': 'S1',
            'exercise_type': 'Abduction',
            ...
        }
        fieldnames = ['exercise_block', 'affected_side', ...]

        result = processor.export_aligned_csvs(
            'logs/my_session',
            metadata,
            fieldnames
        )
        # result = {
        #   'raw': 'logs/my_session_raw.csv',
        #   'sma': 'logs/my_session_sma.csv',
        #   'cwema': 'logs/my_session_cwema.csv',
        # }
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(filepath_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Track output file paths
        output_files = {}

        # Write CSV for each pipeline
        for pipeline_name in [PIPELINE_RAW, PIPELINE_SMA, PIPELINE_CWEMA]:
            # Construct output filename with pipeline suffix
            filepath = f"{filepath_prefix}_{pipeline_name}.csv"
            
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    # Write metadata section (comment lines)
                    for key, value in session_metadata.items():
                        if isinstance(value, bool):
                            value_str = "True" if value else "False"
                        elif isinstance(value, (list, tuple)):
                            value_str = ", ".join(str(v) for v in value)
                        else:
                            value_str = str(value)
                        csvfile.write(f"# {key}: {value_str}\n")
                    
                    # Add pipeline identity as metadata
                    csvfile.write(f"# Pipeline: {pipeline_name.upper()}\n")
                    csvfile.write("#\n")
                    
                    # Write CSV data
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write accumulated rows for this pipeline
                    rows = self._row_accumulator[pipeline_name]
                    writer.writerows(rows)
                
                output_files[pipeline_name] = filepath
            
            except (IOError, OSError) as e:
                raise IOError(
                    f"Failed to write {pipeline_name} CSV to {filepath}: {str(e)}"
                )

        return output_files

    def get_fps_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-pipeline FPS and latency statistics.

        Computes mean latency (ms), mean FPS, and frame count for each
        pipeline from accumulated timing data. Used for thesis feasibility
        reporting (§3.3 pipeline comparison).

        Returns
        -------
        dict
            Per-pipeline stats::

                {
                    "raw":   {"frames": int, "mean_latency_ms": float, "mean_fps": float},
                    "sma":   {"frames": int, "mean_latency_ms": float, "mean_fps": float},
                    "cwema": {"frames": int, "mean_latency_ms": float, "mean_fps": float},
                }

            If no frames have been processed for a pipeline, mean_latency_ms
            and mean_fps are 0.0.
        """
        summary = {}
        for name in [PIPELINE_RAW, PIPELINE_SMA, PIPELINE_CWEMA]:
            times = self._timing[name]
            n = len(times)
            if n > 0:
                mean_sec = sum(times) / n
                mean_ms = round(mean_sec * 1000, 4)
                mean_fps = round(1.0 / mean_sec, 2) if mean_sec > 0 else 0.0
            else:
                mean_ms = 0.0
                mean_fps = 0.0
            summary[name] = {
                "frames": n,
                "mean_latency_ms": mean_ms,
                "mean_fps": mean_fps,
            }
        return summary

    def reset_timing(self) -> None:
        """Clear all per-pipeline timing accumulators."""
        for name in PIPELINE_NAMES:
            self._timing[name].clear()

    def record_angle(self, pipeline_name: str, angle_deg: float) -> None:
        """
        Record a per-pipeline angle value for jitter computation.

        Call this once per frame per pipeline after computing the active
        exercise angle from that pipeline's filtered landmarks. The jitter
        metric accumulates |theta_t - theta_{t-1}| for each consecutive
        pair of valid frames.

        Parameters
        ----------
        pipeline_name : str
            One of "raw", "sma", "cwema".
        angle_deg : float
            Angle in degrees for this frame. NaN/Inf values are skipped
            (treated as detection miss) and reset the previous-angle state.
        """
        if pipeline_name not in PIPELINE_NAMES:
            raise ValueError(
                f"pipeline_name must be one of {PIPELINE_NAMES}, got '{pipeline_name}'"
            )
        state = self._jitter[pipeline_name]
        if not math.isfinite(angle_deg):
            # Detection miss — break the chain
            state["prev"] = None
            return
        if state["prev"] is not None:
            state["deltas"].append(abs(angle_deg - state["prev"]))
        state["prev"] = angle_deg

    def get_jitter_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-pipeline angle-stream jitter statistics.

        Jitter is defined as the Mean Absolute Frame-to-Frame Angle
        Difference (MAD)::

            jitter = mean( |theta_t - theta_{t-1}| )

        Lower values indicate a smoother angle stream. Comparing this
        metric across Raw, SMA, and CW-EMA quantifies each pipeline's
        smoothing effectiveness for thesis feasibility reporting.

        Returns
        -------
        dict
            Per-pipeline jitter stats::

                {
                    "raw":   {"frames": int, "mean_abs_delta_deg": float,
                              "std_abs_delta_deg": float, "max_abs_delta_deg": float},
                    "sma":   { ... },
                    "cwema": { ... },
                }

            ``frames`` is the number of consecutive-frame pairs used.
            All values are 0.0 when no angle pairs have been recorded.
        """
        summary = {}
        for name in [PIPELINE_RAW, PIPELINE_SMA, PIPELINE_CWEMA]:
            deltas = self._jitter[name]["deltas"]
            n = len(deltas)
            if n > 0:
                mean_d = sum(deltas) / n
                var_d = sum((d - mean_d) ** 2 for d in deltas) / n
                std_d = math.sqrt(var_d)
                max_d = max(deltas)
            else:
                mean_d = 0.0
                std_d = 0.0
                max_d = 0.0
            summary[name] = {
                "frames": n,
                "mean_abs_delta_deg": round(mean_d, 4),
                "std_abs_delta_deg": round(std_d, 4),
                "max_abs_delta_deg": round(max_d, 4),
            }
        return summary

    def reset_jitter(self) -> None:
        """Clear all per-pipeline angle-stream jitter accumulators."""
        for name in PIPELINE_NAMES:
            self._jitter[name] = {"prev": None, "deltas": []}

    def __repr__(self) -> str:
        """Return readable representation for debugging."""
        return (
            f"PipelineProcessor("
            f"sma_window_size={self.sma_window_size}, "
            f"ablation_mode='{self.ablation_mode}', "
            f"cwema_filter={self.cwema_filter})"
        )
