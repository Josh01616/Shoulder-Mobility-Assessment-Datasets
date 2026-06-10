"""
Config Loader Module
Loads and validates threshold configuration from JSON file

Phase 5: Task 3.9 / 5.J Alternative - Configurable Threshold File
"""

import json
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads threshold configuration from JSON file.
    
    Provides validated access to all configurable thresholds with
    fallback to hardcoded defaults if config file is missing or invalid.
    """
    
    # Default values (fallback if config.json is missing or invalid)
    DEFAULTS = {
        'rom_classification': {'value': 150},
        'rep_tracking': {
            'abduction': {'start_threshold': {'value': 40}, 'end_threshold': {'value': 30}},
            'flexion': {'start_threshold': {'value': 30}, 'end_threshold': {'value': 20}},
            'min_peak_distance': {'value': 15},
            'min_peak_prominence': {'value': 20}
        },
        'compensation_detection': {
            'trunk_lean': {'value': 15},
            'shoulder_hiking': {'value': 20}
        },
        'spatial_temporal_filtering': {
            'cwema_alpha_base': {'value': 0.4},
            'cwema_c_floor': {'value': 0.05},
            'bone_length_beta': {'value': 0.7},
            'bone_length_segment': {'value': 'left_shoulder_elbow'},
            'sma_window_size': {'value': 5},
            'unscorable_frame_pct': {'value': 0.20}
        },
        'fatigue_detection': {
            'duration_absolute_cap': {'value': 240},
            'medium_threshold': {'value': 10},
            'high_threshold': {'value': 20},
            'severe_threshold': {'value': 30},
            'window_size': {'value': 5},
            'cooldown_reps': {'value': 2},
            'break_durations': {'medium': {'value': 15}, 'high': {'value': 30}},
            'use_fuzzy_inference': {'value': True},
            'compensation_escalation': {'value': 2},
            'min_valid_reps_for_trigger': {'value': 3},
            'consecutive_windows_required': {'value': 2},
            'fuzzy_threshold_low': {'value': 40.0},
            'fuzzy_threshold_high': {'value': 60.0}
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config.json. If None, looks in script directory.
        """
        self.config_path: str
        self.config = {}
        self.load_errors = []
        
        # Auto-detect config path if not provided
        if config_path is None:
            # Look in the same directory as the main script
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(script_dir, 'config.json')
        else:
            self.config_path = config_path
        
        self._load_config()
    
    def _load_config(self):
        """Load and validate config from JSON file."""
        if not os.path.exists(self.config_path):
            self.load_errors.append(f"Config file not found: {self.config_path}")
            self.config = self.DEFAULTS.copy()
            print(f"[ConfigLoader] Warning: Using default thresholds (config.json not found)")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"[ConfigLoader] Loaded thresholds from {self.config_path}")
            self._validate_config()
        except json.JSONDecodeError as e:
            self.load_errors.append(f"JSON parse error: {e}")
            self.config = self.DEFAULTS.copy()
            print(f"[ConfigLoader] Warning: Invalid JSON, using defaults")
        except Exception as e:
            self.load_errors.append(f"Load error: {e}")
            self.config = self.DEFAULTS.copy()
            print(f"[ConfigLoader] Warning: Error loading config, using defaults")
    
    def _validate_config(self):
        """Validate config values against valid_range constraints."""
        validations = [
            ('rom_classification', 'value', 'valid_range'),
            ('compensation_detection.trunk_lean', 'value', 'valid_range'),
            ('compensation_detection.shoulder_hiking', 'value', 'valid_range'),
            ('fatigue_detection.medium_threshold', 'value', 'valid_range'),
            ('fatigue_detection.high_threshold', 'value', 'valid_range'),
            ('fatigue_detection.severe_threshold', 'value', 'valid_range'),
        ]
        
        for path, value_key, range_key in validations:
            section = self._get_nested(path)
            if section and value_key in section and range_key in section:
                value = section[value_key]
                valid_range = section[range_key]
                if len(valid_range) == 2:
                    min_val, max_val = valid_range
                    if not (min_val <= value <= max_val):
                        self.load_errors.append(
                            f"Warning: {path}.{value_key}={value} outside range [{min_val}, {max_val}]"
                        )
                        print(f"[ConfigLoader] {self.load_errors[-1]}")
    
    def _get_nested(self, path: str) -> Optional[Dict]:
        """Get nested config value by dot-separated path."""
        keys = path.split('.')
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path.
        
        Args:
            path: Dot-separated path like 'rom_classification.value'
            default: Default value if path not found
            
        Returns:
            Config value or default
        """
        result = self._get_nested(path)
        return result if result is not None else default
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get the 'value' field from a config section.
        
        Convenience method for sections that have a 'value' field.
        
        Args:
            path: Dot-separated path to section (not including '.value')
            default: Default value if not found
            
        Returns:
            The value field or default
        """
        section = self._get_nested(path)
        if isinstance(section, dict) and 'value' in section:
            return section['value']
        return default
    
    # =========================================================================
    # Convenience accessors for common thresholds
    # =========================================================================
    
    @property
    def rom_threshold(self) -> float:
        """ROM classification threshold in degrees."""
        return self.get_value('rom_classification', 150.0)
    
    @property
    def trunk_lean_threshold(self) -> float:
        """Trunk lean compensation threshold in degrees."""
        return self.get_value('compensation_detection.trunk_lean', 15.0)
    
    @property
    def shoulder_hiking_threshold(self) -> float:
        """Shoulder hiking compensation threshold in percent (as decimal)."""
        # Config stores as percent (e.g., 20), convert to ratio (0.20)
        return self.get_value('compensation_detection.shoulder_hiking', 20.0) / 100.0
    
    def get_rep_thresholds(self, exercise: str) -> Dict[str, float]:
        """
        Get rep tracking thresholds for a specific exercise.
        
        Args:
            exercise: 'Abduction' or 'Flexion'
            
        Returns:
            Dict with 'start' and 'end' threshold values
        """
        exercise_key = exercise.lower()
        defaults = {'start': 30.0, 'end': 20.0}
        
        section = self._get_nested(f'rep_tracking.{exercise_key}')
        if section:
            return {
                'start': section.get('start_threshold', {}).get('value', defaults['start']),
                'end': section.get('end_threshold', {}).get('value', defaults['end'])
            }
        return defaults
    
    @property
    def min_peak_distance(self) -> int:
        """Minimum frames between peaks."""
        return self.get_value('rep_tracking.min_peak_distance', 15)
    
    @property
    def min_peak_prominence(self) -> float:
        """Minimum peak prominence in degrees."""
        return self.get_value('rep_tracking.min_peak_prominence', 20.0)
    
    def get_fatigue_thresholds(self) -> Dict[str, Any]:
        """
        Get all fatigue detection thresholds.
        
        Returns:
            Dict with all fatigue threshold values
        """
        return {
            'duration_absolute_cap': self.get_value('fatigue_detection.duration_absolute_cap', 240),
            'medium': self.get_value('fatigue_detection.medium_threshold', 10.0),
            'high': self.get_value('fatigue_detection.high_threshold', 20.0),
            'severe': self.get_value('fatigue_detection.severe_threshold', 30.0),
            'window_size': self.get_value('fatigue_detection.window_size', 5),
            'cooldown_reps': self.get_value('fatigue_detection.cooldown_reps', 2),
            'break_medium': self.get('fatigue_detection.break_durations.medium.value', 15),
            'break_high': self.get('fatigue_detection.break_durations.high.value', 30),
            'compensation_escalation': self.get_value('fatigue_detection.compensation_escalation', 2),
            'min_valid_reps_for_trigger': self.get_value('fatigue_detection.min_valid_reps_for_trigger', 3),
            'consecutive_windows_required': self.get_value('fatigue_detection.consecutive_windows_required', 2),
            'fuzzy_threshold_low': self.get_value('fatigue_detection.fuzzy_threshold_low', 40.0),
            'fuzzy_threshold_high': self.get_value('fatigue_detection.fuzzy_threshold_high', 60.0)
        }
    
    @property
    def use_fuzzy_inference(self) -> bool:
        """Whether to use Mamdani fuzzy inference for performance deterioration scoring.

        Default is True because the thesis methodology includes the fuzzy inference
        module for temporal performance deterioration monitoring and micro-break
        prompting. Setting this to False in config.json keeps the threshold-based
        fallback available for comparison or troubleshooting.
        """
        val = self.get_value('fatigue_detection.use_fuzzy_inference', True)
        return bool(val)

    # =========================================================================
    # MediaPipe confidence thresholds (MISSING-2)
    # =========================================================================
    
    @property
    def mediapipe_detection_confidence(self) -> float:
        """MediaPipe min_detection_confidence (0.0-1.0).
        
        Controls the threshold for initial pose detection.
        Lower values detect poses more easily but with more false positives.
        """
        val = self.get_value('mediapipe.detection_confidence', 0.5)
        # Clamp to valid MediaPipe range
        return max(0.1, min(1.0, float(val)))
    
    @property
    def mediapipe_tracking_confidence(self) -> float:
        """MediaPipe min_tracking_confidence (0.0-1.0).
        
        Controls the threshold for frame-to-frame landmark tracking.
        Lower values maintain tracking through fast movements but less reliably.
        """
        val = self.get_value('mediapipe.tracking_confidence', 0.5)
        return max(0.1, min(1.0, float(val)))
    
    # =========================================================================
    # Spatial-Temporal Filtering (T9 - Thesis CW-EMA Module Parameters)
    # =========================================================================
    
    @property
    def cwema_alpha_base(self) -> float:
        """Base EMA smoothing factor for confidence-weighted EMA smoothing.
        
        Higher values = more responsive landmark tracking (less lag).
        Lower values = more smoothing (more lag, less noise).
        Typical: 0.4 for 30 FPS video.
        """
        val = self.get_value('spatial_temporal_filtering.cwema_alpha_base', 0.4)
        return max(0.1, min(0.9, float(val)))
    
    @property
    def cwema_c_floor(self) -> float:
        """Minimum confidence floor for EMA alpha computation.
        
        Prevents effective alpha from collapsing when confidence momentarily dips.
        Keeps filter responsive even during brief low-confidence frames.
        Typical: 0.05 (5% of max confidence).
        """
        val = self.get_value('spatial_temporal_filtering.cwema_c_floor', 0.05)
        return max(0.01, min(0.3, float(val)))
    
    @property
    def bone_length_beta(self) -> float:
        """Bone-length constancy penalty weight.
        
        Controls how strictly to enforce anatomical bone-length invariants.
        Used by the BLC (Bone-Length Constancy) validation in spatial-temporal filter.
        Higher = more aggressive flagging of anatomically inconsistent frames.
        Typical: 0.7.
        """
        val = self.get_value('spatial_temporal_filtering.bone_length_beta', 0.7)
        return max(0.5, min(1.0, float(val)))
    
    @property
    def bone_length_segment(self) -> str:
        """MediaPipe segment used as baseline for bone-length constancy validation.
        
        Example: 'left_shoulder_elbow' for shoulder rehab.
        Segment is automatically mirrored based on affected_side.
        """
        return self.get_value('spatial_temporal_filtering.bone_length_segment', 'left_shoulder_elbow')
    
    @property
    def sma_window_size(self) -> int:
        """Simple moving average window size (frames) for SMA pipeline.
        
        Used as comparison baseline in thesis ablation study.
        Smaller = less lag, larger = more stability.
        Typical: 5 frames (≈0.17 sec @ 30 FPS).
        """
        val = self.get_value('spatial_temporal_filtering.sma_window_size', 5)
        return max(3, min(15, int(val)))
    
    @property
    def unscorable_frame_pct(self) -> float:
        """Threshold for flagging rep as unscorable.
        
        Rep flagged as 'unscorable' if >N% of frames violate BLC constraint.
        Automates detection of reps with tracking instability or occlusion.
        Typical: 0.20 (20%).
        """
        val = self.get_value('spatial_temporal_filtering.unscorable_frame_pct', 0.20)
        return max(0.1, min(0.5, float(val)))
    
    # =========================================================================
    # Session configuration (MISSING-5)
    # =========================================================================
    
    @property
    def reps_per_set(self) -> int:
        """Target number of reps per set for auto-completion detection.
        
        Returns 0 if disabled. Default: 10.
        """
        val = self.get_value('session.reps_per_set', 10)
        return max(0, int(val))
    
    @property
    def total_sets(self) -> int:
        """Number of sets per exercise session. Default: 3."""
        val = self.get_value('session.total_sets', 3)
        return max(1, int(val))
    
    @property
    def smoothing_window(self) -> int:
        """Moving average window size (frames) for rep segmentation smoothing.
        
        Default: 5 frames (≈0.17 sec @ 30 FPS).
        """
        val = self.get_value('session.smoothing_window', 5)
        return max(3, min(15, int(val)))
    
    # =========================================================================
    # Compensation persistence thresholds (LC-1)
    # =========================================================================
    
    @property
    def trunk_lean_persistence(self) -> float:
        """Fraction of rep frames where trunk lean must be detected to flag.
        
        Default: 0.3 (30% of frames). Used by LC-7 frame persistence logic.
        """
        val = self.get_value('compensation_detection.trunk_lean_persistence', 0.3)
        return max(0.1, min(0.8, float(val)))
    
    @property
    def shoulder_hiking_persistence(self) -> float:
        """Fraction of rep frames where shoulder hiking must be detected to flag.
        
        Default: 0.3 (30% of frames). Used by LC-7 frame persistence logic.
        """
        val = self.get_value('compensation_detection.shoulder_hiking_persistence', 0.3)
        return max(0.1, min(0.8, float(val)))
    
    # =========================================================================
    # Report interpretation thresholds (LC-3 / LC-4 / LC-9)
    # =========================================================================
    
    @property
    def correct_rom_good_pct(self) -> float:
        """Minimum correct ROM percentage to label as 'Good'. Default: 70."""
        val = self.get_value('report_interpretation.correct_rom_good_pct', 70)
        return float(val)
    
    @property
    def avg_peak_good_angle(self) -> float:
        """Minimum average peak angle to label as 'Good'. Default: 150."""
        val = self.get_value('report_interpretation.avg_peak_good_angle', 150)
        return float(val)
    
    @property
    def compensation_acceptable_pct(self) -> float:
        """Maximum compensation fraction before labeling as 'Frequent'. Default: 0.2."""
        val = self.get_value('report_interpretation.compensation_acceptable_pct', 0.2)
        return float(val)
    
    @property
    def high_fatigue_significant_count(self) -> int:
        """Number of high-fatigue events before labeling as 'Significant'. Default: 3."""
        val = self.get_value('report_interpretation.high_fatigue_significant_count', 3)
        return int(val)
    
    @property
    def high_compensation_rate_pct(self) -> float:
        """Compensation rate threshold for CSV anomaly flag. Default: 0.5."""
        val = self.get_value('report_interpretation.high_compensation_rate_pct', 0.5)
        return float(val)
    
    def get_report_thresholds(self) -> dict:
        """Get all report interpretation thresholds as a dict.
        
        Returns:
            Dict with all report threshold values
        """
        return {
            'correct_rom_good_pct': self.correct_rom_good_pct,
            'avg_peak_good_angle': self.avg_peak_good_angle,
            'compensation_acceptable_pct': self.compensation_acceptable_pct,
            'high_fatigue_significant_count': self.high_fatigue_significant_count,
            'high_compensation_rate_pct': self.high_compensation_rate_pct,
        }
    
    # =========================================================================
    # Calibration tuning parameters (P-2 / P-3)
    # =========================================================================
    
    @property
    def calibration_duration_sec(self) -> float:
        """Duration of calibration countdown window in seconds. Default: 10.0."""
        val = self.get_value('calibration.calibration_duration_sec', 10.0)
        return max(2.0, min(10.0, float(val)))
    
    @property
    def calibration_jitter_threshold_px(self) -> float:
        """Maximum acceptable landmark jitter (std dev in pixels) during calibration. Default: 40.0. Safety clamp max: 50.0."""
        val = self.get_value('calibration.calibration_jitter_threshold_px', 40.0)
        return max(5.0, min(50.0, float(val)))
    
    def has_errors(self) -> bool:
        """Check if any load/validation errors occurred."""
        return len(self.load_errors) > 0
    
    def get_errors(self) -> list:
        """Get list of load/validation errors."""
        return self.load_errors.copy()


# Global config instance (singleton pattern for easy access)
_global_config: Optional[ConfigLoader] = None


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Load or get the global config instance.
    
    Args:
        config_path: Optional path to config.json
        
    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = ConfigLoader(config_path)
    return _global_config


def get_config() -> ConfigLoader:
    """
    Get the global config instance (loads if not already loaded).
    
    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config
