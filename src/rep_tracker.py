"""
Repetition Tracker Module
Implements research-backed rep segmentation with hysteresis thresholds and peak detection

Research Support:
- Canny (1986): Hysteresis principle for noise reduction
- Coates & Wahlström (2023): Turning-point rep counting with guards
- Hsu et al. (2023): Viewpoint-invariant exercise repetition counting
- van den Hoorn et al. (2024): View-dependent measurement accuracy
- Gill et al. (2020): ROM normative values (150° abduction mean, N=2404)
"""

import numpy as np
from collections import deque


class RepetitionTracker:
    """
    Tracks exercise repetitions using hysteresis thresholds and peak detection
    """
    
    # Default exercise-dependent threshold presets (based on baseline angle differences)
    # These can be overridden by config.json
    # Frontal view (abduction): higher baseline (~21-25°) requires higher thresholds
    # Lateral view (flexion): lower baseline (~0-8°) uses standard thresholds
    DEFAULT_EXERCISE_THRESHOLDS = {
        'Abduction': {'start': 40.0, 'end': 30.0},  # Frontal view baseline ~21-25°
        'Flexion': {'start': 30.0, 'end': 20.0}     # Lateral view baseline ~0-8°
    }
    
    # ROM Classification Threshold
    # Research: Gill et al. (2020) - mean abduction 150° in healthy adults (N=2404)
    # Note: This is a pilot-tuned system-angle, not clinical goniometer standard
    DEFAULT_ROM_THRESHOLD = 150.0  # degrees
    
    def __init__(self, 
                 start_threshold=30.0,    # Pilot-tuned heuristic
                 end_threshold=20.0,       # 10° hysteresis gap
                 min_peak_distance=15,     # Frames (~0.5 sec @ 30 FPS)
                 min_peak_prominence=20.0, # Degrees above start threshold
                 smoothing_window=5,       # Frames for moving average
                 config=None):             # ConfigLoader instance (optional)
        """
        Initialize repetition tracker with research-backed parameters
        
        Args:
            start_threshold: Angle (degrees) to enter "in-rep" state
            end_threshold: Angle (degrees) to exit "in-rep" state
            min_peak_distance: Minimum frames between peaks (prevents double-counting)
            min_peak_prominence: Minimum peak height above start threshold
            smoothing_window: Moving average window size (frames)
            config: ConfigLoader instance for threshold configuration (optional)
        """
        self.config = config
        
        # Load thresholds from config if available
        if self.config:
            self.EXERCISE_THRESHOLDS = {
                'Abduction': self.config.get_rep_thresholds('Abduction'),
                'Flexion': self.config.get_rep_thresholds('Flexion')
            }
            self.ROM_CORRECT_THRESHOLD = self.config.rom_threshold
            min_peak_distance = self.config.min_peak_distance
            min_peak_prominence = self.config.min_peak_prominence
            smoothing_window = self.config.smoothing_window  # LC-11: config-driven, default 5
        else:
            self.EXERCISE_THRESHOLDS = self.DEFAULT_EXERCISE_THRESHOLDS.copy()
            self.ROM_CORRECT_THRESHOLD = self.DEFAULT_ROM_THRESHOLD
        
        # Thresholds — apply config-based defaults immediately (BUG-1 fix)
        # If config provided thresholds, use the Flexion preset as the safe default
        # so that even without an explicit set_exercise() call, config values take effect.
        # The Flexion preset is chosen because its thresholds (start=30, end=20) match
        # the constructor defaults, making this a backward-compatible safe default.
        if self.config and 'Flexion' in self.EXERCISE_THRESHOLDS:
            default_thresholds = self.EXERCISE_THRESHOLDS['Flexion']
            self.start_threshold = default_thresholds['start']
            self.end_threshold = default_thresholds['end']
        else:
            self.start_threshold = start_threshold
            self.end_threshold = end_threshold
        self.min_peak_distance = min_peak_distance
        self.min_peak_prominence = min_peak_prominence
        self.smoothing_window = smoothing_window

        # Phase 2: Persistence thresholds (config-driven, cached on init)
        if self.config:
            self.trunk_lean_persistence_threshold = self.config.trunk_lean_persistence
            self.shoulder_hiking_persistence_threshold = self.config.shoulder_hiking_persistence
        else:
            self.trunk_lean_persistence_threshold = 0.3
            self.shoulder_hiking_persistence_threshold = 0.3
        
        # State machine
        self.in_rep = False
        self.rep_count = 0
        self.current_peak_raw = 0.0         # Track RAW peak
        self.frames_since_last_peak = 0
        self.rep_start_frame = None
        
        # Smoothing buffer (for threshold checks only)
        self.angle_buffer = deque(maxlen=smoothing_window)
        
        # Rep history
        self.rep_history = []  # List of (peak_angle, duration_frames)
        
        # Last computed smoothed angle for overlay display (MISSING-6)
        self.last_smoothed_angle = np.nan
        
        # Phase 3.B: Frame counters for compensation persistence ratio (thesis-critical fix)
        # These counters are incremented during active rep to track persistent compensation
        # At rep completion, ratios (n_flagged / N_total) are computed and compared to thresholds
        self._trunk_lean_frame_count = 0
        self._shoulder_hiking_frame_count = 0
        self._valid_frame_count = 0
    
    def reset(self):
        """Reset tracker state (e.g., for new session)"""
        self.in_rep = False
        self.rep_count = 0
        self.current_peak_raw = 0.0
        self.frames_since_last_peak = 0
        self.rep_start_frame = None
        self.angle_buffer.clear()
        self.rep_history = []
        self.last_smoothed_angle = np.nan  # MISSING-6: Reset overlay value
        # Phase 3.B: Reset frame counters for new session
        self._trunk_lean_frame_count = 0
        self._shoulder_hiking_frame_count = 0
        self._valid_frame_count = 0
    
    def set_exercise(self, exercise_type):
        """
        Set exercise-specific thresholds based on expected baseline angles.
        
        Different camera views produce different baseline (resting) angles:
        - Frontal view (abduction): baseline ~21-25° due to natural arm hang angle
        - Lateral view (flexion): baseline ~0-8° (arm aligns with trunk vector)
        
        Args:
            exercise_type: 'Abduction' or 'Flexion'
        """
        if exercise_type in self.EXERCISE_THRESHOLDS:
            thresholds = self.EXERCISE_THRESHOLDS[exercise_type]
            self.start_threshold = thresholds['start']
            self.end_threshold = thresholds['end']
            print(f"Rep tracker: {exercise_type} thresholds set (start={self.start_threshold}°, end={self.end_threshold}°)")
        else:
            # Keep default thresholds if exercise type not recognized
            print(f"Rep tracker: Unknown exercise '{exercise_type}', using default thresholds")
    
    def get_smoothed_angle(self, raw_angle):
        """
        Apply moving average smoothing for threshold checks
        
        Args:
            raw_angle: Raw angle value (degrees) or NaN
            
        Returns:
            Smoothed angle or NaN if input is NaN
        """
        if np.isnan(raw_angle):
            return np.nan
        
        self.angle_buffer.append(raw_angle)
        return np.mean(self.angle_buffer)
    
    def update(self, raw_angle, frame_idx):
        """
        Process one frame and update rep state
        
        Args:
            raw_angle: Raw angle measurement (degrees) or NaN
            frame_idx: Current frame number
            
        Returns:
            tuple: (rep_completed: bool, peak_angle: float or None, rep_duration: int or None)
                   rep_completed = True if a valid rep just finished
                   peak_angle = RAW (unsmoothed) peak angle if rep completed, else None
                   rep_duration = duration in frames if rep completed, else None
        """
        self.frames_since_last_peak += 1
        
        # NaN handling: pause segmentation (research-backed default)
        if np.isnan(raw_angle):
            return False, None, None
        
        # Get smoothed angle for threshold checks only
        smoothed = self.get_smoothed_angle(raw_angle)
        self.last_smoothed_angle = smoothed  # MISSING-6: Expose for overlay display
        
        if not self.in_rep:
            # Check for rep start (using smoothed angle)
            if smoothed > self.start_threshold:
                self.in_rep = True
                self.current_peak_raw = raw_angle  # Store RAW peak
                self.rep_start_frame = frame_idx
        else:
            # Track peak during rep (using RAW angle for accuracy)
            if raw_angle > self.current_peak_raw:
                self.current_peak_raw = raw_angle
            
            # Check for rep end (using smoothed angle)
            if smoothed < self.end_threshold:
                self.in_rep = False
                
                # Validate rep with research-backed constraints
                peak_is_prominent = self.current_peak_raw > (self.start_threshold + self.min_peak_prominence)
                sufficient_gap = self.frames_since_last_peak >= self.min_peak_distance
                
                if peak_is_prominent and sufficient_gap:
                    # Valid rep completed
                    self.rep_count += 1
                    self.frames_since_last_peak = 0
                    
                    # Calculate duration
                    duration = frame_idx - self.rep_start_frame if self.rep_start_frame is not None else 0
                    
                    # Classify ROM (Phase 3.3)
                    rom_label = self.classify_rom(self.current_peak_raw)
                    
                    # Phase 3.B: Compute compensation flags using persistence ratio rule
                    trunk_lean_detected, shoulder_hiking_detected = self._compute_compensation_flags()
                    
                    # Store in history (extended schema for Phase 3.3-3.5)
                    self.rep_history.append({
                        'rep_number': self.rep_count,
                        'peak_angle': self.current_peak_raw,
                        'duration_frames': duration,
                        'start_frame': self.rep_start_frame,
                        'end_frame': frame_idx,
                        'rom_label': rom_label,                      # Phase 3.3: 'correct' or 'insufficient'
                        'trunk_lean_detected': trunk_lean_detected,  # Phase 3.B: Persistence ratio computed
                        'shoulder_hiking_detected': shoulder_hiking_detected  # Phase 3.B: Persistence ratio computed
                    })
                    
                    # Phase 3.B: Reset frame counters for next rep
                    self._reset_frame_counters()
                    
                    return True, self.current_peak_raw, duration
                else:
                    # Rejected: sub-peak or too close to previous rep
                    return False, None, None
        
        return False, None, None
    
    def get_rep_count(self):
        """Get current rep count"""
        return self.rep_count
    
    def get_last_rep_info(self):
        """
        Get information about the last completed rep
        
        Returns:
            dict or None: Last rep info (peak_angle, duration_frames, etc.) or None if no reps
        """
        if self.rep_history:
            return self.rep_history[-1]
        return None
    
    def get_rep_history(self):
        """
        Get full rep history
        
        Returns:
            list: List of rep info dicts
        """
        return self.rep_history
    
    def classify_rom(self, peak_angle):
        """
        Classify range of motion quality based on peak angle.
        
        Phase 3.3: ROM Classification
        Research: Gill et al. (2020) - mean abduction ~150° in healthy adults (N=2404)
        
        Note: This threshold is pilot-tuned for the system's angle computation,
        not a clinical goniometer standard. The 150° value serves as a 
        conservative functional target for rehabilitation exercises.
        
        Args:
            peak_angle: Peak angle achieved during the rep (degrees)
            
        Returns:
            str: 'correct' if peak_angle >= 150°, 'insufficient' otherwise
        """
        if peak_angle >= self.ROM_CORRECT_THRESHOLD:
            return 'correct'
        else:
            return 'insufficient'
    
    def accumulate_frame_compensation(self, trunk_lean_flag, shoulder_hiking_flag):
        """
        Accumulate frame-level compensation detection during active rep.
        
        Called by main application for each frame during an active rep.
        Increments frame counters to enable persistence ratio computation at rep completion.
        
        Phase 3.B: This enables the thesis-required rule:
          A rep is flagged as compensated only if:
          (n_flagged_frames / N_total_frames) > persistence_threshold
        
        Args:
            trunk_lean_flag: Boolean, True if trunk lean detected in this frame
            shoulder_hiking_flag: Boolean, True if shoulder hiking detected in this frame
        """
        if self.in_rep:
            # Increment valid frame counter (counts all frames during rep)
            self._valid_frame_count += 1
            
            # Increment compensation-specific counters
            if trunk_lean_flag:
                self._trunk_lean_frame_count += 1
            if shoulder_hiking_flag:
                self._shoulder_hiking_frame_count += 1
    
    def _compute_compensation_flags(self):
        """
        Compute final compensation flags using persistence ratio rule at rep completion.
        
        Phase 3.B: Implements thesis requirement for compensation persistence.
        Returns True for a compensation type only if the fraction of frames where
        the compensation was detected exceeds the configured threshold.
        
        Returns:
            tuple: (trunk_lean_detected, shoulder_hiking_detected)
                   Both are boolean, set based on persistence ratio > threshold
        """
        # Prevent division by zero
        if self._valid_frame_count == 0:
            return False, False
        
        # Compute persistence ratios
        trunk_lean_ratio = self._trunk_lean_frame_count / self._valid_frame_count
        shoulder_hiking_ratio = self._shoulder_hiking_frame_count / self._valid_frame_count
        
        # Apply threshold rule: flag only if ratio exceeds configured thresholds
        trunk_lean_detected = trunk_lean_ratio > self.trunk_lean_persistence_threshold
        shoulder_hiking_detected = shoulder_hiking_ratio > self.shoulder_hiking_persistence_threshold
        
        return trunk_lean_detected, shoulder_hiking_detected
    
    def _reset_frame_counters(self):
        """
        Reset compensation frame counters after rep completion.
        
        Called at the end of rep processing to prepare for next rep.
        """
        self._trunk_lean_frame_count = 0
        self._shoulder_hiking_frame_count = 0
        self._valid_frame_count = 0
    
    def update_last_rep_compensation(self, trunk_lean_detected=None, shoulder_hiking_detected=None):
        """
        Update compensation flags for the last completed rep.
        
        DEPRECATED: This method is superseded by accumulate_frame_compensation() and
        _compute_compensation_flags() in Phase 3.B.
        
        Kept for backward compatibility but no longer called in main.py.
        
        Args:
            trunk_lean_detected: True if trunk lean was detected during this rep
            shoulder_hiking_detected: True if shoulder hiking was detected during this rep
        """
        if self.rep_history:
            if trunk_lean_detected is not None:
                self.rep_history[-1]['trunk_lean_detected'] = trunk_lean_detected
            if shoulder_hiking_detected is not None:
                self.rep_history[-1]['shoulder_hiking_detected'] = shoulder_hiking_detected
