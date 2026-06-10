"""
Fatigue Module
Implements temporal fatigue detection with dual-mode support:
  - Threshold-based MVP (pilot-tuned, always available)
  - Mamdani fuzzy inference (self-contained, no external dependency)

Research Reference: docs/Temporal_Fatigue_Research.md

Phase 4 MVP Features:
- Early-set baseline computation (median of reps 2-4)
- Percent-change metrics: rom_decline_pct, dur_increase_pct
- Rep validity gating (prevents tracking-inflated durations)
- W=5 sliding window with 2-tier fatigue triggers
- Compensation escalation modifier
- Micro-break prompts with cooldown

Phase 5 Enhancement: Mamdani fuzzy inference (Section 3.7)
- Fuzzy membership functions for ROM decline and duration increase (Low/Medium/High)
- 9-rule Mamdani rule base with transparent IF-THEN logic
- Center-of-singletons defuzzification producing crisp deterioration score (0-100)
- Self-contained implementation using custom trimf_eval()/trapmf_eval() + numpy only
- Controllable via config.json fatigue_detection.use_fuzzy_inference (default: true)
- Fallback to threshold mode when explicitly disabled

Mode selection logic:
  1. If use_fuzzy_inference=True (default): Mamdani fuzzy path active
  2. If use_fuzzy_inference=False: Threshold-based MVP active fallback
  3. Constructor kwarg overrides config if provided explicitly
"""

import numpy as np
from statistics import median

# Mamdani fuzzy inference is self-contained (uses trimf_eval/trapmf_eval + numpy).
# No external fuzzy library dependency required.
FUZZY_AVAILABLE = True


class MamdaniFuzzyInference:
    """
    Mamdani fuzzy inference system for fatigue deterioration scoring.
    
    Inputs:
      - rom_decline_pct: Percent ROM decline from baseline (0-100%)
      - dur_increase_pct: Percent duration increase from baseline (0-100%)
    
    Output:
      - deterioration_score: Crisp fatigue level (0-100) via center-of-singletons defuzzification
    
    Membership Functions (thesis §3.7, σ=20):
      ROM/Dur inputs:  Low trapmf(0-10-20%), Medium trimf(10-20-30%), High trapmf(20-30-100%)
      Deterioration:   Low (0-33), Medium (25-66), High (50-100)
    
    Rule Base (9 Mamdani rules):
      1. IF rom=Low AND dur=Low THEN deterioration=Low
      2. IF rom=Low AND dur=Medium THEN deterioration=Low
      3. IF rom=Low AND dur=High THEN deterioration=Medium
      4. IF rom=Medium AND dur=Low THEN deterioration=Low
      5. IF rom=Medium AND dur=Medium THEN deterioration=Medium
      6. IF rom=Medium AND dur=High THEN deterioration=High
      7. IF rom=High AND dur=Low THEN deterioration=Medium
      8. IF rom=High AND dur=Medium THEN deterioration=High
      9. IF rom=High AND dur=High THEN deterioration=High
    """
    
    def __init__(self):
        """Initialize fuzzy membership function parameters (trapezoidal/triangular, thesis §3.7)."""
        # No need to pre-compute membership values - compute on-demand
        # This avoids numpy divide-by-zero warnings with degenerate triangles
        pass
    
    def compute_deterioration_score(self, rom_decline_pct, dur_increase_pct):
        """
        Compute crisp deterioration score using Mamdani inference.
        
        Args:
            rom_decline_pct: ROM decline (0-100%)
            dur_increase_pct: Duration increase (0-100%)
        
        Returns:
            float: Deterioration score 0-100 (0=low, 100=high)
        """
        # Clamp inputs to valid range (High MF extends to 100 per thesis §3.7)
        rom = np.clip(rom_decline_pct, 0, 100)
        dur = np.clip(dur_increase_pct, 0, 100)
        
        # Step 1: Fuzzify inputs using thesis §3.7 MF definitions (σ=20)
        # Low:  trapmf [0, 0, 0.5σ, σ]    = [0, 0, 10, 20]  — flat plateau 0-10%, descends to 20%
        # Med:  trimf  [0.5σ, σ, 1.5σ]    = [10, 20, 30]     — triangle peaking at 20%
        # High: trapmf [σ, 1.5σ, 100, 100] = [20, 30, 100, 100] — ascends 20-30%, flat plateau 30%+
        rom_low_grade = trapmf_eval(rom, [0, 0, 10, 20])
        rom_medium_grade = trimf_eval(rom, [10, 20, 30])
        rom_high_grade = trapmf_eval(rom, [20, 30, 100, 100])
        
        dur_low_grade = trapmf_eval(dur, [0, 0, 10, 20])
        dur_medium_grade = trimf_eval(dur, [10, 20, 30])
        dur_high_grade = trapmf_eval(dur, [20, 30, 100, 100])
        
        # Step 2: Apply fuzzy rules (9 rules using AND aggregation = min)
        # Rule 1: IF rom=Low AND dur=Low THEN det=Low
        rule1_det_low = min(rom_low_grade, dur_low_grade)
        
        # Rule 2: IF rom=Low AND dur=Medium THEN det=Low
        rule2_det_low = min(rom_low_grade, dur_medium_grade)
        
        # Rule 3: IF rom=Low AND dur=High THEN det=Medium
        rule3_det_medium = min(rom_low_grade, dur_high_grade)
        
        # Rule 4: IF rom=Medium AND dur=Low THEN det=Low
        rule4_det_low = min(rom_medium_grade, dur_low_grade)
        
        # Rule 5: IF rom=Medium AND dur=Medium THEN det=Medium
        rule5_det_medium = min(rom_medium_grade, dur_medium_grade)
        
        # Rule 6: IF rom=Medium AND dur=High THEN det=High
        rule6_det_high = min(rom_medium_grade, dur_high_grade)
        
        # Rule 7: IF rom=High AND dur=Low THEN det=Medium
        rule7_det_medium = min(rom_high_grade, dur_low_grade)
        
        # Rule 8: IF rom=High AND dur=Medium THEN det=High
        rule8_det_high = min(rom_high_grade, dur_medium_grade)
        
        # Rule 9: IF rom=High AND dur=High THEN det=High
        rule9_det_high = min(rom_high_grade, dur_high_grade)
        
        # Step 3: Aggregate output fuzzy sets (OR over all rules for each output)
        # Output Low: max of rule 1, 2, 4
        output_low_grade = max(rule1_det_low, rule2_det_low, rule4_det_low)
        
        # Output Medium: max of rule 3, 5, 7
        output_medium_grade = max(rule3_det_medium, rule5_det_medium, rule7_det_medium)
        
        # Output High: max of rule 6, 8, 9
        output_high_grade = max(rule6_det_high, rule8_det_high, rule9_det_high)
        
        # Step 4: Defuzzify using center-of-singletons method
        # Weighted average of modal output values (Low=10, Medium=50, High=90)
        # This is a well-known real-time approximation of centroid defuzzification.
        # See: thesis §3.7 — center-of-singletons avoids numerical integration overhead.
        numerator = (output_low_grade * 10.0) + (output_medium_grade * 50.0) + (output_high_grade * 90.0)
        denominator = output_low_grade + output_medium_grade + output_high_grade
        
        if denominator < 1e-6:
            # No activation (shouldn't happen with valid input)
            return 0.0
        
        deterioration_score = numerator / denominator
        return float(np.clip(deterioration_score, 0, 100))


def trimf_eval(x, abc):
    """Evaluate triangular membership function at point x.

    Handles left-shoulder [a, a, c] (a==b) correctly: grade=1.0 at x=a
    (peak), decreasing linearly to 0 at x=c. The previous version returned
    0.0 at x=a due to the ``x <= a`` boundary condition catching the peak.

    Standard cases:
        [a, b, c] with a < b < c: triangular, peak at b
        [a, a, c] with a == b:    left-shoulder, peak at a descending to c
        [a, b, b] with b == c:    not used in this system
    """
    a, b, c = abc[0], abc[1], abc[2]
    if x < a or x >= c:
        return 0.0
    if x <= b:
        if b == a:
            if c == a:
                return 0.0
            return (c - x) / (c - a)
        return (x - a) / (b - a)
    else:
        if c == b:
            return 0.0
        return (c - x) / (c - b)


def trapmf_eval(x, abcd):
    """Evaluate trapezoidal membership function at point x.

    Standard trapezoidal MF with four parameters [a, b, c, d]:
      - 0.0 for x <= a or x >= d
      - Rising slope from a to b
      - 1.0 (flat plateau) from b to c
      - Falling slope from c to d

    Special cases used in this system (thesis §3.7, σ=20):
      - Low:  [0, 0, 10, 20]  — left-shoulder: flat 1.0 from 0-10%, descends to 0 at 20%
      - High: [20, 30, 100, 100] — right-shoulder: ascends from 20-30%, flat 1.0 from 30-100%

    Args:
        x: Input value to evaluate
        abcd: List of 4 parameters [a, b, c, d] defining the trapezoid

    Returns:
        float: Membership grade in [0.0, 1.0]
    """
    a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]
    # Handle degenerate case: zero-width trapezoid
    if a == d:
        return 0.0
    # Right-shoulder case: c==d (e.g., [20, 30, 100, 100])
    # x==d should be INSIDE the plateau, not outside
    if c == d and x == d:
        return 1.0
    # Left-shoulder case: a==b (e.g., [0, 0, 10, 20])
    # x==a should be INSIDE the plateau
    if a == b and x == a:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if x < b:
        # Rising slope
        if b == a:
            return 1.0  # Degenerate left edge — treat as flat
        return (x - a) / (b - a)
    if x <= c:
        # Flat plateau
        return 1.0
    # Falling slope (x > c and x < d)
    if d == c:
        return 0.0  # Degenerate right edge
    return (d - x) / (d - c)


class FatigueModule:
    """
    Temporal fatigue detection using dual-mode approach:
    
    1. Threshold-based MVP (always available): Pilot-tuned thresholds from Phase 4
    2. Mamdani fuzzy inference (self-contained, default mode): 
       Transparent fuzzy logic following thesis §3.7
    
    Analyzes rep_history from RepetitionTracker to detect fatigue patterns
    and trigger micro-break interventions.
    """
    
    # Default fatigue thresholds (pilot-tuned; validated in Phase 6)
    # Research: 10% is conservative given ~6% measurement variability
    # These can be overridden by config.json
    DEFAULT_MEDIUM_THRESHOLD = 10.0  # percent decline/increase
    DEFAULT_HIGH_THRESHOLD = 20.0    # percent decline/increase
    DEFAULT_SEVERE_THRESHOLD = 30.0  # percent - triggers stop rule
    
    # Low confidence threshold multiplier (research doc)
    LOW_CONFIDENCE_MULTIPLIER = 1.5  # Raise thresholds when baseline is uncertain
    
    # Rep validity constraints
    DURATION_RELATIVE_CAP = 3.0   # Max 3x baseline duration
    DURATION_ABSOLUTE_CAP = 240   # Thesis §3.7: ~8 seconds @ 30 FPS; prevents tracking artifacts
    
    # Minimum valid reps before triggering fatigue alerts (prevents early false triggers)
    DEFAULT_MIN_VALID_REPS_FOR_TRIGGER = 3
    
    # Denominator floor for percent calculations (prevents noise amplification with low ROM)
    ROM_DENOMINATOR_FLOOR = 60.0  # degrees
    
    # Default window and cooldown values
    DEFAULT_WINDOW_SIZE = 5               # Number of valid reps to analyze
    DEFAULT_COMPENSATION_ESCALATION = 2   # Comp flags in window to trigger form cue
    DEFAULT_COOLDOWN_REPS = 2             # Reps before next prompt after break
    DEFAULT_CONSECUTIVE_WINDOWS_REQUIRED = 2  # Persistent qualifying windows before break
    
    # Default micro-break durations (seconds)
    DEFAULT_MEDIUM_BREAK_DURATION = 15
    DEFAULT_HIGH_BREAK_DURATION = 30
    
    # Default fuzzy score-to-level thresholds (0-100 deterioration score)
    DEFAULT_FUZZY_THRESHOLD_LOW = 40.0
    DEFAULT_FUZZY_THRESHOLD_HIGH = 60.0
    
    def __init__(self, config=None, use_fuzzy_inference=None):
        """
        Initialize fatigue module state.
        
        Args:
            config: ConfigLoader instance for threshold configuration (optional)
            use_fuzzy_inference: bool or None
                - True: Use Mamdani fuzzy inference (default, thesis-aligned)
                - False: Use threshold-based MVP fallback mode
                - None: Read from config, default True when config is unavailable
        """
        self.config = config
        
        # Determine inference mode:
        #   1. Explicit kwarg wins if provided
        #   2. Else read from config if available
        #   3. Else default to True (Mamdani fuzzy inference, thesis-aligned default)
        if use_fuzzy_inference is not None:
            self.use_fuzzy_inference = bool(use_fuzzy_inference)
        elif self.config is not None:
            self.use_fuzzy_inference = self.config.use_fuzzy_inference
        else:
            self.use_fuzzy_inference = True
        
        # Initialize fuzzy inference engine if enabled
        if self.use_fuzzy_inference:
            self.fuzzy_engine = MamdaniFuzzyInference()
        else:
            self.fuzzy_engine = None
        
        # Fuzzy-to-threshold mapping (for consistent trigger behavior)
        # Deterioration score (0-100) maps to fatigue levels:
        #   Low: 0-40, Medium: 30-70, High: 60-100
        # Load thresholds from config if available
        if self.config:
            fatigue_cfg = self.config.get_fatigue_thresholds()
            self.MEDIUM_THRESHOLD = fatigue_cfg['medium']
            self.HIGH_THRESHOLD = fatigue_cfg['high']
            self.SEVERE_THRESHOLD = fatigue_cfg['severe']
            self.WINDOW_SIZE = fatigue_cfg['window_size']
            self.COOLDOWN_REPS = fatigue_cfg['cooldown_reps']
            self.MEDIUM_BREAK_DURATION = fatigue_cfg['break_medium']
            self.HIGH_BREAK_DURATION = fatigue_cfg['break_high']
            self.COMPENSATION_ESCALATION = fatigue_cfg['compensation_escalation']
            self.MIN_VALID_REPS_FOR_TRIGGER = fatigue_cfg['min_valid_reps_for_trigger']
            self.CONSECUTIVE_WINDOWS_REQUIRED = fatigue_cfg['consecutive_windows_required']
            self.FUZZY_THRESHOLD_LOW = fatigue_cfg['fuzzy_threshold_low']
            self.FUZZY_THRESHOLD_HIGH = fatigue_cfg['fuzzy_threshold_high']
            self.DURATION_ABSOLUTE_CAP = fatigue_cfg.get('duration_absolute_cap', 240)
        else:
            self.MEDIUM_THRESHOLD = self.DEFAULT_MEDIUM_THRESHOLD
            self.HIGH_THRESHOLD = self.DEFAULT_HIGH_THRESHOLD
            self.SEVERE_THRESHOLD = self.DEFAULT_SEVERE_THRESHOLD
            self.WINDOW_SIZE = self.DEFAULT_WINDOW_SIZE
            self.COOLDOWN_REPS = self.DEFAULT_COOLDOWN_REPS
            self.MEDIUM_BREAK_DURATION = self.DEFAULT_MEDIUM_BREAK_DURATION
            self.HIGH_BREAK_DURATION = self.DEFAULT_HIGH_BREAK_DURATION
            self.COMPENSATION_ESCALATION = self.DEFAULT_COMPENSATION_ESCALATION
            self.MIN_VALID_REPS_FOR_TRIGGER = self.DEFAULT_MIN_VALID_REPS_FOR_TRIGGER
            self.CONSECUTIVE_WINDOWS_REQUIRED = self.DEFAULT_CONSECUTIVE_WINDOWS_REQUIRED
            self.FUZZY_THRESHOLD_LOW = self.DEFAULT_FUZZY_THRESHOLD_LOW
            self.FUZZY_THRESHOLD_HIGH = self.DEFAULT_FUZZY_THRESHOLD_HIGH
            self.DURATION_ABSOLUTE_CAP = 240
        
        self.reset()
    
    def reset(self):
        """Reset fatigue tracking state (e.g., for new session/set)."""
        # Baseline values
        self.baseline_rom = None
        self.baseline_dur = None
        self.baseline_computed = False
        self.baseline_low_confidence = False
        
        # Cooldown tracking (using counter only; no separate flag needed)
        self.reps_since_last_break = 0
        
        # Consecutive-window trigger tracking (requires 2 consecutive qualifying windows)
        self._consecutive_qualify_count = 0
        
        # Last computed values (for display/logging)
        self.last_fatigue_level = 'Low'
        self.last_metrics = {}
        self.last_fuzzy_score = None  # Track fuzzy deterioration score if used
    
    def compute_baseline(self, rep_history):
        """
        Compute early-set baseline from reps 2-4 (skip rep 1 warm-up).
        
        Uses rep_number to ensure we skip chronological rep 1, not just
        the first item in a filtered list (fixes bug where NaN rep 1 would
        cause wrong reps to be skipped).
        
        Args:
            rep_history: List of rep dicts from RepetitionTracker
            
        Returns:
            tuple: (baseline_rom, baseline_dur, low_confidence)
        """
        if len(rep_history) < 1:
            return None, None, True
        
        # Get valid reps with rep_number 2, 3, or 4 (skip rep 1 warm-up)
        # Uses rep_number field to ensure chronological correctness
        baseline_candidates = [
            r for r in rep_history 
            if r.get('rep_number', 0) in [2, 3, 4] 
            and not np.isnan(r.get('peak_angle', np.nan))
        ]
        
        if len(baseline_candidates) >= 3:
            # Full confidence: have all 3 baseline reps (2, 3, 4)
            baseline_rom = median([r['peak_angle'] for r in baseline_candidates[:3]])
            baseline_dur = median([r['duration_frames'] for r in baseline_candidates[:3]])
            low_confidence = False
        elif len(baseline_candidates) >= 1:
            # Partial: have some reps from 2-4
            baseline_rom = median([r['peak_angle'] for r in baseline_candidates])
            baseline_dur = median([r['duration_frames'] for r in baseline_candidates])
            low_confidence = True
        else:
            # Fallback: use any first valid rep (including rep 1 if that's all we have)
            valid_reps = [r for r in rep_history if not np.isnan(r.get('peak_angle', np.nan))]
            if len(valid_reps) == 0:
                return None, None, True
            baseline_rom = valid_reps[0]['peak_angle']
            baseline_dur = valid_reps[0]['duration_frames']
            low_confidence = True
        
        return baseline_rom, baseline_dur, low_confidence
    
    def is_rep_valid(self, rep, baseline_dur):
        """
        Check if rep is valid for fatigue analysis.
        
        Filters out reps with tracking-inflated durations (dropout artifacts).
        
        Args:
            rep: Rep dict from rep_history
            baseline_dur: Baseline duration in frames
            
        Returns:
            bool: True if rep is valid for fatigue metrics
        """
        if rep is None:
            return False
        
        peak_angle = rep.get('peak_angle', np.nan)
        duration = rep.get('duration_frames', 0)
        
        # Check for NaN peak angle
        if np.isnan(peak_angle):
            return False
        
        # Check duration constraints (both relative and absolute caps per Thesis §3.7)
        # Relative cap: 3x baseline prevents tracking dropout inflation
        if baseline_dur is not None and baseline_dur > 0:
            if duration > self.DURATION_RELATIVE_CAP * baseline_dur:
                return False
        
        # Absolute cap: ~240 frames (~8 sec @ 30 FPS) prevents extremely slow reps
        if duration > self.DURATION_ABSOLUTE_CAP:
            return False
        
        return True
    
    def compute_percent_changes(self, rep, baseline_rom, baseline_dur):
        """
        Compute percent changes from baseline for a single rep.
        
        Args:
            rep: Rep dict with 'peak_angle' and 'duration_frames'
            baseline_rom: Baseline ROM in degrees
            baseline_dur: Baseline duration in frames
            
        Returns:
            tuple: (rom_decline_pct, dur_increase_pct)
                   Positive values indicate deterioration/slowing
                   Clamped to >= 0 (improvement doesn't offset deterioration)
        """
        if baseline_rom is None or baseline_rom <= 0:
            return 0.0, 0.0
        
        current_rom = rep.get('peak_angle', baseline_rom)
        current_dur = rep.get('duration_frames', baseline_dur)
        
        # Use denominator floor to prevent noise amplification with low baseline ROM
        # Research doc: small baselines (e.g., 30°) make small drops appear as large %
        effective_baseline_rom = max(baseline_rom, self.ROM_DENOMINATOR_FLOOR)
        
        # ROM decline: positive = deterioration, clamped to >= 0
        # Clamping: improvement (negative) doesn't mask later fatigue in window mean
        rom_decline_pct = max(0.0, 100.0 * (baseline_rom - current_rom) / effective_baseline_rom)
        
        # Duration increase: positive = slowing, clamped to >= 0
        dur_increase_pct = 0.0
        if baseline_dur is not None and baseline_dur > 0:
            dur_increase_pct = max(0.0, 100.0 * (current_dur - baseline_dur) / baseline_dur)
        
        return rom_decline_pct, dur_increase_pct
    
    def get_window_metrics(self, rep_history, baseline_rom, baseline_dur):
        """
        Compute fatigue metrics over sliding window of last W valid reps.
        
        Args:
            rep_history: Full rep history list
            baseline_rom: Baseline ROM
            baseline_dur: Baseline duration
            
        Returns:
            dict: Metrics including mean_rom_decline, mean_dur_increase, comp_count
        """
        # Get last W valid reps
        valid_reps = []
        for rep in reversed(rep_history):
            if self.is_rep_valid(rep, baseline_dur):
                valid_reps.append(rep)
                if len(valid_reps) >= self.WINDOW_SIZE:
                    break
        
        valid_reps.reverse()  # Restore chronological order
        
        if len(valid_reps) == 0:
            return {
                'mean_rom_decline': 0.0,
                'mean_dur_increase': 0.0,
                'comp_count': 0,
                'valid_rep_count': 0,
                'window_full': False
            }
        
        # Compute percent changes for each valid rep
        rom_declines = []
        dur_increases = []
        comp_count = 0
        
        for rep in valid_reps:
            rom_dec, dur_inc = self.compute_percent_changes(rep, baseline_rom, baseline_dur)
            rom_declines.append(rom_dec)
            dur_increases.append(dur_inc)
            
            # Count compensation flags
            if rep.get('trunk_lean_detected', False) or rep.get('shoulder_hiking_detected', False):
                comp_count += 1
        
        return {
            'mean_rom_decline': np.mean(rom_declines),
            'mean_dur_increase': np.mean(dur_increases),
            'comp_count': comp_count,
            'valid_rep_count': len(valid_reps),
            'window_full': len(valid_reps) >= self.WINDOW_SIZE
        }
    
    def compute_fatigue_level_fuzzy(self, mean_rom_decline, mean_dur_increase, comp_count, valid_rep_count):
        """
        Compute fatigue level using Mamdani fuzzy inference.
        
        Used as enhanced version when self.use_fuzzy_inference is True.
        
        Args:
            mean_rom_decline: Mean ROM decline % from baseline
            mean_dur_increase: Mean duration increase % from baseline
            comp_count: Number of reps with compensation flags
            valid_rep_count: Number of valid reps in window
        
        Returns:
            dict: Fatigue level, triggers, and metrics (same format as threshold version)
        """
        result = {
            'fatigue_level': 'Low',
            'trigger_break': False,
            'trigger_stop': False,
            'trigger_form_cue': False,
            'break_duration': 0,
            'metrics': {}
        }
        
        # Compute fuzzy deterioration score
        assert self.fuzzy_engine is not None  # guaranteed by caller guard
        fuzzy_score = self.fuzzy_engine.compute_deterioration_score(
            mean_rom_decline, mean_dur_increase
        )
        self.last_fuzzy_score = fuzzy_score
        
        # Map fuzzy score to fatigue level
        if fuzzy_score >= self.FUZZY_THRESHOLD_HIGH:
            fatigue_level = 'High'
        elif fuzzy_score >= self.FUZZY_THRESHOLD_LOW:
            fatigue_level = 'Medium'
        else:
            fatigue_level = 'Low'
        
        result['fatigue_level'] = fatigue_level
        self.last_fatigue_level = fatigue_level
        
        # Form cue: compensation escalation (same logic as threshold mode)
        min_reps_required = self.WINDOW_SIZE if self.baseline_low_confidence else self.MIN_VALID_REPS_FOR_TRIGGER
        if comp_count >= self.COMPENSATION_ESCALATION and valid_rep_count >= min_reps_required:
            result['trigger_form_cue'] = True
        
        # Consecutive-window trigger for break: requires 2 consecutive qualifying windows
        # A window "qualifies" if fatigue_level is Medium or High
        window_qualifies = fatigue_level in ('Medium', 'High')
        
        if window_qualifies:
            # Increment consecutive qualifier count
            self._consecutive_qualify_count += 1
        else:
            # Reset counter when window doesn't qualify
            self._consecutive_qualify_count = 0
        
        # Break trigger: only fire if we have 2 consecutive qualifying windows AND cooldown satisfied
        if self.reps_since_last_break >= self.COOLDOWN_REPS and self._consecutive_qualify_count >= self.CONSECUTIVE_WINDOWS_REQUIRED:
            if fatigue_level == 'High':
                result['trigger_break'] = True
                result['break_duration'] = self.HIGH_BREAK_DURATION
                self.reps_since_last_break = 0
                self._consecutive_qualify_count = 0  # Reset after triggering
            elif fatigue_level == 'Medium':
                result['trigger_break'] = True
                result['break_duration'] = self.MEDIUM_BREAK_DURATION
                self.reps_since_last_break = 0
                self._consecutive_qualify_count = 0  # Reset after triggering
        
        return result
    
    def compute_fatigue_level(self, rep_history):
        """
        Compute current fatigue level from rep history.
        
        Main entry point called after each rep completion.
        Automatically uses Mamdani fuzzy inference if enabled, else threshold-based MVP.
        
        Args:
            rep_history: List of rep dicts from RepetitionTracker
            
        Returns:
            dict: {
                'fatigue_level': 'Low', 'Medium', or 'High',
                'trigger_break': bool,
                'trigger_stop': bool,
                'trigger_form_cue': bool,
                'break_duration': int (seconds),
                'metrics': {...}
            }
        """
        result = {
            'fatigue_level': 'Low',
            'trigger_break': False,
            'trigger_stop': False,
            'trigger_form_cue': False,
            'break_duration': 0,
            'metrics': {}
        }
        
        # Need at least a few reps to compute fatigue
        if len(rep_history) < 2:
            self.last_fatigue_level = 'Low'
            self.last_metrics = {}
            return result
        
        # Compute baseline (progressive: recompute until full confidence achieved)
        if not self.baseline_computed or self.baseline_low_confidence:
            baseline_rom, baseline_dur, low_confidence = self.compute_baseline(rep_history)
            if baseline_rom is not None:
                self.baseline_rom = baseline_rom
                self.baseline_dur = baseline_dur
                self.baseline_low_confidence = low_confidence
                # Only lock baseline when full confidence achieved (have reps 2-4)
                if not low_confidence:
                    self.baseline_computed = True
        
        # Cannot compute without baseline
        if self.baseline_rom is None:
            return result
        
        # Get window metrics
        metrics = self.get_window_metrics(rep_history, self.baseline_rom, self.baseline_dur)
        result['metrics'] = metrics
        self.last_metrics = metrics
        
        # Increment reps since last break
        self.reps_since_last_break += 1
        
        # Check for severe decline (stop rule) - applies to both modes
        # Look at most recent rep specifically
        if len(rep_history) > 0:
            last_rep = rep_history[-1]
            if self.is_rep_valid(last_rep, self.baseline_dur):
                rom_dec, _ = self.compute_percent_changes(last_rep, self.baseline_rom, self.baseline_dur)
                if rom_dec >= self.SEVERE_THRESHOLD:
                    result['fatigue_level'] = 'High'
                    result['trigger_stop'] = True
                    self.last_fatigue_level = 'High'
                    return result
        
        mean_rom_decline = metrics['mean_rom_decline']
        mean_dur_increase = metrics['mean_dur_increase']
        comp_count = metrics['comp_count']
        valid_rep_count = metrics['valid_rep_count']
        
        # Route to fuzzy or threshold logic based on mode
        if self.use_fuzzy_inference and self.fuzzy_engine is not None:
            return self.compute_fatigue_level_fuzzy(
                mean_rom_decline, mean_dur_increase, comp_count, valid_rep_count
            )
        else:
            # Original threshold-based MVP logic (fallback mode)
            # Determine effective thresholds (adjust for low confidence baseline)
            # Research doc §6: require stronger evidence when baseline is uncertain
            medium_thresh = self.MEDIUM_THRESHOLD
            high_thresh = self.HIGH_THRESHOLD
            min_reps_required = self.MIN_VALID_REPS_FOR_TRIGGER
            
            if self.baseline_low_confidence:
                medium_thresh *= self.LOW_CONFIDENCE_MULTIPLIER  # 10% -> 15%
                high_thresh *= self.LOW_CONFIDENCE_MULTIPLIER    # 20% -> 30%
                min_reps_required = self.WINDOW_SIZE  # Require full window
            
            # Determine fatigue level (but gate triggers on minimum valid reps)
            fatigue_level = 'Low'
            
            # Only compute fatigue level if we have enough valid reps
            # This prevents early false triggers from 1-2 rep means
            if valid_rep_count >= min_reps_required:
                if mean_rom_decline >= high_thresh or mean_dur_increase >= high_thresh:
                    fatigue_level = 'High'
                elif mean_rom_decline >= medium_thresh or mean_dur_increase >= medium_thresh:
                    fatigue_level = 'Medium'
            
            result['fatigue_level'] = fatigue_level
            self.last_fatigue_level = fatigue_level
            
            # Check compensation escalation (safety modifier)
            # Also requires minimum reps to avoid early false form cues
            if comp_count >= self.COMPENSATION_ESCALATION and valid_rep_count >= min_reps_required:
                result['trigger_form_cue'] = True
            
            # Consecutive-window trigger for break: requires 2 consecutive qualifying windows
            # A window "qualifies" if fatigue_level is Medium or High
            window_qualifies = fatigue_level in ('Medium', 'High')
            
            if window_qualifies:
                # Increment consecutive qualifier count
                self._consecutive_qualify_count += 1
            else:
                # Reset counter when window doesn't qualify
                self._consecutive_qualify_count = 0
            
            # Check if we should trigger a break (respecting cooldown via counter)
            # Cooldown: require COOLDOWN_REPS since last break before allowing new break
            # Consecutive windows: require 2 consecutive qualifying windows before triggering
            if self.reps_since_last_break >= self.COOLDOWN_REPS and self._consecutive_qualify_count >= self.CONSECUTIVE_WINDOWS_REQUIRED:
                if fatigue_level == 'High':
                    result['trigger_break'] = True
                    result['break_duration'] = self.HIGH_BREAK_DURATION
                    self.reps_since_last_break = 0  # Reset cooldown counter
                    self._consecutive_qualify_count = 0  # Reset after triggering
                elif fatigue_level == 'Medium':
                    result['trigger_break'] = True
                    result['break_duration'] = self.MEDIUM_BREAK_DURATION
                    self.reps_since_last_break = 0
                    self._consecutive_qualify_count = 0  # Reset after triggering
            
            return result
    
    def get_last_fatigue_level(self):
        """Get the most recently computed fatigue level."""
        return self.last_fatigue_level
    
    def get_last_metrics(self):
        """Get the most recently computed metrics dict."""
        return self.last_metrics
    
    def get_baseline_info(self):
        """
        Get baseline information for display/debugging.
        
        Returns:
            dict: Baseline ROM, duration, and confidence flag
        """
        return {
            'baseline_rom': self.baseline_rom,
            'baseline_dur': self.baseline_dur,
            'low_confidence': self.baseline_low_confidence,
            'computed': self.baseline_computed
        }
    
    def get_inference_mode(self):
        """
        Get current inference mode (for logging/debugging).
        
        Returns:
            str: 'Mamdani Fuzzy' or 'Threshold-based MVP'
        """
        if self.use_fuzzy_inference and self.fuzzy_engine is not None:
            return 'Mamdani Fuzzy'
        else:
            return 'Threshold-based MVP'
    
    def get_fuzzy_score(self):
        """
        Get the last computed fuzzy deterioration score (if fuzzy mode).
        
        Returns:
            float or None: Deterioration score (0-100) or None if threshold mode
        """
        return self.last_fuzzy_score
