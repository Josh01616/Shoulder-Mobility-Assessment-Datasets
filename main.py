"""
Shoulder Mobility Assessment System
Main entry point for the application
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import csv
import threading
from datetime import datetime
from PIL import Image, ImageTk
import sys
import os
import numpy as np
import platform
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]

# Audio playback (local WAV files via scipy + sounddevice)
try:
    from scipy.io import wavfile
    import sounddevice as sd
    AUDIO_SCIPY_AVAILABLE = True
except ImportError:
    wavfile = None
    sd = None
    AUDIO_SCIPY_AVAILABLE = False

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config_loader import load_config
from pose_processor import PoseProcessor
from rep_tracker import RepetitionTracker
from fatigue_module import FatigueModule
from spatial_temporal_filter import SpatialTemporalFilter


class RehabApp:
    """Main application class"""
    MAX_CALIBRATION_RETRIES = 3
    
    def __init__(self, root):
        self.root = root
        self.root.title("Shoulder Mobility Assessment System")
        self.root.geometry("1280x920")
        self.root.minsize(1200, 850)  # Prevent window from being too small
        
        # Load configuration (Phase 5: Task 3.9 - Configurable thresholds)
        self.config = load_config()
        if self.config.has_errors():
            print(f"[Main] Config warnings: {self.config.get_errors()}")
        
        # Video source (auto-detect or specify)
        self.video_source = self.find_camera()
        if self.video_source is None:
            self.video_source = 0  # Fallback to default
        
        # Video file mode (Phase 5.H)
        self.video_mode = tk.StringVar(value="camera")
        self.video_file_path = None
        
        # Audio feedback (Phase 5.I)
        self.audio_enabled = tk.BooleanVar(value=True)
        self.audio_lock = threading.Lock()  # Prevent overlapping beeps
        self.last_beep_time = 0  # Debounce rapid beeps
        self.last_rep_sound_key = None  # One-shot guard for per-rep audio cues
        
        # Audio assets path for local WAV file playback (local project assets only, no API)
        self.audio_assets_dir = os.path.join(os.path.dirname(__file__), 'assets', 'audio')
        self.audio_cache: dict[str, tuple[int, Any]] = {}  # Cache loaded audio data to avoid repeated disk I/O
        self.audio_cue_registry = {
            'rep_correct': 'success.wav',
            'rep_incorrect': 'error.wav',
            'calibration_start': 'calibration and setup/calibrating.wav',
            'calibration_complete': 'ready.wav',
            'calibration_retry': 'audio cue during exercise/lets try again.wav',
            'set_complete': 'set_complete.wav',
            'set_proceed': 'audio cue during exercise/UI-correct-set-proceed to next see.wav',
            'all_sets_done': 'audio cue during exercise/UI-exercise complete!.wav',
            'take_break_medium': 'fatigue and micro-break/lets pause for a quick break.wav',
            'take_break_high': 'take_a_break.wav',
            'break_complete': 'fatigue and micro-break/go ahead and relax.wav',
            'form_cue_trunk': 'audio cue during exercise/watch your alignment.wav',
            'form_cue_hiking': 'audio cue during exercise/watch your shoulders keep them relaxed.wav',
            'form_cue_general': 'audio cue during exercise/keep it steady.wav',
            'tracking_failed': 'calibration and setup/motion detection failed.wav',
            'low_confidence': 'calibration and setup/failed to detect shoulder angles.wav',
            'adjust_lighting': 'calibration and setup/please adjust lighting for better visibility.wav',
            'countdown_3': '3.wav',
            'countdown_2': '2.wav',
            'countdown_1': '1.wav',
        }
        self.audio_cooldowns = {}

        # Exercise guide assets (Phase 10)
        self.guides_assets_dir = os.path.join(os.path.dirname(__file__), 'assets', 'guides')
        self.guides_shown_in_session = set()  # Track one-time guide display per exercise per session
        
        # Video file playback (Phase 5.H)
        self.video_fps = 30  # Will be set from video file
        self.video_frame_delay = 33  # ms between frames (1000/30)
        self.video_rotation = 0  # Rotation angle from metadata (0, 90, 180, 270)
        
        # Session-level FPS for duration conversions (BUG-3 fix)
        # Set from actual source at session start; 30.0 is safe fallback only
        self.session_fps = 30.0
        
        # FPS tracking
        self.fps = 0
        self.fps_smoothed = 0.0  # Exponentially smoothed FPS for display
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # FPS session statistics (for thesis performance validation)
        self.fps_samples = []  # Store periodic FPS samples for avg calculation
        self.fps_sample_interval = 0.5  # Sample FPS every 0.5 seconds
        self.last_fps_sample_time = time.time()

        # Effective FPS for recording (measured from actual frame processing rate)
        self._recording_frame_count = 0
        self._recording_start_time = None
        self._effective_recording_fps = None  # Measured after first N frames
        
        # Exercise configuration
        self.current_exercise = tk.StringVar(value="Abduction")
        
        # Affected side selector (MISSING-1)
        # Determines which anatomical arm is tracked for rehabilitation
        self.affected_side = tk.StringVar(value="Right")
        
        # Pose processor (with config for thresholds + affected side)
        self.pose_processor = PoseProcessor(
            model_complexity=1, camera_view='frontal', 
            config=self.config, affected_side='Right'
        )
        
        # Repetition tracker (Phase 3.2, with config for thresholds)
        self.rep_tracker = RepetitionTracker(config=self.config)
        self.frame_idx = 0  # Track frame number
        
        # Fatigue module (Phase 4, with config for thresholds)
        self.fatigue_module = FatigueModule(config=self.config)
        self.micro_break_active = False
        self.break_start_time = None
        self.break_duration = 0
        self.last_countdown_update = 0  # Track last countdown update time
        self.break_pause_accumulated = 0  # Track accumulated pause time during break
        
        # Spatial-temporal filter (CW-EMA smoothing for MediaPipe landmarks - T9)
        # Sits between raw pose detection and angle computation
        # Parameters now from config.json (thesis Table 13: tunable synthesis parameters)
        self.spatial_temporal_filter = SpatialTemporalFilter(
            alpha_base=self.config.cwema_alpha_base,      # CW-EMA smoothing factor from config
            c_floor=self.config.cwema_c_floor             # Confidence floor from config
        )
        
        # Compensation tracking (Phase 3.4-3.5)
        # Track if compensation occurred at any point during the current rep
        self.current_rep_trunk_lean = False
        self.current_rep_shoulder_hiking = False
        self.current_rep_low_confidence = False  # Track low-confidence frames per rep
        
        # Track whether pose_processor has been released (BUG-6)
        self._pose_released = False

        # Short-gap pose tolerance (Phase 3: flexion occlusion fix)
        self._last_valid_landmarks = None
        self._last_valid_results = None
        self._pose_gap_frames = 0
        self._max_pose_gap_frames = 5  # Allow up to 5 frames (~0.17s at 30fps) gap
        
        # Session logging (Phase 4.12)
        self.session_log = []  # List of per-rep log entries
        self.session_start_time = None
        self.baseline_log = {}  # Track baseline per set: {set_number: baseline_info}
        
        # UI/UX enhancements (Phase 5.E)
        self.rep_flash_active = False  # For visual rep completion feedback
        self.rep_flash_color = None  # 'green' or 'red'
        self.rep_flash_start = 0  # Timestamp for flash duration
        self.show_session_summary = True  # Show dialog before CSV export
        
        # Set tracking (Phase 4 enhancement)
        self.current_set = 1
        self.total_sets = self.config.total_sets if self.config else 3  # LC-8: config-driven, default 3
        
        # Reps-per-set target for auto set-completion (MISSING-5)
        self.reps_per_set = self.config.reps_per_set if self.config else 10
        self.set_completed_prompted = False  # Guard against duplicate prompts for same set
        self.calibration_duration_sec = self.config.calibration_duration_sec if self.config else 10.0  # P-2: config-driven, default 10.0
        
        # Participant ID (Phase 5.A)
        self.participant_id = tk.StringVar(value="")  # User input via Entry widget
        
        # Video flip option for phone-recorded videos (Phase 5.H)
        self.flip_video = tk.BooleanVar(value=False)
        
        # Mirror display is always enabled (checkbox removed per groupchat decision)
        self.mirror_display = tk.BooleanVar(value=True)
        
        # Video recording (Phase 5.L)
        self.record_annotated = tk.BooleanVar(value=False)
        self.record_raw = tk.BooleanVar(value=False)
        self.video_writer_annotated = None
        self.video_writer_raw = None
        self.recording_paths = {}
        self._recording_initialized = False
        
        # Exercise block tracking (Phase 5.C: for mid-session exercise switching)
        self.exercise_block = 1  # Increments when exercise changes during session
        self.last_exercise = None  # Track last exercise to detect changes
        
        # Tracking Readiness Check (Calibration) - T11
        # Captures baseline data and verifies stable tracking before active exercise
        # Three-phase state machine: waiting_for_detection → countdown → complete
        self.calibration_phase_active = False  # True during entire calibration process
        self.calibration_waiting_for_detection = False  # True: waiting for pose detection before countdown
        self.calibration_countdown_started = False  # True: countdown phase active / baseline being collected
        self.calibration_start_time = None   # Timestamp when countdown started (not waiting phase)
        self.calibration_waiting_start_time = None  # Timestamp when waiting phase started
        self.calibration_detection_confidence_threshold = 0.5  # Min confidence to gate detection
        self.calibration_landmarks_buffer = []  # Buffer of valid landmarks during countdown
        self.calibration_segment_lengths = []  # Segment lengths (shoulder-elbow px) during countdown
        self.baseline_limb_length = {}  # Baseline segment lengths (e.g., shoulder-elbow)
        self.calibration_data = {}  # Stability metrics from calibration
        self.calibration_passed = False  # Flag to track if calibration succeeded
        self.require_calibration_pass = False  # Strict gate used for set-transition recalibration
        self._countdown_last_tone = 0  # Track last countdown tone played (C04-C06)
        self._calibration_retry_count = 0  # Retry counter for strict set-transition calibration
        
        # BLC tracking warning state (cleared each frame)
        self._blc_warning_active = False
        
        # Smoothed angle overlay toggle (MISSING-6)
        self.show_smoothed_overlay = tk.BooleanVar(value=False)
        
        # UI setup
        self.setup_ui()
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.is_paused = False  # Pause state for session
    
    def _get_participant_log_dir(self) -> str:
        """Return per-participant output directory inside logs/.
        
        Creates logs/<participant_id>/ if it doesn't exist.
        Falls back to flat logs/ if participant_id is empty.
        """
        base_logs = os.path.join(os.path.dirname(__file__), 'logs')
        participant_id = self.participant_id.get().strip() if hasattr(self, 'participant_id') else ""
        if participant_id:
            pdir = os.path.join(base_logs, participant_id)
        else:
            pdir = base_logs
        os.makedirs(pdir, exist_ok=True)
        return pdir
        
    def frames_to_seconds(self, frames):
        """Convert frame count to seconds using session FPS (BUG-3 fix).
        
        Uses self.session_fps which is set from the actual video/camera source
        at session start, with a safe fallback of 30.0 FPS.
        
        Args:
            frames: Number of frames (int or float)
            
        Returns:
            Duration in seconds (float)
        """
        fps = self.session_fps if self.session_fps > 0 else 30.0
        return frames / fps

    def _sanitize_fps(self, fps_value, fallback=30.0):
        """Return finite positive FPS in valid range, else fallback."""
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            return float(fallback)

        if not np.isfinite(fps) or fps <= 0 or fps > 120:
            return float(fallback)
        return float(fps)

    def _schedule_next_frame(self, frame_start_time):
        """Schedule next update_frame with source-aware timing.

        File mode compensates for processing time to avoid playback slowdown.
        Camera mode keeps existing near-ASAP behavior.
        """
        base_delay = max(1, int(self.video_frame_delay))

        if self.video_mode.get() == "file":
            elapsed_ms = int((time.perf_counter() - frame_start_time) * 1000)
            delay_ms = max(1, base_delay - elapsed_ms)
        else:
            delay_ms = base_delay

        self.root.after(delay_ms, self.update_frame)
    
    def setup_ui(self):
        """Create the UI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Exercise selection panel
        exercise_frame = ttk.LabelFrame(main_frame, text="Exercise Configuration", padding="10")
        exercise_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Video source row (Phase 5.H)
        ttk.Label(exercise_frame, text="Source:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(exercise_frame, text="Camera", variable=self.video_mode, value="camera", 
                       command=self.on_video_mode_change).grid(row=0, column=1, padx=2)
        ttk.Radiobutton(exercise_frame, text="File", variable=self.video_mode, value="file",
                       command=self.on_video_mode_change).grid(row=0, column=2, padx=2)
        self.browse_btn = ttk.Button(exercise_frame, text="Browse...", command=self.browse_video_file, 
                                     state=tk.DISABLED, width=10)
        self.browse_btn.grid(row=0, column=3, padx=5)
        self.file_label = ttk.Label(exercise_frame, text="", foreground="#666666", width=20)
        self.file_label.grid(row=0, column=4, padx=5, sticky=tk.W)
        
        # Flip video checkbox (for upside-down phone videos)
        self.flip_checkbox = ttk.Checkbutton(exercise_frame, text="Flip Video", variable=self.flip_video)
        self.flip_checkbox.grid(row=0, column=5, padx=5)
        
        # Participant ID input (Phase 5.A)
        ttk.Label(exercise_frame, text="Participant ID:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.participant_entry = ttk.Entry(exercise_frame, textvariable=self.participant_id, width=15)
        self.participant_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(exercise_frame, text="Exercise:", font=("Arial", 10, "bold")).grid(row=1, column=3, sticky=tk.W, padx=5)
        
        self.exercise_menu = ttk.Combobox(exercise_frame, textvariable=self.current_exercise, 
                                      values=["Abduction", "Flexion"], state="readonly", width=12)
        self.exercise_menu.grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        self.exercise_menu.bind("<<ComboboxSelected>>", self.on_exercise_change)
        
        # Affected side selector (MISSING-1)
        ttk.Label(exercise_frame, text="Affected Side:", font=("Arial", 10, "bold")).grid(row=1, column=5, sticky=tk.W, padx=(15, 5))
        self.side_menu = ttk.Combobox(exercise_frame, textvariable=self.affected_side,
                                      values=["Right", "Left"], state="readonly", width=8)
        self.side_menu.grid(row=1, column=6, padx=5, pady=5, sticky=tk.W)
        self.side_menu.bind("<<ComboboxSelected>>", self.on_side_change)
        
        # Camera positioning instructions (dynamic based on exercise + affected side)
        self.instruction_label = ttk.Label(exercise_frame, 
                                           text=self._get_camera_instruction(),
                                           font=("Arial", 10), foreground="#0066cc")
        self.instruction_label.grid(row=2, column=0, columnspan=7, padx=5, pady=(5,0), sticky=tk.W)
        
        # (Session notes UI removed per fix task B)
        
        # Video display area - use tk.Label (not ttk) to control background color
        self.video_label = tk.Label(main_frame, text="📹 Video will appear here\nClick 'Start' to begin session", 
                                     font=("Arial", 14), anchor="center", justify="center",
                                     bg="#f0f0f0")  # Match window background to avoid black bars
        self.video_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        # Info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # FPS display
        self.fps_label = ttk.Label(info_frame, text="FPS: 0.0", font=("Arial", 12))
        self.fps_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        # Flexion angle display
        self.flexion_label = ttk.Label(info_frame, text="Flexion: --", font=("Arial", 12))
        self.flexion_label.grid(row=0, column=1, padx=20)
        
        # Abduction angle display
        self.abduction_label = ttk.Label(info_frame, text="Abduction: --", font=("Arial", 12))
        self.abduction_label.grid(row=0, column=2, padx=20)
        
        # Rep counter display
        self.rep_label = ttk.Label(info_frame, text="Reps: 0", font=("Arial", 14, "bold"), foreground="#006600")
        self.rep_label.grid(row=0, column=3, padx=20)
        
        # Set tracker display (Phase 4 enhancement)
        self.set_label = ttk.Label(
            info_frame,
            text=f"Set: 1 of {self.total_sets}",
            font=("Arial", 12, "bold"),
            foreground="#0066cc"
        )
        self.set_label.grid(row=0, column=4, padx=20)
        
        # Fatigue level display (Phase 4)
        self.fatigue_label = ttk.Label(info_frame, text="Deterioration: Low", font=("Arial", 12, "bold"), foreground="#006600")
        self.fatigue_label.grid(row=0, column=5, padx=20)
        
        # Compensation warnings panel (Phase 3.4-3.5)
        # Consolidated into single label to avoid overlap and redundancy
        warning_frame = ttk.Frame(main_frame)
        warning_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
        
        # Single combined warning label (cleaner UI)
        self.compensation_warning_label = ttk.Label(warning_frame, text="", font=("Arial", 11, "bold"), foreground="#CC6600", wraplength=1000)
        self.compensation_warning_label.grid(row=0, column=0, padx=5, sticky=tk.W)

        # BLC tracking warning label (Fix A: on-screen BLC indicator)
        self.blc_warning_label = ttk.Label(warning_frame, text="", font=("Arial", 11, "bold"), foreground="#CC0000")
        self.blc_warning_label.grid(row=0, column=1, padx=15, sticky=tk.W)
        
        # Keep old labels for backward compatibility (hidden)
        self.trunk_lean_label = self.compensation_warning_label
        self.shoulder_hiking_label = ttk.Label(warning_frame, text="", font=("Arial", 1))
        self.shoulder_hiking_label.grid_forget()  # Hide completely
        
        # Fatigue/Break status panel (Phase 4)
        fatigue_frame = ttk.Frame(main_frame)
        fatigue_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Break prompt label
        self.break_label = ttk.Label(fatigue_frame, text="", font=("Arial", 12, "bold"), foreground="#CC0000")
        self.break_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        # Safety message (always visible during exercise)
        self.safety_label = ttk.Label(fatigue_frame, text="", font=("Arial", 10), foreground="#666666")
        self.safety_label.grid(row=0, column=1, padx=20, sticky=tk.W)
        
        # Status label (moved to row 5 to eliminate gap)
        self.status_label = ttk.Label(main_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Control buttons (adjusted to row 6 after status label move)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=15, padx=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_video)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Set navigation buttons
        ttk.Label(button_frame, text="|", font=("Arial", 12)).grid(row=0, column=3, padx=15)
        
        self.prev_set_btn = ttk.Button(button_frame, text="◀ Prev Set", command=self.previous_set, width=10, state=tk.DISABLED)
        self.prev_set_btn.grid(row=0, column=4, padx=5)
        
        self.next_set_btn = ttk.Button(button_frame, text="Next Set ▶", command=self.next_set, width=10, state=tk.DISABLED)
        self.next_set_btn.grid(row=0, column=5, padx=5)
        
        # Options row (row 7): audio/recording/view toggles and utility actions
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=7, column=0, columnspan=3, pady=5, padx=10)

        self.audio_cb = ttk.Checkbutton(options_frame, text="🔊 Audio", variable=self.audio_enabled)
        self.audio_cb.grid(row=0, column=0, padx=5)

        self.record_annotated_cb = ttk.Checkbutton(options_frame, text="🎥 Rec Annotated", variable=self.record_annotated)
        self.record_annotated_cb.grid(row=0, column=1, padx=5, sticky=tk.W)

        self.record_raw_cb = ttk.Checkbutton(options_frame, text="🎬 Rec Raw", variable=self.record_raw)
        self.record_raw_cb.grid(row=0, column=2, padx=5, sticky=tk.W)

        self.recording_status_label = ttk.Label(options_frame, text="", font=("Arial", 10, "bold"), foreground="#CC0000")
        self.recording_status_label.grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(options_frame, text="|", font=("Arial", 12)).grid(row=0, column=4, padx=10)

        # Smoothed angle overlay toggle (mirror display is always-on)
        self.smooth_cb = ttk.Checkbutton(options_frame, text="Smooth", variable=self.show_smoothed_overlay)
        self.smooth_cb.grid(row=0, column=5, padx=5, sticky=tk.W)

        # Exercise guide quick-view button (Phase 10)
        self.guide_btn = ttk.Button(
            options_frame,
            text="Guide",
            command=lambda: self.show_exercise_guide(force=True),
            width=7
        )
        self.guide_btn.grid(row=0, column=6, padx=5)

        ttk.Label(options_frame, text="|", font=("Arial", 12)).grid(row=0, column=7, padx=10)

        self.reset_btn = ttk.Button(options_frame, text="Reset", command=self.reset_session, width=8)
        self.reset_btn.grid(row=0, column=8, padx=5)

        # Graph visualization button
        self.graph_btn = ttk.Button(options_frame, text="📊 Graph", command=self.show_rep_graph, width=8)
        self.graph_btn.grid(row=0, column=9, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
    
    def on_exercise_change(self, event=None):
        """Handle exercise type change.

        Phase 5 policy: one exercise type per active session.
        Switching exercise while a session is running is blocked to prevent
        cross-view baseline reuse (abduction=frontal, flexion=lateral).
        """
        exercise = self.current_exercise.get()

        # Phase 5: block exercise switching for active sessions.
        if self.is_running and self.last_exercise and self.last_exercise != exercise:
            self.current_exercise.set(self.last_exercise)
            self.status_label.config(
                text="Status: Changing exercise requires stopping the current session and starting a new calibrated session.",
                foreground="#CC6600"
            )
            return
        
        # Store current exercise for change detection
        self.last_exercise = exercise
        
        # Update camera positioning instructions (side-aware)
        self.instruction_label.config(text=self._get_camera_instruction())
        
        # Update pose processor camera view
        camera_view = 'frontal' if exercise == "Abduction" else 'lateral'
        self.pose_processor.set_camera_view(camera_view)
        
        # Reset rep tracker when exercise changes
        self.rep_tracker.reset()
        self.rep_tracker.set_exercise(exercise)  # Set exercise-specific thresholds
        self.rep_label.config(text="Reps: 0", foreground="#006600")  # Reset color
        
        # Reset fatigue module (Phase 4)
        self.fatigue_module.reset()
        # Note: Don't clear baseline_log - exercise_block key prevents collision
        self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
        self.break_label.config(text="")
        self.micro_break_active = False
        
        # Reset compensation tracking
        self._reset_current_rep_compensation()
        self.trunk_lean_label.config(text="")
        self.shoulder_hiking_label.config(text="")
        
        # Update window title
        self.root.title(f"Shoulder Mobility Assessment System - {exercise} ({self.affected_side.get()} side)")
        
        # Update status
        self.status_label.config(text=f"Status: {exercise} selected - Position camera as instructed")

        # Phase 10: Show guide on exercise change when session is not running.
        # (If active-session switching is blocked, this still gives pre-session guidance.)
        if event is not None and not self.is_running:
            self.show_exercise_guide(exercise=exercise)

    def _get_exercise_guide_text(self, exercise):
        """Return short textual exercise guide lines."""
        if exercise == "Abduction":
            return [
                "Frontal view required",
                "Raise arm sideways",
                "Keep trunk upright",
            ]

        return [
            "Side/lateral view required",
            "Raise arm forward",
            "Keep trunk upright",
        ]

    def show_exercise_guide(self, exercise=None, force=False):
        """Show exercise guide popup once per exercise per session unless forced."""
        exercise_name = exercise or self.current_exercise.get()
        if not force and exercise_name in self.guides_shown_in_session:
            return

        guide_lines = self._get_exercise_guide_text(exercise_name)
        guide_text = "\n".join(f"- {line}" for line in guide_lines)

        expected_abduction = os.path.join(self.guides_assets_dir, 'abduction.png')
        expected_flexion = os.path.join(self.guides_assets_dir, 'flexion.png')
        image_path = os.path.join(self.guides_assets_dir, f"{exercise_name.lower()}.png")

        try:
            guide_window = tk.Toplevel(self.root)
            guide_window.title(f"{exercise_name} Exercise Guide")
            guide_window.transient(self.root)
            guide_window.resizable(False, False)

            content = ttk.Frame(guide_window, padding=12)
            content.grid(row=0, column=0, sticky="nsew")

            ttk.Label(
                content,
                text=f"{exercise_name} Guide",
                font=("Arial", 12, "bold")
            ).grid(row=0, column=0, sticky="w", pady=(0, 6))

            ttk.Label(
                content,
                text=guide_text,
                justify=tk.LEFT,
                font=("Arial", 10)
            ).grid(row=1, column=0, sticky="w", pady=(0, 8))

            image_loaded = False
            if os.path.isfile(image_path):
                pil_available = 'Image' in globals() and 'ImageTk' in globals()
                if pil_available:
                    try:
                        resampling_module = getattr(Image, "Resampling", Image)
                        resample = getattr(resampling_module, "LANCZOS", getattr(Image, "LANCZOS"))
                        with Image.open(image_path) as img_raw:
                            guide_img = img_raw.copy()
                        guide_img.thumbnail((760, 420), resample)
                        guide_photo = ImageTk.PhotoImage(guide_img)

                        image_label = ttk.Label(content, image=guide_photo)
                        setattr(image_label, "image", guide_photo)
                        image_label.grid(row=2, column=0, sticky="w", pady=(0, 8))
                        image_loaded = True
                    except Exception:
                        image_loaded = False

                if not image_loaded:
                    ttk.Label(
                        content,
                        text=(
                            f"Guide image found but could not be displayed:\n{image_path}\n"
                            "Using text-only guide."
                        ),
                        justify=tk.LEFT,
                        foreground="#CC6600",
                        wraplength=760
                    ).grid(row=2, column=0, sticky="w", pady=(0, 8))
            else:
                ttk.Label(
                    content,
                    text=(
                        "Guide image not found. Place guide images at:\n"
                        f"- {expected_abduction}\n"
                        f"- {expected_flexion}"
                    ),
                    justify=tk.LEFT,
                    foreground="#CC6600",
                    wraplength=760
                ).grid(row=2, column=0, sticky="w", pady=(0, 8))

            def _close_guide():
                try:
                    guide_window.grab_release()
                except tk.TclError:
                    pass
                guide_window.destroy()

            close_btn = ttk.Button(content, text="Close", command=_close_guide, width=10)
            close_btn.grid(row=3, column=0, sticky="e")

            guide_window.protocol("WM_DELETE_WINDOW", _close_guide)
            guide_window.bind("<Escape>", lambda _e: _close_guide())
            close_btn.focus_set()

            guide_window.grab_set()
            self.root.wait_window(guide_window)
        except tk.TclError:
            # Safe fallback if Toplevel cannot be created.
            messagebox.showinfo(
                f"{exercise_name} Guide",
                (
                    f"{guide_text}\n\n"
                    "Guide images expected at:\n"
                    f"- {expected_abduction}\n"
                    f"- {expected_flexion}"
                )
            )

        if not force:
            self.guides_shown_in_session.add(exercise_name)

    def _get_camera_instruction(self):
        """Generate camera positioning instruction based on current exercise and affected side.
        
        MISSING-1: Dynamic instruction text ensures the user always sees correct
        guidance for their selected affected side.
        
        Returns:
            str: Instruction text for the current exercise+side combination
        """
        side = self.affected_side.get()  # 'Right' or 'Left'
        exercise = self.current_exercise.get()
        
        if exercise == "Abduction":
            return (f"📹 Stand FACING camera (frontal view). "
                    f"Raise your {side.upper()} arm only. "
                    f"Keep both shoulders visible.")
        else:  # Flexion
            return (f"📹 Stand with your {side.upper()} SIDE toward camera (lateral view). "
                    f"Raise your {side.upper()} arm only.")
    
    def on_side_change(self, event=None):
        """Handle affected side change (MISSING-1).
        
        Propagates the selected side to PoseProcessor and updates all
        side-dependent UI elements. Resets rep tracking since landmark
        mapping changes.
        """
        side = self.affected_side.get()
        
        # Propagate to pose processor (rebuilds landmark sets)
        self.pose_processor.set_affected_side(side)
        
        # Update camera instructions
        self.instruction_label.config(text=self._get_camera_instruction())
        
        # Reset rep tracker (landmark mapping changed — stale state invalid)
        self.rep_tracker.reset()
        self.rep_label.config(text="Reps: 0", foreground="#006600")
        
        # Reset fatigue module
        self.fatigue_module.reset()
        self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
        self.break_label.config(text="")
        self.micro_break_active = False
        
        # Reset compensation
        self._reset_current_rep_compensation()
        self.compensation_warning_label.config(text="")
        
        # Update window title
        exercise = self.current_exercise.get()
        self.root.title(f"Shoulder Mobility Assessment System - {exercise} ({side} side)")
        
        self.status_label.config(text=f"Status: Affected side set to {side}")
        print(f"[Main] Affected side changed to: {side}")
    
    def on_video_mode_change(self):
        """Handle video mode toggle (Phase 5.H)"""
        if self.video_mode.get() == "file":
            self.browse_btn.config(state=tk.NORMAL)
        else:
            self.browse_btn.config(state=tk.DISABLED)
            self.file_label.config(text="")
            self.video_file_path = None
    
    def browse_video_file(self):
        """Open file browser to select video (Phase 5.H)"""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), "data", "videos")
        )
        if path:
            self.video_file_path = path
            name = os.path.basename(path)
            self.file_label.config(text=name[:25] + "..." if len(name) > 25 else name, foreground="#006600")
    
    def play_sound(self, sound_type='success', audio_filename=None):
        """
        Play audio feedback on rep completion (Phase 5.I) - Non-blocking with debounce.
        
        Supports three fallback layers:
        1. Local WAV files from assets/audio/ (if audio_filename provided or standard names exist)
        2. System beeps via winsound (Windows only)
        3. Silent if both fail (graceful degradation)
        
        Args:
            sound_type: 'success' or 'error' (used for system beep fallback)
            audio_filename: Optional .wav filename to load from assets/audio/ (e.g., 'ding.wav')
        """
        if not self.audio_enabled.get():
            return
        
        # Debounce: skip if beep was played very recently (within 200ms)
        current_time = time.time()
        if current_time - self.last_beep_time < 0.2:
            return
        self.last_beep_time = current_time
        
        def _play_audio():
            """Non-blocking audio playback thread target"""
            if not self.audio_lock.acquire(blocking=False):
                return  # Skip if locked (previous beep still playing)
            
            try:
                # Layer 1: Try local WAV file playback
                if AUDIO_SCIPY_AVAILABLE and wavfile is not None and sd is not None and os.path.isdir(self.audio_assets_dir):
                    # Determine filename to try
                    wav_file = None
                    if audio_filename:
                        wav_file = audio_filename if audio_filename.endswith('.wav') else f"{audio_filename}.wav"
                    else:
                        # Default naming: "success.wav" or "error.wav"
                        wav_file = f"{sound_type}.wav"
                    
                    wav_path = os.path.join(self.audio_assets_dir, wav_file)
                    
                    # Try to load and play local WAV file
                    if os.path.isfile(wav_path):
                        try:
                            # Check cache first to avoid repeated I/O
                            if wav_path not in self.audio_cache:
                                samplerate, data = wavfile.read(wav_path)
                                self.audio_cache[wav_path] = (samplerate, data)
                            else:
                                samplerate, data = self.audio_cache[wav_path]
                            
                            # Play audio (blocking, so thread usage is essential)
                            sd.play(data, samplerate)
                            sd.wait()  # Wait for playback to finish
                            return  # Success - exit early
                        except Exception as e:
                            print(f"[Audio] Error playing local WAV {wav_path}: {e}")
                            # Fall through to system beep
                
                # Layer 2: Fall back to system beep (Windows only)
                if winsound is not None:
                    freq = 1000 if sound_type == 'success' else 400
                    dur = 150 if sound_type == 'success' else 250
                    try:
                        winsound.Beep(freq, dur)
                        return  # Success - exit early
                    except Exception as e:
                        print(f"[Audio] Error playing system beep: {e}")
                
                # Layer 3: Silent fallback (both layers failed)
                print(f"[Audio] No audio available for '{sound_type}' (no local files and no winsound)")
                
            except Exception as e:
                print(f"[Audio] Unexpected error in audio playback: {e}")
            finally:
                self.audio_lock.release()
        
        # Start non-blocking playback thread
        try:
            t = threading.Thread(target=_play_audio, daemon=True)
            t.start()
        except Exception as e:
            print(f"[Audio] Failed to start audio thread: {e}")

    def play_cue(self, cue_key, cooldown_sec=0.0):
        """Play a named audio cue with cooldown-based debounce."""
        if cue_key not in self.audio_cue_registry:
            print(f"[Audio] Unknown cue key: {cue_key}")
            return

        if cooldown_sec > 0:
            last_played = self.audio_cooldowns.get(cue_key, 0)
            if time.time() - last_played < cooldown_sec:
                return

        self.audio_cooldowns[cue_key] = time.time()
        relative_path = self.audio_cue_registry[cue_key]
        self.play_sound(audio_filename=relative_path)

    def _reset_current_rep_compensation(self):
        """Reset all current-rep compensation tracking flags (BUG-4 fix).
        
        Centralizes compensation state cleanup to prevent stale flags from
        leaking across set boundaries, exercise switches, or session resets.
        """
        self.current_rep_trunk_lean = False
        self.current_rep_shoulder_hiking = False
        self.current_rep_low_confidence = False
    
    def _handle_set_completion(self):
        """Handle automatic set-completion detection (MISSING-5).
        
        Called when rep count reaches reps_per_set target. Shows a non-blocking
        confirmation dialog to advance to the next set or continue.
        Uses root.after() to defer execution outside the frame loop.
        """
        # Pause tracking immediately at set completion to prevent extra reps
        # from being processed while the prompt is displayed.
        if self.is_running and not self.is_paused:
            self.toggle_pause()
            self.status_label.config(
                text=f"Status: Set {self.current_set} complete - awaiting next set decision",
                foreground="#0066cc"
            )

        def _prompt():
            if self.current_set < self.total_sets:
                response = messagebox.askyesno(
                    "Set Complete",
                    f"🎯 Set {self.current_set} complete!\n"
                    f"{self.rep_tracker.get_rep_count()} of {self.reps_per_set} reps done.\n\n"
                    f"Advance to Set {self.current_set + 1}?\n\n"
                    f"Yes = Next set\n"
                    f"No = Continue in current set"
                )
                if response:
                    self.play_cue('set_proceed', cooldown_sec=5.0)
                    self.next_set(auto_from_completion=True)
                else:
                    # Continue current set safely without re-prompting this set.
                    if self.is_running and self.is_paused:
                        self.toggle_pause()
                        self.status_label.config(
                            text=f"Status: Running - Continue Set {self.current_set}",
                            foreground="#000000"
                        )
            else:
                messagebox.showinfo(
                    "All Sets Complete",
                    f"🏆 Set {self.current_set} of {self.total_sets} complete!\n"
                    f"{self.rep_tracker.get_rep_count()} reps done.\n\n"
                    f"All sets finished. You can stop or continue exercising."
                )
                self.play_cue('all_sets_done', cooldown_sec=5.0)
                self.status_label.config(
                    text="Status: All sets complete - Session paused",
                    foreground="#0066cc"
                )
        
        # Defer to avoid blocking the frame loop
        self.root.after(100, _prompt)
    
    def perform_calibration(self, landmarks):
        """
        Perform tracking readiness check during session start (T11).
        
        Two-phase gated calibration:
        Phase A (waiting_for_detection): Wait for pose detection + confidence ≥ 0.5
        Phase B (countdown): Collect valid frames for calibration_duration_sec, build baseline
        
        Thesis §3.5: Runs for ~10 seconds (configurable) while user holds neutral pose.
        Validates landmark stability and records baseline limb length for BLC.
        
        Args:
            landmarks: dict from pose_processor.process_frame()
            
        Returns:
            tuple: (calibration_complete, passed)
                - calibration_complete: bool (calibration_duration_sec elapsed)
                - passed: bool (tracking was stable)
        """
        if not self.calibration_phase_active:
            return False, False
        
        # ===== PHASE A: WAITING FOR DETECTION =====
        if self.calibration_waiting_for_detection:
            # Check if we have critical landmarks with adequate confidence
            landmark_check = self._check_critical_landmarks_for_calibration(landmarks)
            
            if landmark_check['valid']:
                # Detection successful - transition to countdown phase
                self.calibration_waiting_for_detection = False
                self.calibration_countdown_started = True
                self.calibration_start_time = time.time()
                self.calibration_landmarks_buffer = []
                self.calibration_segment_lengths = []
                # Fall through to Phase B on same frame
            else:
                # Phase A timeout guard: stop session if no pose is detected for too long.
                waiting_start = self.calibration_waiting_start_time or time.time()
                self.calibration_waiting_start_time = waiting_start
                waiting_elapsed = time.time() - waiting_start

                if waiting_elapsed >= 60.0:
                    print("[Calibration] TIMEOUT: No valid pose detected for 60s during Phase A")

                    self.calibration_phase_active = False
                    self.calibration_waiting_for_detection = False
                    self.calibration_countdown_started = False
                    self.require_calibration_pass = False

                    try:
                        messagebox.showwarning(
                            "Calibration Timeout",
                            "No valid pose was detected for 60 seconds during calibration.\n"
                            "The session will now stop."
                        )
                    except Exception:
                        pass

                    self.stop_video()
                    self.status_label.config(
                        text="Status: Calibration timeout - no pose detected (session stopped)",
                        foreground="#CC0000"
                    )

                # Still waiting - show detection UI
                return False, None  # Not complete, still waiting
        
        # ===== PHASE B: COUNTDOWN / COLLECTION =====
        if self.calibration_countdown_started:
            elapsed = time.time() - (self.calibration_start_time or time.time())
            
            # Only buffer valid frames (confidence ≥ threshold + critical landmarks present)
            if landmarks:
                landmark_check = self._check_critical_landmarks_for_calibration(landmarks)
                if landmark_check['valid']:
                    self.calibration_landmarks_buffer.append(landmarks)
                    # Also collect segment length for baseline computation
                    if landmark_check['segment_length'] is not None:
                        self.calibration_segment_lengths.append(landmark_check['segment_length'])
            
            # Check if countdown complete (duration from config)
            if elapsed >= self.calibration_duration_sec:
                # Calibration window finished - validate stability
                passed = self._validate_calibration_stability()
                return True, passed
            
            # Still calibrating - return incomplete
            return False, None
        
        # Should not reach here
        return False, None
    
    def _check_critical_landmarks_for_calibration(self, landmarks):
        """
        Check if critical landmarks are present and visible for calibration gate.
        
        Validates that TRACKED_SHOULDER and TRACKED_ELBOW are present with 
        visibility ≥ confidence_threshold. Also computes segment length for baseline.
        
        Args:
            landmarks: dict from pose_processor.process_frame()
        
        Returns:
            dict with keys:
                - 'valid': bool (landmarks present and confident)
                - 'segment_length': float or None (shoulder-elbow distance in px)
        """
        shoulder_key = 'TRACKED_SHOULDER'
        elbow_key = 'TRACKED_ELBOW'
        
        # Check if critical landmarks exist
        if shoulder_key not in landmarks or elbow_key not in landmarks:
            return {'valid': False, 'segment_length': None}
        
        sx, sy, s_vis = landmarks[shoulder_key]
        ex, ey, e_vis = landmarks[elbow_key]
        
        # Check confidence threshold
        if s_vis < self.calibration_detection_confidence_threshold or \
           e_vis < self.calibration_detection_confidence_threshold:
            return {'valid': False, 'segment_length': None}
        
        # Compute segment length (shoulder to elbow)
        try:
            segment_length = np.sqrt((ex - sx)**2 + (ey - sy)**2)
            return {'valid': True, 'segment_length': float(segment_length)}
        except (TypeError, ValueError):
            return {'valid': False, 'segment_length': None}
    
    def _validate_calibration_stability(self):
        """
        Validate that landmarks were stable during calibration.
        Checks for low jitter and consistent visibility.
        Computes baseline limb length if passing.
        
        Uses median of buffered segment lengths for baseline (robust to outliers).
        
        Returns:
            bool: True if calibration passes (tracking stable), False otherwise
        """
        if len(self.calibration_landmarks_buffer) < 10:
            # Not enough frames to validate
            return False

        # Use only the final stable window (last ~3 seconds at 30 FPS = 90 frames)
        # to avoid penalizing early walk-in movement during calibration countdown
        recent_buffer = self.calibration_landmarks_buffer[-90:]
        
        # Extract a key landmark across frames for stability check
        # Use the exercise-specific tracked shoulder
        exercise = self.current_exercise.get()
        shoulder_key = 'TRACKED_SHOULDER'
        
        # Collect shoulder positions from buffer
        shoulder_positions = []
        visibility_scores = []
        for landmarks_dict in recent_buffer:
            if shoulder_key in landmarks_dict:
                x, y, vis = landmarks_dict[shoulder_key]
                shoulder_positions.append((x, y))
                visibility_scores.append(vis)
        
        if len(shoulder_positions) < 10:
            # Not enough valid observations
            return False
        
        # Check visibility: average should be > 0.5
        avg_visibility = np.mean(visibility_scores)
        if avg_visibility < 0.5:
            return False
        
        # Check stability: standard deviation of positions should be small
        positions_array = np.array(shoulder_positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        
        # Jitter threshold: ~5 pixels standard deviation is acceptable
        # High jitter (>15 px) indicates unstable tracking
        max_jitter = self.config.calibration_jitter_threshold_px if self.config else 40.0
        if x_std > max_jitter or y_std > max_jitter:
            return False
        
        # Calibration passed - record baseline using median of collected segment lengths
        # This is more robust than using first frame (filters outliers)
        if self._record_baseline_limb_length_from_window():
            self.calibration_data = {
                'avg_visibility': float(avg_visibility),
                'shoulder_x_std': float(x_std),
                'shoulder_y_std': float(y_std),
                'frame_count': len(self.calibration_landmarks_buffer),
                'segment_length_samples': len(self.calibration_segment_lengths),
            }
            return True
        
        return False
    
    def _record_baseline_limb_length_from_window(self):
        """
        Record baseline limb length from calibration window using median.
        
        Uses median of all collected segment lengths (shoulder-elbow distance)
        during the calibration countdown. Median is robust to outliers and provides
        a representative stable baseline for BLC filter.
        
        Returns:
            bool: True if baseline recorded successfully
        """
        if len(self.calibration_segment_lengths) < 5:
            # Not enough valid segment samples
            return False

        # Use only the final stable window (last ~3 seconds at 30 FPS = 90 frames)
        # to avoid skewing baseline with frames from walking into position
        recent_segment_lengths = self.calibration_segment_lengths[-90:]
        
        try:
            # Use median for robustness (outlier-resistant)
            import statistics
            baseline_length = statistics.median(recent_segment_lengths)
        except (ValueError, TypeError, Exception) as e:
            print(f"[Calibration] Error computing median segment length: {e}")
            return False
        
        if not (0 < baseline_length < 10000):  # Sanity check: reasonable px range
            return False
        
        # Store baseline (pixels in current video)
        self.baseline_limb_length = {
            'segment': 'upperarm',
            'shoulder_elbow_px': float(baseline_length),
            'exercise': self.current_exercise.get(),
            'method': 'median_window',  # Mark that we used median, not first frame
            'samples': len(recent_segment_lengths),
        }
        
        # Pass baseline to SpatialTemporalFilter for BLC frame-level checks (T2/T13)
        self.spatial_temporal_filter.set_blc_baseline(float(baseline_length))
        
        return True
    
    def reset_session(self):
        """Reset session data without stopping video (Phase 5.K)"""
        # Reset all tracking
        self.rep_tracker.reset()
        self.fatigue_module.reset()
        self.spatial_temporal_filter.reset()  # Clear filter state for fresh smoothing
        self.session_log = []
        self.current_set = 1
        self.exercise_block = 1
        self.frame_idx = 0
        self.baseline_log = {}
        self._reset_current_rep_compensation()
        self.set_completed_prompted = False  # MISSING-5: Reset set-completion guard
        self.last_rep_sound_key = None
        self.rep_flash_active = False
        self.rep_flash_color = None
        self.rep_flash_start = 0
        self.guides_shown_in_session = set()
        
        # Reset calibration state (T11: Tracking Readiness Check)
        self.calibration_phase_active = False
        self.calibration_waiting_for_detection = False
        self.calibration_countdown_started = False
        self.calibration_start_time = None
        self.calibration_waiting_start_time = None
        self.calibration_landmarks_buffer = []
        self.calibration_segment_lengths = []
        self.baseline_limb_length = {}
        self._last_valid_landmarks = None
        self._last_valid_results = None
        self._pose_gap_frames = 0
        self.calibration_data = {}
        self.calibration_passed = False
        self.require_calibration_pass = False
        self._countdown_last_tone = 0  # Reset countdown tone tracker
        self._calibration_retry_count = 0  # Reset strict calibration retry counter
        
        # Reset FPS statistics for new session
        self.fps_samples = []
        self.last_fps_sample_time = time.time()
        self._recording_frame_count = 0
        self._recording_start_time = None
        self._effective_recording_fps = None
        
        # Reset UI displays
        self.rep_label.config(text="Reps: 0", foreground="#006600")
        self.set_label.config(text=f"Set: 1 of {self.total_sets}")
        self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
        self.break_label.config(text="")
        self.compensation_warning_label.config(text="")
        # Micro-break is ADVISORY (decision-support): PT/researcher supervises enforcement.
        # The system recommends/rest-counts, but does not hard-block rep counting.
        self.micro_break_active = False
        self._recording_initialized = False
        self.recording_paths = {}

        if self.is_running:
            self.status_label.config(text="Status: Session reset - Continue exercising")
        else:
            self.status_label.config(text="Status: Session reset - Ready to start")
    
    def toggle_pause(self):
        """Pause/Resume session (keeps video capture open)"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.config(text="Resume")
            self.status_label.config(text="Status: PAUSED - Click Resume to continue")
            # Track when pause started (for break timer)
            self.pause_start_time = time.time()
        else:
            self.pause_btn.config(text="Pause")
            self.status_label.config(text="Status: Running")
            # Accumulate pause duration for break timer
            if hasattr(self, 'pause_start_time') and self.pause_start_time:
                self.break_pause_accumulated += time.time() - self.pause_start_time
                self.pause_start_time = None
            # Clear stale smoothing buffer on resume (BUG-5 fix):
            # Old angles from before the pause can influence threshold detection
            # and cause false rep starts when motion resumes.
            self.rep_tracker.angle_buffer.clear()
            # Resume frame updates
            self.last_frame_time = time.time()  # Reset FPS timer
            self.update_frame()
    
    def show_rep_graph(self):
        """Show comprehensive, easy-to-read session progress graph with multi-set support"""
        if not self.session_log:
            messagebox.showinfo("No Data", "No reps recorded yet. Complete some reps first.")
            return
        
        # Group data by exercise block AND set (supports mid-session exercise switching)
        sets_data = {}
        for entry in self.session_log:
            # Key: (exercise_block, exercise_name, set_number) for unique identification
            block = entry.get('exercise_block', 1)
            exercise = entry.get('exercise', 'Unknown')
            set_num = entry.get('set_number', 1)
            key = (block, exercise, set_num)
            if key not in sets_data:
                sets_data[key] = []
            sets_data[key].append(entry)
        
        num_sets = len(sets_data)
        
        # Create popup window with scrollable canvas
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Session Progress")
        graph_window.geometry("1100x800")
        
        # Create main container with scrollbar
        main_container = ttk.Frame(graph_window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        canvas_scroll = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        # Center the content by creating window at center
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="n")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel for scrolling
        def on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", on_mousewheel)
        
        # Unbind mousewheel when window closes
        def on_close():
            canvas_scroll.unbind_all("<MouseWheel>")
            graph_window.destroy()
        graph_window.protocol("WM_DELETE_WINDOW", on_close)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Fixed height per set for consistent, readable graphs
        height_per_set = 5.5  # inches per set
        fig_height = max(7, num_sets * height_per_set)
        
        if num_sets == 1:
            fig, axes = plt.subplots(1, 1, figsize=(13, 7))
            axes = [axes]
        else:
            fig, axes = plt.subplots(num_sets, 1, figsize=(13, fig_height))
            if num_sets == 1:
                axes = [axes]
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import numpy as np
        
        # Fatigue colors with HIGHER OPACITY for visibility
        fatigue_colors = {
            'Low': ('#A5D6A7', 0.5),      # Light green
            'Medium': ('#FFE082', 0.6),    # Light amber  
            'High': ('#EF9A9A', 0.7)       # Light red
        }
        
        # Process each set
        for set_idx, (key, set_entries) in enumerate(sorted(sets_data.items())):
            ax = axes[set_idx]
            
            # Unpack key: (exercise_block, exercise_name, set_number)
            block, exercise_name, set_num = key
            
            # Extract data for this set
            reps = list(range(1, len(set_entries) + 1))  # 1-based rep numbers within set
            peak_angles = [r['peak_angle'] for r in set_entries]
            rom_labels = [r['rom_label'] for r in set_entries]
            fatigue_levels = [r['fatigue_level'] for r in set_entries]
            trunk_lean = [r['trunk_lean_detected'] for r in set_entries]
            shoulder_hiking = [r['shoulder_hiking_detected'] for r in set_entries]
            durations = [r['duration_frames'] for r in set_entries]
            
            # ===== GET CORRECT PER-SET BASELINE =====
            # Use stored baseline_log for this specific set (computed from reps 2-4 of this set)
            # Key is (exercise_block, set_number) to support mid-session exercise switching
            baseline_rom = None
            baseline_dur = None
            baseline_low_conf = False
            
            baseline_key = (block, set_num)
            if baseline_key in self.baseline_log:
                baseline_info = self.baseline_log[baseline_key]
                baseline_rom = baseline_info.get('baseline_rom')
                baseline_dur = baseline_info.get('baseline_dur')
                baseline_low_conf = baseline_info.get('low_confidence', False)
            elif block == self.exercise_block and set_num == self.current_set:
                # Fallback: if viewing current set, use live fatigue module values
                baseline_rom = self.fatigue_module.baseline_rom
                baseline_dur = self.fatigue_module.baseline_dur
                baseline_low_conf = self.fatigue_module.baseline_low_confidence
            
            # If still no baseline, compute from set's reps 2-4 directly
            # Uses rep_number field (same logic as fatigue_module.compute_baseline)
            if baseline_rom is None and len(set_entries) >= 2:
                # Get reps with rep_number 2, 3, or 4 from this set (skip rep 1 warm-up)
                baseline_candidates = [
                    e for e in set_entries 
                    if e.get('rep_number', 0) in [2, 3, 4]
                ]
                if len(baseline_candidates) >= 1:
                    from statistics import median
                    baseline_rom = median([e['peak_angle'] for e in baseline_candidates])
                    baseline_dur = median([e['duration_frames'] for e in baseline_candidates])
                    baseline_low_conf = len(baseline_candidates) < 3
            
            # Background: Fatigue level shading
            for i, (rep, fatigue) in enumerate(zip(reps, fatigue_levels)):
                color, alpha = fatigue_colors.get(fatigue, ('#FFFFFF', 0.0))
                ax.axvspan(rep - 0.45, rep + 0.45, color=color, alpha=alpha, zorder=0)
            
            # Main line: Peak angles
            ax.plot(reps, peak_angles, '-', color='#1976D2', linewidth=2.5, zorder=2)
            
            # Color markers by ROM classification
            for i, (rep, angle, rom) in enumerate(zip(reps, peak_angles, rom_labels)):
                color = '#2E7D32' if rom == 'correct' else '#C62828'
                ax.scatter(rep, angle, color=color, s=140, zorder=4, 
                          edgecolors='white', linewidth=2)
            
            # Add trend line if enough data points (shows ROM decline/improvement)
            z = None
            trend_color = '#4CAF50'
            if len(reps) >= 4:
                z = np.polyfit(reps, peak_angles, 1)
                p = np.poly1d(z)
                trend_color = '#4CAF50' if z[0] >= 0 else '#FF5722'  # Green if improving, orange if declining
                ax.plot(reps, p(reps), '--', color=trend_color, linewidth=1.5, alpha=0.7, zorder=1)
            
            # Reference lines
            ax.axhline(y=150, color='#1565C0', linestyle='--', linewidth=2, 
                      alpha=0.8, zorder=1)
            
            # Baseline ROM line (PER-SET baseline from reps 2-4)
            if baseline_rom:
                baseline_label = f'Baseline ({baseline_rom:.0f}°)'
                if baseline_low_conf:
                    baseline_label += ' *'  # Mark low confidence
                ax.axhline(y=baseline_rom, color='#7B1FA2', linestyle=':', 
                          linewidth=2.5, zorder=1)
            
            # ===== DURATION VISUALIZATION =====
            # Convert durations to seconds for readability
            durations_sec = [self.frames_to_seconds(d) for d in durations]
            baseline_dur_sec = self.frames_to_seconds(baseline_dur) if baseline_dur else None
            
            # Add duration annotations above each point (cleaner than bars)
            for i, (rep, angle, dur_sec) in enumerate(zip(reps, peak_angles, durations_sec)):
                # Show duration as text above the marker with background for visibility
                dur_color = '#006400' if baseline_dur_sec and dur_sec <= baseline_dur_sec * 1.2 else '#B71C1C'
                ax.annotate(f'{dur_sec:.1f}s', (rep, angle + 10), fontsize=9, 
                           ha='center', va='bottom', color=dur_color, fontweight='bold',
                           zorder=7,
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                    edgecolor='none', alpha=0.8))
            
            # Compensation markers at bottom - SEPARATE markers for each type
            y_bottom = max(0, min(peak_angles) - 25) if min(peak_angles) > 70 else 45
            
            has_trunk = False
            has_hiking = False
            has_both = False
            
            for i, rep in enumerate(reps):
                if trunk_lean[i] and shoulder_hiking[i]:
                    ax.scatter(rep, y_bottom, marker='D', s=180, color='#D32F2F', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_both = True
                elif trunk_lean[i]:
                    ax.scatter(rep, y_bottom, marker='v', s=180, color='#FF6F00', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_trunk = True
                elif shoulder_hiking[i]:
                    ax.scatter(rep, y_bottom, marker='^', s=180, color='#E65100', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_hiking = True
            
            # Styling
            ax.set_xlabel('Rep Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Peak Angle (°)', fontsize=11, fontweight='bold')
            
            # Title for each set with exercise name and average ROM
            avg_rom = sum(peak_angles) / len(peak_angles)
            set_title = f'{exercise_name} - Set {set_num}' if num_sets > 1 else f'{exercise_name} (Avg: {avg_rom:.0f}°)'
            if num_sets > 1:
                set_title += f'  (Avg: {avg_rom:.0f}°)'
            ax.set_title(set_title, fontsize=14, fontweight='bold', pad=20)
            
            ax.set_xticks(reps)
            ax.grid(axis='y', alpha=0.3, linestyle='-')
            # Extra top margin for duration labels (larger font now)
            ax.set_ylim(bottom=max(0, min(peak_angles) - 40), top=max(peak_angles) + 45)
            
            if len(reps) > 15:
                ax.tick_params(axis='x', labelsize=8)
            
            # Build legend elements dynamically
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markersize=10, 
                       markerfacecolor='#2E7D32', markeredgecolor='white', 
                       label='Correct ROM (≥150°)'),
                Line2D([0], [0], marker='o', color='w', markersize=10, 
                       markerfacecolor='#C62828', markeredgecolor='white', 
                       label='Insufficient ROM (<150°)'),
                Line2D([0], [0], color='#1565C0', linestyle='--', linewidth=2, 
                       label='Target ROM (150°)'),
            ]
            
            # Add baseline to legend if available
            if baseline_rom:
                conf_note = ' (low conf.)' if baseline_low_conf else ''
                legend_elements.append(
                    Line2D([0], [0], color='#7B1FA2', linestyle=':', linewidth=2.5, 
                           label=f'Baseline ROM ({baseline_rom:.0f}°){conf_note}')
                )
            
            # Trend line legend
            if z is not None and len(reps) >= 4:
                trend_label = 'Trend (improving)' if z[0] >= 0 else 'Trend (declining)'
                legend_elements.append(
                    Line2D([0], [0], color=trend_color, linestyle='--', linewidth=1.5, 
                           alpha=0.7, label=trend_label)
                )
            
            # Add fatigue level legend
            legend_elements.extend([
                Patch(facecolor='#A5D6A7', alpha=0.7, edgecolor='gray', 
                      label='Low Deter.'),
                Patch(facecolor='#FFE082', alpha=0.8, edgecolor='gray', 
                      label='Med Deter.'),
                Patch(facecolor='#EF9A9A', alpha=0.9, edgecolor='gray', 
                      label='High Deter.'),
            ])  # type: ignore[arg-type]
            
            # Duration annotation legend
            if baseline_dur_sec:
                legend_elements.append(
                    Line2D([0], [0], marker='', color='w', markersize=0,
                           label=f'Duration labels (baseline: {baseline_dur_sec:.1f}s)')
                )
            
            # Add compensation markers to legend (only if present in this set)
            if has_trunk or has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='v', color='w', markersize=10,
                           markerfacecolor='#FF6F00', markeredgecolor='black',
                           label='▼ Trunk Lean')
                )
            if has_hiking or has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='^', color='w', markersize=10,
                           markerfacecolor='#E65100', markeredgecolor='black',
                           label='▲ Shoulder Hiking')
                )
            if has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='D', color='w', markersize=9,
                           markerfacecolor='#D32F2F', markeredgecolor='black',
                           label='◆ Both Compensations')
                )
            
            # Position legend outside plot area
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.95)
            
            # Stats for this set
            total = len(set_entries)
            correct = sum(1 for r in set_entries if r['rom_label'] == 'correct')
            trunk_count = sum(1 for t in trunk_lean if t)
            hiking_count = sum(1 for h in shoulder_hiking if h)
            high_fatigue = sum(1 for f in fatigue_levels if f == 'High')
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Comprehensive stats box with baseline info
            stats_line1 = f"Reps: {total}  |  Correct: {correct}/{total} ({100*correct/total:.0f}%)  |  High Deter.: {high_fatigue}"
            stats_line2 = f"Trunk Lean: {trunk_count}  |  Shoulder Hiking: {hiking_count}  |  Avg Duration: {avg_duration:.0f} frames"
            
            # Add baseline info (now correctly per-set)
            if baseline_rom and baseline_dur:
                stats_line3 = f"Baseline (reps 2-4): {baseline_rom:.0f}° ROM, {baseline_dur:.0f} frames ({self.frames_to_seconds(baseline_dur):.1f}s)"
                if baseline_low_conf:
                    stats_line3 += " [low confidence]"
            else:
                stats_line3 = "Baseline: Not yet computed (need reps 2-4)"
            
            stats_text = f"{stats_line1}\n{stats_line2}\n{stats_line3}"
            
            # Position stats below the chart
            ax.text(0.5, -0.20 if num_sets == 1 else -0.32, stats_text, 
                   transform=ax.transAxes, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1', 
                            edgecolor='#90A4AE', linewidth=1.5),
                   family='monospace')
        
        # Overall title if multiple sets
        if num_sets > 1:
            fig.suptitle('Session Progress - All Sets', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=(0, 0.04, 0.82, 0.97 if num_sets > 1 else 0.98))
        
        # Embed in scrollable frame (centered)
        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=20, pady=10)
        
        # Button frame for actions (fixed at bottom of window, not scrollable)
        btn_frame = ttk.Frame(graph_window)
        btn_frame.pack(side=tk.BOTTOM, pady=8)
        
        # Center the canvas content
        def center_canvas(event=None):
            canvas_scroll.update_idletasks()
            canvas_width = canvas_scroll.winfo_width()
            frame_width = scrollable_frame.winfo_reqwidth()
            x_offset = max(0, (canvas_width - frame_width) // 2)
            canvas_scroll.coords(canvas_scroll.find_all()[0], x_offset, 0)
        
        canvas_scroll.bind("<Configure>", center_canvas)
        graph_window.after(100, center_canvas)
        
        # Save PNG button
        def save_png():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            participant_id = self.participant_id.get().strip() or "TEST_USER"
            filename = f"graph_{participant_id}_{timestamp}.png"
            
            logs_dir = self._get_participant_log_dir()
            filepath = os.path.join(logs_dir, filename)
            
            try:
                fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                messagebox.showinfo("Saved", f"Graph saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save graph:\n{str(e)}")
        
        ttk.Button(btn_frame, text="💾 Save PNG", command=save_png).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=graph_window.destroy).pack(side=tk.LEFT, padx=5)

    def find_camera(self):
        """Auto-detect available camera (tries indices 0-5)"""
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Camera found at index {i}")
                    return i
        print("No camera detected")
        return None
        
    def start_video(self):
        """Start video capture with participant ID validation (Phase 5.A)"""
        if not self.is_running:
            # Validate participant ID (Phase 5.A)
            participant_id = self.participant_id.get().strip()
            if not participant_id:
                self.status_label.config(text="Status: Error - Please enter Participant ID", foreground="#CC0000")
                return
            
            # Validate alphanumeric only (allow _ and -)
            if not participant_id.replace('_', '').replace('-', '').isalnum():
                self.status_label.config(text="Status: Error - Participant ID must be alphanumeric (allows _ and -)", foreground="#CC0000")
                return
            
            # Validate file mode has file selected (Phase 5.H)
            if self.video_mode.get() == "file" and not self.video_file_path:
                self.status_label.config(text="Status: Error - Please select a video file", foreground="#CC0000")
                return
            
            # Determine video source
            source = self.video_file_path if self.video_mode.get() == "file" else self.video_source

            # Phase 10: start-of-session guide display (one-time per exercise per session)
            self.guides_shown_in_session = set()
            self.show_exercise_guide(exercise=self.current_exercise.get())
            
            # Show loading state (Phase 5.E)
            loading_msg = "Loading video file..." if self.video_mode.get() == "file" else "Initializing camera..."
            self.status_label.config(text=f"Status: {loading_msg}", foreground="#0066cc")
            self.video_label.config(text=f"📷 {loading_msg}\nPlease wait", image='')
            self.root.update()  # Force UI refresh
            
            self.cap = cv2.VideoCapture(source)  # type: ignore[arg-type]
            
            if not self.cap.isOpened():
                self.status_label.config(text=f"Status: Error - Cannot open video source", foreground="#CC0000")
                self.video_label.config(text="❌ Video initialization failed\nCheck source")
                return
            
            self.is_running = True
            self.is_paused = False
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL, text="Pause")
            self.stop_btn.config(state=tk.NORMAL)
            self.prev_set_btn.config(state=tk.NORMAL)
            self.next_set_btn.config(state=tk.NORMAL)
            
            # Get video FPS and rotation for proper playback (video files only)
            if self.video_mode.get() == "file":
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.video_fps = self._sanitize_fps(source_fps, fallback=30.0)
                # Required file-mode behavior: delay from source FPS, fallback ~33ms
                self.video_frame_delay = max(1, int(1000 / self.video_fps))
                
                # Detect rotation once at startup (not every frame)
                try:
                    self.video_rotation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))
                except:
                    self.video_rotation = 0
                
                # Set session FPS from detected video FPS (BUG-3 fix)
                self.session_fps = self.video_fps
                mode_str = f"File @ {self.video_fps:.0f}fps"
            else:
                self.video_frame_delay = 10  # Camera: process ASAP
                self.video_rotation = 0
                
                # Set session FPS from camera (BUG-3 fix)
                camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.session_fps = self._sanitize_fps(camera_fps, fallback=30.0)
                mode_str = "Camera"
            
            self.status_label.config(text=f"Status: Running ({mode_str})", foreground="#000000")
            
            # Reset FPS counter
            self.frame_count = 0
            self.start_time = time.time()
            self.fps_samples = []
            self.last_fps_sample_time = time.time()
            self._recording_frame_count = 0
            self._recording_start_time = None
            self._effective_recording_fps = None
            
            # Reset rep tracker and frame index
            self.frame_idx = 0
            self.current_set = 1
            self.rep_tracker.reset()
            self.rep_tracker.set_exercise(self.current_exercise.get())  # Set exercise-specific thresholds
            self.last_rep_sound_key = None  # Reset one-shot per-rep audio guard
            self.set_completed_prompted = False
            self.rep_flash_active = False
            self.rep_flash_color = None
            self.rep_flash_start = 0
            self.rep_label.config(text="Reps: 0", foreground="#006600")  # Reset color
            self.set_label.config(text=f"Set: {self.current_set} of {self.total_sets}")
            
            # Reset spatial-temporal filter for new session
            self.spatial_temporal_filter.reset()
            self._last_valid_landmarks = None
            self._last_valid_results = None
            self._pose_gap_frames = 0
            
            # Reset fatigue module (Phase 4)
            self.fatigue_module.reset()
            self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
            self.break_label.config(text="")
            self.safety_label.config(text="⚠️ If you feel pain, stop and consult your PT")
            self.micro_break_active = False
            
            # Reset compensation tracking
            self._reset_current_rep_compensation()
            self.trunk_lean_label.config(text="")
            self.shoulder_hiking_label.config(text="")
            
            # Initialize session log (Phase 4.12)
            self.session_log = []
            self.baseline_log = {}
            self.session_start_time = datetime.now()
            
            # Initialize exercise block tracking (Phase 5.C)
            self.exercise_block = 1
            self.last_exercise = self.current_exercise.get()
            
            # Disable participant ID and affected side editing during session (Phase 5.A + MISSING-1)
            self.participant_entry.config(state='disabled')
            self.side_menu.config(state='disabled')
            
            # Initialize session recording (Phase 5.L)
            self._init_recording(participant_id)
            
            # Activate Tracking Readiness Check (T11: gated 3-sec calibration with detection waiting)
            # Phase A: Wait for landmark detection before starting countdown
            self.calibration_phase_active = True
            self.calibration_waiting_for_detection = True
            self.calibration_countdown_started = False
            self.calibration_waiting_start_time = time.time()
            self.calibration_start_time = None  # Will be set when countdown begins
            self.calibration_landmarks_buffer = []
            self.calibration_segment_lengths = []
            self.calibration_passed = False
            self.require_calibration_pass = False
            self._countdown_last_tone = 0  # Reset countdown tone tracker
            self._calibration_retry_count = 0  # Reset strict calibration retry counter
            self.calibration_data = {}
            self.baseline_limb_length = {}
            self.status_label.config(
                text="Status: Calibrating - Get into position (frontal/lateral as needed)...",
                foreground="#0066cc"
            )
            
            # Start video loop
            self.update_frame()
    
    def stop_video(self):
        """Stop video capture"""
        if self.is_running:
            had_data = bool(self.session_log)
            self.is_running = False
            self.is_paused = False
            self.guides_shown_in_session = set()
            
            # Release recording writers (Phase 5.L) — before cap release
            self._release_recording()
            
            if self.cap:
                self.cap.release()
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED, text="Pause")
            self.stop_btn.config(state=tk.DISABLED)
            self.prev_set_btn.config(state=tk.DISABLED)
            self.next_set_btn.config(state=tk.DISABLED)
            
            # Re-enable participant ID and affected side editing (Phase 5.A + MISSING-1)
            self.participant_entry.config(state='normal')
            self.side_menu.config(state='readonly')
            
            # Show session summary dialog before export (Phase 5.E)
            if self.session_log and self.show_session_summary:
                self.show_session_summary_dialog()
            
            # Export session log to CSV (Phase 4.12)
            if self.session_log:
                self.export_session_csv()
                
                # Ask if user wants PDF report
                if messagebox.askyesno("PDF Report", "Generate PDF report?\n(Recommended for professional documentation)"):
                    self.export_session_pdf()
            
            self.status_label.config(text="Status: Stopped", foreground="#000000")
            self.video_label.config(image='', text="📹 Video will appear here\nClick 'Start' to begin session", bg="#f0f0f0")

            if had_data:
                self.root.after(100, self._show_post_session_dialog)

    def _show_post_session_dialog(self):
        """Show post-session options dialog for multi-session workflow."""
        existing_dialog = getattr(self, '_post_session_dialog', None)
        if existing_dialog is not None:
            try:
                if existing_dialog.winfo_exists():
                    existing_dialog.lift()
                    existing_dialog.focus_force()
                    return
            except tk.TclError:
                pass

        dialog = tk.Toplevel(self.root)
        self._post_session_dialog = dialog
        dialog.title("Session Complete")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text="Session exported successfully.\nWhat would you like to do?",
            font=("Arial", 12),
            justify=tk.CENTER
        ).pack(padx=20, pady=(20, 15))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=20, pady=(0, 20))

        def _close_dialog():
            self._post_session_dialog = None
            dialog.destroy()

        def continue_same():
            self.reset_session()
            self.status_label.config(
                text="Status: Ready - Same participant, new session",
                foreground="#006600"
            )
            _close_dialog()

        def new_participant():
            self.reset_session()
            self.participant_id.set("")
            self.participant_entry.config(state='normal')
            self.participant_entry.focus_set()
            self.status_label.config(
                text="Status: Ready - Enter new participant ID",
                foreground="#0066cc"
            )
            _close_dialog()

        def return_ready():
            self.status_label.config(text="Status: Ready", foreground="#000000")
            _close_dialog()

        def exit_app():
            _close_dialog()
            self.on_closing()

        ttk.Button(btn_frame, text="Continue Same Participant", command=continue_same, width=28).pack(pady=3)
        ttk.Button(btn_frame, text="New Participant", command=new_participant, width=28).pack(pady=3)
        ttk.Button(btn_frame, text="Return to Ready Screen", command=return_ready, width=28).pack(pady=3)
        ttk.Button(btn_frame, text="Exit Application", command=exit_app, width=28).pack(pady=3)

        dialog.protocol("WM_DELETE_WINDOW", return_ready)
    
    def export_session_pdf(self):
        """
        Export comprehensive session report as PDF with graph and statistics.
        Professional format for physical therapists.
        """
        if not self.session_log:
            messagebox.showinfo("No Data", "No session data to export")
            return
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "PDF report generation requires the 'reportlab' package.\n\n"
                "Install it with: pip install reportlab\n\n"
                "Session data has been saved to CSV."
            )
            return
        
        try:
            
            # Generate filename
            timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S') if self.session_start_time else datetime.now().strftime('%Y%m%d_%H%M%S')
            participant_id = self.participant_id.get().strip() or "TEST_USER"
            exercises = list(set(rep['exercise'] for rep in self.session_log))
            exercise_str = "_".join(sorted(exercises))
            
            filename = f"report_{participant_id}_{exercise_str}_{timestamp}.pdf"
            logs_dir = self._get_participant_log_dir()
            filepath = os.path.join(logs_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=letter,
                                   topMargin=0.75*inch, bottomMargin=0.75*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1565C0'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#1976D2'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # ===== PAGE 1: TITLE AND SUMMARY =====
            story.append(Paragraph("Shoulder Mobility Assessment System", title_style))
            story.append(Paragraph("Mobility Assessment Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Compute session duration (start -> end) for PT-facing metadata
            session_end_time = None
            if self.session_log:
                last_timestamp = self.session_log[-1].get('timestamp')
                if isinstance(last_timestamp, str):
                    try:
                        session_end_time = datetime.fromisoformat(last_timestamp)
                    except ValueError:
                        session_end_time = None

            if session_end_time is None:
                session_end_time = datetime.now()

            if self.session_start_time:
                duration_seconds = max(0, int((session_end_time - self.session_start_time).total_seconds()))
                duration_hours, duration_remainder = divmod(duration_seconds, 3600)
                duration_minutes, duration_secs = divmod(duration_remainder, 60)
                if duration_hours > 0:
                    session_duration = f"{duration_hours}h {duration_minutes}m {duration_secs}s"
                else:
                    session_duration = f"{duration_minutes}m {duration_secs}s"
            else:
                session_duration = "N/A"

            rom_threshold = self.config.rom_threshold if self.config else 150
            
            # Session metadata
            metadata = [
                ["Participant ID:", participant_id],
                ["Affected Side:", self.affected_side.get()],  # MISSING-1
                ["Exercise(s):", exercise_str],
                ["Session Date:", self.session_start_time.strftime('%Y-%m-%d %H:%M') if self.session_start_time else "N/A"],
                ["Total Repetitions:", str(len(self.session_log))],
                ["Sets Completed:", str(len(set(r['set_number'] for r in self.session_log)))],
                ["Session Duration:", session_duration],
                ["ROM Threshold:", f"{rom_threshold:g}°"],
            ]
            
            metadata_table = Table(metadata, colWidths=[2.5*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 0.4*inch))
            
            # Performance summary
            story.append(Paragraph("Performance Summary", heading_style))
            
            total_reps = len(self.session_log)
            correct_reps = sum(1 for r in self.session_log if r['rom_label'] == 'correct')
            correct_pct = (correct_reps / total_reps * 100) if total_reps > 0 else 0
            avg_peak = sum(r['peak_angle'] for r in self.session_log) / total_reps if total_reps > 0 else 0
            best_peak = max(r['peak_angle'] for r in self.session_log) if total_reps > 0 else 0
            
            trunk_lean_count = sum(1 for r in self.session_log if r['trunk_lean_detected'])
            hiking_count = sum(1 for r in self.session_log if r['shoulder_hiking_detected'])
            
            high_fatigue = sum(1 for r in self.session_log if r['fatigue_level'] == 'High')
            medium_fatigue = sum(1 for r in self.session_log if r['fatigue_level'] == 'Medium')

            has_unscorable_flag = all('unscorable_flag' in r for r in self.session_log)
            unscorable_count = sum(1 for r in self.session_log if r.get('unscorable_flag', False)) if has_unscorable_flag else None

            if any('micro_break_triggered' in r for r in self.session_log):
                micro_break_count = sum(1 for r in self.session_log if r.get('micro_break_triggered', False))
            elif any('trigger_break' in r for r in self.session_log):
                micro_break_count = sum(1 for r in self.session_log if r.get('trigger_break', False))
            else:
                micro_break_count = None
            
            # LC-9: Report interpretation thresholds from config (same defaults as previously hardcoded)
            _rom_good_pct = self.config.correct_rom_good_pct if self.config else 70
            _peak_good_angle = self.config.avg_peak_good_angle if self.config else 150
            _comp_acceptable = self.config.compensation_acceptable_pct if self.config else 0.2
            _fatigue_sig_count = self.config.high_fatigue_significant_count if self.config else 3
            
            summary_data = [
                ["Metric", "Value", "Interpretation"],
                ["Correct ROM", f"{correct_reps}/{total_reps} ({correct_pct:.0f}%)", 
                 "Good" if correct_pct >= _rom_good_pct else "Needs Improvement"],
                ["Average Peak Angle", f"{avg_peak:.1f}°", 
                 "Good" if avg_peak >= _peak_good_angle else "Below Target"],
                ["Best Peak Angle", f"{best_peak:.1f}°",
                 "Good" if best_peak >= _peak_good_angle else "Below Target"],
                ["Trunk Lean Events", str(trunk_lean_count), 
                 "Acceptable" if trunk_lean_count <= total_reps * _comp_acceptable else "Frequent"],
                ["Shoulder Hiking Events", str(hiking_count), 
                 "Acceptable" if hiking_count <= total_reps * _comp_acceptable else "Frequent"],
                ["High Deterioration Events", str(high_fatigue), 
                 "Low" if high_fatigue <= _fatigue_sig_count else "Significant"],
                ["Unscorable Reps", str(unscorable_count) if unscorable_count is not None else "N/A",
                 "None" if unscorable_count == 0 else ("Review Tracking" if unscorable_count is not None else "N/A")],
                ["Micro-Break Prompts", str(micro_break_count) if micro_break_count is not None else "N/A",
                 "None" if micro_break_count == 0 else ("Triggered" if micro_break_count is not None else "N/A")],
            ]
            
            summary_table = Table(summary_data, colWidths=[2.2*inch, 2*inch, 2.3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            story.append(PageBreak())
            
            # ===== VISUAL PROGRESS ANALYSIS (One page per set for readability) =====
            story.append(Paragraph("Visual Progress Analysis", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Generate comprehensive graph (one image per set)
            temp_graph_paths = []
            self.save_graph_to_file_comprehensive(logs_dir, timestamp, temp_graph_paths)
            
            # Add each set's graph on its own page for readability
            for i, temp_path in enumerate(temp_graph_paths):
                if i > 0:
                    story.append(PageBreak())
                    story.append(Paragraph(f"Visual Progress Analysis (continued)", heading_style))
                    story.append(Spacer(1, 0.2*inch))
                
                # Use full page width for readability
                img = RLImage(temp_path, width=7.5*inch, height=6*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            
            story.append(PageBreak())
            
            # ===== PAGE 2+: DETAILED REP-BY-REP DATA (FULL SESSION) =====
            story.append(Paragraph("Detailed Repetition Log", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Rep data table - FULL session (no limit)
            # Table will automatically span multiple pages if needed
            rep_data = [["Rep", "Set", "Exercise", "Peak°", "ROM", "Duration", "Deterioration", "Form"]]
            
            for i, rep in enumerate(self.session_log, 1):
                comp_flags = []
                if rep['trunk_lean_detected']:
                    comp_flags.append("TL")
                if rep['shoulder_hiking_detected']:
                    comp_flags.append("SH")
                comp_str = "+".join(comp_flags) if comp_flags else "-"
                
                # Exercise abbreviation (Abd/Flex)
                exercise_abbr = "Abd" if rep['exercise'] == 'Abduction' else "Flex"
                
                rep_data.append([
                    str(i),
                    str(rep['set_number']),
                    exercise_abbr,
                    f"{rep['peak_angle']:.0f}°",
                    "✓" if rep['rom_label'] == 'correct' else "✗",
                    f"{self.frames_to_seconds(rep['duration_frames']):.1f}s",
                    rep['fatigue_level'][0],  # L/M/H
                    comp_str
                ])
            
            rep_table = Table(rep_data, colWidths=[0.45*inch, 0.4*inch, 0.55*inch, 0.65*inch, 0.5*inch, 0.7*inch, 0.6*inch, 0.65*inch], repeatRows=1)
            rep_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(rep_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Legend for abbreviations
            legend_text = "<b>Legend:</b> Abd=Abduction, Flex=Flexion, TL=Trunk Lean, SH=Shoulder Hiking, L/M/H=Low/Med/High Deterioration"
            story.append(Paragraph(legend_text, styles['Normal']))

            story.append(Spacer(1, 0.2*inch))
            footer_note = "This report summarizes key session outputs for clinical review. The complete technical dataset, including all per-frame parameters, is available in the CSV export."
            story.append(Paragraph(footer_note, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
            
            # Build PDF
            doc.build(story)
            
            # Clean up temporary graphs
            for temp_path in temp_graph_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            messagebox.showinfo("PDF Exported", f"Session report saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to create PDF:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_graph_to_file(self, filepath):
        """Helper function to save current session graph to file (used for PDF export)"""
        if not self.session_log:
            return
        
        # Group data by exercise block AND set (supports mid-session exercise switching)
        sets_data = {}
        for entry in self.session_log:
            block = entry.get('exercise_block', 1)
            exercise = entry.get('exercise', 'Unknown')
            set_num = entry.get('set_number', 1)
            key = (block, exercise, set_num)
            if key not in sets_data:
                sets_data[key] = []
            sets_data[key].append(entry)
        
        num_sets = len(sets_data)
        
        # Create figure (same logic as show_rep_graph but without tkinter window)
        if num_sets == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        else:
            fig, axes = plt.subplots(num_sets, 1, figsize=(10, 2.5 + num_sets * 2.5))
            if num_sets == 1:
                axes = [axes]
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import numpy as np
        
        fatigue_colors = {
            'Low': ('#A5D6A7', 0.5),
            'Medium': ('#FFE082', 0.6),
            'High': ('#EF9A9A', 0.7)
        }
        
        # Process each set (simplified version of show_rep_graph)
        for set_idx, (key, set_entries) in enumerate(sorted(sets_data.items())):
            ax = axes[set_idx]
            block, exercise_name, set_num = key
            
            reps = list(range(1, len(set_entries) + 1))
            peak_angles = [r['peak_angle'] for r in set_entries]
            rom_labels = [r['rom_label'] for r in set_entries]
            fatigue_levels = [r['fatigue_level'] for r in set_entries]
            durations = [r['duration_frames'] for r in set_entries]
            
            # Background fatigue shading
            for i, (rep, fatigue) in enumerate(zip(reps, fatigue_levels)):
                color, alpha = fatigue_colors.get(fatigue, ('#FFFFFF', 0.0))
                ax.axvspan(rep - 0.45, rep + 0.45, color=color, alpha=alpha, zorder=0)
            
            # Main line
            ax.plot(reps, peak_angles, '-', color='#1976D2', linewidth=2, zorder=2)
            
            # Markers
            for i, (rep, angle, rom) in enumerate(zip(reps, peak_angles, rom_labels)):
                color = '#2E7D32' if rom == 'correct' else '#C62828'
                ax.scatter(rep, angle, color=color, s=100, zorder=4, 
                          edgecolors='white', linewidth=1.5)
            
            # Reference line
            ax.axhline(y=150, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
            
            # Styling
            avg_rom = sum(peak_angles) / len(peak_angles)
            set_title = f'{exercise_name} - Set {set_num}  (Avg: {avg_rom:.0f}°)' if num_sets > 1 else f'{exercise_name} (Avg: {avg_rom:.0f}°)'
            ax.set_title(set_title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Rep Number', fontsize=10)
            ax.set_ylabel('Peak Angle (°)', fontsize=10)
            ax.set_xticks(reps)
            ax.grid(axis='y', alpha=0.3)
        
        if num_sets > 1:
            fig.suptitle('Session Progress - All Sets', fontsize=14, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=(0, 0, 1, 0.96 if num_sets > 1 else 1))
        fig.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def save_graph_to_file_comprehensive(self, logs_dir, timestamp, temp_paths_list):
        """
        Generate comprehensive graphs for PDF export - one per set with full legends.
        Matches the Session Progress window appearance for professional reports.
        """
        if not self.session_log:
            return
        
        # Group data by exercise block AND set (supports mid-session exercise switching)
        sets_data = {}
        for entry in self.session_log:
            block = entry.get('exercise_block', 1)
            exercise = entry.get('exercise', 'Unknown')
            set_num = entry.get('set_number', 1)
            key = (block, exercise, set_num)
            if key not in sets_data:
                sets_data[key] = []
            sets_data[key].append(entry)
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import numpy as np
        from statistics import median
        
        fatigue_colors = {
            'Low': ('#A5D6A7', 0.5),
            'Medium': ('#FFE082', 0.6),
            'High': ('#EF9A9A', 0.7)
        }
        
        # Generate one graph per set for readability
        for key, set_entries in sorted(sets_data.items()):
            block, exercise_name, set_num = key
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            
            reps = list(range(1, len(set_entries) + 1))
            peak_angles = [r['peak_angle'] for r in set_entries]
            rom_labels = [r['rom_label'] for r in set_entries]
            fatigue_levels = [r['fatigue_level'] for r in set_entries]
            trunk_lean = [r['trunk_lean_detected'] for r in set_entries]
            shoulder_hiking = [r['shoulder_hiking_detected'] for r in set_entries]
            durations = [r['duration_frames'] for r in set_entries]
            
            # Get per-set baseline using (block, set_num) key
            baseline_rom = None
            baseline_dur = None
            baseline_low_conf = False
            
            baseline_key = (block, set_num)
            if baseline_key in self.baseline_log:
                baseline_info = self.baseline_log[baseline_key]
                baseline_rom = baseline_info.get('baseline_rom')
                baseline_dur = baseline_info.get('baseline_dur')
                baseline_low_conf = baseline_info.get('low_confidence', False)
            
            # Fallback: compute from set's reps 2-4
            if baseline_rom is None and len(set_entries) >= 2:
                baseline_candidates = [
                    e for e in set_entries 
                    if e.get('rep_number', 0) in [2, 3, 4]
                ]
                if len(baseline_candidates) >= 1:
                    baseline_rom = median([e['peak_angle'] for e in baseline_candidates])
                    baseline_dur = median([e['duration_frames'] for e in baseline_candidates])
                    baseline_low_conf = len(baseline_candidates) < 3
            
            # Background fatigue shading
            for i, (rep, fatigue) in enumerate(zip(reps, fatigue_levels)):
                color, alpha = fatigue_colors.get(fatigue, ('#FFFFFF', 0.0))
                ax.axvspan(rep - 0.45, rep + 0.45, color=color, alpha=alpha, zorder=0)
            
            # Main line
            ax.plot(reps, peak_angles, '-', color='#1976D2', linewidth=2.5, zorder=2)
            
            # Markers with ROM classification colors
            for i, (rep, angle, rom) in enumerate(zip(reps, peak_angles, rom_labels)):
                color = '#2E7D32' if rom == 'correct' else '#C62828'
                ax.scatter(rep, angle, color=color, s=120, zorder=4, 
                          edgecolors='white', linewidth=2)
            
            # Trend line
            trend_color = '#4CAF50'
            z = None
            if len(reps) >= 4:
                z = np.polyfit(reps, peak_angles, 1)
                p = np.poly1d(z)
                trend_color = '#4CAF50' if z[0] >= 0 else '#FF5722'
                ax.plot(reps, p(reps), '--', color=trend_color, linewidth=1.5, alpha=0.7, zorder=1)
            
            # Reference lines
            ax.axhline(y=150, color='#1565C0', linestyle='--', linewidth=2, alpha=0.8, zorder=1)
            
            if baseline_rom:
                ax.axhline(y=baseline_rom, color='#7B1FA2', linestyle=':', linewidth=2.5, zorder=1)
            
            # Duration annotations
            baseline_dur_sec = self.frames_to_seconds(baseline_dur) if baseline_dur else None
            for i, (rep, angle, dur) in enumerate(zip(reps, peak_angles, durations)):
                dur_sec = self.frames_to_seconds(dur)
                dur_color = '#006400' if baseline_dur_sec and dur_sec <= baseline_dur_sec * 1.2 else '#B71C1C'
                ax.annotate(f'{dur_sec:.1f}s', (rep, angle + 8), fontsize=8, 
                           ha='center', va='bottom', color=dur_color, fontweight='bold',
                           zorder=7, bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                              edgecolor='none', alpha=0.8))
            
            # Compensation markers
            y_bottom = max(0, min(peak_angles) - 25) if min(peak_angles) > 70 else 45
            has_trunk = has_hiking = has_both = False
            
            for i, rep in enumerate(reps):
                if trunk_lean[i] and shoulder_hiking[i]:
                    ax.scatter(rep, y_bottom, marker='D', s=150, color='#D32F2F', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_both = True
                elif trunk_lean[i]:
                    ax.scatter(rep, y_bottom, marker='v', s=150, color='#FF6F00', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_trunk = True
                elif shoulder_hiking[i]:
                    ax.scatter(rep, y_bottom, marker='^', s=150, color='#E65100', 
                              zorder=5, edgecolors='black', linewidth=1.5)
                    has_hiking = True
            
            # Styling
            avg_rom = sum(peak_angles) / len(peak_angles)
            ax.set_title(f'{exercise_name} - Set {set_num}  (Avg: {avg_rom:.0f}°)', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Rep Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Peak Angle (°)', fontsize=11, fontweight='bold')
            ax.set_xticks(reps)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(bottom=max(0, min(peak_angles) - 40), top=max(peak_angles) + 40)
            
            # Build comprehensive legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markersize=9, 
                       markerfacecolor='#2E7D32', markeredgecolor='white', 
                       label='Correct ROM (≥150°)'),
                Line2D([0], [0], marker='o', color='w', markersize=9, 
                       markerfacecolor='#C62828', markeredgecolor='white', 
                       label='Insufficient ROM (<150°)'),
                Line2D([0], [0], color='#1565C0', linestyle='--', linewidth=2, 
                       label='Target ROM (150°)'),
            ]
            
            if baseline_rom:
                conf_note = ' (low conf.)' if baseline_low_conf else ''
                legend_elements.append(
                    Line2D([0], [0], color='#7B1FA2', linestyle=':', linewidth=2.5, 
                           label=f'Baseline ROM ({baseline_rom:.0f}°){conf_note}')
                )
            
            if z is not None and len(reps) >= 4:
                trend_label = 'Trend (improving)' if z[0] >= 0 else 'Trend (declining)'
                legend_elements.append(
                    Line2D([0], [0], color=trend_color, linestyle='--', linewidth=1.5, 
                           alpha=0.7, label=trend_label)
                )
            
            legend_elements.extend([
                Patch(facecolor='#A5D6A7', alpha=0.7, edgecolor='gray', label='Low Deter.'),
                Patch(facecolor='#FFE082', alpha=0.8, edgecolor='gray', label='Med Deter.'),
                Patch(facecolor='#EF9A9A', alpha=0.9, edgecolor='gray', label='High Deter.'),
            ])  # type: ignore[arg-type]
            
            if baseline_dur_sec:
                legend_elements.append(
                    Line2D([0], [0], marker='', color='w', markersize=0,
                           label=f'Duration labels (baseline: {baseline_dur_sec:.1f}s)')
                )
            
            if has_trunk or has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='v', color='w', markersize=9,
                           markerfacecolor='#FF6F00', markeredgecolor='black', label='▼ Trunk Lean')
                )
            if has_hiking or has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='^', color='w', markersize=9,
                           markerfacecolor='#E65100', markeredgecolor='black', label='▲ Shoulder Hiking')
                )
            if has_both:
                legend_elements.append(
                    Line2D([0], [0], marker='D', color='w', markersize=8,
                           markerfacecolor='#D32F2F', markeredgecolor='black', label='◆ Both Compensations')
                )
            
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.95)
            
            # Stats box
            total = len(set_entries)
            correct = sum(1 for r in set_entries if r['rom_label'] == 'correct')
            trunk_count = sum(1 for t in trunk_lean if t)
            hiking_count = sum(1 for h in shoulder_hiking if h)
            high_fatigue = sum(1 for f in fatigue_levels if f == 'High')
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            stats_line1 = f"Reps: {total}  |  Correct: {correct}/{total} ({100*correct/total:.0f}%)  |  High Deter.: {high_fatigue}"
            stats_line2 = f"Trunk Lean: {trunk_count}  |  Shoulder Hiking: {hiking_count}  |  Avg Duration: {avg_duration:.0f} frames"
            if baseline_rom and baseline_dur:
                stats_line3 = f"Baseline (reps 2-4): {baseline_rom:.0f}° ROM, {baseline_dur:.0f} frames ({self.frames_to_seconds(baseline_dur):.1f}s)"
            else:
                stats_line3 = "Baseline: Not computed"
            
            ax.text(0.5, -0.15, f"{stats_line1}\n{stats_line2}\n{stats_line3}", 
                   transform=ax.transAxes, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1', 
                            edgecolor='#90A4AE', linewidth=1.5),
                   family='monospace')
            
            plt.tight_layout(rect=(0, 0.08, 0.82, 0.98))
            
            # Save this set's graph (include exercise and block for unique filename)
            temp_path = os.path.join(logs_dir, f"temp_graph_{exercise_name}_b{block}_set{set_num}_{timestamp}.png")
            fig.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            temp_paths_list.append(temp_path)

    def update_frame(self):
        """Read and display video frame with pose processing"""
        if self.is_running and not self.is_paused:
            if self.cap is None:
                return
            frame_start_time = time.perf_counter()
            ret, frame = self.cap.read()
            
            # End-of-video detection for file mode (auto-pause instead of infinite no-op loop)
            if not ret and self.video_mode.get() == "file":
                self.is_paused = True
                self.pause_btn.config(text="Resume")
                self.status_label.config(
                    text="Status: VIDEO ENDED — Pause/Next Set/Stop, or Resume to re-read last frame",
                    foreground="#CC6600"
                )
                print("[Main] Video file ended — session auto-paused. Use Next Set then Resume for next clip, or Stop to finish.")
                return
            
            if ret:
                # Apply video rotation (detected once at startup for performance)
                if self.video_rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.video_rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.video_rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Manual flip override (user checkbox)
                if self.flip_video.get():
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                # Overlay text layout (Phase 2: prevent overlap)
                overlay_layout = self._get_overlay_layout(frame)
                
                # Record raw frame BEFORE any drawing (Phase 5.L)
                if self.record_raw.get():
                    self._write_raw_frame(frame)


                
                # Process frame with pose estimation
                results, landmarks = self.pose_processor.process_frame(frame)
                
                # T11: Tracking Readiness Check — intercept frames during calibration phase
                if self.calibration_phase_active:
                    calibration_complete, passed = self.perform_calibration(landmarks)

                    # Calibration timeout may stop the session inside perform_calibration().
                    if not self.is_running:
                        return
                    
                    # Draw skeleton during calibration so user can see tracking
                    if results.pose_landmarks:
                        frame = self.pose_processor.draw_landmarks(frame, results)

                    # Mirror display base frame (no text overlays) for readable post-flip redraw.
                    display_base_frame = frame.copy()

                    # Stable calibration text layout (Phase 2):
                    # y=55 status, y=85 instruction, y=120 warning/detail
                    cal_status_text = ""
                    cal_status_color = (0, 150, 255)
                    cal_status_scale = 0.85
                    cal_instruction_text = ""
                    cal_instruction_color = (255, 255, 200)
                    cal_warning_text = ""

                    if self.calibration_waiting_for_detection:
                        # Phase A: Waiting for user to get into position
                        cal_status_text = "WAITING FOR DETECTION"
                        cal_status_color = (0, 150, 255)
                        cal_instruction_text = "Get into position - ensure pose is visible"
                        exercise = self.current_exercise.get()
                        guidance = "FRONTAL" if exercise == "Abduction" else "LATERAL"
                        cal_warning_text = f"View: {guidance}"
                    elif self.calibration_countdown_started:
                        # Phase B: Countdown in progress
                        self.play_cue('calibration_start', cooldown_sec=15.0)
                        elapsed = time.time() - (self.calibration_start_time or time.time())
                        remaining = max(0, self.calibration_duration_sec - elapsed)
                        # Countdown tones C04-C06: fire once at 3s, 2s, 1s remaining
                        remaining_int = int(remaining)
                        if remaining_int != self._countdown_last_tone:
                            if remaining_int == 3:
                                self.play_sound('success', audio_filename='3')  # C04
                            elif remaining_int == 2:
                                self.play_sound('success', audio_filename='2')  # C05
                            elif remaining_int == 1:
                                self.play_sound('success', audio_filename='1')  # C06
                            self._countdown_last_tone = remaining_int

                        cal_status_text = "CALIBRATING"
                        cal_status_color = (0, 200, 0)
                        cal_instruction_text = f"Hold neutral position: {remaining:.1f}s"
                        cal_instruction_color = (0, 255, 100)
                        cal_warning_text = "Arms relaxed at sides"
                    
                    # Handle calibration completion
                    if calibration_complete:
                        strict_required = bool(getattr(self, 'require_calibration_pass', False))
                        self.calibration_passed = bool(passed)

                        if strict_required and not passed:
                            self._calibration_retry_count += 1

                            if self._calibration_retry_count >= self.MAX_CALIBRATION_RETRIES:
                                # Max retries exceeded — proceed with warning instead of looping.
                                self.require_calibration_pass = False
                                self.calibration_phase_active = False
                                self.calibration_waiting_for_detection = False
                                self.calibration_countdown_started = False
                                self.calibration_passed = False

                                self.status_label.config(
                                    text="Status: Calibration unstable after 3 retries - proceeding with caution",
                                    foreground="#CC6600"
                                )
                                print("[Calibration] WARNING: Max retries exceeded, proceeding without stable calibration")

                                cal_status_text = "CALIBRATION WARNING"
                                cal_status_color = (0, 165, 255)
                                cal_status_scale = 0.95
                                cal_instruction_text = "Max retries exceeded - proceed with caution"
                                cal_instruction_color = (200, 165, 100)
                                cal_warning_text = ""
                            else:
                                # Set-transition strict mode: retry calibration until max retry limit.
                                self.calibration_phase_active = True
                                self.calibration_waiting_for_detection = True
                                self.calibration_countdown_started = False
                                self.calibration_start_time = None
                                self.calibration_waiting_start_time = time.time()
                                self.calibration_landmarks_buffer = []
                                self.calibration_segment_lengths = []
                                self.calibration_data = {}
                                self.baseline_limb_length = {}
                                self._countdown_last_tone = 0
                                self.spatial_temporal_filter.reset_blc()

                                self.status_label.config(
                                    text="Status: Calibration unstable - hold neutral pose to retry",
                                    foreground="#CC6600"
                                )
                                print("[Calibration] WARNING: Set-transition calibration failed, retrying")
                                self.play_cue('calibration_retry', cooldown_sec=10.0)

                                cal_status_text = "CALIBRATION WARNING"
                                cal_status_color = (0, 165, 255)
                                cal_status_scale = 0.95
                                cal_instruction_text = "Unstable tracking - retrying calibration"
                                cal_instruction_color = (200, 165, 100)
                                cal_warning_text = "Rep tracking remains paused"
                        else:
                            self.calibration_phase_active = False
                            self.require_calibration_pass = False

                            if passed:
                                self.status_label.config(
                                    text="Status: Calibration passed ✓ - Begin exercise now",
                                    foreground="#006600"
                                )
                                print(f"[Calibration] PASSED: {self.calibration_data}")
                                self.play_sound('success', audio_filename='ready')  # C07
                                cal_status_text = "CALIBRATION PASSED"
                                cal_status_color = (0, 200, 0)
                                cal_status_scale = 0.95
                                cal_instruction_text = "BLC baseline established"
                                cal_instruction_color = (100, 255, 100)
                                cal_warning_text = ""
                            else:
                                self.status_label.config(
                                    text="Status: Calibration warning ⚠ - Tracking may be unstable (proceeding anyway)",
                                    foreground="#CC6600"
                                )
                                print("[Calibration] WARNING: Tracking unstable, proceeding anyway")
                                cal_status_text = "CALIBRATION WARNING"
                                cal_status_color = (0, 165, 255)
                                cal_status_scale = 0.95
                                cal_instruction_text = "Unstable tracking - proceed with caution"
                                cal_instruction_color = (200, 165, 100)
                                cal_warning_text = ""

                    # Draw calibration overlays (consistent y-offsets)
                    self._draw_overlay_text(
                        frame, cal_status_text, overlay_layout['guidance_y'],
                        cal_status_color, font_scale=cal_status_scale, thickness=2
                    )
                    self._draw_overlay_text(
                        frame, cal_instruction_text, overlay_layout['instruction_y'],
                        cal_instruction_color, font_scale=0.62, thickness=2
                    )
                    self._draw_overlay_text(
                        frame, cal_warning_text, overlay_layout['warning_y'],
                        (220, 220, 220), font_scale=0.60, thickness=1
                    )
                    
                    # Display frame
                    # Record annotated frame during calibration (Phase 4/5.L)
                    session_label_text = None
                    if self.record_annotated.get():
                        session_label_text = self._get_annotated_session_label(calibrating=True)
                        self._draw_overlay_text(
                            frame, session_label_text, overlay_layout['session_y'],
                            (255, 255, 255), font_scale=0.58, thickness=2
                        )
                        self._write_annotated_frame(frame)

                    # Build display frame (mirror uses clean frame, then HUD text redraw).
                    display_frame = frame
                    if self.mirror_display.get():
                        display_frame = cv2.flip(display_base_frame, 1)
                        self._draw_overlay_text(
                            display_frame, cal_status_text, overlay_layout['guidance_y'],
                            cal_status_color, font_scale=cal_status_scale, thickness=2, mirror=True
                        )
                        self._draw_overlay_text(
                            display_frame, cal_instruction_text, overlay_layout['instruction_y'],
                            cal_instruction_color, font_scale=0.62, thickness=2, mirror=True
                        )
                        self._draw_overlay_text(
                            display_frame, cal_warning_text, overlay_layout['warning_y'],
                            (220, 220, 220), font_scale=0.60, thickness=1, mirror=True
                        )
                        if session_label_text:
                            self._draw_overlay_text(
                                display_frame, session_label_text, overlay_layout['session_y'],
                                (255, 255, 255), font_scale=0.58, thickness=2, mirror=True
                            )

                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    max_width = 800
                    max_height = 550
                    height, width = frame_rgb.shape[:2]
                    scale_w = max_width / width
                    scale_h = max_height / height
                    scale = min(scale_w, scale_h)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk  # type: ignore[attr-defined]
                    self.video_label.config(image=imgtk, text="")
                    
                    self._schedule_next_frame(frame_start_time)
                    return  # Skip rep tracking during calibration
                
                # Apply spatial-temporal filtering (CW-EMA) to smooth landmarks before angle computation
                # Filter sits between raw MediaPipe landmarks and downstream angle/rule logic
                if landmarks:
                    filtered_landmarks = self.spatial_temporal_filter.filter_landmarks(landmarks)
                    # Safe fallback: if filter returns empty (e.g. invalid data), use raw landmarks
                    landmarks_for_angles = filtered_landmarks if filtered_landmarks else landmarks
                else:
                    # No landmarks detected - pass empty dict (filter will preserve state)
                    landmarks_for_angles = landmarks
                
                # BLC frame-level check (T2/T13: Bone-Length Constancy)
                # Runs after CW-EMA filtering; accumulates per-rep flagged-frame counts
                # Fix A: capture result for on-screen BLC warning display
                self._blc_warning_active = False
                if landmarks_for_angles:
                    blc_result = self.spatial_temporal_filter.check_bone_length(landmarks_for_angles)
                    if blc_result is not None and blc_result.flagged:
                        self._blc_warning_active = True
                
                # Compute angles using filtered landmarks
                flexion_angle, abduction_angle, low_confidence = self.pose_processor.compute_angles(landmarks_for_angles)

                # Mirror display base frame (no text overlays) for readable post-flip redraw.
                # Analysis/recording still use the original (unflipped) frame pipeline.
                display_base_frame = frame.copy()
                
                # Enhanced camera positioning feedback (Phase 5.E + Phase 3 flexion fix)
                pose_available = landmarks is not None and bool(landmarks) and bool(results.pose_landmarks)
                exercise = self.current_exercise.get()
                tracking_pause_active = False

                if pose_available:
                    # Pose detected — reset gap counter, cache valid state
                    self._pose_gap_frames = 0
                    self._last_valid_landmarks = landmarks.copy()
                    self._last_valid_results = results
                elif exercise == "Flexion" and self._pose_gap_frames < self._max_pose_gap_frames and self._last_valid_landmarks:
                    # Flexion short-gap tolerance: brief occlusions get gentler warning.
                    self._pose_gap_frames += 1
                    tracking_pause_active = True
                    print(f"[Flexion] Pose gap frame {self._pose_gap_frames}/{self._max_pose_gap_frames} — using cached landmarks")
                    self._draw_overlay_text(
                        frame, "BRIEF TRACKING PAUSE", overlay_layout['warning_y'],
                        (0, 165, 255), font_scale=0.65, thickness=2
                    )
                else:
                    # True pose loss — show positioning guidance.
                    self._pose_gap_frames += 1
                    self._last_valid_landmarks = None
                    self._last_valid_results = None
                    self.play_cue('tracking_failed', cooldown_sec=10.0)
                    guidance = "FRONTAL VIEW" if exercise == "Abduction" else "LATERAL VIEW"
                    self._draw_overlay_text(
                        frame, f"Position: {guidance} required", overlay_layout['guidance_y'],
                        (0, 165, 255), font_scale=0.65, thickness=2
                    )
                    self._draw_overlay_text(
                        frame, "Ensure full body visible", overlay_layout['instruction_y'],
                        (255, 255, 255), font_scale=0.60, thickness=2
                    )
                    self._draw_overlay_text(
                        frame, "NO POSE DETECTED", overlay_layout['warning_y'],
                        (0, 0, 255), font_scale=0.82, thickness=2
                    )

                if low_confidence and pose_available:
                    # Low confidence warning (only when pose IS detected but quality is poor)
                    self.play_cue('low_confidence', cooldown_sec=10.0)
                    self._draw_overlay_text(
                        frame, "LOW DETECTION CONFIDENCE", overlay_layout['guidance_y'],
                        (0, 165, 255), font_scale=0.65, thickness=2
                    )
                    self._draw_overlay_text(
                        frame, "Improve lighting/position", overlay_layout['instruction_y'],
                        (255, 255, 255), font_scale=0.60, thickness=2
                    )
                
                # Compute compensation detection (Phase 3.4-3.5)
                # Use filtered landmarks for compensation rules (part of downstream pipeline)
                trunk_lean_angle, trunk_lean_flag = self.pose_processor.compute_trunk_lean(landmarks_for_angles)
                shoulder_hiking_ratio, shoulder_hiking_flag = self.pose_processor.compute_shoulder_hiking(landmarks_for_angles)
                
                # Phase 3.B: Accumulate frame-level compensation for persistence ratio calculation
                # This replaces the old boolean flagging that marked entire rep on single frame detection
                self.rep_tracker.accumulate_frame_compensation(trunk_lean_flag, shoulder_hiking_flag)
                
                # Track low confidence separately (not part of persistence ratio)
                if low_confidence:
                    self.current_rep_low_confidence = True
                
                # Draw pose landmarks on frame
                if results.pose_landmarks:
                    frame = self.pose_processor.draw_landmarks(frame, results)
                    if self.mirror_display.get():
                        display_base_frame = self.pose_processor.draw_landmarks(display_base_frame, results)
                
                # Draw angles on frame (now in top RIGHT corner)
                # MISSING-6: Pass smoothed angle for optional overlay
                frame = self.pose_processor.draw_angles(
                    frame, flexion_angle, abduction_angle, low_confidence,
                    smoothed_angle=self.rep_tracker.last_smoothed_angle,
                    show_smoothed=self.show_smoothed_overlay.get()
                )
                
                # Draw compensation warnings on BOTTOM LEFT (avoid overlap)
                comp_y = overlay_layout['compensation_y_start']
                if trunk_lean_flag:
                    self._draw_overlay_text(
                        frame, f"TRUNK LEAN: {trunk_lean_angle:.1f}deg", comp_y,
                        (0, 140, 255), font_scale=0.7, thickness=2
                    )
                    comp_y += overlay_layout['compensation_step']
                if shoulder_hiking_flag:
                    self._draw_overlay_text(
                        frame, f"SHOULDER HIKING: {shoulder_hiking_ratio*100:.1f}%", comp_y,
                        (0, 140, 255), font_scale=0.7, thickness=2
                    )

                # Fix A: BLC on-screen warning (drawn on frame below compensation)
                if self._blc_warning_active:
                    blc_y = comp_y + overlay_layout['compensation_step']
                    self._draw_overlay_text(
                        frame, "BLC TRACKING WARNING", blc_y,
                        (0, 0, 255), font_scale=0.65, thickness=2
                    )
                    # Update Tkinter BLC label
                    try:
                        self.blc_warning_label.config(text="⚠ BLC Tracking Warning", foreground="#CC0000")
                    except (AttributeError, tk.TclError):
                        pass
                else:
                    try:
                        self.blc_warning_label.config(text="")
                    except (AttributeError, tk.TclError):
                        pass
                
                # Rep tracking (Phase 3.2)
                exercise = self.current_exercise.get()
                active_angle = flexion_angle if exercise == "Flexion" else abduction_angle
                
                rep_completed, peak_angle, duration = self.rep_tracker.update(active_angle, self.frame_idx)
                self.frame_idx += 1

                # Track effective FPS for recording accuracy (Phase 4)
                if self._recording_start_time is None:
                    self._recording_start_time = time.time()
                self._recording_frame_count += 1
                if self._recording_frame_count == 60:  # Sample after 60 frames (~2 seconds)
                    elapsed = time.time() - self._recording_start_time
                    if elapsed > 0:
                        self._effective_recording_fps = self._recording_frame_count / elapsed
                        print(f"[Recording] Measured effective FPS: {self._effective_recording_fps:.1f}")
                
                # MISSING-3: Lock exercise dropdown while in active rep
                # This prevents mid-rep switching which corrupts state
                if hasattr(self, 'exercise_menu'):
                    if self.rep_tracker.in_rep:
                        self.exercise_menu.config(state='disabled')
                    else:
                        self.exercise_menu.config(state='readonly')
                
                if rep_completed:
                    # Phase 3.B: Compensation flags are now computed by rep_tracker using persistence ratio
                    # No need to call update_last_rep_compensation() — it's handled in rep_tracker.update()
                    
                    # Get last rep info for ROM display
                    last_rep = self.rep_tracker.get_last_rep_info()
                    rom_label = last_rep['rom_label'] if last_rep else 'unknown'
                    trunk_lean_detected = last_rep['trunk_lean_detected'] if last_rep else False
                    shoulder_hiking_detected = last_rep['shoulder_hiking_detected'] if last_rep else False

                    # Keep compatibility with existing current-rep fields used in UI/logging paths.
                    # Values now come from persistence-ratio rep completion logic (not sticky frame flags).
                    self.current_rep_trunk_lean = trunk_lean_detected
                    self.current_rep_shoulder_hiking = shoulder_hiking_detected
                    
                    # Update rep counter with color feedback (Phase 3.3)
                    rep_color = "#006600" if rom_label == 'correct' else "#CC0000"  # Green for correct, red for insufficient
                    self.rep_label.config(
                        text=f"Reps: {self.rep_tracker.get_rep_count()} ({rom_label.upper()})",
                        foreground=rep_color
                    )
                    
                    # Trigger visual rep flash feedback (Phase 5.E)
                    self.rep_flash_active = True
                    self.rep_flash_color = 'green' if rom_label == 'correct' else 'red'
                    self.rep_flash_start = time.time()
                    
                    # Play audio feedback (Phase 5.I)
                    rep_sound_key = (self.exercise_block, self.current_set, self.rep_tracker.get_rep_count())
                    if rep_sound_key != self.last_rep_sound_key:
                        self.play_sound('success' if rom_label == 'correct' else 'error')
                        self.last_rep_sound_key = rep_sound_key
                    
                    # Compute fatigue level (Phase 4)
                    fatigue_result = self.fatigue_module.compute_fatigue_level(
                        self.rep_tracker.get_rep_history()
                    )
                    
                    # Capture baseline info when first computed for this set
                    # Key is (exercise_block, set_number) to support mid-session exercise switching
                    baseline_key = (self.exercise_block, self.current_set)
                    if baseline_key not in self.baseline_log:
                        baseline_info = self.fatigue_module.get_baseline_info()
                        if baseline_info['computed'] and baseline_info['baseline_rom'] is not None:
                            self.baseline_log[baseline_key] = baseline_info.copy()
                    
                    # Update fatigue display
                    fatigue_level = fatigue_result['fatigue_level']
                    if fatigue_level == 'High':
                        fatigue_color = "#CC0000"  # Red
                    elif fatigue_level == 'Medium':
                        fatigue_color = "#CC6600"  # Orange
                    else:
                        fatigue_color = "#006600"  # Green
                    
                    self.fatigue_label.config(
                        text=f"Deterioration: {fatigue_level}",
                        foreground=fatigue_color
                    )
                    
                    # Handle micro-break trigger
                    if fatigue_result['trigger_stop']:
                        # Severe decline - stop rule
                        self.break_label.config(
                            text="⛔ STOP: Sudden ROM decline detected. Please rest and consult PT.",
                            foreground="#CC0000"
                        )
                        self.micro_break_active = True
                        print(f"STOP RULE TRIGGERED: Severe ROM decline")
                    elif fatigue_result['trigger_break'] and not self.micro_break_active:
                        # Micro-break triggered
                        self.micro_break_active = True
                        if fatigue_level == 'Medium':
                            self.play_cue('take_break_medium', cooldown_sec=5.0)
                        else:
                            self.play_cue('take_break_high', cooldown_sec=5.0)
                        self.break_start_time = time.time()
                        self.break_duration = fatigue_result['break_duration']
                        self.break_pause_accumulated = 0  # Reset pause accumulator for new break
                        
                        if fatigue_level == 'High':
                            self.break_label.config(
                                text=f"🛑 REST {self.break_duration}s - High deterioration detected. Consider stopping.",
                                foreground="#CC0000"
                            )
                        else:
                            self.break_label.config(
                                text=f"⏸️ SHORT BREAK {self.break_duration}s - Take a moment to rest.",
                                foreground="#CC6600"
                            )

                        print(f"MICRO-BREAK: {fatigue_level} fatigue, {self.break_duration}s break")

                    # Micro-break recommendation is advisory (not hard-blocking).
                    # If a new rep completes while an advisory break timer is active,
                    # auto-dismiss the current break prompt and continue tracking.
                    if self.micro_break_active and self.break_start_time is not None and not fatigue_result['trigger_break']:
                        self.micro_break_active = False
                        self.break_start_time = None
                        self.break_pause_accumulated = 0
                        self.break_label.config(text="")

                    # Handle form cue for compensation escalation
                    if fatigue_result['trigger_form_cue']:
                        # Differentiated form cue audio
                        last_rep = self.rep_tracker.get_last_rep_info()
                        if last_rep and last_rep.get('trunk_lean_detected'):
                            self.play_cue('form_cue_trunk', cooldown_sec=15.0)
                        elif last_rep and last_rep.get('shoulder_hiking_detected'):
                            self.play_cue('form_cue_hiking', cooldown_sec=15.0)
                        else:
                            self.play_cue('form_cue_general', cooldown_sec=15.0)

                        current_text = self.break_label.cget("text")
                        # Only append to active break messages, not stale "break complete" messages
                        if self.micro_break_active:
                            # Append to active break message
                            if "Form check" not in current_text:
                                self.break_label.config(
                                    text=current_text + " | 📋 Form check: Watch your posture!",
                                    foreground="#CC6600"
                                )
                        else:
                            # No active break - show form cue standalone (clears stale messages)
                            self.break_label.config(
                                text="📋 Form check: Watch your posture!",
                                foreground="#CC6600"
                            )
                    elif not self.micro_break_active and not fatigue_result['trigger_break'] and not fatigue_result['trigger_stop']:
                        # No alerts - clear any stale break messages
                        self.break_label.config(text="")
                    
                    # Print detailed rep summary
                    metrics = fatigue_result.get('metrics', {})
                    print(f"Rep {self.rep_tracker.get_rep_count()} completed: "
                          f"Peak={peak_angle:.1f}°, ROM={rom_label}, "
                          f"TrunkLean={trunk_lean_detected}, "
                          f"Hiking={shoulder_hiking_detected}, "
                          f"Duration={duration} frames, "
                          f"Fatigue={fatigue_level}, "
                          f"ROM_decline={metrics.get('mean_rom_decline', 0):.1f}%, "
                          f"Dur_increase={metrics.get('mean_dur_increase', 0):.1f}%")
                    
                    # T13: Check if rep is unscorable due to BLC flagging
                    unscorable = self.spatial_temporal_filter.is_rep_unscorable
                    
                    # Thesis §15: deterioration_score_Fi from Mamdani fuzzy inference
                    deterioration_score = self.fatigue_module.get_fuzzy_score()
                    
                    # Log rep to session (Phase 4.12 + 5.C)
                    self.session_log.append({
                        'participant_id': self.participant_id.get().strip() or 'TEST_USER',
                        'exercise_block': self.exercise_block,  # Phase 5.C: Track exercise sessions
                        'affected_side': self.affected_side.get(),  # MISSING-1: Track affected side
                        'set_number': self.current_set,
                        'rep_number': self.rep_tracker.get_rep_count(),
                        'timestamp': datetime.now().isoformat(),
                        'exercise': exercise,
                        'peak_angle': round(peak_angle, 1) if peak_angle is not None else 0.0,
                        'duration_frames': duration,
                        'rom_label': rom_label,
                        'unscorable_flag': unscorable,  # T13: BLC-based unscorable detection
                        'tracking_valid': not self.current_rep_low_confidence,  # Thesis §15
                        'trunk_lean_detected': trunk_lean_detected,
                        'shoulder_hiking_detected': shoulder_hiking_detected,
                        'fatigue_level': fatigue_level,
                        'deterioration_score_Fi': round(deterioration_score, 2) if deterioration_score is not None else '',
                        'micro_break_triggered': fatigue_result['trigger_break'],
                        'stop_triggered': fatigue_result['trigger_stop'],
                        'form_cue_triggered': fatigue_result['trigger_form_cue'],
                        'mean_rom_decline_pct': round(metrics.get('mean_rom_decline', 0), 2),
                        'mean_dur_increase_pct': round(metrics.get('mean_dur_increase', 0), 2),
                        'valid_reps_in_window': metrics.get('valid_rep_count', 0)
                    })
                    
                    # Reset compensation and tracking-quality flags for next rep
                    self.current_rep_trunk_lean = False
                    self.current_rep_shoulder_hiking = False
                    self.current_rep_low_confidence = False
                    
                    # Reset ONLY BLC per-rep counters for the next rep.
                    # IMPORTANT: do NOT call self.spatial_temporal_filter.reset() here.
                    # full reset() clears the CW-EMA _states dict, causing a cold-start
                    # on the first frames of every new rep (transient spike in filtered
                    # angle) and also clears the BLC baseline requiring re-set every rep.
                    # reset_blc_rep_counters() resets only _blc_flagged_count and
                    # _blc_total_count while preserving both _states and _blc_baseline.
                    self.spatial_temporal_filter.reset_blc_rep_counters()
                    
                    # MISSING-5: Check if set target reached
                    if (self.reps_per_set > 0 
                            and self.rep_tracker.get_rep_count() >= self.reps_per_set
                            and not self.set_completed_prompted):
                        self.set_completed_prompted = True  # Guard against duplicate prompts
                        self.play_sound('success', audio_filename='set_complete')  # C08
                        self._handle_set_completion()
                
                # Update micro-break countdown (optimized: once per second)
                if self.micro_break_active and self.break_start_time is not None:
                    # Account for accumulated pause time
                    elapsed = time.time() - self.break_start_time - self.break_pause_accumulated
                    remaining = max(0, self.break_duration - elapsed)
                    current_time = time.time()
                    
                    if remaining > 0:
                        # Update countdown only once per second to reduce overhead
                        if current_time - self.last_countdown_update >= 1.0:
                            current_text = self.break_label.cget("text")
                            if "REST" in current_text:
                                # High fatigue break
                                self.break_label.config(
                                    text=f"🛑 REST {int(remaining)}s remaining - High deterioration detected. Consider stopping.",
                                    foreground="#CC0000"
                                )
                            elif "BREAK" in current_text:
                                # Medium fatigue break
                                self.break_label.config(
                                    text=f"⏸️ SHORT BREAK {int(remaining)}s remaining - Take a moment to rest.",
                                    foreground="#CC6600"
                                )
                            self.last_countdown_update = current_time
                    else:
                        # Break complete
                        self.break_label.config(text="✅ Break complete. Continue when ready.", foreground="#006600")
                        self.play_cue('break_complete', cooldown_sec=5.0)
                        self.micro_break_active = False
                        self.break_start_time = None
                        self.break_pause_accumulated = 0  # Reset pause accumulator
                
                # Update compensation warning label (Phase 3.4-3.5) - Combined for cleaner UI
                warnings = []
                if trunk_lean_flag:
                    warnings.append(f"⚠ Trunk Lean: {trunk_lean_angle:.1f}°")
                if shoulder_hiking_flag:
                    warnings.append(f"⚠ Shoulder Hiking: {shoulder_hiking_ratio*100:.1f}%")
                
                if warnings:
                    self.compensation_warning_label.config(text=" | ".join(warnings))
                else:
                    self.compensation_warning_label.config(text="")
                
                # Update angle displays
                if not np.isnan(flexion_angle):
                    self.flexion_label.config(text=f"Flexion: {flexion_angle:.1f}°")
                else:
                    self.flexion_label.config(text="Flexion: --")
                
                if not np.isnan(abduction_angle):
                    self.abduction_label.config(text=f"Abduction: {abduction_angle:.1f}°")
                else:
                    self.abduction_label.config(text="Abduction: --")
                
                # Calculate and display FPS with exponential smoothing
                current_time = time.time()
                # P-6: Guard against div-by-zero on first frame or camera stall
                frame_dt = current_time - self.last_frame_time
                fps_instant = 1.0 / frame_dt if frame_dt > 1e-6 else 0.0
                self.last_frame_time = current_time
                
                # Exponential smoothing: smooth_fps = smooth_fps * 0.9 + instant_fps * 0.1
                # Provides stable display without buffer overhead
                if self.fps_smoothed < 0.1:  # Essentially zero (avoid float equality)
                    self.fps_smoothed = fps_instant  # Initialize on first frame
                else:
                    self.fps_smoothed = self.fps_smoothed * 0.9 + fps_instant * 0.1
                
                # Track FPS statistics for thesis performance validation
                # Sample every fps_sample_interval seconds (not every frame) to reduce overhead
                if current_time - self.last_fps_sample_time >= self.fps_sample_interval:
                    self.fps_samples.append(self.fps_smoothed)
                    self.last_fps_sample_time = current_time
                
                # Update FPS display with smoothed value
                self.fps_label.config(text=f"FPS: {self.fps_smoothed:.1f}")
                
                # Apply visual rep flash feedback overlay (Phase 5.E)
                if self.rep_flash_active:
                    # Red flash longer (500ms) vs green (300ms) for better visibility
                    flash_duration = 0.5 if self.rep_flash_color == 'red' else 0.3
                    elapsed = time.time() - self.rep_flash_start
                    if elapsed < flash_duration:
                        # Create colored overlay
                        overlay = frame.copy()
                        if self.rep_flash_color == 'green':
                            # Green flash for correct ROM
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 200, 0), -1)
                        else:
                            # Red flash for insufficient ROM (stronger intensity)
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 220), -1)
                        # Fade effect based on time (alpha decreases over duration)
                        alpha = 0.35 * (1 - elapsed / flash_duration)
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                        if self.mirror_display.get():
                            overlay_display = display_base_frame.copy()
                            if self.rep_flash_color == 'green':
                                cv2.rectangle(overlay_display, (0, 0), (display_base_frame.shape[1], display_base_frame.shape[0]), (0, 200, 0), -1)
                            else:
                                cv2.rectangle(overlay_display, (0, 0), (display_base_frame.shape[1], display_base_frame.shape[0]), (0, 0, 220), -1)
                            display_base_frame = cv2.addWeighted(overlay_display, alpha, display_base_frame, 1 - alpha, 0)
                    else:
                        self.rep_flash_active = False
                
                # Record annotated frame AFTER all drawing (Phase 5.L)
                # Records at source resolution (before Tkinter resize) for quality
                session_label_text = None
                if self.record_annotated.get():
                    # Burn session label into annotated video (Phase 4)
                    session_label_text = self._get_annotated_session_label(calibrating=False)

                    self._draw_overlay_text(
                        frame, session_label_text, overlay_layout['session_y'],
                        (255, 255, 255), font_scale=0.58, thickness=2
                    )
                    self._write_annotated_frame(frame)

                # Build display frame (mirror uses clean frame, then HUD text redraw).
                display_frame = frame
                if self.mirror_display.get():
                    display_frame = cv2.flip(display_base_frame, 1)

                    # Re-draw session label on mirrored frame (readable)
                    if session_label_text:
                        self._draw_overlay_text(
                            display_frame, session_label_text, overlay_layout['session_y'],
                            (255, 255, 255), font_scale=0.58, thickness=2, mirror=True
                        )
                    
                    # Re-draw warning texts (top-left area) with fixed layout
                    if not pose_available:
                        if tracking_pause_active:
                            self._draw_overlay_text(
                                display_frame, "BRIEF TRACKING PAUSE", overlay_layout['warning_y'],
                                (0, 165, 255), font_scale=0.65, thickness=2, mirror=True
                            )
                        else:
                            guidance = "FRONTAL VIEW" if exercise == "Abduction" else "LATERAL VIEW"
                            self._draw_overlay_text(
                                display_frame, f"Position: {guidance} required", overlay_layout['guidance_y'],
                                (0, 165, 255), font_scale=0.65, thickness=2, mirror=True
                            )
                            self._draw_overlay_text(
                                display_frame, "Ensure full body visible", overlay_layout['instruction_y'],
                                (255, 255, 255), font_scale=0.60, thickness=2, mirror=True
                            )
                            self._draw_overlay_text(
                                display_frame, "NO POSE DETECTED", overlay_layout['warning_y'],
                                (0, 0, 255), font_scale=0.82, thickness=2, mirror=True
                            )
                    elif low_confidence and pose_available:
                        self._draw_overlay_text(
                            display_frame, "LOW DETECTION CONFIDENCE", overlay_layout['guidance_y'],
                            (0, 165, 255), font_scale=0.65, thickness=2, mirror=True
                        )
                        self._draw_overlay_text(
                            display_frame, "Improve lighting/position", overlay_layout['instruction_y'],
                            (255, 255, 255), font_scale=0.60, thickness=2, mirror=True
                        )
                    
                    # Re-draw angle texts (top-right area → mirrored to top-left)
                    angle_x = overlay_layout['angle_x']
                    if not np.isnan(flexion_angle):
                        self._draw_overlay_text(
                            display_frame, f"Flexion: {flexion_angle:.1f}deg", 30,
                            (0, 255, 0), font_scale=0.6, thickness=2, mirror=True, x=angle_x
                        )
                    if not np.isnan(abduction_angle):
                        self._draw_overlay_text(
                            display_frame, f"Abduction: {abduction_angle:.1f}deg", 60,
                            (0, 255, 0), font_scale=0.6, thickness=2, mirror=True, x=angle_x
                        )
                    if self.show_smoothed_overlay.get() and self.rep_tracker.last_smoothed_angle is not None and not np.isnan(self.rep_tracker.last_smoothed_angle):
                        self._draw_overlay_text(
                            display_frame, f"Smoothed: {self.rep_tracker.last_smoothed_angle:.1f}deg", 90,
                            (255, 255, 0), font_scale=0.55, thickness=2, mirror=True, x=angle_x
                        )
                    
                    # Re-draw compensation texts (bottom-left area)
                    comp_y = overlay_layout['compensation_y_start']
                    if trunk_lean_flag:
                        self._draw_overlay_text(
                            display_frame, f"TRUNK LEAN: {trunk_lean_angle:.1f}deg", comp_y,
                            (0, 140, 255), font_scale=0.7, thickness=2, mirror=True
                        )
                        comp_y += overlay_layout['compensation_step']
                    if shoulder_hiking_flag:
                        self._draw_overlay_text(
                            display_frame, f"SHOULDER HIKING: {shoulder_hiking_ratio*100:.1f}%", comp_y,
                            (0, 140, 255), font_scale=0.7, thickness=2, mirror=True
                        )

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display while maintaining aspect ratio (no black bars)
                height, width = frame_rgb.shape[:2]
                max_width = 900  # Increased from 800 for better visibility
                max_height = 600
                
                # Calculate scaling to fit within max dimensions while maintaining aspect ratio
                scale_w = max_width / width
                scale_h = max_height / height
                scale = min(scale_w, scale_h)  # Use smaller scale to fit within both constraints
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Convert to PhotoImage
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_label.imgtk = imgtk  # type: ignore[attr-defined]  # Keep reference
                self.video_label.config(image=imgtk, text="")
            
            # Schedule next update based on video source
            self._schedule_next_frame(frame_start_time)
        elif self.is_running and self.is_paused:
            # When paused, don't reschedule - toggle_pause() will call update_frame() to resume
            pass
    
    def show_session_summary_dialog(self):
        """Show session summary dialog before CSV export (Phase 5.E)"""
        from tkinter import messagebox
        
        if not self.session_log:
            return
        
        # Calculate summary statistics
        total_reps = len(self.session_log)
        correct_reps = sum(1 for r in self.session_log if r['rom_label'] == 'correct')
        incorrect_reps = total_reps - correct_reps
        correct_pct = (correct_reps / total_reps * 100) if total_reps > 0 else 0
        
        trunk_lean_count = sum(1 for r in self.session_log if r['trunk_lean_detected'])
        hiking_count = sum(1 for r in self.session_log if r['shoulder_hiking_detected'])
        
        avg_peak = sum(r['peak_angle'] for r in self.session_log) / total_reps if total_reps > 0 else 0
        
        # Count fatigue events
        medium_fatigue = sum(1 for r in self.session_log if r['fatigue_level'] == 'Medium')
        high_fatigue = sum(1 for r in self.session_log if r['fatigue_level'] == 'High')
        
        # Get exercises performed
        exercises = list(set(r['exercise'] for r in self.session_log))
        exercise_str = ", ".join(sorted(exercises))
        
        # Build summary message
        summary = f"""SESSION COMPLETE

📊 Performance Summary:
━━━━━━━━━━━━━━━━━━━━━━━
Total Repetitions: {total_reps}
✅ Correct ROM: {correct_reps} ({correct_pct:.0f}%)
❌ Insufficient ROM: {incorrect_reps}
📐 Average Peak Angle: {avg_peak:.1f}°

⚠️ Compensation Detected:
Trunk Lean: {trunk_lean_count} reps
Shoulder Hiking: {hiking_count} reps

💪 Deterioration Events:
Medium: {medium_fatigue}
High: {high_fatigue}

🏋️ Exercise(s): {exercise_str}
🦾 Affected Side: {self.affected_side.get()}

Session data will be saved to logs/{self.participant_id.get().strip() or 'TEST_USER'}/ folder."""
        
        messagebox.showinfo("Session Summary", summary)
    
    # ── Display Text Rendering (Phase 6) ──────────────────────────────

    def _get_annotated_session_label(self, calibrating=False):
        """Build burn-in label text for annotated video frames only."""
        exercise_name = self.current_exercise.get()
        set_num = self.current_set

        if calibrating:
            return f"{exercise_name} | Set {set_num} | Calibrating..."

        if self.set_completed_prompted and not self.rep_tracker.in_rep:
            return f"{exercise_name} | Set {set_num} Complete"

        rep_count = self.rep_tracker.get_rep_count()
        if rep_count <= 0:
            return f"{exercise_name} | Set {set_num}"

        last_rep = self.rep_tracker.get_last_rep_info()
        rom_label = last_rep['rom_label'] if last_rep else None
        if rom_label:
            return f"{exercise_name} | Set {set_num} | Rep {rep_count} | ROM: {rom_label}"

        return f"{exercise_name} | Set {set_num} | Rep {rep_count} | ROM: pending"

    def _get_overlay_layout(self, frame):
        """Return stable overlay text positions to avoid on-screen overlap."""
        h, w = frame.shape[:2]
        return {
            'left_x': 10,
            'session_y': 25,
            'guidance_y': 55,
            'instruction_y': 85,
            'warning_y': 120,
            'warning_step': 30,
            'compensation_y_start': max(40, h - 70),
            'compensation_step': 35,
            'angle_x': max(10, w - 250),
        }

    def _clamp_text_position(self, frame, text, x, y, font, scale, thickness, padding=4):
        """Clamp text origin so the full text stays visible within frame bounds."""
        h, w = frame.shape[:2]
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        min_x = padding
        max_x = max(min_x, w - text_w - padding)
        clamped_x = max(min_x, min(int(x), max_x))

        min_y = text_h + baseline + padding
        max_y = max(min_y, h - padding)
        clamped_y = max(min_y, min(int(y), max_y))

        return clamped_x, clamped_y

    def _draw_overlay_text(self, frame, text, y, color, font_scale=0.65,
                           thickness=2, mirror=False, x=None,
                           line_type=cv2.LINE_AA):
        """Draw overlay text with safe clamped positioning.

        Note: `mirror` is kept for backward compatibility with existing call sites.
        Text is always drawn readable (left-to-right) and never coordinate-mirrored.
        """
        if not text:
            return

        x_pos = 10 if x is None else int(x)
        y_pos = int(y)
        x_pos, y_pos = self._clamp_text_position(
            frame, text, x_pos, y_pos,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        cv2.putText(
            frame, text, (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type
        )
    
    def _put_text_mirrored(self, frame, text, org, font, scale, color, thickness, line_type=cv2.LINE_8, mirror=False):
        """
        Legacy helper retained for compatibility with older call sites.
        Text is drawn readable and clamped; coordinates are not mirrored.
        """
        x, y = org

        x, y = self._clamp_text_position(frame, text, x, y, font, scale, thickness)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, line_type)
    

    # ── Session Recording (Phase 5.L) ──────────────────────────────────
    
    def _init_recording(self, participant_id):
        """
        Initialize video recording writers if enabled (Phase 5.L).
        
        Uses lazy initialization: writers are created on the first frame
        (in _write_raw_frame / _write_annotated_frame) so we get the actual 
        frame dimensions from the camera/video, not guessed values.
        
        Codec: MJPG — fast encoding, no external dependencies, good quality.
        - CPU cost: ~1-3ms per write() call (negligible vs MediaPipe ~30ms).
        - File size: ~5-15 MB/min at 640x480, acceptable for session recordings.
        """
        self.video_writer_annotated = None
        self.video_writer_raw = None
        self.recording_paths = {}
        self._recording_initialized = False  # Lazy init flag
        
        want_annotated = self.record_annotated.get()
        want_raw = self.record_raw.get()
        
        # Disable checkboxes during session to prevent mid-session toggling
        self.record_annotated_cb.config(state=tk.DISABLED)
        self.record_raw_cb.config(state=tk.DISABLED)
        
        if not want_annotated and not want_raw:
            self.recording_status_label.config(text="")
            return
        
        # Build output paths using per-participant log directory
        exercise = self.current_exercise.get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = self._get_participant_log_dir()
        os.makedirs(logs_dir, exist_ok=True)
        
        base_name = f"session_{participant_id}_{exercise}_{timestamp}"
        
        if want_annotated:
            path = os.path.join(logs_dir, f"{base_name}_annotated.avi")
            self.recording_paths['annotated'] = path
        
        if want_raw:
            path = os.path.join(logs_dir, f"{base_name}_raw.avi")
            self.recording_paths['raw'] = path

        # Mark for lazy init (actual VideoWriter created on first frame)
        self._recording_initialized = False
        
        # Update UI status
        rec_types = []
        if want_annotated:
            rec_types.append("Annotated")
        if want_raw:
            rec_types.append("Raw")
        self.recording_status_label.config(
            text=f"▶ REC: {' + '.join(rec_types)}", foreground="#CC0000"
        )
    
    def _lazy_init_writers(self, frame):
        """
        Create VideoWriter objects on the first frame (lazy initialization).
        
        This ensures we use the actual frame dimensions from the camera/video,
        avoiding dimension mismatches that cause silent write failures.
        
        Recording FPS strategy:
        - Camera mode: use 15 FPS (typical effective rate after MediaPipe processing)
        - File mode: match the source video FPS for accurate playback speed
        """
        if self._recording_initialized:
            return
        
        h, w = frame.shape[:2]
        fourcc_func = getattr(cv2, "VideoWriter_fourcc")
        fourcc = fourcc_func(*"MJPG")
        
        # Determine recording FPS (Fix C: use measured effective FPS for camera mode)
        if self.video_mode.get() == "file":
            rec_fps = self._sanitize_fps(self.video_fps, fallback=30.0)
        else:
            # Camera mode: use measured effective FPS if available (accurate playback).
            # Camera-reported CAP_PROP_FPS is often unreliable (e.g. reports 30 but
            # actual processing rate is ~15-18 FPS after MediaPipe). Using the
            # camera-reported value causes exported video to play fast-forwarded.
            if self._effective_recording_fps is not None:
                rec_fps = self._sanitize_fps(self._effective_recording_fps, fallback=15.0)
            else:
                # Fallback: conservative estimate matching typical processing throughput
                rec_fps = 15.0
            print(f"[Recording] Using recording FPS: {rec_fps:.1f}")
        
        if 'annotated' in self.recording_paths:
            self.video_writer_annotated = cv2.VideoWriter(
                self.recording_paths['annotated'], fourcc, rec_fps, (w, h)
            )
            if not self.video_writer_annotated.isOpened():
                print(f"Warning: Could not open annotated video writer")
                self.video_writer_annotated = None
        
        if 'raw' in self.recording_paths:
            self.video_writer_raw = cv2.VideoWriter(
                self.recording_paths['raw'], fourcc, rec_fps, (w, h)
            )
            if not self.video_writer_raw.isOpened():
                print(f"Warning: Could not open raw video writer")
                self.video_writer_raw = None
        
        self._recording_initialized = True
    
    def _write_raw_frame(self, frame):
        """
        Write a raw (clean, pre-annotation) frame to the raw video writer.
        Called BEFORE any cv2.putText / draw_landmarks calls in update_frame().
        
        Performance note: cv2.VideoWriter.write() is ~1-3ms per call.
        We copy the frame here because the original will be drawn on in-place.
        """
        if not self._recording_initialized:
            self._lazy_init_writers(frame)
        
        if self.video_writer_raw is not None:
            # Must copy because the frame will be modified in-place by drawing calls
            try:
                self.video_writer_raw.write(frame.copy())
            except Exception as e:
                print(f"[Recording] Raw write error: {e}")
                try:
                    self.video_writer_raw.release()
                except Exception:
                    pass
                self.video_writer_raw = None
    
    def _write_annotated_frame(self, frame):
        """
        Write an annotated (with skeleton overlay) frame to the annotated video writer.
        Called AFTER all drawing calls but BEFORE Tkinter resize in update_frame().
        
        Performance note: No copy needed — frame is fully drawn and about to be
        converted to RGB + resized for display anyway.
        """
        if not self._recording_initialized:
            self._lazy_init_writers(frame)
        
        if self.video_writer_annotated is not None:
            try:
                self.video_writer_annotated.write(frame)
            except Exception as e:
                print(f"[Recording] Annotated write error: {e}")
                try:
                    self.video_writer_annotated.release()
                except Exception:
                    pass
                self.video_writer_annotated = None


    
    def _release_recording(self):
        """
        Release all video writers and re-enable recording checkboxes.
        Called from stop_video() and on_closing().
        """
        saved_files = []
        
        if self.video_writer_annotated is not None:
            self.video_writer_annotated.release()
            self.video_writer_annotated = None
            if 'annotated' in self.recording_paths:
                saved_files.append(self.recording_paths['annotated'])
        
        if self.video_writer_raw is not None:
            self.video_writer_raw.release()
            self.video_writer_raw = None
            if 'raw' in self.recording_paths:
                saved_files.append(self.recording_paths['raw'])


        
        self.recording_paths = {}
        self._recording_initialized = False
        
        # Re-enable checkboxes
        try:
            self.record_annotated_cb.config(state=tk.NORMAL)
            self.record_raw_cb.config(state=tk.NORMAL)
            self.recording_status_label.config(text="")
        except (AttributeError, tk.TclError):
            pass  # UI may not exist during shutdown
        
        # Log saved video files
        if saved_files:
            for f in saved_files:
                print(f"[Recording] Saved: {f}")

        # Phase 4: Post-session duration validation
        if saved_files and self._recording_frame_count > 0:
            session_elapsed = time.time() - (self._recording_start_time or time.time())
            if session_elapsed > 1.0:
                actual_fps = self._recording_frame_count / session_elapsed
                # Compare with writer FPS
                writer_fps = self.session_fps if self.session_fps > 0 else 20.0
                expected_duration = self._recording_frame_count / writer_fps
                ratio = expected_duration / session_elapsed if session_elapsed > 0 else 1.0
                if abs(ratio - 1.0) > 0.10:  # > 10% mismatch
                    print(f"[Recording] ⚠ Duration mismatch: {ratio:.2f}x "
                          f"(writer FPS={writer_fps:.1f}, actual FPS={actual_fps:.1f}, "
                          f"frames={self._recording_frame_count}, elapsed={session_elapsed:.1f}s)")
                    print(f"[Recording] Video may play {'slower' if ratio > 1.0 else 'faster'} than real-time")
                else:
                    print(f"[Recording] ✓ Duration match OK (ratio={ratio:.2f})")
    
    def export_session_csv(self):
        """
        Export session log to CSV file (Phase 4.12).
        
        Enhanced version with metadata and summary statistics.
        Saves to logs/<participant_id>/ directory with timestamp filename.
        """
        if not self.session_log:
            print("No session data to export")
            return
        
        # Create filename with timestamp (Phase 5.A + 5.C)
        timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S') if self.session_start_time else datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get participant ID (Phase 5.A)
        participant_id = self.participant_id.get().strip() or "TEST_USER"
        
        # Get all unique exercises in session (Phase 5.C)
        exercises = list(set(rep['exercise'] for rep in self.session_log))
        exercise_str = "_".join(sorted(exercises)) if len(exercises) > 1 else exercises[0]
        
        filename = f"session_{participant_id}_{exercise_str}_{timestamp}.csv"
        
        # Ensure per-participant log directory exists
        logs_dir = self._get_participant_log_dir()
        filepath = os.path.join(logs_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                # Write metadata section (comments starting with #)
                csvfile.write("# Shoulder Mobility Assessment System - Session Export\n")
                csvfile.write(f"# Participant ID: {participant_id}\n")
                
                # List all exercises if multiple (Phase 5.C)
                if len(exercises) > 1:
                    csvfile.write(f"# Exercise Types: {', '.join(sorted(exercises))} (multi-exercise session)\n")
                else:
                    csvfile.write(f"# Exercise Type: {exercises[0]}\n")
                
                csvfile.write(f"# Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write(f"# Affected Side: {self.affected_side.get()}\n")  # MISSING-1
                csvfile.write(f"# Total Sets: {self.total_sets}\n")
                
                # Session notes removed (fix task B) — write empty for CSV compatibility
                csvfile.write("# Session Notes: (none)\n")
                
                # FPS performance metric (for thesis validation)
                fps_avg = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else self.fps_smoothed
                csvfile.write(f"# Average FPS: {fps_avg:.1f}\n")
                
                # Write per-set baseline info (key is now (exercise_block, set_number))
                if self.baseline_log:
                    csvfile.write("#\n")
                    for key in sorted(self.baseline_log.keys()):
                        block, set_num = key
                        baseline_info = self.baseline_log[key]
                        # Find exercise name for this block
                        exercise_for_block = next(
                            (r['exercise'] for r in self.session_log if r.get('exercise_block') == block), 
                            'Unknown'
                        )
                        csvfile.write(f"# {exercise_for_block} Block {block} Set {set_num} Baseline ROM: {baseline_info['baseline_rom']:.1f}°\n")
                        csvfile.write(f"# {exercise_for_block} Block {block} Set {set_num} Baseline Duration: {baseline_info['baseline_dur']:.0f} frames\n")
                        if baseline_info.get('low_confidence', False):
                            csvfile.write(f"# {exercise_for_block} Block {block} Set {set_num} Baseline: LOW CONFIDENCE\n")
                
                # System configuration
                rom_threshold = self.config.rom_threshold if self.config else 150
                fatigue_thresholds = self.config.get_fatigue_thresholds() if self.config else {}
                medium_threshold = fatigue_thresholds.get('medium', 10)
                high_threshold = fatigue_thresholds.get('high', 20)
                severe_threshold = fatigue_thresholds.get('severe', 30)
                fatigue_window_size = fatigue_thresholds.get('window_size', 5)

                csvfile.write(f"# MediaPipe Model Complexity: 1\n")
                csvfile.write(f"# MediaPipe Detection Confidence: {self.pose_processor.detection_confidence}\n")  # MISSING-2
                csvfile.write(f"# MediaPipe Tracking Confidence: {self.pose_processor.tracking_confidence}\n")  # MISSING-2
                csvfile.write(f"# Reps Per Set Target: {self.reps_per_set}\n")  # MISSING-5
                csvfile.write(f"# ROM Threshold (correct): ≥{rom_threshold}°\n")
                csvfile.write(
                    f"# Deterioration Thresholds: Medium={medium_threshold}%, High={high_threshold}%, "
                    f"Severe={severe_threshold}%\n"
                )
                csvfile.write(f"# Sliding Window Size: {fatigue_window_size} reps\n")
                csvfile.write("#\n")
                
                # Define CSV columns — thesis §15 output contract
                fieldnames = [
                    'participant_id',  # Thesis §15: per-row participant ID
                    'exercise_block',  # Phase 5.C: Track separate exercise sessions
                    'affected_side',   # MISSING-1: Track affected side per rep
                    'set_number',
                    'rep_number',
                    'timestamp',
                    'exercise',
                    'peak_angle',
                    'duration_frames',
                    'rom_label',
                    'unscorable_flag',
                    'tracking_valid',  # Thesis §15: no low-confidence frames in rep
                    'trunk_lean_detected',
                    'shoulder_hiking_detected',
                    'fatigue_level',
                    'deterioration_score_Fi',  # Thesis §15: Mamdani fuzzy score (0-100)
                    'micro_break_triggered',
                    'stop_triggered',
                    'form_cue_triggered',
                    'mean_rom_decline_pct',
                    'mean_dur_increase_pct',
                    'valid_reps_in_window'
                ]
                
                # Write rep-level data
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.session_log)
                
                # Write summary statistics
                csvfile.write("\n# SESSION SUMMARY\n")
                
                total_reps = len(self.session_log)
                correct_reps = sum(1 for r in self.session_log if r['rom_label'] == 'correct')
                insufficient_reps = total_reps - correct_reps
                
                trunk_lean_count = sum(1 for r in self.session_log if r['trunk_lean_detected'])
                hiking_count = sum(1 for r in self.session_log if r['shoulder_hiking_detected'])
                
                medium_fatigue_count = sum(1 for r in self.session_log if r['fatigue_level'] == 'Medium')
                high_fatigue_count = sum(1 for r in self.session_log if r['fatigue_level'] == 'High')
                
                micro_breaks = sum(1 for r in self.session_log if r['micro_break_triggered'])
                stop_alerts = sum(1 for r in self.session_log if r['stop_triggered'])
                
                avg_peak_angle = sum(r['peak_angle'] for r in self.session_log) / total_reps if total_reps > 0 else 0
                avg_duration = sum(r['duration_frames'] for r in self.session_log) / total_reps if total_reps > 0 else 0
                
                csvfile.write(f"# Total Reps: {total_reps}\n")
                csvfile.write(f"# Correct ROM: {correct_reps} ({correct_reps/total_reps*100:.1f}%)\n")
                csvfile.write(f"# Insufficient ROM: {insufficient_reps} ({insufficient_reps/total_reps*100:.1f}%)\n")
                csvfile.write(f"# Average Peak Angle: {avg_peak_angle:.1f}°\n")
                csvfile.write(f"# Average Duration: {avg_duration:.1f} frames\n")
                csvfile.write(f"# Trunk Lean Detected: {trunk_lean_count} reps ({trunk_lean_count/total_reps*100:.1f}%)\n")
                csvfile.write(f"# Shoulder Hiking Detected: {hiking_count} reps ({hiking_count/total_reps*100:.1f}%)\n")
                csvfile.write(f"# Medium Deterioration Events: {medium_fatigue_count}\n")
                csvfile.write(f"# High Deterioration Events: {high_fatigue_count}\n")
                csvfile.write(f"# Micro-Breaks Triggered: {micro_breaks}\n")
                csvfile.write(f"# Stop Alerts: {stop_alerts}\n")
                csvfile.write("#\n")
                
                # Anomaly detection
                csvfile.write("# ANOMALIES DETECTED:\n")
                anomalies_found = False
                
                # Check for extremely long duration reps (>5 seconds @ typical FPS)
                long_reps = [r for r in self.session_log if r['duration_frames'] > 100]
                if long_reps:
                    csvfile.write(f"# - {len(long_reps)} rep(s) with unusually long duration (>100 frames):\n")
                    for r in long_reps:
                        csvfile.write(f"#   Set {r['set_number']}, Rep {r['rep_number']}: {r['duration_frames']} frames\n")
                    anomalies_found = True
                
                # Check for compensation rate >50%
                # LC-9: CSV anomaly threshold from config (default 0.5 = 50%)
                _comp_rate_threshold = self.config.high_compensation_rate_pct if self.config else 0.5
                
                if trunk_lean_count / total_reps > _comp_rate_threshold:
                    csvfile.write(f"# - High trunk lean rate: {trunk_lean_count/total_reps*100:.1f}% (>{_comp_rate_threshold*100:.0f}% threshold)\n")
                    anomalies_found = True
                
                if hiking_count / total_reps > _comp_rate_threshold:
                    csvfile.write(f"# - High shoulder hiking rate: {hiking_count/total_reps*100:.1f}% (>{_comp_rate_threshold*100:.0f}% threshold)\n")
                    csvfile.write("#   Consider checking threshold sensitivity or form execution\n")
                    anomalies_found = True
                
                if not anomalies_found:
                    csvfile.write("# - None detected\n")
            
            print(f"Session exported to: {filepath}")
            print(f"Total reps logged: {total_reps}")
            self.status_label.config(text=f"Status: Stopped - Session saved to {filename}")
            
            # Export summary statistics to separate file (Phase 5.L)
            self.export_summary_file(participant_id, exercises, timestamp, total_reps, 
                                     correct_reps, avg_peak_angle, trunk_lean_count, 
                                     hiking_count, medium_fatigue_count, high_fatigue_count)
        except Exception as e:
            print(f"Error exporting session CSV: {e}")
            self.status_label.config(text=f"Status: Stopped - Error saving session")
    
    def export_summary_file(self, participant_id, exercises, timestamp, total_reps,
                           correct_reps, avg_peak_angle, trunk_lean_count, 
                           hiking_count, medium_fatigue_count, high_fatigue_count):
        """Export summary statistics to a separate text file (Phase 5.L)"""
        try:
            exercise_str = "_".join(sorted(exercises)) if len(exercises) > 1 else exercises[0]
            filename = f"summary_{participant_id}_{exercise_str}_{timestamp}.txt"
            
            logs_dir = self._get_participant_log_dir()
            filepath = os.path.join(logs_dir, filename)
            
            correct_pct = (correct_reps / total_reps * 100) if total_reps > 0 else 0
            trunk_pct = (trunk_lean_count / total_reps * 100) if total_reps > 0 else 0
            hiking_pct = (hiking_count / total_reps * 100) if total_reps > 0 else 0
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("  MOBILITY ASSESSMENT SESSION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Participant ID: {participant_id}\n")
                f.write(f"Affected Side: {self.affected_side.get()}\n")  # MISSING-1
                f.write(f"Exercise(s): {', '.join(sorted(exercises))}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Sets: {self.total_sets}\n")
                
                # Session notes removed (fix task B)
                f.write("\n")
                
                f.write("-" * 50 + "\n")
                f.write("  PERFORMANCE METRICS\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Repetitions: {total_reps}\n")
                f.write(f"Correct ROM (>=150 deg): {correct_reps} ({correct_pct:.1f}%)\n")
                f.write(f"Insufficient ROM: {total_reps - correct_reps} ({100 - correct_pct:.1f}%)\n")
                f.write(f"Average Peak Angle: {avg_peak_angle:.1f} deg\n\n")
                
                f.write("-" * 50 + "\n")
                f.write("  COMPENSATION ANALYSIS\n")
                f.write("-" * 50 + "\n")
                f.write(f"Trunk Lean Detected: {trunk_lean_count} reps ({trunk_pct:.1f}%)\n")
                f.write(f"Shoulder Hiking Detected: {hiking_count} reps ({hiking_pct:.1f}%)\n\n")
                
                f.write("-" * 50 + "\n")
                f.write("  DETERIORATION INDICATORS\n")
                f.write("-" * 50 + "\n")
                f.write(f"Medium Deterioration Events: {medium_fatigue_count}\n")
                f.write(f"High Deterioration Events: {high_fatigue_count}\n\n")
                
                # Performance rating
                f.write("-" * 50 + "\n")
                f.write("  OVERALL ASSESSMENT\n")
                f.write("-" * 50 + "\n")
                
                if correct_pct >= 80 and trunk_pct < 20 and hiking_pct < 20:
                    rating = "EXCELLENT"
                    note = "Patient shows good ROM and minimal compensation."
                elif correct_pct >= 60 and trunk_pct < 40:
                    rating = "GOOD"
                    note = "Acceptable performance with room for improvement."
                elif correct_pct >= 40:
                    rating = "NEEDS IMPROVEMENT"
                    note = "Consider modifying exercise prescription."
                else:
                    rating = "REQUIRES ATTENTION"
                    note = "Significant difficulties observed. PT review recommended."
                
                f.write(f"Rating: {rating}\n")
                f.write(f"Notes: {note}\n")
                f.write("\n" + "=" * 50 + "\n")
                
            print(f"Summary exported to: {filepath}")
        except Exception as e:
            print(f"Error exporting summary: {e}")
    
    def previous_set(self):
        """Navigate to previous set with data-loss confirmation (BUG-7 fix)"""
        if self.current_set > 1:
            # Check if current set has in-progress data that would be lost (BUG-7 fix)
            has_current_set_data = self.rep_tracker.get_rep_count() > 0 or self.rep_tracker.in_rep
            if has_current_set_data:
                response = messagebox.askyesno(
                    "Confirm Set Change",
                    f"Set {self.current_set} has {self.rep_tracker.get_rep_count()} rep(s) recorded.\n"
                    f"Going to the previous set will reset current tracking data.\n\n"
                    f"Note: Already-logged reps in the session remain saved.\n\n"
                    f"Continue?"
                )
                if not response:
                    return
            
            self.current_set -= 1
            self.set_label.config(text=f"Set: {self.current_set} of {self.total_sets}")
            
            # Reset trackers for new set
            self.rep_tracker.reset()
            self.rep_tracker.set_exercise(self.current_exercise.get())
            self.rep_label.config(text="Reps: 0", foreground="#006600")
            
            self.fatigue_module.reset()
            self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
            self.break_label.config(text="")
            self.micro_break_active = False
            self._reset_current_rep_compensation()  # BUG-4 fix
            self.compensation_warning_label.config(text="")
            self.set_completed_prompted = False  # MISSING-5: Reset for new set
            self.require_calibration_pass = False
            
            self.status_label.config(text=f"Status: Ready - Set {self.current_set} of {self.total_sets}")
    
    def next_set(self, auto_from_completion=False):
        """Navigate to next set.

        When running, starts a fresh calibration phase for the new set and
        resets BLC baseline/counters before rep tracking continues.
        """
        if self.current_set < self.total_sets:
            self.current_set += 1
            self.set_label.config(text=f"Set: {self.current_set} of {self.total_sets}")
            
            # Reset trackers for new set
            self.rep_tracker.reset()
            self.rep_tracker.set_exercise(self.current_exercise.get())
            self.rep_label.config(text="Reps: 0", foreground="#006600")
            
            self.fatigue_module.reset()
            self.fatigue_label.config(text="Deterioration: Low", foreground="#006600")
            self.break_label.config(text="")
            self.micro_break_active = False
            self._reset_current_rep_compensation()  # BUG-4 fix
            self.compensation_warning_label.config(text="")
            self.set_completed_prompted = False  # MISSING-5: Reset for new set

            # Set-transition BLC reset (baseline + rep counters) before recalibration.
            self.spatial_temporal_filter.reset_blc()
            self.baseline_limb_length = {}

            # Re-run calibration for new set before rep tracking resumes.
            if self.is_running:
                self.calibration_phase_active = True
                self.calibration_waiting_for_detection = True
                self.calibration_countdown_started = False
                self.calibration_waiting_start_time = time.time()
                self.calibration_start_time = None
                self.calibration_landmarks_buffer = []
                self.calibration_segment_lengths = []
                self.calibration_data = {}
                self.calibration_passed = False
                self.require_calibration_pass = True
                self._countdown_last_tone = 0
                self._calibration_retry_count = 0

                # Auto-flow from set-complete prompt should immediately continue
                # into calibration (tracking remains gated by calibration_phase_active).
                if auto_from_completion and self.is_paused:
                    self.toggle_pause()

                self.status_label.config(
                    text=f"Status: Set {self.current_set} - Recalibrating before reps...",
                    foreground="#0066cc"
                )
            
            if not self.is_running:
                self.status_label.config(text=f"Status: Ready - Set {self.current_set} of {self.total_sets}")
    
    def on_closing(self):
        """Handle window close event with confirmation during active session (Phase 5.B)"""
        # Confirm if session is running (Phase 5.B)
        if self.is_running:
            from tkinter import messagebox
            response = messagebox.askyesnocancel(
                "Confirm Exit",
                "Session is still running. Do you want to save and exit?\n\n"
                "Yes = Save and exit\n"
                "No = Exit without saving\n"
                "Cancel = Continue session"
            )
            
            if response is None:  # Cancel
                return
            elif response:  # Yes - save
                self.stop_video()  # This triggers CSV export
            else:  # No - don't save
                self.is_running = False
                self._release_recording()  # Release recording writers (Phase 5.L)
                if self.cap:
                    self.cap.release()
        else:
            self.stop_video()
        
        # Release MediaPipe resources on shutdown (BUG-6 fix)
        # Safe to call even if already released — release() is idempotent
        if not self._pose_released:
            try:
                self.pose_processor.release()
                self._pose_released = True
            except Exception:
                pass  # Best-effort cleanup on shutdown
        
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = RehabApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
