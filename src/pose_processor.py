"""
Pose Processor Module
Handles MediaPipe BlazePose integration, landmark extraction, and angle computation
"""

import cv2
import numpy as np
import mediapipe as mp


class PoseProcessor:
    """Process video frames to extract pose landmarks and compute shoulder angles"""
    
    # MediaPipe landmark indices (constant across all instances)
    # MediaPipe landmarks are labeled from the SUBJECT's perspective:
    #   MediaPipe "LEFT"  = Subject's anatomical LEFT side
    #   MediaPipe "RIGHT" = Subject's anatomical RIGHT side
    MP_LANDMARKS = {
        'LEFT_SHOULDER': 11,  'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13,     'RIGHT_ELBOW': 14,
        'LEFT_HIP': 23,       'RIGHT_HIP': 24,
        'LEFT_EAR': 7,        'RIGHT_EAR': 8,
    }
    
    def __init__(self, model_complexity=1, camera_view='frontal', config=None,
                 affected_side='Right'):
        """
        Initialize MediaPipe Pose
        
        Args:
            model_complexity: 0 (Lite), 1 (Full), 2 (Heavy) - default 1
            camera_view: 'frontal' (abduction) or 'lateral' (flexion) - default 'frontal'
            config: ConfigLoader instance for threshold configuration (optional)
            affected_side: 'Right' or 'Left' — which anatomical side is being rehabilitated
        """
        self.config = config
        self.mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
        self.mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]
        
        # MediaPipe confidence thresholds (MISSING-2: configurable via config.json)
        if self.config:
            detection_conf = self.config.mediapipe_detection_confidence
            tracking_conf = self.config.mediapipe_tracking_confidence
        else:
            detection_conf = 0.5
            tracking_conf = 0.5
        
        # Initialize pose estimator
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        
        # Store for auditing/export (MISSING-2)
        self.detection_confidence = detection_conf
        self.tracking_confidence = tracking_conf
        
        # Affected side (MISSING-1): determines which anatomical arm is tracked
        # 'Right' = track user's right arm (MediaPipe RIGHT indices: 12,14,24,8)
        # 'Left'  = track user's left arm  (MediaPipe LEFT indices: 11,13,23,7)
        self.affected_side = affected_side
        
        # Compensation Detection Thresholds (Phase 3.4-3.5)
        if self.config:
            self.TRUNK_LEAN_THRESHOLD = self.config.trunk_lean_threshold
            self.SHOULDER_HIKING_THRESHOLD = self.config.shoulder_hiking_threshold
        else:
            self.TRUNK_LEAN_THRESHOLD = 15.0    # degrees deviation from vertical
            self.SHOULDER_HIKING_THRESHOLD = 0.45  # 45% asymmetry ratio (thesis §3.6.4)
        
        # Build side-aware landmark sets and set initial view
        self.camera_view = camera_view
        self._rebuild_landmarks()
        
        # Visibility threshold
        self.VISIBILITY_THRESHOLD = 0.5
    
    def _rebuild_landmarks(self):
        """Build landmark dictionaries for current affected_side and camera_view.
        
        MISSING-1: Centralized side-aware landmark mapping.
        
        Uses generic keys so that downstream code (compute_angles, compensation)
        does not branch on side. Key naming convention:
        
        TRACKED_*  = the affected (rehabilitated) side
        CONTRA_*   = the contralateral (unaffected) side
        
        MediaPipe landmark convention (subject's perspective):
          affected_side='Right' → tracked = MediaPipe RIGHT indices (12,14,24,8)
          affected_side='Left'  → tracked = MediaPipe LEFT  indices (11,13,23,7)

        This applies to both frontal and lateral views. Mirror display
        is visual-only and occurs after pose processing.
        """
        MP = self.MP_LANDMARKS
        
        if self.affected_side == 'Right':
            # Tracked = MediaPipe RIGHT (subject's anatomical RIGHT)
            tracked_shoulder = MP['RIGHT_SHOULDER']
            tracked_elbow    = MP['RIGHT_ELBOW']
            tracked_hip      = MP['RIGHT_HIP']
            tracked_ear      = MP['RIGHT_EAR']
            contra_shoulder  = MP['LEFT_SHOULDER']
            contra_hip       = MP['LEFT_HIP']
            contra_ear       = MP['LEFT_EAR']
        else:
            # Tracked = MediaPipe LEFT (subject's anatomical LEFT)
            tracked_shoulder = MP['LEFT_SHOULDER']
            tracked_elbow    = MP['LEFT_ELBOW']
            tracked_hip      = MP['LEFT_HIP']
            tracked_ear      = MP['LEFT_EAR']
            contra_shoulder  = MP['RIGHT_SHOULDER']
            contra_hip       = MP['RIGHT_HIP']
            contra_ear       = MP['RIGHT_EAR']
        
        # Frontal view: both sides visible
        self.FRONTAL_LANDMARKS = {
            'TRACKED_SHOULDER': tracked_shoulder,
            'TRACKED_ELBOW':    tracked_elbow,
            'TRACKED_HIP':      tracked_hip,
            'TRACKED_EAR':      tracked_ear,
            'CONTRA_SHOULDER':  contra_shoulder,
            'CONTRA_HIP':       contra_hip,
            'CONTRA_EAR':       contra_ear,
        }
        
        # Lateral view: only the affected (tracked) side is visible
        # TRACKED_EAR excluded: during flexion, arm routinely occludes face.
        # Ear is not needed for flexion angle computation or lateral trunk lean.
        # Shoulder hiking is N/A for lateral view (returns early in compute_shoulder_hiking).
        self.LATERAL_LANDMARKS = {
            'TRACKED_SHOULDER': tracked_shoulder,
            'TRACKED_ELBOW':    tracked_elbow,
            'TRACKED_HIP':      tracked_hip,
        }
        
        # Apply current camera view
        if self.camera_view == 'lateral':
            self.LANDMARKS = self.LATERAL_LANDMARKS
        else:
            self.LANDMARKS = self.FRONTAL_LANDMARKS
    
    def set_affected_side(self, side):
        """Change the affected (tracked) side and rebuild landmark sets.
        
        Args:
            side: 'Right' or 'Left'
        """
        if side not in ('Right', 'Left'):
            raise ValueError("affected_side must be 'Right' or 'Left'")
        self.affected_side = side
        self._rebuild_landmarks()
        
    def process_frame(self, frame):
        """
        Process a single frame to extract landmarks
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            results: MediaPipe pose results object (or None if no detection)
            landmarks_dict: Dictionary of landmark coordinates {name: (x, y, visibility)}
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)  # type: ignore[union-attr]
        
        # Extract landmarks if detected
        landmarks_dict = {}
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            
            for name, idx in self.LANDMARKS.items():
                landmark = results.pose_landmarks.landmark[idx]
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * w
                y = landmark.y * h
                visibility = landmark.visibility
                landmarks_dict[name] = (x, y, visibility)
            
            # T12: Add raw ear landmarks for head-tilt computation
            # Head tilt must use the raw (non-side-remapped) ears
            left_ear_lm = results.pose_landmarks.landmark[self.MP_LANDMARKS['LEFT_EAR']]
            right_ear_lm = results.pose_landmarks.landmark[self.MP_LANDMARKS['RIGHT_EAR']]
            landmarks_dict['left_ear_raw'] = (left_ear_lm.x * w, left_ear_lm.y * h, left_ear_lm.visibility)
            landmarks_dict['right_ear_raw'] = (right_ear_lm.x * w, right_ear_lm.y * h, right_ear_lm.visibility)
        
        return results, landmarks_dict
    
    def compute_angles(self, landmarks_dict):
        """
        Compute shoulder joint angle using trunk-relative method.
        
        Uses generic TRACKED_*/CONTRA_* landmark keys set by _rebuild_landmarks(),
        so this method is side-agnostic (MISSING-1).
        
        Unified formula for both exercises:
        - Frontal view: angle represents shoulder abduction
        - Lateral view: angle represents shoulder flexion
        
        Angle convention:
        - 0° = arm at side (neutral position)
        - 90° = arm at shoulder level (horizontal)
        - 180° = arm overhead (full elevation)
        
        Research validation: van den Hoorn et al. (2024, Sensors)
        - R² = 0.98 for both abduction and flexion
        - DOI: 10.3390/s24020534
        
        Args:
            landmarks_dict: Dictionary of landmarks from process_frame()
            
        Returns:
            flexion_angle: Shoulder flexion angle in degrees (or np.nan if invalid)
            abduction_angle: Shoulder abduction angle in degrees (or np.nan if invalid)
            low_confidence: Boolean flag if any landmarks have low visibility
        """
        flexion_angle = np.nan
        abduction_angle = np.nan
        low_confidence = False
        
        # Required landmarks depend on camera view (side-agnostic keys)
        if self.camera_view == 'lateral':
            required_landmarks = ['TRACKED_SHOULDER', 'TRACKED_ELBOW', 'TRACKED_HIP']
        else:
            required_landmarks = ['TRACKED_SHOULDER', 'TRACKED_ELBOW', 'TRACKED_HIP', 'CONTRA_HIP']
        
        # Check visibility
        for landmark_name in required_landmarks:
            if landmark_name in landmarks_dict:
                _, _, visibility = landmarks_dict[landmark_name]
                if visibility < self.VISIBILITY_THRESHOLD:
                    low_confidence = True
            else:
                low_confidence = True
        
        if not low_confidence:
            # Extract coordinates for tracked (affected) arm
            shoulder = np.array(landmarks_dict['TRACKED_SHOULDER'][:2])
            elbow = np.array(landmarks_dict['TRACKED_ELBOW'][:2])
            tracked_hip = np.array(landmarks_dict['TRACKED_HIP'][:2])
            
            # Compute hip reference (view-dependent)
            if self.camera_view == 'lateral':
                hip_reference = tracked_hip
            else:
                # Frontal view: use bilateral hip midpoint for stability
                contra_hip = np.array(landmarks_dict['CONTRA_HIP'][:2])
                hip_reference = (tracked_hip + contra_hip) / 2.0
            
            # Compute joint angle using unified trunk-relative method
            joint_angle = self._compute_joint_angle(shoulder, elbow, hip_reference)
            
            # Assign angle based on camera view
            if self.camera_view == 'lateral':
                flexion_angle = joint_angle
                abduction_angle = np.nan
            else:
                abduction_angle = joint_angle
                flexion_angle = np.nan
        
        return flexion_angle, abduction_angle, low_confidence
    
    def _compute_joint_angle(self, shoulder, elbow, hip):
        """
        Compute anatomical joint angle between upper arm and trunk.
        
        Clinical definition:
        - 0° = arm resting at side (parallel to trunk)
        - 90° = arm horizontal (perpendicular to trunk)
        - 180° = arm overhead (opposite to trunk)
        
        Formula: θ = 180° - arccos(v_arm · v_trunk / |v_arm||v_trunk|)
        
        This inverts the raw vector angle so that:
        - When arm aligns with trunk (vectors parallel): raw angle ≈ 0° → output 180° - 0° = 180°... WAIT, this is wrong.
        
        Correction: When arm is DOWN (at rest), it points opposite to trunk vector.
        - Trunk vector: hip -> shoulder (points upward)
        - Arm vector at rest: shoulder -> elbow (points downward)
        - These are ~180° apart, so raw angle ≈ 180°
        - We want output = 0° at rest, so: θ = 180° - raw_angle = 180° - 180° = 0° ✓
        
        When arm is overhead:
        - Arm vector: shoulder -> elbow (points upward, same direction as trunk)
        - Raw angle ≈ 0° (vectors parallel)
        - Output: θ = 180° - 0° = 180° ✓
        
        Args:
            shoulder: (x, y) coordinates
            elbow: (x, y) coordinates  
            hip: (x, y) coordinates (hip midpoint)
            
        Returns:
            Angle in degrees (0-180), where 0° = rest, 180° = overhead
        """
        # Upper arm vector (shoulder -> elbow)
        v_arm = elbow - shoulder
        
        # Trunk vector (hip -> shoulder, pointing upward along body)
        v_trunk = shoulder - hip
        
        # Guard against degenerate geometry (BUG-2 fix):
        # If two landmarks collapse to the same coordinates, the vector norm
        # approaches zero and normalization would produce NaN/Inf.
        # Return np.nan to signal an invalid frame rather than poisoning
        # downstream angle smoothing and rep detection.
        EPSILON = 1e-6
        arm_norm = np.linalg.norm(v_arm)
        trunk_norm = np.linalg.norm(v_trunk)
        if arm_norm < EPSILON or trunk_norm < EPSILON:
            return np.nan
        
        # Normalize vectors (safe — norms verified above)
        v_arm_norm = v_arm / arm_norm
        v_trunk_norm = v_trunk / trunk_norm
        
        # Compute angle between vectors with numerical stability
        dot_product = np.dot(v_arm_norm, v_trunk_norm)
        raw_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        # Invert to match clinical convention (0° at rest, 180° overhead)
        anatomical_angle = 180.0 - raw_angle
        
        return anatomical_angle
    
    def set_camera_view(self, camera_view):
        """
        Change camera view mode (updates required landmarks).
        
        Args:
            camera_view: 'frontal' (for abduction) or 'lateral' (for flexion)
        """
        if camera_view not in ['frontal', 'lateral']:
            raise ValueError("camera_view must be 'frontal' or 'lateral'")
        
        self.camera_view = camera_view
        self._rebuild_landmarks()  # Reapply side+view combination
    
    def compute_trunk_lean(self, landmarks_dict):
        """
        Compute trunk lean angle from vertical.
        
        Phase 3.4: Trunk Lean Detection (side-agnostic via TRACKED_*/CONTRA_* keys)
        Research context: Nakatsuchi et al. 2021, Christensen et al. 2021 (analogous)
        
        Measurement:
        - Frontal view: Uses trunk MIDLINE (mid-shoulder -> mid-hip) for unbiased detection
        - Lateral view: Uses tracked-side landmarks only
        
        Args:
            landmarks_dict: Dictionary of landmarks from process_frame()
            
        Returns:
            tuple: (trunk_lean_angle, is_compensating)
        """
        if self.camera_view == 'lateral':
            required = ['TRACKED_SHOULDER', 'TRACKED_HIP']
        else:
            required = ['TRACKED_SHOULDER', 'CONTRA_SHOULDER', 'TRACKED_HIP', 'CONTRA_HIP']
        
        for lm in required:
            if lm not in landmarks_dict:
                return np.nan, False
            _, _, vis = landmarks_dict[lm]
            if vis < self.VISIBILITY_THRESHOLD:
                return np.nan, False
        
        if self.camera_view == 'lateral':
            shoulder = np.array(landmarks_dict['TRACKED_SHOULDER'][:2])
            hip = np.array(landmarks_dict['TRACKED_HIP'][:2])
        else:
            # Frontal view: use midline (mid-shoulder -> mid-hip)
            tracked_shoulder = np.array(landmarks_dict['TRACKED_SHOULDER'][:2])
            contra_shoulder = np.array(landmarks_dict['CONTRA_SHOULDER'][:2])
            tracked_hip = np.array(landmarks_dict['TRACKED_HIP'][:2])
            contra_hip = np.array(landmarks_dict['CONTRA_HIP'][:2])
            
            shoulder = (tracked_shoulder + contra_shoulder) / 2.0
            hip = (tracked_hip + contra_hip) / 2.0
        
        v_trunk = shoulder - hip
        v_vertical = np.array([0, -1])
        
        trunk_norm = np.linalg.norm(v_trunk)
        if trunk_norm < 1e-6:
            return np.nan, False
        
        v_trunk_norm = v_trunk / trunk_norm
        dot_product = np.dot(v_trunk_norm, v_vertical)
        trunk_lean_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        is_compensating = trunk_lean_angle > self.TRUNK_LEAN_THRESHOLD
        return trunk_lean_angle, is_compensating
    
    def compute_head_tilt(self, landmarks_dict):
        """
        Compute head tilt angle from ear landmarks.
        
        T12: Head-Tilt Suppression for Shoulder Hiking
        Used to reduce false-positive shoulder hiking flags caused by natural head tilt.
        When the head tilts significantly, ear-to-shoulder asymmetry changes due to
        geometry, not actual shoulder hiking compensation. This filter suppresses hiking
        detection when head tilt > 15°.
        
        Args:
            landmarks_dict: Dictionary of landmarks from process_frame()
            
        Returns:
            Head tilt angle in degrees (positive = head tilted right, negative = left).
            Returns np.nan if landmarks are unavailable or invalid.
        """
        # Need raw MediaPipe ear indices to compute head tilt (not side-specific)
        left_ear_idx = self.MP_LANDMARKS['LEFT_EAR']   # 7
        right_ear_idx = self.MP_LANDMARKS['RIGHT_EAR']  # 8
        
        # Attempt to access raw landmarks. If process_frame() did not populate them,
        # this will fail safely and return np.nan.
        if 'left_ear_raw' not in landmarks_dict or 'right_ear_raw' not in landmarks_dict:
            return np.nan
        
        left_ear_x, left_ear_y, left_vis = landmarks_dict['left_ear_raw']
        right_ear_x, right_ear_y, right_vis = landmarks_dict['right_ear_raw']
        
        # Both ears must be visible
        if left_vis < self.VISIBILITY_THRESHOLD or right_vis < self.VISIBILITY_THRESHOLD:
            return np.nan
        
        # Compute ear-to-ear vector (right ear -> left ear)
        dx = left_ear_x - right_ear_x
        dy = left_ear_y - right_ear_y

        if np.hypot(dx, dy) < 1e-6:
            return np.nan

        # arctan2 returns angle of vector from horizontal.
        # For level ears: dy ~= 0, dx > 0 -> angle ~= 0 deg.
        # For tilted head: larger |dy| increases |angle|.
        angle_rad = np.arctan2(dy, dx)
        head_tilt_angle = np.degrees(angle_rad)

        return head_tilt_angle
    
    def compute_shoulder_hiking(self, landmarks_dict):
        """
        Detect shoulder hiking compensation using bilateral ear-shoulder asymmetry.
        
        Phase 3.5: Shoulder Hiking Detection (side-aware via TRACKED_*/CONTRA_* keys)
        Research context: Mohamed et al. 2020 (scapular compensation)
        
        MISSING-1 side-awareness:
        The comparison is between TRACKED (affected) side and CONTRA (unaffected) side.
        When the affected shoulder hikes (shorter ear-to-shoulder distance on that side),
        the asymmetry ratio detects it regardless of which anatomical side is affected.
        
        Frontal view only — lateral view lacks contralateral visibility.
        
        Args:
            landmarks_dict: Dictionary of landmarks from process_frame()
            
        Returns:
            tuple: (asymmetry_ratio, is_compensating)
        """
        if self.camera_view == 'lateral':
            return np.nan, False
        
        required = ['TRACKED_EAR', 'TRACKED_SHOULDER', 'CONTRA_EAR', 'CONTRA_SHOULDER']
        
        for lm in required:
            if lm not in landmarks_dict:
                return np.nan, False
            _, _, vis = landmarks_dict[lm]
            if vis < self.VISIBILITY_THRESHOLD:
                return np.nan, False
        
        # Ear-to-shoulder vertical distances (image Y points DOWN)
        tracked_ear_y = landmarks_dict['TRACKED_EAR'][1]
        tracked_shoulder_y = landmarks_dict['TRACKED_SHOULDER'][1]
        contra_ear_y = landmarks_dict['CONTRA_EAR'][1]
        contra_shoulder_y = landmarks_dict['CONTRA_SHOULDER'][1]
        
        tracked_distance = tracked_shoulder_y - tracked_ear_y
        contra_distance = contra_shoulder_y - contra_ear_y
        
        if tracked_distance <= 0 or contra_distance <= 0:
            return np.nan, False
        
        avg_distance = (tracked_distance + contra_distance) / 2.0
        asymmetry = abs(tracked_distance - contra_distance) / avg_distance
        
        is_compensating = asymmetry > self.SHOULDER_HIKING_THRESHOLD
        
        # T12: Head-tilt suppression — reduce false positives from natural head motion
        # If the head is tilted > 15°, the ear-to-shoulder asymmetry may be geometric
        # rather than true shoulder hiking compensation. Suppress the flag.
        if is_compensating:
            head_tilt = self.compute_head_tilt(landmarks_dict)
            if not np.isnan(head_tilt) and abs(head_tilt) > 15.0:
                is_compensating = False  # Suppress hiking flag due to head tilt
        
        return asymmetry, is_compensating
    
    def draw_landmarks(self, frame, results):
        """
        Draw skeleton overlay on frame
        
        Args:
            frame: BGR image from OpenCV
            results: MediaPipe pose results
            
        Returns:
            Annotated frame
        """
        if results.pose_landmarks:
            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def draw_angles(self, frame, flexion_angle, abduction_angle, low_confidence,
                     smoothed_angle=None, show_smoothed=False):
        """
        Draw angle values on frame
        
        MISSING-6: Optionally displays the smoothed angle value used by the
        rep tracker for threshold decisions. Helps with debugging and calibration.
        
        Args:
            frame: BGR image from OpenCV
            flexion_angle: Flexion angle in degrees (or np.nan)
            abduction_angle: Abduction angle in degrees (or np.nan)
            low_confidence: Boolean flag for visibility warning
            smoothed_angle: Smoothed angle value from rep_tracker (or None/np.nan)
            show_smoothed: Whether to render the smoothed angle overlay
            
        Returns:
            Annotated frame
        """
        # Move angles to TOP RIGHT to avoid overlap with pose detection warnings
        height, width = frame.shape[:2]
        x_pos = width - 250  # Right side
        y_offset = 30
        
        # Flexion angle
        if np.isnan(flexion_angle):
            flex_text = "Flexion: --"
        else:
            flex_text = f"Flexion: {flexion_angle:.1f}deg"
        cv2.putText(frame, flex_text, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Abduction angle
        if np.isnan(abduction_angle):
            abd_text = "Abduction: --"
        else:
            abd_text = f"Abduction: {abduction_angle:.1f}deg"
        cv2.putText(frame, abd_text, (x_pos, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # MISSING-6: Smoothed angle overlay (when enabled)
        if show_smoothed and smoothed_angle is not None and not np.isnan(smoothed_angle):
            smooth_text = f"Smoothed: {smoothed_angle:.1f}deg"
            cv2.putText(frame, smooth_text, (x_pos, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)  # Cyan/yellow
        
        return frame
    
    def release(self):
        """Release MediaPipe resources (idempotent — safe to call multiple times, BUG-6 fix)"""
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass  # Best-effort cleanup
            self.pose = None
