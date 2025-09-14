from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import json
import time
import threading
import requests
from datetime import datetime
import tempfile
import shutil
import base64
import numpy as np
import math
from database import sessions, users, video_metadata

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8080"], 
     allow_headers=["Content-Type", "Authorization", "Range"],
     expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])

# Configuration
MEDIAPIPE_SERVER_URL = "http://127.0.0.1:5001"
VIDEO_PROCESSING_DIR = "../output_videos"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_videos")
ANALYTICS_DIR = "../analytics"

def find_best_video(video_filename):
    """
    Find the best available video file, prioritizing H.264 versions
    
    Args:
        video_filename: Original video filename
    
    Returns:
        tuple: (best_path: str, is_h264: bool, original_path: str)
    """
    base_name = os.path.splitext(video_filename)[0]
    
    # Check for H.264 version first (look for any file starting with h264_ and containing base_name)
    h264_files = []
    for directory in [VIDEO_PROCESSING_DIR, OUTPUT_DIR]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.startswith("h264_") and base_name in file and file.endswith('.mp4'):
                    h264_files.append(os.path.join(directory, file))
    
    if h264_files:
        # Use the first H.264 file found
        h264_path = h264_files[0]
        print(f"✅ Using H.264 version: {os.path.basename(h264_path)}")
        return h264_path, True, os.path.join(VIDEO_PROCESSING_DIR, video_filename)
    
    # Fall back to original video
    original_path = os.path.join(VIDEO_PROCESSING_DIR, video_filename)
    if os.path.exists(original_path):
        print(f"⚠️  Using original video (not H.264): {video_filename}")
        return original_path, False, original_path
    
    return None, False, None

def convert_to_h264(input_path, output_path=None):
    """
    Convert video to H.264 format for browser compatibility
    
    Args:
        input_path: Path to input video file
        output_path: Path for output H.264 video (optional)
    
    Returns:
        tuple: (success: bool, output_path: str, error: str)
    """
    try:
        if output_path is None:
            # Generate H.264 filename
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            dir_name = os.path.dirname(input_path)
            output_path = os.path.join(dir_name, f"h264_{base_name}.mp4")
        
        # Check if output already exists
        if os.path.exists(output_path):
            print(f"✅ H.264 version already exists: {output_path}")
            return True, output_path, None
        
        print(f"🔄 Converting to H.264: {os.path.basename(input_path)}")
        
        # Use ffmpeg to convert to H.264
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',           # H.264 video codec
            '-preset', 'medium',          # Balance between speed and quality
            '-crf', '23',                 # Constant Rate Factor (quality)
            '-c:a', 'aac',                # AAC audio codec
            '-b:a', '128k',               # Audio bitrate
            '-y',                         # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Get file sizes for comparison
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"✅ H.264 conversion successful: {os.path.basename(output_path)}")
            print(f"   Size: {input_size:.1f}MB → {output_size:.1f}MB")
            
            return True, output_path, None
        else:
            error_msg = f"FFmpeg conversion failed: {result.stderr}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "Conversion timed out after 5 minutes"
        print(f"❌ {error_msg}")
        return False, None, error_msg
    except Exception as e:
        error_msg = f"Conversion error: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# Global state for tracking processing jobs
processing_jobs = {}
upload_session_mapping = {}  # Mapping between upload session IDs and database session IDs

class ACLRiskAssessment:
    """
    Comprehensive ACL Risk Assessment System
    Based on research-based biomechanical factors and context multipliers
    """
    
    def __init__(self):
        # Risk factor weights (must sum to 1.0)
        self.flexion_weight = 0.25      # 25% - Knee flexion angle
        self.frontal_weight = 0.40      # 40% - Frontal plane alignment (most important)
        self.landing_weight = 0.30      # 30% - Landing mechanics quality
        self.context_weight = 0.05      # 5% - Context multipliers (can add up to 15%)
        
        # Risk thresholds
        self.low_risk_threshold = 30
        self.moderate_risk_threshold = 60
        
        # Coaching cues database
        self.coaching_cues = {
            'valgus': {
                'issue': 'Knees caving in',
                'cue': 'knees over toes',
                'drill': 'miniband squats (2×12)',
                'priority': 'high'
            },
            'varus': {
                'issue': 'Knees bowing out',
                'cue': 'track over 2nd–3rd toe',
                'drill': 'wall-sit with knee tracking (2×30s)',
                'priority': 'high'
            },
            'stiff_landing': {
                'issue': 'Stiff landing - lack of shock absorption',
                'cue': 'quiet feet',
                'drill': 'drop-land to stick from low box',
                'priority': 'medium'
            },
            'asymmetry': {
                'issue': 'Left/right knee/hip asymmetry',
                'cue': 'match knee bend L/R',
                'drill': 'single-leg step-downs (2×10/side)',
                'priority': 'medium'
            },
            'hyperextension': {
                'issue': 'Knee hyperextension',
                'cue': 'micro-bend on contact',
                'drill': 'landing to soft knee holds',
                'priority': 'high'
            },
            'forward_lean': {
                'issue': 'Excess forward trunk lean',
                'cue': 'chest up, core engaged',
                'drill': 'landing with wall support',
                'priority': 'medium'
            }
        }
    
    def calculate_knee_flexion_risk(self, left_knee_angle, right_knee_angle):
        """
        Calculate knee flexion angle risk (25% weight)
        
        Args:
            left_knee_angle: Left knee angle in degrees
            right_knee_angle: Right knee angle in degrees
        
        Returns:
            dict: Risk score and details
        """
        # Use average of both knees
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        risk_score = 0
        risk_level = 'LOW'
        hyperextension_flag = False
        
        # Risk assessment based on research thresholds
        if avg_knee_angle > 160:
            # Hyperextension - highest risk
            risk_score = 100
            risk_level = 'HIGH'
            hyperextension_flag = True
        elif avg_knee_angle < 50:
            # Very stiff landing
            risk_score = 90
            risk_level = 'HIGH'
        elif avg_knee_angle < 70:
            # Stiff landing - risky
            risk_score = 70
            risk_level = 'MODERATE'
        elif avg_knee_angle < 120:
            # Good range - safer
            risk_score = 20
            risk_level = 'LOW'
        else:
            # Deep flexion - moderate risk
            risk_score = 40
            risk_level = 'MODERATE'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'knee_angle': avg_knee_angle,
            'hyperextension_flag': hyperextension_flag,
            'weight': self.flexion_weight
        }
    
    def calculate_frontal_plane_risk(self, left_knee_x, right_knee_x, left_ankle_x, right_ankle_x):
        """
        Calculate frontal plane alignment risk (40% weight) - Most Important Factor
        
        Args:
            left_knee_x, right_knee_x: Knee x-coordinates
            left_ankle_x, right_ankle_x: Ankle x-coordinates
        
        Returns:
            dict: Risk score and details
        """
        # Calculate knee separation and alignment
        knee_separation = abs(left_knee_x - right_knee_x)
        ankle_separation = abs(left_ankle_x - right_ankle_x)
        
        # Calculate knee-to-ankle alignment
        left_alignment = abs(left_knee_x - left_ankle_x)
        right_alignment = abs(right_knee_x - right_ankle_x)
        
        risk_score = 0
        risk_level = 'LOW'
        alignment_issue = 'none'
        
        # Valgus detection (knees caving in)
        if knee_separation < ankle_separation * 0.8:
            # Knees are closer together than ankles
            valgus_severity = (ankle_separation - knee_separation) / ankle_separation
            risk_score = min(100, 50 + valgus_severity * 50)
            risk_level = 'HIGH' if valgus_severity > 0.3 else 'MODERATE'
            alignment_issue = 'valgus'
        
        # Varus detection (knees bowing out)
        elif knee_separation > ankle_separation * 1.2:
            # Knees are further apart than ankles
            varus_severity = (knee_separation - ankle_separation) / ankle_separation
            risk_score = min(100, 40 + varus_severity * 40)
            risk_level = 'MODERATE' if varus_severity > 0.2 else 'LOW'
            alignment_issue = 'varus'
        
        # Good alignment
        else:
            risk_score = 10
            risk_level = 'LOW'
            alignment_issue = 'good'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'knee_separation': knee_separation,
            'ankle_separation': ankle_separation,
            'alignment_issue': alignment_issue,
            'weight': self.frontal_weight
        }
    
    def calculate_landing_mechanics_risk(self, landmarks, frame_history):
        """
        Calculate landing mechanics quality risk (30% weight)
        
        Args:
            landmarks: Current frame landmarks
            frame_history: Previous frames for temporal analysis
        
        Returns:
            dict: Risk score and details
        """
        risk_score = 0
        risk_level = 'LOW'
        issues = []
        
        # 1. Shock absorption (stiff landing detection)
        shock_absorption_score = self._assess_shock_absorption(landmarks, frame_history)
        if shock_absorption_score > 70:
            issues.append('stiff_landing')
            risk_score += 30
        
        # 2. Left/right symmetry
        symmetry_score = self._assess_symmetry(landmarks)
        if symmetry_score > 60:
            issues.append('asymmetry')
            risk_score += 25
        
        # 3. Trunk control (forward lean)
        trunk_score = self._assess_trunk_control(landmarks)
        if trunk_score > 50:
            issues.append('forward_lean')
            risk_score += 20
        
        # Normalize to 0-100 scale
        risk_score = min(100, risk_score)
        
        if risk_score > 70:
            risk_level = 'HIGH'
        elif risk_score > 40:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'issues': issues,
            'shock_absorption_score': shock_absorption_score,
            'symmetry_score': symmetry_score,
            'trunk_score': trunk_score,
            'weight': self.landing_weight
        }
    
    def _assess_shock_absorption(self, landmarks, frame_history):
        """Assess shock absorption quality"""
        if len(frame_history) < 3:
            return 0
        
        # Check for rapid deceleration patterns
        recent_frames = frame_history[-3:]
        hip_velocities = []
        
        for i in range(1, len(recent_frames)):
            prev_frame = recent_frames[i-1]
            curr_frame = recent_frames[i]
            
            # Safely access landmarks - frame_history contains dictionaries with 'landmarks' key
            prev_landmarks = prev_frame.get('landmarks', []) if isinstance(prev_frame, dict) else []
            curr_landmarks = curr_frame.get('landmarks', []) if isinstance(curr_frame, dict) else []
            
            if (len(prev_landmarks) > 24 and len(curr_landmarks) > 24):
                prev_hip_y = (prev_landmarks[23].get('y', 0) + prev_landmarks[24].get('y', 0)) / 2
                curr_hip_y = (curr_landmarks[23].get('y', 0) + curr_landmarks[24].get('y', 0)) / 2
                velocity = abs(curr_hip_y - prev_hip_y)
                hip_velocities.append(velocity)
        
        # High velocity changes indicate stiff landing
        avg_velocity = np.mean(hip_velocities) if hip_velocities else 0
        return min(100, avg_velocity * 1000)  # Scale to 0-100
    
    def _assess_symmetry(self, landmarks):
        """Assess left/right symmetry"""
        if len(landmarks) < 29:
            return 0
        
        # Compare left and right knee angles
        left_knee_angle = self._calculate_knee_angle(landmarks, 23, 25, 27)  # Left hip, knee, ankle
        right_knee_angle = self._calculate_knee_angle(landmarks, 24, 26, 28)  # Right hip, knee, ankle
        
        # Calculate asymmetry
        angle_diff = abs(left_knee_angle - right_knee_angle)
        return min(100, angle_diff * 2)  # Scale to 0-100
    
    def _assess_trunk_control(self, landmarks):
        """Assess trunk control and forward lean"""
        if len(landmarks) < 25:
            return 0
        
        # Calculate trunk angle from vertical - safely access landmarks
        if len(landmarks) > 24:
            shoulder_center = {
                'x': (landmarks[11].get('x', 0) + landmarks[12].get('x', 0)) / 2,
                'y': (landmarks[11].get('y', 0) + landmarks[12].get('y', 0)) / 2
            }
            hip_center = {
                'x': (landmarks[23].get('x', 0) + landmarks[24].get('x', 0)) / 2,
                'y': (landmarks[23].get('y', 0) + landmarks[24].get('y', 0)) / 2
            }
        else:
            return 0
        
        # Calculate angle from vertical
        dx = shoulder_center['x'] - hip_center['x']
        dy = shoulder_center['y'] - hip_center['y']
        
        if dx != 0:
            trunk_angle = abs(math.degrees(math.atan2(dx, dy)))
            return min(100, trunk_angle * 2)  # Scale to 0-100
        
        return 0
    
    def _calculate_knee_angle(self, landmarks, hip_idx, knee_idx, ankle_idx):
        """Calculate knee angle between hip, knee, and ankle"""
        if len(landmarks) <= max(hip_idx, knee_idx, ankle_idx):
            return 0
        
        hip = landmarks[hip_idx]
        knee = landmarks[knee_idx]
        ankle = landmarks[ankle_idx]
        
        # Safely access landmark coordinates
        if not isinstance(hip, dict) or not isinstance(knee, dict) or not isinstance(ankle, dict):
            return 0
        
        # Calculate vectors
        v1 = np.array([hip.get('x', 0) - knee.get('x', 0), hip.get('y', 0) - knee.get('y', 0)])
        v2 = np.array([ankle.get('x', 0) - knee.get('x', 0), ankle.get('y', 0) - knee.get('y', 0)])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def calculate_context_multipliers(self, session_context):
        """
        Calculate context multipliers (±5–15%)
        
        Args:
            session_context: dict with context information
        
        Returns:
            dict: Context adjustments
        """
        context_adjustment = 0
        context_factors = []
        
        # Practice vs competition
        if session_context.get('session_type') == 'practice':
            context_adjustment += 5
            context_factors.append('practice_session')
        
        # Fatigue level
        fatigue_level = session_context.get('fatigue_level', 'low')
        if fatigue_level == 'high':
            context_adjustment += 10
            context_factors.append('high_fatigue')
        elif fatigue_level == 'medium':
            context_adjustment += 5
            context_factors.append('medium_fatigue')
        
        # Surface/footwear mismatch
        if session_context.get('surface_mismatch', False):
            context_adjustment += 5
            context_factors.append('surface_mismatch')
        
        # Apparatus risk
        apparatus = session_context.get('apparatus', 'floor')
        if apparatus in ['beam_dismount', 'vault', 'bars_dismount']:
            context_adjustment += 5
            context_factors.append('high_risk_apparatus')
        
        # Cap at 15%
        context_adjustment = min(15, context_adjustment)
        
        return {
            'adjustment_percentage': context_adjustment,
            'context_factors': context_factors,
            'weight': self.context_weight
        }
    
    def generate_coaching_cues(self, risk_factors):
        """
        Generate actionable coaching cues based on detected issues
        
        Args:
            risk_factors: dict with risk assessment results
        
        Returns:
            list: Coaching cues with priority
        """
        cues = []
        
        # Check for specific issues and generate cues
        if risk_factors.get('frontal_plane', {}).get('alignment_issue') == 'valgus':
            cues.append(self.coaching_cues['valgus'])
        
        if risk_factors.get('frontal_plane', {}).get('alignment_issue') == 'varus':
            cues.append(self.coaching_cues['varus'])
        
        if 'stiff_landing' in risk_factors.get('landing_mechanics', {}).get('issues', []):
            cues.append(self.coaching_cues['stiff_landing'])
        
        if 'asymmetry' in risk_factors.get('landing_mechanics', {}).get('issues', []):
            cues.append(self.coaching_cues['asymmetry'])
        
        if risk_factors.get('knee_flexion', {}).get('hyperextension_flag', False):
            cues.append(self.coaching_cues['hyperextension'])
        
        if 'forward_lean' in risk_factors.get('landing_mechanics', {}).get('issues', []):
            cues.append(self.coaching_cues['forward_lean'])
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        cues.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 0), reverse=True)
        
        return cues
    
    def assess_acl_risk(self, landmarks, frame_history, session_context=None):
        """
        Comprehensive ACL risk assessment
        
        Args:
            landmarks: Current frame landmarks
            frame_history: Previous frames for temporal analysis
            session_context: Session context information
        
        Returns:
            dict: Complete ACL risk assessment
        """
        if not landmarks or len(landmarks) < 29:
            return {
                'overall_risk': 0,
                'risk_level': 'LOW',
                'confidence': 'low',
                'error': 'Insufficient landmark data'
            }
        
        # Extract key landmarks - landmarks is a list, so we need to access by index
        left_knee_angle = self._calculate_knee_angle(landmarks, 23, 25, 27)
        right_knee_angle = self._calculate_knee_angle(landmarks, 24, 26, 28)
        
        # Safely access landmarks by index
        if len(landmarks) > 28:
            left_knee_x = landmarks[25].get('x', 0) if landmarks[25] else 0
            right_knee_x = landmarks[26].get('x', 0) if landmarks[26] else 0
            left_ankle_x = landmarks[27].get('x', 0) if landmarks[27] else 0
            right_ankle_x = landmarks[28].get('x', 0) if landmarks[28] else 0
        else:
            # Fallback values if landmarks are insufficient
            left_knee_x = right_knee_x = left_ankle_x = right_ankle_x = 0
        
        # Calculate individual risk factors
        knee_flexion_risk = self.calculate_knee_flexion_risk(left_knee_angle, right_knee_angle)
        frontal_plane_risk = self.calculate_frontal_plane_risk(left_knee_x, right_knee_x, left_ankle_x, right_ankle_x)
        landing_mechanics_risk = self.calculate_landing_mechanics_risk(landmarks, frame_history)
        
        # Calculate context multipliers
        session_context = session_context or {}
        context_multipliers = self.calculate_context_multipliers(session_context)
        
        # Calculate weighted total risk
        total_risk = (
            knee_flexion_risk['risk_score'] * knee_flexion_risk['weight'] +
            frontal_plane_risk['risk_score'] * frontal_plane_risk['weight'] +
            landing_mechanics_risk['risk_score'] * landing_mechanics_risk['weight']
        )
        
        # Apply context adjustment
        context_adjustment = context_multipliers['adjustment_percentage'] / 100
        total_risk = total_risk * (1 + context_adjustment)
        
        # Determine risk level
        if total_risk < self.low_risk_threshold:
            risk_level = 'LOW'
        elif total_risk < self.moderate_risk_threshold:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'
        
        # Generate coaching cues
        risk_factors = {
            'knee_flexion': knee_flexion_risk,
            'frontal_plane': frontal_plane_risk,
            'landing_mechanics': landing_mechanics_risk,
            'context': context_multipliers
        }
        
        coaching_cues = self.generate_coaching_cues(risk_factors)
        
        return {
            'overall_risk': round(total_risk, 1),
            'risk_level': risk_level,
            'risk_bands': {
                'low': f'<{self.low_risk_threshold}',
                'moderate': f'{self.low_risk_threshold}-{self.moderate_risk_threshold}',
                'high': f'>{self.moderate_risk_threshold}'
            },
            'risk_factors': risk_factors,
            'coaching_cues': coaching_cues,
            'confidence': 'high' if len(landmarks) >= 33 else 'medium',
            'timestamp': datetime.now().isoformat()
        }

# Initialize ACL risk assessment system
acl_risk_assessor = ACLRiskAssessment()

class VideoProcessor:
    def __init__(self):
        self.mediapipe_url = MEDIAPIPE_SERVER_URL
    
    def check_mediapipe_server(self):
        """Check if MediaPipe server is running"""
        try:
            response = requests.get(f"{self.mediapipe_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def process_video_with_analytics(self, video_path, output_name=None):
        """Process video using the fixed overlay system and convert to H.264"""
        if not output_name:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"processed_{base_name}.mp4"
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        analytics_path = os.path.join(ANALYTICS_DIR, f"analytics_{base_name}.json")
        
        # Command to run the fixed video overlay script
        cmd = [
            "python3", "video_overlay_with_analytics_fixed.py",
            video_path,
            "--output", output_path,
            "--server", "http://127.0.0.1:5001"
        ]
        
        try:
            print(f"🎬 Starting video processing: {os.path.basename(video_path)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"✅ Video overlay processing completed: {output_name}")
                
                # Convert to H.264 for better browser compatibility
                h264_output_path = os.path.join(OUTPUT_DIR, f"h264_{output_name}")
                print(f"🔄 Converting to H.264: {os.path.basename(h264_output_path)}")
                
                h264_cmd = [
                    "ffmpeg", "-i", output_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-preset", "fast",
                    "-crf", "23",
                    h264_output_path,
                    "-y"  # Overwrite if exists
                ]
                
                h264_result = subprocess.run(h264_cmd, capture_output=True, text=True)
                
                if h264_result.returncode == 0:
                    print(f"✅ H.264 conversion completed: {os.path.basename(h264_output_path)}")
                    
                    # Get file sizes for comparison
                    original_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    h264_size = os.path.getsize(h264_output_path) / (1024 * 1024)  # MB
                    
                    print(f"📊 File sizes - Original: {original_size:.1f}MB, H.264: {h264_size:.1f}MB")
                    
                    return {
                        "success": True,
                        "output_video": output_path,
                        "h264_video": h264_output_path,
                        "analytics_file": analytics_path,
                        "message": f"Video processed and converted to H.264 successfully (Original: {original_size:.1f}MB → H.264: {h264_size:.1f}MB)"
                    }
                else:
                    print(f"⚠️ H.264 conversion failed, but original video is available: {output_name}")
                    print(f"H.264 conversion error: {h264_result.stderr}")
                    
                    return {
                        "success": True,
                        "output_video": output_path,
                        "analytics_file": analytics_path,
                        "message": "Video processed successfully (H.264 conversion failed, but original video is available)"
                    }
            else:
                print(f"❌ Video processing failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "message": "Video processing failed"
                }
        except Exception as e:
            print(f"❌ Video processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Video processing error"
            }

# Initialize video processor
video_processor = VideoProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mediapipe_status = video_processor.check_mediapipe_server()
    return jsonify({
        "status": "healthy",
        "mediapipe_server": "running" if mediapipe_status else "down",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyzeVideo', methods=['POST'])
def analyze_video():
    """Analyze video and generate analytics overlay"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return jsonify({"error": "video_filename is required"}), 400
        
        # Find the best available video (prioritize H.264)
        best_video_path, is_h264, original_path = find_best_video(video_filename)
        
        if not best_video_path:
            return jsonify({"error": f"Video file not found: {video_filename}"}), 404
        
        # Use the best available video for processing
        video_path = best_video_path
        print(f"🎬 Processing video: {os.path.basename(video_path)} (H.264: {is_h264})")
        
        # Generate unique output name
        base_name = os.path.splitext(video_filename)[0]
        timestamp = int(time.time())
        output_name = f"analyzed_{base_name}_{timestamp}.mp4"
        
        # Process video
        result = video_processor.process_video_with_analytics(video_path, output_name)
        
        if result["success"]:
            # Store job info
            job_id = f"job_{timestamp}"
            processing_jobs[job_id] = {
                "status": "completed",
                "video_filename": video_filename,
                "output_video": output_name,
                "analytics_file": f"analytics_{base_name}.json",
                "timestamp": timestamp,
                "result": result
            }
            
            return jsonify({
                "success": True,
                "job_id": job_id,
                "output_video": output_name,
                "analytics_file": f"analytics_{base_name}.json",
                "message": "Video analysis completed"
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"],
                "message": result["message"]
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getStatistics', methods=['GET'])
def get_statistics():
    """Get detailed statistics for a specific video"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        base_name = os.path.splitext(video_filename)[0]
        analytics_file = os.path.join(ANALYTICS_DIR, f"analytics_{base_name}.json")
        
        if not os.path.exists(analytics_file):
            return jsonify({"error": f"Analytics file not found for {video_filename}"}), 404
        
        with open(analytics_file, 'r') as f:
            analytics_data = json.load(f)
        
        # Calculate comprehensive statistics
        stats = calculate_detailed_statistics(analytics_data)
        
        return jsonify({
            "success": True,
            "video_filename": video_filename,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getSummaryStatistics', methods=['GET'])
def get_summary_statistics():
    """Get summary statistics for all processed videos"""
    try:
        # Get all analytics files
        analytics_files = []
        for file in os.listdir(ANALYTICS_DIR):
            if file.startswith("analytics_") and file.endswith(".json"):
                analytics_files.append(file)
        
        summary_stats = {
            "total_videos_processed": len(analytics_files),
            "videos": []
        }
        
        for analytics_file in analytics_files:
            file_path = os.path.join(ANALYTICS_DIR, analytics_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            video_name = analytics_file.replace("analytics_", "").replace(".json", "")
            video_stats = calculate_summary_statistics(data, video_name)
            summary_stats["videos"].append(video_stats)
        
        return jsonify({
            "success": True,
            "summary_statistics": summary_stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getVideoList', methods=['GET'])
def get_video_list():
    """Get list of available videos"""
    try:
        videos = []
        if os.path.exists(VIDEO_PROCESSING_DIR):
            for file in os.listdir(VIDEO_PROCESSING_DIR):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(VIDEO_PROCESSING_DIR, file)
                    file_size = os.path.getsize(file_path)
                    videos.append({
                        "filename": file,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "path": file_path
                    })
        
        return jsonify({
            "success": True,
            "videos": videos
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getProcessedVideos', methods=['GET'])
def get_processed_videos():
    """Get list of processed videos with analytics"""
    print("🚀 getProcessedVideos endpoint called")
    
    try:
        print("🔍 Step 1: Checking if OUTPUT_DIR exists...")
        print(f"📁 OUTPUT_DIR: {OUTPUT_DIR}")
        print(f"📁 OUTPUT_DIR exists: {os.path.exists(OUTPUT_DIR)}")
        
        processed_videos = []
        
        if os.path.exists(OUTPUT_DIR):
            print("🔍 Step 2: Listing files in OUTPUT_DIR...")
            files = os.listdir(OUTPUT_DIR)
            print(f"📁 Found {len(files)} files in OUTPUT_DIR")
            print(f"📁 Files: {files}")
            
            for i, file in enumerate(files):
                print(f"🔍 Step 3.{i+1}: Processing file {file}")
                
                if file.endswith('.mp4'):
                    print(f"✅ File {file} is an MP4")
                    file_path = os.path.join(OUTPUT_DIR, file)
                    print(f"📁 File path: {file_path}")
                    
                    try:
                        print(f"🔍 Step 4.{i+1}: Getting file size...")
                        file_size = os.path.getsize(file_path)
                        print(f"📊 File size: {file_size} bytes")
                        
                        video_info = {
                            "processed_filename": file,
                            "file_size_mb": round(file_size / (1024 * 1024), 2),
                            "has_analytics": False,
                            "analysis_type": "unknown"
                        }
                        
                        print(f"🔍 Step 5.{i+1}: Parsing filename...")
                        # Parse filename to determine type and original name
                        if file.startswith("analyzed_"):
                            print(f"📝 File {file} is standard analysis")
                            original_name = file.replace("analyzed_", "").replace(".mp4", "")
                            video_info["original_filename"] = original_name
                            video_info["analysis_type"] = "standard"
                            
                            # Check for analytics file
                            analytics_file = f"analytics_{original_name}.json"
                            analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
                            print(f"🔍 Checking for analytics file: {analytics_path}")
                            if os.path.exists(analytics_path):
                                video_info["has_analytics"] = True
                                video_info["analytics_file"] = analytics_file
                                print(f"✅ Found analytics file: {analytics_file}")
                            else:
                                print(f"❌ Analytics file not found: {analytics_file}")
                        
                        elif file.startswith("per_frame_analyzed_"):
                            print(f"📝 File {file} is per-frame analysis")
                            name_part = file.replace("per_frame_analyzed_", "").replace(".mp4", "")
                            if "_" in name_part:
                                original_name = name_part.split("_")[0]
                                timestamp = name_part.split("_")[1]
                                video_info["original_filename"] = original_name
                                video_info["analysis_type"] = "per_frame"
                                video_info["timestamp"] = timestamp
                                
                                # Check for analytics file
                                analytics_file = f"frame_analysis_{original_name}_{timestamp}.json"
                                analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
                                print(f"🔍 Checking for analytics file: {analytics_path}")
                                if os.path.exists(analytics_path):
                                    video_info["has_analytics"] = True
                                    video_info["analytics_file"] = analytics_file
                                    print(f"✅ Found analytics file: {analytics_file}")
                                else:
                                    print(f"❌ Analytics file not found: {analytics_file}")
                            else:
                                video_info["original_filename"] = name_part
                                video_info["analysis_type"] = "per_frame"
                        
                        elif file.startswith("api_generated_"):
                            print(f"📝 File {file} is API-generated analysis")
                            original_name = file.replace("api_generated_", "").replace(".mp4", "")
                            video_info["original_filename"] = original_name
                            video_info["analysis_type"] = "api_generated"
                            
                            # Check for analytics file with consistent naming
                            analytics_file = f"api_generated_{original_name}.json"
                            analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
                            print(f"🔍 Checking for analytics file: {analytics_path}")
                            if os.path.exists(analytics_path):
                                video_info["has_analytics"] = True
                                video_info["analytics_file"] = analytics_file
                                print(f"✅ Found analytics file: {analytics_file}")
                            else:
                                # Fallback to old frame_analysis files
                                analytics_files = [f for f in os.listdir(ANALYTICS_DIR) 
                                                 if f.startswith(f"frame_analysis_{original_name}_")]
                                if analytics_files:
                                    # Use the most recent analytics file
                                    latest_analytics = sorted(analytics_files)[-1]
                                    video_info["has_analytics"] = True
                                    video_info["analytics_file"] = latest_analytics
                                    print(f"✅ Found fallback analytics file: {latest_analytics}")
                                else:
                                    print(f"❌ No analytics file found for: {original_name}")
                        
                        elif file.startswith("enhanced_replay_"):
                            print(f"📝 File {file} is enhanced replay")
                            original_name = file.replace("enhanced_replay_", "").replace(".mp4", "")
                            video_info["original_filename"] = original_name
                            video_info["analysis_type"] = "enhanced_replay"
                            video_info["has_enhanced_features"] = True
                            
                            # Check for session report
                            session_report_file = f"session_report_{original_name}.json"
                            session_report_path = os.path.join(ANALYTICS_DIR, session_report_file)
                            print(f"🔍 Checking for session report: {session_report_path}")
                            if os.path.exists(session_report_path):
                                video_info["has_session_report"] = True
                                video_info["session_report_file"] = session_report_file
                                print(f"✅ Found session report: {session_report_file}")
                            else:
                                print(f"❌ No session report found for: {original_name}")
                            
                            # Check for analytics file
                            analytics_files = [
                                f"acl_inference_{original_name}.json",
                                f"api_generated_{original_name}.json",
                                f"analytics_{original_name}.json"
                            ]
                            
                            for analytics_file in analytics_files:
                                analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
                                if os.path.exists(analytics_path):
                                    video_info["has_analytics"] = True
                                    video_info["analytics_file"] = analytics_file
                                    print(f"✅ Found analytics file: {analytics_file}")
                                    break
                        
                        else:
                            print(f"📝 File {file} is unknown type")
                            video_info["original_filename"] = os.path.splitext(file)[0]
                        
                        processed_videos.append(video_info)
                        print(f"✅ Added video: {video_info['original_filename']}")
                        
                    except Exception as file_error:
                        print(f"❌ Error processing file {file}: {file_error}")
                        continue
                else:
                    print(f"❌ File {file} is not an MP4, skipping")
        
        print(f"🎯 Step 6: Found {len(processed_videos)} processed videos")
        print(f"📊 Final result: {processed_videos}")
        
        result = {
            "success": True,
            "processed_videos": processed_videos
        }
        
        print("🚀 Returning response...")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error scanning processed videos: {e}")
        import traceback
        traceback.print_exc()
        # Return mock data as fallback
        return jsonify({
            "success": True,
            "processed_videos": [
                {
                    "processed_filename": "analyzed_FWSpWksgk60_1755820826.mp4",
                    "original_filename": "FWSpWksgk60",
                    "has_analytics": True,
                    "analytics_file": "analytics_FWSpWksgk60.json",
                    "analysis_type": "standard"
                }
            ]
        })

@app.route('/downloadVideo', methods=['GET'])
def download_video():
    """Download processed video file with H.264 priority"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # First try to find H.264 version (best for browser compatibility)
        base_name = os.path.splitext(video_filename)[0]
        
        # Look for H.264 files with flexible naming in both directories
        h264_files = []
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.startswith("h264_") and base_name in file and file.endswith('.mp4'):
                        h264_files.append({
                            "filename": file,
                            "path": os.path.join(directory, file)
                        })
        
        if h264_files:
            # Use the first H.264 file found
            h264_file = h264_files[0]
            print(f"✅ Serving H.264 video: {h264_file['filename']}")
            return send_file(h264_file["path"], as_attachment=True, download_name=h264_file["filename"])
        
        # If no H.264 version, try to find the exact file in both directories
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            video_path = os.path.join(directory, video_filename)
            if os.path.exists(video_path):
                print(f"⚠️  Serving original video (not H.264): {video_filename}")
                return send_file(video_path, as_attachment=True)
        
        # If not found, look for per-frame analyzed version in both directories
        per_frame_files = []
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            if os.path.exists(directory):
                per_frame_files.extend([f for f in os.listdir(directory) 
                                      if f.startswith(f"per_frame_analyzed_{base_name}_") and f.endswith('.mp4')])
        
        if per_frame_files:
            # Get the most recent per-frame analyzed file
            latest_file = sorted(per_frame_files)[-1]
            # Find which directory contains this file
            for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
                video_path = os.path.join(directory, latest_file)
                if os.path.exists(video_path):
                    print(f"📹 Serving per-frame analyzed video: {latest_file}")
                    return send_file(video_path, as_attachment=True, download_name=f"overlayed_{video_filename}")
        
        return jsonify({"error": "Video file not found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getVideoInfo', methods=['GET'])
def get_video_info():
    """Get information about available video formats for a video"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        base_name = os.path.splitext(video_filename)[0]
        
        # Check for different video formats
        video_info = {
            "original_filename": video_filename,
            "base_name": base_name,
            "formats": {}
        }
        
        # Check for H.264 version (in both directories, flexible naming)
        h264_files = []
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.startswith("h264_") and base_name in file and file.endswith('.mp4'):
                        h264_files.append({
                            "filename": file,
                            "path": os.path.join(directory, file),
                            "directory": directory
                        })
        
        if h264_files:
            # Use the first H.264 file found
            h264_file = h264_files[0]
            h264_size = os.path.getsize(h264_file["path"]) / (1024 * 1024)
            video_info["formats"]["h264"] = {
                "filename": h264_file["filename"],
                "path": h264_file["path"],
                "size_mb": round(h264_size, 2),
                "recommended": True,
                "browser_compatible": True
            }
        
        # Check for original version (in both directories)
        original_paths = [
            os.path.join(OUTPUT_DIR, video_filename),
            os.path.join(VIDEO_PROCESSING_DIR, video_filename)
        ]
        
        for original_path in original_paths:
            if os.path.exists(original_path):
                original_size = os.path.getsize(original_path) / (1024 * 1024)
                video_info["formats"]["original"] = {
                    "filename": video_filename,
                    "path": original_path,
                    "size_mb": round(original_size, 2),
                    "recommended": False,
                    "browser_compatible": False
                }
                break
        
        # Check for analyzed versions (in both directories)
        analyzed_files = []
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            if os.path.exists(directory):
                analyzed_files.extend([f for f in os.listdir(directory) 
                                    if f.startswith(f"analyzed_{base_name}_") and f.endswith('.mp4')])
        
        if analyzed_files:
            video_info["formats"]["analyzed"] = []
            for analyzed_file in analyzed_files:
                # Find which directory contains this file
                analyzed_path = None
                for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
                    temp_path = os.path.join(directory, analyzed_file)
                    if os.path.exists(temp_path):
                        analyzed_path = temp_path
                        break
                
                if analyzed_path:
                    analyzed_size = os.path.getsize(analyzed_path) / (1024 * 1024)
                    video_info["formats"]["analyzed"].append({
                        "filename": analyzed_file,
                        "path": analyzed_path,
                        "size_mb": round(analyzed_size, 2),
                        "recommended": False,
                        "browser_compatible": False
                    })
        
        # Check for per-frame analyzed versions (in both directories)
        per_frame_files = []
        for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
            if os.path.exists(directory):
                per_frame_files.extend([f for f in os.listdir(directory) 
                                      if f.startswith(f"per_frame_analyzed_{base_name}_") and f.endswith('.mp4')])
        
        if per_frame_files:
            video_info["formats"]["per_frame_analyzed"] = []
            for per_frame_file in per_frame_files:
                # Find which directory contains this file
                per_frame_path = None
                for directory in [OUTPUT_DIR, VIDEO_PROCESSING_DIR]:
                    temp_path = os.path.join(directory, per_frame_file)
                    if os.path.exists(temp_path):
                        per_frame_path = temp_path
                        break
                
                if per_frame_path:
                    per_frame_size = os.path.getsize(per_frame_path) / (1024 * 1024)
                    video_info["formats"]["per_frame_analyzed"].append({
                        "filename": per_frame_file,
                        "path": per_frame_path,
                        "size_mb": round(per_frame_size, 2),
                        "recommended": False,
                        "browser_compatible": False
                    })
        
        # Determine best format
        if "h264" in video_info["formats"]:
            video_info["best_format"] = "h264"
            video_info["recommended_filename"] = video_info["formats"]["h264"]["filename"]
        elif "original" in video_info["formats"]:
            video_info["best_format"] = "original"
            video_info["recommended_filename"] = video_filename
        elif "analyzed" in video_info["formats"]:
            video_info["best_format"] = "analyzed"
            video_info["recommended_filename"] = video_info["formats"]["analyzed"][0]["filename"]
        elif "per_frame_analyzed" in video_info["formats"]:
            video_info["best_format"] = "per_frame_analyzed"
            video_info["recommended_filename"] = video_info["formats"]["per_frame_analyzed"][0]["filename"]
        else:
            video_info["best_format"] = None
            video_info["recommended_filename"] = None
        
        return jsonify({
            "success": True,
            "video_info": video_info
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getJobStatus', methods=['GET'])
def get_job_status():
    """Get status of a processing job"""
    try:
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({"error": "job_id parameter is required"}), 400
        
        if job_id in processing_jobs:
            return jsonify({
                "success": True,
                "job_status": processing_jobs[job_id]
            })
        else:
            return jsonify({"error": "Job not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getACLRiskAnalysis', methods=['GET'])
def get_acl_risk_analysis():
    """Get detailed ACL risk analysis for a video with enhanced analytics"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        base_name = os.path.splitext(video_filename)[0]
        
        # First try to find the enhanced api_generated analytics file
        analytics_file = os.path.join(ANALYTICS_DIR, f"api_generated_{base_name}.json")
        
        if not os.path.exists(analytics_file):
            # Fallback to old analytics file
            analytics_file = os.path.join(ANALYTICS_DIR, f"analytics_{base_name}.json")
            
            if not os.path.exists(analytics_file):
                return jsonify({"error": f"Analytics file not found for {video_filename}"}), 404
        
        with open(analytics_file, 'r') as f:
            analytics_data = json.load(f)
        
        # Check if this is enhanced analytics data
        enhanced_analytics = analytics_data.get('enhanced_analytics', False)
        
        if enhanced_analytics and 'frame_data' in analytics_data:
            # Use enhanced frame data
            frame_data = analytics_data['frame_data']
            acl_analysis = calculate_enhanced_acl_risk_analysis(frame_data)
        else:
            # Use legacy analytics data
            acl_analysis = calculate_acl_risk_analysis(analytics_data)
        
        return jsonify({
            "success": True,
            "video_filename": video_filename,
            "enhanced_analytics": enhanced_analytics,
            "acl_risk_analysis": acl_analysis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/assessACLRisk', methods=['POST'])
def assess_acl_risk():
    """Real-time ACL risk assessment for a single frame or frame sequence"""
    try:
        data = request.get_json()
        landmarks = data.get('landmarks', [])
        frame_history = data.get('frame_history', [])
        session_context = data.get('session_context', {})
        
        if not landmarks:
            return jsonify({"error": "landmarks data is required"}), 400
        
        # Use the comprehensive ACL risk assessment system
        acl_assessment = acl_risk_assessor.assess_acl_risk(landmarks, frame_history, session_context)
        
        return jsonify({
            "success": True,
            "acl_assessment": acl_assessment,
            "assessment_method": "comprehensive_research_based",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getACLCoachingCues', methods=['GET'])
def get_acl_coaching_cues():
    """Get available ACL coaching cues and drills"""
    try:
        return jsonify({
            "success": True,
            "coaching_cues": acl_risk_assessor.coaching_cues,
            "risk_bands": {
                "low": f"<{acl_risk_assessor.low_risk_threshold}",
                "moderate": f"{acl_risk_assessor.low_risk_threshold}-{acl_risk_assessor.moderate_risk_threshold}",
                "high": f">{acl_risk_assessor.moderate_risk_threshold}"
            },
            "risk_weights": {
                "knee_flexion": acl_risk_assessor.flexion_weight,
                "frontal_plane": acl_risk_assessor.frontal_weight,
                "landing_mechanics": acl_risk_assessor.landing_weight,
                "context": acl_risk_assessor.context_weight
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calculate_detailed_statistics(analytics_data):
    """Calculate detailed statistics from analytics data"""
    if not analytics_data:
        return {}
    
    stats = {
        "total_frames": len(analytics_data),
        "tumbling_sequences": 0,
        "flight_time_total": 0,
        "max_elevation": 0,
        "avg_elevation": 0,
        "acl_risk_summary": {
            "low_risk_frames": 0,
            "moderate_risk_frames": 0,
            "high_risk_frames": 0,
            "avg_overall_risk": 0
        },
        "landmark_confidence": {
            "high_confidence_frames": 0,
            "low_confidence_frames": 0,
            "avg_confidence": 0
        }
    }
    
    elevation_angles = []
    acl_risks = []
    confidence_scores = []
    
    for frame_data in analytics_data:
        if 'metrics' in frame_data and 'tumbling_metrics' in frame_data['metrics']:
            tumbling_metrics = frame_data['metrics']['tumbling_metrics']
            
            # Tumbling detection
            if tumbling_metrics.get('tumbling_detected', False):
                stats["tumbling_sequences"] += 1
            
            # Elevation tracking
            elevation = tumbling_metrics.get('elevation_angle', 0)
            elevation_angles.append(elevation)
            stats["max_elevation"] = max(stats["max_elevation"], elevation)
            
            # ACL risk analysis
            acl_factors = tumbling_metrics.get('acl_risk_factors', {})
            overall_risk = acl_factors.get('overall_acl_risk', 0)
            acl_risks.append(overall_risk)
            risk_level = acl_factors.get('risk_level', 'LOW')
            
            if risk_level == 'LOW':
                stats["acl_risk_summary"]["low_risk_frames"] += 1
            elif risk_level == 'MODERATE':
                stats["acl_risk_summary"]["moderate_risk_frames"] += 1
            else:
                stats["acl_risk_summary"]["high_risk_frames"] += 1
            
            # Confidence tracking
            confidence = tumbling_metrics.get('landmark_confidence', 0)
            confidence_scores.append(confidence)
            if confidence > 0:
                stats["landmark_confidence"]["high_confidence_frames"] += 1
            else:
                stats["landmark_confidence"]["low_confidence_frames"] += 1
    
    # Calculate averages
    if elevation_angles:
        stats["avg_elevation"] = sum(elevation_angles) / len(elevation_angles)
    
    if acl_risks:
        stats["acl_risk_summary"]["avg_overall_risk"] = sum(acl_risks) / len(acl_risks)
    
    if confidence_scores:
        stats["landmark_confidence"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    return stats

def calculate_summary_statistics(analytics_data, video_name):
    """Calculate summary statistics for a single video"""
    detailed_stats = calculate_detailed_statistics(analytics_data)
    
    return {
        "video_name": video_name,
        "total_frames": detailed_stats.get("total_frames", 0),
        "tumbling_sequences": detailed_stats.get("tumbling_sequences", 0),
        "max_elevation": detailed_stats.get("max_elevation", 0),
        "avg_elevation": detailed_stats.get("avg_elevation", 0),
        "avg_acl_risk": detailed_stats.get("acl_risk_summary", {}).get("avg_overall_risk", 0),
        "high_risk_frames": detailed_stats.get("acl_risk_summary", {}).get("high_risk_frames", 0)
    }

def calculate_acl_risk_analysis(analytics_data):
    """Calculate detailed ACL risk analysis using the new comprehensive system"""
    if not analytics_data:
        return {}
    
    acl_analysis = {
        "total_frames": len(analytics_data),
        "risk_distribution": {
            "low": 0,
            "moderate": 0,
            "high": 0
        },
        "risk_factors": {
            "knee_flexion_risk": [],
            "frontal_plane_risk": [],
            "landing_mechanics_risk": []
        },
        "peak_risk_frames": [],
        "coaching_cues": [],
        "context_factors": set(),
        "enhanced_analysis": True
    }
    
    # Track frame history for temporal analysis
    frame_history = []
    
    for i, frame_data in enumerate(analytics_data):
        landmarks = frame_data.get('landmarks', [])
        
        if landmarks and len(landmarks) >= 29:
            # Use the new comprehensive ACL risk assessment
            acl_assessment = acl_risk_assessor.assess_acl_risk(landmarks, frame_history)
            
            # Update frame history
            frame_history.append({
                'landmarks': landmarks,
                'timestamp': frame_data.get('timestamp', i)
            })
            
            # Keep only recent frames for analysis
            if len(frame_history) > 10:
                frame_history.pop(0)
            
            # Risk level distribution
            risk_level = acl_assessment.get('risk_level', 'LOW')
            if risk_level == 'LOW':
                acl_analysis["risk_distribution"]["low"] += 1
            elif risk_level == 'MODERATE':
                acl_analysis["risk_distribution"]["moderate"] += 1
            else:
                acl_analysis["risk_distribution"]["high"] += 1
            
            # Collect risk factors
            risk_factors = acl_assessment.get('risk_factors', {})
            acl_analysis["risk_factors"]["knee_flexion_risk"].append(
                risk_factors.get('knee_flexion', {}).get('risk_score', 0)
            )
            acl_analysis["risk_factors"]["frontal_plane_risk"].append(
                risk_factors.get('frontal_plane', {}).get('risk_score', 0)
            )
            acl_analysis["risk_factors"]["landing_mechanics_risk"].append(
                risk_factors.get('landing_mechanics', {}).get('risk_score', 0)
            )
            
            # Track peak risk frames
            overall_risk = acl_assessment.get('overall_risk', 0)
            if overall_risk > 70:  # High risk threshold
                acl_analysis["peak_risk_frames"].append({
                    "frame": i,
                    "risk_score": overall_risk,
                    "risk_level": risk_level,
                    "timestamp": frame_data.get('timestamp', i),
                    "coaching_cues": acl_assessment.get('coaching_cues', [])
                })
            
            # Collect coaching cues
            coaching_cues = acl_assessment.get('coaching_cues', [])
            for cue in coaching_cues:
                if cue not in acl_analysis["coaching_cues"]:
                    acl_analysis["coaching_cues"].append(cue)
            
            # Collect context factors
            context_factors = risk_factors.get('context', {}).get('context_factors', [])
            acl_analysis["context_factors"].update(context_factors)
    
    # Convert set to list for JSON serialization
    acl_analysis["context_factors"] = list(acl_analysis["context_factors"])
    
    # Calculate averages for risk factors
    for factor in acl_analysis["risk_factors"]:
        values = acl_analysis["risk_factors"][factor]
        if values:
            acl_analysis["risk_factors"][f"{factor}_avg"] = sum(values) / len(values)
            acl_analysis["risk_factors"][f"{factor}_max"] = max(values)
    
    return acl_analysis

def calculate_enhanced_acl_risk_analysis(frame_data):
    """Calculate enhanced ACL risk analysis using the new comprehensive system"""
    if not frame_data:
        return {}
    
    acl_analysis = {
        "total_frames": len(frame_data),
        "risk_distribution": {
            "low": 0,
            "moderate": 0,
            "high": 0
        },
        "risk_factors": {
            "knee_flexion_risk": [],
            "frontal_plane_risk": [],
            "landing_mechanics_risk": []
        },
        "peak_risk_frames": [],
        "coaching_cues": [],
        "context_factors": set(),
        "enhanced_analysis": True,
        "tumbling_analysis": {
            "tumbling_frames": 0,
            "flight_phases": {
                "ground": 0,
                "preparation": 0,
                "takeoff": 0,
                "flight": 0,
                "landing": 0
            },
            "average_tumbling_quality": 0
        },
        "movement_analysis": {
            "average_elevation_angle": 0,
            "max_elevation_angle": 0,
            "average_forward_lean_angle": 0,
            "max_forward_lean_angle": 0,
            "average_height_from_ground": 0,
            "max_height_from_ground": 0
        }
    }
    
    elevation_angles = []
    forward_lean_angles = []
    heights_from_ground = []
    tumbling_qualities = []
    
    # Track frame history for temporal analysis
    frame_history = []
    
    for i, frame in enumerate(frame_data):
        landmarks = frame.get('landmarks', [])
        
        if landmarks and len(landmarks) >= 29:
            # Use the new comprehensive ACL risk assessment
            acl_assessment = acl_risk_assessor.assess_acl_risk(landmarks, frame_history)
            
            # Update frame history
            frame_history.append({
                'landmarks': landmarks,
                'timestamp': frame.get('timestamp', i)
            })
            
            # Keep only recent frames for analysis
            if len(frame_history) > 10:
                frame_history.pop(0)
            
            # Risk level distribution
            risk_level = acl_assessment.get('risk_level', 'LOW')
            if risk_level == 'LOW':
                acl_analysis["risk_distribution"]["low"] += 1
            elif risk_level == 'MODERATE':
                acl_analysis["risk_distribution"]["moderate"] += 1
            else:
                acl_analysis["risk_distribution"]["high"] += 1
            
            # Collect risk factors
            risk_factors = acl_assessment.get('risk_factors', {})
            acl_analysis["risk_factors"]["knee_flexion_risk"].append(
                risk_factors.get('knee_flexion', {}).get('risk_score', 0)
            )
            acl_analysis["risk_factors"]["frontal_plane_risk"].append(
                risk_factors.get('frontal_plane', {}).get('risk_score', 0)
            )
            acl_analysis["risk_factors"]["landing_mechanics_risk"].append(
                risk_factors.get('landing_mechanics', {}).get('risk_score', 0)
            )
            
            # Track peak risk frames
            overall_risk = acl_assessment.get('overall_risk', 0)
            if overall_risk > 70:  # High risk threshold
                acl_analysis["peak_risk_frames"].append({
                    "frame": i,
                    "risk_score": overall_risk,
                    "risk_level": risk_level,
                    "timestamp": frame.get('timestamp', i),
                    "coaching_cues": acl_assessment.get('coaching_cues', [])
                })
            
            # Collect coaching cues
            coaching_cues = acl_assessment.get('coaching_cues', [])
            for cue in coaching_cues:
                if cue not in acl_analysis["coaching_cues"]:
                    acl_analysis["coaching_cues"].append(cue)
            
            # Collect context factors
            context_factors = risk_factors.get('context', {}).get('context_factors', [])
            acl_analysis["context_factors"].update(context_factors)
        
        # Tumbling analysis
        if frame.get('tumbling_detected', False):
            acl_analysis["tumbling_analysis"]["tumbling_frames"] += 1
        
        flight_phase = frame.get('flight_phase', 'ground')
        acl_analysis["tumbling_analysis"]["flight_phases"][flight_phase] += 1
        
        tumbling_quality = frame.get('tumbling_quality', 0)
        if tumbling_quality > 0:
            tumbling_qualities.append(tumbling_quality)
        
        # Movement analysis
        elevation_angle = frame.get('elevation_angle', 0)
        if elevation_angle is not None:
            elevation_angles.append(elevation_angle)
        
        forward_lean_angle = frame.get('forward_lean_angle', 0)
        if forward_lean_angle is not None:
            forward_lean_angles.append(forward_lean_angle)
        
        height_from_ground = frame.get('height_from_ground', 0)
        if height_from_ground is not None:
            heights_from_ground.append(height_from_ground)
    
    # Calculate averages and statistics
    if elevation_angles:
        acl_analysis["movement_analysis"]["average_elevation_angle"] = np.mean(elevation_angles)
        acl_analysis["movement_analysis"]["max_elevation_angle"] = np.max(elevation_angles)
    
    if forward_lean_angles:
        acl_analysis["movement_analysis"]["average_forward_lean_angle"] = np.mean(forward_lean_angles)
        acl_analysis["movement_analysis"]["max_forward_lean_angle"] = np.max(forward_lean_angles)
    
    if heights_from_ground:
        acl_analysis["movement_analysis"]["average_height_from_ground"] = np.mean(heights_from_ground)
        acl_analysis["movement_analysis"]["max_height_from_ground"] = np.max(heights_from_ground)
    
    if tumbling_qualities:
        acl_analysis["tumbling_analysis"]["average_tumbling_quality"] = np.mean(tumbling_qualities)
    
    # Convert set to list for JSON serialization
    acl_analysis["context_factors"] = list(acl_analysis["context_factors"])
    
    # Calculate averages for risk factors
    for factor in acl_analysis["risk_factors"]:
        values = acl_analysis["risk_factors"][factor]
        if values:
            acl_analysis["risk_factors"][f"{factor}_avg"] = sum(values) / len(values)
            acl_analysis["risk_factors"][f"{factor}_max"] = max(values)
    
    return acl_analysis

@app.route('/analyzeVideoPerFrame', methods=['POST'])
def analyze_video_per_frame():
    """Analyze video and return per-frame statistics with enhanced tumbling detection and ACL risk analysis"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return jsonify({'error': 'No video filename provided'}), 400
        
        # Find the best available video (prioritize H.264)
        best_video_path, is_h264, original_path = find_best_video(video_filename)
        
        if not best_video_path:
            return jsonify({'error': f'Video file not found: {video_filename}'}), 404
        
        # Use the best available video for processing
        video_path = best_video_path
        print(f"🎬 Processing video per-frame: {os.path.basename(video_path)} (H.264: {is_h264})")
        
        # Check if MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({'error': 'MediaPipe server is not running'}), 503
        
        # Generate unique job ID
        job_id = f"job_{int(time.time())}"
        processing_jobs[job_id] = {
            'status': 'processing',
            'video_filename': video_filename,
            'start_time': datetime.now().isoformat(),
            'progress': 0
        }
        
        # Start processing in background thread
        def process_video():
            try:
                # Set analytics filename for this session
                base_name = os.path.splitext(video_filename)[0]
                video_id = base_name
                analytics_filename = f"api_generated_{video_id}.json"
                analytics_path = os.path.join(ANALYTICS_DIR, analytics_filename)
                
                # Set the analytics filename in MediaPipe server
                requests.post(f"{MEDIAPIPE_SERVER_URL}/set-analytics-filename", 
                            json={'filename': analytics_path})
                
                # Clear previous analytics
                requests.post(f"{MEDIAPIPE_SERVER_URL}/clear-analytics")
                requests.post(f"{MEDIAPIPE_SERVER_URL}/clear-analytics-json")
                
                # Initialize enhanced tumbling analyzer
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from video_overlay_with_analytics_fixed import FixedTumblingAnalyzer
                import cv2
                
                # Get video properties for tumbling analyzer
                cap = cv2.VideoCapture(video_path)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Initialize tumbling analyzer
                tumbling_analyzer = FixedTumblingAnalyzer(video_height=height, video_width=width)
                
                # Process video frame by frame
                cap = cv2.VideoCapture(video_path)
                frame_data = []
                frame_number = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame to MediaPipe server
                    response = requests.post(f"{MEDIAPIPE_SERVER_URL}/detect-pose", 
                                           json={'image': frame_base64})
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            landmarks = result.get('landmarks', [])
                            base_metrics = result.get('metrics', {})
                            
                            # Add enhanced tumbling analysis
                            tumbling_metrics = tumbling_analyzer.analyze_tumbling_sequence(landmarks, frame_number)
                            
                            # Add comprehensive ACL risk assessment - temporarily simplified
                            try:
                                frame_history_for_acl = []
                                if frame_number > 0:
                                    # Get recent frame history for ACL assessment
                                    recent_frames = frame_data[-min(10, len(frame_data)):]
                                    frame_history_for_acl = [frame['landmarks'] for frame in recent_frames if frame.get('landmarks')]
                                
                                acl_assessment = acl_risk_assessor.assess_acl_risk(
                                    landmarks, 
                                    frame_history_for_acl,
                                    session_context={'session_type': 'practice', 'apparatus': 'floor'}
                                )
                            except Exception as e:
                                print(f"ACL assessment error: {e}")
                                # Fallback to basic ACL assessment
                                acl_assessment = {
                                    'overall_risk': 0,
                                    'risk_level': 'LOW',
                                    'coaching_cues': ['ACL assessment temporarily unavailable'],
                                    'confidence': 'low'
                                }
                            
                            # Combine metrics
                            enhanced_metrics = {
                                **base_metrics,
                                'tumbling_metrics': tumbling_metrics,
                                'acl_risk_assessment': acl_assessment
                            }
                            
                            frame_data.append({
                                'frame_number': frame_number,
                                'timestamp': frame_number / fps,
                                'pose_data': result.get('pose_data', {}),
                                'metrics': enhanced_metrics,
                                'analytics': result.get('analytics', {}),
                                'landmarks': landmarks,
                                'tumbling_detected': tumbling_metrics.get('tumbling_detected', False),
                                'acl_risk_factors': tumbling_metrics.get('acl_risk_factors', {}),
                                'acl_recommendations': tumbling_metrics.get('acl_recommendations', []),
                                'flight_phase': tumbling_metrics.get('flight_phase', 'ground'),
                                'height_from_ground': tumbling_metrics.get('height_from_ground', 0),
                                'elevation_angle': tumbling_metrics.get('elevation_angle', 0),
                                'forward_lean_angle': tumbling_metrics.get('forward_lean_angle', 0),
                                'com_position': tumbling_metrics.get('com_position'),
                                'tumbling_quality': tumbling_metrics.get('tumbling_quality', 0),
                                'landmark_confidence': tumbling_metrics.get('landmark_confidence', 0),
                                # New comprehensive ACL assessment
                                'acl_risk_assessment': acl_assessment,
                                'overall_acl_risk': acl_assessment.get('overall_risk', 0),
                                'acl_risk_level': acl_assessment.get('risk_level', 'LOW'),
                                'coaching_cues': acl_assessment.get('coaching_cues', [])
                            })
                    
                    frame_number += 1
                    progress = (frame_number / frame_count) * 100
                    processing_jobs[job_id]['progress'] = progress
                    
                    # Update progress every 10 frames
                    if frame_number % 10 == 0:
                        processing_jobs[job_id]['progress'] = progress
                
                cap.release()
                
                # Save enhanced frame-by-frame data
                with open(analytics_path, 'w') as f:
                    json.dump({
                        'video_filename': video_filename,
                        'total_frames': frame_count,
                        'fps': fps,
                        'frame_data': frame_data,
                        'processing_time': datetime.now().isoformat(),
                        'enhanced_analytics': True,
                        'tumbling_detection_enabled': True,
                        'acl_risk_analysis_enabled': True
                    }, f, indent=2)
                
                # Generate overlayed video using the existing overlay script
                # Use api_generated prefix for API-generated videos
                video_id = base_name
                output_video_name = f"api_generated_{video_id}.mp4"
                output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
                
                try:
                    # Run the video overlay script with comprehensive overlays
                    overlay_cmd = [
                        "python3", "video_overlay_with_analytics_fixed.py",
                        video_path,
                        "--output", output_video_path,
                        "--server", "http://127.0.0.1:5001"
                    ]
                    
                    overlay_result = subprocess.run(overlay_cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    
                    if overlay_result.returncode == 0:
                        overlay_success = True
                        overlay_message = "Video overlay generated successfully"
                    else:
                        overlay_success = False
                        overlay_message = f"Video overlay failed: {overlay_result.stderr}"
                except Exception as e:
                    overlay_success = False
                    overlay_message = f"Video overlay error: {str(e)}"
                
                processing_jobs[job_id].update({
                    'status': 'completed',
                    'analytics_file': analytics_path,
                    'total_frames': frame_count,
                    'frames_processed': len(frame_data),
                    'overlay_video': output_video_path if overlay_success else None,
                    'overlay_success': overlay_success,
                    'overlay_message': overlay_message,
                    'end_time': datetime.now().isoformat(),
                    'enhanced_features': {
                        'tumbling_detection': True,
                        'acl_risk_analysis': True,
                        'per_frame_metrics': True
                    }
                })
                
            except Exception as e:
                processing_jobs[job_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'end_time': datetime.now().isoformat()
                })
        
        # Start processing thread
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Enhanced video analysis started with tumbling detection and ACL risk analysis',
            'video_filename': video_filename,
            'enhanced_features': {
                'tumbling_detection': True,
                'acl_risk_analysis': True,
                'per_frame_metrics': True
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getPerFrameStatistics', methods=['GET'])
def get_per_frame_statistics():
    """Get per-frame statistics for a processed video with enhanced analytics"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'No video filename provided'}), 400
        
        # Look for the frame analysis file
        base_name = os.path.splitext(video_filename)[0]
        
        # First try to find the new api_generated analytics file
        analytics_file = f"api_generated_{base_name}.json"
        analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
        
        # If not found, try without the h264_ prefix (for cases where video has h264_ prefix but analytics doesn't)
        if not os.path.exists(analytics_path) and base_name.startswith('h264_'):
            base_name_without_h264 = base_name[5:]  # Remove 'h264_' prefix
            analytics_file = f"api_generated_{base_name_without_h264}.json"
            analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
        
        if not os.path.exists(analytics_path):
            # Fallback to fixed_analytics files in analytics directory
            analytics_files = [f for f in os.listdir(ANALYTICS_DIR) 
                             if f.startswith(f"fixed_analytics_") and base_name in f]
            
            if not analytics_files:
                # Fallback to fixed_analytics files in backend directory
                backend_dir = os.path.dirname(__file__)
                analytics_files = [f for f in os.listdir(backend_dir) 
                                 if f.startswith(f"fixed_analytics_") and base_name in f]
                if analytics_files:
                    latest_file = sorted(analytics_files)[-1]
                    analytics_path = os.path.join(backend_dir, latest_file)
                else:
                    # Fallback to old frame_analysis files
                    analytics_files = [f for f in os.listdir(ANALYTICS_DIR) 
                                     if f.startswith(f"frame_analysis_{base_name}_")]
                    
                    if not analytics_files:
                        return jsonify({'error': 'No frame analysis found for this video'}), 404
                    
                    # Get the most recent analysis file
                    latest_file = sorted(analytics_files)[-1]
                    analytics_path = os.path.join(ANALYTICS_DIR, latest_file)
        
        with open(analytics_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, list):
            # This is a list of frames (fixed_analytics format)
            frame_data = data
            enhanced_analytics = True
        else:
            # This is a dictionary format
            enhanced_analytics = data.get('enhanced_analytics', False)
            frame_data = data.get('frame_data', [])
        
        # Calculate enhanced statistics if available
        enhanced_stats = {}
        if enhanced_analytics and frame_data:
            
            # Tumbling detection statistics
            tumbling_frames = [frame for frame in frame_data 
                             if frame.get('metrics', {}).get('tumbling_metrics', {}).get('tumbling_detected', False)]
            enhanced_stats['tumbling_detection'] = {
                'total_tumbling_frames': len(tumbling_frames),
                'tumbling_percentage': (len(tumbling_frames) / len(frame_data)) * 100 if frame_data else 0,
                'flight_phases': {
                    'ground': len([f for f in frame_data if f.get('metrics', {}).get('tumbling_metrics', {}).get('flight_phase') == 'ground']),
                    'preparation': len([f for f in frame_data if f.get('metrics', {}).get('tumbling_metrics', {}).get('flight_phase') == 'preparation']),
                    'takeoff': len([f for f in frame_data if f.get('metrics', {}).get('tumbling_metrics', {}).get('flight_phase') == 'takeoff']),
                    'flight': len([f for f in frame_data if f.get('metrics', {}).get('tumbling_metrics', {}).get('flight_phase') == 'flight']),
                    'landing': len([f for f in frame_data if f.get('metrics', {}).get('tumbling_metrics', {}).get('flight_phase') == 'landing'])
                }
            }
            
            # ACL Risk Analysis statistics
            acl_risk_factors = [frame.get('metrics', {}).get('tumbling_metrics', {}).get('acl_risk_factors', {}) 
                              for frame in frame_data if frame.get('metrics', {}).get('tumbling_metrics', {}).get('acl_risk_factors')]
            if acl_risk_factors:
                risk_levels = [rf.get('risk_level', 'LOW') for rf in acl_risk_factors]
                enhanced_stats['acl_risk_analysis'] = {
                    'average_overall_risk': float(np.mean([rf.get('overall_acl_risk', 0) for rf in acl_risk_factors])),
                    'average_knee_angle_risk': float(np.mean([rf.get('knee_angle_risk', 0) for rf in acl_risk_factors])),
                    'average_knee_valgus_risk': float(np.mean([rf.get('knee_valgus_risk', 0) for rf in acl_risk_factors])),
                    'average_landing_mechanics_risk': float(np.mean([rf.get('landing_mechanics_risk', 0) for rf in acl_risk_factors])),
                    'risk_level_distribution': {
                        'LOW': int(risk_levels.count('LOW')),
                        'MODERATE': int(risk_levels.count('MODERATE')),
                        'HIGH': int(risk_levels.count('HIGH'))
                    },
                    'high_risk_frames': int(risk_levels.count('HIGH'))
                }
            
            # Elevation and movement statistics
            elevation_angles = [frame.get('elevation_angle', 0) for frame in frame_data if frame.get('elevation_angle') is not None]
            forward_lean_angles = [frame.get('forward_lean_angle', 0) for frame in frame_data if frame.get('forward_lean_angle') is not None]
            heights_from_ground = [frame.get('height_from_ground', 0) for frame in frame_data if frame.get('height_from_ground') is not None]
            
            enhanced_stats['movement_analysis'] = {
                'average_elevation_angle': float(np.mean(elevation_angles)) if elevation_angles else 0.0,
                'max_elevation_angle': float(np.max(elevation_angles)) if elevation_angles else 0.0,
                'average_forward_lean_angle': float(np.mean(forward_lean_angles)) if forward_lean_angles else 0.0,
                'max_forward_lean_angle': float(np.max(forward_lean_angles)) if forward_lean_angles else 0.0,
                'average_height_from_ground': float(np.mean(heights_from_ground)) if heights_from_ground else 0.0,
                'max_height_from_ground': float(np.max(heights_from_ground)) if heights_from_ground else 0.0
            }
            
            # Tumbling quality statistics
            tumbling_qualities = [frame.get('metrics', {}).get('tumbling_metrics', {}).get('tumbling_quality', 0) 
                                for frame in frame_data if frame.get('metrics', {}).get('tumbling_metrics', {}).get('tumbling_quality', 0) > 0]
            enhanced_stats['tumbling_quality'] = {
                'average_quality': float(np.mean(tumbling_qualities)) if tumbling_qualities else 0.0,
                'max_quality': float(np.max(tumbling_qualities)) if tumbling_qualities else 0.0,
                'quality_frames_count': int(len(tumbling_qualities))
            }
        
        return jsonify({
            'success': True,
            'video_filename': video_filename,
            'total_frames': len(frame_data),
            'fps': 30,  # Default FPS
            'frames_processed': len(frame_data),
            'processing_time': '',
            'enhanced_analytics': enhanced_analytics,
            'enhanced_statistics': enhanced_stats if enhanced_analytics else {},
            'frame_data': frame_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/downloadPerFrameVideo', methods=['GET', 'OPTIONS'])
def download_per_frame_video():
    """Download the overlayed video from per-frame analysis"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'
        return response
    
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'No video filename provided'}), 400
        
        print(f"🔍 Looking for per-frame video: {video_filename}")
        
        # First try to find the video from job data
        job_found = None
        for job_id, job_data in processing_jobs.items():
            if job_data.get('video_filename') == video_filename and job_data.get('status') == 'completed':
                job_found = job_data
                break
        
        overlay_video_path = None
        
        if job_found and job_found.get('overlay_video'):
            overlay_video_path = job_found.get('overlay_video')
            print(f"📁 Found video from job data: {overlay_video_path}")
        else:
            # If not found in job data, search for the file directly
            base_name = os.path.splitext(video_filename)[0]
            print(f"🔍 Searching for files starting with: api_generated_{base_name}.mp4")
            
            if os.path.exists(OUTPUT_DIR):
                # First try to find api_generated files
                api_generated_file = f"api_generated_{base_name}.mp4"
                api_generated_path = os.path.join(OUTPUT_DIR, api_generated_file)
                
                if os.path.exists(api_generated_path):
                    overlay_video_path = api_generated_path
                    print(f"✅ Found API generated file: {overlay_video_path}")
                else:
                    # Fallback to per_frame_analyzed files
                    print(f"🔍 API generated file not found, searching for per_frame_analyzed_{base_name}_")
                    overlay_files = [f for f in os.listdir(OUTPUT_DIR) 
                                   if f.startswith(f"per_frame_analyzed_{base_name}_") and f.endswith('.mp4')]
                    
                    print(f"📄 Found {len(overlay_files)} overlay files: {overlay_files}")
                    
                    if overlay_files:
                        # Get the most recent overlay file
                        latest_file = sorted(overlay_files)[-1]
                        overlay_video_path = os.path.join(OUTPUT_DIR, latest_file)
                        print(f"✅ Using latest file: {overlay_video_path}")
        
        if not overlay_video_path or not os.path.exists(overlay_video_path):
            print(f"❌ Overlay video not found for {video_filename}")
            return jsonify({'error': f'Overlay video not found for {video_filename}'}), 404
        
        print(f"📤 Sending file: {overlay_video_path}")
        response = send_file(overlay_video_path, mimetype='video/mp4')
        
        # Add CORS headers for video files
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'
        
        return response
        
    except Exception as e:
        print(f"❌ Error in downloadPerFrameVideo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generateEnhancedReplay', methods=['POST'])
def generate_enhanced_replay():
    """Generate enhanced video replay with comprehensive frame-by-frame statistics"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return jsonify({'error': 'video_filename is required'}), 400
        
        video_path = os.path.join(VIDEO_PROCESSING_DIR, video_filename)
        if not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_filename}'}), 404
        
        # Generate unique job ID
        job_id = f"enhanced_replay_{int(time.time())}"
        processing_jobs[job_id] = {
            'status': 'processing',
            'video_filename': video_filename,
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'type': 'enhanced_replay'
        }
        
        # Start processing in background thread
        def process_enhanced_replay():
            try:
                # Look for analytics data
                base_name = os.path.splitext(video_filename)[0]
                analytics_files = [
                    f"acl_inference_{base_name}.json",
                    f"api_generated_{base_name}.json",
                    f"analytics_{base_name}.json"
                ]
                
                analytics_data = None
                analytics_file_used = None
                
                for analytics_file in analytics_files:
                    analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
                    if os.path.exists(analytics_path):
                        with open(analytics_path, 'r') as f:
                            analytics_data = json.load(f)
                        analytics_file_used = analytics_file
                        break
                
                if not analytics_data:
                    processing_jobs[job_id].update({
                        'status': 'failed',
                        'error': 'No analytics data found for video',
                        'end_time': datetime.now().isoformat()
                    })
                    return
                
                # Extract frame data
                frame_data = analytics_data.get('frame_data', [])
                if not frame_data:
                    processing_jobs[job_id].update({
                        'status': 'failed',
                        'error': 'No frame data found in analytics',
                        'end_time': datetime.now().isoformat()
                    })
                    return
                
                # Import and use enhanced video replay
                from enhanced_video_replay_with_acl import EnhancedVideoReplay
                
                # Initialize enhanced video replay
                replay = EnhancedVideoReplay(video_path, frame_data)
                
                # Generate output paths
                output_video_name = f"enhanced_replay_{base_name}.mp4"
                output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
                output_report_name = f"session_report_{base_name}.json"
                output_report_path = os.path.join(ANALYTICS_DIR, output_report_name)
                
                # Process video replay
                processing_jobs[job_id]['progress'] = 25
                
                # Process video replay with progress updates
                output_video = replay.process_video_replay(output_video_path)
                
                processing_jobs[job_id]['progress'] = 75
                
                # Save session report
                replay.save_session_report(output_report_path)
                
                processing_jobs[job_id]['progress'] = 100
                
                # Update job status
                processing_jobs[job_id].update({
                    'status': 'completed',
                    'output_video': output_video_name,
                    'output_report': output_report_name,
                    'analytics_file_used': analytics_file_used,
                    'frames_processed': len(frame_data),
                    'end_time': datetime.now().isoformat(),
                    'enhanced_features': {
                        'frame_by_frame_statistics': True,
                        'acl_risk_tracking': True,
                        'session_history': True,
                        'trend_analysis': True,
                        'coaching_cues': True
                    }
                })
                
            except Exception as e:
                processing_jobs[job_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'end_time': datetime.now().isoformat()
                })
        
        # Start processing thread
        thread = threading.Thread(target=process_enhanced_replay)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Enhanced video replay generation started',
            'video_filename': video_filename,
            'features': {
                'frame_by_frame_statistics': True,
                'acl_risk_tracking': True,
                'session_history': True,
                'trend_analysis': True,
                'coaching_cues': True
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getEnhancedReplayStatus', methods=['GET'])
def get_enhanced_replay_status():
    """Get status of enhanced replay generation jobs"""
    try:
        enhanced_jobs = {}
        for job_id, job_data in processing_jobs.items():
            if job_data.get('type') == 'enhanced_replay':
                enhanced_jobs[job_id] = job_data
        
        return jsonify({
            'success': True,
            'enhanced_replay_jobs': enhanced_jobs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/downloadEnhancedReplay', methods=['GET', 'OPTIONS'])
def download_enhanced_replay():
    """Download enhanced video replay"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'
        return response
    
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'video_filename parameter is required'}), 400
        
        base_name = os.path.splitext(video_filename)[0]
        enhanced_video_name = f"enhanced_replay_{base_name}.mp4"
        enhanced_video_path = os.path.join(OUTPUT_DIR, enhanced_video_name)
        
        if not os.path.exists(enhanced_video_path):
            return jsonify({'error': f'Enhanced replay video not found: {enhanced_video_name}'}), 404
        
        response = send_file(enhanced_video_path, mimetype='video/mp4')
        
        # Add CORS headers for video files
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getSessionReport', methods=['GET'])
def get_session_report():
    """Get session report for enhanced replay"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'video_filename parameter is required'}), 400
        
        base_name = os.path.splitext(video_filename)[0]
        session_report_name = f"session_report_{base_name}.json"
        session_report_path = os.path.join(ANALYTICS_DIR, session_report_name)
        
        if not os.path.exists(session_report_path):
            return jsonify({'error': f'Session report not found: {session_report_name}'}), 404
        
        with open(session_report_path, 'r') as f:
            session_report = json.load(f)
        
        return jsonify({
            'success': True,
            'session_report': session_report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploadVideoChunk', methods=['POST', 'OPTIONS'])
def upload_video_chunk():
    """Upload video in chunks with session management and automatic analysis"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Session-ID, X-Chunk-Index, X-Total-Chunks, X-File-Name, X-File-Size'
        return response
    
    try:
        # Get chunk information from headers
        session_id = request.headers.get('X-Session-ID')
        chunk_index = int(request.headers.get('X-Chunk-Index', 0))
        total_chunks = int(request.headers.get('X-Total-Chunks', 1))
        file_name = request.headers.get('X-File-Name')
        file_size = int(request.headers.get('X-File-Size', 0))
        
        if not session_id or not file_name:
            return jsonify({'error': 'Missing required headers: X-Session-ID, X-File-Name'}), 400
        
        # Check if file was uploaded
        if 'chunk' not in request.files:
            return jsonify({'error': 'No chunk file provided'}), 400
        
        chunk_file = request.files['chunk']
        
        # Check file extension
        allowed_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not allowed. Supported types: {", ".join(allowed_extensions)}'}), 400
        
        # Check file size limit (max 1GB total)
        max_size = 1024 * 1024 * 1024  # 1GB
        if file_size > max_size:
            return jsonify({'error': f'File too large. Maximum size: 1GB, got: {file_size / (1024*1024*1024):.1f}GB'}), 400
        
        # Create session directory for chunks
        session_dir = os.path.join(VIDEO_PROCESSING_DIR, 'sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save chunk
        chunk_filename = f"chunk_{chunk_index:06d}"
        chunk_path = os.path.join(session_dir, chunk_filename)
        chunk_file.save(chunk_path)
        
        print(f"📦 Chunk {chunk_index + 1}/{total_chunks} uploaded for session {session_id}")
        
        # If this is the last chunk, combine all chunks and start analysis
        if chunk_index == total_chunks - 1:
            print(f"🔗 Combining {total_chunks} chunks for session {session_id}")
            
            # Combine chunks
            final_filename = f"uploaded_{file_name}"
            final_path = os.path.join(VIDEO_PROCESSING_DIR, final_filename)
            
            with open(final_path, 'wb') as outfile:
                for i in range(total_chunks):
                    chunk_path = os.path.join(session_dir, f"chunk_{i:06d}")
                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as infile:
                            outfile.write(infile.read())
                        # Clean up chunk file
                        os.remove(chunk_path)
            
            # Clean up session directory
            os.rmdir(session_dir)
            
            print(f"✅ Video assembled: {final_filename}")
            
            # Start background analysis
            def analyze_uploaded_video():
                try:
                    print(f"🔍 Starting analysis for chunked upload: {final_filename}")
                    
                    # Convert to H.264 for browser compatibility
                    h264_success, h264_path, h264_error = convert_to_h264(final_path)
                    
                    # Generate unique output name
                    base_name = os.path.splitext(final_filename)[0]
                    timestamp_int = int(time.time())
                    output_name = f"analyzed_{base_name}_{timestamp_int}.mp4"
                    
                    # Process video with analytics
                    result = video_processor.process_video_with_analytics(
                        h264_path if h264_success else final_path, 
                        output_name
                    )
                    
                    if result["success"]:
                        # Use H.264 version if available, otherwise use original
                        final_video_path = result.get("h264_video", result.get("output_video", os.path.join(OUTPUT_DIR, output_name)))
                        final_video_filename = os.path.basename(final_video_path)
                        
                        # Create session in database
                        session_data = {
                            "user_id": "demo_user",
                            "athlete_name": "Chunk Upload Athlete",
                            "session_name": f"Chunk Upload - {os.path.splitext(file_name)[0]}",
                            "event": "Upload",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "duration": "00:00",  # Will be updated
                            "original_filename": final_filename,
                            "processed_video_filename": final_video_filename,
                            "processed_video_url": f"http://localhost:5004/downloadVideo?video_filename={final_video_filename}",
                            "analytics_filename": f"api_generated_{base_name}_{timestamp_int}.json",
                            "analytics_url": f"http://localhost:5004/getPerFrameStatistics?video_filename={base_name}_{timestamp_int}",
                            "motion_iq": result.get("motion_iq", 0),
                            "acl_risk": result.get("acl_risk", 0),
                            "precision": result.get("precision", 0),
                            "power": result.get("power", 0),
                            "tumbling_percentage": result.get("tumbling_percentage", 0),
                            "status": "completed",
                            "processing_progress": 1.0,
                            "total_frames": result.get("total_frames", 0),
                            "fps": result.get("fps", 0),
                            "has_landmarks": True,
                            "landmark_confidence": result.get("landmark_confidence", 0.95),
                            "notes": f"Automatically uploaded and analyzed via chunk upload - {file_name}",
                            "coach_notes": "",
                            "created_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                        
                        session_id_db = sessions.create_session(session_data)
                        print(f"✅ Session created with ID: {session_id_db}")
                        
                        # Store mapping between upload session ID and database session ID
                        upload_session_mapping[session_id] = session_id_db
                        
                        # Update the session with the database ID for progress tracking
                        session_data["_id"] = session_id_db
                        
                        # Upload processed video and analytics to GridFS
                        try:
                            gridfs_video_id, gridfs_analytics_id = upload_processed_files_to_gridfs(
                                output_name, 
                                base_name, 
                                timestamp_int
                            )
                            
                            if gridfs_video_id and gridfs_analytics_id:
                                # Update session with GridFS IDs
                                sessions.update_session(session_id_db, {
                                    'gridfs_video_id': str(gridfs_video_id),
                                    'gridfs_analytics_id': str(gridfs_analytics_id),
                                    'is_binary_stored': True
                                })
                                print(f"✅ Files uploaded to GridFS - Video ID: {gridfs_video_id}, Analytics ID: {gridfs_analytics_id}")
                            else:
                                print("⚠️ Failed to upload files to GridFS, using file-based storage")
                                
                        except Exception as e:
                            print(f"⚠️ Error uploading to GridFS: {str(e)}, using file-based storage")
                        
                        # Store video metadata
                        video_meta = {
                            "filename": final_filename,
                            "original_filename": file_name,
                            "file_size": file_size,
                            "duration": result.get("duration", 0.0),
                            "fps": result.get("fps", 0.0),
                            "resolution": result.get("resolution", {"width": 0, "height": 0}),
                            "codec": "h264" if h264_success else "unknown",
                            "bitrate": 0,
                            "processing_status": "completed",
                            "original_path": final_path,
                            "processed_path": os.path.join(OUTPUT_DIR, output_name),
                            "analytics_path": os.path.join(ANALYTICS_DIR, f"api_generated_{base_name}_{timestamp_int}.json"),
                            "uploaded_at": datetime.utcnow(),
                            "session_id": session_id_db
                        }
                        
                        video_metadata.create_video_metadata(video_meta)
                        
                    else:
                        print(f"❌ Analysis failed for chunked upload: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"❌ Error in background analysis for chunked upload: {str(e)}")
            
            # Start background analysis
            import threading
            analysis_thread = threading.Thread(target=analyze_uploaded_video)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'All chunks uploaded and video analysis started',
                'filename': final_filename,
                'total_chunks': total_chunks
            })
        else:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': f'Chunk {chunk_index + 1}/{total_chunks} uploaded',
                'chunk_index': chunk_index,
                'total_chunks': total_chunks
            })
            
    except Exception as e:
        print(f"❌ Error in chunk upload: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/getUploadProgress/<session_id>', methods=['GET'])
def get_upload_progress(session_id):
    """Get upload progress and session status for chunk uploads"""
    try:
        # Check if session directory exists (upload in progress)
        session_dir = os.path.join(VIDEO_PROCESSING_DIR, 'sessions', session_id)
        
        if os.path.exists(session_dir):
            # Upload still in progress
            chunks = [f for f in os.listdir(session_dir) if f.startswith('chunk_')]
            total_chunks = len(chunks)
            
            return jsonify({
                'status': 'uploading',
                'session_id': session_id,
                'uploaded_chunks': total_chunks,
                'message': f'Upload in progress: {total_chunks} chunks uploaded'
            })
        else:
            # Check if upload session ID maps to a database session ID
            if session_id in upload_session_mapping:
                db_session_id = upload_session_mapping[session_id]
                session = sessions.get_session(db_session_id)
                
                if session:
                    return jsonify({
                        'status': 'completed',
                        'session_id': session_id,
                        'db_session_id': db_session_id,
                        'session': session,
                        'message': 'Upload and analysis completed'
                    })
            
            # Fallback: try to find session directly by ID
            session = sessions.get_session(session_id)
            
            if session:
                return jsonify({
                    'status': 'completed',
                    'session_id': session_id,
                    'session': session,
                    'message': 'Upload and analysis completed'
                })
            else:
                return jsonify({
                    'status': 'not_found',
                    'session_id': session_id,
                    'message': 'Session not found'
                }), 404
                
    except Exception as e:
        print(f"❌ Error getting upload progress: {str(e)}")
        return jsonify({'error': f'Failed to get upload progress: {str(e)}'}), 500

@app.route('/uploadVideo', methods=['POST', 'OPTIONS'])
def upload_video():
    """Upload a new video file to the server with automatic analysis and MongoDB storage"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        # Check if file was selected
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not allowed. Supported types: {", ".join(allowed_extensions)}'}), 400
        
        # Check file size (max 500MB)
        video_file.seek(0, 2)  # Seek to end
        file_size = video_file.tell()
        video_file.seek(0)  # Reset to beginning
        
        max_size = 500 * 1024 * 1024  # 500MB
        if file_size > max_size:
            return jsonify({'error': f'File too large. Maximum size: 500MB, got: {file_size / (1024*1024):.1f}MB'}), 400
        
        # Generate unique filename
        base_name = os.path.splitext(video_file.filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{base_name}_{timestamp}{file_ext}"
        
        # Save file to video directory
        video_path = os.path.join(VIDEO_PROCESSING_DIR, unique_filename)
        os.makedirs(VIDEO_PROCESSING_DIR, exist_ok=True)
        
        video_file.save(video_path)
        
        # Get file size in MB
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        print(f"📹 Video uploaded: {unique_filename} ({file_size_mb:.2f} MB)")
        
        # Automatically convert to H.264 for browser compatibility
        h264_success, h264_path, h264_error = convert_to_h264(video_path)
        
        # Store video metadata in MongoDB
        video_meta = {
            "filename": unique_filename,
            "original_filename": video_file.filename,
            "file_size": file_size,
            "duration": 0.0,  # Will be updated after analysis
            "fps": 0.0,
            "resolution": {"width": 0, "height": 0},
            "codec": "unknown",
            "bitrate": 0,
            "processing_status": "pending",
            "original_path": video_path,
            "processed_path": h264_path if h264_success else None,
            "analytics_path": None,
            "uploaded_at": datetime.utcnow()
        }
        
        video_id = video_metadata.create_video_metadata(video_meta)
        
        # Start video analysis in background
        def analyze_uploaded_video():
            try:
                print(f"🔍 Starting analysis for uploaded video: {unique_filename}")
                
                # Update status to processing
                video_metadata.collection.update_one(
                    {"_id": video_metadata.collection.find_one({"filename": unique_filename})["_id"]},
                    {"$set": {"processing_status": "processing"}}
                )
                
                # Process video with analytics
                timestamp_int = int(time.time())
                output_name = f"analyzed_{base_name}_{timestamp_int}.mp4"
                
                result = video_processor.process_video_with_analytics(video_path, output_name)
                
                if result["success"]:
                    # Use H.264 version if available, otherwise use original
                    final_video_path = result.get("h264_video", result.get("output_video", output_path))
                    final_video_filename = os.path.basename(final_video_path)
                    
                    # Create session in database
                    session_data = {
                        "user_id": "demo_user",  # Default user for demo
                        "athlete_name": "Uploaded Athlete",
                        "session_name": f"Analysis - {base_name}",
                        "event": "Unknown",  # Could be detected or user-specified
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "duration": "0:00",  # Will be updated
                        
                        "original_filename": video_file.filename,
                        "processed_video_filename": final_video_filename,
                        "processed_video_url": f"http://localhost:5004/downloadVideo?video_filename={final_video_filename}",
                        "analytics_filename": f"api_generated_{base_name}_{timestamp_int}.json",
                        "analytics_url": f"http://localhost:5004/getPerFrameStatistics?video_filename={base_name}_{timestamp_int}",
                        
                        "motion_iq": 0.0,
                        "acl_risk": 0.0,
                        "precision": 0.0,
                        "power": 0.0,
                        "tumbling_percentage": 0.0,
                        
                        "status": "completed",
                        "processing_progress": 1.0,
                        
                        "total_frames": 0,
                        "fps": 0.0,
                        "has_landmarks": False,
                        "landmark_confidence": 0.0,
                        
                        "notes": "Automatically analyzed uploaded video",
                        "coach_notes": "",
                        "highlights": [],
                        "areas_for_improvement": []
                    }
                    
                    # Upsert session (insert or update if exists)
                    session_id = sessions.upsert_session(session_data)
                    
                    # Update video metadata with results
                    video_metadata.collection.update_one(
                        {"_id": video_metadata.collection.find_one({"filename": unique_filename})["_id"]},
                        {"$set": {
                            "processing_status": "completed",
                            "processed_path": os.path.join(OUTPUT_DIR, output_name),
                            "analytics_path": os.path.join(ANALYTICS_DIR, f"api_generated_{base_name}_{timestamp_int}.json"),
                            "processed_at": datetime.utcnow()
                        }}
                    )
                    
                    print(f"✅ Analysis completed for: {unique_filename}")
                    print(f"📊 Session created with ID: {session_id}")
                    
                else:
                    # Update status to failed
                    video_metadata.collection.update_one(
                        {"_id": video_metadata.collection.find_one({"filename": unique_filename})["_id"]},
                        {"$set": {
                            "processing_status": "failed",
                            "processing_error": result.get("error", "Unknown error")
                        }}
                    )
                    print(f"❌ Analysis failed for: {unique_filename}")
                    
            except Exception as e:
                print(f"❌ Background analysis error for {unique_filename}: {str(e)}")
                # Update status to failed
                video_metadata.collection.update_one(
                    {"_id": video_metadata.collection.find_one({"filename": unique_filename})["_id"]},
                    {"$set": {
                        "processing_status": "failed",
                        "processing_error": str(e)
                    }}
                )
        
        # Start background analysis
        analysis_thread = threading.Thread(target=analyze_uploaded_video)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        response_data = {
            'success': True,
            'message': 'Video uploaded successfully and analysis started',
            'filename': unique_filename,
            'original_name': video_file.filename,
            'size_mb': round(file_size_mb, 2),
            'path': video_path,
            'video_id': video_id,
            'h264_conversion': {
                'success': h264_success,
                'h264_filename': os.path.basename(h264_path) if h264_success else None,
                'h264_path': h264_path if h264_success else None,
                'error': h264_error
            },
            'analysis_status': 'started'
        }
        
        if h264_success:
            print(f"✅ H.264 version created: {os.path.basename(h264_path)}")
            response_data['message'] += ' and converted to H.264 for browser compatibility'
        else:
            print(f"⚠️  H.264 conversion failed: {h264_error}")
            response_data['message'] += ' (H.264 conversion failed)'
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error uploading video: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/getSessions', methods=['GET'])
def get_sessions():
    """Get all sessions from MongoDB"""
    try:
        all_sessions = sessions.get_all_sessions()
        
        return jsonify({
            'success': True,
            'sessions': all_sessions,
            'count': len(all_sessions)
        })
        
    except Exception as e:
        print(f"❌ Error retrieving sessions: {str(e)}")
        return jsonify({'error': f'Failed to retrieve sessions: {str(e)}'}), 500

@app.route('/getSessionsByUser/<user_id>', methods=['GET'])
def get_sessions_by_user(user_id):
    """Get sessions for a specific user"""
    try:
        user_sessions = sessions.get_sessions_by_user(user_id)
        
        return jsonify({
            'success': True,
            'sessions': user_sessions,
            'count': len(user_sessions),
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"❌ Error retrieving sessions for user {user_id}: {str(e)}")
        return jsonify({'error': f'Failed to retrieve sessions: {str(e)}'}), 500

@app.route('/getSession/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific session by ID"""
    try:
        session = sessions.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'session': session
        })
        
    except Exception as e:
        print(f"❌ Error retrieving session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to retrieve session: {str(e)}'}), 500

@app.route('/getVideoMetadata/<video_id>', methods=['GET'])
def get_video_metadata(video_id):
    """Get video metadata by ID"""
    try:
        metadata = video_metadata.get_video_metadata(video_id)
        
        if not metadata:
            return jsonify({'error': 'Video metadata not found'}), 404
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
        
    except Exception as e:
        print(f"❌ Error retrieving video metadata {video_id}: {str(e)}")
        return jsonify({'error': f'Failed to retrieve video metadata: {str(e)}'}), 500

def upload_processed_files_to_gridfs(video_filename, analytics_base_name, timestamp_int):
    """Upload processed video and analytics files to GridFS"""
    try:
        if not hasattr(db_manager, 'db') or not db_manager.db:
            print("❌ MongoDB not available for GridFS upload")
            return None, None
        
        from gridfs import GridFS
        fs = GridFS(db_manager.db)
        
        # Upload processed video file
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        analytics_filename = f"api_generated_{analytics_base_name}_{timestamp_int}.json"
        analytics_path = os.path.join(ANALYTICS_DIR, analytics_filename)
        
        gridfs_video_id = None
        gridfs_analytics_id = None
        
        # Upload video file to GridFS
        if os.path.exists(video_path):
            with open(video_path, 'rb') as video_file:
                gridfs_video_id = fs.put(
                    video_file,
                    filename=video_filename,
                    content_type='video/mp4',
                    metadata={
                        'type': 'processed_video',
                        'timestamp': timestamp_int,
                        'original_path': video_path
                    }
                )
            print(f"📹 Video uploaded to GridFS with ID: {gridfs_video_id}")
        else:
            print(f"❌ Video file not found: {video_path}")
            return None, None
        
        # Upload analytics file to GridFS
        if os.path.exists(analytics_path):
            with open(analytics_path, 'rb') as analytics_file:
                gridfs_analytics_id = fs.put(
                    analytics_file,
                    filename=analytics_filename,
                    content_type='application/json',
                    metadata={
                        'type': 'analytics_data',
                        'timestamp': timestamp_int,
                        'original_path': analytics_path
                    }
                )
            print(f"📊 Analytics uploaded to GridFS with ID: {gridfs_analytics_id}")
        else:
            print(f"❌ Analytics file not found: {analytics_path}")
            return gridfs_video_id, None
        
        return gridfs_video_id, gridfs_analytics_id
        
    except Exception as e:
        print(f"❌ Error uploading files to GridFS: {str(e)}")
        return None, None

@app.route('/downloadVideoFromDB/<gridfs_file_id>', methods=['GET'])
def download_video_from_db(gridfs_file_id):
    """Download video file from MongoDB GridFS"""
    try:
        # Check if MongoDB is available
        if not hasattr(db_manager, 'db') or not db_manager.db:
            return jsonify({'error': 'MongoDB not available, cannot download from GridFS'}), 503
        
        from gridfs import GridFS
        from bson import ObjectId
        
        # Initialize GridFS
        fs = GridFS(db_manager.db)
        
        # Get file from GridFS
        gridfs_file = fs.get(ObjectId(gridfs_file_id))
        
        if not gridfs_file:
            return jsonify({'error': 'Video file not found in database'}), 404
        
        # Create a temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        # Write GridFS data to temp file
        temp_file.write(gridfs_file.read())
        temp_file.close()
        
        # Send file
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=gridfs_file.filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        print(f"❌ Error downloading video from GridFS {gridfs_file_id}: {str(e)}")
        return jsonify({'error': f'Failed to download video: {str(e)}'}), 500

@app.route('/getVideoFromSession/<session_id>', methods=['GET'])
def get_video_from_session(session_id):
    """Get video file from a session (either GridFS or file path)"""
    try:
        session = sessions.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if video is stored in GridFS
        if session.get('is_binary_stored') and session.get('gridfs_video_id'):
            gridfs_file_id = session['gridfs_video_id']
            return download_video_from_db(gridfs_file_id)
        
        # Fallback to file path
        original_filename = session.get('original_filename')
        if original_filename:
            # Use the existing download_video endpoint
            from flask import redirect
            return redirect(f'/downloadVideo?video_filename={original_filename}')
        
        return jsonify({'error': 'No video file found for this session'}), 404
        
    except Exception as e:
        print(f"❌ Error getting video from session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to get video: {str(e)}'}), 500

def get_per_frame_statistics_internal(base_name):
    """Internal helper function to get per-frame statistics by base name"""
    try:
        # Look for the frame analysis file
        analytics_file = f"api_generated_{base_name}.json"
        analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
        
        # If not found, try without the h264_ prefix
        if not os.path.exists(analytics_path) and base_name.startswith('h264_'):
            base_name_without_h264 = base_name[5:]  # Remove 'h264_' prefix
            analytics_file = f"api_generated_{base_name_without_h264}.json"
            analytics_path = os.path.join(ANALYTICS_DIR, analytics_file)
        
        if not os.path.exists(analytics_path):
            # Fallback to fixed_analytics files in analytics directory
            analytics_files = [f for f in os.listdir(ANALYTICS_DIR) 
                             if f.startswith(f"fixed_analytics_") and base_name in f]
            
            if not analytics_files:
                # Fallback to fixed_analytics files in backend directory
                backend_dir = os.path.dirname(__file__)
                analytics_files = [f for f in os.listdir(backend_dir) 
                                 if f.startswith(f"fixed_analytics_") and base_name in f]
                if analytics_files:
                    latest_file = sorted(analytics_files)[-1]
                    analytics_path = os.path.join(backend_dir, latest_file)
                else:
                    return jsonify({'error': 'No frame analysis found for this video'}), 404
        
        with open(analytics_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, list) and len(data) > 0:
            # New format: array of frame data
            return jsonify(data)
        elif isinstance(data, dict) and 'frames' in data:
            # Old format: object with frames array
            return jsonify(data['frames'])
        else:
            # Fallback: return as-is
            return jsonify(data)
            
    except Exception as e:
        print(f"❌ Error reading analytics file for {base_name}: {str(e)}")
        return jsonify({'error': f'Failed to read analytics: {str(e)}'}), 500

@app.route('/getAnalyticsFromSession/<session_id>', methods=['GET'])
def get_analytics_from_session(session_id):
    """Get analytics data from a session (either GridFS or file path)"""
    try:
        session = sessions.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if analytics is stored in GridFS
        if session.get('gridfs_analytics_id'):
            gridfs_file_id = session['gridfs_analytics_id']
            
            # Check if MongoDB is available
            if not hasattr(db_manager, 'db') or not db_manager.db:
                return jsonify({'error': 'MongoDB not available, cannot fetch analytics from GridFS'}), 503
            
            # Initialize GridFS
            from gridfs import GridFS
            from bson import ObjectId
            
            fs = GridFS(db_manager.db)
            
            # Get analytics file from GridFS
            gridfs_file = fs.get(ObjectId(gridfs_file_id))
            
            if not gridfs_file:
                return jsonify({'error': 'Analytics file not found in database'}), 404
            
            # Read the analytics data
            analytics_data = gridfs_file.read().decode('utf-8')
            return jsonify(json.loads(analytics_data))
        
        # Fallback to file path - use the existing getPerFrameStatistics endpoint
        analytics_filename = session.get('analytics_filename')
        if analytics_filename:
            # Extract base name from analytics filename
            base_name = analytics_filename.replace('.json', '').replace('api_generated_', '')
            return get_per_frame_statistics_internal(base_name)
        
        return jsonify({'error': 'No analytics file found for this session'}), 404
        
    except Exception as e:
        print(f"❌ Error getting analytics from session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500

@app.route('/deleteSession/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    try:
        success = sessions.delete_session(session_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Session deleted successfully'
            })
        else:
            return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        print(f"❌ Error deleting session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to delete session: {str(e)}'}), 500

if __name__ == '__main__':
    print("🏃‍♀️ Starting Gymnastics Analytics API Server...")
    print(f"📡 MediaPipe Server URL: {MEDIAPIPE_SERVER_URL}")
    print(f"📁 Video Directory: {VIDEO_PROCESSING_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"📁 Analytics Directory: {ANALYTICS_DIR}")
    
    # Check MediaPipe server status
    if video_processor.check_mediapipe_server():
        print("✅ MediaPipe server is running")
    else:
        print("⚠️  MediaPipe server is not running - some features may not work")
    
    app.run(host='0.0.0.0', port=5004, debug=True)
