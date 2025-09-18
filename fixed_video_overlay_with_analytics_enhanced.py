#!/usr/bin/env python3
"""
Enhanced Video Overlay with Improved Knee Valgus Calculation
This version includes enhanced ACL risk assessment with proper knee valgus calculation
and comprehensive real-time analytics overlay.
"""

import cv2
import numpy as np
import requests
import json
import time
import base64
import os
import sys
from datetime import datetime
from collections import deque
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
SERVER_URL = "https://extraordinary-gentleness-production.up.railway.app"
TUMBLING_DETECTION_ENABLED = True

class EnhancedTumblingAnalyzer:
    """Enhanced tumbling analyzer with improved knee valgus calculation"""
    
    def __init__(self, video_height=1080, video_width=1920):
        self.tumbling_sequence = False
        self.takeoff_frame = None
        self.landing_frame = None
        self.flight_frames = []
        self.com_trajectory = []
        self.elevation_angles = []
        self.ground_heights = []
        
        # Video properties for normalization
        self.video_height = video_height
        self.video_width = video_width
        
        # Buffers for noise filtering
        self.height_buffer = deque(maxlen=5)  # 5-frame moving average
        self.velocity_buffer = deque(maxlen=3)  # 3-frame velocity smoothing
        
        # Calibrated thresholds (will be adjusted based on video)
        self.takeoff_velocity_threshold = 0.02  # Normalized units
        self.flight_height_threshold = 0.05     # Normalized units
        self.landing_height_threshold = 0.02    # Normalized units
        self.min_flight_duration = 3            # Minimum frames in flight
        
        # State tracking
        self.flight_start_frame = None
        self.flight_duration = 0
        self.sequence_count = 0
        
        # Ground reference calibration
        self.ground_level = None
        self.ground_calibrated = False
        self.ground_samples = []
        
        # Enhanced ACL risk tracking
        self.max_height = 0
        
        # Pose smoothing for flickering reduction
        self.pose_buffer = deque(maxlen=5)  # 5-frame pose buffer
        self.pose_buffer_size = 5
        self.last_valid_pose = None
        
    def calibrate_ground_level(self, landmarks):
        """Calibrate ground level from ankle positions when person is standing"""
        if not landmarks or len(landmarks) < 29:
            return
            
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        if left_ankle and right_ankle:
            # Use the lower ankle as ground reference
            ground_y = min(left_ankle['y'], right_ankle['y'])
            self.ground_samples.append(ground_y)
            
            # Calibrate after 10 samples
            if len(self.ground_samples) >= 10:
                self.ground_level = np.mean(self.ground_samples)
                self.ground_calibrated = True
                print(f"✅ Ground level calibrated: {self.ground_level:.4f}")
    
    def calculate_height_from_ground(self, landmarks):
        """Calculate height from ground with proper coordinate system"""
        if not landmarks or len(landmarks) < 29:
            return 0
            
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        if left_ankle and right_ankle:
            # Use the higher ankle as reference (person might be on one foot)
            ankle_y = max(left_ankle['y'], right_ankle['y'])
            
            # If ground is calibrated, use it as reference
            if self.ground_calibrated:
                # Height = distance from ground level to ankle
                height = self.ground_level - ankle_y
            else:
                # Fallback: normalize by video height
                height = ankle_y / self.video_height
            
            # Add to buffer for smoothing
            self.height_buffer.append(height)
            
            # Return smoothed height
            if len(self.height_buffer) >= 3:
                return np.mean(list(self.height_buffer))
            else:
                return height
        
        return 0
    
    def calculate_proper_elevation_angle(self, landmarks):
        """Calculate elevation angle relative to ground plane"""
        if not landmarks or len(landmarks) < 33:
            return 0
            
        # Use shoulder to hip line to determine body orientation
        left_shoulder = landmarks[11] if len(landmarks) > 11 else None
        right_shoulder = landmarks[12] if len(landmarks) > 12 else None
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            # Calculate body center line
            shoulder_center = {
                'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                'z': (left_shoulder['z'] + right_shoulder['z']) / 2
            }
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2
            }
            
            # Calculate body vector (from hip to shoulder)
            body_vector = np.array([
                shoulder_center['x'] - hip_center['x'],
                shoulder_center['y'] - hip_center['y'],
                shoulder_center['z'] - hip_center['z']
            ])
            
            # Define ground plane normal (assuming camera is level)
            # In MediaPipe: Y-axis is vertical, so ground normal is [0, 1, 0]
            ground_normal = np.array([0, 1, 0])
            
            # Calculate angle between body vector and ground plane
            # This gives us the elevation angle from horizontal
            body_magnitude = np.linalg.norm(body_vector)
            if body_magnitude > 0.001:  # Avoid division by zero
                # Project body vector onto ground plane
                body_ground_projection = body_vector - np.dot(body_vector, ground_normal) * ground_normal
                projection_magnitude = np.linalg.norm(body_ground_projection)
                
                if projection_magnitude > 0.001:
                    # Calculate elevation angle (angle from ground plane)
                    elevation_angle = np.arccos(projection_magnitude / body_magnitude) * 180 / np.pi
                    
                    # Determine if leaning forward or backward
                    if body_vector[1] < 0:  # Negative Y means leaning forward
                        elevation_angle = -elevation_angle
                    
                    return elevation_angle
        
        return 0
    
    def calculate_forward_lean_angle(self, landmarks):
        """Calculate forward/backward lean angle relative to vertical"""
        if not landmarks or len(landmarks) < 33:
            return 0
            
        # Use shoulder to hip line
        left_shoulder = landmarks[11] if len(landmarks) > 11 else None
        right_shoulder = landmarks[12] if len(landmarks) > 12 else None
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            # Calculate body center line
            shoulder_center = {
                'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                'z': (left_shoulder['z'] + right_shoulder['z']) / 2
            }
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2
            }
            
            # Calculate body vector (from hip to shoulder)
            body_vector = np.array([
                shoulder_center['x'] - hip_center['x'],
                shoulder_center['y'] - hip_center['y'],
                shoulder_center['z'] - hip_center['z']
            ])
            
            # Define vertical reference (Y-axis in MediaPipe)
            vertical = np.array([0, 1, 0])
            
            # Calculate angle between body vector and vertical
            body_magnitude = np.linalg.norm(body_vector)
            if body_magnitude > 0.001:
                cos_angle = np.dot(body_vector, vertical) / body_magnitude
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                # Determine direction of lean
                if body_vector[2] > 0:  # Positive Z means leaning forward
                    angle = -angle
                
                return angle
        
        return 0
    
    def calculate_center_of_mass(self, landmarks):
        """Calculate center of mass from landmarks with improved method"""
        if not landmarks or len(landmarks) < 33:
            return None
            
        com_points = []
        weights = []
        
        # Head (10% of body weight)
        if len(landmarks) > 0 and landmarks[0].get('visibility', 0) > 0.5:
            com_points.append([landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']])
            weights.append(0.1)
        
        # Torso (50% of body weight) - average of shoulders and hips
        if (len(landmarks) > 12 and len(landmarks) > 24 and
            landmarks[11].get('visibility', 0) > 0.5 and
            landmarks[12].get('visibility', 0) > 0.5 and
            landmarks[23].get('visibility', 0) > 0.5 and
            landmarks[24].get('visibility', 0) > 0.5):
            
            shoulder_center = [
                (landmarks[11]['x'] + landmarks[12]['x']) / 2,
                (landmarks[11]['y'] + landmarks[12]['y']) / 2,
                (landmarks[11]['z'] + landmarks[12]['z']) / 2
            ]
            hip_center = [
                (landmarks[23]['x'] + landmarks[24]['x']) / 2,
                (landmarks[23]['y'] + landmarks[24]['y']) / 2,
                (landmarks[23]['z'] + landmarks[24]['z']) / 2
            ]
            torso_center = [
                (shoulder_center[0] + hip_center[0]) / 2,
                (shoulder_center[1] + hip_center[1]) / 2,
                (shoulder_center[2] + hip_center[2]) / 2
            ]
            com_points.append(torso_center)
            weights.append(0.5)
        
        # Arms (10% each) - average of shoulders and wrists
        if len(landmarks) > 11 and len(landmarks) > 15:
            left_arm_center = [
                (landmarks[11]['x'] + landmarks[15]['x']) / 2,
                (landmarks[11]['y'] + landmarks[15]['y']) / 2,
                (landmarks[11]['z'] + landmarks[15]['z']) / 2
            ]
            com_points.append(left_arm_center)
            weights.append(0.1)
            
        if len(landmarks) > 12 and len(landmarks) > 16:
            right_arm_center = [
                (landmarks[12]['x'] + landmarks[16]['x']) / 2,
                (landmarks[12]['y'] + landmarks[16]['y']) / 2,
                (landmarks[12]['z'] + landmarks[16]['z']) / 2
            ]
            com_points.append(right_arm_center)
            weights.append(0.1)
        
        # Legs (10% each) - average of hips and ankles
        if len(landmarks) > 23 and len(landmarks) > 27:
            left_leg_center = [
                (landmarks[23]['x'] + landmarks[27]['x']) / 2,
                (landmarks[23]['y'] + landmarks[27]['y']) / 2,
                (landmarks[23]['z'] + landmarks[27]['z']) / 2
            ]
            com_points.append(left_leg_center)
            weights.append(0.1)
            
        if len(landmarks) > 24 and len(landmarks) > 28:
            right_leg_center = [
                (landmarks[24]['x'] + landmarks[28]['x']) / 2,
                (landmarks[24]['y'] + landmarks[28]['y']) / 2,
                (landmarks[24]['z'] + landmarks[28]['z']) / 2
            ]
            com_points.append(right_leg_center)
            weights.append(0.1)
        
        # Calculate weighted average COM
        if com_points and weights:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            com_x = sum(point[0] * weight for point, weight in zip(com_points, normalized_weights))
            com_y = sum(point[1] * weight for point, weight in zip(com_points, normalized_weights))
            com_z = sum(point[2] * weight for point, weight in zip(com_points, normalized_weights))
            
            return {'x': com_x, 'y': com_y, 'z': com_z}
        
        return None
    
    def calculate_knee_angle(self, hip, knee, ankle):
        """Calculate knee angle from three points"""
        if not all([hip, knee, ankle]):
            return 0
            
        # Create vectors
        v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y'], hip['z'] - knee['z']])
        v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y'], ankle['z'] - knee['z']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def calculate_valgus_angle(self, hip, knee, ankle):
        """Calculate valgus angle (inward/outward deviation) of knee"""
        try:
            # Convert to pixel coordinates
            hip_x = hip['x'] * self.width
            hip_y = hip['y'] * self.height
            knee_x = knee['x'] * self.width
            knee_y = knee['y'] * self.height
            ankle_x = ankle['x'] * self.width
            ankle_y = ankle['y'] * self.height
            
            # Calculate vectors
            thigh_vector = np.array([knee_x - hip_x, knee_y - hip_y])
            shin_vector = np.array([ankle_x - knee_x, ankle_y - knee_y])
            
            # Normalize vectors
            thigh_norm = np.linalg.norm(thigh_vector)
            shin_norm = np.linalg.norm(shin_vector)
            
            if thigh_norm == 0 or shin_norm == 0:
                return 0.0
            
            thigh_unit = thigh_vector / thigh_norm
            shin_unit = shin_vector / shin_norm
            
            # Calculate valgus angle (deviation from straight line)
            # Positive = valgus (inward), Negative = varus (outward)
            valgus_angle = np.arcsin(np.cross(thigh_unit, shin_unit))
            
            return valgus_angle
            
        except:
            return 0.0
    
    def calculate_enhanced_acl_risk_factors(self, landmarks):
        """Calculate enhanced ACL risk factors with improved knee valgus calculation"""
        if not landmarks or len(landmarks) < 33:
            return {
                'knee_angle_risk': 0,
                'knee_valgus_risk': 0,
                'landing_mechanics_risk': 0,
                'overall_acl_risk': 0,
                'risk_level': 'LOW'
            }
        
        risk_factors = {
            'knee_angle_risk': 0,
            'knee_valgus_risk': 0,
            'landing_mechanics_risk': 0,
            'overall_acl_risk': 0,
            'risk_level': 'LOW'
        }
        
        # Get knee and hip landmarks
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        left_knee = landmarks[25] if len(landmarks) > 25 else None
        right_knee = landmarks[26] if len(landmarks) > 26 else None
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        if all([left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
            # Calculate knee angles
            left_knee_angle = self.calculate_knee_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_knee_angle(right_hip, right_knee, right_ankle)
            
            # 1. Knee Angle Risk (deep flexion increases risk)
            # Risk is highest when knee angle < 120 degrees
            left_knee_risk = 1 if left_knee_angle < 120 else 0
            right_knee_risk = 1 if right_knee_angle < 120 else 0
            risk_factors['knee_angle_risk'] = max(left_knee_risk, right_knee_risk)
            
            # 2. Enhanced Knee Valgus Risk (knees caving inward)
            left_valgus = self.calculate_valgus_angle(left_hip, left_knee, left_ankle)
            right_valgus = self.calculate_valgus_angle(right_hip, right_knee, right_ankle)
            
            # Valgus risk if deviation > 0.05 radians (≈ 2.9 degrees)
            left_valgus_risk = 1 if abs(left_valgus) > 0.05 else 0
            right_valgus_risk = 1 if abs(right_valgus) > 0.05 else 0
            risk_factors['knee_valgus_risk'] = max(left_valgus_risk, right_valgus_risk)
            
            # 3. Landing Mechanics Risk
            landing_risk = self.calculate_landing_mechanics_risk(landmarks)
            risk_factors['landing_mechanics_risk'] = landing_risk
        
        # Calculate overall ACL risk
        risk_factors['overall_acl_risk'] = (
            risk_factors['knee_angle_risk'] +
            risk_factors['knee_valgus_risk'] +
            risk_factors['landing_mechanics_risk']
        )
        
        # Determine risk level
        if risk_factors['overall_acl_risk'] == 0:
            risk_factors['risk_level'] = 'LOW'
        elif risk_factors['overall_acl_risk'] == 1:
            risk_factors['risk_level'] = 'MEDIUM'
        else:
            risk_factors['risk_level'] = 'HIGH'
        
        return risk_factors
    
    def calculate_landing_mechanics_risk(self, landmarks):
        """Calculate landing mechanics risk factors"""
        if not landmarks or len(landmarks) < 33:
            return 0
        
        landing_risk = 0
        
        # Get relevant landmarks
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        left_knee = landmarks[25] if len(landmarks) > 25 else None
        right_knee = landmarks[26] if len(landmarks) > 26 else None
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        if all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            # 1. Check for stiff landing (straight legs)
            left_knee_angle = self.calculate_knee_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_knee_angle(right_hip, right_knee, right_ankle)
            
            # Stiff landing risk (knees too straight)
            if left_knee_angle > 160 or right_knee_angle > 160:
                landing_risk += 1
            
            # 2. Check for asymmetrical landing
            knee_angle_diff = abs(left_knee_angle - right_knee_angle)
            if knee_angle_diff > 20:  # More than 20 degrees difference
                landing_risk += 1
        
        return min(1, landing_risk)
    
    def get_acl_risk_recommendations(self, risk_factors):
        """Get recommendations based on ACL risk factors"""
        recommendations = []
        
        if risk_factors['knee_angle_risk'] == 1:
            recommendations.append("Monitor knee position - avoid deep flexion")
        
        if risk_factors['knee_valgus_risk'] == 1:
            recommendations.append("Keep knees aligned over toes - prevent inward collapse")
        
        if risk_factors['landing_mechanics_risk'] == 1:
            recommendations.append("Focus on soft, controlled landing mechanics")
        
        if risk_factors['risk_level'] == 'HIGH':
            recommendations.append("HIGH ACL RISK - Consider technique modification")
        elif risk_factors['risk_level'] == 'MEDIUM':
            recommendations.append("Moderate ACL risk - monitor landing technique")
        else:
            recommendations.append("Good form! Maintain current technique")
        
        return recommendations
    
    def analyze_tumbling_sequence(self, landmarks, frame_number):
        """Analyze tumbling sequence with enhanced metrics"""
        enhanced_metrics = {
            'tumbling_detected': False,
            'com_position': None,
            'elevation_angle': 0,
            'forward_lean_angle': 0,
            'height_from_ground': 0,
            'flight_phase': 'ground',
            'tumbling_quality': 0,
            'landmark_confidence': 0,
            'acl_risk_factors': {
                'knee_angle_risk': 0,
                'knee_valgus_risk': 0,
                'landing_mechanics_risk': 0,
                'overall_acl_risk': 0,
                'risk_level': 'LOW'
            },
            'acl_recommendations': []
        }
        
        # Calibrate ground level if not done
        if not self.ground_calibrated:
            self.calibrate_ground_level(landmarks)
        
        # Validate landmarks first
        if not landmarks or len(landmarks) < 33:
            enhanced_metrics['landmark_confidence'] = 0
            return enhanced_metrics
        
        # Check visibility of key landmarks
        key_landmarks = [0, 11, 12, 23, 24, 27, 28]  # Head, shoulders, hips, ankles
        visible_count = 0
        
        for idx in key_landmarks:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                if landmark.get('visibility', 0) > 0.5:  # Visibility threshold
                    visible_count += 1
        
        # Require at least 5 out of 7 key landmarks to be visible
        if visible_count >= 5:
            enhanced_metrics['landmark_confidence'] = 1
        else:
            enhanced_metrics['landmark_confidence'] = 0
            return enhanced_metrics
        
        # Calculate COM with improved method
        com = self.calculate_center_of_mass(landmarks)
        if com:
            enhanced_metrics['com_position'] = [com['x'], com['y'], com['z']]
            self.com_trajectory.append(com)
        
        # Calculate height from ground
        height = self.calculate_height_from_ground(landmarks)
        enhanced_metrics['height_from_ground'] = height
        self.ground_heights.append(height)
        
        # Update max height
        if height > self.max_height:
            self.max_height = height
        
        # Calculate proper elevation angle (relative to ground)
        elevation_angle = self.calculate_proper_elevation_angle(landmarks)
        enhanced_metrics['elevation_angle'] = elevation_angle
        self.elevation_angles.append(elevation_angle)
        
        # Calculate forward lean angle (relative to vertical)
        forward_lean = self.calculate_forward_lean_angle(landmarks)
        enhanced_metrics['forward_lean_angle'] = forward_lean
        
        # Calculate enhanced ACL risk factors
        acl_risk_factors = self.calculate_enhanced_acl_risk_factors(landmarks)
        enhanced_metrics['acl_risk_factors'] = acl_risk_factors
        enhanced_metrics['acl_recommendations'] = self.get_acl_risk_recommendations(acl_risk_factors)
        
        # Simple tumbling detection based on height
        if height > self.flight_height_threshold:
            enhanced_metrics['flight_phase'] = 'tumbling'
            enhanced_metrics['tumbling_detected'] = True
        
        return enhanced_metrics
    
    def smooth_pose(self, pose_data):
        """Apply temporal smoothing to pose landmarks to reduce flickering"""
        if not pose_data or 'landmarks' not in pose_data:
            return self.last_valid_pose
        
        # Add current pose to buffer
        self.pose_buffer.append(pose_data)
        
        # If buffer is not full, return current pose
        if len(self.pose_buffer) < self.pose_buffer_size:
            self.last_valid_pose = pose_data
            return pose_data
        
        # Calculate weighted average of landmarks
        smoothed_landmarks = []
        for i in range(33):  # MediaPipe has 33 landmarks
            x_sum = 0
            y_sum = 0
            z_sum = 0
            visibility_sum = 0
            weight_sum = 0
            
            # Weight more recent poses higher
            for j, pose in enumerate(self.pose_buffer):
                if 'landmarks' in pose and i < len(pose['landmarks']):
                    landmark = pose['landmarks'][i]
                    weight = (j + 1) / len(self.pose_buffer)  # Linear weighting
                    
                    x_sum += landmark.get('x', 0) * weight
                    y_sum += landmark.get('y', 0) * weight
                    z_sum += landmark.get('z', 0) * weight
                    visibility_sum += landmark.get('visibility', 0) * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                smoothed_landmark = {
                    'x': x_sum / weight_sum,
                    'y': y_sum / weight_sum,
                    'z': z_sum / weight_sum,
                    'visibility': visibility_sum / weight_sum
                }
                smoothed_landmarks.append(smoothed_landmark)
            else:
                # Fallback to current landmark
                if i < len(pose_data['landmarks']):
                    smoothed_landmarks.append(pose_data['landmarks'][i])
                else:
                    smoothed_landmarks.append({'x': 0, 'y': 0, 'z': 0, 'visibility': 0})
        
        # Create smoothed pose data
        smoothed_pose = {
            'landmarks': smoothed_landmarks,
            'success': True
        }
        
        self.last_valid_pose = smoothed_pose
        return smoothed_pose
    
    def validate_and_correct_pose(self, pose_data):
        """Validate and correct pose coordinates to prevent flickering"""
        if not pose_data or 'landmarks' not in pose_data:
            return self.last_valid_pose
        
        corrected_landmarks = []
        for landmark in pose_data['landmarks']:
            # Clamp coordinates to valid range [0, 1]
            corrected_landmark = {
                'x': max(0, min(1, landmark.get('x', 0))),
                'y': max(0, min(1, landmark.get('y', 0))),
                'z': max(0, min(1, landmark.get('z', 0))),
                'visibility': max(0, min(1, landmark.get('visibility', 0)))
            }
            corrected_landmarks.append(corrected_landmark)
        
        corrected_pose = {
            'landmarks': corrected_landmarks,
            'success': True
        }
        
        return corrected_pose

class FixedVideoOverlayWithAnalytics:
    """Enhanced video overlay with improved knee valgus calculation and comprehensive analytics"""
    
    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        # Ensure output path uses the actual video filename, not temp names
        if output_path:
            self.output_path = output_path
        else:
            # Extract the original filename from the path, handling temp files
            original_filename = os.path.basename(video_path)
            # If it's a temp file, try to extract the original name
            if original_filename.startswith('temp_'):
                # Extract original name from temp filename like "temp_1234567890_originalname.mp4"
                parts = original_filename.split('_', 2)
                if len(parts) >= 3:
                    original_filename = parts[2]  # Get the original filename part
            self.output_path = f"fixed_overlayed_analytics_{original_filename}"
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize enhanced tumbling analyzer
        self.tumbling_analyzer = EnhancedTumblingAnalyzer(video_height=self.height, video_width=self.width)
        
        # Setup optimized HTTP session with connection pooling
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Video writer with H.264 codec for browser compatibility
        # Try different H.264 codecs in order of preference
        codecs_to_try = ['avc1', 'h264', 'X264']
        self.out = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
                if self.out.isOpened():
                    print(f"✅ Using video codec: {codec}")
                    break
                else:
                    self.out.release()
                    self.out = None
            except Exception as e:
                print(f"⚠️ Codec {codec} failed: {e}")
                continue
        
        # Fallback to mp4v if H.264 codecs fail
        if self.out is None or not self.out.isOpened():
            print("⚠️ H.264 codecs failed, falling back to mp4v (may not be browser-compatible)")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        # Analytics tracking
        self.frame_analytics = []
        self.current_frame = 0
        
    def draw_landmarks(self, frame, landmarks):
        """Draw MediaPipe landmarks on frame with enhanced visibility and anti-flickering"""
        if not landmarks:
            return frame
        
        # MediaPipe pose landmark names for reference
        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # Draw all landmarks with visibility-based sizing
        for i, landmark in enumerate(landmarks):
            if landmark.get('visibility', 0) > 0.3:  # Only draw visible landmarks
                x = int(landmark['x'] * self.width)
                y = int(landmark['y'] * self.height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                
                # Size based on visibility (8-12 pixels)
                radius = int(8 + (landmark.get('visibility', 0) - 0.3) * 5.7)
                radius = max(8, min(12, radius))
                
                # Color based on landmark type
                if i in [0]:  # Head
                    color = (255, 0, 0)  # Blue
                elif i in [11, 12]:  # Shoulders
                    color = (0, 255, 0)  # Green
                elif i in [23, 24]:  # Hips
                    color = (0, 0, 255)  # Red
                elif i in [25, 26]:  # Knees
                    color = (255, 255, 0)  # Cyan
                elif i in [27, 28]:  # Ankles
                    color = (255, 0, 255)  # Magenta
                else:
                    color = (255, 255, 255)  # White for other landmarks
                
                # Draw landmark with white outline for better visibility
                cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 2)  # White outline
                cv2.circle(frame, (x, y), radius, color, -1)  # Colored fill
                
                # Add landmark index for debugging (only for high visibility)
                if landmark.get('visibility', 0) > 0.8:
                    cv2.putText(frame, str(i), (x + radius + 2, y - radius - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw skeleton connections
        frame = self.draw_skeleton_connections(frame, landmarks)
        
        return frame
    
    def draw_skeleton_connections(self, frame, landmarks):
        """Draw skeleton connections between landmarks with enhanced visibility and anti-flickering"""
        if not landmarks or len(landmarks) < 33:
            return frame
            
        # MediaPipe pose skeleton connections
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
            (9, 10),  # Mouth
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (17, 21),
            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (18, 22),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), (29, 32), (30, 32),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Enhanced visibility threshold
                if (start_landmark.get('visibility', 0) > 0.3 and 
                    end_landmark.get('visibility', 0) > 0.3):
                    
                    start_x = int(start_landmark['x'] * self.width)
                    start_y = int(start_landmark['y'] * self.height)
                    end_x = int(end_landmark['x'] * self.width)
                    end_y = int(end_landmark['y'] * self.height)
                    
                    # Ensure coordinates are within frame bounds
                    start_x = max(0, min(start_x, self.width - 1))
                    start_y = max(0, min(start_y, self.height - 1))
                    end_x = max(0, min(end_x, self.width - 1))
                    end_y = max(0, min(end_y, self.height - 1))
                    
                    # Calculate average visibility for color coding
                    avg_visibility = (start_landmark.get('visibility', 0) + end_landmark.get('visibility', 0)) / 2
                    
                    # Color code based on visibility
                    if avg_visibility > 0.8:
                        color = (0, 255, 0)  # Green for high visibility
                        thickness = 5
                    elif avg_visibility > 0.6:
                        color = (0, 255, 255)  # Yellow for medium visibility
                        thickness = 4
                    else:
                        color = (0, 165, 255)  # Orange for low visibility
                        thickness = 3
                    
                    # Draw connection line with white outline for better visibility
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness + 1)  # White outline
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)  # Colored fill
        
        return frame
    
    def draw_com_trajectory(self, frame, com_trajectory):
        """Draw COM trajectory on frame"""
        if len(com_trajectory) < 2:
            return frame
            
        # Draw trajectory line
        points = []
        for com in com_trajectory[-20:]:  # Last 20 points
            x = int(com['x'] * self.width)
            y = int(com['y'] * self.height)
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            points.append((x, y))
        
        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
        
        # Draw current COM
        if com_trajectory:
            current_com = com_trajectory[-1]
            x = int(current_com['x'] * self.width)
            y = int(current_com['y'] * self.height)
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
            cv2.putText(frame, "COM", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def draw_comprehensive_analytics_overlay(self, frame, analytics_data):
        """Draw comprehensive analytics information on frame with all key metrics"""
        # Create overlay background
        overlay = frame.copy()
        
        # Enhanced analytics panel background - positioned in top-left corner
        panel_width = 280
        panel_height = 350
        panel_x = 10
        panel_y = 10
        
        # Semi-transparent background
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(overlay, "ENHANCED REAL-TIME ANALYTICS", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        y_offset = panel_y + 50
        
        # Frame info
        cv2.putText(overlay, f"Frame: {self.current_frame}/{self.total_frames}", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Ground calibration status
        if self.tumbling_analyzer.ground_calibrated:
            cv2.putText(overlay, "Ground: CALIBRATED", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(overlay, "Ground: CALIBRATING...", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_offset += 25
        
        # Flight phase
        flight_phase = analytics_data.get('flight_phase', 'ground')
        cv2.putText(overlay, f"Phase: {flight_phase.upper()}", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 25
        
        # Tumbling detection
        if analytics_data.get('tumbling_detected', False):
            cv2.putText(overlay, "TUMBLING DETECTED!", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
        
        # Height from ground
        height = analytics_data.get('height_from_ground', 0)
        height_text = f"Height: {height:.1f}cm"
        if height < 0:
            height_text += " ↓"
        cv2.putText(overlay, height_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Max height
        cv2.putText(overlay, f"Max H: {self.tumbling_analyzer.max_height:.1f}cm", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Elevation angle
        elevation = analytics_data.get('elevation_angle', 0)
        cv2.putText(overlay, f"Elevation: {elevation:.1f}°", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Forward lean angle
        lean = analytics_data.get('forward_lean_angle', 0)
        cv2.putText(overlay, f"Forward Lean: {lean:.1f}°", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
        
        # COM coordinates
        com = analytics_data.get('com_position')
        if com:
            cv2.putText(overlay, f"COM: ({com[0]:.3f}, {com[1]:.3f})", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y_offset += 20
        
        # Landmark confidence
        confidence = analytics_data.get('landmark_confidence', 0)
        confidence_color = (0, 255, 0) if confidence > 0 else (255, 0, 0)
        cv2.putText(overlay, f"Confidence: {'HIGH' if confidence > 0 else 'LOW'}", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, confidence_color, 1)
        y_offset += 25
        
        # Enhanced ACL Risk Assessment Section
        cv2.putText(overlay, "ENHANCED ACL RISK ASSESSMENT", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += 25
        
        # Overall ACL Risk
        acl_risk_factors = analytics_data.get('acl_risk_factors', {})
        risk_level = acl_risk_factors.get('risk_level', 'LOW')
        
        # Color code based on risk level
        risk_color = (0, 255, 0) if risk_level == 'LOW' else (255, 255, 0) if risk_level == 'MEDIUM' else (255, 0, 0)
        cv2.putText(overlay, f"ACL Risk Level: {risk_level}", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
        y_offset += 20
        
        # Individual risk factors
        knee_angle_risk = acl_risk_factors.get('knee_angle_risk', 0)
        knee_angle_text = "Knee Angle: RISKY" if knee_angle_risk == 1 else "Knee Angle: SAFE"
        knee_angle_color = (255, 0, 0) if knee_angle_risk == 1 else (0, 255, 0)
        cv2.putText(overlay, knee_angle_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, knee_angle_color, 1)
        y_offset += 18
        
        knee_valgus_risk = acl_risk_factors.get('knee_valgus_risk', 0)
        valgus_text = "Knee Valgus: RISKY" if knee_valgus_risk == 1 else "Knee Valgus: SAFE"
        valgus_color = (255, 0, 0) if knee_valgus_risk == 1 else (0, 255, 0)
        cv2.putText(overlay, valgus_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, valgus_color, 1)
        y_offset += 18
        
        landing_mechanics_risk = acl_risk_factors.get('landing_mechanics_risk', 0)
        landing_text = "Landing: RISKY" if landing_mechanics_risk == 1 else "Landing: SAFE"
        landing_color = (255, 0, 0) if landing_mechanics_risk == 1 else (0, 255, 0)
        cv2.putText(overlay, landing_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, landing_color, 1)
        y_offset += 25
        
        # ACL Recommendations
        acl_recommendations = analytics_data.get('acl_recommendations', [])
        if acl_recommendations:
            cv2.putText(overlay, "Recommendations:", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 18
            
            for i, rec in enumerate(acl_recommendations[:2]):  # Show first 2 recommendations
                if y_offset < panel_y + panel_height - 20:  # Don't overflow panel
                    cv2.putText(overlay, f"• {rec[:35]}...", (panel_x + 15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 16
        
        # Blend overlay with original frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame with enhanced analytics and pose smoothing"""
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            # Send frame to server with retry mechanism
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        f"{SERVER_URL}/detect-pose",
                        json={'image': frame_base64},
                        timeout=5,  # Increased timeout for reliability
                        headers={'Connection': 'keep-alive'}  # Reuse connections
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.1)  # Brief delay before retry
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    # Get raw pose data
                    raw_pose_data = result
                    
                    # Validate and correct pose coordinates
                    validated_pose = self.tumbling_analyzer.validate_and_correct_pose(raw_pose_data)
                    
                    # Apply temporal smoothing to reduce flickering
                    smoothed_pose = self.tumbling_analyzer.smooth_pose(validated_pose)
                    
                    if smoothed_pose and 'landmarks' in smoothed_pose:
                        landmarks = smoothed_pose['landmarks']
                        metrics = result.get('metrics', {})
                        
                        # Add enhanced tumbling analysis
                        if TUMBLING_DETECTION_ENABLED:
                            tumbling_metrics = self.tumbling_analyzer.analyze_tumbling_sequence(landmarks, self.current_frame)
                            metrics['tumbling_metrics'] = tumbling_metrics
                        
                        # Store analytics
                        frame_analytics = {
                            'frame_number': self.current_frame,
                            'landmarks': landmarks,
                            'metrics': metrics
                        }
                        self.frame_analytics.append(frame_analytics)
                        
                        # Draw landmarks with smoothed data
                        frame = self.draw_landmarks(frame, landmarks)
                        
                        # Draw COM trajectory
                        if self.tumbling_analyzer.com_trajectory:
                            frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                        
                        # Draw comprehensive analytics overlay
                        frame = self.draw_comprehensive_analytics_overlay(frame, tumbling_metrics)
                        
                        # Store last valid tumbling metrics for continuity
                        self.last_tumbling_metrics = tumbling_metrics
                    else:
                        # Use last valid pose if available to maintain overlay continuity
                        if self.tumbling_analyzer.last_valid_pose:
                            landmarks = self.tumbling_analyzer.last_valid_pose['landmarks']
                            frame = self.draw_landmarks(frame, landmarks)
                            
                            # Draw COM trajectory if available
                            if self.tumbling_analyzer.com_trajectory:
                                frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                            
                            # Draw analytics overlay with last valid data
                            if hasattr(self, 'last_tumbling_metrics'):
                                frame = self.draw_comprehensive_analytics_overlay(frame, self.last_tumbling_metrics)
                        
                        # Only show "NO POSE DETECTED" if we have no valid pose at all
                        if not self.tumbling_analyzer.last_valid_pose:
                            cv2.putText(frame, "NO POSE DETECTED", (self.width//2 - 100, self.height//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Use last valid pose if available to maintain overlay continuity
                    if self.tumbling_analyzer.last_valid_pose:
                        landmarks = self.tumbling_analyzer.last_valid_pose['landmarks']
                        frame = self.draw_landmarks(frame, landmarks)
                        
                        # Draw COM trajectory if available
                        if self.tumbling_analyzer.com_trajectory:
                            frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                        
                        # Draw analytics overlay with last valid data
                        if hasattr(self, 'last_tumbling_metrics'):
                            frame = self.draw_comprehensive_analytics_overlay(frame, self.last_tumbling_metrics)
                    
                    # Only show "NO POSE DETECTED" if we have no valid pose at all
                    if not self.tumbling_analyzer.last_valid_pose:
                        cv2.putText(frame, "NO POSE DETECTED", (self.width//2 - 100, self.height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            else:
                # Server error - use last valid pose if available
                if self.tumbling_analyzer.last_valid_pose:
                    landmarks = self.tumbling_analyzer.last_valid_pose['landmarks']
                    frame = self.draw_landmarks(frame, landmarks)
                    
                    # Draw COM trajectory if available
                    if self.tumbling_analyzer.com_trajectory:
                        frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                    
                    # Draw analytics overlay with last valid data
                    if hasattr(self, 'last_tumbling_metrics'):
                        frame = self.draw_comprehensive_analytics_overlay(frame, self.last_tumbling_metrics)
                else:
                    cv2.putText(frame, "SERVER ERROR", (self.width//2 - 80, self.height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        except Exception as e:
            # Connection error - use last valid pose if available
            if self.tumbling_analyzer.last_valid_pose:
                landmarks = self.tumbling_analyzer.last_valid_pose['landmarks']
                frame = self.draw_landmarks(frame, landmarks)
                
                # Draw COM trajectory if available
                if self.tumbling_analyzer.com_trajectory:
                    frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                
                # Draw analytics overlay with last valid data
                if hasattr(self, 'last_tumbling_metrics'):
                    frame = self.draw_comprehensive_analytics_overlay(frame, self.last_tumbling_metrics)
            else:
                cv2.putText(frame, f"ERROR: {str(e)[:20]}", (self.width//2 - 80, self.height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def process_video(self):
        """Process entire video with enhanced analytics overlay"""
        print(f"🎬 Processing video with ENHANCED knee valgus calculation: {os.path.basename(self.video_path)}")
        print(f"📊 Output: {self.output_path}")
        print(f"📐 Resolution: {self.width}x{self.height}")
        print(f"🎯 FPS: {self.fps}")
        print(f"📈 Total frames: {self.total_frames}")
        print("="*60)
        
        frame_count = 0
        print(f"🚀 Processing EVERY frame for consistent overlays (no frame skipping)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process EVERY frame to avoid missing overlays
            processed_frame = self.process_frame(frame)
            
            # Write frame
            self.out.write(processed_frame)
            
            frame_count += 1
            self.current_frame = frame_count
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / self.total_frames) * 100
                print(f"   Processed {frame_count}/{self.total_frames} frames ({progress:.1f}%)")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        
        print(f"✅ Enhanced video processing completed!")
        print(f"📊 Analytics data collected for {len(self.frame_analytics)} frames")
        
        # Save analytics data with correct filename
        original_filename = os.path.basename(self.video_path)
        # Handle temp files to extract original name
        if original_filename.startswith('temp_'):
            parts = original_filename.split('_', 2)
            if len(parts) >= 3:
                original_filename = parts[2]  # Get the original filename part
        
        analytics_filename = f"fixed_analytics_{original_filename}.json"
        with open(analytics_filename, 'w') as f:
            json.dump(self.frame_analytics, f, indent=2)
        
        print(f"💾 Analytics saved: {analytics_filename}")
        
        # Post-process video to ensure browser compatibility
        self.ensure_browser_compatibility()
        
        return self.frame_analytics
    
    def ensure_browser_compatibility(self):
        """Ensure the video is encoded in a browser-compatible format"""
        try:
            import subprocess
            
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ ffmpeg not available, skipping video conversion")
                return
            
            # Create a temporary file for the converted video
            temp_output = self.output_path.replace('.mp4', '_browser_compatible.mp4')
            
            # Use ffmpeg to convert to proper H.264
            cmd = [
                'ffmpeg', '-i', self.output_path,
                '-c:v', 'libx264',  # H.264 video codec
                '-c:a', 'aac',      # AAC audio codec
                '-preset', 'fast',  # Fast encoding
                '-crf', '23',       # Good quality
                '-movflags', '+faststart',  # Optimize for streaming
                '-pix_fmt', 'yuv420p',  # Ensure compatible pixel format
                '-y',               # Overwrite output file
                temp_output
            ]
            
            print(f"🔄 Converting video to browser-compatible format...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace the original file with the converted one
                import shutil
                shutil.move(temp_output, self.output_path)
                print(f"✅ Video converted to browser-compatible H.264 format")
            else:
                print(f"⚠️ Video conversion failed: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
        except Exception as e:
            print(f"⚠️ Video compatibility conversion failed: {e}")
            # Clean up temp file if it exists
            if 'temp_output' in locals() and os.path.exists(temp_output):
                os.remove(temp_output)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create video overlay with ENHANCED knee valgus calculation')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--server', '-s', default='https://extraordinary-gentleness-production.up.railway.app', help='MediaPipe server URL')
    
    args = parser.parse_args()
    
    # Update server URL
    global SERVER_URL
    SERVER_URL = args.server
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    # Check server connection
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding: HTTP {response.status_code}")
            return
        print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Process video
    overlay = FixedVideoOverlayWithAnalytics(args.video_path, args.output)
    analytics_data = overlay.process_video()
    
    print(f"\n🎉 Enhanced video overlay with analytics completed!")
    print(f"📹 Output video: {overlay.output_path}")
    
    # Use the same logic for analytics filename
    original_filename = os.path.basename(args.video_path)
    if original_filename.startswith('temp_'):
        parts = original_filename.split('_', 2)
        if len(parts) >= 3:
            original_filename = parts[2]
    
    print(f"📊 Analytics file: fixed_analytics_{original_filename}.json")

if __name__ == "__main__":
    main()
