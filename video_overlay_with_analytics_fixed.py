#!/usr/bin/env python3
"""
Fixed Video Overlay with Proper Elevation Calculation
This version fixes the elevation calculation by using ground reference and proper coordinate system.
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

# Configuration
SERVER_URL = "http://localhost:5001"
TUMBLING_DETECTION_ENABLED = True

class FixedTumblingAnalyzer:
    """Fixed tumbling analyzer with proper elevation calculation"""
    
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
    
    def detect_takeoff(self, landmarks, frame_number):
        """Detect takeoff with improved velocity calculation"""
        if not landmarks or len(landmarks) < 33:
            return False
            
        # Calculate current height
        current_height = self.calculate_height_from_ground(landmarks)
        
        # Calculate velocity if we have previous data
        if hasattr(self, 'previous_height') and self.previous_height is not None:
            # Calculate velocity (change in height per frame)
            velocity = current_height - self.previous_height
            
            # Add to velocity buffer for smoothing
            self.velocity_buffer.append(velocity)
            
            # Use smoothed velocity for detection
            if len(self.velocity_buffer) >= 2:
                smoothed_velocity = np.mean(list(self.velocity_buffer))
                
                # Dynamic threshold based on video properties
                threshold = self.takeoff_velocity_threshold
                
                # Check for takeoff (positive velocity above threshold)
                if smoothed_velocity > threshold:
                    # Additional validation: check if we're coming from ground
                    if self.previous_height < self.landing_height_threshold:
                        return True
        
        self.previous_height = current_height
        return False
    
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
    
    def validate_landmarks(self, landmarks):
        """Validate that landmarks are visible and reasonable"""
        if not landmarks or len(landmarks) < 33:
            return False
            
        # Check visibility of key landmarks
        key_landmarks = [0, 11, 12, 23, 24, 27, 28]  # Head, shoulders, hips, ankles
        visible_count = 0
        
        for idx in key_landmarks:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                if landmark.get('visibility', 0) > 0.5:  # Visibility threshold
                    visible_count += 1
        
        # Require at least 5 out of 7 key landmarks to be visible
        return visible_count >= 5
    
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
    
    def detect_tumbling_start(self, landmarks, frame_number):
        """Improved tumbling start detection with validation"""
        if not landmarks or len(landmarks) < 33:
            return False
            
        # Check for crouch position with improved angle calculation
        left_knee = landmarks[25] if len(landmarks) > 25 else None
        right_knee = landmarks[26] if len(landmarks) > 26 else None
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        if all([left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle]):
            # Calculate knee angles
            knee_angle_left = self.calculate_knee_angle(left_hip, left_knee, left_ankle)
            knee_angle_right = self.calculate_knee_angle(right_hip, right_knee, right_ankle)
            
            # More realistic tumbling preparation angles (60-150 degrees)
            if (knee_angle_left < 150 and knee_angle_right < 150 and 
                knee_angle_left > 60 and knee_angle_right > 60):
                
                # Additional validation: check if person is on ground
                height = self.calculate_height_from_ground(landmarks)
                if height < self.landing_height_threshold:
                    return True
                    
        return False
    
    def analyze_tumbling_sequence(self, landmarks, frame_number):
        """Analyze tumbling sequence with improved logic"""
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
        if not self.validate_landmarks(landmarks):
            enhanced_metrics['landmark_confidence'] = 0
            return enhanced_metrics
        
        enhanced_metrics['landmark_confidence'] = 1
        
        # Calculate COM with improved method
        com = self.calculate_center_of_mass(landmarks)
        if com:
            enhanced_metrics['com_position'] = com
            self.com_trajectory.append(com)
        
        # Calculate height from ground
        height = self.calculate_height_from_ground(landmarks)
        enhanced_metrics['height_from_ground'] = height
        self.ground_heights.append(height)
        
        # Calculate proper elevation angle (relative to ground)
        elevation_angle = self.calculate_proper_elevation_angle(landmarks)
        enhanced_metrics['elevation_angle'] = elevation_angle
        self.elevation_angles.append(elevation_angle)
        
        # Calculate forward lean angle (relative to vertical)
        forward_lean = self.calculate_forward_lean_angle(landmarks)
        enhanced_metrics['forward_lean_angle'] = forward_lean
        
        # Calculate ACL risk factors
        acl_risk_factors = self.calculate_acl_risk_factors(landmarks)
        enhanced_metrics['acl_risk_factors'] = acl_risk_factors
        enhanced_metrics['acl_recommendations'] = self.get_acl_risk_recommendations(acl_risk_factors)
        
        # Improved flight phase detection
        if self.detect_tumbling_start(landmarks, frame_number):
            self.tumbling_sequence = True
            enhanced_metrics['flight_phase'] = 'preparation'
            
        elif self.detect_takeoff(landmarks, frame_number):
            self.takeoff_frame = frame_number
            enhanced_metrics['flight_phase'] = 'takeoff'
            enhanced_metrics['tumbling_detected'] = True
            
        elif self.tumbling_sequence and height > self.flight_height_threshold:
            # Track flight duration
            if self.flight_start_frame is None:
                self.flight_start_frame = frame_number
            
            self.flight_duration += 1
            enhanced_metrics['flight_phase'] = 'flight'
            enhanced_metrics['tumbling_detected'] = True
            
            # Only calculate quality after minimum flight duration
            if self.flight_duration >= self.min_flight_duration:
                trajectory_smoothness = self.calculate_trajectory_smoothness()
                elevation_consistency = self.calculate_elevation_consistency()
                enhanced_metrics['tumbling_quality'] = (trajectory_smoothness + elevation_consistency) / 2
                
        elif (self.tumbling_sequence and 
              height < self.landing_height_threshold and 
              self.flight_duration >= self.min_flight_duration):
            # Landing phase
            enhanced_metrics['flight_phase'] = 'landing'
            enhanced_metrics['tumbling_detected'] = True
            self.landing_frame = frame_number
            
            # Reset for next sequence
            self.tumbling_sequence = False
            self.flight_start_frame = None
            self.flight_duration = 0
            self.sequence_count += 1
        
        return enhanced_metrics
    
    def calculate_trajectory_smoothness(self):
        """Calculate trajectory smoothness with outlier detection"""
        if len(self.com_trajectory) < 5:
            return 0
            
        # Calculate velocities
        velocities = []
        for i in range(1, len(self.com_trajectory)):
            prev = self.com_trajectory[i-1]
            curr = self.com_trajectory[i]
            velocity = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2 + (curr['z'] - prev['z'])**2)
            velocities.append(velocity)
        
        # Remove outliers (velocity spikes)
        if velocities:
            velocities = np.array(velocities)
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            
            # Remove velocities more than 2 standard deviations from mean
            filtered_velocities = velocities[np.abs(velocities - mean_vel) <= 2 * std_vel]
            
            if len(filtered_velocities) > 0:
                variance = np.var(filtered_velocities)
                smoothness = max(0, 100 - variance * 1000)
                return smoothness
        
        return 0
    
    def calculate_elevation_consistency(self):
        """Calculate elevation consistency with outlier detection"""
        if len(self.elevation_angles) < 5:
            return 0
            
        angles = np.array(self.elevation_angles)
        
        # Remove outliers
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        filtered_angles = angles[np.abs(angles - mean_angle) <= 2 * std_angle]
        
        if len(filtered_angles) > 0:
            variance = np.var(filtered_angles)
            consistency = max(0, 100 - variance * 0.1)
            return consistency
        
        return 0
    
    def calculate_acl_risk_factors(self, landmarks):
        """Calculate ACL risk factors from landmarks"""
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
            # Risk is highest when knee angle < 90 degrees (deep flexion)
            left_knee_risk = max(0, (90 - left_knee_angle) / 90 * 100) if left_knee_angle < 90 else 0
            right_knee_risk = max(0, (90 - right_knee_angle) / 90 * 100) if right_knee_angle < 90 else 0
            risk_factors['knee_angle_risk'] = max(left_knee_risk, right_knee_risk)
            
            # 2. Knee Valgus Risk (knees caving inward)
            left_valgus = self.calculate_knee_valgus(left_hip, left_knee, left_ankle)
            right_valgus = self.calculate_knee_valgus(right_hip, right_knee, right_ankle)
            risk_factors['knee_valgus_risk'] = max(left_valgus, right_valgus)
            
            # 3. Landing Mechanics Risk
            landing_risk = self.calculate_landing_mechanics_risk(landmarks)
            risk_factors['landing_mechanics_risk'] = landing_risk
        
        # Calculate overall ACL risk (weighted average)
        risk_factors['overall_acl_risk'] = (
            risk_factors['knee_angle_risk'] * 0.3 +
            risk_factors['knee_valgus_risk'] * 0.4 +
            risk_factors['landing_mechanics_risk'] * 0.3
        )
        
        # Determine risk level
        if risk_factors['overall_acl_risk'] < 30:
            risk_factors['risk_level'] = 'LOW'
        elif risk_factors['overall_acl_risk'] < 60:
            risk_factors['risk_level'] = 'MODERATE'
        else:
            risk_factors['risk_level'] = 'HIGH'
        
        return risk_factors
    
    def calculate_knee_valgus(self, hip, knee, ankle):
        """Calculate knee valgus (inward collapse) risk"""
        if not all([hip, knee, ankle]):
            return 0
        
        # Calculate vectors
        hip_to_knee = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
        knee_to_ankle = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
        
        # Calculate angle between these vectors
        dot_product = np.dot(hip_to_knee, knee_to_ankle)
        hip_to_knee_norm = np.linalg.norm(hip_to_knee)
        knee_to_ankle_norm = np.linalg.norm(knee_to_ankle)
        
        if hip_to_knee_norm > 0 and knee_to_ankle_norm > 0:
            cos_angle = dot_product / (hip_to_knee_norm * knee_to_ankle_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            # Valgus risk increases as angle deviates from 180 degrees
            # (knees pointing inward)
            valgus_deviation = abs(180 - angle)
            valgus_risk = min(100, valgus_deviation * 2)  # Scale to 0-100
            
            return valgus_risk
        
        return 0
    
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
                landing_risk += 40
            
            # 2. Check for asymmetrical landing
            knee_angle_diff = abs(left_knee_angle - right_knee_angle)
            if knee_angle_diff > 20:  # More than 20 degrees difference
                landing_risk += 30
            
            # 3. Check for excessive forward lean (poor posture)
            forward_lean = self.calculate_forward_lean_angle(landmarks)
            if abs(forward_lean) > 45:  # More than 45 degrees forward lean
                landing_risk += 20
            
            # 4. Check for high landing impact (sudden height change)
            if hasattr(self, 'previous_height') and self.previous_height is not None:
                current_height = self.calculate_height_from_ground(landmarks)
                height_change = abs(current_height - self.previous_height)
                if height_change > 0.05:  # Sudden height change
                    landing_risk += 10
        
        return min(100, landing_risk)
    
    def get_acl_risk_recommendations(self, risk_factors):
        """Get recommendations based on ACL risk factors"""
        recommendations = []
        
        if risk_factors['knee_angle_risk'] > 50:
            recommendations.append("Deep knee flexion detected - consider landing with more bend")
        
        if risk_factors['knee_valgus_risk'] > 50:
            recommendations.append("Knee valgus detected - keep knees aligned over toes")
        
        if risk_factors['landing_mechanics_risk'] > 50:
            recommendations.append("Poor landing mechanics - focus on soft, controlled landing")
        
        if risk_factors['overall_acl_risk'] > 60:
            recommendations.append("HIGH ACL RISK - Consider technique modification")
        elif risk_factors['overall_acl_risk'] > 30:
            recommendations.append("Moderate ACL risk - monitor landing technique")
        else:
            recommendations.append("Good landing mechanics - maintain current form")
        
        return recommendations

class FixedVideoOverlayWithAnalytics:
    """Fixed video overlay with proper elevation calculation"""
    
    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        self.output_path = output_path or f"fixed_overlayed_analytics_{os.path.basename(video_path)}"
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize fixed tumbling analyzer
        self.tumbling_analyzer = FixedTumblingAnalyzer(video_height=self.height, video_width=self.width)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        # Analytics tracking
        self.frame_analytics = []
        self.current_frame = 0
        
    def draw_landmarks(self, frame, landmarks):
        """Draw MediaPipe landmarks on frame"""
        if not landmarks:
            return frame
            
        # Draw key landmarks
        key_points = {
            0: (255, 0, 0),    # Head - Blue
            11: (0, 255, 0),   # Left shoulder - Green
            12: (0, 255, 0),   # Right shoulder - Green
            23: (0, 0, 255),   # Left hip - Red
            24: (0, 0, 255),   # Right hip - Red
            25: (255, 255, 0), # Left knee - Cyan
            26: (255, 255, 0), # Right knee - Cyan
            27: (255, 0, 255), # Left ankle - Magenta
            28: (255, 0, 255), # Right ankle - Magenta
        }
        
        for idx, color in key_points.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                # Lower visibility threshold to show more landmarks
                if landmark.get('visibility', 0) > 0.1:
                    x = int(landmark['x'] * self.width)
                    y = int(landmark['y'] * self.height)
                    cv2.circle(frame, (x, y), 6, color, -1)  # Slightly larger circles
                    cv2.putText(frame, str(idx), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw skeleton connections
        frame = self.draw_skeleton_connections(frame, landmarks)
        
        return frame
    
    def draw_skeleton_connections(self, frame, landmarks):
        """Draw skeleton connections between landmarks"""
        if not landmarks or len(landmarks) < 33:
            return frame
            
        # Define skeleton connections (MediaPipe pose connections)
        connections = [
            # Head and shoulders
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # Shoulders
            (9, 10),
            # Left arm
            (11, 12), (12, 14), (14, 16),
            # Right arm  
            (11, 13), (13, 15), (15, 17),
            # Torso
            (11, 23), (12, 24), (23, 24),
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32),
            # Feet
            (27, 31), (28, 32)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Only draw if both landmarks have reasonable visibility
                if (start_landmark.get('visibility', 0) > 0.1 and 
                    end_landmark.get('visibility', 0) > 0.1):
                    
                    start_x = int(start_landmark['x'] * self.width)
                    start_y = int(start_landmark['y'] * self.height)
                    end_x = int(end_landmark['x'] * self.width)
                    end_y = int(end_landmark['y'] * self.height)
                    
                    # Draw connection line
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        
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
            points.append((x, y))
        
        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
        
        # Draw current COM
        if com_trajectory:
            current_com = com_trajectory[-1]
            x = int(current_com['x'] * self.width)
            y = int(current_com['y'] * self.height)
            cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
            cv2.putText(frame, "COM", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def draw_analytics_overlay(self, frame, analytics_data):
        """Draw analytics information on frame"""
        # Create overlay background
        overlay = frame.copy()
        
        # Analytics panel background - white background for black text
        panel_width = 500
        panel_height = 450
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (255, 255, 255), -1)
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), 2)
        
        # Title
        cv2.putText(overlay, "FIXED REAL-TIME ANALYTICS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        y_offset = 60
        
        # Frame info
        cv2.putText(overlay, f"Frame: {self.current_frame}/{self.total_frames}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        # Ground calibration status
        if self.tumbling_analyzer.ground_calibrated:
            cv2.putText(overlay, "Ground: CALIBRATED", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(overlay, "Ground: CALIBRATING...", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        # Flight phase
        flight_phase = analytics_data.get('flight_phase', 'ground')
        cv2.putText(overlay, f"Phase: {flight_phase.upper()}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30
        
        # Tumbling detection
        if analytics_data.get('tumbling_detected', False):
            cv2.putText(overlay, "TUMBLING DETECTED!", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 30
        
        # Height from ground
        height = analytics_data.get('height_from_ground', 0)
        cv2.putText(overlay, f"Height: {height:.3f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        # Fixed elevation angle (relative to ground)
        elevation = analytics_data.get('elevation_angle', 0)
        cv2.putText(overlay, f"Elevation: {elevation:.1f}°", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        # Forward lean angle (relative to vertical)
        lean = analytics_data.get('forward_lean_angle', 0)
        cv2.putText(overlay, f"Forward Lean: {lean:.1f}°", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        # COM coordinates
        com = analytics_data.get('com_position')
        if com:
            cv2.putText(overlay, f"COM: ({com['x']:.3f}, {com['y']:.3f}, {com['z']:.3f})", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 25
        
        # Tumbling quality
        quality = analytics_data.get('tumbling_quality', 0)
        if quality > 0:
            quality_color = (0, 255, 0) if quality > 70 else (255, 255, 0) if quality > 40 else (255, 0, 0)
            cv2.putText(overlay, f"Quality: {quality:.1f}/100", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
            y_offset += 25
        
        # Landmark confidence
        confidence = analytics_data.get('landmark_confidence', 0)
        confidence_color = (0, 255, 0) if confidence > 0 else (255, 0, 0)
        cv2.putText(overlay, f"Confidence: {'HIGH' if confidence > 0 else 'LOW'}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
        y_offset += 30
        
        # ACL Risk Assessment Section
        cv2.putText(overlay, "ACL RISK ASSESSMENT", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        # Overall ACL Risk
        acl_risk_factors = analytics_data.get('acl_risk_factors', {})
        overall_risk = acl_risk_factors.get('overall_acl_risk', 0)
        risk_level = acl_risk_factors.get('risk_level', 'LOW')
        
        # Color code based on risk level
        risk_color = (0, 255, 0) if risk_level == 'LOW' else (255, 255, 0) if risk_level == 'MODERATE' else (255, 0, 0)
        cv2.putText(overlay, f"ACL Risk: {overall_risk:.1f}/100 ({risk_level})", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
        y_offset += 25
        
        # Individual risk factors
        knee_angle_risk = acl_risk_factors.get('knee_angle_risk', 0)
        cv2.putText(overlay, f"Knee Angle Risk: {knee_angle_risk:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        knee_valgus_risk = acl_risk_factors.get('knee_valgus_risk', 0)
        cv2.putText(overlay, f"Knee Valgus Risk: {knee_valgus_risk:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        landing_mechanics_risk = acl_risk_factors.get('landing_mechanics_risk', 0)
        cv2.putText(overlay, f"Landing Mechanics Risk: {landing_mechanics_risk:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
        
        # ACL Recommendations
        acl_recommendations = analytics_data.get('acl_recommendations', [])
        if acl_recommendations:
            cv2.putText(overlay, "Recommendations:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 20
            
            for i, rec in enumerate(acl_recommendations[:2]):  # Show first 2 recommendations
                if y_offset < panel_height - 30:  # Don't overflow panel
                    cv2.putText(overlay, f"• {rec[:40]}...", (30, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 18
        
        # Blend overlay with original frame
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame with analytics"""
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            # Send frame to server
            response = requests.post(
                f"{SERVER_URL}/detect-pose",
                json={'image': frame_base64},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    landmarks = result.get('landmarks', [])
                    metrics = result.get('metrics', {})
                    
                    # Add tumbling analysis
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
                    
                    # Draw landmarks
                    frame = self.draw_landmarks(frame, landmarks)
                    
                    # Draw COM trajectory
                    if self.tumbling_analyzer.com_trajectory:
                        frame = self.draw_com_trajectory(frame, self.tumbling_analyzer.com_trajectory)
                    
                    # Draw analytics overlay
                    frame = self.draw_analytics_overlay(frame, tumbling_metrics)
                    
                else:
                    # No pose detected
                    cv2.putText(frame, "NO POSE DETECTED", (self.width//2 - 100, self.height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            else:
                # Server error
                cv2.putText(frame, "SERVER ERROR", (self.width//2 - 80, self.height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        except Exception as e:
            # Connection error
            cv2.putText(frame, f"ERROR: {str(e)[:20]}", (self.width//2 - 80, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def process_video(self):
        """Process entire video with analytics overlay"""
        print(f"🎬 Processing video with FIXED elevation calculation: {os.path.basename(self.video_path)}")
        print(f"📊 Output: {self.output_path}")
        print(f"📐 Resolution: {self.width}x{self.height}")
        print(f"🎯 FPS: {self.fps}")
        print(f"📈 Total frames: {self.total_frames}")
        print("="*60)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
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
        
        print(f"✅ Video processing completed!")
        print(f"📊 Analytics data collected for {len(self.frame_analytics)} frames")
        
        # Save analytics data
        analytics_filename = f"fixed_analytics_{os.path.basename(self.video_path)}.json"
        with open(analytics_filename, 'w') as f:
            json.dump(self.frame_analytics, f, indent=2)
        
        print(f"💾 Analytics saved: {analytics_filename}")
        
        return self.frame_analytics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create video overlay with FIXED elevation calculation')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--server', '-s', default='http://localhost:5001', help='MediaPipe server URL')
    
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
    
    print(f"\n🎉 Fixed video overlay with analytics completed!")
    print(f"📹 Output video: {overlay.output_path}")
    print(f"📊 Analytics file: fixed_analytics_{os.path.basename(args.video_path)}.json")

if __name__ == "__main__":
    main()
