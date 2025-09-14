#!/usr/bin/env python3
"""
Enhanced MediaPipe Pose Detection Server with Coordinate Extraction
Combines pose landmarks with bounding box coordinates for comprehensive analysis
"""

import cv2
import numpy as np
import base64
import json
import time
import logging
from flask import Flask, request, jsonify
from collections import deque
import mediapipe as mp
from scipy.signal import savgol_filter
import math
import os
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=True,  # Changed to True to avoid timestamp issues
    model_complexity=1,  # Reduced complexity to avoid SSL issues
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Frame history for temporal analysis
MAX_HISTORY = 60
frame_history = deque(maxlen=MAX_HISTORY)
analytics_history = deque(maxlen=100)

# Analytics update interval (seconds)
ANALYTICS_INTERVAL = 5
last_analytics_update = 0

# Timestamp management to avoid MediaPipe conflicts
last_timestamp = 0

# JSON file for analytics - will be set dynamically by client
ANALYTICS_JSON_FILE = None

# Coordinate API configuration
COORDINATE_API_URL = "http://localhost:8082"
COORDINATE_API_ENABLED = False  # Fallback to landmark-based coordinates only

# Landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def image_to_base64(image):
    """Convert image to base64 string"""
    if image is None:
        return None
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_string):
    """Convert base64 string to image"""
    if not base64_string:
        return None
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle other numpy scalar types
        return obj.item()
    else:
        return obj

def check_coordinate_api_health():
    """Check if coordinate API is available"""
    try:
        response = requests.get(f"{COORDINATE_API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_coordinates_from_image(image, frame_id):
    """Get bounding box coordinates for an image using the coordinate API"""
    if not COORDINATE_API_ENABLED:
        return None
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image)
        
        # Send to coordinate API
        response = requests.post(
            f"{COORDINATE_API_URL}/process_image",
            json={
                'image': image_base64,
                'frame_id': frame_id
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('detections', [])
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting coordinates: {e}")
        return None

def calculate_bounding_box_from_landmarks(landmarks):
    """Calculate bounding box from pose landmarks as fallback"""
    if not landmarks or len(landmarks) < 33:
        return None
    
    # Get all visible landmarks
    visible_landmarks = [lm for lm in landmarks if lm and lm.get('visibility', 0) > 0.5]
    
    if not visible_landmarks:
        return None
    
    # Calculate bounding box
    x_coords = [lm['x'] for lm in visible_landmarks]
    y_coords = [lm['y'] for lm in visible_landmarks]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 0.1
    width = x_max - x_min
    height = y_max - y_min
    
    bbox = {
        'x1': max(0, x_min - width * padding),
        'y1': max(0, y_min - height * padding),
        'x2': min(1, x_max + width * padding),
        'y2': min(1, y_min + height * padding)
    }
    
    center = {
        'x': (bbox['x1'] + bbox['x2']) / 2,
        'y': (bbox['y1'] + bbox['y2']) / 2
    }
    
    return {
        'person_id': 0,
        'bbox': bbox,
        'center': center,
        'confidence': 0.8,  # Estimated confidence
        'source': 'landmarks'
    }

def calculate_angle_3d(p1, p2, p3):
    """Calculate 3D angle between three points"""
    if not all([p1, p2, p3]):
        return 0
    
    # Convert to numpy arrays
    a = np.array([p1['x'], p1['y'], p1.get('z', 0)])
    b = np.array([p2['x'], p2['y'], p2.get('z', 0)])
    c = np.array([p3['x'], p3['y'], p3.get('z', 0)])
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_angle(p1, p2, p3, use_3d=True):
    """Calculate angle between three points (2D or 3D)"""
    if use_3d:
        return calculate_angle_3d(p1, p2, p3)
    
    if not all([p1, p2, p3]):
        return 0
    
    # 2D calculation
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_joint_velocity(landmarks, frame_history, joint_indices):
    """Calculate joint velocity based on frame history"""
    if len(frame_history) < 2:
        return {}
    
    velocities = {}
    current_frame = landmarks
    previous_frame = frame_history[-1].get('landmarks', [])
    
    for joint_name, joint_idx in joint_indices.items():
        if (joint_idx < len(current_frame) and joint_idx < len(previous_frame) and
            current_frame[joint_idx] and previous_frame[joint_idx]):
            
            current_pos = np.array([current_frame[joint_idx]['x'], current_frame[joint_idx]['y']])
            previous_pos = np.array([previous_frame[joint_idx]['x'], previous_frame[joint_idx]['y']])
            
            # Calculate velocity (pixels per frame)
            velocity = np.linalg.norm(current_pos - previous_pos)
            velocities[joint_name] = velocity
    
    return velocities

def calculate_flight_time(frame_history):
    """Calculate flight time based on foot position changes"""
    if len(frame_history) < 10:
        return 0
    
    # Look for periods where both feet are off the ground
    flight_frames = 0
    for frame_data in frame_history:
        landmarks = frame_data.get('landmarks', [])
        if len(landmarks) >= 29:
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            if left_ankle and right_ankle:
                # Check if feet are elevated (y position higher than hips)
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                if left_hip and right_hip:
                    hip_y = (left_hip['y'] + right_hip['y']) / 2
                    ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
                    
                    if ankle_y < hip_y - 0.1:  # Feet above hips
                        flight_frames += 1
    
    # Convert to time (assuming 30 fps)
    return flight_frames / 30.0

def calculate_angle_of_elevation(landmarks):
    """Calculate angle of elevation from hip to shoulder line"""
    if len(landmarks) < 25:
        return 0
    
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    if all([left_hip, right_hip, left_shoulder, right_shoulder]):
        # Calculate hip and shoulder centers
        hip_center = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2
        }
        shoulder_center = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2
        }
        
        # Calculate angle from horizontal
        dx = shoulder_center['x'] - hip_center['x']
        dy = hip_center['y'] - shoulder_center['y']  # Inverted Y axis
        
        if dx != 0:
            angle = math.degrees(math.atan2(dy, dx))
            return angle
    
    return 0

def calculate_acl_risk(landmarks):
    """Calculate ACL injury risk based on knee angles and positions"""
    if len(landmarks) < 29:
        return 0
    
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]
    
    risk_factors = 0
    
    # Check knee angles
    if all([left_hip, left_knee, left_ankle]):
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if left_knee_angle < 90:  # Deep flexion
            risk_factors += 1
    
    if all([right_hip, right_knee, right_ankle]):
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if right_knee_angle < 90:  # Deep flexion
            risk_factors += 1
    
    # Check knee valgus (knees caving in)
    if all([left_knee, right_knee]):
        knee_separation = abs(left_knee['x'] - right_knee['x'])
        if knee_separation < 0.1:  # Knees too close
            risk_factors += 1
    
    return min(risk_factors / 3.0, 1.0)  # Normalize to 0-1

def calculate_stakeholder_metrics(landmarks, frame_history, event_type="floor"):
    """Calculate comprehensive stakeholder metrics"""
    if not landmarks or len(landmarks) < 33:
        return {}
    
    metrics = {}
    
    # Basic joint angles
    if len(landmarks) > 28:
        # Shoulder angles
        if all([landmarks[11], landmarks[13], landmarks[15]]):  # Left shoulder
            metrics['left_shoulder_angle'] = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if all([landmarks[12], landmarks[14], landmarks[16]]):  # Right shoulder
            metrics['right_shoulder_angle'] = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        
        # Elbow angles
        if all([landmarks[11], landmarks[13], landmarks[15]]):  # Left elbow
            metrics['left_elbow_angle'] = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if all([landmarks[12], landmarks[14], landmarks[16]]):  # Right elbow
            metrics['right_elbow_angle'] = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        
        # Hip angles
        if all([landmarks[23], landmarks[25], landmarks[27]]):  # Left hip
            metrics['left_hip_angle'] = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        if all([landmarks[24], landmarks[26], landmarks[28]]):  # Right hip
            metrics['right_hip_angle'] = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        
        # Knee angles
        if all([landmarks[23], landmarks[25], landmarks[27]]):  # Left knee
            metrics['left_knee_angle'] = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        if all([landmarks[24], landmarks[26], landmarks[28]]):  # Right knee
            metrics['right_knee_angle'] = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    
    # Dynamic metrics
    joint_indices = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26
    }
    
    velocities = calculate_joint_velocity(landmarks, frame_history, joint_indices)
    metrics.update(velocities)
    
    # Flight time
    metrics['flight_time'] = calculate_flight_time(frame_history)
    
    # Angle of elevation
    metrics['angle_of_elevation'] = calculate_angle_of_elevation(landmarks)
    
    # ACL risk
    metrics['acl_risk'] = calculate_acl_risk(landmarks)
    
    # Event-specific metrics
    if event_type == "floor":
        metrics.update(calculate_floor_metrics(landmarks, frame_history))
    elif event_type == "vault":
        metrics.update(calculate_vault_metrics(landmarks, frame_history))
    elif event_type == "beam":
        metrics.update(calculate_beam_metrics(landmarks, frame_history))
    elif event_type == "bars":
        metrics.update(calculate_bars_metrics(landmarks, frame_history))
    
    # Ensure all boolean comparisons are explicitly cast
    metrics = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in metrics.items()}
    
    return metrics

def calculate_floor_metrics(landmarks, frame_history):
    """Floor exercise specific metrics"""
    metrics = {}
    
    # Split leap detection
    if len(landmarks) > 28:
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        if all([left_hip, right_hip, left_ankle, right_ankle]):
            # Calculate split angle
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2
            }
            
            # Vector from hip center to each ankle
            left_vector = {
                'x': left_ankle['x'] - hip_center['x'],
                'y': left_ankle['y'] - hip_center['y']
            }
            right_vector = {
                'x': right_ankle['x'] - hip_center['x'],
                'y': right_ankle['y'] - hip_center['y']
            }
            
            # Calculate angle between vectors
            dot_product = left_vector['x'] * right_vector['x'] + left_vector['y'] * right_vector['y']
            left_magnitude = math.sqrt(left_vector['x']**2 + left_vector['y']**2)
            right_magnitude = math.sqrt(right_vector['x']**2 + right_vector['y']**2)
            
            if left_magnitude > 0 and right_magnitude > 0:
                cos_angle = dot_product / (left_magnitude * right_magnitude)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                split_angle = math.degrees(math.acos(cos_angle))
                metrics['split_angle'] = split_angle
    
    return metrics

def calculate_vault_metrics(landmarks, frame_history):
    """Vault specific metrics"""
    metrics = {}
    
    # Height off table (estimated from shoulder position)
    if len(landmarks) > 12:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        if left_shoulder and right_shoulder:
            shoulder_height = (left_shoulder['y'] + right_shoulder['y']) / 2
            metrics['shoulder_height'] = shoulder_height
    
    return metrics

def calculate_beam_metrics(landmarks, frame_history):
    """Balance beam specific metrics"""
    metrics = {}
    
    # Balance detection (hip movement)
    if len(frame_history) > 5:
        hip_positions = []
        for frame_data in frame_history[-5:]:
            frame_landmarks = frame_data.get('landmarks', [])
            if len(frame_landmarks) > 24:
                left_hip = frame_landmarks[23]
                right_hip = frame_landmarks[24]
                if left_hip and right_hip:
                    hip_center = (left_hip['x'] + right_hip['x']) / 2
                    hip_positions.append(hip_center)
        
        if len(hip_positions) > 1:
            # Calculate hip movement variance
            hip_variance = np.var(hip_positions)
            metrics['balance_stability'] = 1.0 - min(hip_variance * 10, 1.0)
    
    return metrics

def calculate_bars_metrics(landmarks, frame_history):
    """Uneven bars specific metrics"""
    metrics = {}
    
    # Handstand angle detection
    if len(landmarks) > 28:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            # Calculate body line angle
            shoulder_center = {
                'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                'y': (left_shoulder['y'] + right_shoulder['y']) / 2
            }
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2
            }
            
            # Angle from vertical
            dx = shoulder_center['x'] - hip_center['x']
            dy = shoulder_center['y'] - hip_center['y']
            
            if dx != 0:
                angle = math.degrees(math.atan2(dx, dy))
                metrics['handstand_angle'] = abs(angle)
    
    return metrics

def save_analytics_to_json(metrics, event_type, filename=None):
    """Save analytics to JSON file"""
    try:
        if filename is None:
            filename = f"analytics_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        timestamp = datetime.now().isoformat()
        converted_metrics = convert_numpy_types(metrics)
        
        analytics_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "metrics": converted_metrics
        }
        
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        existing_data.append(analytics_entry)
        
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Analytics saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving analytics to JSON: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    coordinate_api_ok = check_coordinate_api_health()
    return jsonify({
        'status': 'ok',
        'mediapipe': 'running',
        'coordinate_api': 'available' if coordinate_api_ok else 'unavailable',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/detect-pose', methods=['POST'])
def detect_pose():
    """Detect pose in single image with coordinate extraction"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        event_type = data.get('event_type', 'floor')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure monotonically increasing timestamps to avoid MediaPipe conflicts
        global last_timestamp
        current_time = time.time()
        if current_time <= last_timestamp:
            current_time = last_timestamp + 0.001  # Add 1ms to ensure increase
        last_timestamp = current_time
        
        # Process with MediaPipe
        results = pose.process(image_rgb)
        
        landmarks = []
        metrics = {}
        
        if results.pose_landmarks:
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # Calculate metrics
            metrics = calculate_stakeholder_metrics(landmarks, list(frame_history), event_type)
        
        # Get coordinate data (using landmark-based calculation)
        coordinates = [calculate_bounding_box_from_landmarks(landmarks)] if landmarks else []
        
        # Update frame history
        if landmarks:
            frame_history.append({
                'landmarks': landmarks,
                'metrics': metrics,
                'timestamp': time.time()
            })
        
        # Update analytics history
        if metrics:
            analytics_history.append({
                'timestamp': time.time(),
                'metrics': metrics,
                'event_type': event_type
            })
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metrics': metrics,
            'coordinates': coordinates,
            'landmarks_count': len(landmarks)
        })
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-pose-batch', methods=['POST'])
def detect_pose_batch():
    """Process multiple frames with coordinate extraction"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        event_type = data.get('event_type', 'floor')
        
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
        
        results = []
        processed_count = 0
        
        for i, frame_data in enumerate(frames):
            try:
                image_base64 = frame_data.get('image')
                frame_id = frame_data.get('frame_id', i)
                
                if not image_base64:
                    continue
                
                # Convert base64 to image
                image = base64_to_image(image_base64)
                if image is None:
                    continue
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Ensure monotonically increasing timestamps to avoid MediaPipe conflicts
                global last_timestamp
                current_time = time.time()
                if current_time <= last_timestamp:
                    current_time = last_timestamp + 0.001  # Add 1ms to ensure increase
                last_timestamp = current_time
                
                # Process with MediaPipe
                results_mp = pose.process(image_rgb)
                
                landmarks = []
                metrics = {}
                
                if results_mp.pose_landmarks:
                    # Extract landmarks
                    for landmark in results_mp.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    # Calculate metrics
                    metrics = calculate_stakeholder_metrics(landmarks, list(frame_history), event_type)
                    processed_count += 1
                
                # Get coordinate data (using landmark-based calculation)
                coordinates = [calculate_bounding_box_from_landmarks(landmarks)] if landmarks else []
                
                # Update frame history
                if landmarks:
                    frame_history.append({
                        'landmarks': landmarks,
                        'metrics': metrics,
                        'timestamp': time.time()
                    })
                
                results.append({
                    'frame_id': frame_id,
                    'landmarks': landmarks,
                    'metrics': metrics,
                    'coordinates': coordinates,
                    'success': len(landmarks) > 0
                })
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                results.append({
                    'frame_id': i,
                    'landmarks': [],
                    'metrics': {},
                    'coordinates': [],
                    'success': False,
                    'error': str(e)
                })
        
        # Log analytics every 5 seconds
        global last_analytics_update
        current_time = time.time()
        if current_time - last_analytics_update >= ANALYTICS_INTERVAL:
            if analytics_history:
                latest_metrics = analytics_history[-1]['metrics']
                save_analytics_to_json(latest_metrics, event_type, ANALYTICS_JSON_FILE)
                last_analytics_update = current_time
                logger.info(f"Analytics logged at {datetime.now().strftime('%H:%M:%S')}")
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_count': processed_count,
            'total_frames': len(frames)
        })
        
    except Exception as e:
        logger.error(f"Error in batch pose detection: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/detect-pose1', methods=['POST'])
def detect_pose1():
    """Detect pose in single image with coordinate extraction"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        event_type = data.get('event_type', 'floor')

        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        # Convert base64 to image
        t0 = time.time()
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        t1 = time.time()

        # Resize to speed up MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (640, 480))   # 🔑 Downscale
        t2 = time.time()

        # Ensure monotonically increasing timestamps
        global last_timestamp
        current_time = time.time()
        if current_time <= last_timestamp:
            current_time = last_timestamp + 0.001
        last_timestamp = current_time

        # Run inference
        results = pose.process(image_rgb)
        t3 = time.time()

        landmarks, metrics = [], {}
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            metrics = calculate_stakeholder_metrics(landmarks, list(frame_history), event_type)

        # Bounding box
        coordinates = [calculate_bounding_box_from_landmarks(landmarks)] if landmarks else []

        # Update histories
        if landmarks:
            frame_history.append({
                'landmarks': landmarks,
                'metrics': metrics,
                'timestamp': time.time()
            })

        if metrics:
            analytics_history.append({
                'timestamp': time.time(),
                'metrics': metrics,
                'event_type': event_type
            })

        # Timing logs
        logger.info(f"⏱ Base64 decode: {t1 - t0:.2f}s | Resize: {t2 - t1:.2f}s | Inference: {t3 - t2:.2f}s | Total: {t3 - t0:.2f}s")

        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metrics': metrics,
            'coordinates': coordinates,
            'landmarks_count': len(landmarks),
            'timing': {
                'decode': round(t1 - t0, 2),
                'resize': round(t2 - t1, 2),
                'inference': round(t3 - t2, 2),
                'total': round(t3 - t0, 2)
            }
        })

    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get-analytics', methods=['GET'])
def get_analytics():
    """Get latest analytics"""
    if not analytics_history:
        return jsonify({'error': 'No analytics available'}), 404
    
    latest = analytics_history[-1]
    return jsonify({
        'success': True,
        'analytics': convert_numpy_types(latest)
    })

@app.route('/set-analytics-filename', methods=['POST'])
def set_analytics_filename():
    """Set the filename for analytics JSON output"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if filename:
            global ANALYTICS_JSON_FILE
            ANALYTICS_JSON_FILE = filename
            return jsonify({'success': True, 'message': f'Analytics filename set to {filename}'})
        else:
            return jsonify({'error': 'No filename provided'}), 400
    except Exception as e:
        logger.error(f"Error setting analytics filename: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics-json', methods=['GET'])
def get_analytics_json():
    """Get the full analytics JSON file"""
    if not ANALYTICS_JSON_FILE or not os.path.exists(ANALYTICS_JSON_FILE):
        return jsonify({'error': 'No analytics file available'}), 404
    
    try:
        with open(ANALYTICS_JSON_FILE, 'r') as f:
            data = json.load(f)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-analytics', methods=['POST'])
def clear_analytics():
    """Clear analytics history"""
    global analytics_history
    analytics_history.clear()
    return jsonify({'success': True, 'message': 'Analytics history cleared'})

@app.route('/clear-analytics-json', methods=['POST'])
def clear_analytics_json():
    """Clear the analytics JSON file"""
    global ANALYTICS_JSON_FILE
    if ANALYTICS_JSON_FILE and os.path.exists(ANALYTICS_JSON_FILE):
        try:
            os.remove(ANALYTICS_JSON_FILE)
            ANALYTICS_JSON_FILE = None
            return jsonify({'success': True, 'message': 'Analytics file cleared'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'success': True, 'message': 'No file to clear'})

@app.route('/get-coordinates', methods=['POST'])
def get_coordinates():
    """Get coordinates for an image"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get coordinates
        coordinates = get_coordinates_from_image(image, 0)
        
        return jsonify({
            'success': True,
            'coordinates': coordinates or []
        })
        
    except Exception as e:
        logger.error(f"Error getting coordinates: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced MediaPipe Server with Coordinate Extraction")
    print(f"📍 Coordinate API URL: {COORDINATE_API_URL}")
    print(f"📍 Coordinate API Enabled: {COORDINATE_API_ENABLED}")
    
    # Check coordinate API health
    if COORDINATE_API_ENABLED:
        coordinate_ok = check_coordinate_api_health()
        print(f"📍 Coordinate API Status: {'✅ Available' if coordinate_ok else '❌ Unavailable'}")

    port = int(os.environ.get("PORT", 5001))
    
    app.run(host='0.0.0.0', port=port, debug=True)
