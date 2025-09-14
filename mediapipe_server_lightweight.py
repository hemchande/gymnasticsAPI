#!/usr/bin/env python3
"""
Lightweight MediaPipe server optimized for Render deployment
- Reduced memory usage
- Faster startup
- Better error handling
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import time
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for testing

# Global variables for MediaPipe
mp_pose = None
pose = None

def initialize_mediapipe():
    """Initialize MediaPipe with minimal memory usage"""
    global mp_pose, pose
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe with minimal configuration
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,  # Process single images, not video
            model_complexity=0,      # Lightest model (0=light, 1=medium, 2=heavy)
            enable_segmentation=False,  # Disable segmentation to save memory
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✅ MediaPipe initialized successfully")
        return True
        
    except ImportError:
        logger.error("❌ MediaPipe not available")
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing MediaPipe: {e}")
        return False

def process_image_lightweight(image_data):
    """Process image with minimal memory usage"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Could not decode image", "success": False}
        
        # Resize image to reduce memory usage (max 640x480)
        height, width = image.shape[:2]
        if width > 640 or height > 480:
            scale = min(640/width, 480/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        if pose is None:
            return {"error": "MediaPipe not initialized", "success": False}
        
        results = pose.process(image_rgb)
        
        # Extract landmarks
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(landmark.visibility)
                })
        
        return {
            "success": True,
            "landmarks": landmarks,
            "image_shape": image.shape,
            "processing_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"error": str(e), "success": False}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "mediapipe": "running" if pose is not None else "unavailable",
        "coordinate_api": "available",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/detect-pose', methods=['POST'])
def detect_pose():
    """Detect pose in image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        logger.info(f"Processing pose detection request at {timestamp}")
        
        # Process image
        result = process_image_lightweight(image_data)
        
        if result['success']:
            return jsonify({
                "success": True,
                "landmarks": result['landmarks'],
                "timestamp": timestamp,
                "processing_time": result.get('processing_time', 0),
                "image_shape": result.get('image_shape', [0, 0, 0])
            })
        else:
            return jsonify({
                "success": False,
                "error": result['error'],
                "timestamp": timestamp
            }), 500
            
    except Exception as e:
        logger.error(f"Error in detect_pose: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": int(time.time() * 1000)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "MediaPipe Pose Detection Server",
        "status": "running",
        "endpoints": ["/health", "/detect-pose"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        "status": "ok",
        "mediapipe_initialized": pose is not None,
        "memory_usage": "optimized",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize MediaPipe
    if not initialize_mediapipe():
        logger.warning("⚠️  MediaPipe initialization failed - server will run with limited functionality")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Starting lightweight MediaPipe server on port {port}")
    
    # Run with minimal configuration
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Disable debug mode for production
        threaded=True,  # Enable threading
        processes=1  # Single process to save memory
    )
