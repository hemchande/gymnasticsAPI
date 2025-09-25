#!/usr/bin/env python3
"""
Updated Gymnastics API Server with Cloudflare Stream Integration
This server integrates Cloudflare Stream for video upload and streaming while maintaining MongoDB for metadata
"""

from flask import Flask, request, jsonify, send_file, Response, stream_template, make_response
from flask_cors import CORS
from flask_compress import Compress
import json
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
import io
import gridfs
from bson import ObjectId
import re
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
import hashlib

# Import database modules
from database import db_manager, sessions, users, video_metadata

# Custom JSON encoder to handle ObjectId and datetime
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)

# Helper function to convert ObjectIds to strings in dictionaries
def convert_objectids_to_strings(obj):
    """Recursively convert ObjectId instances to strings in dictionaries and lists"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids_to_strings(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj

app = Flask(__name__)
app.json_encoder = JSONEncoder
CORS(app, 
     origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8080", "https://motionlabsai-qb7r-5ljq6f0oj-hemchandeishagmailcoms-projects.vercel.app", "https://www.motionlabsai.com","https://motionlabsai-qb7r-9n46w958x-hemchandeishagmailcoms-projects.vercel.app","https://motionlabsai-qb7r-hemchandeishagmailcoms-projects.vercel.app","https://motionlabsai-qb7r.vercel.app"], 
     allow_headers=["Content-Type", "Authorization", "Range", "X-Requested-With", "Accept", "Origin"],
     expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
     supports_credentials=True)

# Enable response compression
Compress(app)

# Configuration
MEDIAPIPE_SERVER_URL = "https://extraordinary-gentleness-production.up.railway.app"
VIDEO_PROCESSING_DIR = "../output_videos"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_videos")
ANALYTICS_DIR = "../analytics"
TEMP_DIR = "/tmp" if os.path.exists("/tmp") else "."

# Cloudflare Stream Configuration
CLOUDFLARE_ACCOUNT_ID = "f2b0714a082195118f53d0b8327f6635"
CLOUDFLARE_API_TOKEN = "DEmkpIDn5SLgpjTOoDqYrPivnOpD9gnqbVICwzTQ"
CLOUDFLARE_STREAM_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/stream"
CLOUDFLARE_STREAM_DOMAIN = "customer-cxebs7nmdazhytrk.cloudflarestream.com"

# Performance Configuration
MAX_WORKERS = 4  # Limit concurrent video processing
CHUNK_SIZE = 32 * 1024  # 32KB chunks for streaming (reduced for better performance)
MEMORY_THRESHOLD = 80  # Memory usage threshold (%)
REQUEST_TIMEOUT = 30  # Request timeout in seconds
CACHE_SIZE = 100  # LRU cache size for frequently accessed data
MAX_ANALYTICS_FRAMES = 50  # Limit frames returned per request
VIDEO_STREAM_TIMEOUT = 60  # Video streaming timeout in seconds
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB max video size for streaming

# Initialize GridFS (only if MongoDB is available)
fs = None
if db_manager.db is not None:
    fs = gridfs.GridFS(db_manager.db)
    print("✅ GridFS initialized")
else:
    print("⚠️ GridFS not available (MongoDB not connected)")

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

class MongoDBVideoProcessor:
    """Video processor that works with MongoDB GridFS"""
    
    def __init__(self):
        self.mediapipe_url = MEDIAPIPE_SERVER_URL
        self.fs = fs
    
    def check_mediapipe_server(self):
        """Check if MediaPipe server is running"""
        try:
            response = requests.get(f"{self.mediapipe_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def find_best_video(self, video_filename):
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
    
    def process_video_with_analytics(self, video_path, output_name=None):
        """Process video using the fixed overlay system and convert to H.264"""
        if not output_name:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"processed_{base_name}.mp4"
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # Analytics file is created in current directory with "fixed_analytics_" prefix
        analytics_path = f"fixed_analytics_{os.path.basename(video_path)}.json"
        
        # Command to run the fixed video overlay script
        cmd = [
            "python3", "fixed_video_overlay_with_analytics_enhanced.py",  # Changed from "video_overlay_with_analytics_fixed.py",
            video_path,
            "--output", output_path,
            "--server", MEDIAPIPE_SERVER_URL
        ]
        
        try:
            print(f"🎬 Starting video processing: {os.path.basename(video_path)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"✅ Video overlay processing completed: {output_name}")
                
                # The video is already processed, no need for additional H.264 conversion
                # Use the output_name directly since it's already the final processed video
                h264_output_path = os.path.join(OUTPUT_DIR, output_name)
                print(f"✅ Using processed video directly: {os.path.basename(h264_output_path)}")
                
                # The video is already processed and ready to use
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 Processed video size: {file_size:.1f}MB")
                
                return {
                    "success": True,
                    "output_video": output_path,
                    "h264_video": output_path,  # Same as output_video since it's already processed
                    "analytics_file": analytics_path,
                    "message": f"Video processed successfully ({file_size:.1f}MB)"
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
    
    def upload_processed_video_to_gridfs(self, video_path, analytics_path, metadata=None):
        """Upload processed video and analytics to GridFS and create session"""
        try:
            # Upload video to GridFS
            video_filename = os.path.basename(video_path)
            video_metadata = metadata or {}
            video_metadata.update({
                "processing_type": "analyzed_video",
                "upload_timestamp": datetime.now().isoformat()
            })
            
            with open(video_path, 'rb') as video_file:
                video_id = self.fs.put(
                    video_file,
                    filename=video_filename,
                    metadata=video_metadata,
                    contentType='video/mp4'
                )
            
            print(f"✅ Video uploaded to GridFS: {video_filename} (ID: {video_id})")
            
            # Upload analytics to GridFS
            analytics_id = None
            if analytics_path and os.path.exists(analytics_path):
                analytics_filename = os.path.basename(analytics_path)
                analytics_metadata = {
                    "video_filename": video_filename,
                    "analysis_type": "per_frame_analytics",
                    "upload_timestamp": datetime.now().isoformat()
                }
                
                with open(analytics_path, 'rb') as analytics_file:
                    analytics_id = self.fs.put(
                        analytics_file,
                        filename=analytics_filename,
                        metadata=analytics_metadata,
                        contentType='application/json'
                    )
                
                print(f"✅ Analytics uploaded to GridFS: {analytics_filename} (ID: {analytics_id})")
            
            return video_id, analytics_id
            
        except Exception as e:
            print(f"❌ Error uploading to GridFS: {e}")
            return None, None
    
    def get_analytics_from_gridfs(self, analytics_id):
        """Retrieve analytics file from GridFS with caching"""
        try:
            if isinstance(analytics_id, str):
                analytics_id = ObjectId(analytics_id)
            
            grid_out = self.fs.get(analytics_id)
            return json.loads(grid_out.read().decode('utf-8'))
        except Exception as e:
            print(f"❌ Error retrieving analytics from GridFS: {e}")
            return None

# Initialize video processor
video_processor = MongoDBVideoProcessor()

def convert_timestamps_to_relative(analytics_data):
    """Convert Unix timestamps to relative timestamps for video synchronization"""
    try:
        # Handle both list format and dict format
        if isinstance(analytics_data, list):
            # Direct list of frames
            frame_data = analytics_data
        elif 'frame_data' in analytics_data and isinstance(analytics_data['frame_data'], list):
            # Dictionary with frame_data key
            frame_data = analytics_data['frame_data']
        else:
            # No frame data found
            return analytics_data
            
        if len(frame_data) == 0:
            return analytics_data
            
        # Get the first timestamp as the baseline
        first_timestamp = None
        for frame in frame_data:
            if 'metrics' in frame and 'timestamp' in frame['metrics']:
                first_timestamp = frame['metrics']['timestamp']
                break
        
        if first_timestamp is None:
            print("⚠️ No timestamps found in frame data")
            return analytics_data
        
        # Convert all timestamps to relative time (seconds from start)
        for frame in frame_data:
            if 'metrics' in frame and 'timestamp' in frame['metrics']:
                original_timestamp = frame['metrics']['timestamp']
                relative_time = original_timestamp - first_timestamp
                frame['timestamp'] = relative_time  # Add relative timestamp at frame level
                frame['metrics']['relative_timestamp'] = relative_time  # Keep relative timestamp in metrics too
        
        print(f"✅ Converted {len(frame_data)} frames to relative timestamps (baseline: {first_timestamp})")
        
        return analytics_data
        
    except Exception as e:
        print(f"❌ Error converting timestamps: {e}")
        return analytics_data

# Global OPTIONS handler for CORS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Cloudflare Stream API Helper Functions
def upload_to_cloudflare_stream(video_path, metadata=None):
    """
    Upload a video to Cloudflare Stream
    
    Args:
        video_path (str): Path to the video file
        metadata (dict): Optional metadata for the video
    
    Returns:
        dict: Cloudflare Stream response with video ID and details
    """
    try:
        print(f"🌊 Uploading video to Cloudflare Stream: {video_path}")
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
        }
        
        # Prepare files and data
        files = {
            'file': open(video_path, 'rb')
        }
        
        data = {}
        if metadata:
            data['meta'] = json.dumps(metadata)
        
        # Upload to Cloudflare Stream
        response = requests.post(
            CLOUDFLARE_STREAM_BASE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300  # 5 minute timeout for uploads
        )
        
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                video_data = result.get('result', {})
                print(f"✅ Video uploaded to Cloudflare Stream successfully")
                print(f"📊 Video UID: {video_data.get('uid')}")
                print(f"📊 Video Size: {video_data.get('size')} bytes")
                return video_data
            else:
                print(f"❌ Cloudflare Stream upload failed: {result.get('errors', [])}")
                return None
        else:
            print(f"❌ Cloudflare Stream upload failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading to Cloudflare Stream: {e}")
        return None

def get_cloudflare_stream_url(video_uid):
    """
    Get the streaming URL for a Cloudflare Stream video
    
    Args:
        video_uid (str): Cloudflare Stream video UID
    
    Returns:
        str: Streaming URL for the video
    """
    # Use iframe URL for now - this should work with HTML5 video elements
    # Cloudflare Stream iframe URLs can be used directly in video src
    return f"https://{CLOUDFLARE_STREAM_DOMAIN}/{video_uid}/iframe"

def get_cloudflare_video_info(video_uid):
    """
    Get video information from Cloudflare Stream
    
    Args:
        video_uid (str): Cloudflare Stream video UID
    
    Returns:
        dict: Video information from Cloudflare Stream
    """
    try:
        headers = {
            'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
        }
        
        response = requests.get(
            f"{CLOUDFLARE_STREAM_BASE_URL}/{video_uid}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('result', {})
            else:
                print(f"❌ Failed to get Cloudflare video info: {result.get('errors', [])}")
                return None
        else:
            print(f"❌ Failed to get Cloudflare video info with status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting Cloudflare video info: {e}")
        return None

def list_cloudflare_videos(limit=100):
    """
    List videos from Cloudflare Stream
    
    Args:
        limit (int): Maximum number of videos to return
    
    Returns:
        list: List of video information from Cloudflare Stream
    """
    try:
        headers = {
            'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
        }
        
        response = requests.get(
            f"{CLOUDFLARE_STREAM_BASE_URL}?limit={limit}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('result', [])
            else:
                print(f"❌ Failed to list Cloudflare videos: {result.get('errors', [])}")
                return []
        else:
            print(f"❌ Failed to list Cloudflare videos with status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing Cloudflare videos: {e}")
        return []

def enable_cloudflare_stream_download(video_uid):
    """
    Enable MP4 download for a Cloudflare Stream video
    
    Args:
        video_uid (str): Cloudflare Stream video UID
    
    Returns:
        dict: Download info with status and URL, or None if failed
    """
    try:
        print(f"🌊 Enabling MP4 download for video: {video_uid}")
        
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        
        # Enable MP4 download
        response = requests.post(
            f"{CLOUDFLARE_STREAM_BASE_URL}/{video_uid}/downloads",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                download_info = result.get('result', {}).get('default', {})
                print(f"✅ MP4 download enabled: {download_info.get('status')}")
                return download_info
            else:
                print(f"❌ Failed to enable MP4 download: {result.get('errors', [])}")
                return None
        else:
            print(f"❌ Failed to enable MP4 download with status {response.status_code}")
            print(f"❌ Response body: {response.text}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                print(f"❌ Could not parse error response as JSON")
            return None
            
    except Exception as e:
        print(f"❌ Error enabling MP4 download: {e}")
        return None

def get_cloudflare_stream_download_status(video_uid):
    """
    Get download status for a Cloudflare Stream video
    
    Args:
        video_uid (str): Cloudflare Stream video UID
    
    Returns:
        dict: Download info with status and URL, or None if failed
    """
    try:
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        
        response = requests.get(
            f"{CLOUDFLARE_STREAM_BASE_URL}/{video_uid}/downloads",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                download_info = result.get('result', {}).get('default', {})
                return download_info
            else:
                print(f"❌ Failed to get download status: {result.get('errors', [])}")
                return None
        else:
            print(f"❌ Failed to get download status with status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting download status: {e}")
        return None

def download_video_from_cloudflare_stream(video_uid, output_path):
    """
    Download a video from Cloudflare Stream to local file
    
    Args:
        video_uid (str): Cloudflare Stream video UID
        output_path (str): Local path to save the video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🌊 Downloading video from Cloudflare Stream: {video_uid}")
        
        # First, check if download is already available
        download_info = get_cloudflare_stream_download_status(video_uid)
        
        if not download_info:
            # Enable MP4 download if not available
            print(f"🔄 Enabling MP4 download for video: {video_uid}")
            download_info = enable_cloudflare_stream_download(video_uid)
            
            if not download_info:
                print(f"❌ Failed to enable MP4 download for video: {video_uid}")
                return False
        
        # Check if download is ready
        if download_info.get('status') == 'ready':
            download_url = download_info.get('url')
            print(f"✅ Download is ready: {download_url}")
        else:
            # Poll until download is ready
            print(f"⏳ Download status: {download_info.get('status')}, polling until ready...")
            max_attempts = 30  # 5 minutes max
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(10)  # Wait 10 seconds between polls
                download_info = get_cloudflare_stream_download_status(video_uid)
                
                if not download_info:
                    print(f"❌ Failed to get download status")
                    return False
                
                status = download_info.get('status')
                percent = download_info.get('percentComplete', 0)
                
                print(f"📊 Download progress: {status} ({percent}%)")
                
                if status == 'ready':
                    download_url = download_info.get('url')
                    print(f"✅ Download is ready: {download_url}")
                    break
                elif status == 'error':
                    print(f"❌ Download failed")
                    return False
                
                attempt += 1
            
            if attempt >= max_attempts:
                print(f"❌ Download timeout after {max_attempts} attempts")
                return False
        
        # Download the video
        if not download_url:
            print(f"❌ No download URL available")
            return False
        
        print(f"📥 Downloading from: {download_url}")
        
        # Download the video
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        response = requests.get(download_url, headers=headers, stream=True, timeout=300)
        
        if response.status_code == 200:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write video to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Video downloaded successfully: {output_path} ({file_size} bytes)")
            return True
        else:
            print(f"❌ Failed to download video with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading video from Cloudflare Stream: {e}")
        return False

# Performance monitoring
def check_memory_usage():
    """Check current memory usage and trigger cleanup if needed"""
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MEMORY_THRESHOLD:
            print(f"⚠️ High memory usage: {memory_percent:.1f}%")
            gc.collect()  # Force garbage collection
            return True
        return False
    except Exception as e:
        print(f"❌ Error checking memory: {e}")
        return False

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used if hasattr(psutil, 'virtual_memory') else 0
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used if hasattr(psutil, 'virtual_memory') else 0
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if execution_time > 5.0:  # Log slow operations
                print(f"⏱️ Slow operation: {func.__name__} took {execution_time:.2f}s, memory delta: {memory_delta/1024/1024:.1f}MB")
            
            # Check memory usage after operation
            check_memory_usage()
    
    return wrapper

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check MongoDB connection
        db_manager.client.admin.command('ping')
        mongodb_status = "connected"
    except Exception as e:
        mongodb_status = f"error: {str(e)}"
    
    # Check MediaPipe server
    try:
        response = requests.get(f"{MEDIAPIPE_SERVER_URL}/health", timeout=5)
        mediapipe_status = "running" if response.status_code == 200 else "error"
    except Exception:
        mediapipe_status = "error"
    
    # Check Cloudflare Stream
    try:
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        response = requests.get(f"{CLOUDFLARE_STREAM_BASE_URL}?limit=1", headers=headers, timeout=10)
        cloudflare_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        cloudflare_status = "error"
    
    return jsonify({
        "status": "healthy",
        "mongodb": mongodb_status,
        "mediapipe_server": mediapipe_status,
        "cloudflare_stream": cloudflare_status,
        "timestamp": datetime.now().isoformat()
    })

# Cloudflare Stream Upload Endpoint
@app.route('/stream/upload', methods=['POST'])
@monitor_performance
def upload_to_stream():
    """Upload a video to Cloudflare Stream and create a session in MongoDB"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Get metadata from form
        athlete_name = request.form.get('athlete_name', 'Unknown Athlete')
        event = request.form.get('event', 'Unknown Event')
        session_name = request.form.get('session_name', file.filename)
        description = request.form.get('description', '')
        
        # Save uploaded file temporarily
        temp_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{file.filename}")
        file.save(temp_path)
        
        try:
            # Prepare metadata for Cloudflare Stream
            metadata = {
                "name": file.filename,
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "description": description,
                "upload_source": "gymnastics_api_server_updated2"
            }
            
            # Upload to Cloudflare Stream
            cloudflare_video = upload_to_cloudflare_stream(temp_path, metadata)
            
            if not cloudflare_video:
                return jsonify({"error": "Failed to upload video to Cloudflare Stream"}), 500
            
            # Get streaming URL
            stream_url = get_cloudflare_stream_url(cloudflare_video.get('uid'))
            
            # Create session in MongoDB with Cloudflare Stream integration
            session_data = {
                "athlete_name": athlete_name,
                "session_name": session_name,
                "event": event,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "duration": f"{cloudflare_video.get('duration', 0):.1f}s",
                "original_filename": file.filename,
                "processed_video_filename": file.filename,
                "processed_video_url": stream_url,
                "status": "uploaded",
                "processing_status": "completed",
                "is_binary_stored": False,  # Stored in Cloudflare Stream, not GridFS
                "meta": {
                    "cloudflare_stream_id": cloudflare_video.get('uid'),
                    "cloudflare_uid": cloudflare_video.get('uid'),
                    "upload_source": "cloudflare_stream",
                    "upload_timestamp": datetime.now().isoformat(),
                    "video_size": cloudflare_video.get('size'),
                    "video_duration": cloudflare_video.get('duration'),
                    "ready_to_stream": cloudflare_video.get('readyToStream', False),
                    "stream_url": stream_url,
                    "thumbnail": cloudflare_video.get('thumbnail'),
                    "description": description
                },
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Save session to MongoDB
            session_id = sessions.create_session(session_data)
            
            print(f"✅ Session created with Cloudflare Stream integration: {session_id}")
            
            return jsonify({
                "success": True,
                "message": "Video uploaded to Cloudflare Stream successfully",
                "session_id": session_id,
                "video": {
                    "id": cloudflare_video.get('uid'),
                    "uid": cloudflare_video.get('uid'),
                    "filename": file.filename,
                    "size": cloudflare_video.get('size'),
                    "duration": cloudflare_video.get('duration'),
                    "ready_to_stream": cloudflare_video.get('readyToStream', False),
                    "stream_url": stream_url,
                    "thumbnail": cloudflare_video.get('thumbnail'),
                    "created": cloudflare_video.get('created')
                }
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"❌ Error in stream upload: {e}")
        return jsonify({"error": str(e)}), 500

# List Cloudflare Stream Videos
@app.route('/stream/videos', methods=['GET'])
def list_stream_videos():
    """List videos from Cloudflare Stream"""
    try:
        limit = request.args.get('limit', 100, type=int)
        videos = list_cloudflare_videos(limit)
        
        # Convert to response format
        video_list = []
        for video in videos:
            stream_url = get_cloudflare_stream_url(video.get('uid'))
            video_list.append({
                "id": video.get('uid'),
                "uid": video.get('uid'),
                "size": video.get('size'),
                "duration": video.get('duration'),
                "ready_to_stream": video.get('readyToStream', False),
                "stream_url": stream_url,
                "thumbnail": video.get('thumbnail'),
                "created": video.get('created'),
                "meta": video.get('meta', {})
            })
        
        return jsonify({
            "success": True,
            "videos": video_list,
            "count": len(video_list)
        })
        
    except Exception as e:
        print(f"❌ Error listing stream videos: {e}")
        return jsonify({"error": str(e)}), 500

# Get Cloudflare Stream Video Info
@app.route('/stream/video/<video_id>', methods=['GET'])
def get_stream_video(video_id):
    """Get video information from Cloudflare Stream"""
    try:
        video_info = get_cloudflare_video_info(video_id)
        
        if not video_info:
            return jsonify({"error": "Video not found"}), 404
        
        stream_url = get_cloudflare_stream_url(video_id)
        
        return jsonify({
            "success": True,
            "video": {
                "id": video_info.get('uid'),
                "uid": video_info.get('uid'),
                "size": video_info.get('size'),
                "duration": video_info.get('duration'),
                "ready_to_stream": video_info.get('readyToStream', False),
                "stream_url": stream_url,
                "thumbnail": video_info.get('thumbnail'),
                "created": video_info.get('created'),
                "meta": video_info.get('meta', {})
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting stream video: {e}")
        return jsonify({"error": str(e)}), 500

# Enhanced getVideo endpoint with Cloudflare Stream integration
@app.route('/getVideo', methods=['GET', 'HEAD'])
def get_video():
    """Enhanced getVideo endpoint with Cloudflare Stream integration"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        print(f"🎬 Requesting video: {video_filename}")
        
        # Priority 1: Check if video is in Cloudflare Stream via MongoDB
        session = sessions.get_session_by_video_filename(video_filename)
        if session and session.get('meta', {}).get('cloudflare_uid'):
            cloudflare_uid = session['meta']['cloudflare_uid']
            
            # For analyzed videos, check if there's an analyzed stream URL
            if session.get('meta', {}).get('analyzed_stream_url'):
                analyzed_url = session['meta']['analyzed_stream_url']
                print(f"🌊 Using analyzed Cloudflare Stream URL: {analyzed_url}")
                return Response(
                    response=json.dumps({"redirect_url": analyzed_url}),
                    status=302,
                    headers={"Location": analyzed_url}
                )
            else:
                # Use original stream URL
                stream_url = session['meta'].get('stream_url')
                if stream_url:
                    print(f"🌊 Using original Cloudflare Stream URL: {stream_url}")
                    return Response(
                        response=json.dumps({"redirect_url": stream_url}),
                        status=302,
                        headers={"Location": stream_url}
                    )
                else:
                    # Fallback to constructed URL
                    stream_url = get_cloudflare_stream_url(cloudflare_uid)
                    print(f"🌊 Constructed Cloudflare Stream URL: {stream_url}")
                    return Response(
                        response=json.dumps({"redirect_url": stream_url}),
                        status=302,
                        headers={"Location": stream_url}
                    )
        
        # Priority 2: Check local cache (if implemented)
        # This would check for locally cached videos
        
        # Priority 3: Fall back to GridFS
        if session and session.get('gridfs_video_id'):
            video_id = session['gridfs_video_id']
            return stream_video_from_gridfs(video_id, video_filename)
        
        return jsonify({"error": "Video not found"}), 404
        
    except Exception as e:
        print(f"❌ Error in getVideo: {e}")
        return jsonify({"error": str(e)}), 500

# GridFS streaming function (from original server)
def stream_video_from_gridfs(video_id, video_filename):
    """Stream video from GridFS with range support"""
    try:
        # Get video from GridFS
        video_file = fs.get(ObjectId(video_id))
        if not video_file:
            return jsonify({"error": "Video not found in GridFS"}), 404
        
        file_size = video_file.length
        
        # Handle range requests
        range_header = request.headers.get('Range')
        if range_header:
            # Parse range header
            range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                
                # Ensure end doesn't exceed file size
                end = min(end, file_size - 1)
                
                # Read the requested range
                video_file.seek(start)
                data = video_file.read(end - start + 1)
                
                # Create response with range headers
                response = Response(
                    data,
                    status=206,
                    headers={
                        'Content-Range': f'bytes {start}-{end}/{file_size}',
                        'Accept-Ranges': 'bytes',
                        'Content-Length': str(len(data)),
                        'Content-Type': 'video/mp4',
                        'Cache-Control': 'public, max-age=3600'
                    }
                )
                return response
        
        # Serve entire file
        video_file.seek(0)
        return Response(
            video_file,
            mimetype='video/mp4',
            headers={
                'Content-Length': str(file_size),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600'
            }
        )
        
    except Exception as e:
        print(f"❌ Error streaming video from GridFS: {e}")
        return jsonify({"error": str(e)}), 500

# Get sessions endpoint (enhanced with Cloudflare Stream info)
@app.route('/getSessions', methods=['GET'])
def get_sessions():
    """Get all sessions with Cloudflare Stream integration info"""
    try:
        all_sessions = sessions.get_all_sessions()
        
        # Enhance sessions with Cloudflare Stream info
        enhanced_sessions = []
        for session in all_sessions:
            enhanced_session = session.copy()
            
            # Add Cloudflare Stream info if available
            if session.get('meta', {}).get('cloudflare_uid'):
                cloudflare_uid = session['meta']['cloudflare_uid']
                enhanced_session['cloudflare_stream_url'] = get_cloudflare_stream_url(cloudflare_uid)
                enhanced_session['has_cloudflare_stream'] = True
            else:
                enhanced_session['has_cloudflare_stream'] = False
            
            # Convert ObjectIds to strings
            enhanced_session = convert_objectids_to_strings(enhanced_session)
            enhanced_sessions.append(enhanced_session)
        
        return jsonify({
            "sessions": enhanced_sessions,
            "count": len(enhanced_sessions)
        })
        
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return jsonify({"error": str(e)}), 500

# Get sessions by user ID endpoint
@app.route('/getSessionsByUser/<user_id>', methods=['GET'])
def get_sessions_by_user(user_id):
    """Get sessions for a specific user with Cloudflare Stream integration info"""
    try:
        # Get all sessions first
        all_sessions = sessions.get_all_sessions()
        
        # Filter sessions by user_id
        user_sessions = []
        for session in all_sessions:
            # Check if session belongs to the user
            session_user_id = session.get('user_id', '')
            session_athlete_name = session.get('athlete_name', '')
            
            # More restrictive matching logic:
            # 1. Direct user_id match (exact match)
            # 2. Athlete name matches user email (for sessions without proper user_id)
            # 3. Only include "demo_user" sessions if they were created by this specific user
            should_include = False
            
            if session_user_id == user_id or session_user_id == user_id.lower():
                should_include = True
            elif session_athlete_name == user_id or session_athlete_name == user_id.lower():
                should_include = True
            elif not session_user_id and session_athlete_name:
                # If no user_id set but athlete_name exists, include it
                should_include = True
            
            # Don't include "demo_user" sessions for real users unless they specifically match
            
            if should_include:
                enhanced_session = session.copy()
                
                # Add Cloudflare Stream info if available
                if session.get('meta', {}).get('cloudflare_uid'):
                    cloudflare_uid = session['meta']['cloudflare_uid']
                    enhanced_session['cloudflare_stream_url'] = get_cloudflare_stream_url(cloudflare_uid)
                    enhanced_session['has_cloudflare_stream'] = True
                else:
                    enhanced_session['has_cloudflare_stream'] = False
                
                # Convert ObjectIds to strings
                enhanced_session = convert_objectids_to_strings(enhanced_session)
                user_sessions.append(enhanced_session)
        
        print(f"🔍 Found {len(user_sessions)} sessions for user: {user_id}")
        
        return jsonify({
            "sessions": user_sessions,
            "count": len(user_sessions),
            "user_id": user_id
        })
        
    except Exception as e:
        print(f"❌ Error getting sessions for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500

# Get session by ID endpoint (enhanced)
@app.route('/getSession/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific session with Cloudflare Stream integration info"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Add Cloudflare Stream info if available
        if session.get('meta', {}).get('cloudflare_uid'):
            cloudflare_uid = session['meta']['cloudflare_uid']
            session['cloudflare_stream_url'] = get_cloudflare_stream_url(cloudflare_uid)
            session['has_cloudflare_stream'] = True
        else:
            session['has_cloudflare_stream'] = False
        
        # Convert ObjectIds to strings
        session = convert_objectids_to_strings(session)
        
        return jsonify(session)
        
    except Exception as e:
        print(f"❌ Error getting session: {e}")
        return jsonify({"error": str(e)}), 500

# Sync existing GridFS video to Cloudflare Stream
@app.route('/stream/sync', methods=['POST'])
def sync_video_to_stream():
    """Sync an existing GridFS video to Cloudflare Stream"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Check if already synced to Cloudflare Stream
        if session.get('meta', {}).get('cloudflare_uid'):
            return jsonify({
                "message": "Video already synced to Cloudflare Stream",
                "cloudflare_uid": session['meta']['cloudflare_uid']
            })
        
        # Get video from GridFS
        if not session.get('gridfs_video_id'):
            return jsonify({"error": "No GridFS video ID found"}), 404
        
        video_id = session['gridfs_video_id']
        video_file = fs.get(ObjectId(video_id))
        if not video_file:
            return jsonify({"error": "Video not found in GridFS"}), 404
        
        # Save GridFS video to temporary file
        temp_path = os.path.join(TEMP_DIR, f"sync_{int(time.time())}_{video_filename}")
        with open(temp_path, 'wb') as f:
            f.write(video_file.read())
        
        try:
            # Prepare metadata
            metadata = {
                "name": video_filename,
                "athlete_name": session.get('athlete_name', 'Unknown'),
                "event": session.get('event', 'Unknown'),
                "session_id": str(session['_id']),
                "sync_source": "gridfs"
            }
            
            # Upload to Cloudflare Stream
            cloudflare_video = upload_to_cloudflare_stream(temp_path, metadata)
            
            if not cloudflare_video:
                return jsonify({"error": "Failed to upload video to Cloudflare Stream"}), 500
            
            # Get streaming URL
            stream_url = get_cloudflare_stream_url(cloudflare_video.get('uid'))
            
            # Update session with Cloudflare Stream info
            update_data = {
                "meta.cloudflare_stream_id": cloudflare_video.get('uid'),
                "meta.cloudflare_uid": cloudflare_video.get('uid'),
                "meta.cloudflare_stream_url": stream_url,
                "meta.sync_timestamp": datetime.now().isoformat(),
                "processed_video_url": stream_url
            }
            
            sessions.update_session(session_id, update_data)
            
            print(f"✅ Video synced to Cloudflare Stream: {cloudflare_video.get('uid')}")
            
            return jsonify({
                "success": True,
                "message": "Video synced to Cloudflare Stream successfully",
                "session_id": session_id,
                "video": {
                    "id": cloudflare_video.get('uid'),
                    "uid": cloudflare_video.get('uid'),
                    "filename": video_filename,
                    "size": cloudflare_video.get('size'),
                    "duration": cloudflare_video.get('duration'),
                    "ready_to_stream": cloudflare_video.get('readyToStream', False),
                    "stream_url": stream_url,
                    "thumbnail": cloudflare_video.get('thumbnail'),
                    "created": cloudflare_video.get('created')
                }
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"❌ Error syncing video to stream: {e}")
        return jsonify({"error": str(e)}), 500

# Legacy endpoints for compatibility
@app.route('/getProcessedVideos', methods=['GET'])
def get_processed_videos():
    """Get processed videos (legacy endpoint)"""
    try:
        sessions_list = sessions.get_all_sessions()
        processed_videos = []
        
        for session in sessions_list:
            if session.get('processed_video_filename'):
                video_info = {
                    "filename": session['processed_video_filename'],
                    "athlete_name": session.get('athlete_name', 'Unknown'),
                    "event": session.get('event', 'Unknown'),
                    "date": session.get('date', ''),
                    "duration": session.get('duration', ''),
                    "has_cloudflare_stream": bool(session.get('meta', {}).get('cloudflare_uid')),
                    "cloudflare_stream_url": get_cloudflare_stream_url(session['meta']['cloudflare_uid']) if session.get('meta', {}).get('cloudflare_uid') else None
                }
                processed_videos.append(video_info)
        
        return jsonify({
            "processed_videos": processed_videos,
            "count": len(processed_videos)
        })
        
    except Exception as e:
        print(f"❌ Error getting processed videos: {e}")
        return jsonify({"error": str(e)}), 500

# Regular video upload endpoint (with Cloudflare Stream integration)
@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    """Upload video to Cloudflare Stream only (no GridFS)"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Get metadata from form data
        athlete_name = request.form.get('athlete_name', 'Unknown Athlete')
        event = request.form.get('event', 'Floor Exercise')
        session_name = request.form.get('session_name', f'{athlete_name} - {event}')
        user_id = request.form.get('user_id', 'demo_user')  # Get user_id from form data
        auto_analyze = request.form.get('auto_analyze', 'true').lower() == 'true'
        
        print(f"📤 Uploading video to Cloudflare Stream: {video_file.filename}")
        print(f"👤 Athlete: {athlete_name}")
        print(f"🏃 Event: {event}")
        print(f"📝 Session: {session_name}")
        print(f"🔄 Auto-analyze: {auto_analyze}")
        
        # Upload video to Cloudflare Stream
        cloudflare_info = None
        try:
            # Save file temporarily for Cloudflare upload
            temp_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{video_file.filename}")
            video_file.seek(0)  # Reset file pointer
            video_file.save(temp_path)
            
            # Upload to Cloudflare Stream
            metadata = {
                "name": video_file.filename,
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "upload_source": "gymnastics_api_server_updated2"
            }
            
            cloudflare_video = upload_to_cloudflare_stream(temp_path, metadata)
            
            if not cloudflare_video:
                raise Exception("Failed to upload to Cloudflare Stream")
            
            stream_url = get_cloudflare_stream_url(cloudflare_video.get('uid'))
            
            cloudflare_info = {
                "id": cloudflare_video.get('uid'),
                "uid": cloudflare_video.get('uid'),
                "stream_url": stream_url,
                "ready_to_stream": cloudflare_video.get('readyToStream', False),
                "size": cloudflare_video.get('size'),
                "duration": cloudflare_video.get('duration')
            }
            
            print(f"✅ Video uploaded to Cloudflare Stream: {cloudflare_video.get('uid')}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"❌ Cloudflare Stream upload failed: {e}")
            return jsonify({"error": f"Failed to upload to Cloudflare Stream: {str(e)}"}), 500
        
        # Create session record with Cloudflare Stream info only
        session_data = {
            "user_id": user_id,  # Use user_id from form data
            "athlete_name": athlete_name,
            "session_name": session_name,
            "event": event,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "duration": "00:00",  # Will be updated after processing
            "original_filename": video_file.filename,
            "processed_video_filename": video_file.filename,
            "processed_video_url": stream_url,
            "status": "uploaded",
            "processing_status": "pending",
            "is_binary_stored": False,  # No GridFS storage
            "gridfs_video_id": None,  # No GridFS ID
            "created_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "updated_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "meta": {
                "cloudflare_stream_id": cloudflare_video.get('uid'),
                "cloudflare_uid": cloudflare_video.get('uid'),
                "upload_source": "cloudflare_stream",
                "upload_timestamp": datetime.now().isoformat(),
                "video_size": cloudflare_video.get('size'),
                "video_duration": cloudflare_video.get('duration'),
                "ready_to_stream": cloudflare_video.get('readyToStream', False),
                "stream_url": stream_url,
                "thumbnail": cloudflare_video.get('thumbnail')
            }
        }
        
        # Save session to MongoDB
        session_id = sessions.create_session(session_data)
        
        print(f"✅ Session created: {session_id}")
        
        response_data = {
            "success": True,
            "message": "Video uploaded successfully to Cloudflare Stream",
            "session_id": session_id,
            "video_id": cloudflare_video.get('uid'),  # Use Cloudflare UID as video ID
            "filename": video_file.filename,
            "size_mb": round(cloudflare_video.get('size', 0) / (1024 * 1024), 2),
            "gridfs_stored": False,
            "cloudflare_stored": True,
            "cloudflare": cloudflare_info
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error uploading video: {e}")
        return jsonify({"error": str(e)}), 500

# Analytics endpoints (keeping from original server)
@app.route('/getAnalytics/<analytics_id>', methods=['GET'])
def get_analytics(analytics_id):
    """Get analytics data from MongoDB GridFS"""
    try:
        print(f"🔍 Getting analytics for ID: {analytics_id}")
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        # Convert timestamps to relative time for video synchronization
        processed_analytics = convert_timestamps_to_relative(analytics_data)
        
        return jsonify({
            "analytics_id": analytics_id,
            "analytics": processed_analytics,
            "metadata": {
                "total_frames": len(processed_analytics) if isinstance(processed_analytics, list) else 
                               len(processed_analytics.get('frame_data', [])) if isinstance(processed_analytics, dict) else 0,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting analytics: {e}")
        return jsonify({
            "error": "Failed to retrieve analytics",
            "details": str(e)
        }), 500

@app.route('/getPerFrameStatistics', methods=['GET'])
def get_per_frame_statistics():
    """Get per-frame statistics by video filename"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        print(f"🔍 Getting per-frame statistics for video: {video_filename}")
        
        # Try to find analytics data for this video filename
        # First, check if we have a session with this video
        sessions_collection = db_manager.db.sessions
        session = sessions_collection.find_one({"video_filename": video_filename})
        
        if session and session.get('analytics_id'):
            # Get analytics from GridFS using the analytics_id
            analytics_id = session['analytics_id']
            print(f"📊 Found session with analytics_id: {analytics_id}")
            
            analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
            if analytics_data:
                # Check if the analytics data has the expected structure
                if isinstance(analytics_data, dict) and 'frame_data' in analytics_data:
                    # Convert timestamps to relative time for video synchronization
                    processed_data = convert_timestamps_to_relative(analytics_data)
                    return jsonify(processed_data)
                elif isinstance(analytics_data, list):
                    # Convert list format to expected format and fix timestamps
                    processed_data = convert_timestamps_to_relative({
                        "frame_data": analytics_data,
                        "enhanced_statistics": {},
                        "analytics_id": str(analytics_id)
                    })
                    return jsonify(processed_data)
                else:
                    print(f"⚠️ Unexpected analytics data format for {analytics_id}")
        
        # If no session found, try to find analytics files in GridFS by filename
        print(f"🔍 Searching GridFS for analytics files matching: {video_filename}")
        files_collection = db_manager.db.fs.files
        
        # Try different filename patterns
        search_patterns = [
            video_filename,
            video_filename.replace('.mp4', ''),
            f"fixed_analytics_{video_filename}",
            f"analytics_{video_filename.replace('.mp4', '')}.json"
        ]
        
        for pattern in search_patterns:
            print(f"🔍 Searching for pattern: {pattern}")
            analytics_file = files_collection.find_one({"filename": {"$regex": pattern, "$options": "i"}})
            
            if analytics_file:
                print(f"✅ Found analytics file: {analytics_file['filename']}")
                analytics_id = analytics_file['_id']
                
                analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
                if analytics_data:
                    if isinstance(analytics_data, dict) and 'frame_data' in analytics_data:
                        # Convert timestamps to relative time for video synchronization
                        processed_data = convert_timestamps_to_relative(analytics_data)
                        return jsonify(processed_data)
                    elif isinstance(analytics_data, list):
                        # Convert list format to expected format and fix timestamps
                        processed_data = convert_timestamps_to_relative({
                            "frame_data": analytics_data,
                            "enhanced_statistics": {},
                            "analytics_id": str(analytics_id)
                        })
                        return jsonify(processed_data)
        
        # If no analytics found, return empty response
        print(f"❌ No analytics found for video: {video_filename}")
        return jsonify({
            "error": f"No analytics found for video: {video_filename}",
            "frame_data": [],
            "enhanced_statistics": {}
        }), 404
        
    except Exception as e:
        print(f"❌ Error getting per-frame statistics: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/analyzeVideo1', methods=['POST'])
def analyze_video_from_cloudflare_or_gridfs():
    """Analyze video from Cloudflare Stream (priority) or GridFS by downloading it locally first, then processing"""
    try:
        data = request.get_json()
        
        # Support both old and new parameter formats
        session_id = data.get('session_id')
        cloudflare_stream_id = data.get('cloudflare_stream_id')
        video_filename = data.get('video_filename')
        
        # If session_id is provided, use that to find the session
        if session_id:
            print(f"🔍 Analyzing video for session: {session_id}")
            session = sessions.get_session(session_id)
            if not session:
                print(f"❌ Session not found: {session_id}")
                return jsonify({"error": f"Session not found: {session_id}"}), 404
            
            print(f"✅ Session found: {session.get('_id')}")
            print(f"📊 Session meta: {session.get('meta', {})}")
            
            # Extract video filename from session
            video_filename = session.get('processed_video_filename') or session.get('original_filename')
            if not video_filename:
                print(f"❌ No video filename found in session")
                return jsonify({"error": "No video filename found in session"}), 400
                
            print(f"🔍 Found video filename in session: {video_filename}")
        elif not video_filename:
            return jsonify({"error": "Either session_id or video_filename is required"}), 400
        
        # Legacy parameter support
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        user_id = data.get('user_id', 'demo_user')
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        print(f"🔍 Looking for video: {video_filename}")
        
        # Check if Railway MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }), 503
        
        # First, try to find the session to check for Cloudflare Stream or GridFS video ID
        if session_id:
            # Use the session we already found
            session = sessions.get_session(session_id)
        else:
            # Fallback to finding by video filename
            session = sessions.get_session_by_video_filename(video_filename)
        
        video_path = None
        
        if session:
            # Priority 1: Check for Cloudflare Stream video
            cloudflare_uid = None
            if cloudflare_stream_id:
                # Use the provided Cloudflare Stream ID
                cloudflare_uid = cloudflare_stream_id
                print(f"🌊 Using provided Cloudflare Stream ID: {cloudflare_uid}")
            elif session.get('meta') and session['meta'].get('cloudflare_stream_id'):
                cloudflare_uid = session['meta']['cloudflare_stream_id']
            elif session.get('meta') and session['meta'].get('cloudflare_uid'):
                cloudflare_uid = session['meta']['cloudflare_uid']
            
            if cloudflare_uid:
                print(f"🌊 Found session with Cloudflare Stream UID: {cloudflare_uid}")
                
                # Check if video is ready to stream first
                try:
                    import requests
                    headers = {
                        'Authorization': f'Bearer {CLOUDFLARE_STREAM_TOKEN}',
                        'Content-Type': 'application/json'
                    }
                    
                    # Get video details to check if it's ready
                    video_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_STREAM_ACCOUNT_ID}/stream/{cloudflare_uid}"
                    response = requests.get(video_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        video_data = response.json()
                        if video_data.get('success'):
                            video_info = video_data['result']
                            ready_to_stream = video_info.get('readyToStream', False)
                            print(f"🌊 Video ready status: {ready_to_stream}")
                            
                            if not ready_to_stream:
                                return jsonify({
                                    "error": "Video is still processing. Please wait a few minutes and try again.",
                                    "message": "Cloudflare Stream is still processing your video. This usually takes 2-5 minutes for videos of this size.",
                                    "video_ready": False,
                                    "cloudflare_uid": cloudflare_uid
                                }), 202  # 202 = Accepted but processing
                    else:
                        print(f"⚠️ Could not check video status: {response.status_code}")
                        
                except Exception as e:
                    print(f"⚠️ Error checking video status: {e}")
                    # Continue with download attempt anyway
                
                try:
                    # Download video from Cloudflare Stream to local temp file
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{video_filename}")
                    
                    if download_video_from_cloudflare_stream(cloudflare_uid, temp_video_path):
                        video_path = temp_video_path
                        print(f"✅ Downloaded video from Cloudflare Stream to: {video_path}")
                    else:
                        print(f"⚠️ Failed to download from Cloudflare Stream, falling back to GridFS")
                        cloudflare_uid = None  # Reset to try GridFS
                        
                except Exception as cloudflare_error:
                    print(f"⚠️ Cloudflare Stream download failed: {cloudflare_error}")
                    cloudflare_uid = None  # Reset to try GridFS
            
            # Priority 2: Fall back to GridFS if Cloudflare Stream failed or not available
            if not video_path and session.get('gridfs_video_id'):
                print(f"🔍 Falling back to GridFS video ID: {session['gridfs_video_id']}")
                try:
                    from bson import ObjectId
                    gridfs_file = fs.get(ObjectId(session['gridfs_video_id']))
                    print(f"✅ Found video in GridFS by session ID: {gridfs_file.filename}")
                    
                    # Download video from GridFS to local temp file
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{gridfs_file.filename}")
                    with open(temp_video_path, 'wb') as temp_file:
                        temp_file.write(gridfs_file.read())
                    
                    video_path = temp_video_path
                    print(f"📥 Downloaded original video from GridFS to: {video_path}")
                except Exception as gridfs_error:
                    print(f"⚠️ GridFS lookup by session ID failed: {gridfs_error}")
                    gridfs_file = None
        
        if not video_path:
            # Fallback: try to find video in GridFS by filename
            print(f"🔍 No session found, searching GridFS for: {video_filename}")
            
            # Search for video in GridFS
            gridfs_file = None
            try:
                # Try to find by exact filename first
                gridfs_file = fs.find_one({"filename": video_filename})
                
                # If not found, try to find by partial match
                if not gridfs_file:
                    base_name = os.path.splitext(video_filename)[0]
                    for file_doc in fs.find({"filename": {"$regex": base_name, "$options": "i"}}):
                        if file_doc.filename.endswith('.mp4'):
                            gridfs_file = file_doc
                            break
                
                if gridfs_file:
                    print(f"✅ Found video in GridFS: {gridfs_file.filename}")
                
                    # Download video from GridFS to temporary local file
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{gridfs_file.filename}")
                    with open(temp_video_path, 'wb') as temp_file:
                        temp_file.write(gridfs_file.read())
                    
                    video_path = temp_video_path
                    print(f"📥 Downloaded video from GridFS to: {video_path}")
                    print(f"✅ Video downloaded successfully: {os.path.getsize(temp_video_path)} bytes")
                
            except Exception as e:
                print(f"❌ Error accessing GridFS: {e}")
                return jsonify({"error": f"Failed to access video in GridFS: {str(e)}"}), 500
        
        # Final fallback: if no video found in GridFS, try local search (but warn about it)
        if not video_path:
            print(f"⚠️ No video found in GridFS, trying local search as last resort: {video_filename}")
            best_video_path, is_h264, original_path = video_processor.find_best_video(video_filename)
            
            if best_video_path and os.path.exists(best_video_path):
                print(f"⚠️ Found local video (this may be a processed video): {best_video_path}")
                video_path = best_video_path
            else:
                return jsonify({"error": f"Video file not found in GridFS or locally: {video_filename}"}), 404
        
        # Process the video (same as original analyzeVideo)
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        # Generate unique output name
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"h264_analyzed_{base_name}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Check if analytics already exist for this video
        analytics_filename = f"fixed_analytics_{video_filename}.json"
        analytics_path = os.path.join(".", analytics_filename)
        
        if not os.path.exists(analytics_path):
            analytics_path = os.path.join(OUTPUT_DIR, analytics_filename)
        
        # Check if we can reuse existing processed video
        existing_videos = []
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                if f"10 sec" in file and file.endswith('.mp4') and 'h264' in file:
                    existing_videos.append(os.path.join(OUTPUT_DIR, file))
        
        if existing_videos and os.path.exists(analytics_path):
            # Use existing processed video and analytics
            actual_output_path = existing_videos[-1]  # Use the most recent one
            print(f"🚀 Using existing processed video: {actual_output_path}")
            print(f"🚀 Using existing analytics: {analytics_path}")
        else:
            # Process video with analytics (SYNC for now to ensure file exists)
            print(f"🔄 Processing video with MediaPipe...")
            result = video_processor.process_video_with_analytics(video_path, output_name)
            
            if not result or not result.get("success", False):
                return jsonify({"error": "Video processing failed"}), 500
            
            # Get the actual output paths from the result
            actual_output_path = result.get("h264_video") or result.get("output_video")
            if not actual_output_path or not os.path.exists(actual_output_path):
                return jsonify({"error": "Processed video file not found"}), 500
            
            print(f"✅ Video processing completed: {actual_output_path}")
        
        # Upload processed video and analytics to GridFS
        # The analytics file is created with the original filename, not the temp filename
        analytics_filename = f"fixed_analytics_{video_filename}.json"
        analytics_path = os.path.join(".", analytics_filename)
        
        if not os.path.exists(analytics_path):
            analytics_path = os.path.join(OUTPUT_DIR, analytics_filename)
        
        # If still not found, try with the temp filename as fallback
        if not os.path.exists(analytics_path):
            temp_analytics_filename = f"fixed_analytics_{os.path.basename(video_path)}.json"
            temp_analytics_path = os.path.join(".", temp_analytics_filename)
            if os.path.exists(temp_analytics_path):
                analytics_path = temp_analytics_path
                analytics_filename = temp_analytics_filename
        
        print(f"📤 Uploading processed video and analytics to GridFS...")
        print(f"📊 Video file path: {actual_output_path}")
        print(f"📊 Video file exists: {os.path.exists(actual_output_path)}")
        print(f"📊 Analytics file path: {analytics_path}")
        print(f"📊 Analytics file exists: {os.path.exists(analytics_path)}")
        if os.path.exists(analytics_path):
            file_size = os.path.getsize(analytics_path)
            print(f"📊 Analytics file size: {file_size} bytes")
        
        video_id, analytics_id = video_processor.upload_processed_video_to_gridfs(
            actual_output_path, analytics_path, {
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "user_id": user_id,
                "date": date,
                "original_filename": video_filename,
                "processed_video_filename": output_name,
                "processing_timestamp": timestamp
            }
        )
        
        if not video_id:
            return jsonify({"error": "Failed to upload processed video to MongoDB"}), 500
        
        # Check if original video was from Cloudflare Stream and upload processed video there too
        cloudflare_processed_video = None
        processed_video_url = f"http://localhost:5004/getVideo?video_filename={os.path.basename(actual_output_path)}"
        
        if session and session.get('meta') and (session['meta'].get('cloudflare_stream_id') or session['meta'].get('cloudflare_uid')):
            print(f"🌊 Original video was from Cloudflare Stream, uploading processed video...")
            try:
                # Upload processed video to Cloudflare Stream
                processed_metadata = {
                    "name": f"analyzed_{os.path.basename(actual_output_path)}",
                    "description": f"Analyzed video for {athlete_name} - {event}",
                    "athlete_name": athlete_name,
                    "event": event,
                    "session_name": session_name,
                    "analysis_type": "pose_detection",
                    "processing_timestamp": timestamp
                }
                
                cloudflare_processed_video = upload_to_cloudflare_stream(actual_output_path, processed_metadata)
                if cloudflare_processed_video:
                    processed_video_url = get_cloudflare_stream_url(cloudflare_processed_video.get('uid'))
                    print(f"✅ Processed video uploaded to Cloudflare Stream: {cloudflare_processed_video.get('uid')}")
                else:
                    print(f"⚠️ Failed to upload processed video to Cloudflare Stream, using GridFS URL")
            except Exception as e:
                print(f"⚠️ Error uploading processed video to Cloudflare Stream: {e}")

        # Create session record
        session_data = {
            "user_id": user_id,
            "athlete_name": athlete_name,
            "session_name": session_name,
            "event": event,
            "date": date,
            "duration": "00:00",
            "original_filename": video_filename,
            "processed_video_filename": os.path.basename(actual_output_path),
            "processed_video_url": processed_video_url,
            "analytics_filename": analytics_filename if analytics_id else None,
            "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}" if analytics_id else None,
            "motion_iq": 0.0,
            "acl_risk": 0.0,
            "precision": 0.0,
            "power": 0.0,
            "tumbling_percentage": 0.0,
            "landmark_confidence": 0.0,
            "total_frames": 0,
            "fps": 0.0,
            "highlights": [],
            "areas_for_improvement": [],
            "coach_notes": "",
            "notes": f"Video analyzed with Railway MediaPipe server: {video_filename}",
            "status": "completed",
            "processing_progress": 1.0,
            "processing_status": "completed",
            "gridfs_video_id": video_id,
            "gridfs_analytics_id": analytics_id,
            "is_binary_stored": True,
            "has_landmarks": True,
            "created_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "updated_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
        
        # Add Cloudflare Stream metadata if processed video was uploaded there
        if cloudflare_processed_video:
            # Preserve original Cloudflare Stream metadata and add analyzed video info
            original_meta = session.get('meta', {}) if session else {}
            
            session_data["meta"] = {
                # Original video Cloudflare Stream info
                "cloudflare_stream_id": original_meta.get('cloudflare_stream_id'),
                "cloudflare_uid": original_meta.get('cloudflare_uid'),
                "upload_source": original_meta.get('upload_source', 'cloudflare_stream'),
                "upload_timestamp": original_meta.get('upload_timestamp'),
                "video_size": original_meta.get('video_size'),
                "video_duration": original_meta.get('video_duration'),
                "ready_to_stream": original_meta.get('ready_to_stream', False),
                "stream_url": original_meta.get('stream_url'),
                "thumbnail": original_meta.get('thumbnail'),
                
                # Analyzed video Cloudflare Stream info
                "analyzed_cloudflare_stream_id": cloudflare_processed_video.get('uid'),
                "analyzed_cloudflare_uid": cloudflare_processed_video.get('uid'),
                "analyzed_upload_timestamp": datetime.now().isoformat(),
                "analyzed_video_size": cloudflare_processed_video.get('size'),
                "analyzed_video_duration": cloudflare_processed_video.get('duration'),
                "analyzed_ready_to_stream": cloudflare_processed_video.get('readyToStream', False),
                "analyzed_stream_url": processed_video_url,
                "analyzed_thumbnail": cloudflare_processed_video.get('thumbnail'),
                "description": f"Analyzed video for {athlete_name} - {event}",
                "analysis_type": "pose_detection"
            }
        
        # Check if we should update existing session or create new one
        existing_session_id = None
        if session and session.get('_id'):
            existing_session_id = str(session['_id'])
            print(f"🔄 Updating existing session: {existing_session_id}")
            sessions.update_session(existing_session_id, session_data)
            session_id = existing_session_id
        else:
            print(f"🆕 Creating new session")
            session_id = sessions.create_session(session_data)
        
        print(f"✅ Session processed: {session_id}")
        
        # Clean up temporary file if it was downloaded from GridFS
        if video_path.startswith("/tmp") or video_path.startswith(".") and "temp_" in video_path:
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up temporary file: {video_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file: {e}")
        
        response_data = {
            "success": True,
            "message": "Video analysis completed and uploaded to MongoDB",
            "session_id": str(session_id),
            "video_id": str(video_id),
            "analytics_id": str(analytics_id) if analytics_id else None,
            "output_video": os.path.basename(actual_output_path),
            "analytics_file": analytics_filename if analytics_id else None,
            "download_url": f"http://localhost:5004/getVideo?video_filename={os.path.basename(actual_output_path)}",
            "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}" if analytics_id else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add Cloudflare Stream information if processed video was uploaded there
        if cloudflare_processed_video:
            response_data["cloudflare_stream"] = {
                "video_id": cloudflare_processed_video.get('uid'),
                "stream_url": processed_video_url,
                "size": cloudflare_processed_video.get('size'),
                "duration": cloudflare_processed_video.get('duration'),
                "ready_to_stream": cloudflare_processed_video.get('readyToStream', False),
                "thumbnail": cloudflare_processed_video.get('thumbnail')
            }
            response_data["message"] += " and uploaded to Cloudflare Stream"
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in analyzeVideo1: {e}")
        return jsonify({
            "error": "Video analysis failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Gymnastics API Server with Cloudflare Stream Integration")
    print(f"🌊 Cloudflare Stream Account ID: {CLOUDFLARE_ACCOUNT_ID}")
    print(f"🌊 Cloudflare Stream Domain: {CLOUDFLARE_STREAM_DOMAIN}")
    print(f"📊 MongoDB Database: {db_manager.db.name}")
    print(f"🔧 Max Workers: {MAX_WORKERS}")
    print(f"💾 Memory Threshold: {MEMORY_THRESHOLD}%")
    
    # Test Cloudflare Stream connection
    try:
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        response = requests.get(f"{CLOUDFLARE_STREAM_BASE_URL}?limit=1", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Cloudflare Stream connection successful")
        else:
            print(f"⚠️ Cloudflare Stream connection issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Cloudflare Stream connection error: {e}")
    
    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)
