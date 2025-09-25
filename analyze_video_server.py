#!/usr/bin/env python3
"""
Dedicated Video Analysis Server
Handles only the /analyzeVideo1 endpoint to prevent worker timeouts
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess
import os
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
     origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8080", "https://motionlabsai-qb7r-5ljq6f0oj-hemchandeishagmailcoms-projects.vercel.app", "https://www.motionlabsai.com"], 
     allow_headers=["Content-Type", "Authorization", "Range", "X-Requested-With", "Accept", "Origin"],
     expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
     supports_credentials=True)

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
MAX_WORKERS = 2  # Reduced for dedicated analysis server
CHUNK_SIZE = 32 * 1024
MEMORY_THRESHOLD = 80
REQUEST_TIMEOUT = 60  # Increased timeout for analysis
CACHE_SIZE = 50
MAX_ANALYTICS_FRAMES = 50
VIDEO_STREAM_TIMEOUT = 120  # Increased for analysis
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200MB max for analysis

# Initialize GridFS
fs = None
if db_manager.db is not None:
    fs = gridfs.GridFS(db_manager.db)
    print("✅ GridFS initialized for analysis server")
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
    
    def process_video_with_analytics(self, video_path, output_name=None):
        """Process video using the fixed overlay system and convert to H.264"""
        if not output_name:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"processed_{base_name}.mp4"
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        analytics_path = f"fixed_analytics_{os.path.basename(video_path)}.json"
        
        # Command to run the fixed video overlay script
        cmd = [
            "python3", "fixed_video_overlay_with_analytics_enhanced.py",
            video_path,
            "--output", output_path,
            "--server", MEDIAPIPE_SERVER_URL
        ]
        
        try:
            print(f"🎬 Starting video processing: {os.path.basename(video_path)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✅ Video overlay processing completed: {output_name}")
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 Processed video size: {file_size:.1f}MB")
                
                return {
                    "success": True,
                    "output_video": output_path,
                    "h264_video": output_path,
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
        except subprocess.TimeoutExpired:
            print(f"❌ Video processing timeout")
            return {
                "success": False,
                "error": "Processing timeout",
                "message": "Video processing timed out"
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

# Initialize video processor
video_processor = MongoDBVideoProcessor()

# Cloudflare Stream Helper Functions
def upload_to_cloudflare_stream(video_path, metadata=None):
    """Upload a video to Cloudflare Stream"""
    try:
        print(f"🌊 Uploading video to Cloudflare Stream: {video_path}")
        
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        files = {'file': open(video_path, 'rb')}
        data = {}
        if metadata:
            data['meta'] = json.dumps(metadata)
        
        response = requests.post(
            CLOUDFLARE_STREAM_BASE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300
        )
        
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                video_data = result.get('result', {})
                print(f"✅ Video uploaded to Cloudflare Stream successfully")
                return video_data
            
        print(f"❌ Cloudflare Stream upload failed")
        return None
            
    except Exception as e:
        print(f"❌ Error uploading to Cloudflare Stream: {e}")
        return None

def get_cloudflare_stream_url(video_uid):
    """Get the streaming URL for a Cloudflare Stream video"""
    return f"https://{CLOUDFLARE_STREAM_DOMAIN}/{video_uid}/iframe"

def download_video_from_cloudflare_stream(video_uid, output_path):
    """Download a video from Cloudflare Stream to local file"""
    try:
        print(f"🌊 Downloading video from Cloudflare Stream: {video_uid}")
        
        # Enable MP4 download
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        response = requests.post(
            f"{CLOUDFLARE_STREAM_BASE_URL}/{video_uid}/downloads",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            # Poll for download readiness
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(10)
                status_response = requests.get(
                    f"{CLOUDFLARE_STREAM_BASE_URL}/{video_uid}/downloads",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    if status_result.get('success'):
                        download_info = status_result.get('result', {}).get('default', {})
                        if download_info.get('status') == 'ready':
                            download_url = download_info.get('url')
                            break
                
            if download_url:
                # Download the video
                response = requests.get(download_url, headers=headers, stream=True, timeout=300)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Video downloaded successfully: {output_path} ({file_size} bytes)")
                    return True
                    
        return False
        
    except Exception as e:
        print(f"❌ Error downloading video from Cloudflare Stream: {e}")
        return False

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db_manager.client.admin.command('ping')
        mongodb_status = "connected"
    except Exception as e:
        mongodb_status = f"error: {str(e)}"
    
    try:
        response = requests.get(f"{MEDIAPIPE_SERVER_URL}/health", timeout=5)
        mediapipe_status = "running" if response.status_code == 200 else "error"
    except Exception:
        mediapipe_status = "error"
    
    return jsonify({
        "status": "healthy",
        "service": "video_analysis_server",
        "mongodb": mongodb_status,
        "mediapipe_server": mediapipe_status,
        "timestamp": datetime.now().isoformat()
    })

# Main Analysis Endpoint
@app.route('/analyzeVideo1', methods=['POST'])
def analyze_video():
    """Analyze video from Cloudflare Stream or GridFS"""
    try:
        data = request.get_json()
        
        # Support both old and new parameter formats
        session_id = data.get('session_id')
        cloudflare_stream_id = data.get('cloudflare_stream_id')
        video_filename = data.get('video_filename')
        
        if session_id:
            print(f"🔍 Analyzing video for session: {session_id}")
            session = sessions.get_session(session_id)
            if not session:
                print(f"❌ Session not found: {session_id}")
                return jsonify({"error": f"Session not found: {session_id}"}), 404
            
            print(f"✅ Session found: {session.get('_id')}")
            video_filename = session.get('processed_video_filename') or session.get('original_filename')
            if not video_filename:
                return jsonify({"error": "No video filename found in session"}), 400
        elif not video_filename:
            return jsonify({"error": "Either session_id or video_filename is required"}), 400
        
        # Legacy parameter support
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        user_id = data.get('user_id', 'demo_user')
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        print(f"🔍 Looking for video: {video_filename}")
        
        # Check if MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }), 503
        
        # Find the session to check for Cloudflare Stream or GridFS video ID
        if session_id:
            session = sessions.get_session(session_id)
        else:
            session = sessions.get_session_by_video_filename(video_filename)
        
        video_path = None
        
        if session:
            # Priority 1: Check for Cloudflare Stream video
            cloudflare_uid = None
            if cloudflare_stream_id:
                cloudflare_uid = cloudflare_stream_id
                print(f"🌊 Using provided Cloudflare Stream ID: {cloudflare_uid}")
            elif session.get('meta') and session['meta'].get('cloudflare_stream_id'):
                cloudflare_uid = session['meta']['cloudflare_stream_id']
            elif session.get('meta') and session['meta'].get('cloudflare_uid'):
                cloudflare_uid = session['meta']['cloudflare_uid']
            
            if cloudflare_uid:
                print(f"🌊 Found session with Cloudflare Stream UID: {cloudflare_uid}")
                
                try:
                    # Download video from Cloudflare Stream to local temp file
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{video_filename}")
                    
                    if download_video_from_cloudflare_stream(cloudflare_uid, temp_video_path):
                        video_path = temp_video_path
                        print(f"✅ Downloaded video from Cloudflare Stream to: {video_path}")
                    else:
                        print(f"⚠️ Failed to download from Cloudflare Stream, falling back to GridFS")
                        cloudflare_uid = None
                        
                except Exception as cloudflare_error:
                    print(f"⚠️ Cloudflare Stream download failed: {cloudflare_error}")
                    cloudflare_uid = None
            
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
        
        if not video_path:
            # Fallback: try to find video in GridFS by filename
            print(f"🔍 No session found, searching GridFS for: {video_filename}")
            
            try:
                gridfs_file = fs.find_one({"filename": video_filename})
                
                if not gridfs_file:
                    base_name = os.path.splitext(video_filename)[0]
                    for file_doc in fs.find({"filename": {"$regex": base_name, "$options": "i"}}):
                        if file_doc.filename.endswith('.mp4'):
                            gridfs_file = file_doc
                            break
                
                if gridfs_file:
                    print(f"✅ Found video in GridFS: {gridfs_file.filename}")
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{gridfs_file.filename}")
                    with open(temp_video_path, 'wb') as temp_file:
                        temp_file.write(gridfs_file.read())
                    
                    video_path = temp_video_path
                    print(f"📥 Downloaded video from GridFS to: {video_path}")
                
            except Exception as e:
                print(f"❌ Error accessing GridFS: {e}")
                return jsonify({"error": f"Failed to access video in GridFS: {str(e)}"}), 500
        
        if not video_path:
            return jsonify({"error": f"Video file not found: {video_filename}"}), 404
        
        # Process the video
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        # Generate unique output name
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"h264_analyzed_{base_name}_{timestamp}.mp4"
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process video with analytics
        print(f"🔄 Processing video with MediaPipe...")
        result = video_processor.process_video_with_analytics(video_path, output_name)
        
        if not result or not result.get("success", False):
            # Clean up temp file
            if video_path.startswith(TEMP_DIR):
                try:
                    os.remove(video_path)
                except:
                    pass
            return jsonify({"error": "Video processing failed", "details": result.get("error", "Unknown error")}), 500
        
        # Get the actual output paths from the result
        actual_output_path = result.get("h264_video") or result.get("output_video")
        if not actual_output_path or not os.path.exists(actual_output_path):
            return jsonify({"error": "Processed video file not found"}), 500
        
        print(f"✅ Video processing completed: {actual_output_path}")
        
        # Upload processed video and analytics to GridFS
        analytics_filename = f"fixed_analytics_{video_filename}.json"
        analytics_path = os.path.join(".", analytics_filename)
        
        if not os.path.exists(analytics_path):
            analytics_path = os.path.join(OUTPUT_DIR, analytics_filename)
        
        print(f"📤 Uploading processed video and analytics to GridFS...")
        
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

        # Create or update session record
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
            original_meta = session.get('meta', {}) if session else {}
            
            session_data["meta"] = {
                "cloudflare_stream_id": original_meta.get('cloudflare_stream_id'),
                "cloudflare_uid": original_meta.get('cloudflare_uid'),
                "upload_source": original_meta.get('upload_source', 'cloudflare_stream'),
                "upload_timestamp": original_meta.get('upload_timestamp'),
                "video_size": original_meta.get('video_size'),
                "video_duration": original_meta.get('video_duration'),
                "ready_to_stream": original_meta.get('ready_to_stream', False),
                "stream_url": original_meta.get('stream_url'),
                "thumbnail": original_meta.get('thumbnail'),
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
        
        # Update existing session or create new one
        if session and session.get('_id'):
            existing_session_id = str(session['_id'])
            print(f"🔄 Updating existing session: {existing_session_id}")
            sessions.update_session(existing_session_id, session_data)
            session_id = existing_session_id
        else:
            print(f"🆕 Creating new session")
            session_id = sessions.create_session(session_data)
        
        print(f"✅ Session processed: {session_id}")
        
        # Clean up temporary file
        if video_path.startswith(TEMP_DIR):
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up temporary file: {video_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file: {e}")
        
        response_data = {
            "success": True,
            "message": "Video analysis completed",
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
    print("🚀 Starting Dedicated Video Analysis Server")
    print(f"🌊 Cloudflare Stream Account ID: {CLOUDFLARE_ACCOUNT_ID}")
    print(f"📊 MongoDB Database: {db_manager.db.name}")
    print(f"🔧 Max Workers: {MAX_WORKERS}")
    print(f"⏱️ Request Timeout: {REQUEST_TIMEOUT}s")
    
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)
