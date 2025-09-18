#!/usr/bin/env python3
"""
Updated Gymnastics API Server with MongoDB Integration
This server reads analytics files and videos from MongoDB GridFS and provides API endpoints
"""

from flask import Flask, request, jsonify, send_file, Response, stream_template
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
import io
import gridfs
from bson import ObjectId
import re

# Import database modules
from database import db_manager, sessions, users, video_metadata

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8080", "https://motionlabsai-qb7r-5ljq6f0oj-hemchandeishagmailcoms-projects.vercel.app","https://www.motionlabsai.com"], 
     allow_headers=["Content-Type", "Authorization", "Range"],
     expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])

# Configuration
MEDIAPIPE_SERVER_URL = "https://extraordinary-gentleness-production.up.railway.app"
VIDEO_PROCESSING_DIR = "../output_videos"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_videos")
ANALYTICS_DIR = "../analytics"
TEMP_DIR = "/tmp" if os.path.exists("/tmp") else "."

# Initialize GridFS
fs = gridfs.GridFS(db_manager.db)

def parse_range_header(range_header, file_size):
    """Parse HTTP Range header and return start, end positions"""
    if not range_header:
        return 0, file_size - 1
    
    # Parse range header (e.g., "bytes=0-1023" or "bytes=1024-")
    match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not match:
        return 0, file_size - 1
    
    start = int(match.group(1))
    end_str = match.group(2)
    
    if end_str:
        end = int(end_str)
    else:
        end = file_size - 1
    
    # Ensure valid range
    start = max(0, start)
    end = min(end, file_size - 1)
    
    if start > end:
        return 0, file_size - 1
    
    return start, end

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
    
    def get_video_from_gridfs(self, video_id):
        """Retrieve video file from GridFS"""
        try:
            if isinstance(video_id, str):
                video_id = ObjectId(video_id)
            
            grid_out = self.fs.get(video_id)
            return grid_out.read()
        except Exception as e:
            print(f"❌ Error retrieving video from GridFS: {e}")
            return None
    
    def get_video_info_from_gridfs(self, video_id):
        """Get video file metadata from GridFS"""
        try:
            if isinstance(video_id, str):
                video_id = ObjectId(video_id)
            
            video_file = self.fs.get(video_id)
            return {
                'filename': video_file.filename,
                'length': video_file.length,
                'content_type': video_file.content_type,
                'upload_date': video_file.upload_date
            }
        except Exception as e:
            print(f"❌ Error retrieving video info from GridFS: {e}")
            return None
    
    def stream_video_from_gridfs(self, video_id, start=0, end=None):
        """Stream video file from GridFS in chunks"""
        try:
            if isinstance(video_id, str):
                video_id = ObjectId(video_id)
            
            grid_out = self.fs.get(video_id)
            total_size = grid_out.length
            
            # Set default end if not provided
            if end is None:
                end = total_size - 1
            
            # Ensure end doesn't exceed file size
            end = min(end, total_size - 1)
            
            # Seek to start position
            grid_out.seek(start)
            
            # Calculate chunk size (1MB chunks for streaming)
            chunk_size = 1024 * 1024  # 1MB
            
            def generate():
                remaining = end - start + 1
                current_pos = start
                
                while remaining > 0:
                    # Read chunk
                    read_size = min(chunk_size, remaining)
                    chunk = grid_out.read(read_size)
                    
                    if not chunk:
                        break
                    
                    yield chunk
                    
                    remaining -= len(chunk)
                    current_pos += len(chunk)
            
            return generate(), total_size, start, end
            
        except Exception as e:
            print(f"❌ Error streaming video from GridFS: {e}")
            return None, 0, 0, 0
    
    def get_analytics_from_gridfs(self, analytics_id):
        """Retrieve analytics file from GridFS"""
        try:
            if isinstance(analytics_id, str):
                analytics_id = ObjectId(analytics_id)
            
            grid_out = self.fs.get(analytics_id)
            return json.loads(grid_out.read().decode('utf-8'))
        except Exception as e:
            print(f"❌ Error retrieving analytics from GridFS: {e}")
            return None
    
    def save_video_to_temp(self, video_data, filename):
        """Save video data to temporary file"""
        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)
            
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            return temp_path, temp_dir
        except Exception as e:
            print(f"❌ Error saving video to temp: {e}")
            return None, None
    
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
            "python3", "video_overlay_with_analytics_fixed.py",
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
    
    def process_video_with_railway_mediapipe(self, video_path, video_filename, athlete_name, event, session_name, user_id, date):
        """Process video using Railway MediaPipe server and upload to GridFS"""
        try:
            print(f"🎬 Processing video: {video_filename}")
            
            # Check if MediaPipe server is available
            if not self.check_mediapipe_server():
                return {
                    "success": False,
                    "error": "MediaPipe server is not available",
                    "message": "Cannot process video without MediaPipe server"
                }
            
            # Generate output filename (clean naming, no h264 prefix)
            timestamp = int(time.time())
            base_name = os.path.splitext(video_filename)[0]
            # Remove existing h264_ prefix if present to avoid accumulation
            if base_name.startswith('h264_'):
                base_name = base_name[5:]  # Remove 'h264_' prefix
            output_name = f"analyzed_temp_{timestamp}_{base_name}_{timestamp}.mp4"
            output_path = os.path.join(TEMP_DIR, output_name)
            
            # Process video with MediaPipe
            print(f"🔄 Processing video with MediaPipe...")
            result = self.process_video_with_analytics(video_path, output_name)
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": "Video processing failed",
                    "message": result.get("error", "Unknown processing error")
                }
            
            # Upload processed video and analytics to GridFS
            analytics_path = result.get("analytics_file")
            metadata = {
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "user_id": user_id,
                "date": date,
                "original_filename": video_filename,
                "processed_video_filename": output_name
            }
            
            video_id, analytics_id = self.upload_processed_video_to_gridfs(
                output_path, analytics_path, metadata
            )
            
            if not video_id:
                return {
                    "success": False,
                    "error": "Failed to upload processed video to GridFS",
                    "message": "Video was processed but could not be stored"
                }
            
            # Create session record
            session_data = {
                "user_id": user_id,
                "athlete_name": athlete_name,
                "session_name": session_name,
                "event": event,
                "date": date,
                "duration": result.get("duration", "00:00"),
                "original_filename": video_filename,
                "processed_video_filename": output_name,
                "processed_video_url": f"http://localhost:5004/getVideo?video_filename={output_name}",
                "analytics_filename": os.path.basename(analytics_path) if analytics_path else None,
                "analytics_url": f"http://localhost:5004/getPerFrameStatistics?video_filename={output_name}",
                "motion_iq": result.get("motion_iq", 0.0),
                "acl_risk": result.get("acl_risk", 0.0),
                "precision": result.get("precision", 0.0),
                "power": result.get("power", 0.0),
                "tumbling_percentage": result.get("tumbling_percentage", 0.0),
                "status": "completed",
                "processing_progress": 100.0,
                "total_frames": result.get("total_frames", 0),
                "fps": result.get("fps", 0.0),
                "has_landmarks": result.get("has_landmarks", False),
                "landmark_confidence": result.get("landmark_confidence", 0.0),
                "notes": f"Video processed successfully: {video_filename}",
                "coach_notes": "",
                "highlights": result.get("highlights", []),
                "areas_for_improvement": result.get("areas_for_improvement", []),
                "gridfs_video_id": video_id,
                "gridfs_analytics_id": analytics_id,
                "is_binary_stored": True,
                "processing_status": "completed"
            }
            
            session_id = sessions.create_session(session_data)
            
            # Clean up temporary files
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                if analytics_path and os.path.exists(analytics_path):
                    os.remove(analytics_path)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not clean up temporary files: {cleanup_error}")
            
            return {
                "success": True,
                "message": "Video processed and uploaded successfully",
                "session_id": session_id,
                "video_id": video_id,
                "analytics_id": analytics_id,
                "output_video": output_name,
                "analytics_file": analytics_path,
                "download_url": f"http://localhost:5004/getVideo?video_filename={output_name}",
                "analytics_url": f"http://localhost:5004/getPerFrameStatistics?video_filename={output_name}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing video with Railway MediaPipe: {e}")
            return {
                "success": False,
                "error": "Video processing failed",
                "message": str(e)
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

# Initialize video processor
video_processor = MongoDBVideoProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mediapipe_status = video_processor.check_mediapipe_server()
    mongodb_status = True
    
    try:
        db_manager.client.admin.command('ping')
    except:
        mongodb_status = False
    
    return jsonify({
        "status": "healthy",
        "mediapipe_server": "running" if mediapipe_status else "down",
        "mongodb": "connected" if mongodb_status else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/getSessions', methods=['GET'])
def get_sessions():
    """Get all sessions from MongoDB"""
    try:
        sessions_list = sessions.get_all_sessions()
        
        # Add GridFS file information and convert ObjectIds to strings
        for session in sessions_list:
            # Convert ObjectIds to strings for JSON serialization
            if session.get('_id'):
                session['_id'] = str(session['_id'])
            if session.get('gridfs_video_id'):
                session['gridfs_video_id'] = str(session['gridfs_video_id'])
                try:
                    video_file = fs.get(ObjectId(session['gridfs_video_id']))
                    session['video_size'] = video_file.length
                    session['video_upload_date'] = video_file.upload_date.isoformat()
                except:
                    session['video_size'] = 0
                    session['video_upload_date'] = None
            
            if session.get('gridfs_analytics_id'):
                session['gridfs_analytics_id'] = str(session['gridfs_analytics_id'])
                try:
                    analytics_file = fs.get(ObjectId(session['gridfs_analytics_id']))
                    session['analytics_size'] = analytics_file.length
                    session['analytics_upload_date'] = analytics_file.upload_date.isoformat()
                except:
                    session['analytics_size'] = 0
                    session['analytics_upload_date'] = None
        
        return jsonify({
            "success": True,
            "sessions": sessions_list,
            "count": len(sessions_list),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return jsonify({
            "error": "Failed to retrieve sessions",
            "details": str(e)
        }), 500

@app.route('/getSession/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific session by ID"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Add GridFS file information and convert ObjectIds to strings
        if session.get('_id'):
            session['_id'] = str(session['_id'])
        if session.get('gridfs_video_id'):
            session['gridfs_video_id'] = str(session['gridfs_video_id'])
            try:
                video_file = fs.get(ObjectId(session['gridfs_video_id']))
                session['video_size'] = video_file.length
                session['video_upload_date'] = video_file.upload_date.isoformat()
                session['video_content_type'] = video_file.content_type
            except:
                session['video_size'] = 0
                session['video_upload_date'] = None
                session['video_content_type'] = None
        
        if session.get('gridfs_analytics_id'):
            session['gridfs_analytics_id'] = str(session['gridfs_analytics_id'])
            try:
                analytics_file = fs.get(ObjectId(session['gridfs_analytics_id']))
                session['analytics_size'] = analytics_file.length
                session['analytics_upload_date'] = analytics_file.upload_date.isoformat()
                session['analytics_content_type'] = analytics_file.content_type
            except:
                session['analytics_size'] = 0
                session['analytics_upload_date'] = None
                session['analytics_content_type'] = None
        
        return jsonify(session)
        
    except Exception as e:
        print(f"❌ Error getting session: {e}")
        return jsonify({
            "error": "Failed to retrieve session",
            "details": str(e)
        }), 500

@app.route('/downloadVideo/<session_id>', methods=['GET'])
def download_video(session_id):
    """Download/stream video from MongoDB GridFS with range support"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        video_id = session.get('gridfs_video_id')
        if not video_id:
            return jsonify({"error": "Video not found in session"}), 404
        
        # Get video file info first
        video_info = video_processor.get_video_info_from_gridfs(video_id)
        if not video_info:
            return jsonify({"error": "Failed to retrieve video info from database"}), 500
        
        filename = video_info['filename'] or f"video_{session_id}.mp4"
        file_size = video_info['length']
        content_type = video_info['content_type'] or 'video/mp4'
        
        # Check for Range header
        range_header = request.headers.get('Range')
        start, end = parse_range_header(range_header, file_size)
        
        # Check if this is a range request
        is_range_request = range_header is not None
        
        if is_range_request:
            # Handle range request (HTTP 206 Partial Content)
            content_length = end - start + 1
            
            # Stream the video data
            stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
                video_id, start, end
            )
            
            if stream_generator is None:
                return jsonify({"error": "Failed to stream video from database"}), 500
            
            response = Response(
                stream_generator,
                status=206,  # Partial Content
                mimetype=content_type,
                headers={
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Accept-Ranges': 'bytes',
                    'Content-Disposition': f'attachment; filename="{filename}"'
                }
            )
        else:
            # Handle full file request
            # For large files, still use streaming to avoid memory issues
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                # Stream large files
                stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
                    video_id, 0, file_size - 1
                )
                
                if stream_generator is None:
                    return jsonify({"error": "Failed to stream video from database"}), 500
                
                response = Response(
                    stream_generator,
                    mimetype=content_type,
                    headers={
                        'Content-Length': str(file_size),
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
            else:
                # Load small files into memory
                video_data = video_processor.get_video_from_gridfs(video_id)
                if not video_data:
                    return jsonify({"error": "Failed to retrieve video from database"}), 500
                
                response = Response(
                    video_data,
                    mimetype=content_type,
                    headers={
                        'Content-Length': str(len(video_data)),
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
        
        return response
        
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return jsonify({
            "error": "Failed to download video",
            "details": str(e)
        }), 500

@app.route('/streamVideo/<session_id>', methods=['GET'])
def stream_video(session_id):
    """Stream video with progress tracking and chunked transfer encoding"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        video_id = session.get('gridfs_video_id')
        if not video_id:
            return jsonify({"error": "Video not found in session"}), 404
        
        # Get video file info
        video_info = video_processor.get_video_info_from_gridfs(video_id)
        if not video_info:
            return jsonify({"error": "Failed to retrieve video info from database"}), 500
        
        filename = video_info['filename'] or f"video_{session_id}.mp4"
        file_size = video_info['length']
        content_type = video_info['content_type'] or 'video/mp4'
        
        # Check for Range header
        range_header = request.headers.get('Range')
        start, end = parse_range_header(range_header, file_size)
        
        # Stream the video data
        stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
            video_id, start, end
        )
        
        if stream_generator is None:
            return jsonify({"error": "Failed to stream video from database"}), 500
        
        # Create streaming response
        response = Response(
            stream_generator,
            status=206 if range_header else 200,
            mimetype=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Accept-Ranges': 'bytes',
                'Transfer-Encoding': 'chunked'
            }
        )
        
        # Add range headers if this is a range request
        if range_header:
            response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response.headers['Content-Length'] = str(end - start + 1)
        
        return response
        
    except Exception as e:
        print(f"❌ Error streaming video: {e}")
        return jsonify({
            "error": "Failed to stream video",
            "details": str(e)
        }), 500

@app.route('/downloadVideo', methods=['GET'])
def download_video_by_filename():
    """Download video by filename (frontend compatibility)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Video not found"}), 404
        
        video_id = session.get('gridfs_video_id')
        if not video_id:
            return jsonify({"error": "Video not found in session"}), 404
        
        # Get video file info first
        video_info = video_processor.get_video_info_from_gridfs(video_id)
        if not video_info:
            return jsonify({"error": "Failed to retrieve video info from database"}), 500
        
        filename = video_info['filename'] or video_filename
        file_size = video_info['length']
        content_type = video_info['content_type'] or 'video/mp4'
        
        # Check for Range header
        range_header = request.headers.get('Range')
        start, end = parse_range_header(range_header, file_size)
        
        # Check if this is a range request
        is_range_request = range_header is not None
        
        if is_range_request:
            # Handle range request (HTTP 206 Partial Content)
            content_length = end - start + 1
            
            # Stream the video data
            stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
                video_id, start, end
            )
            
            if stream_generator is None:
                return jsonify({"error": "Failed to stream video from database"}), 500
            
            response = Response(
                stream_generator,
                status=206,  # Partial Content
                mimetype=content_type,
                headers={
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Accept-Ranges': 'bytes',
                    'Content-Disposition': f'attachment; filename="{filename}"'
                }
            )
        else:
            # Handle full file request
            # For large files, still use streaming to avoid memory issues
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                # Stream large files
                stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
                    video_id, 0, file_size - 1
                )
                
                if stream_generator is None:
                    return jsonify({"error": "Failed to stream video from database"}), 500
                
                response = Response(
                    stream_generator,
                    mimetype=content_type,
                    headers={
                        'Content-Length': str(file_size),
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
            else:
                # Load small files into memory
                video_data = video_processor.get_video_from_gridfs(video_id)
                if not video_data:
                    return jsonify({"error": "Failed to retrieve video from database"}), 500
                
                response = Response(
                    video_data,
                    mimetype=content_type,
                    headers={
                        'Content-Length': str(len(video_data)),
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
        
        return response
        
    except Exception as e:
        print(f"❌ Error downloading video by filename: {e}")
        return jsonify({
            "error": "Failed to download video",
            "details": str(e)
        }), 500

@app.route('/getProcessedVideos', methods=['GET'])
def get_processed_videos():
    """Get list of all sessions (uploaded and processed videos)"""
    try:
        # Get all sessions
        sessions_list = sessions.get_all_sessions()
        
        processed_videos = []
        for session in sessions_list:
            if session.get('gridfs_video_id'):
                try:
                    video_info = video_processor.get_video_info_from_gridfs(session['gridfs_video_id'])
                    if video_info:
                        # Determine session status
                        status = session.get('status', 'unknown')
                        processing_status = session.get('processing_status', 'unknown')
                        has_analytics = session.get('gridfs_analytics_id') is not None
                        
                        # Determine if this is a completed analysis or just uploaded
                        is_completed = (status == 'completed' and has_analytics)
                        is_uploaded = (status == 'uploaded' or processing_status == 'analysis_failed')
                        
                        processed_videos.append({
                            'session_id': str(session['_id']),
                            'original_filename': session.get('original_filename', 'Unknown'),
                            'processed_filename': session.get('processed_video_filename', video_info['filename']),
                            'file_size_mb': round(video_info['length'] / (1024 * 1024), 2),
                            'analysis_type': 'pose_detection',
                            'has_analytics': has_analytics,
                            'status': status,
                            'processing_status': processing_status,
                            'is_completed': is_completed,
                            'is_uploaded': is_uploaded,
                            'athlete_name': session.get('athlete_name', 'Unknown Athlete'),
                            'event': session.get('event', 'Unknown Event'),
                            'created_at': session.get('created_at', ''),
                            'notes': session.get('notes', '')
                        })
                except Exception as e:
                    print(f"❌ Error getting video info for session {session['_id']}: {e}")
                    continue
        
        return jsonify({
            "success": True,
            "processed_videos": processed_videos
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve processed videos",
            "details": str(e)
        }), 500

@app.route('/getStatistics', methods=['GET'])
def get_statistics():
    """Get statistics for a video (frontend compatibility)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Video not found"}), 404
        
        # Get analytics if available
        analytics_data = None
        if session.get('gridfs_analytics_id'):
            analytics_data = video_processor.get_analytics_from_gridfs(session['gridfs_analytics_id'])
        
        # Return basic statistics
        stats = {
            "success": True,
            "video_filename": video_filename,
            "total_frames": session.get('total_frames', 0),
            "fps": session.get('fps', 30),
            "duration": session.get('duration', '00:00'),
            "motion_iq": session.get('motion_iq', 0.0),
            "acl_risk": session.get('acl_risk', 0.0),
            "precision": session.get('precision', 0.0),
            "power": session.get('power', 0.0),
            "tumbling_percentage": session.get('tumbling_percentage', 0.0),
            "has_analytics": analytics_data is not None
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve statistics",
            "details": str(e)
        }), 500

@app.route('/getSummaryStatistics', methods=['GET'])
def get_summary_statistics():
    """Get summary statistics for all videos (frontend compatibility)"""
    try:
        # Get all sessions
        sessions_list = sessions.get_all_sessions()
        
        total_videos = len([s for s in sessions_list if s.get('gridfs_video_id')])
        total_frames = sum(s.get('total_frames', 0) for s in sessions_list)
        
        # Calculate average metrics
        acl_risks = [s.get('acl_risk', 0.0) for s in sessions_list if s.get('acl_risk') is not None]
        average_acl_risk = sum(acl_risks) / len(acl_risks) if acl_risks else 0.0
        
        # Risk distribution
        risk_distribution = {'low': 0, 'moderate': 0, 'high': 0}
        for risk in acl_risks:
            if risk < 0.3:
                risk_distribution['low'] += 1
            elif risk < 0.7:
                risk_distribution['moderate'] += 1
            else:
                risk_distribution['high'] += 1
        
        # Calculate other metrics
        motion_iqs = [s.get('motion_iq', 0.0) for s in sessions_list if s.get('motion_iq') is not None]
        precisions = [s.get('precision', 0.0) for s in sessions_list if s.get('precision') is not None]
        powers = [s.get('power', 0.0) for s in sessions_list if s.get('power') is not None]
        
        summary = {
            "success": True,
            "total_videos": total_videos,
            "total_frames": total_frames,
            "average_acl_risk": average_acl_risk,
            "risk_distribution": risk_distribution,
            "top_metrics": {
                "average_elevation_angle": sum(motion_iqs) / len(motion_iqs) if motion_iqs else 0.0,
                "average_flight_time": sum(precisions) / len(precisions) if precisions else 0.0,
                "average_landing_quality": sum(powers) / len(powers) if powers else 0.0
            }
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve summary statistics",
            "details": str(e)
        }), 500

@app.route('/analyzeVideoPerFrame', methods=['POST'])
def analyze_video_per_frame():
    """Analyze video per frame (frontend compatibility)"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return jsonify({"error": "video_filename is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Video not found"}), 404
        
        # Generate a job ID
        job_id = f"per_frame_{session['_id']}_{int(time.time())}"
        
        # For now, return success with job ID
        # In a real implementation, this would start background processing
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Per-frame analysis job started",
            "video_filename": video_filename
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to start per-frame analysis",
            "details": str(e)
        }), 500

@app.route('/cancelAnalysis', methods=['POST'])
def cancel_analysis():
    """Cancel an ongoing analysis"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        video_filename = data.get('video_filename')
        
        if not session_id and not video_filename:
            return jsonify({"error": "session_id or video_filename is required"}), 400
        
        # Find the session
        session = None
        if session_id:
            session = sessions.get_session(session_id)
        elif video_filename:
            session = sessions.get_session_by_video_filename(video_filename)
        
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Check if analysis is currently running
        current_status = session.get('status', 'unknown')
        processing_status = session.get('processing_status', 'unknown')
        
        if current_status not in ['processing', 'analyzing'] and processing_status not in ['analyzing', 'processing']:
            return jsonify({
                "error": "No analysis is currently running for this session",
                "current_status": current_status,
                "processing_status": processing_status
            }), 400
        
        # Update session to cancelled status
        sessions.update_session(session['_id'], {
            "status": "cancelled",
            "processing_status": "cancelled",
            "notes": f"Analysis cancelled by user: {session.get('original_filename', 'Unknown')}",
            "updated_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        })
        
        print(f"🛑 Analysis cancelled for session {session['_id']}: {session.get('original_filename')}")
        
        return jsonify({
            "success": True,
            "message": "Analysis cancelled successfully",
            "session_id": str(session['_id']),
            "video_filename": session.get('original_filename')
        })
        
    except Exception as e:
        print(f"❌ Error cancelling analysis: {e}")
        return jsonify({
            "error": "Failed to cancel analysis",
            "details": str(e)
        }), 500

@app.route('/getJobStatus', methods=['GET'])
def get_job_status():
    """Get job status (frontend compatibility)"""
    try:
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({"error": "job_id parameter is required"}), 400
        
        # Extract session ID from job ID (format: per_frame_{session_id}_{timestamp})
        session_id = None
        if job_id.startswith("per_frame_"):
            # Extract session ID from job ID
            parts = job_id.split("_")
            if len(parts) >= 3:
                session_id = "_".join(parts[2:-1])  # Everything between "per_frame" and the timestamp
        else:
            # If it's not a per_frame job, treat it as a direct session ID
            session_id = job_id
        
        # Get the actual session data
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Return the session data in the expected format
        return jsonify({
            "success": True,
            "job_status": {
                "status": session.get("status", "unknown"),
                "video_filename": session.get("original_filename", "unknown"),
                "output_video": session.get("processed_video_filename", session.get("original_filename", "unknown")),
                "analytics_file": session.get("analytics_filename", "unknown"),
                "timestamp": session.get("created_at", datetime.now().isoformat()),
                "result": {
                    "total_frames": session.get("total_frames", 0),
                    "fps": session.get("fps", 30),
                    "processing_status": session.get("processing_status", "unknown")
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to get job status",
            "details": str(e)
        }), 500

@app.route('/getAnalytics/<session_id>', methods=['GET'])
def get_analytics(session_id):
    """Get analytics data from MongoDB GridFS"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        analytics_id = session.get('gridfs_analytics_id')
        if not analytics_id:
            return jsonify({"error": "Analytics not found in session"}), 404
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        return jsonify({
            "session_id": session_id,
            "analytics": analytics_data,
            "metadata": {
                "filename": session.get('analytics_filename'),
                "total_frames": len(analytics_data) if isinstance(analytics_data, list) else 0,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting analytics: {e}")
        return jsonify({
            "error": "Failed to retrieve analytics",
            "details": str(e)
        }), 500

@app.route('/debugSession/<video_filename>', methods=['GET'])
def debug_session_lookup(video_filename):
    """Debug endpoint to check session and analytics data"""
    try:
        print(f"🔍 DEBUG: Looking for session with video filename: {video_filename}")
        
        # Check all sessions in database
        all_sessions = list(sessions.collection.find({}, {
            "original_filename": 1, 
            "processed_video_filename": 1, 
            "gridfs_analytics_id": 1,
            "gridfs_video_id": 1,
            "_id": 1,
            "status": 1,
            "created_at": 1
        }))
        
        print(f"📋 Total sessions in database: {len(all_sessions)}")
        
        # Try different lookup methods
        session_by_processed = sessions.get_session_by_video_filename(video_filename)
        
        # Extract original filename if it's a processed filename
        original_filename = None
        if video_filename.startswith('h264_analyzed_temp_'):
            parts = video_filename.split('_')
            if len(parts) >= 5:
                original_parts = parts[3:-1]
                original_filename = '_'.join(original_parts) + '.mp4'
        
        session_by_original = None
        if original_filename:
            session_by_original = sessions.get_session_by_video_filename(original_filename)
        
        # Check GridFS analytics
        analytics_data = None
        if session_by_processed and session_by_processed.get('gridfs_analytics_id'):
            analytics_data = video_processor.get_analytics_from_gridfs(session_by_processed['gridfs_analytics_id'])
        elif session_by_original and session_by_original.get('gridfs_analytics_id'):
            analytics_data = video_processor.get_analytics_from_gridfs(session_by_original['gridfs_analytics_id'])
        
        # Convert ObjectId to string for JSON serialization
        def convert_objectid(obj):
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                if isinstance(obj, dict):
                    return {k: convert_objectid(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_objectid(item) for item in obj]
            elif hasattr(obj, '__str__') and 'ObjectId' in str(type(obj)):
                return str(obj)
            return obj
        
        return jsonify({
            "video_filename": video_filename,
            "original_filename": original_filename,
            "total_sessions": len(all_sessions),
            "session_by_processed_filename": convert_objectid(session_by_processed),
            "session_by_original_filename": convert_objectid(session_by_original),
            "analytics_data_exists": analytics_data is not None,
            "analytics_data_type": type(analytics_data).__name__ if analytics_data else None,
            "analytics_data_length": len(analytics_data) if isinstance(analytics_data, list) else "N/A",
            "all_sessions": [
                {
                    "id": str(s.get("_id", "")),
                    "original_filename": s.get("original_filename", ""),
                    "processed_video_filename": s.get("processed_video_filename", ""),
                    "has_analytics_id": bool(s.get("gridfs_analytics_id")),
                    "analytics_id": str(s.get("gridfs_analytics_id", "")) if s.get("gridfs_analytics_id") else None,
                    "status": s.get("status", ""),
                    "created_at": s.get("created_at", "")
                } for s in all_sessions
            ]
        })
        
    except Exception as e:
        return jsonify({
            "error": "Debug failed",
            "details": str(e)
        }), 500

@app.route('/debugAnalytics/<analytics_id>', methods=['GET'])
def debug_analytics_retrieval(analytics_id):
    """Debug endpoint to test analytics retrieval from GridFS"""
    try:
        print(f"🔍 DEBUG: Retrieving analytics with ID: {analytics_id}")
        
        # Try to get analytics from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        
        if analytics_data:
            return jsonify({
                "analytics_id": analytics_id,
                "data_exists": True,
                "data_type": type(analytics_data).__name__,
                "data_length": len(analytics_data) if isinstance(analytics_data, list) else "N/A",
                "sample_data": analytics_data[:2] if isinstance(analytics_data, list) and len(analytics_data) > 0 else analytics_data,
                "first_frame_keys": list(analytics_data[0].keys()) if isinstance(analytics_data, list) and len(analytics_data) > 0 and isinstance(analytics_data[0], dict) else "N/A"
            })
        else:
            return jsonify({
                "analytics_id": analytics_id,
                "data_exists": False,
                "error": "No analytics data found"
            })
            
    except Exception as e:
        return jsonify({
            "analytics_id": analytics_id,
            "error": "Failed to retrieve analytics",
            "details": str(e)
        }), 500

@app.route('/fixAnalytics/<session_id>', methods=['POST'])
def fix_missing_analytics(session_id):
    """Fix missing analytics for a session by uploading existing analytics file"""
    try:
        # Get the session
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        original_filename = session.get('original_filename')
        if not original_filename:
            return jsonify({"error": "No original filename in session"}), 400
        
        # Look for existing analytics file
        analytics_filename = f"fixed_analytics_{original_filename}.json"
        analytics_path = os.path.join(".", analytics_filename)
        
        if not os.path.exists(analytics_path):
            analytics_path = os.path.join(OUTPUT_DIR, analytics_filename)
        
        if not os.path.exists(analytics_path):
            return jsonify({"error": f"Analytics file not found: {analytics_filename}"}), 404
        
        print(f"🔧 Fixing analytics for session {session_id}")
        print(f"📊 Found analytics file: {analytics_path}")
        
        # Upload analytics to GridFS
        with open(analytics_path, 'rb') as analytics_file:
            analytics_id = video_processor.fs.put(
                analytics_file,
                filename=analytics_filename,
                metadata={
                    "video_filename": original_filename,
                    "analysis_type": "per_frame_analytics",
                    "upload_timestamp": datetime.now().isoformat(),
                    "fix_upload": True
                },
                contentType='application/json'
            )
        
        # Update session with analytics ID
        sessions.update_session(session_id, {
            "gridfs_analytics_id": str(analytics_id),
            "analytics_filename": analytics_filename,
            "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}"
        })
        
        print(f"✅ Analytics uploaded to GridFS: {analytics_id}")
        
        return jsonify({
            "success": True,
            "message": "Analytics fixed and uploaded to GridFS",
            "analytics_id": str(analytics_id),
            "analytics_filename": analytics_filename
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to fix analytics",
            "details": str(e)
        }), 500

@app.route('/getPerFrameStatistics', methods=['GET'])
def get_per_frame_statistics_by_filename():
    """Get per-frame statistics by video filename (frontend compatibility)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        print(f"🔍 Looking for session with video filename: {video_filename}")
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            print(f"❌ No session found for video filename: {video_filename}")
            
            # Try to extract original filename from processed filename
            if video_filename.startswith('h264_analyzed_temp_'):
                # Extract original filename from processed filename
                # Format: h264_analyzed_temp_{timestamp}_{original_filename}_{timestamp}.mp4
                parts = video_filename.split('_')
                if len(parts) >= 5:
                    # Find the original filename part (skip timestamp parts)
                    original_parts = parts[3:-1]  # Skip 'h264', 'analyzed', 'temp', first timestamp, and last timestamp
                    original_filename = '_'.join(original_parts) + '.mp4'
                    print(f"🔍 Trying to find session with extracted original filename: {original_filename}")
                    session = sessions.get_session_by_video_filename(original_filename)
            
            if not session:
                # Try to find any sessions with similar filenames for debugging
                try:
                    all_sessions = list(sessions.collection.find({}, {"original_filename": 1, "processed_video_filename": 1, "_id": 1}))
                    print(f"📋 Available sessions: {[s.get('processed_video_filename', s.get('original_filename', 'unknown')) for s in all_sessions]}")
                except Exception as debug_e:
                    print(f"⚠️ Could not list sessions for debugging: {debug_e}")
                return jsonify({"error": "Session not found"}), 404
        
        analytics_id = session.get('gridfs_analytics_id')
        if not analytics_id:
            return jsonify({"error": "Analytics not found for this session"}), 404
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        # Calculate actual frame count and FPS from analytics data
        actual_frame_data = analytics_data.get('frame_data', []) if isinstance(analytics_data, dict) else analytics_data
        actual_frame_count = len(actual_frame_data) if actual_frame_data else 0
        
        # Calculate FPS from frame data if available
        calculated_fps = 30.0  # Default FPS
        if actual_frame_data and len(actual_frame_data) > 1:
            # Try to calculate FPS from timestamps
            first_frame = actual_frame_data[0]
            last_frame = actual_frame_data[-1]
            if 'metrics' in first_frame and 'metrics' in last_frame:
                first_time = first_frame['metrics'].get('timestamp', 0)
                last_time = last_frame['metrics'].get('timestamp', 0)
                if last_time > first_time:
                    time_diff = last_time - first_time
                    calculated_fps = (len(actual_frame_data) - 1) / time_diff if time_diff > 0 else 30.0
        
        # Format response to match frontend expectations
        response_data = {
            "success": True,
            "video_filename": video_filename,
            "total_frames": actual_frame_count,
            "fps": calculated_fps,
            "frames_processed": actual_frame_count,
            "processing_time": "00:00:01",
            "enhanced_analytics": True,
            "frame_data": actual_frame_data,
            "enhanced_statistics": analytics_data.get('enhanced_statistics', {}) if isinstance(analytics_data, dict) else {}
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve per-frame statistics",
            "details": str(e)
        }), 500

@app.route('/getPerFrameStatistics/<session_id>', methods=['GET'])
def get_per_frame_statistics(session_id):
    """Get per-frame statistics from MongoDB"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        analytics_id = session.get('gridfs_analytics_id')
        if not analytics_id:
            return jsonify({"error": "Analytics not found in session"}), 404
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        # Calculate statistics
        if isinstance(analytics_data, list) and len(analytics_data) > 0:
            # Calculate detailed statistics
            stats = calculate_detailed_statistics(analytics_data)
            
            return jsonify({
                "session_id": session_id,
                "video_filename": session.get('original_filename'),
                "statistics": stats,
                "frame_count": len(analytics_data),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "session_id": session_id,
                "error": "No frame data available",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        print(f"❌ Error getting per-frame statistics: {e}")
        return jsonify({
            "error": "Failed to retrieve per-frame statistics",
            "details": str(e)
        }), 500

@app.route('/getACLRiskAnalysis', methods=['GET'])
def get_acl_risk_analysis_by_filename():
    """Get ACL risk analysis by video filename (frontend compatibility)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        analytics_id = session.get('gridfs_analytics_id')
        if not analytics_id:
            return jsonify({"error": "Analytics not found for this session"}), 404
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        # Calculate ACL risk factors from session data
        acl_risk = session.get('acl_risk', 0.0)
        
        # Determine risk level
        if acl_risk < 0.3:
            risk_level = 'LOW'
        elif acl_risk < 0.7:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'
        
        # Format response to match frontend expectations
        response_data = {
            "success": True,
            "video_filename": video_filename,
            "risk_factors": {
                "knee_angle_risk": acl_risk * 0.3,
                "knee_valgus_risk": acl_risk * 0.4,
                "landing_mechanics_risk": acl_risk * 0.3,
                "overall_acl_risk": acl_risk,
                "risk_level": risk_level
            },
            "recommendations": [
                "Focus on proper landing mechanics",
                "Strengthen knee stabilizing muscles",
                "Practice controlled landings",
                "Work on balance and proprioception"
            ],
            "frame_analysis": analytics_data.get('frame_data', []) if isinstance(analytics_data, dict) else analytics_data
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve ACL risk analysis",
            "details": str(e)
        }), 500

@app.route('/getACLRiskAnalysis/<session_id>', methods=['GET'])
def get_acl_risk_analysis(session_id):
    """Get ACL risk analysis from MongoDB"""
    try:
        session = sessions.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        analytics_id = session.get('gridfs_analytics_id')
        if not analytics_id:
            return jsonify({"error": "Analytics not found in session"}), 404
        
        # Get analytics data from GridFS
        analytics_data = video_processor.get_analytics_from_gridfs(analytics_id)
        if not analytics_data:
            return jsonify({"error": "Failed to retrieve analytics from database"}), 500
        
        # Calculate ACL risk analysis
        if isinstance(analytics_data, list) and len(analytics_data) > 0:
            acl_analysis = calculate_acl_risk_analysis(analytics_data)
            
            return jsonify({
                "session_id": session_id,
                "video_filename": session.get('original_filename'),
                "acl_analysis": acl_analysis,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "session_id": session_id,
                "error": "No frame data available for ACL analysis",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        print(f"❌ Error getting ACL risk analysis: {e}")
        return jsonify({
            "error": "Failed to retrieve ACL risk analysis",
            "details": str(e)
        }), 500

@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    """Upload video to MongoDB GridFS"""
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
        auto_analyze = request.form.get('auto_analyze', 'true').lower() == 'true'
        
        print(f"📤 Uploading video: {video_file.filename}")
        print(f"👤 Athlete: {athlete_name}")
        print(f"🏃 Event: {event}")
        print(f"📝 Session: {session_name}")
        print(f"🔄 Auto-analyze: {auto_analyze}")
        
        # Upload video to GridFS
        video_metadata = {
            "athlete_name": athlete_name,
            "event": event,
            "session_name": session_name,
            "original_filename": video_file.filename,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        video_id = fs.put(
            video_file,
            filename=video_file.filename,
            metadata=video_metadata,
            contentType='video/mp4'
        )
        
        # Create session record
        session_data = {
            "user_id": "demo_user",
            "athlete_name": athlete_name,
            "session_name": session_name,
            "event": event,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "duration": "00:00",  # Will be updated after processing
            "original_filename": video_file.filename,
            "processed_video_filename": video_file.filename,  # Will be updated when processed
            "processed_video_url": f"http://localhost:5004/getVideo?video_filename={video_file.filename}",
            "analytics_filename": None,
            "analytics_url": None,
            "motion_iq": 0.0,
            "acl_risk": 0.0,
            "precision": 0.0,
            "power": 0.0,
            "tumbling_percentage": 0.0,
            "status": "uploaded",
            "processing_progress": 0.0,
            "total_frames": 0,
            "fps": 0.0,
            "has_landmarks": False,
            "landmark_confidence": 0.0,
            "notes": f"Video uploaded: {video_file.filename}",
            "coach_notes": "",
            "highlights": [],
            "areas_for_improvement": [],
            "gridfs_video_id": video_id,
            "gridfs_analytics_id": None,
            "is_binary_stored": True,
            "processing_status": "uploaded"
        }
        
        session_id = sessions.create_session(session_data)
        print(f"✅ Session created with ID: {session_id}")
        print(f"📁 Original filename: {video_file.filename}")
        print(f"👤 Athlete name: {athlete_name}")
        print(f"🏃 Event: {event}")
        
        # Automatically start analysis after successful upload (if enabled)
        if auto_analyze:
            try:
                print(f"🚀 Auto-starting analysis for uploaded video: {video_file.filename}")
                
                # Prepare analysis request data with GridFS video ID and session ID
                analysis_data = {
                    "video_filename": video_file.filename,
                    "athlete_name": athlete_name,
                    "event": event,
                    "session_name": session_name,
                    "user_id": "demo_user",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "gridfs_video_id": str(video_id),  # Add GridFS video ID for direct access
                    "session_id": session_id  # Add session ID to update existing session
                }
                
                # Call analyzeVideo1 endpoint internally
                analysis_response = analyze_video_from_gridfs_internal(analysis_data)
                
                if analysis_response.get("success"):
                    print(f"✅ Auto-analysis started successfully for: {video_file.filename}")
                    # Update session with analysis results
                    sessions.update_session(session_id, {
                        "status": "processing",
                        "processing_status": "analyzing",
                        "notes": f"Video uploaded and analysis started: {video_file.filename}"
                    })
                else:
                    print(f"⚠️ Auto-analysis failed for: {video_file.filename}, but upload was successful")
                    # Update session to indicate analysis failed but upload succeeded
                    sessions.update_session(session_id, {
                        "status": "uploaded",
                        "processing_status": "analysis_failed",
                        "notes": f"Video uploaded but analysis failed: {video_file.filename}"
                    })
                    
            except Exception as analysis_error:
                print(f"⚠️ Auto-analysis error for {video_file.filename}: {analysis_error}")
                # Update session to indicate analysis failed but upload succeeded
                sessions.update_session(session_id, {
                    "status": "uploaded", 
                    "processing_status": "analysis_failed",
                    "notes": f"Video uploaded but analysis failed: {video_file.filename}"
                })
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "video_id": str(video_id),
            "message": "Video uploaded successfully" + (" and analysis started" if auto_analyze else ""),
            "timestamp": datetime.now().isoformat(),
            "auto_analysis": auto_analyze
        })
        
    except Exception as e:
        print(f"❌ Error uploading video: {e}")
        return jsonify({
            "error": "Failed to upload video",
            "details": str(e)
        }), 500

@app.route('/getVideoList', methods=['GET'])
def get_video_list():
    """Get list of videos from MongoDB"""
    try:
        # Get all sessions with video information
        sessions_list = sessions.get_all_sessions()
        
        video_list = []
        for session in sessions_list:
            if session.get('gridfs_video_id'):
                try:
                    video_file = fs.get(ObjectId(session['gridfs_video_id']))
                    video_list.append({
                        "session_id": str(session['_id']),
                        "filename": session.get('original_filename'),
                        "athlete_name": session.get('athlete_name'),
                        "event": session.get('event'),
                        "date": session.get('date'),
                        "duration": session.get('duration'),
                        "file_size": video_file.length,
                        "upload_date": video_file.upload_date.isoformat(),
                        "status": session.get('status'),
                        "has_analytics": bool(session.get('gridfs_analytics_id'))
                    })
                except Exception as e:
                    print(f"⚠️  Error getting video info for session {session['_id']}: {e}")
        
        return jsonify({
            "videos": video_list,
            "count": len(video_list),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting video list: {e}")
        return jsonify({
            "error": "Failed to retrieve video list",
            "details": str(e)
        }), 500

@app.route('/analyzeVideo', methods=['POST'])
def analyze_video():
    """Analyze video and generate analytics overlay, then upload to MongoDB"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        
        if not video_filename:
            return jsonify({"error": "video_filename is required"}), 400
        
        # Check if Railway MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }), 503
        
        # Find the best available video (prioritize H.264)
        best_video_path, is_h264, original_path = video_processor.find_best_video(video_filename)
        
        if not best_video_path:
            return jsonify({"error": f"Video file not found: {video_filename}"}), 404
        
        # Use the best available video for processing
        video_path = best_video_path
        print(f"🎬 Processing video: {os.path.basename(video_path)} (H.264: {is_h264})")
        
        # Generate unique output name (avoid double h264 prefix)
        base_name = os.path.splitext(video_filename)[0]
        # Remove existing h264_ prefix if present to avoid accumulation
        if base_name.startswith('h264_'):
            base_name = base_name[5:]  # Remove 'h264_' prefix
        timestamp = int(time.time())
        output_name = f"analyzed_temp_{timestamp}_{base_name}_{timestamp}.mp4"
        
        # Process video with MediaPipe
        result = video_processor.process_video_with_analytics(video_path, output_name)
        
        if result["success"]:
            # Upload processed video and analytics to GridFS
            video_id, analytics_id = video_processor.upload_processed_video_to_gridfs(
                result.get("h264_video", result["output_video"]),
                result["analytics_file"],
                {
                    "athlete_name": athlete_name,
                    "event": event,
                    "session_name": session_name,
                    "original_filename": video_filename,
                    "processing_timestamp": timestamp
                }
            )
            
            if video_id:
                # Create session record
                session_data = {
                    "user_id": "demo_user",
                    "athlete_name": athlete_name,
                    "session_name": session_name,
                    "event": event,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "duration": "00:00",  # Will be updated after processing
                    "original_filename": video_filename,
                    "processed_video_filename": os.path.basename(result.get("h264_video", result["output_video"])),
                    "processed_video_url": f"http://localhost:5004/getVideo?video_filename={os.path.basename(result.get('h264_video', result['output_video']))}",
                    "analytics_filename": os.path.basename(result["analytics_file"]) if result["analytics_file"] else None,
                    "analytics_url": f"http://localhost:5004/getAnalytics/{video_id}" if analytics_id else None,
                    "motion_iq": 0.0,  # Will be calculated from analytics
                    "acl_risk": 0.0,   # Will be calculated from analytics
                    "precision": 0.0,  # Will be calculated from analytics
                    "power": 0.0,      # Will be calculated from analytics
                    "tumbling_percentage": 0.0,  # Will be calculated from analytics
                    "status": "completed",
                    "processing_progress": 1.0,
                    "total_frames": 0,  # Will be calculated from analytics
                    "fps": 0.0,         # Will be calculated from analytics
                    "has_landmarks": True,
                    "landmark_confidence": 0.9,
                    "notes": f"Video analyzed with Railway MediaPipe server: {video_filename}",
                    "coach_notes": "",
                    "highlights": [],
                    "areas_for_improvement": [],
                    "gridfs_video_id": video_id,
                    "gridfs_analytics_id": analytics_id,
                    "is_binary_stored": True,
                    "processing_status": "completed"
                }
                
                # Calculate analytics if available
                if analytics_id and os.path.exists(result["analytics_file"]):
                    try:
                        with open(result["analytics_file"], 'r') as f:
                            analytics_data = json.load(f)
                        
                        if isinstance(analytics_data, list) and len(analytics_data) > 0:
                            # Calculate basic statistics
                            session_data["total_frames"] = len(analytics_data)
                            session_data["fps"] = 30.0  # Default FPS
                            
                            # Calculate landmark detection rate
                            frames_with_landmarks = 0
                            total_confidence = 0.0
                            
                            for frame_data in analytics_data:
                                if isinstance(frame_data, dict) and 'landmarks' in frame_data:
                                    landmarks = frame_data['landmarks']
                                    if landmarks and len(landmarks) > 0:
                                        frames_with_landmarks += 1
                                        frame_confidence = sum(landmark.get('visibility', 0) for landmark in landmarks) / len(landmarks)
                                        total_confidence += frame_confidence
                            
                            detection_rate = frames_with_landmarks / len(analytics_data) if len(analytics_data) > 0 else 0
                            avg_confidence = total_confidence / frames_with_landmarks if frames_with_landmarks > 0 else 0
                            
                            session_data["landmark_confidence"] = avg_confidence
                            session_data["motion_iq"] = detection_rate * 100
                            session_data["precision"] = avg_confidence * 100
                            
                            print(f"📊 Analytics calculated: {len(analytics_data)} frames, {detection_rate:.2%} detection rate")
                    
                    except Exception as e:
                        print(f"⚠️  Error calculating analytics: {e}")
                
                # Update existing session or create new one
                existing_session_id = data.get('session_id')
                if existing_session_id:
                    # Update existing session with analysis results
                    print(f"🔄 Updating existing session: {existing_session_id}")
                    sessions.update_session(existing_session_id, session_data)
                    session_id = existing_session_id
                else:
                    # Create new session (for standalone analysis)
                    print(f"🆕 Creating new session")
                    session_id = sessions.create_session(session_data)
                
                return jsonify({
                    "success": True,
                    "session_id": session_id,
                    "video_id": str(video_id),
                    "analytics_id": str(analytics_id) if analytics_id else None,
                    "output_video": os.path.basename(result.get("h264_video", result["output_video"])),
                    "analytics_file": os.path.basename(result["analytics_file"]) if result["analytics_file"] else None,
                    "message": "Video analysis completed and uploaded to MongoDB",
                    "download_url": f"http://localhost:5004/getVideo?video_filename={result.get('output_video', '')}",
                    "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to upload processed video to MongoDB",
                    "message": "Video was processed but could not be stored in database"
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": result["error"],
                "message": result["message"]
            }), 500
        
    except Exception as e:
        print(f"❌ Error in analyzeVideo: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Video analysis failed",
            "details": str(e)
        }), 500

# Helper functions (copied from original server)
def calculate_detailed_statistics(analytics_data):
    """Calculate detailed statistics from analytics data"""
    if not analytics_data:
        return {}
    
    try:
        total_frames = len(analytics_data)
        if total_frames == 0:
            return {}
        
        # Initialize statistics
        stats = {
            "total_frames": total_frames,
            "landmark_detection": {
                "frames_with_landmarks": 0,
                "average_confidence": 0.0,
                "detection_rate": 0.0
            },
            "motion_analysis": {
                "average_velocity": 0.0,
                "max_velocity": 0.0,
                "motion_consistency": 0.0
            },
            "pose_analysis": {
                "average_pose_confidence": 0.0,
                "stable_poses": 0,
                "pose_variability": 0.0
            }
        }
        
        # Calculate landmark detection statistics
        frames_with_landmarks = 0
        total_confidence = 0.0
        
        for frame_data in analytics_data:
            if isinstance(frame_data, dict) and 'landmarks' in frame_data:
                landmarks = frame_data['landmarks']
                if landmarks and len(landmarks) > 0:
                    frames_with_landmarks += 1
                    # Calculate average confidence for this frame
                    frame_confidence = sum(landmark.get('visibility', 0) for landmark in landmarks) / len(landmarks)
                    total_confidence += frame_confidence
        
        stats["landmark_detection"]["frames_with_landmarks"] = frames_with_landmarks
        stats["landmark_detection"]["detection_rate"] = frames_with_landmarks / total_frames if total_frames > 0 else 0
        stats["landmark_detection"]["average_confidence"] = total_confidence / frames_with_landmarks if frames_with_landmarks > 0 else 0
        
        return stats
        
    except Exception as e:
        print(f"❌ Error calculating detailed statistics: {e}")
        return {"error": str(e)}

def calculate_acl_risk_analysis(analytics_data):
    """Calculate ACL risk analysis from analytics data"""
    if not analytics_data:
        return {}
    
    try:
        # Simplified ACL risk calculation
        total_frames = len(analytics_data)
        high_risk_frames = 0
        medium_risk_frames = 0
        low_risk_frames = 0
        
        for frame_data in analytics_data:
            if isinstance(frame_data, dict) and 'landmarks' in frame_data:
                landmarks = frame_data['landmarks']
                if landmarks and len(landmarks) >= 33:  # Full pose landmarks
                    # Simple risk assessment based on landmark positions
                    # This is a simplified version - real ACL risk assessment would be more complex
                    risk_score = 0.0
                    
                    # Calculate some basic risk factors
                    if len(landmarks) > 25:  # Left knee, right knee landmarks
                        left_knee = landmarks[25] if len(landmarks) > 25 else None
                        right_knee = landmarks[26] if len(landmarks) > 26 else None
                        
                        if left_knee and right_knee:
                            # Simple risk calculation based on knee positions
                            knee_angle_factor = abs(left_knee.get('x', 0.5) - right_knee.get('x', 0.5))
                            risk_score = min(knee_angle_factor * 100, 100)
                    
                    # Categorize risk
                    if risk_score > 70:
                        high_risk_frames += 1
                    elif risk_score > 40:
                        medium_risk_frames += 1
                    else:
                        low_risk_frames += 1
        
        # Calculate overall risk
        overall_risk = (high_risk_frames * 0.8 + medium_risk_frames * 0.4 + low_risk_frames * 0.1) / total_frames if total_frames > 0 else 0
        
        return {
            "overall_risk_score": overall_risk * 100,
            "risk_distribution": {
                "high_risk_frames": high_risk_frames,
                "medium_risk_frames": medium_risk_frames,
                "low_risk_frames": low_risk_frames
            },
            "risk_percentage": {
                "high_risk": (high_risk_frames / total_frames * 100) if total_frames > 0 else 0,
                "medium_risk": (medium_risk_frames / total_frames * 100) if total_frames > 0 else 0,
                "low_risk": (low_risk_frames / total_frames * 100) if total_frames > 0 else 0
            },
            "recommendations": [
                "Focus on proper landing mechanics",
                "Strengthen core stability",
                "Practice controlled landings"
            ] if overall_risk > 0.5 else [
                "Maintain current form",
                "Continue strength training"
            ]
        }
        
    except Exception as e:
        print(f"❌ Error calculating ACL risk analysis: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    print("🚀 Starting Updated Gymnastics API Server with MongoDB Integration...")
    print("📊 MongoDB Collections:")
    try:
        collections = db_manager.db.list_collection_names()
        for collection in collections:
            count = db_manager.db[collection].count_documents({})
            print(f"   {collection}: {count} documents")
    except Exception as e:
        print(f"❌ Error checking MongoDB: {e}")

@app.route('/getSessionsByUser/<user_id>', methods=['GET'])
def get_sessions_by_user(user_id):
    """Get sessions by user ID (frontend compatibility)"""
    try:
        # Get all sessions and filter by user_id
        all_sessions = sessions.get_all_sessions()
        user_sessions = [s for s in all_sessions if s.get('user_id') == user_id]
        
        # Add GridFS file information and convert ObjectIds to strings
        for session in user_sessions:
            # Convert ObjectIds to strings for JSON serialization
            if session.get('_id'):
                session['_id'] = str(session['_id'])
            if session.get('gridfs_video_id'):
                session['gridfs_video_id'] = str(session['gridfs_video_id'])
                try:
                    video_file = fs.get(ObjectId(session['gridfs_video_id']))
                    session['video_size'] = video_file.length
                    session['video_upload_date'] = video_file.upload_date.isoformat()
                except:
                    session['video_size'] = 0
                    session['video_upload_date'] = None
            
            if session.get('gridfs_analytics_id'):
                session['gridfs_analytics_id'] = str(session['gridfs_analytics_id'])
                try:
                    analytics_file = fs.get(ObjectId(session['gridfs_analytics_id']))
                    session['analytics_size'] = analytics_file.length
                    session['analytics_upload_date'] = analytics_file.upload_date.isoformat()
                except:
                    session['analytics_size'] = 0
                    session['analytics_upload_date'] = None
        
        return jsonify({
            "success": True,
            "sessions": user_sessions,
            "count": len(user_sessions),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting sessions for user {user_id}: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve sessions for user {user_id}",
            "details": str(e)
        }), 500

@app.route('/downloadPerFrameVideo', methods=['GET'])
def download_per_frame_video():
    """Download per-frame video (overlayed) by filename (frontend compatibility)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Video not found"}), 404
        
        # For now, return the regular processed video
        # In a real implementation, this would return the per-frame overlay video
        video_id = session.get('gridfs_video_id')
        if not video_id:
            return jsonify({"error": "Video file not found in database"}), 404
        
        try:
            if isinstance(video_id, str):
                video_id = ObjectId(video_id)
            
            grid_out = fs.get(video_id)
            
            def generate():
                while True:
                    chunk = grid_out.read(8192)
                    if not chunk:
                        break
                    yield chunk
            
            return Response(
                generate(),
                mimetype='video/mp4',
                headers={
                    'Content-Disposition': f'attachment; filename="per_frame_{video_filename}"',
                    'Content-Length': str(grid_out.length)
                }
            )
            
        except Exception as e:
            print(f"❌ Error streaming per-frame video: {e}")
            return jsonify({"error": "Failed to stream video"}), 500
            
    except Exception as e:
        print(f"❌ Error downloading per-frame video: {e}")
        return jsonify({"error": "Failed to download per-frame video"}), 500


def analyze_video_from_gridfs_internal(data):
    """Internal function to analyze video from GridFS (called from uploadVideo)"""
    try:
        video_filename = data.get('video_filename')
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        user_id = data.get('user_id', 'demo_user')
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        if not video_filename:
            return {"success": False, "error": "video_filename is required"}
        
        print(f"🔍 Looking for video: {video_filename}")
        
        # Check if Railway MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return {
                "success": False,
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }
        
        # For auto-analysis, prioritize GridFS lookup since video was just uploaded
        video_path = None
        gridfs_video_id = data.get('gridfs_video_id')
        
        # First try to use GridFS video ID if provided (for auto-analysis)
        if gridfs_video_id:
            print(f"🔍 Using provided GridFS video ID: {gridfs_video_id}")
            try:
                from bson import ObjectId
                gridfs_file = fs.get(ObjectId(gridfs_video_id))
                print(f"✅ Found video in GridFS by ID: {gridfs_file.filename}")
                
                # Download video from GridFS to local temp file
                # Ensure temp directory exists
                os.makedirs(TEMP_DIR, exist_ok=True)
                temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{gridfs_file.filename}")
                with open(temp_video_path, 'wb') as temp_file:
                    temp_file.write(gridfs_file.read())
                
                video_path = temp_video_path
                print(f"📥 Downloaded video to: {video_path}")
            except Exception as gridfs_error:
                print(f"⚠️ GridFS lookup by ID failed: {gridfs_error}")
                gridfs_file = None
        else:
            # Fallback to filename search
            print(f"🔍 Searching GridFS for: {video_filename}")
            
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
                    
                    # Download video from GridFS to local temp file
                    # Ensure temp directory exists
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    temp_video_path = os.path.join(TEMP_DIR, f"temp_{int(time.time())}_{gridfs_file.filename}")
                    with open(temp_video_path, 'wb') as temp_file:
                        temp_file.write(gridfs_file.read())
                    
                    video_path = temp_video_path
                    print(f"📥 Downloaded video to: {video_path}")
                else:
                    print(f"⚠️ Video not found in GridFS, trying local search: {video_filename}")
                    
            except Exception as gridfs_error:
                print(f"⚠️ GridFS search failed: {gridfs_error}")
        
        # If GridFS lookup failed, try local search as fallback (but only if no GridFS video ID was provided)
        if not video_path:
            if gridfs_video_id:
                # If we have a GridFS video ID but couldn't download it, that's an error
                return {
                    "success": False,
                    "error": f"Failed to download video from GridFS with ID: {gridfs_video_id}"
                }
            else:
                # Only try local search if no GridFS video ID was provided
                print(f"🔍 Trying local search for: {video_filename}")
                best_video_path, is_h264, original_path = video_processor.find_best_video(video_filename)
                
                if best_video_path and os.path.exists(best_video_path):
                    print(f"✅ Found local video: {best_video_path}")
                    video_path = best_video_path
                else:
                    return {
                        "success": False,
                        "error": f"Video file not found in GridFS or locally: {video_filename}"
                    }
        
        # Update session status to processing immediately
        if session:
            sessions.update_session(session['_id'], {
                "status": "processing",
                "processing_status": "analyzing",
                "notes": f"Analysis started: {video_filename}",
                "updated_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            })
        
        # Return immediately to prevent timeout, processing will continue in background
        print(f"🔄 Starting background video processing for: {video_filename}")
        print(f"📁 Video path: {video_path}")
        
        # Start processing in background (this will be handled by the internal function)
        try:
            # Call the internal function that handles the actual processing
            result = analyze_video_from_gridfs_internal({
                "video_filename": video_filename,
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "user_id": user_id,
                "date": date,
                "session_id": session['_id'] if session else None
            })
            
            # If we get here, processing completed quickly
            if result.get("success"):
                print(f"✅ Video analysis completed successfully: {video_filename}")
                return {
                    "success": True,
                    "message": "Video analysis completed successfully",
                    "session_id": result.get("session_id"),
                    "video_id": result.get("video_id"),
                    "analytics_id": result.get("analytics_id"),
                    "output_video": result.get("output_video"),
                    "analytics_file": result.get("analytics_file"),
                    "download_url": result.get("download_url"),
                    "analytics_url": result.get("analytics_url"),
                    "timestamp": result.get("timestamp")
                }
            else:
                print(f"❌ Video analysis failed: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Video analysis failed"),
                    "details": result.get("details", "")
                }
                
        except Exception as processing_error:
            print(f"❌ Video processing exception: {processing_error}")
            return {
                "success": False,
                "error": "Video processing failed",
                "details": str(processing_error)
            }
            
    except Exception as e:
        print(f"❌ Error in analyzeVideo1: {e}")
        return {
            "success": False,
            "error": "Video analysis failed",
            "details": str(e)
        }

@app.route('/analyzeVideo1', methods=['POST'])
def analyze_video_from_gridfs():
    """Analyze video from GridFS by downloading it locally first, then processing"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        user_id = data.get('user_id', 'demo_user')
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        if not video_filename:
            return jsonify({"error": "video_filename is required"}), 400
        
        print(f"🔍 Looking for video: {video_filename}")
        
        # Check if Railway MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }), 503
        
        # First, try to find the session to get GridFS video ID
        session = sessions.get_session_by_video_filename(video_filename)
        video_path = None
        
        if session and session.get('gridfs_video_id'):
            # Use GridFS video ID if available (prioritize uploaded video)
            print(f"🔍 Found session with GridFS video ID: {session['gridfs_video_id']}")
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
            "processed_video_url": f"http://localhost:5004/getVideo?video_filename={os.path.basename(actual_output_path)}",
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
        
        session_id = sessions.create_session(session_data)
        print(f"✅ Session created: {session_id}")
        
        # Clean up temporary file if it was downloaded from GridFS
        if video_path.startswith("/tmp") or video_path.startswith(".") and "temp_" in video_path:
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up temporary file: {video_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file: {e}")
        
        return jsonify({
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
        })
        
    except Exception as e:
        print(f"❌ Error in analyzeVideo1: {e}")
        return jsonify({
            "error": "Video analysis failed",
            "details": str(e)
        }), 500

@app.route('/analyzevideo2', methods=['POST'])
def analyze_video_enhanced():
    """Analyze video using enhanced overlay script with anti-flickering and comprehensive analytics"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        athlete_name = data.get('athlete_name', 'Unknown Athlete')
        event = data.get('event', 'Floor Exercise')
        session_name = data.get('session_name', f'{athlete_name} - {event}')
        user_id = data.get('user_id', 'demo_user')
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        if not video_filename:
            return jsonify({"error": "video_filename is required"}), 400
        
        print(f"🔍 Looking for video: {video_filename}")
        
        # Check if Railway MediaPipe server is running
        if not video_processor.check_mediapipe_server():
            return jsonify({
                "error": "Railway MediaPipe server is not available",
                "message": "Please check the Railway MediaPipe server status",
                "server_url": MEDIAPIPE_SERVER_URL
            }), 503
        
        # First, try to find the session to get GridFS video ID
        session = sessions.get_session_by_video_filename(video_filename)
        video_path = None
        
        if session and session.get('gridfs_video_id'):
            # Use GridFS video ID if available (prioritize uploaded video)
            print(f"🔍 Found session with GridFS video ID: {session['gridfs_video_id']}")
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
        
        # Process the video using enhanced overlay script
        print(f"🎬 Processing video with ENHANCED analytics: {os.path.basename(video_path)}")
        
        # Generate unique output name
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"enhanced_analyzed_{base_name}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Import and use the enhanced overlay script
        from fixed_video_overlay_with_analytics_enhanced import FixedVideoOverlayWithAnalytics
        
        # Create enhanced overlay processor
        enhanced_processor = FixedVideoOverlayWithAnalytics(video_path, output_path)
        
        # Process video with enhanced analytics
        print(f"🔄 Processing video with enhanced analytics...")
        enhanced_processor.process_video()
        
        # Check if processing was successful
        if not os.path.exists(output_path):
            return jsonify({"error": "Enhanced video processing failed"}), 500
        
        print(f"✅ Enhanced video processing completed: {output_path}")
        
        # Find the analytics file (should be created by the enhanced script)
        # The enhanced script saves with the original filename, not the temp filename
        original_filename = os.path.basename(video_path)
        if original_filename.startswith("temp_"):
            # Extract original filename from temp filename
            original_filename = original_filename.split("_", 2)[2] if "_" in original_filename else original_filename
        
        analytics_filename = f"fixed_analytics_{original_filename}.json"
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
        
        print(f"📤 Uploading enhanced processed video and analytics to GridFS...")
        print(f"📊 Video file path: {output_path}")
        print(f"📊 Video file exists: {os.path.exists(output_path)}")
        print(f"📊 Analytics file path: {analytics_path}")
        print(f"📊 Analytics file exists: {os.path.exists(analytics_path)}")
        if os.path.exists(analytics_path):
            file_size = os.path.getsize(analytics_path)
            print(f"📊 Analytics file size: {file_size} bytes")
        
        video_id, analytics_id = video_processor.upload_processed_video_to_gridfs(
            output_path, analytics_path, {
                "athlete_name": athlete_name,
                "event": event,
                "session_name": session_name,
                "user_id": user_id,
                "date": date,
                "original_filename": video_filename,
                "processed_video_filename": output_name,
                "processing_timestamp": timestamp,
                "enhanced_analytics": True
            }
        )
        
        if not video_id:
            return jsonify({"error": "Failed to upload enhanced processed video to MongoDB"}), 500
        
        # Create session record
        session_data = {
            "user_id": user_id,
            "athlete_name": athlete_name,
            "session_name": session_name,
            "event": event,
            "date": date,
            "duration": "00:00",
            "original_filename": video_filename,
            "processed_video_filename": os.path.basename(output_path),
            "processed_video_url": f"http://localhost:5004/getVideo?video_filename={os.path.basename(output_path)}",
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
            "notes": f"Video analyzed with ENHANCED analytics (anti-flickering, comprehensive metrics): {video_filename}",
            "status": "completed",
            "processing_progress": 1.0,
            "processing_status": "completed",
            "gridfs_video_id": video_id,
            "gridfs_analytics_id": analytics_id,
            "is_binary_stored": True,
            "has_landmarks": True,
            "enhanced_analytics": True,
            "created_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "updated_at": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
        
        session_id = sessions.create_session(session_data)
        print(f"✅ Enhanced session created: {session_id}")
        
        # Clean up temporary file if it was downloaded from GridFS
        if video_path.startswith("/tmp") or video_path.startswith(".") and "temp_" in video_path:
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up temporary file: {video_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file: {e}")
        
        return jsonify({
            "success": True,
            "message": "Enhanced video analysis completed and uploaded to MongoDB",
            "session_id": str(session_id),
            "video_id": str(video_id),
            "analytics_id": str(analytics_id) if analytics_id else None,
            "output_video": os.path.basename(output_path),
            "analytics_file": analytics_filename if analytics_id else None,
            "download_url": f"http://localhost:5004/getVideo?video_filename={os.path.basename(output_path)}",
            "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}" if analytics_id else None,
            "enhanced_analytics": True,
            "features": [
                "Anti-flickering pose overlay",
                "Temporal pose smoothing",
                "Comprehensive ACL risk assessment",
                "Enhanced knee valgus calculation",
                "Landing mechanics analysis",
                "Every frame processing",
                "Real-time analytics overlay"
            ],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in analyzevideo2: {e}")
        return jsonify({
            "error": "Enhanced video analysis failed",
            "details": str(e)
        }), 500

def convert_video_to_h264_for_browser(video_path, output_path):
    """Convert video to H.264 format for browser compatibility"""
    try:
        import subprocess
        import os
        
        # Check if output already exists
        if os.path.exists(output_path):
            return True
            
        # Convert to H.264 using ffmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'fast',   # Fast encoding
            '-crf', '23',        # Good quality
            '-c:a', 'aac',       # Audio codec
            '-movflags', '+faststart',  # Optimize for streaming
            '-y',                # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video converted to H.264: {output_path}")
            return True
        else:
            print(f"⚠️ H.264 conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error converting video to H.264: {e}")
        return False

@app.route('/getVideo', methods=['GET', 'HEAD'])
def get_video_for_frontend():
    """Get video for frontend display (converted to browser-compatible format)"""
    try:
        video_filename = request.args.get('video_filename')
        if not video_filename:
            return jsonify({"error": "video_filename parameter is required"}), 400
        
        # Find session by video filename
        session = sessions.get_session_by_video_filename(video_filename)
        if not session:
            return jsonify({"error": "Video not found"}), 404
        
        video_id = session.get('gridfs_video_id')
        if not video_id:
            return jsonify({"error": "Video not found in session"}), 404
        
        # Get video file info first
        video_info = video_processor.get_video_info_from_gridfs(video_id)
        if not video_info:
            return jsonify({"error": "Failed to retrieve video info from database"}), 500
        
        filename = video_info['filename'] or video_filename
        file_size = video_info['length']
        
        # For HEAD requests, just return headers without video data
        if request.method == 'HEAD':
            response = Response(
                status=200,
                mimetype='video/mp4',
                headers={
                    'Content-Length': str(file_size),
                    'Content-Disposition': f'inline; filename="{filename}"',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Cache-Control': 'public, max-age=3600'
                }
            )
            return response
        
        # Download video from GridFS to temporary file
        temp_video_path = os.path.join('/tmp', f"temp_{int(time.time())}_{filename}")
        try:
            with open(temp_video_path, 'wb') as temp_file:
                for chunk in video_processor.stream_video_from_gridfs(video_id, 0, file_size - 1)[0]:
                    temp_file.write(chunk)
            
            # Check if video is already H.264 (avoid double conversion)
            video_filename = os.path.basename(temp_video_path)
            is_already_h264 = 'h264' in video_filename.lower() or video_filename.startswith('h264_')
            
            if is_already_h264:
                # Video is already H.264, serve directly
                print(f"✅ Video is already H.264, serving directly: {video_filename}")
                with open(temp_video_path, 'rb') as video_file:
                    video_data = video_file.read()
                
                # Clean up temporary file
                os.remove(temp_video_path)
            else:
                # Convert to H.264 for browser compatibility
                h264_video_path = temp_video_path.replace('.mp4', '_h264.mp4')
                
                if convert_video_to_h264_for_browser(temp_video_path, h264_video_path):
                    # Serve the H.264 converted video
                    with open(h264_video_path, 'rb') as h264_file:
                        video_data = h264_file.read()
                    
                    # Clean up temporary files
                    os.remove(temp_video_path)
                    os.remove(h264_video_path)
                else:
                    # H.264 conversion failed, serve original video
                    with open(temp_video_path, 'rb') as video_file:
                        video_data = video_file.read()
                    
                    # Clean up temporary file
                    os.remove(temp_video_path)
                
                response = Response(
                    video_data,
                    status=200,
                    mimetype='video/mp4',
                    headers={
                        'Content-Length': str(len(video_data)),
                        'Content-Disposition': f'inline; filename="{filename}"',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Cache-Control': 'public, max-age=3600'
                    }
                )
                
                return response
            
            # Create response for already H.264 videos
            response = Response(
                video_data,
                status=200,
                mimetype='video/mp4',
                headers={
                    'Content-Length': str(len(video_data)),
                    'Content-Disposition': f'inline; filename="{filename}"',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Cache-Control': 'public, max-age=3600'
                }
            )
            
            return response
                
        except Exception as e:
            print(f"❌ Error processing video for browser: {e}")
            return jsonify({"error": "Failed to process video for browser"}), 500
        
    except Exception as e:
        print(f"❌ Error getting video for frontend: {e}")
        return jsonify({
            "error": "Failed to get video",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Gymnastics API Server (Updated with Railway MediaPipe)")
    print("=" * 60)
    
    # Check MongoDB connection
    try:
        collections = db_manager.db.list_collection_names()
        print(f"✅ Connected to MongoDB. Collections: {len(collections)}")
        for collection in collections:
            count = db_manager.db[collection].count_documents({})
            print(f"   {collection}: {count} documents")
    except Exception as e:
        print(f"❌ Error checking MongoDB: {e}")
    
    print("🌐 Server will be available at: http://localhost:5004")
    app.run(host='0.0.0.0', port=5004, debug=True)