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
from database2 import db_manager, sessions, users, video_metadata

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8080", "https://motionlabsai-qb7r-5ljq6f0oj-hemchandeishagmailcoms-projects.vercel.app","https://www.motionlabsai.com"], 
     allow_headers=["Content-Type", "Authorization", "Range"],
     expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])

# Configuration
MEDIAPIPE_SERVER_URL = "https://extraordinary-gentleness-production.up.railway.app"
VIDEO_PROCESSING_DIR = "../output_videos"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_videos")
ANALYTICS_DIR = "../analytics"

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
            if os.path.exists(analytics_path):
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
    """Get list of processed videos (frontend compatibility)"""
    try:
        # Get all sessions with video information
        sessions_list = sessions.get_all_sessions()
        
        processed_videos = []
        for session in sessions_list:
            if session.get('gridfs_video_id'):
                try:
                    video_info = video_processor.get_video_info_from_gridfs(session['gridfs_video_id'])
                    if video_info:
                        processed_videos.append({
                            'original_filename': session.get('original_filename', 'Unknown'),
                            'processed_filename': video_info['filename'],
                            'file_size_mb': round(video_info['length'] / (1024 * 1024), 2),
                            'analysis_type': 'pose_detection',
                            'has_analytics': session.get('gridfs_analytics_id') is not None
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

@app.route('/getJobStatus', methods=['GET'])
def get_job_status():
    """Get job status (frontend compatibility)"""
    try:
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({"error": "job_id parameter is required"}), 400
        
        # For now, return completed status
        # In a real implementation, this would check actual job status
        return jsonify({
            "status": "completed",
            "video_filename": "unknown",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "analytics_file": "analytics.json",
            "total_frames": 1000,
            "frames_processed": 1000,
            "overlay_video": "overlay.mp4",
            "overlay_success": True,
            "overlay_message": "Overlay completed successfully"
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

@app.route('/getPerFrameStatistics', methods=['GET'])
def get_per_frame_statistics_by_filename():
    """Get per-frame statistics by video filename (frontend compatibility)"""
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
        
        # Format response to match frontend expectations
        response_data = {
            "success": True,
            "video_filename": video_filename,
            "total_frames": session.get('total_frames', 0),
            "fps": session.get('fps', 30),
            "frames_processed": session.get('total_frames', 0),
            "processing_time": "00:00:01",
            "enhanced_analytics": True,
            "frame_data": analytics_data.get('frame_data', []) if isinstance(analytics_data, dict) else analytics_data,
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
            "processed_video_filename": video_file.filename,
            "processed_video_url": f"http://localhost:5004/downloadVideo/{video_id}",
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
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "video_id": str(video_id),
            "message": "Video uploaded successfully",
            "timestamp": datetime.now().isoformat()
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
        
        # Generate unique output name
        base_name = os.path.splitext(video_filename)[0]
        timestamp = int(time.time())
        output_name = f"analyzed_{base_name}_{timestamp}.mp4"
        
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
                    "processed_video_url": f"http://localhost:5004/downloadVideo/{video_id}",
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
                
                # Create session in MongoDB
                session_id = sessions.create_session(session_data)
                
                return jsonify({
                    "success": True,
                    "session_id": session_id,
                    "video_id": str(video_id),
                    "analytics_id": str(analytics_id) if analytics_id else None,
                    "output_video": os.path.basename(result.get("h264_video", result["output_video"])),
                    "analytics_file": os.path.basename(result["analytics_file"]) if result["analytics_file"] else None,
                    "message": "Video analysis completed and uploaded to MongoDB",
                    "download_url": f"http://localhost:5004/downloadVideo/{session_id}",
                    "analytics_url": f"http://localhost:5004/getAnalytics/{session_id}",
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
        
        # First try to find video locally (existing behavior)
        best_video_path, is_h264, original_path = video_processor.find_best_video(video_filename)
        
        if best_video_path and os.path.exists(best_video_path):
            print(f"✅ Found local video: {best_video_path}")
            video_path = best_video_path
        else:
            # Try to find video in GridFS by filename
            print(f"🔍 Video not found locally, searching GridFS for: {video_filename}")
            
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
                
                if not gridfs_file:
                    return jsonify({"error": f"Video file not found in GridFS: {video_filename}"}), 404
                
                print(f"✅ Found video in GridFS: {gridfs_file.filename}")
                
                # Download video from GridFS to temporary local file
                temp_dir = "/tmp" if os.path.exists("/tmp") else "."
                temp_filename = f"temp_{int(time.time())}_{gridfs_file.filename}"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                print(f"📥 Downloading video to: {temp_path}")
                with open(temp_path, 'wb') as temp_file:
                    temp_file.write(gridfs_file.read())
                
                video_path = temp_path
                print(f"✅ Video downloaded successfully: {os.path.getsize(temp_path)} bytes")
                
            except Exception as e:
                print(f"❌ Error accessing GridFS: {e}")
                return jsonify({"error": f"Failed to access video in GridFS: {str(e)}"}), 500
        
        # Process the video (same as original analyzeVideo)
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        # Generate unique output name
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"h264_analyzed_{base_name}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process video with analytics
        print(f"🔄 Processing video with MediaPipe...")
        success = video_processor.process_video_with_analytics(video_path, output_name)
        
        if not success:
            return jsonify({"error": "Video processing failed"}), 500
        
        print(f"✅ Video processing completed: {output_path}")
        
        # Upload processed video and analytics to GridFS
        analytics_filename = f"fixed_analytics_{os.path.basename(video_path)}.json"
        analytics_path = os.path.join(".", analytics_filename)
        
        if not os.path.exists(analytics_path):
            analytics_path = os.path.join(OUTPUT_DIR, analytics_filename)
        
        print(f"📤 Uploading processed video and analytics to GridFS...")
        video_id, analytics_id = video_processor.upload_processed_video_to_gridfs(
            output_path, analytics_path, {
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
            "processed_video_filename": output_name,
            "processed_video_url": f"http://localhost:5004/downloadVideo/{video_id}",
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
            "output_video": output_name,
            "analytics_file": analytics_filename if analytics_id else None,
            "download_url": f"http://localhost:5004/downloadVideo/{video_id}",
            "analytics_url": f"http://localhost:5004/getAnalytics/{analytics_id}" if analytics_id else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in analyzeVideo1: {e}")
        return jsonify({
            "error": "Video analysis failed",
            "details": str(e)
        }), 500

@app.route('/getVideo', methods=['GET'])
def get_video_for_frontend():
    """Get video for frontend display (simplified, no range requests)"""
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
        
        # Stream the entire video without range requests
        stream_generator, total_size, stream_start, stream_end = video_processor.stream_video_from_gridfs(
            video_id, 0, file_size - 1
        )
        
        if stream_generator is None:
            return jsonify({"error": "Failed to stream video from database"}), 500
        
        response = Response(
            stream_generator,
            status=200,  # OK
            mimetype=content_type,
            headers={
                'Content-Length': str(file_size),
                'Content-Disposition': f'inline; filename="{filename}"',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Cache-Control': 'public, max-age=3600'
            }
        )
        
        return response
        
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
