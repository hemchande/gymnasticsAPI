"""
MongoDB Database Configuration and Models for Gymnastics Analytics
"""

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
from bson import ObjectId
from urllib.parse import quote_plus

# MongoDB Configuration - Following JS example pattern
username = "hemchande"
password = "He10072638"  # Using the password from your JS example (without special chars)
escaped_password = quote_plus(password)
MONGODB_URI = "mongodb+srv://hemchande:He10072638@cluster0.dhmdwbm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "gymnastics_analytics"

class DatabaseManager:
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB with improved timeout handling"""
        try:
            # Connection options with better timeout handling and SSL bypass for development
            self.client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=10000,  # Increased to 10 seconds
                connectTimeoutMS=20000,          # 20 second connection timeout
                socketTimeoutMS=60000,           # 60 second socket timeout
                server_api=ServerApi('1'),
                # Additional options for better compatibility
                retryWrites=True,
                w='majority',
                # Add heartbeat frequency for better connection monitoring
                heartbeatFrequencyMS=10000,
                # SSL options for development (bypass certificate verification)
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True
            )
            self.db = self.client[DATABASE_NAME]
            
            # Send a ping to confirm connection with timeout
            self.client.admin.command('ping', maxTimeMS=5000)
            print("✅ Successfully connected to MongoDB!")
            
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            raise e
    
    def get_collection(self, collection_name: str):
        """Get a collection from the database"""
        return self.db[collection_name]
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()

# Database Collections
class User:
    def __init__(self, db_manager: DatabaseManager):
        self.collection = db_manager.get_collection("users")
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user"""
        user_data["created_at"] = datetime.utcnow()
        user_data["updated_at"] = datetime.utcnow()
        
        result = self.collection.insert_one(user_data)
        return str(result.inserted_id)
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = self.collection.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception:
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user data"""
        update_data["updated_at"] = datetime.utcnow()
        result = self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

class Session:
    def __init__(self, db_manager: DatabaseManager):
        self.collection = db_manager.get_collection("sessions")
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new session"""
        session_data["created_at"] = datetime.utcnow()
        session_data["updated_at"] = datetime.utcnow()
        
        result = self.collection.insert_one(session_data)
        return str(result.inserted_id)
    
    def upsert_session(self, session_data: Dict[str, Any]) -> str:
        """Upsert session (insert or update if exists)"""
        # Use filename as unique identifier for upsert
        filename = session_data.get("original_filename")
        if not filename:
            raise ValueError("original_filename is required for upsert")
        
        existing_session = self.collection.find_one({"original_filename": filename})
        
        session_data["updated_at"] = datetime.utcnow()
        
        if existing_session:
            # Update existing session
            result = self.collection.update_one(
                {"original_filename": filename},
                {"$set": session_data}
            )
            return str(existing_session["_id"])
        else:
            # Create new session
            session_data["created_at"] = datetime.utcnow()
            result = self.collection.insert_one(session_data)
            return str(result.inserted_id)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            session = self.collection.find_one({"_id": ObjectId(session_id)})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception:
            return None
    
    def get_sessions_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        try:
            sessions = list(self.collection.find({"user_id": user_id}))
            for session in sessions:
                session["_id"] = str(session["_id"])
            return sessions
        except Exception:
            return []
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions (for admin/demo purposes)"""
        try:
            sessions = list(self.collection.find({}))
            for session in sessions:
                session["_id"] = str(session["_id"])
            return sessions
        except Exception:
            return []
    
    def get_session_by_video_filename(self, video_filename: str) -> Optional[Dict[str, Any]]:
        """Get session by video filename (for frontend compatibility)"""
        try:
            # Try to find by processed_video_filename first
            session = self.collection.find_one({"processed_video_filename": video_filename})
            if session:
                session["_id"] = str(session["_id"])
                return session
            
            # Try to find by original_filename
            session = self.collection.find_one({"original_filename": video_filename})
            if session:
                session["_id"] = str(session["_id"])
                return session
            
            # Try to find by any filename field that contains the video filename
            session = self.collection.find_one({
                "$or": [
                    {"processed_video_filename": {"$regex": video_filename, "$options": "i"}},
                    {"original_filename": {"$regex": video_filename, "$options": "i"}}
                ]
            })
            if session:
                session["_id"] = str(session["_id"])
                return session
            
            return None
        except Exception:
            return None
    
    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update session data"""
        update_data["updated_at"] = datetime.utcnow()
        result = self.collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        result = self.collection.delete_one({"_id": ObjectId(session_id)})
        return result.deleted_count > 0

class VideoMetadata:
    def __init__(self, db_manager: DatabaseManager):
        self.collection = db_manager.get_collection("video_metadata")
    
    def create_video_metadata(self, metadata: Dict[str, Any]) -> str:
        """Create video metadata record"""
        metadata["created_at"] = datetime.utcnow()
        metadata["updated_at"] = datetime.utcnow()
        
        result = self.collection.insert_one(metadata)
        return str(result.inserted_id)
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video metadata by ID"""
        try:
            video = self.collection.find_one({"_id": ObjectId(video_id)})
            if video:
                video["_id"] = str(video["_id"])
            return video
        except Exception:
            return None
    
    def get_video_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get video metadata by filename"""
        video = self.collection.find_one({"filename": filename})
        if video:
            video["_id"] = str(video["_id"])
        return video

# Database Schema Definitions
def create_session_schema():
    """Define the session document schema"""
    return {
        "user_id": str,  # Reference to user
        "athlete_name": str,
        "session_name": str,
        "event": str,  # Floor, Vault, Bars, Beam
        "date": str,  # ISO date string
        "duration": str,  # Duration in MM:SS format
        
        # Video files
        "original_filename": str,  # Original uploaded filename
        "processed_video_filename": str,  # Processed video with overlays
        "processed_video_url": str,  # URL to access processed video
        "analytics_filename": str,  # JSON analytics file
        "analytics_url": str,  # URL to access analytics
        
        # Analysis results
        "motion_iq": float,
        "acl_risk": float,
        "precision": float,
        "power": float,
        "tumbling_percentage": float,
        
        # Status and metadata
        "status": str,  # pending, processing, completed, failed
        "processing_progress": float,  # 0.0 to 1.0
        
        # Analysis details
        "total_frames": int,
        "fps": float,
        "has_landmarks": bool,
        "landmark_confidence": float,
        
        # Notes and feedback
        "notes": str,
        "coach_notes": str,
        "highlights": List[str],
        "areas_for_improvement": List[str],
        
        # Timestamps
        "created_at": datetime,
        "updated_at": datetime
    }

def create_user_schema():
    """Define the user document schema"""
    return {
        "username": str,
        "email": str,
        "role": str,  # coach, athlete, admin
        "first_name": str,
        "last_name": str,
        
        # Preferences
        "preferences": {
            "default_event": str,
            "notification_settings": Dict[str, Any]
        },
        
        # Timestamps
        "created_at": datetime,
        "updated_at": datetime
    }

def create_video_metadata_schema():
    """Define the video metadata document schema"""
    return {
        "filename": str,
        "original_filename": str,
        "file_size": int,  # in bytes
        "duration": float,  # in seconds
        "fps": float,
        "resolution": {
            "width": int,
            "height": int
        },
        "codec": str,
        "bitrate": int,
        
        # Processing status
        "processing_status": str,  # pending, processing, completed, failed
        "processing_error": str,
        
        # File paths
        "original_path": str,
        "processed_path": str,
        "analytics_path": str,
        
        # Timestamps
        "uploaded_at": datetime,
        "processed_at": datetime,
        "created_at": datetime,
        "updated_at": datetime
    }

# Try to initialize MongoDB, fallback to local storage if it fails
try:
    db_manager = DatabaseManager()
    # Export MongoDB collections
    users = User(db_manager)
    sessions = Session(db_manager)
    video_metadata = VideoMetadata(db_manager)
    print("✅ Using MongoDB for data storage")
except Exception as e:
    print(f"⚠️  MongoDB not available ({e}), falling back to local storage")
    # Import local storage fallback
    from local_storage import local_users, local_sessions, local_video_metadata
    # Use local storage
    users = local_users
    sessions = local_sessions
    video_metadata = local_video_metadata
    # Create a dummy db_manager for compatibility
    class DummyDBManager:
        def __init__(self):
            self.client = None
            self.db = None
        def get_collection(self, name):
            return None
        def close(self):
            pass
    db_manager = DummyDBManager()
    print("✅ Using local storage for data storage")

if __name__ == "__main__":
    # Test storage
    try:
        if hasattr(db_manager, 'client') and db_manager.client:
            # MongoDB test
            db_manager.client.admin.command('ping')
            print("✅ MongoDB connection successful!")
            print(f"✅ Database: {DATABASE_NAME}")
            print(f"✅ Collections: {db_manager.db.list_collection_names()}")
        else:
            # Local storage test
            print("✅ Local storage test:")
            print(f"✅ Sessions count: {len(sessions.get_all_sessions())}")
            if hasattr(users, 'storage_manager'):
                users_count = len(users.storage_manager._read_json_file(users.file_path))
                video_count = len(video_metadata.storage_manager._read_json_file(video_metadata.file_path))
                print(f"✅ Users count: {users_count}")
                print(f"✅ Video metadata count: {video_count}")
            else:
                print("✅ Using MongoDB collections")
        
    except Exception as e:
        print(f"❌ Storage test failed: {e}")

