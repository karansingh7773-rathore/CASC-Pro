"""
Configuration Template for CASC Pro
Copy this file to 'config.py' and fill in your actual credentials.
NEVER commit config.py to GitHub!
"""
import os

# ============================================================================
# API KEYS - REPLACE WITH YOUR ACTUAL KEYS
# ============================================================================

# Open Router API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Google AI Studio Configuration (for Gemini 2.0 Flash)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_ai_studio_api_key_here")

# NVIDIA API Configuration (for Q&A)
# Get your key from: https://build.nvidia.com/
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "nvapi-your_nvidia_api_key_here")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1"

# Model Configuration - CASC Pro
LIVE_COMMENTATOR_MODEL = "nousresearch/nous-hermes-2-vision-7b:free"
QNA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"  # NVIDIA Nemotron for Q&A
GEMINI_MODEL = "gemini-2.0-flash-exp"

# ============================================================================
# AZURE COSMOS DB - REPLACE WITH YOUR CREDENTIALS
# ============================================================================

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://your-account.documents.azure.com:443/")
COSMOS_KEY = os.getenv("COSMOS_KEY", "your_cosmos_primary_key_here")
COSMOS_DATABASE_NAME = "CASC_DB"
COSMOS_CONTAINER_NAME = "SecurityEvents"

# ============================================================================
# ALGORITHM CONSTANTS
# ============================================================================

THROTTLE_SECONDS = 3.75

# ============================================================================
# YOLO CONFIGURATION
# ============================================================================

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_PERSON_CLASS_ID = 0

# ============================================================================
# VIDEO CONFIGURATION
# ============================================================================

# Default: Use webcam
VIDEO_SOURCE = 0

# Or use a video file:
# VIDEO_SOURCE = r"D:\CASC2\videos\your_video.mp4"

# Video storage
TEMP_VIDEO_DIR = "d:\\CASC2\\temp_videos"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
