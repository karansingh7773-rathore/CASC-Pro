# CASC Pro - Contextual Aware Security Cam

A Python-based intelligent security camera system that uses computer vision and AI to analyze motion events in real-time.

##  Overview

CASC combines local OpenCV motion detection with cloud-based AI analysis (via Open Router's Llama 4 Maverick) to provide contextual awareness of security events. The system uses a sophisticated two-thread architecture to ensure smooth video playback while performing intensive API operations in the background.

##  Architecture

### Two-Thread Model

**Thread 1: Main/Sentry Thread**
- Processes video frames in real-time
- Runs fast local OpenCV detection (Haar Cascades for faces/bodies)
- Manages state machine (Idle → Motion_Started → Motion_In_Progress → Motion_Stopped)
- Queues jobs for API analysis without blocking video

**Thread 2: Worker/Detective Thread**
- Processes queued analysis jobs
- Makes API calls to Open Router (3+ second response time)
- Writes results to Azure Cosmos DB
- Operates asynchronously to prevent video freezing

### State Machine Logic

1. **Idle**: No motion detected, system monitoring
2. **Motion Started**: Motion detected, event logged, waiting to classify as short/long
3. **Motion In Progress**: Continuous motion, rolling API analysis every 3.75 seconds
4. **Motion Stopped**: Event ends, fork logic determines final processing

##  Algorithm: Hybrid-Throttled-Event Model

- **Throttle**: 3.75 seconds between API calls (respects 16 calls/minute limit)
- **Short Events** (< 3.75s): Single API analysis comparing start/end frames
- **Long Events** (≥ 3.75s): Multiple rolling analyses + final summary generation

##  Core Technologies

- **Python 3.8+**: Main programming language
- **OpenCV**: Video processing and local motion detection
- **Threading & Queue**: Asynchronous job processing
- **Requests**: HTTP API calls to Open Router
- **Azure Cosmos DB**: NoSQL storage for event logs and insights
- **Open Router**: API gateway for Llama 4 Maverick vision model

##  Installation

### 1. Clone or Create Project Directory

```bash
mkdir d:\CASC2
cd d:\CASC2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# Open Router API
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Azure Cosmos DB
COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
COSMOS_KEY=your_cosmos_primary_key_here

# Azure Blob Storage (optional, for frame storage)
BLOB_CONNECTION_STRING=your_blob_connection_string_here
```

### 4. Update config.py

Edit `config.py` to set your credentials if not using environment variables:

```python
OPENROUTER_API_KEY = "sk-or-v1-..."
COSMOS_ENDPOINT = "https://your-account.documents.azure.com:443/"
COSMOS_KEY = "your-key-here..."
```

## 🎬 Video Setup & Upload

### Option 1: Using Your Webcam (Default)

The system uses your webcam by default. No video upload needed!

```python
# In config.py
VIDEO_SOURCE = 0  # Uses default webcam
```

### Option 2: Using a Video File (Recommended for Testing)

**Step 1: Create Videos Folder**

```bash
# Create the videos directory
mkdir d:\CASC2\videos
```

**Step 2: Add Your Video**

Copy your video file to `d:\CASC2\videos\`. Supported formats:
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`
- `.wmv`
- `.flv`

Example:
```
d:\CASC2\videos\
  ├── my_security_footage.mp4
  ├── test_video.mp4
  └── sample_motion.avi
```

**Step 3: Update Config**

Edit `config.py` and set the video path:

```python
# In config.py
VIDEO_SOURCE = r"d:\CASC2\videos\my_security_footage.mp4"
```

**Step 4: Interactive Selection (Easiest)**

Run the video analyzer test:

```bash
python test_video_analyzer.py
```

This will:
- Show all videos in your `videos` folder
- Let you preview and select a video
- Auto-update `config.py` for you

### Where to Place Your Videos

```
📁 d:\CASC2\
   ├── main.py
   ├── config.py
   ├── ...
   └── 📁 videos\          ← Create this folder
       ├── video1.mp4      ← Your videos go here
       ├── video2.mp4
       └── test_footage.avi
```

**Important Notes:**
- Use the full path with `r"..."` prefix in config.py
- Video files can be large - ensure you have enough disk space
- For best results, use videos with clear motion events

## 🧪 Testing

### Test Suite Overview

Run tests to verify each component before running the full system:

| Test File | Purpose | Command |
|-----------|---------|---------|
| `test_cosmos_db.py` | Test database operations | `python test_cosmos_db.py` |
| `test_api_helper.py` | Test API calls to Open Router | `python test_api_helper.py` |
| `test_qa_function.py` | Test Q&A functionality | `python test_qa_function.py` |
| `test_video_analyzer.py` | Select and preview videos | `python test_video_analyzer.py` |

### Running Tests

**1. Test Cosmos DB Connection**

```bash
python test_cosmos_db.py
```

This will:
- ✅ Create a test event
- ✅ Add insights
- ✅ Update event status
- ✅ Query events
- ✅ Verify all database operations

**2. Test API Integration**

```bash
python test_api_helper.py
```

This will:
- ✅ Test vision analysis API
- ✅ Test summary generation API
- ✅ Test Q&A API
- ⚠️ Requires valid API key in config.py

**3. Test Q&A Function**

```bash
python test_qa_function.py
```

This will:
- ✅ Create a test event with insights
- ✅ Run sample questions
- ✅ Verify AI can answer questions about events

**4. Select and Test Video**

```bash
python test_video_analyzer.py
```

This will:
- 📹 List all videos in `d:\CASC2\videos\`
- 🎬 Let you preview video properties
- ⚙️ Auto-update config.py with selected video

### Quick Test Checklist

Before running the main application:

- [ ] Created `d:\CASC2\videos\` folder
- [ ] Added test video to videos folder
- [ ] Updated API keys in `config.py`
- [ ] Ran `test_cosmos_db.py` successfully
- [ ] Ran `test_api_helper.py` successfully  
- [ ] Ran `test_video_analyzer.py` and selected video
- [ ] Ready to run `python main.py`!

##  Usage

### Running the Main Application

```bash
python main.py
```

**Before First Run:**
1. ✅ Configure your API keys in `config.py`
2. ✅ Select your video source (webcam or file)
3. ✅ Run tests to verify setup

**Controls:**
- Press `q` to quit the application
- The video window shows live detection and motion status

### Video Source Options

**Using Webcam:**
```python
# In config.py
VIDEO_SOURCE = 0  # Default webcam
VIDEO_SOURCE = 1  # Second webcam (if available)
```

**Using Video File:**
```python
# In config.py
VIDEO_SOURCE = r"d:\CASC2\videos\test_video.mp4"
```

**Interactive Selection:**
```bash
python test_video_analyzer.py
# Then follow prompts to select video
```

### Q&A Function

Query past events using natural language:

```python
from qa_function import ask_question

answer = ask_question("was anyone on the ground during the fight?")
print(answer)
```

Or run standalone:

```bash
python qa_function.py
```

##  Project Structure

```
d:\CASC2\
├── main.py              # Main application with two-thread architecture
├── config.py            # Configuration and constants
├── cosmos_db.py         # Cosmos DB helper functions
├── api_helper.py        # Open Router API integration
├── qa_function.py       # Q&A query function
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

##  Event Data Schema

Each security event is stored in Cosmos DB with this structure:

```json
{
  "id": "evt_1729603325",
  "StartTime": "2025-10-22T12:02:05Z",
  "EndTime": "2025-10-22T12:12:05Z",
  "Status": "Complete",
  "StartFrameURL": "https://...",
  "EventSummary": "A fight broke out between 4 men...",
  "Insights": [
    {
      "Timestamp": "2025-10-22T12:02:08Z",
      "Analysis": {
        "Summary": "A group of 4 men are gathering...",
        "People Count": 4,
        "Emotions/Posture": "agitated",
        "Threat Assessment": 6,
        "Key Objects": ["none"]
      }
    }
  ]
}
```

##  AI Analysis Features

For each motion event, the AI analyzes:

1. **Summary**: One-sentence description of the action
2. **People Count**: Number of visible people
3. **Emotions/Posture**: Behavioral assessment (calm, agitated, fighting, etc.)
4. **Threat Assessment**: Scale of 1-10 (Calm to Critical)
5. **Key Objects**: Relevant objects detected (package, bag, weapon, etc.)

##  Configuration Options

Edit `config.py` to customize:

- `THROTTLE_SECONDS`: Time between API calls (default: 3.75s)
- `MOTION_DETECTION_THRESHOLD`: Sensitivity of motion detection (default: 5000 pixels)
- `VIDEO_SOURCE`: Camera index or video file path
- Model selection and API endpoints

##  Troubleshooting

**Video won't open:**
- Check `VIDEO_SOURCE` in config.py uses `r"..."` prefix for paths
- Verify video file exists at the specified path
- Ensure video format is supported (.mp4, .avi, .mov, etc.)
- Run `test_video_analyzer.py` to check video properties
- For webcam: ensure it's not being used by another application

**No videos found in test_video_analyzer.py:**
- Create the folder: `mkdir d:\CASC2\videos`
- Copy video files to that folder
- Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv

**Motion detection not working:**
- Adjust `MOTION_DETECTION_THRESHOLD` in config.py
- Too high = less sensitive (needs more movement)
- Too low = more sensitive (detects small changes)
- Recommended range: 3000-8000

##  Performance Notes

- **Video FPS**: 20-30 FPS (local processing only)
- **API Response Time**: 3-5 seconds per analysis
- **Database Latency**: < 100ms for writes
- **Queue Processing**: Asynchronous, no blocking

##  Security Considerations

- Store API keys in environment variables, not in code
- Use Azure Key Vault for production deployments
- Implement proper access controls on Cosmos DB
- Consider encrypting stored frame data

##  License

This is a prototype project. Add your license here.

##  Contributing

This is a prototype. Contributions welcome!

## Support

For issues or questions, please create an issue in the repository.

---

**Built with ❤️ for contextual security awareness**