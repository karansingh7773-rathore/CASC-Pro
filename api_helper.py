import requests
import base64
import cv2
import json
from datetime import datetime
import config
import os
import time
import google.generativeai as genai

def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def encode_video_to_base64(video_path):
    """Encode video file to base64 string."""
    with open(video_path, 'rb') as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

# =========================================================================
# MAVERICK: Live Commentary (NOT saved to database)
# ========================================================================

def call_maverick_live_commentary(frame_before, frame_now):
    """
    Call Maverick for live security commentary during motion.
    This is NOT saved to database - only displayed in terminal.
    """
    base64_before = encode_frame_to_base64(frame_before)
    base64_now = encode_frame_to_base64(frame_now)
    
    prompt = """You are a live security commentator (CASC AI). Describe the CHANGE between 'before' and 'after' and the CURRENT status. Be brief.
- Current Action/Change:
- People Count Now:
- Current Posture/Emotion:
- Current Threat Level (1-10):"""
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.MAVERICK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_before}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_now}"}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        print("\n" + "="*70)
        print("[MAVERICK - LIVE COMMENTARY]")
        print("="*70)
        
        response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        commentary = result['choices'][0]['message']['content'].strip()
        print(commentary)
        print("="*70 + "\n")
        
    except requests.exceptions.HTTPError as e:
        if "429" in str(e) or "503" in str(e):
            print(f"[MAVERICK] Service temporarily unavailable, trying fallback model...")
            try_fallback_model(frame_before, frame_now)
        else:
            print(f"[MAVERICK ERROR] Live commentary failed: {e}\n")
    except Exception as e:
        print(f"[MAVERICK ERROR] Live commentary failed: {e}\n")

def try_fallback_model(frame_before, frame_now):
    """Try a fallback model if the primary Maverick model fails."""
    fallback_models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "google/gemini-flash-1.5:free"
    ]
    
    base64_before = encode_frame_to_base64(frame_before)
    base64_now = encode_frame_to_base64(frame_now)
    
    # Simplified prompt for fallback (text-only models)
    prompt = """Analyze these two security camera frames (before/after). Describe:
1. What changed between the frames?
2. How many people are visible?
3. Current threat level (1-10)?
4. Brief action description?

Before frame: [IMAGE_DATA]
After frame: [IMAGE_DATA]"""
    
    for model in fallback_models:
        try:
            print(f"[MAVERICK] Trying fallback model: {model}")
            
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # For text-only models, use a simplified approach
            if "instruct" in model and "vision" not in model:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Security camera detected motion. Provide brief commentary: Action/Change, People Count, Emotion/Posture, Threat Level (1-10). Keep it under 50 words."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            else:
                # For vision models
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Brief security commentary on these frames:"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_before}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_now}"}}
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            
            response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            
            commentary = result['choices'][0]['message']['content'].strip()
            print("\n" + "="*70)
            print(f"[MAVERICK - FALLBACK: {model}]")
            print("="*70)
            print(commentary)
            print("="*70 + "\n")
            return  # Success, exit function
            
        except Exception as e:
            print(f"[MAVERICK] Fallback model {model} failed: {e}")
            continue
    
    # If all models fail
    print("\n" + "="*70)
    print("[MAVERICK - BASIC MOTION DETECTED]")
    print("="*70)
    print("- Current Action/Change: Motion detected in frame")
    print("- People Count Now: Unknown (API unavailable)")
    print("- Current Posture/Emotion: Unknown")
    print("- Current Threat Level (1-10): 5 (default)")
    print("="*70 + "\n")

def call_maverick_live_commentary_text_only():
    """
    Simple text-only live commentary when vision models fail.
    """
    import time
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    prompt = f"You are a security AI. Motion was detected at {current_time}. Provide a brief 2-line security status update mentioning: current alert level, recommended action, and general assessment. Be professional and concise."
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.MAVERICK_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        commentary = result['choices'][0]['message']['content'].strip()
        
        print("\n" + "="*70)
        print("[MAVERICK - LIVE SECURITY UPDATE]")
        print("="*70)
        print(f"Time: {current_time}")
        print(f"Status: {commentary}")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("[MAVERICK - MOTION DETECTED]")
        print("="*70)
        print(f"Time: {current_time}")
        print("Status: Motion detected - Monitoring continues")
        print("Alert Level: Medium - Standard surveillance protocol")
        print("="*70 + "\n")

# ============================================================================
# GEMINI 2.5 FLASH: Video Analysis via Google AI Studio (SAVED to database)
# ============================================================================

def upload_video_to_gemini(video_path):
    """
    Upload video to Google AI Studio using the Files API.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        str: File URI for the uploaded video
    """
    try:
        # Configure the API
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        print(f"[GEMINI] Uploading video file: {video_path}")
        
        # Upload the file
        video_file = genai.upload_file(path=video_path)
        
        print(f"[GEMINI] Upload complete. File URI: {video_file.uri}")
        
        # Wait for file to be processed
        while video_file.state.name == "PROCESSING":
            print("[GEMINI] Processing video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        return video_file.uri
        
    except Exception as e:
        print(f"[GEMINI ERROR] Failed to upload video: {e}")
        return None

def upload_video_to_gemini_api(video_path):
    """Upload video using Google AI File API (REST) - Simplified approach."""
    try:
        print(f"[GEMINI] Uploading video file: {video_path}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[GEMINI] Video size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:
            print("[GEMINI ERROR] Video file too large (max 100MB)")
            return None
        
        # Use multipart upload (simpler method)
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={config.GOOGLE_API_KEY}"
        
        with open(video_path, 'rb') as video_file:
            files = {
                'file': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            
            # Metadata as form data
            data = {
                'file': json.dumps({
                    'display_name': os.path.basename(video_path)
                })
            }
            
            headers = {
                'X-Goog-Upload-Protocol': 'multipart'
            }
            
            response = requests.post(url, headers=headers, files=files, timeout=120)
        
        if response.status_code not in [200, 201]:
            print(f"[GEMINI ERROR] Upload failed: {response.text}")
            return None
        
        result = response.json()
        file_info = result.get('file', {})
        file_uri = file_info.get('uri')
        file_name = file_info.get('name')
        
        if not file_uri:
            print(f"[GEMINI ERROR] No file URI in response: {result}")
            return None
        
        print(f"[GEMINI] Video uploaded successfully: {file_name}")
        
        # Wait for processing
        print("[GEMINI] Waiting for video processing...")
        max_wait = 120  # Increased timeout for larger videos
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={config.GOOGLE_API_KEY}"
            status_response = requests.get(status_url)
            
            if status_response.status_code == 200:
                state = status_response.json().get('state')
                
                if state == 'ACTIVE':
                    print("[GEMINI] Video processing complete")
                    return file_uri
                elif state == 'FAILED':
                    print("[GEMINI ERROR] Video processing failed")
                    return None
                elif state == 'PROCESSING':
                    print("[GEMINI] Still processing... (patience)")
            
            time.sleep(3)
        
        print("[GEMINI ERROR] Video processing timeout")
        return None
        
    except Exception as e:
        print(f"[GEMINI ERROR] Failed to upload video: {e}")
        import traceback
        traceback.print_exc()
        return None


def call_gemini_video_analysis(video_path, event_id):
    """
    Call Google AI Studio's Gemini 2.5 Flash to analyze the entire event video.
    This IS saved to database as the official record.
    
    Args:
        video_path: Path to the recorded event video
        event_id: Event identifier
    
    Returns:
        str: Comprehensive summary from Gemini
    """
    
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"[GEMINI ERROR] Video file not found: {video_path}")
            return None
        
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"\n[GEMINI] Analyzing video: {os.path.basename(video_path)} ({video_size_mb:.2f} MB)")
        
        # Upload video to Gemini
        file_uri = upload_video_to_gemini(video_path)
        
        if not file_uri:
            print("[GEMINI ERROR] Failed to upload video")
            return None
        
        # Prepare the generation request
        prompt = """You are analyzing security camera footage. Provide a comprehensive, detailed summary of all significant actions, people, objects, and the overall narrative in this security camera video footage.

Include:
1. OVERVIEW: What happened overall in this event?
2. TIMELINE: Describe the sequence of events from start to finish
3. PEOPLE: Who was involved? How many people? What did they do?
4. OBJECTS: Any significant objects or items visible?
5. THREAT ASSESSMENT: Overall security evaluation (Low/Medium/High) and why
6. KEY OBSERVATIONS: Critical details a security officer should know
7. RECOMMENDATIONS: Should this require further review or action?

Be thorough, detailed, and professional."""
        
        generation_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={config.GOOGLE_API_KEY}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "file_data": {
                                "mime_type": "video/mp4",
                                "file_uri": file_uri
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }
        
        print("[GEMINI] Generating comprehensive video analysis...")
        print("="*70)
        
        response = requests.post(generation_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Extract the summary
        if 'candidates' in result and len(result['candidates']) > 0:
            summary = result['candidates'][0]['content']['parts'][0]['text']
            
            print("\n" + "="*70)
            print("[GEMINI 2.5 FLASH - VIDEO ANALYSIS SUMMARY]")
            print("="*70)
            print(summary)
            print("="*70 + "\n")
            
            return summary
        else:
            print("[GEMINI ERROR] No response generated")
            return None
        
    except Exception as e:
        print(f"[GEMINI ERROR] Video analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Q&A: Using Free Open Router Model (reads Gemini 2.5's saved summary)
# ============================================================================

def call_qna_model(event_summary, question):
    """
    Call free Open Router model for Q&A based on Gemini 2.5's saved video summary.
    
    Args:
        event_summary: The comprehensive summary from Gemini 2.5 video analysis
        question: User's question
    
    Returns:
        str: Answer from the Q&A model
    """
    prompt = f"""You are answering questions about a security camera event based on the analysis summary below.

SECURITY EVENT SUMMARY (from Gemini 2.5 Flash):
{event_summary}

USER QUESTION: {question}

Provide a clear, accurate answer based ONLY on the information in the summary above. If the information is not in the summary, say so."""
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.QNA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        print(f"[Q&A ERROR] Failed: {e}")
        return None

# =========================================================================
# LIVE COMMENTATOR: Context-Aware Live Updates
# ============================================================================

def call_live_commentator(frame_before, frame_now, previous_commentary=None):
    """
    Call Open Router image model for context-aware live commentary.
    
    Args:
        frame_before: Previous frame
        frame_now: Current frame
        previous_commentary: Previous commentary from DB (for context)
    
    Returns:
        str: Current analysis/commentary
    """
    base64_before = encode_frame_to_base64(frame_before)
    base64_now = encode_frame_to_base64(frame_now)
    
    # Build context-aware prompt
    if previous_commentary:
        prompt = f"""You are a live security camera AI providing continuous updates.

PREVIOUS UPDATE (3.75 seconds ago):
{previous_commentary}

Now analyze the NEW 'before' and 'after' frames and describe what has CHANGED or PROGRESSED since the previous update.

Provide:
- Current Action/Change: (How has the situation evolved?)
- People Count Now:
- Current Posture/Emotion:
- Current Threat Level (1-10):

Be concise and focus on changes."""
    else:
        prompt = """You are a live security camera AI. Analyze these frames and provide initial assessment.

Provide:
- Current Action/Change:
- People Count Now:
- Current Posture/Emotion:
- Current Threat Level (1-10):"""
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.LIVE_COMMENTATOR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_before}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_now}"}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        print("\n" + "="*70)
        print("[LIVE COMMENTATOR - CONTEXT-AWARE UPDATE]")
        print("="*70)
        
        response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        commentary = result['choices'][0]['message']['content'].strip()
        print(commentary)
        print("="*70 + "\n")
        
        return commentary
        
    except Exception as e:
        print(f"[LIVE COMMENTATOR ERROR] {e}\n")
        return f"Motion detected at {datetime.now().strftime('%H:%M:%S')} - Analysis unavailable"

# =========================================================================
# GEMINI: Post-Event Video Analysis
# ============================================================================

def call_gemini_final_analysis(video_path, event_id):
    """
    Call Gemini 2.0 Flash via REST API for comprehensive video analysis.
    Uses base64 encoding for smaller videos or file upload for larger ones.
    """
    try:
        print(f"\n[GEMINI] Preparing video analysis for event {event_id}...")
        
        # Check file size - DEFINE EARLY
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[GEMINI] Video size: {file_size_mb:.2f} MB")
        
        # Create prompt
        prompt = """Analyze this security camera video footage comprehensively.

Provide:
1. OVERVIEW: What happened in this event?
2. TIMELINE: Chronological description of events
3. PEOPLE: Who was involved? Count? Actions?
4. OBJECTS: Significant objects or items?
5. THREAT ASSESSMENT: Overall security evaluation (Low/Medium/High) with reasoning
6. KEY OBSERVATIONS: Critical details for security personnel
7. RECOMMENDATIONS: Required actions or follow-up?

Be thorough and professional."""
        
        summary = None
        
        # Method 1: Try file upload for videos > 5MB
        if file_size_mb > 5:
            print("[GEMINI] Using file upload method...")
            file_uri = upload_video_to_gemini_api(video_path)
            
            if file_uri:
                # Generate content with file URI
                generation_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GOOGLE_API_KEY}"
                
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "file_data": {
                                        "mime_type": "video/mp4",
                                        "file_uri": file_uri
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2048
                    }
                }
                
                print("[GEMINI] Requesting analysis from Gemini 2.0 Flash...")
                response = requests.post(generation_url, json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    summary = result['candidates'][0]['content']['parts'][0]['text']
        
        # Method 2: Use base64 inline for smaller videos (< 20MB limit)
        if not summary and file_size_mb <= 20:
            print("[GEMINI] Using base64 inline method...")
            
            video_base64 = encode_video_to_base64(video_path)
            
            generation_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GOOGLE_API_KEY}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "video/mp4",
                                    "data": video_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048
                }
            }
            
            print("[GEMINI] Sending video for analysis (this may take 30-60 seconds)...")
            response = requests.post(generation_url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                summary = result['candidates'][0]['content']['parts'][0]['text']
        
        # Display the complete summary
        if summary:
            print("\n" + "="*70)
            print("[GEMINI - FINAL VIDEO ANALYSIS]")
            print("="*70)
            print(summary)
            print("="*70)
            print("\n[GEMINI] Analysis complete - Full summary saved to database")
            return summary
        
        # Error cases
        if file_size_mb > 20:
            print(f"[GEMINI ERROR] Video too large ({file_size_mb:.2f} MB) - max 20MB")
        else:
            print("[GEMINI ERROR] No response generated from API")
        
        return None
        
    except Exception as e:
        print(f"[GEMINI ERROR] Video analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Q&A: Post-Event Question Answering
# ============================================================================

def call_qa_model(event_summary, question):
    """
    Call Open Router text model for Q&A based on Gemini summary.
    
    Args:
        event_summary: Gemini's comprehensive summary
        question: User's question
    
    Returns:
        str: Answer
    """
    prompt = f"""You are analyzing a security event based on the comprehensive analysis below.

EVENT ANALYSIS:
{event_summary}

USER QUESTION: {question}

Provide a clear, accurate answer based ONLY on the analysis above."""
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.QNA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(config.OPENROUTER_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Q&A ERROR] {e}")
        return "Unable to answer at this time."
