import requests
import base64
import cv2
import json
from datetime import datetime
import config
import os
import time
import sys
import threading  # ADD THIS - was missing at top level
from openai import OpenAI

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
        
        print(f"[AI] Uploading video file: {video_path}")
        
        # Upload the file
        video_file = genai.upload_file(path=video_path)

        print(f"[AI] Upload complete. File URI: {video_file.uri}")
        
        # Wait for file to be processed
        while video_file.state.name == "PROCESSING":
            print("[AI] Processing video...")
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
        print(f"[AI] Uploading video file: {video_path}")
        sys.stdout.flush()
        
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[AI] Video size: {file_size_mb:.2f} MB")
        sys.stdout.flush()
        
        if file_size_mb > 100:
            print("[AI ERROR] Video file too large (max 100MB)")
            return None
        
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={config.GOOGLE_API_KEY}"
        
        with open(video_path, 'rb') as video_file:
            files = {
                'file': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            
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
            print(f"[AI ERROR] Upload failed: {response.text}")
            return None
        
        result = response.json()
        file_info = result.get('file', {})
        file_uri = file_info.get('uri')
        file_name = file_info.get('name')
        
        if not file_uri:
            print(f"[AI ERROR] No file URI in response")
            return None
        
        print(f"[AI] Video uploaded successfully: {file_name}")
        sys.stdout.flush()
        
        print("[AI] Waiting for video processing...")
        sys.stdout.flush()
        max_wait = 120
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={config.GOOGLE_API_KEY}"
            status_response = requests.get(status_url)
            
            if status_response.status_code == 200:
                state = status_response.json().get('state')
                
                if state == 'ACTIVE':
                    print("[AI] Video processing complete")
                    sys.stdout.flush()
                    return file_uri
                elif state == 'FAILED':
                    print("[AI ERROR] Video processing failed")
                    return None
            
            time.sleep(3)
        
        print("[AI ERROR] Video processing timeout")
        return None
        
    except Exception as e:
        print(f"[AI ERROR] Failed to upload video: {e}")
        sys.stdout.flush()
        return None

def call_gemini_final_analysis(video_path, event_id):
    """
    Call Gemini 2.0 Flash via REST API for comprehensive video analysis.
    Simulates streaming with progress indicators and chunked display.
    """
    try:
        print(f"\n[AI] Preparing video analysis for event {event_id}...")
        sys.stdout.flush()
        
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[AI] Video size: {file_size_mb:.2f} MB")
        sys.stdout.flush()
        
        prompt = """Analyze this security camera video footage comprehensively.

Provide a DETAILED report with:
1. OVERVIEW: What happened in this event? (3-4 sentences minimum)
2. TIMELINE: Chronological description of events with timestamps
3. PEOPLE: Who was involved? Count? Detailed actions and behaviors
4. OBJECTS: Significant objects or items visible in the footage
5. THREAT ASSESSMENT: Overall security evaluation (Low/Medium/High) with detailed reasoning
6. KEY OBSERVATIONS: 5-7 critical details for security personnel
7. RECOMMENDATIONS: Specific required actions or follow-up procedures

Be thorough, detailed, and professional. Provide AT LEAST 500 words."""
        
        summary = None
        
        if file_size_mb > 5:
            print("[AI] Using file upload method...")
            sys.stdout.flush()
            file_uri = upload_video_to_gemini_api(video_path)
            
            if file_uri:
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
                        "maxOutputTokens": 4096
                    }
                }
                
                print("[AI] Requesting comprehensive analysis...")
                print("[AI] AI is analyzing video frames... (this may take 30-90 seconds)")
                sys.stdout.flush()
                
                # Show progress animation while waiting
                stop_animation = threading.Event()
                
                def progress_animation():
                    """Display animated progress indicator."""
                    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    idx = 0
                    while not stop_animation.is_set():
                        print(f"\r[AI] Processing {frames[idx % len(frames)]}", end="", flush=True)
                        time.sleep(0.1)
                        idx += 1
                    print("\r[AI] Processing ✓ Complete!     ", flush=True)
                
                animation_thread = threading.Thread(target=progress_animation, daemon=True)
                animation_thread.start()
                
                response = requests.post(generation_url, json=payload, timeout=240)
                
                # Stop animation
                stop_animation.set()
                animation_thread.join(timeout=1)
                
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    summary = result['candidates'][0]['content']['parts'][0]['text']
        
        if not summary and file_size_mb <= 20:
            print("[AI] Using base64 inline method...")
            sys.stdout.flush()
            
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
                    "maxOutputTokens": 4096
                }
            }
            
            print("[AI] Sending video for comprehensive analysis...")
            print("[AI] AI is analyzing video frames... (this may take 30-90 seconds)")
            sys.stdout.flush()
            
            # Show progress animation while waiting
            stop_animation = threading.Event()
            
            def progress_animation():
                """Display animated progress indicator."""
                frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                idx = 0
                while not stop_animation.is_set():
                    print(f"\r[AI] Processing {frames[idx % len(frames)]}", end="", flush=True)
                    time.sleep(0.1)
                    idx += 1
                print("\r[AI] Processing ✓ Complete!     ", flush=True)
            
            animation_thread = threading.Thread(target=progress_animation, daemon=True)
            animation_thread.start()
            
            response = requests.post(generation_url, json=payload, timeout=240)
            
            # Stop animation
            stop_animation.set()
            animation_thread.join(timeout=1)
            
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                summary = result['candidates'][0]['content']['parts'][0]['text']
        
        # Display with streaming-like chunked output
        if summary:
            print("\n" + "="*50)
            print("[AI - FINAL VIDEO ANALYSIS]")
            print("="*50)
            sys.stdout.flush()
            
            # Simulate streaming by displaying in word chunks
            words = summary.split()
            line_buffer = []
            max_line_length = 80
            
            for word in words:
                line_buffer.append(word)
                current_line = " ".join(line_buffer)
                
                if len(current_line) > max_line_length or word.endswith(('.', '!', '?', ':')):
                    print(" ".join(line_buffer), flush=True)
                    time.sleep(0.02)
                    line_buffer = []
            
            # Print remaining words
            if line_buffer:
                print(" ".join(line_buffer), flush=True)
            
            print("\n" + "="*50)
            print(f"\n[AI] Analysis complete - {len(summary)} characters (~{len(words)} words)")
            print("[AI] Full summary saved to database")
            sys.stdout.flush()
            
            return summary
        
        if file_size_mb > 20:
            print(f"[AI ERROR] Video too large ({file_size_mb:.2f} MB) - max 20MB")
        else:
            print("[AI ERROR] No response generated from API")
        
        sys.stdout.flush()
        return None
        
    except Exception as e:
        print(f"[AI ERROR] Video analysis failed: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Q&A: Using Free Open Router Model (reads Gemini 2.5's saved summary)
# ============================================================================

def call_qa_model(event_summary, question):
    """
    Call NVIDIA Nemotron model for Q&A based on Gemini summary.
    Uses OpenAI-compatible API from NVIDIA.
    
    Args:
        event_summary: Gemini's comprehensive summary
        question: User's question
    
    Returns:
        str: Answer from NVIDIA Nemotron
    """
    try:
        # Initialize NVIDIA client
        client = OpenAI(
            base_url=config.NVIDIA_API_URL,
            api_key=config.NVIDIA_API_KEY
        )
        
        # Construct the system prompt with event context
        system_prompt = f"""You are a professional security analyst AI assistant. 
You have access to a comprehensive security event analysis and must answer questions based ONLY on this information.

SECURITY EVENT ANALYSIS:
{event_summary}

Instructions:
- Answer questions accurately based only on the information provided
- If information is not in the analysis, clearly state that
- Be concise but thorough
- Use professional security terminology
- Provide specific details from the analysis when relevant"""
        
        print(f"\n[NVIDIA Q&A] Processing question with Nemotron...")
        print(f"[NVIDIA Q&A] Question: {question}")
        sys.stdout.flush()
        
        # Create chat completion with streaming
        completion = client.chat.completions.create(
            model=config.QNA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,  # Reasonable limit for Q&A
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        
        # Collect streaming response
        print("[NVIDIA Q&A] Response: ", end="", flush=True)
        full_response = ""
        
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n")  # New line after response
        sys.stdout.flush()
        
        return full_response.strip()
        
    except Exception as e:
        print(f"\n[NVIDIA Q&A ERROR] Failed: {e}")
        sys.stdout.flush()
        
        # Fallback to Open Router if NVIDIA fails
        print("[Q&A] Attempting fallback to Open Router...")
        return call_qa_model_fallback(event_summary, question)

def call_qa_model_fallback(event_summary, question):
    """
    Fallback Q&A function using Open Router if NVIDIA fails.
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
        "model": "mistralai/mistral-7b-instruct:free",
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
        print(f"[Q&A FALLBACK ERROR] {e}")
        return "Unable to answer at this time. Please check your API configuration."

# =========================================================================
# LIVE COMMENTATOR: Context-Aware Live Updates (NVIDIA Vision Model)
# ============================================================================

def call_live_commentator(frame_before, frame_now, previous_commentary=None):
    """
    Call NVIDIA Llama 3.2 Vision model for context-aware live commentary with streaming.
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
    
    # Prepare message content with images
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_before}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_now}"}
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Accept": "text/event-stream"  # Enable streaming
    }
    
    payload = {
        "model": config.LIVE_COMMENTATOR_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": True
    }
    
    try:
        print("\n" + "="*70)
        print("[NVIDIA LIVE COMMENTATOR - STREAMING UPDATE]")
        print("="*70)
        sys.stdout.flush()
        
        response = requests.post(config.NVIDIA_VISION_URL, headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        full_commentary = ""
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8")
                
                # Skip empty lines and "data: " prefix
                if line_text.startswith("data: "):
                    line_text = line_text[6:]  # Remove "data: " prefix
                
                # Skip [DONE] signal
                if line_text.strip() == "[DONE]":
                    break
                
                try:
                    # Parse JSON chunk
                    chunk_data = json.loads(line_text)
                    
                    # Extract content from chunk
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        delta = chunk_data["choices"][0].get("delta", {})
                        content_chunk = delta.get("content", "")
                        
                        if content_chunk:
                            print(content_chunk, end="", flush=True)
                            full_commentary += content_chunk
                
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON
        
        print("\n" + "="*70 + "\n")
        sys.stdout.flush()
        
        return full_commentary.strip() if full_commentary else "Analysis in progress..."
        
    except Exception as e:
        print(f"[NVIDIA COMMENTATOR ERROR] {e}\n")
        sys.stdout.flush()
        return f"Motion detected at {datetime.now().strftime('%H:%M:%S')} - Analysis unavailable"

def call_live_commentator(frames, event_id, frame_count, event_duration, previous_commentary=None):
    """
    Call NVIDIA Llama 3.2 Vision model for context-aware live commentary using sliding window.
    
    Args:
        frames: List of frames (up to 10) showing progression
        event_id: Event identifier
        frame_count: Number of frames in the sequence
        event_duration: How long the event has been occurring
        previous_commentary: Previous commentary for context
    """
    # Encode all frames to base64
    encoded_frames = [encode_frame_to_base64(frame) for frame in frames]
    
    # Build context-aware prompt for sliding window
    if previous_commentary:
        prompt = f"""You are a live security camera AI analyzing an ongoing event (ID: {event_id}).

EVENT CONTEXT:
- Event Duration: {event_duration:.1f} seconds
- Frames in sequence: {frame_count} (oldest to newest)

PREVIOUS ANALYSIS:
{previous_commentary}

TASK: Analyze this sequence of {frame_count} frames showing the event progression.

Frame 1 (oldest) → Frame {frame_count} (most recent)

Provide:
1. PROGRESSION: How did the situation evolve across these frames?
2. PEOPLE TRACKING: Track individuals - entries, exits, movements, interactions
3. BEHAVIOR CHANGE: Any escalation or de-escalation in behavior?
4. CURRENT STATUS: What is happening RIGHT NOW (latest frames)?
5. THREAT TRAJECTORY: Is threat increasing, stable, or decreasing?
6. CURRENT THREAT LEVEL (1-10): Based on latest frames

Be concise but comprehensive. Focus on CHANGES and PROGRESSION."""
    else:
        prompt = f"""You are a live security camera AI analyzing event {event_id}.

Analyze this sequence of {frame_count} security camera frames (oldest to newest).

Frame 1 (start) → Frame {frame_count} (current)

Provide:
1. INITIAL OBSERVATION: What happened at the start?
2. PROGRESSION: How did things develop across frames?
3. PEOPLE: Count and describe individuals across the sequence
4. ACTIONS: Key actions observed in the progression
5. CURRENT STATUS: What is happening NOW (latest frames)?
6. THREAT LEVEL (1-10): Current threat assessment

Be detailed and track changes across the frame sequence."""
    
    # Prepare message content with all frames
    content = [{"type": "text", "text": prompt}]
    
    # Add all frames to content
    for i, encoded_frame in enumerate(encoded_frames, 1):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_frame}",
                "detail": "low" if frame_count > 5 else "high"  # Use low detail for many frames
            }
        })
    
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Accept": "text/event-stream"
    }
    
    payload = {
        "model": config.LIVE_COMMENTATOR_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,  # Increased for detailed analysis
        "temperature": 0.6,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": True
    }
    
    try:
        print("\n" + "="*50)
        print(f"[NVIDIA SLIDING WINDOW - {frame_count} FRAMES]")
        print("="*50)
        sys.stdout.flush()
        
        response = requests.post(config.NVIDIA_VISION_URL, headers=headers, json=payload, stream=True, timeout=45)
        response.raise_for_status()
        
        full_commentary = ""
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8")
                
                if line_text.startswith("data: "):
                    line_text = line_text[6:]
                
                if line_text.strip() == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(line_text)
                    
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        delta = chunk_data["choices"][0].get("delta", {})
                        content_chunk = delta.get("content", "")
                        
                        if content_chunk:
                            print(content_chunk, end="", flush=True)
                            full_commentary += content_chunk
                
                except json.JSONDecodeError:
                    continue
        
        print("\n" + "="*50 + "\n")
        sys.stdout.flush()
        
        return full_commentary.strip() if full_commentary else "Analysis in progress..."
        
    except Exception as e:
        print(f"[NVIDIA COMMENTATOR ERROR] {e}\n")
        sys.stdout.flush()
        return f"Motion detected at {datetime.now().strftime('%H:%M:%S')} - Analysis unavailable"

def call_gemini_final_analysis(video_path, event_id):
    """
    Call Gemini 2.0 Flash via REST API for comprehensive video analysis.
    Simulates streaming with progress indicators and chunked display.
    """
    try:
        print(f"\n[AI] Preparing video analysis for event {event_id}...")
        sys.stdout.flush()
        
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[AI] Video size: {file_size_mb:.2f} MB")
        sys.stdout.flush()
        
        prompt = """Analyze this security camera video footage comprehensively.

Provide a DETAILED report with:
1. OVERVIEW: What happened in this event? (3-4 sentences minimum)
2. TIMELINE: Chronological description of events with timestamps
3. PEOPLE: Who was involved? Count? Detailed actions and behaviors
4. OBJECTS: Significant objects or items visible in the footage
5. THREAT ASSESSMENT: Overall security evaluation (Low/Medium/High) with detailed reasoning
6. KEY OBSERVATIONS: 5-7 critical details for security personnel
7. RECOMMENDATIONS: Specific required actions or follow-up procedures

Be thorough, detailed, and professional. Provide AT LEAST 500 words."""
        
        summary = None
        
        if file_size_mb > 5:
            print("[AI] Using file upload method...")
            sys.stdout.flush()
            file_uri = upload_video_to_gemini_api(video_path)
            
            if file_uri:
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
                        "maxOutputTokens": 4096
                    }
                }
                
                print("[AI] Requesting comprehensive analysis...")
                print("[AI] AI is analyzing video frames... (this may take 30-90 seconds)")
                sys.stdout.flush()
                
                # Show progress animation while waiting
                stop_animation = threading.Event()
                
                def progress_animation():
                    """Display animated progress indicator."""
                    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    idx = 0
                    while not stop_animation.is_set():
                        print(f"\r[AI] Processing {frames[idx % len(frames)]}", end="", flush=True)
                        time.sleep(0.1)
                        idx += 1
                    print("\r[AI] Processing ✓ Complete!     ", flush=True)
                
                animation_thread = threading.Thread(target=progress_animation, daemon=True)
                animation_thread.start()
                
                response = requests.post(generation_url, json=payload, timeout=240)
                
                # Stop animation
                stop_animation.set()
                animation_thread.join(timeout=1)
                
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    summary = result['candidates'][0]['content']['parts'][0]['text']
        
        if not summary and file_size_mb <= 20:
            print("[AI] Using base64 inline method...")
            sys.stdout.flush()
            
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
                    "maxOutputTokens": 4096
                }
            }
            
            print("[AI] Sending video for comprehensive analysis...")
            print("[AI] AI is analyzing video frames... (this may take 30-90 seconds)")
            sys.stdout.flush()
            
            # Show progress animation while waiting
            stop_animation = threading.Event()
            
            def progress_animation():
                """Display animated progress indicator."""
                frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                idx = 0
                while not stop_animation.is_set():
                    print(f"\r[AI] Processing {frames[idx % len(frames)]}", end="", flush=True)
                    time.sleep(0.1)
                    idx += 1
                print("\r[AI] Processing ✓ Complete!     ", flush=True)
            
            animation_thread = threading.Thread(target=progress_animation, daemon=True)
            animation_thread.start()
            
            response = requests.post(generation_url, json=payload, timeout=240)
            
            # Stop animation
            stop_animation.set()
            animation_thread.join(timeout=1)
            
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                summary = result['candidates'][0]['content']['parts'][0]['text']
        
        # Display with streaming-like chunked output
        if summary:
            print("\n" + "="*50)
            print("[AI - FINAL VIDEO ANALYSIS]")
            print("="*50)
            sys.stdout.flush()
            
            # Simulate streaming by displaying in word chunks
            words = summary.split()
            line_buffer = []
            max_line_length = 80
            
            for word in words:
                line_buffer.append(word)
                current_line = " ".join(line_buffer)
                
                if len(current_line) > max_line_length or word.endswith(('.', '!', '?', ':')):
                    print(" ".join(line_buffer), flush=True)
                    time.sleep(0.02)
                    line_buffer = []
            
            # Print remaining words
            if line_buffer:
                print(" ".join(line_buffer), flush=True)
            
            print("\n" + "="*50)
            print(f"\n[AI] Analysis complete - {len(summary)} characters (~{len(words)} words)")
            print("[AI] Full summary saved to database")
            sys.stdout.flush()
            
            return summary
        
        if file_size_mb > 20:
            print(f"[AI ERROR] Video too large ({file_size_mb:.2f} MB) - max 20MB")
        else:
            print("[AI ERROR] No response generated from API")
        
        sys.stdout.flush()
        return None
        
    except Exception as e:
        print(f"[AI ERROR] Video analysis failed: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return None
