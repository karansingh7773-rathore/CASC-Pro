import cv2
import time
import threading
import queue
import uuid
from datetime import datetime
import os
import config
from cosmos_db import CosmosDBHelper
from yolo_detector import YOLODetector
from api_helper import call_live_commentator, call_gemini_final_analysis
import sys

# Initialize
db = CosmosDBHelper()
job_queue = queue.Queue()

# State variables
motion_detected = False
current_event_id = None
event_start_time = None
last_live_call_time = None
frame_event_start = None
frame_previous_live = None
motion_segment_frames = []

# NEW: Sliding window for live commentary
frame_window = []  # Stores last 10 frames for sliding window analysis
last_frame_sample_time = 0  # When we last added a frame to window

# NEW: Track all events for final Gemini analysis
all_events = []  # List of all event IDs processed

def save_frames_to_video(frames, output_path, fps=15):
    """Save collected frames to video file."""
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True

def combine_all_event_videos(event_ids, output_path, fps=15):
    """Combine all event videos into one master video for Gemini."""
    all_frames = []
    
    for event_id in event_ids:
        video_path = os.path.join(config.TEMP_VIDEO_DIR, f"{event_id}.mp4")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
    
    if all_frames:
        save_frames_to_video(all_frames, output_path, fps)
        return True
    return False

def worker_thread():
    """Thread 2: Worker/Detective - Handles all slow API operations."""
    print("[Worker Thread] Started\n")
    sys.stdout.flush()
    
    while True:
        job = job_queue.get()
        
        # Check for sentinel (shutdown signal)
        if job is None:
            print("[Worker Thread] Received shutdown signal")
            sys.stdout.flush()
            time.sleep(1)  # Give time for any pending prints
            job_queue.task_done()
            break
        
        try:
            if job["job_type"] == "LIVE_COMMENTARY":
                event_id = job["event_id"]
                frames = job["frames"]
                frame_count = job["frame_count"]
                event_duration = job["event_duration"]
                
                # Get previous commentary from DB for context
                event_doc = db.get_event(event_id)
                previous_commentary = event_doc.get("LastLiveCommentary") if event_doc else None
                
                # Call live commentator with sliding window of frames
                current_analysis = call_live_commentator(
                    frames,
                    event_id,
                    frame_count,
                    event_duration,
                    previous_commentary
                )
                
                # Update DB with new commentary
                db.update_live_commentary(event_id, current_analysis)
            
            elif job["job_type"] == "SAVE_EVENT_VIDEO":
                # Just save the video, don't analyze yet
                event_id = job["event_id"]
                video_path = job["video_file_path"]
                print(f"[Worker] Event video saved: {event_id}")
                sys.stdout.flush()
            
            elif job["job_type"] == "GEMINI_FINAL_ANALYSIS":
                # This happens ONCE at the very end
                master_video_path = job["master_video_path"]
                
                print("\n" + "="*70)
                print("[AI] Starting comprehensive analysis of ALL events")
                print("[AI] This may take 60-120 seconds for detailed analysis")
                print("="*70)
                sys.stdout.flush()
                
                # Call Gemini for comprehensive analysis
                summary = call_gemini_final_analysis(master_video_path, "all_events")
                
                if summary:
                    print(f"\n[Worker] AI analysis complete!")
                    print(f"[Worker] Summary length: {len(summary)} characters")
                    print(f"[Worker] Word count: ~{len(summary.split())} words")
                    sys.stdout.flush()
                    
                    # Save to a master document
                    master_event_id = "master_session_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    db.create_event(master_event_id, datetime.utcnow().isoformat() + "Z", master_video_path)
                    db.update_gemini_summary(master_event_id, summary, datetime.utcnow().isoformat() + "Z")
                    print(f"[Worker] Summary saved to database as event: {master_event_id}")
                    sys.stdout.flush()
                else:
                    print(f"[Worker] AI analysis failed - no summary generated")
                    sys.stdout.flush()
                
                # Clean up master video
                try:
                    os.remove(master_video_path)
                    print(f"[Worker] Cleaned up master video file")
                    sys.stdout.flush()
                except Exception as cleanup_error:
                    print(f"[Worker] Could not clean up video: {cleanup_error}")
                    sys.stdout.flush()
        
        except Exception as e:
            print(f"[Worker ERROR] Job processing failed: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
        
        finally:
            job_queue.task_done()
    
    print("[Worker Thread] Stopped")
    sys.stdout.flush()

def main():
    """Thread 1: Main/Sentry - Video processing and YOLO detection."""
    global motion_detected, current_event_id, event_start_time
    global last_live_call_time, frame_event_start, frame_previous_live
    global motion_segment_frames, all_events
    global frame_window, last_frame_sample_time
    
    print("="*50)
    print("CASC PRO - Contextual Aware Security Cam")
    print("YOLOv8n Detection | Live Commentary | AI Analysis")
    print("="*50 + "\n")
    
    # Initialize YOLO
    detector = YOLODetector()
    
    # Start worker thread
    worker = threading.Thread(target=worker_thread, daemon=True)
    worker.start()
    
    # Open video
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    frame_delay = max(1, int(1000 / fps))
    
    print(f"[System] Video FPS: {fps}")
    print(f"[System] Press 'q' to quit\n")
    print("="*50 + "\n")
    
    frame_count = 0
    event_count = 0  # Track total number of events
    
    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n[System] End of video reached")
            break
        
        frame_count += 1
        
        # Run YOLO detection
        annotated_frame, has_person, person_count, detections = detector.detect_people(frame)
        
        current_time = time.time()
        
        # ============================================================
        # STATE MACHINE
        # ============================================================
        
        # State 2: Motion Started
        if has_person and not motion_detected:
            event_count += 1
            print(f"\n{'='*50}")
            print(f"[ALERT] PERSON DETECTED - Event #{event_count} Started (Count: {person_count})")
            print(f"{'='*50}")
            
            motion_detected = True
            event_start_time = current_time
            last_live_call_time = current_time
            frame_event_start = frame.copy()
            frame_previous_live = frame.copy()
            current_event_id = f"evt_{uuid.uuid4()}"
            motion_segment_frames = [frame.copy()]
            
            # Initialize sliding window with first frame
            frame_window = [frame.copy()]
            last_frame_sample_time = current_time
            
            # Create event in DB
            start_time_iso = datetime.utcnow().isoformat() + "Z"
            db.create_event(current_event_id, start_time_iso)
            print(f"[DB] Event created: {current_event_id}\n")
        
        # State 3: Motion In Progress
        elif has_person and motion_detected:
            # Collect frames for video recording
            motion_segment_frames.append(frame.copy())
            
            # NEW: Sample frames for sliding window at configured interval
            if (current_time - last_frame_sample_time) >= config.FRAME_SAMPLING_INTERVAL:
                frame_window.append(frame.copy())
                last_frame_sample_time = current_time
                
                # Keep only last MAX_FRAME_WINDOW frames (sliding window)
                if len(frame_window) > config.MAX_FRAME_WINDOW:
                    frame_window.pop(0)  # Remove oldest frame
            
            # Check if it's time for live commentary with sliding window
            if (current_time - last_live_call_time) >= config.THROTTLE_SECONDS:
                elapsed = current_time - event_start_time
                frame_count_in_window = len(frame_window)
                
                print(f"\n[Live Commentary] Requesting sliding window analysis")
                print(f"[Live Commentary] Elapsed: {elapsed:.1f}s | Frames in window: {frame_count_in_window}")
                
                job = {
                    "job_type": "LIVE_COMMENTARY",
                    "event_id": current_event_id,
                    "frames": frame_window.copy(),  # Send all frames in window
                    "frame_count": frame_count_in_window,
                    "event_duration": elapsed
                }
                job_queue.put(job)
                
                last_live_call_time = current_time
                frame_previous_live = frame.copy()
            
            # Display status
            elapsed = int(current_time - event_start_time)
            cv2.putText(annotated_frame, f"RECORDING ({elapsed}s) - People: {person_count} - Window: {len(frame_window)} frames", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # State 4: Motion Stopped
        elif not has_person and motion_detected:
            print(f"\n{'='*50}")
            print("[ALERT] PERSON LEFT - Event Ended")
            print(f"{'='*50}")
            
            motion_detected = False
            event_end_time = current_time
            event_duration = event_end_time - event_start_time
            
            print(f"[System] Event duration: {event_duration:.2f}s")
            print(f"[System] Frames collected: {len(motion_segment_frames)}")
            print(f"[System] Frames in final window: {len(frame_window)}")
            
            # Save video segment
            video_path = os.path.join(config.TEMP_VIDEO_DIR, f"{current_event_id}.mp4")
            save_frames_to_video(motion_segment_frames, video_path, fps)
            print(f"[System] Video saved: {video_path}")
            
            # Add to all events list
            all_events.append(current_event_id)
            
            # Update event status in DB
            db.update_event_status(current_event_id, "Complete", datetime.utcnow().isoformat() + "Z")
            print(f"[System] Event marked as complete (AI analysis will happen at video end)\n")
            
            # Reset state
            current_event_id = None
            motion_segment_frames = []
            frame_event_start = None
            frame_previous_live = None
            frame_window = []  # Clear sliding window
        
        # State 1: Idle
        else:
            cv2.putText(annotated_frame, "Status: MONITORING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('CASC Pro - YOLOv8 Detection', annotated_frame)
        
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            print("\n[System] User quit requested")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("[System] Video processing finished.")
    print(f"[System] Total events captured: {len(all_events)}")
    print(f"[System] Total frames processed: {frame_count}")
    print("="*50 + "\n")
    
    # NOW: Combine all event videos and send to AI ONCE
    if all_events:
        print("[System] Combining all event videos for AI analysis...")
        master_video_path = os.path.join(config.TEMP_VIDEO_DIR, "master_session.mp4")
        
        if combine_all_event_videos(all_events, master_video_path, fps):
            print(f"[System] Master video created: {master_video_path}")
            print("[System] Queuing for AI comprehensive analysis...\n")
            
            # Queue AI analysis
            job = {
                "job_type": "GEMINI_FINAL_ANALYSIS",
                "master_video_path": master_video_path
            }
            job_queue.put(job)
        else:
            print("[System] No valid event videos to analyze")
    else:
        print("[System] No events captured - skipping AI analysis")
    
    print("\n" + "="*50)
    print("[System] Waiting for background analysis to complete...")
    print("[System] Please be patient - AI may take 60-120 seconds")
    print("="*50 + "\n")
    sys.stdout.flush()
    
    # Signal worker to finish and wait
    job_queue.put(None)
    
    # Wait with timeout
    print("[System] Waiting for worker thread to finish...")
    sys.stdout.flush()
    job_queue.join()
    
    # Give extra time for final prints and flushes
    print("[System] Ensuring all output is displayed...")
    sys.stdout.flush()
    time.sleep(3)  # Increased from 2 to 3 seconds
    
    print("\n" + "="*50)
    print("[System] All background tasks complete. Ready for Q&A.")
    print("="*50 + "\n")
    sys.stdout.flush()

    # Q&A Loop
    from qa_function import ask_question
    
    while True:
        try:
            question = input("Ask about an event (or type 'exit'): ").strip()
            if question.lower() in ['exit', 'quit', '']:
                print("\n[System] Exiting CASC Pro")
                break
            
            answer = ask_question(question)
            print(f"\nAnswer: {answer}\n")
        
        except KeyboardInterrupt:
            print("\n\n[System] Exiting CASC Pro")
            break

if __name__ == "__main__":
    main()
