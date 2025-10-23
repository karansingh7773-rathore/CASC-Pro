"""
Test video analyzer - Run the main CASC system with a specific video.
This is useful for testing with pre-recorded videos.

Run with: python test_video_analyzer.py
"""
import cv2
import os
import config

def list_available_videos():
    """List all video files in the videos directory."""
    videos_dir = "d:\\CASC2\\videos"
    
    if not os.path.exists(videos_dir):
        print(f"⚠️  Videos directory not found: {videos_dir}")
        return []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    videos = []
    
    for file in os.listdir(videos_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append(os.path.join(videos_dir, file))
    
    return videos

def test_video_properties(video_path):
    """Test if video can be opened and display its properties."""
    print("=" * 60)
    print(f"Testing Video: {os.path.basename(video_path)}")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open video file")
        return False
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n📊 Video Properties:")
    print(f"   - Resolution: {width} x {height}")
    print(f"   - FPS: {fps}")
    print(f"   - Frame Count: {frame_count}")
    print(f"   - Duration: {duration:.2f} seconds")
    
    # Play video preview (first 5 seconds or 150 frames)
    print(f"\n🎬 Playing video preview...")
    print(f"   Press 'q' to stop preview, or wait for auto-stop")
    
    frame_counter = 0
    max_preview_frames = min(150, frame_count)  # Show first 5 seconds max
    
    while frame_counter < max_preview_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame counter to display
        cv2.putText(frame, f"Frame: {frame_counter}/{frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Video Preview - Press Q to exit', frame)
        
        # Important: waitKey with delay to allow frame updates
        # Wait 1ms for webcam speed, or calculate based on FPS for video files
        delay = max(1, int(1000 / fps)) if fps > 0 else 30
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("\n   Preview stopped by user")
            break
        
        frame_counter += 1
    
    cv2.destroyAllWindows()
    cap.release()
    
    if frame_counter > 0:
        print(f"   ✅ Preview completed ({frame_counter} frames shown)")
        return True
    else:
        print(f"   ❌ Could not read frames")
        return False

def select_video_interactive():
    """Interactive video selection."""
    print("\n" + "=" * 60)
    print("📹 Video Selection")
    print("=" * 60)
    
    videos = list_available_videos()
    
    if not videos:
        print("\n⚠️  No videos found in d:\\CASC2\\videos\\")
        print("\n💡 Options:")
        print("   1. Create the 'videos' folder: d:\\CASC2\\videos\\")
        print("   2. Place your video files there (.mp4, .avi, etc.)")
        print("   3. Or use webcam by setting VIDEO_SOURCE = 0 in config.py")
        return None
    
    print(f"\n✅ Found {len(videos)} video(s):\n")
    
    for i, video in enumerate(videos, 1):
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"   {i}. {os.path.basename(video)} ({size_mb:.2f} MB)")
    
    print(f"   0. Use webcam instead")
    
    while True:
        try:
            choice = input(f"\nSelect video (0-{len(videos)}): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                return 0  # Webcam
            elif 1 <= choice_num <= len(videos):
                return videos[choice_num - 1]
            else:
                print(f"⚠️  Please enter a number between 0 and {len(videos)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled")
            return None

def main():
    """Main test function."""
    print("🎥 CASC Video Analyzer Test")
    print("=" * 60)
    
    # Check if videos directory exists
    videos_dir = "d:\\CASC2\\videos"
    if not os.path.exists(videos_dir):
        print(f"\n📁 Creating videos directory: {videos_dir}")
        os.makedirs(videos_dir)
        print("✅ Directory created")
        print("\n💡 Please place your test videos in this folder and run again.")
        return
    
    # Select video
    video_source = select_video_interactive()
    
    if video_source is None:
        return
    
    # Test video properties
    if video_source != 0:
        success = test_video_properties(video_source)
        if not success:
            return
        
        # Ask if user wants to update config
        print("\n" + "=" * 60)
        update = input("\n❓ Update config.py with this video? (y/n): ").strip().lower()
        
        if update == 'y':
            print(f"\n💡 To use this video, update config.py:")
            print(f'   VIDEO_SOURCE = r"{video_source}"')
            
            # Option to auto-update
            auto = input("\n❓ Auto-update config.py now? (y/n): ").strip().lower()
            if auto == 'y':
                try:
                    with open('config.py', 'r') as f:
                        content = f.read()
                    
                    # Replace VIDEO_SOURCE line
                    import re
                    new_content = re.sub(
                        r'VIDEO_SOURCE = .*',
                        f'VIDEO_SOURCE = r"{video_source}"',
                        content
                    )
                    
                    with open('config.py', 'w') as f:
                        f.write(new_content)
                    
                    print("✅ config.py updated!")
                    print("\n💡 Now run: python main.py")
                except Exception as e:
                    print(f"❌ Error updating config: {e}")
    else:
        print("\n💡 To use webcam, update config.py:")
        print("   VIDEO_SOURCE = 0")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled")
