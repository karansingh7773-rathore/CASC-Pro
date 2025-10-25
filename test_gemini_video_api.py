"""
Test Gemini Video API
This script tests if Gemini can analyze video files correctly.
"""
import sys
import os
import config
from api_helper import call_gemini_final_analysis

def test_gemini_video_analysis():
    """Test Gemini video analysis with a sample video."""
    print("="*50)
    print("Gemini Video API Test")
    print("="*50 + "\n")
    
    # Check if test video exists
    test_video_path = os.path.join(config.TEMP_VIDEO_DIR, "videoplayback.mp4")
    
    if not os.path.exists(test_video_path):
        print("[ERROR] Test video not found!")
        print(f"[ERROR] Expected location: {test_video_path}")
        print("\n[INFO] Please:")
        print(f"  1. Create folder: {config.TEMP_VIDEO_DIR}")
        print("  2. Copy a test video as: test_video.mp4")
        print("  3. Run this script again")
        return False
    
    # Check video size
    file_size_mb = os.path.getsize(test_video_path) / (1024 * 1024)
    print(f"[INFO] Test video found: {test_video_path}")
    print(f"[INFO] Video size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 20:
        print(f"[WARNING] Video is large ({file_size_mb:.2f} MB)")
        print("[WARNING] This may take longer to upload and process")
    
    print("\n[TEST] Starting Gemini video analysis...")
    print("[TEST] This may take 30-120 seconds depending on video size")
    print("="*50 + "\n")
    
    # Call Gemini
    summary = call_gemini_final_analysis(test_video_path, "test_event_001")
    
    # Check results
    print("\n" + "="*50)
    print("[TEST] Analysis Results")
    print("="*50)
    
    if summary:
        print(f"[SUCCESS] Gemini returned analysis!")
        print(f"[SUCCESS] Summary length: {len(summary)} characters")
        print(f"[SUCCESS] Word count: ~{len(summary.split())} words")
        
        # Display first 500 characters
        print("\n[PREVIEW] First 500 characters:")
        print("-"*50)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        print("-"*50)
        
        # Save full summary to file
        output_file = os.path.join(config.TEMP_VIDEO_DIR, "test_summary.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n[INFO] Full summary saved to: {output_file}")
        
        return True
    else:
        print("[FAILURE] Gemini did not return a summary")
        print("[FAILURE] Check the error messages above")
        return False

def check_configuration():
    """Check if Gemini API is configured correctly."""
    print("\n" + "="*50)
    print("Configuration Check")
    print("="*50)
    
    checks_passed = True
    
    # Check Google API Key
    if config.GOOGLE_API_KEY and config.GOOGLE_API_KEY != "your_google_ai_studio_api_key_here":
        print("[OK] Google API Key is configured")
    else:
        print("[FAIL] Google API Key is missing or default")
        print("[INFO] Get your key from: https://aistudio.google.com/app/apikey")
        print("[INFO] Set it in config.py: GOOGLE_API_KEY = 'your_key_here'")
        checks_passed = False
    
    # Check Gemini Model
    print(f"[INFO] Using model: {config.GEMINI_MODEL}")
    
    # Check temp directory
    if os.path.exists(config.TEMP_VIDEO_DIR):
        print(f"[OK] Temp directory exists: {config.TEMP_VIDEO_DIR}")
    else:
        print(f"[INFO] Creating temp directory: {config.TEMP_VIDEO_DIR}")
        os.makedirs(config.TEMP_VIDEO_DIR, exist_ok=True)
    
    print("="*50)
    return checks_passed

if __name__ == "__main__":
    print("\n Gemini Video API Test Suite\n")
    
    # Step 1: Check configuration
    if not check_configuration():
        print("\n[ERROR] Configuration check failed!")
        print("[ERROR] Please fix the issues above before testing")
        sys.exit(1)
    
    # Step 2: Run test
    try:
        success = test_gemini_video_analysis()
        
        if success:
            print("\n" + "="*50)
            print("✅ TEST PASSED - Gemini Video API is working!")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ TEST FAILED - See errors above")
            print("="*50)
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Test cancelled by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
