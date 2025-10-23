"""
Test file for API Helper (Open Router integration).
Run with: python test_api_helper.py
"""
import cv2
import numpy as np
from api_helper import call_vision_api, call_summary_api, call_qa_api

def create_test_frame(text="BEFORE", color=(0, 255, 0)):
    """Create a simple test frame with text."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                2, color, 3, cv2.LINE_AA)
    return frame

def test_vision_api():
    """Test the vision API with mock frames."""
    print("=" * 60)
    print("Testing Vision API")
    print("=" * 60)
    
    print("\n📝 Creating test frames...")
    frame_before = create_test_frame("BEFORE", (0, 255, 0))
    frame_after = create_test_frame("AFTER", (0, 0, 255))
    
    print("✅ Test frames created")
    
    print("\n📝 Calling Vision API (this may take 3-5 seconds)...")
    result = call_vision_api(frame_before, frame_after)
    
    if result:
        print("✅ Vision API call successful!")
        print(f"\n📊 Result:")
        print(f"   Timestamp: {result['Timestamp']}")
        print(f"   Analysis:")
        for key, value in result['Analysis'].items():
            print(f"      - {key}: {value}")
        return True
    else:
        print("❌ Vision API call failed")
        return False

def test_summary_api():
    """Test the summary API with mock insights."""
    print("\n" + "=" * 60)
    print("Testing Summary API")
    print("=" * 60)
    
    mock_insights = [
        {
            "Timestamp": "2025-01-15T10:00:00Z",
            "Analysis": {
                "Summary": "Two people entered the frame",
                "People Count": 2,
                "Emotions/Posture": "calm, walking",
                "Threat Assessment": 2,
                "Key Objects": ["none"]
            }
        },
        {
            "Timestamp": "2025-01-15T10:00:05Z",
            "Analysis": {
                "Summary": "People are talking to each other",
                "People Count": 2,
                "Emotions/Posture": "calm, conversing",
                "Threat Assessment": 1,
                "Key Objects": ["none"]
            }
        },
        {
            "Timestamp": "2025-01-15T10:00:10Z",
            "Analysis": {
                "Summary": "One person left the frame",
                "People Count": 1,
                "Emotions/Posture": "calm, walking",
                "Threat Assessment": 1,
                "Key Objects": ["bag"]
            }
        }
    ]
    
    print("\n📝 Calling Summary API (this may take 3-5 seconds)...")
    summary = call_summary_api(mock_insights)
    
    if summary:
        print("✅ Summary API call successful!")
        print(f"\n📊 Generated Summary:")
        print(f"   {summary}")
        return True
    else:
        print("❌ Summary API call failed")
        return False

def test_qa_api():
    """Test the Q&A API with mock context."""
    print("\n" + "=" * 60)
    print("Testing Q&A API")
    print("=" * 60)
    
    mock_context = {
        "EventID": "evt_test_12345",
        "StartTime": "2025-01-15T10:00:00Z",
        "EndTime": "2025-01-15T10:01:00Z",
        "EventSummary": "Two people had a conversation and one left with a bag.",
        "Insights": [
            {
                "Timestamp": "2025-01-15T10:00:00Z",
                "Analysis": {
                    "Summary": "Two people entered talking",
                    "People Count": 2,
                    "Emotions/Posture": "calm",
                    "Threat Assessment": 2,
                    "Key Objects": ["none"]
                }
            }
        ]
    }
    
    test_question = "How many people were involved?"
    
    print(f"\n📝 Question: {test_question}")
    print("📝 Calling Q&A API (this may take 3-5 seconds)...")
    
    answer = call_qa_api(mock_context, test_question)
    
    if answer:
        print("✅ Q&A API call successful!")
        print(f"\n📊 Answer:")
        print(f"   {answer}")
        return True
    else:
        print("❌ Q&A API call failed")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting API Helper Tests")
    print("⚠️  Note: These tests require valid API credentials in config.py")
    print("⚠️  Each test may take 3-5 seconds due to API response time\n")
    
    try:
        # Test 1: Vision API
        vision_success = test_vision_api()
        
        # Test 2: Summary API
        summary_success = test_summary_api()
        
        # Test 3: Q&A API
        qa_success = test_qa_api()
        
        # Final Report
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"Vision API: {'✅ PASSED' if vision_success else '❌ FAILED'}")
        print(f"Summary API: {'✅ PASSED' if summary_success else '❌ FAILED'}")
        print(f"Q&A API: {'✅ PASSED' if qa_success else '❌ FAILED'}")
        
        if vision_success and summary_success and qa_success:
            print("\n🎉 All API tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check your API configuration.")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
