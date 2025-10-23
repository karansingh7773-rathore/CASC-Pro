"""
Test file for Q&A function.
Run with: python test_qa_function.py
"""
from qa_function import ask_question
from cosmos_db import CosmosDBHelper
from datetime import datetime
import time

def setup_test_event():
    """Create a test event in Cosmos DB for Q&A testing."""
    print("=" * 60)
    print("Setting up test event for Q&A")
    print("=" * 60)
    
    db = CosmosDBHelper()
    
    test_event_id = f"qa_test_evt_{int(time.time())}"
    start_time = datetime.utcnow().isoformat() + "Z"
    
    # Create event
    db.create_event(test_event_id, start_time)
    
    # Add insights
    insights = [
        {
            "Timestamp": datetime.utcnow().isoformat() + "Z",
            "Analysis": {
                "Summary": "A man in a red shirt entered the frame carrying a large box",
                "People Count": 1,
                "Emotions/Posture": "calm, carrying heavy object",
                "Threat Assessment": 2,
                "Key Objects": ["box", "package"]
            }
        },
        {
            "Timestamp": datetime.utcnow().isoformat() + "Z",
            "Analysis": {
                "Summary": "The man set the box down and looked around nervously",
                "People Count": 1,
                "Emotions/Posture": "nervous, looking around",
                "Threat Assessment": 5,
                "Key Objects": ["box"]
            }
        },
        {
            "Timestamp": datetime.utcnow().isoformat() + "Z",
            "Analysis": {
                "Summary": "The man opened the box and pulled out a phone",
                "People Count": 1,
                "Emotions/Posture": "focused, crouching",
                "Threat Assessment": 3,
                "Key Objects": ["box", "phone"]
            }
        }
    ]
    
    for insight in insights:
        db.append_insight(test_event_id, insight)
    
    # Add summary
    summary = "A man in a red shirt brought a large box, looked around nervously, then opened it to retrieve a phone."
    db.update_event_summary(test_event_id, summary)
    
    # Complete the event
    end_time = datetime.utcnow().isoformat() + "Z"
    db.update_event_status(test_event_id, "Complete", end_time)
    
    print(f"✅ Test event created: {test_event_id}")
    return test_event_id

def test_qa_questions():
    """Test various Q&A queries."""
    print("\n" + "=" * 60)
    print("Testing Q&A Function")
    print("=" * 60)
    
    # List of test questions
    questions = [
        "What was the person wearing?",
        "What objects were involved in this event?",
        "What was the threat level?",
        "Was the person nervous or calm?",
        "What did the person do with the box?",
        "How many people were in the scene?"
    ]
    
    print("\n📝 Running Q&A tests...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("🔍 Processing (may take 3-5 seconds)...")
        
        answer = ask_question(question)
        
        print(f"💡 Answer: {answer}")
        print("-" * 60 + "\n")
    
    print("✅ Q&A testing complete!")

if __name__ == "__main__":
    try:
        # Setup test event
        event_id = setup_test_event()
        
        # Wait a moment for DB to propagate
        print("\n⏳ Waiting 2 seconds for DB to propagate...")
        time.sleep(2)
        
        # Run Q&A tests
        test_qa_questions()
        
        print("\n" + "=" * 60)
        print("🎉 All Q&A tests completed!")
        print("=" * 60)
        print(f"\n💡 Test event ID: {event_id}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
