"""
Test file for Cosmos DB operations.
Run with: python test_cosmos_db.py
"""
import time
from datetime import datetime
from cosmos_db import CosmosDBHelper

def test_cosmos_db():
    """Test all Cosmos DB operations."""
    print("=" * 60)
    print("Testing Cosmos DB Operations")
    print("=" * 60)
    
    db = CosmosDBHelper()
    
    # Test 1: Create Event
    print("\n📝 Test 1: Creating event...")
    test_event_id = f"test_evt_{int(time.time())}"
    start_time = datetime.utcnow().isoformat() + "Z"
    
    result = db.create_event(test_event_id, start_time, "https://example.com/frame.jpg")
    assert result == True, "Failed to create event"
    print("✅ Event created successfully")
    
    # Test 2: Append Insight
    print("\n📝 Test 2: Appending insight...")
    test_insight = {
        "Timestamp": datetime.utcnow().isoformat() + "Z",
        "Analysis": {
            "Summary": "Test summary: Person walking",
            "People Count": 1,
            "Emotions/Posture": "calm, walking",
            "Threat Assessment": 2,
            "Key Objects": ["none"]
        }
    }
    
    result = db.append_insight(test_event_id, test_insight)
    assert result == True, "Failed to append insight"
    print("✅ Insight appended successfully")
    
    # Test 3: Append Another Insight
    print("\n📝 Test 3: Appending second insight...")
    test_insight_2 = {
        "Timestamp": datetime.utcnow().isoformat() + "Z",
        "Analysis": {
            "Summary": "Test summary: Person stopped",
            "People Count": 1,
            "Emotions/Posture": "calm, standing",
            "Threat Assessment": 1,
            "Key Objects": ["phone"]
        }
    }
    
    result = db.append_insight(test_event_id, test_insight_2)
    assert result == True, "Failed to append second insight"
    print("✅ Second insight appended successfully")
    
    # Test 4: Get Event
    print("\n📝 Test 4: Retrieving event...")
    event = db.get_event(test_event_id)
    assert event is not None, "Failed to retrieve event"
    assert len(event['Insights']) == 2, "Insights count mismatch"
    print(f"✅ Event retrieved: {event['id']}")
    print(f"   - Insights count: {len(event['Insights'])}")
    
    # Test 5: Update Event Summary
    print("\n📝 Test 5: Updating event summary...")
    test_summary = "Test event: A person walked through the frame and stopped to check their phone."
    
    result = db.update_event_summary(test_event_id, test_summary)
    assert result == True, "Failed to update summary"
    print("✅ Summary updated successfully")
    
    # Test 6: Update Event Status
    print("\n📝 Test 6: Updating event status...")
    end_time = datetime.utcnow().isoformat() + "Z"
    
    result = db.update_event_status(test_event_id, "Complete", end_time)
    assert result == True, "Failed to update status"
    print("✅ Status updated successfully")
    
    # Test 7: Query Events
    print("\n📝 Test 7: Querying events...")
    results = db.query_events("person")
    assert len(results) > 0, "No events found in query"
    print(f"✅ Found {len(results)} event(s) matching 'person'")
    
    # Test 8: Verify Final State
    print("\n📝 Test 8: Verifying final event state...")
    final_event = db.get_event(test_event_id)
    print(f"✅ Final Event State:")
    print(f"   - ID: {final_event['id']}")
    print(f"   - Status: {final_event['Status']}")
    print(f"   - Summary: {final_event['EventSummary']}")
    print(f"   - Insights: {len(final_event['Insights'])}")
    print(f"   - Start: {final_event['StartTime']}")
    print(f"   - End: {final_event['EndTime']}")
    
    print("\n" + "=" * 60)
    print("✅ All Cosmos DB tests passed!")
    print("=" * 60)
    
    return test_event_id

if __name__ == "__main__":
    try:
        event_id = test_cosmos_db()
        print(f"\n💡 Test event ID: {event_id}")
        print("You can view this event in Azure Portal under Cosmos DB.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
