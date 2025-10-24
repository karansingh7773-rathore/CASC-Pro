from cosmos_db import CosmosDBHelper
from api_helper import call_qa_model
import sys

def ask_question(text_query):
    """
    Q&A function using NVIDIA Nemotron based on Gemini summaries from database.
    """
    db = CosmosDBHelper()
    
    print(f"\n[Q&A] Searching for: '{text_query}'")
    sys.stdout.flush()
    
    # First try to find master session
    try:
        # Query for master session events
        query = "SELECT * FROM c WHERE STARTSWITH(c.id, 'master_session_') ORDER BY c.StartTime DESC"
        items = list(db.container.query_items(query=query, enable_cross_partition_query=True))
        
        if items:
            event = items[0]  # Most recent master session
            summary = event.get("GeminiSummary")
            
            if summary:
                print(f"[Q&A] Found comprehensive session summary ({len(summary)} characters)")
                sys.stdout.flush()
                return call_qa_model(summary, text_query)
    except Exception as e:
        print(f"[Q&A] Error querying master sessions: {e}")
        sys.stdout.flush()
    
    # Fallback to text search
    events = db.query_events(text_query)
    
    if not events:
        events = db.get_recent_events(5)
        if not events:
            return "No events found in database."
    
    event = events[0]
    summary = event.get("GeminiSummary") or event.get("VideoSummary")
    
    if not summary:
        return "Event found but analysis not yet complete."
    
    return call_qa_model(summary, text_query)

if __name__ == "__main__":
    print("="*70)
    print("CASC Pro - Interactive Q&A Mode (Powered by NVIDIA Nemotron)")
    print("="*70)
    print("Ask questions about security events stored in the database.")
    print("Type 'exit' or 'quit' to stop.\n")
    sys.stdout.flush()
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit', '']:
                print("\nExiting Q&A mode.")
                break
            
            answer = ask_question(question)
            print(f"\nFinal Answer: {answer}\n")
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            print("\n\nExiting Q&A mode.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.stdout.flush()
