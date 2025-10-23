from cosmos_db import CosmosDBHelper
from api_helper import call_qa_model

def ask_question(text_query):
    """
    Q&A function using Gemini summaries from database.
    """
    db = CosmosDBHelper()
    
    print(f"\n[Q&A] Searching for: '{text_query}'")
    
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
                return call_qa_model(summary, text_query)
    except Exception as e:
        print(f"[Q&A] Error querying master sessions: {e}")
    
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

# Interactive Q&A mode
if __name__ == "__main__":
    print("="*70)
    print("CASC - Interactive Q&A Mode")
    print("="*70)
    print("Model: DeepSeek Chat v3.1 (via Open Router)")
    print("Reading from: Gemini 2.5 Flash summaries in database")
    print("="*70)
    print("Ask questions about security events stored in the database.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', '']:
                print("\nExiting Q&A mode.")
                break
            
            answer = ask_question(question)
            
        except KeyboardInterrupt:
            print("\n\nExiting Q&A mode.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
