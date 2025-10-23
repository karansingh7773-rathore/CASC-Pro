from azure.cosmos import CosmosClient, exceptions, PartitionKey
from datetime import datetime
import config

class CosmosDBHelper:
    def __init__(self):
        self.client = CosmosClient(config.COSMOS_ENDPOINT, config.COSMOS_KEY)
        self.database = self.client.create_database_if_not_exists(id=config.COSMOS_DATABASE_NAME)
        self.container = self.database.create_container_if_not_exists(
            id=config.COSMOS_CONTAINER_NAME,
            partition_key=PartitionKey(path="/id"),
        )
    
    def create_event(self, event_id, start_time, video_path=None):
        """Create a new event document in Cosmos DB with LastLiveCommentary field."""
        event_doc = {
            "id": event_id,
            "StartTime": start_time,
            "EndTime": None,
            "Status": "InProgress",
            "VideoFilePath": video_path,
            "LastLiveCommentary": None,  # NEW: For Maverick context
            "Insights": [],  # NEW: Array of live commentary entries
            "GeminiSummary": None  # Will be filled by Gemini after event ends
        }
        try:
            self.container.create_item(body=event_doc)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to create event {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")  # Show just the error message, not full stack
            return False
    
    def update_event_status(self, event_id, status, end_time=None):
        """Update event status to Complete."""
        try:
            event_doc = self.container.read_item(item=event_id, partition_key=event_id)
            event_doc["Status"] = status
            if end_time:
                event_doc["EndTime"] = end_time
            self.container.replace_item(item=event_id, body=event_doc)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to update event status for {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return False
    
    def update_event_summary(self, event_id, summary, video_path=None):
        """Update the VideoSummary field with Gemini's analysis."""
        try:
            event_doc = self.container.read_item(item=event_id, partition_key=event_id)
            event_doc["VideoSummary"] = summary
            if video_path:
                event_doc["VideoFilePath"] = video_path
            self.container.replace_item(item=event_id, body=event_doc)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to update summary for {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return False
    
    def update_live_commentary(self, event_id, commentary):
        """Update LastLiveCommentary and append to Insights array."""
        try:
            event_doc = self.container.read_item(item=event_id, partition_key=event_id)
            event_doc["LastLiveCommentary"] = commentary
            
            # Append to Insights array with timestamp
            insight_entry = {
                "Timestamp": datetime.utcnow().isoformat() + "Z",
                "Commentary": commentary
            }
            event_doc["Insights"].append(insight_entry)
            
            self.container.replace_item(item=event_id, body=event_doc)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to update commentary for {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return False
    
    def update_gemini_summary(self, event_id, summary, end_time):
        """Update the GeminiSummary field and mark event as Complete."""
        try:
            event_doc = self.container.read_item(item=event_id, partition_key=event_id)
            event_doc["GeminiSummary"] = summary
            event_doc["Status"] = "Complete"
            event_doc["EndTime"] = end_time
            self.container.replace_item(item=event_id, body=event_doc)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to update Gemini summary for {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return False
    
    def get_event(self, event_id):
        """Retrieve an event document by ID."""
        try:
            return self.container.read_item(item=event_id, partition_key=event_id)
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to retrieve event {event_id}")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return None
    
    def query_events(self, query_text):
        """Query events containing specific text in VideoSummary."""
        query = f"SELECT * FROM c WHERE CONTAINS(LOWER(c.VideoSummary), LOWER('{query_text}'))"
        try:
            items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
            return items
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to query events")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return []
    
    def get_recent_events(self, limit=10):
        """
        Retrieve the most recent events ordered by StartTime.
        Used for providing context/memory to AI models.
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            list: Recent events ordered by StartTime (newest first)
        """
        query = f"SELECT TOP {limit} * FROM c ORDER BY c.StartTime DESC"
        try:
            items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
            return items
        except exceptions.CosmosHttpResponseError as e:
            print(f"[DB ERROR] Failed to retrieve recent events")
            print(f"[DB ERROR] Reason: {str(e).split(',')[0]}")
            return []
