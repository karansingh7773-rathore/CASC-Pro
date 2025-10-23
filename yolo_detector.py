"""
YOLOv8n Person Detection Module
"""
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] ultralytics not installed. Please install: pip install ultralytics")

import cv2
import config

class YOLODetector:
    def __init__(self):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        print("[YOLO] Loading YOLOv8n model...")
        try:
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print("[YOLO] Model loaded successfully")
        except Exception as e:
            print(f"[YOLO ERROR] Failed to load model: {e}")
            print("[YOLO] Model will be downloaded on first use...")
            self.model = YOLO('yolov8n.pt')  # Auto-download
    
    def detect_people(self, frame):
        """
        Detect people in frame using YOLOv8n.
        
        Args:
            frame: OpenCV image frame
        
        Returns:
            tuple: (annotated_frame, has_person, person_count, detections)
        """
        # Run inference
        results = self.model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        # Get detections
        detections = results[0].boxes
        
        # Filter for 'person' class (class_id = 0 in COCO)
        person_detections = []
        for box in detections:
            class_id = int(box.cls[0])
            if class_id == config.YOLO_PERSON_CLASS_ID:
                person_detections.append(box)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Add person count
        person_count = len(person_detections)
        has_person = person_count > 0
        
        cv2.putText(annotated_frame, f"People Detected: {person_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame, has_person, person_count, person_detections
