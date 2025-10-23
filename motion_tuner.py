"""
Motion Detection Parameter Tuner
Run this script to visually adjust motion detection sensitivity
"""
import cv2
import config

def nothing(x):
    pass

def main():
    print("="*70)
    print("CASC - Motion Detection Parameter Tuner")
    print("="*70)
    print("Use the sliders to adjust motion detection parameters.")
    print("Press 'q' to quit and save values.")
    print("="*70 + "\n")
    
    # Open video source
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Get video FPS
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if video_fps == 0:
        video_fps = 30
    frame_delay = max(1, int(1000 / video_fps))
    
    # Create window with trackbars
    cv2.namedWindow('Motion Detection Tuner')
    cv2.createTrackbar('Min Area', 'Motion Detection Tuner', config.MIN_CONTOUR_AREA, 10000, nothing)
    cv2.createTrackbar('Blur Size', 'Motion Detection Tuner', config.MOTION_BLUR_SIZE, 51, nothing)
    cv2.createTrackbar('Threshold', 'Motion Detection Tuner', config.MOTION_THRESHOLD, 100, nothing)
    
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            prev_frame = None
            continue
        
        # Get current trackbar values
        min_area = cv2.getTrackbarPos('Min Area', 'Motion Detection Tuner')
        blur_size = cv2.getTrackbarPos('Blur Size', 'Motion Detection Tuner')
        threshold = cv2.getTrackbarPos('Threshold', 'Motion Detection Tuner')
        
        # Ensure blur size is odd
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size < 3:
            blur_size = 3
        
        if prev_frame is not None:
            # Convert to grayscale
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply blur
            gray1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 0)
            gray2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 0)
            
            # Compute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Threshold
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate
            thresh = cv2.dilate(thresh, None, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours and filter by size
            motion_detected = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Area: {int(area)}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display status
            status = "MOTION DETECTED" if motion_detected else "NO MOTION"
            color = (0, 0, 255) if motion_detected else (255, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display parameters
            cv2.putText(frame, f"Min Area: {min_area}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Blur: {blur_size}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Threshold: {threshold}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show threshold image in corner
            thresh_resized = cv2.resize(thresh, (320, 180))
            frame[0:180, frame.shape[1]-320:frame.shape[1]] = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
        
        prev_frame = frame.copy()
        
        cv2.imshow('Motion Detection Tuner', frame)
        
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            print("\n" + "="*70)
            print("Optimal Parameters Found:")
            print("="*70)
            print(f"MIN_CONTOUR_AREA = {min_area}")
            print(f"MOTION_BLUR_SIZE = {blur_size}")
            print(f"MOTION_THRESHOLD = {threshold}")
            print("\nUpdate these values in config.py")
            print("="*70)
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
