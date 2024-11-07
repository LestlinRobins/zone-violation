import cv2
import numpy as np
import time
from datetime import datetime
import argparse
import torch

class ZoneViolationDetector:
    def __init__(self, video_source=0, confidence_threshold=0.2, zone_coords=None, target_classes=None):
        """
        Initialize the Zone Violation Detector
        
        Args:
            video_source: Camera index or video file path
            confidence_threshold: Minimum confidence for detection
            zone_coords: List of points defining zone boundary [(x1,y1), (x2,y2), ...]
            target_classes: List of classes to detect in the zone
        """
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.zone_coords = zone_coords
        
        # Initialize video capture
        self.cap = cv2.VideoCapture("video2.mp4")
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {video_source}")
            
        # Get the first frame to initialize dimensions
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read from video source")
            
        self.height, self.width = self.frame.shape[:2]
        
        # If zone not provided, initialize to None (will be set interactively)
        self.zone_polygon = None
        if zone_coords:
            self.zone_polygon = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Classes to detect
        self.target_classes = target_classes if target_classes else ['car', 'truck', 'person', 'bottle', 'backpack', 'bus']
        self.class_ids = [self.model.names.index(cls) for cls in self.target_classes if cls in self.model.names]
        
        # For violation tracking
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts
        self.violations = []
        self.violation_count = 0
        
    def select_zone(self):
        """Allow user to draw a polygon zone on the first frame"""
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(self.frame, points[-2], points[-1], (0, 255, 0), 2)
                cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Zone", self.frame)
                
        # Create a copy of the frame for drawing
        selecting_frame = self.frame.copy()
        self.frame = selecting_frame
        
        cv2.imshow("Select Zone", self.frame)
        cv2.setMouseCallback("Select Zone", mouse_callback)
        
        print("Click to select points for the restricted zone. Press 'c' to complete or 'r' to reset.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(points) > 2:
                # Complete the polygon
                cv2.line(self.frame, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow("Select Zone", self.frame)
                self.zone_polygon = np.array(points, np.int32).reshape((-1, 1, 2))
                break
            elif key == ord('r'):
                # Reset the points
                points = []
                self.frame = selecting_frame.copy()
                cv2.imshow("Select Zone", self.frame)
        
        cv2.waitKey(1000)
        cv2.destroyWindow("Select Zone")
        return self.zone_polygon
    
    def is_in_zone(self, x, y, w, h):
        """Check if object is in the defined zone using its bottom center point"""
        if self.zone_polygon is None:
            return False
        
        # Use the bottom center point of the bounding box
        bottom_center = (int(x + w/2), int(y + h))
        
        # Check if the point is inside the polygon
        return cv2.pointPolygonTest(self.zone_polygon, bottom_center, False) >= 0
    
    def detect_violations(self, frame):
        """Detect objects and check if they're in the restricted zone"""
        # Convert frame to RGB for the model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_frame)
        detections = results.pandas().xyxy[0]
        
        current_violations = []
        
        # Draw the zone
        if self.zone_polygon is not None:
            cv2.polylines(frame, [self.zone_polygon], True, (0, 255, 0), 2)
            
        # Process each detection
        for _, detection in detections.iterrows():
            if detection['confidence'] < self.confidence_threshold:
                continue
                
            class_id = int(detection['class'])
            if class_id not in self.class_ids:
                continue
                
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            w, h = x2 - x1, y2 - y1
            label = f"{self.model.names[class_id]}: {detection['confidence']:.2f}"
            
            # Check if the object is in the zone
            if self.is_in_zone(x1, y1, w, h):
                # Draw red for violation
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add to current violations
                current_violations.append({
                    'class': self.model.names[class_id],
                    'confidence': detection['confidence'],
                    'bbox': (x1, y1, x2, y2),
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # Draw green for normal detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Generate alert if violations detected
        current_time = time.time()
        if current_violations and (current_time - self.last_alert_time) > self.alert_cooldown:
            self.generate_alert(current_violations)
            self.last_alert_time = current_time
            
        # Add violation count to frame
        cv2.putText(
            frame, 
            f"Violations: {self.violation_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
            
        return frame
    
    def generate_alert(self, violations):
        """Generate an alert for violations"""
        self.violations.extend(violations)
        self.violation_count += len(violations)
        
        print("\n" + "="*50)
        print(f"⚠️ ALERT! Violation detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for v in violations:
            print(f"- {v['class']} detected in restricted zone (confidence: {v['confidence']:.2f})")
            
        print("="*50 + "\n")
        
        # Here you can add additional alert methods like:
        # - Sending email
        # - Push notification
        # - Writing to a log file
        # - Triggering an alarm sound
    
    def save_report(self, filename="violation_report.txt"):
        """Save a report of all violations"""
        with open(filename, 'w') as f:
            f.write(f"Violation Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Violations: {self.violation_count}\n\n")
            
            for i, v in enumerate(self.violations, 1):
                f.write(f"Violation #{i}:\n")
                f.write(f"  Time: {v['time']}\n")
                f.write(f"  Object: {v['class']}\n")
                f.write(f"  Confidence: {v['confidence']:.2f}\n")
                f.write(f"  Bounding Box: {v['bbox']}\n\n")
    
    def run(self):
        """Main loop to process video frames"""
        if self.zone_polygon is None:
            self.select_zone()
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # If video file ends, restart
                if isinstance(self.video_source, str):
                    self.cap = cv2.VideoCapture(self.video_source)
                    continue
                else:
                    break
                    
            # Process the frame
            processed_frame = self.detect_violations(frame)
            
            # Display the result
            cv2.imshow("Zone Violation Detector", processed_frame)
            
            # Break the loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save report
        self.save_report()
        
def main():
    parser = argparse.ArgumentParser(description="Zone Violation Detector")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or file path)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--classes", type=str, default="car,truck,person,bottle,backpack", 
                       help="Target classes to detect separated by commas")
    
    args = parser.parse_args()
    
    # Parse video source (int for webcam index or string for file path)
    source = args.source
    if source.isdigit():
        source = int(source)
        
    # Parse classes
    target_classes = args.classes.split(',')
    
    detector = ZoneViolationDetector(
        video_source=source,
        confidence_threshold=args.confidence,
        target_classes=target_classes
    )
    
    detector.run()

if __name__ == "__main__":
    main()