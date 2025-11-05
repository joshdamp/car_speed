import cv2
import numpy as np
from collections import defaultdict
import math

# Constants
PIXELS_PER_METER = 8.8
FRAMES_PER_SECOND = 18
MIN_DETECTION_AREA = 1200  # Minimum area to consider as valid detection
TRACKING_THRESHOLD = 80  # Reduced from 150 - stricter tracking to prevent false associations
NMS_THRESHOLD = 0.15  # Very aggressive - merge if any overlap exists

def apply_nms(detections, nms_threshold=NMS_THRESHOLD):
    """Apply Non-Maximum Suppression with merging to remove overlapping detections"""
    if len(detections) <= 1:
        return detections
    
    detections = list(detections)
    keep = []
    
    while detections:
        # Start with the first detection
        current = detections.pop(0)
        merged = [current]
        
        # Find all overlapping detections
        to_remove = []
        for i, other in enumerate(detections):
            x1_min = max(current[0], other[0])
            y1_min = max(current[1], other[1])
            x1_max = min(current[0] + current[2], other[0] + other[2])
            y1_max = min(current[1] + current[3], other[1] + other[3])
            
            if x1_max > x1_min and y1_max > y1_min:
                # Calculate IoU
                intersection = (x1_max - x1_min) * (y1_max - y1_min)
                area1 = current[2] * current[3]
                area2 = other[2] * other[3]
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # If overlapping enough, merge them
                if iou >= nms_threshold:
                    merged.append(other)
                    to_remove.append(i)
        
        # Remove merged detections from list (in reverse order to maintain indices)
        for i in sorted(to_remove, reverse=True):
            detections.pop(i)
        
        # Merge all overlapping boxes into one
        merged_box = merge_boxes([np.array(m) for m in merged])
        keep.append(tuple(merged_box))
    
    return keep

def merge_boxes(boxes):
    """Merge multiple overlapping boxes into one"""
    if len(boxes) == 0:
        return boxes[0]
    
    boxes = np.array(boxes)
    
    # Calculate bounding box of all boxes
    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 0] + boxes[:, 2])
    y_max = np.max(boxes[:, 1] + boxes[:, 3])
    
    w = x_max - x_min
    h = y_max - y_min
    
    return [x_min, y_min, w, h]

def preprocess_frame(gray):
    """Apply preprocessing to improve vehicle detection"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Closing: fill small holes inside objects
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Opening: remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}  # Store vehicle positions over time
        self.next_id = 0
        self.speeds = {}  # Store current speeds (real-time)
        self.tracked_vehicles = set()  # Vehicles that have been speed-checked
        self.speed_history = {}  # Store speed history for each vehicle
        
    def get_velocity(self, vehicle_id):
        """Calculate current velocity of a vehicle"""
        if vehicle_id not in self.vehicles:
            return None
        
        positions = self.vehicles[vehicle_id]['positions']
        
        # Need at least 2 positions to calculate velocity
        if len(positions) < 2:
            return 0
        
        # Use last 3 positions for velocity calculation
        recent = positions[-3:] if len(positions) >= 3 else positions
        
        start_pos = recent[0]
        end_pos = recent[-1]
        
        # Calculate pixel distance
        pixel_distance = math.sqrt(
            (end_pos[0] - start_pos[0])**2 + 
            (end_pos[1] - start_pos[1])**2
        )
        
        # Calculate time elapsed
        frames_elapsed = end_pos[2] - start_pos[2]
        
        if frames_elapsed == 0:
            return 0
        
        return pixel_distance / frames_elapsed
    
    def update(self, detections, frame_number):
        """Update vehicle positions and calculate speeds"""
        current_centroids = []
        
        # Calculate centroids of current detections
        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy, x, y, w, h))
        
        # Match current detections with existing vehicles
        if len(self.vehicles) == 0:
            for centroid in current_centroids:
                self.vehicles[self.next_id] = {
                    'positions': [(centroid[0], centroid[1], frame_number)],
                    'bbox': (centroid[2], centroid[3], centroid[4], centroid[5]),
                    'first_detected': frame_number
                }
                self.next_id += 1
        else:
            # Find closest match for each current detection
            matched = set()
            for centroid in current_centroids:
                min_dist = float('inf')
                min_id = -1
                
                for vid, data in self.vehicles.items():
                    if vid in matched:
                        continue
                    last_pos = data['positions'][-1]
                    dist = math.sqrt((centroid[0] - last_pos[0])**2 + 
                                   (centroid[1] - last_pos[1])**2)
                    
                    # Only match if distance is within threshold
                    if dist < TRACKING_THRESHOLD:
                        # Additional check: if vehicle is stationary, be more strict
                        velocity = self.get_velocity(vid)
                        if velocity is not None and velocity < 1.0:  # Stationary or very slow
                            # For stationary vehicles, require tighter match (50% of threshold)
                            if dist < TRACKING_THRESHOLD * 0.5:
                                if dist < min_dist:
                                    min_dist = dist
                                    min_id = vid
                        else:
                            # For moving vehicles, normal threshold
                            if dist < min_dist:
                                min_dist = dist
                                min_id = vid
                
                if min_id != -1:
                    # Update existing vehicle
                    self.vehicles[min_id]['positions'].append(
                        (centroid[0], centroid[1], frame_number)
                    )
                    self.vehicles[min_id]['bbox'] = (
                        centroid[2], centroid[3], centroid[4], centroid[5]
                    )
                    matched.add(min_id)
                else:
                    # New vehicle
                    self.vehicles[self.next_id] = {
                        'positions': [(centroid[0], centroid[1], frame_number)],
                        'bbox': (centroid[2], centroid[3], centroid[4], centroid[5]),
                        'first_detected': frame_number
                    }
                    self.next_id += 1
        
        # Remove old vehicles that haven't been seen in recent frames
        to_remove = []
        for vid, data in self.vehicles.items():
            last_frame = data['positions'][-1][2]
            # Remove if not seen in 25 frames (increased from 15 for longer display)
            if last_frame < frame_number - 25:
                to_remove.append(vid)
        
        for vid in to_remove:
            # Clean up from speeds dictionary too
            if vid in self.speeds:
                del self.speeds[vid]
            if vid in self.tracked_vehicles:
                self.tracked_vehicles.discard(vid)
            del self.vehicles[vid]
    
    def calculate_speed(self, vehicle_id):
        """Calculate real-time speed for a vehicle based on recent movement"""
        if vehicle_id not in self.vehicles:
            return None
        
        positions = self.vehicles[vehicle_id]['positions']
        
        # Need at least 5 frames for reliable speed calculation
        # Reduced from 8 for faster detection
        if len(positions) < 5:
            return None
        
        # Use last 5 positions for faster speed measurement
        # Reduced from 8 to get much faster speed detection
        recent_positions = positions[-5:]
        
        # Calculate average speed over recent frames
        speeds = []
        for i in range(len(recent_positions) - 1):
            start_pos = recent_positions[i]
            end_pos = recent_positions[i + 1]
            
            # Calculate pixel distance
            pixel_distance = math.sqrt(
                (end_pos[0] - start_pos[0])**2 + 
                (end_pos[1] - start_pos[1])**2
            )
            
            # Calculate time elapsed (in seconds)
            frames_elapsed = end_pos[2] - start_pos[2]
            time_elapsed = frames_elapsed / FRAMES_PER_SECOND
            
            if time_elapsed > 0:
                # Convert to meters
                distance_meters = pixel_distance / PIXELS_PER_METER
                
                # Calculate speed in m/s then km/h
                speed_ms = distance_meters / time_elapsed
                speed_kmh = speed_ms * 3.6
                
                speeds.append(speed_kmh)
        
        # Return average of recent speeds, or None if no valid speeds
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            return avg_speed
        
        return None
    
    def get_display_info(self, current_frame_number):
        """Get vehicles and their speeds for display"""
        display_data = []
        
        for vid, data in self.vehicles.items():
            bbox = data['bbox']
            
            # Only display vehicles that were detected in recent frames (within last 10 frames)
            last_frame = data['positions'][-1][2]
            if current_frame_number - last_frame > 10:
                continue  # Skip stale detections
            
            # Calculate speed after vehicle has been tracked for at least 10 frames
            # Reduced from 15 for much faster speed identification
            speed = None
            if len(data['positions']) >= 10:
                speed = self.calculate_speed(vid)
            
            # Update speeds with real-time values
            # Only store speed if vehicle is actually moving (> 5 km/h to avoid noise)
            if speed is not None and speed > 5:
                self.speeds[vid] = speed
                self.tracked_vehicles.add(vid)
            elif vid in self.speeds and (speed is None or speed <= 2):
                # Remove speed if vehicle has stopped
                del self.speeds[vid]
            
            speed_text = f"{int(self.speeds[vid])} km/h" if vid in self.speeds else "..."
            
            display_data.append({
                'id': vid,
                'bbox': bbox,
                'speed': speed_text,
                'has_speed': vid in self.speeds
            })
        
        return display_data

def main():
    # Load Haar Cascade
    car_cascade = cv2.CascadeClassifier('cars_haar.xml')
    
    # Open video
    cap = cv2.VideoCapture('cars.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Video writer (optional - to save output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_speed.mp4', fourcc, FRAMES_PER_SECOND, (width, height))
    
    frame_number = 0
    
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_number += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve detection
        gray = preprocess_frame(gray)
        
        # Detect vehicles
        cars = car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Changed from 1.1 - more selective
            minNeighbors=8,  # Increased from 6 - much stricter
            minSize=(50, 50)  # Increased from (40,40) - filter smaller detections
        )
        
        # Filter detections by area
        valid_cars = []
        for (x, y, w, h) in cars:
            if w * h > MIN_DETECTION_AREA:
                valid_cars.append((x, y, w, h))
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        valid_cars = apply_nms(valid_cars)
        # Update tracker
        tracker.update(valid_cars, frame_number)
        
        # Get display information
        display_data = tracker.get_display_info(frame_number)
        
        # Draw on frame
        for vehicle in display_data:
            x, y, w, h = vehicle['bbox']
            vehicle_id = vehicle['id']
            speed_text = vehicle['speed']
            
            # Draw bounding box in green
            box_color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw vehicle ID and speed in red
            text_color = (0, 0, 255)  # Red
            label = f"{speed_text}"
            cv2.putText(
                frame, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                text_color, 
                2
            )
        
        # Display frame info
        info_text = f"Frame: {frame_number} | Vehicles: {len(display_data)}"
        cv2.putText(
            frame, 
            info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Write frame
        out.write(frame)
        
        # Display
        cv2.imshow('Vehicle Speed Tracker', frame)
        
        # Check for quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n=== Speed Detection Summary ===")
    print(f"Total tracked vehicles: {tracker.next_id}")
    print(f"Vehicles with speed calculated: {len(tracker.speeds)}")
    print("\nVehicle Speeds:")
    for vid, speed in sorted(tracker.speeds.items()):
        print(f"  Vehicle {vid}: {speed:.2f} km/h")

if __name__ == "__main__":
    main()