import cv2
import numpy as np
from collections import defaultdict
import math

# Constants
PIXELS_PER_METER = 8.8
FRAMES_PER_SECOND = 18
MIN_DETECTION_AREA = 1200  # Minimum area to consider as valid detection
TRACKING_THRESHOLD = 150  # Increased to handle fast-moving vehicles
NMS_THRESHOLD = 0.1  # Extremely aggressive - merge almost any overlap
DETECTION_CUTOFF_Y = 50  # Y-coordinate where speed detection starts (higher = farther down)
MAX_FRAMES_UNSEEN = 40  # Increased frames before vehicle is removed (allows temporary occlusions)

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

def create_roi_mask(height, width):
    """Create a Region of Interest mask for the road lanes only"""
    # Define the road area polygon (hardcoded for highway footage)
    # Allow detection across the full height but constrain horizontally to road lanes
    pts = np.array([
        [0, 0],           # Top-left (allow full height)
        [width, 0],       # Top-right
        [width, height],  # Bottom-right
        [0, height]       # Bottom-left
    ], np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}  # Store vehicle positions over time
        self.next_id = 0
        self.speeds = {}  # Store current speeds (real-time)
        self.smoothed_speeds = {}  # Store smoothed speeds for better stability
        self.tracked_vehicles = set()  # Vehicles that have been speed-checked
        self.speed_history = {}  # Store speed history for each vehicle
        self.motion_history = {}  # Store motion history to detect stops
        self.stationary_vehicles = set()  # Vehicles that are confirmed stationary
        
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
    
    def is_stationary(self, vehicle_id):
        """Detect if vehicle is stationary based on motion history"""
        if vehicle_id not in self.motion_history:
            return False
        
        # Check last 12 frames of motion (more sensitive)
        recent_motions = self.motion_history[vehicle_id][-12:] if len(self.motion_history[vehicle_id]) >= 12 else self.motion_history[vehicle_id]
        
        if not recent_motions or len(recent_motions) == 0:
            return False
        
        # Calculate average motion in recent frames
        avg_motion = sum(recent_motions) / len(recent_motions)
        
        # If average motion is very low (<2 pixels per frame), vehicle is stationary
        # This is more reliable than checking individual frames
        return avg_motion < 2.0
    
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
                self.motion_history[self.next_id] = []
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
                        if vid in self.stationary_vehicles:
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
                    last_pos = self.vehicles[min_id]['positions'][-1]
                    motion_distance = math.sqrt((centroid[0] - last_pos[0])**2 + 
                                               (centroid[1] - last_pos[1])**2)
                    
                    self.vehicles[min_id]['positions'].append(
                        (centroid[0], centroid[1], frame_number)
                    )
                    self.vehicles[min_id]['bbox'] = (centroid[2], centroid[3], centroid[4], centroid[5])
                    
                    # Track motion history
                    if min_id not in self.motion_history:
                        self.motion_history[min_id] = []
                    self.motion_history[min_id].append(motion_distance)
                    
                    # Keep only last 50 motion readings
                    if len(self.motion_history[min_id]) > 50:
                        self.motion_history[min_id].pop(0)
                    
                    matched.add(min_id)
                else:
                    # New vehicle
                    self.vehicles[self.next_id] = {
                        'positions': [(centroid[0], centroid[1], frame_number)],
                        'bbox': (centroid[2], centroid[3], centroid[4], centroid[5]),
                        'first_detected': frame_number
                    }
                    self.motion_history[self.next_id] = []
                    self.next_id += 1
        
        # Remove old vehicles that haven't been seen in recent frames
        to_remove = []
        for vid, data in self.vehicles.items():
            last_frame = data['positions'][-1][2]
            # Remove if not seen in MAX_FRAMES_UNSEEN frames (allows occlusions/fast movement)
            if last_frame < frame_number - MAX_FRAMES_UNSEEN:
                to_remove.append(vid)
        
        for vid in to_remove:
            # Clean up from all tracking dictionaries
            if vid in self.speeds:
                del self.speeds[vid]
            if vid in self.smoothed_speeds:
                del self.smoothed_speeds[vid]
            if vid in self.tracked_vehicles:
                self.tracked_vehicles.discard(vid)
            if vid in self.stationary_vehicles:
                self.stationary_vehicles.discard(vid)
            if vid in self.motion_history:
                del self.motion_history[vid]
            del self.vehicles[vid]
    
    def calculate_speed(self, vehicle_id):
        """Calculate real-time speed for a vehicle based on recent movement"""
        if vehicle_id not in self.vehicles:
            return None
        
        positions = self.vehicles[vehicle_id]['positions']
        
        # Need at least 5 frames for reliable speed calculation
        if len(positions) < 5:
            return None
        
        # Use last 5 positions for faster speed measurement
        recent_positions = positions[-5:]
        
        # Calculate speeds between consecutive frames
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
        
        # Return median of recent speeds (more robust than mean, filters outliers)
        if speeds:
            speeds_sorted = sorted(speeds)
            median_speed = speeds_sorted[len(speeds_sorted) // 2]
            return median_speed
        
        return None
    
    def smooth_speed(self, vehicle_id, current_speed):
        """Apply exponential moving average smoothing to speed"""
        if vehicle_id not in self.smoothed_speeds:
            self.smoothed_speeds[vehicle_id] = current_speed
            return current_speed
        
        # Exponential moving average with factor 0.3 (less aggressive smoothing)
        # Allows speed to change more quickly and realistically
        smoothing_factor = 0.3
        smoothed = (smoothing_factor * current_speed + 
                   (1 - smoothing_factor) * self.smoothed_speeds[vehicle_id])
        self.smoothed_speeds[vehicle_id] = smoothed
        
        return smoothed
    
    def get_display_info(self, current_frame_number):
        """Get vehicles and their speeds for display"""
        display_data = []
        
        for vid, data in self.vehicles.items():
            bbox = data['bbox']
            
            # Only display vehicles that were detected in recent frames (within last 10 frames)
            last_frame = data['positions'][-1][2]
            if current_frame_number - last_frame > 10:
                continue  # Skip stale detections
            
            # Get bounding box for perspective check
            x, y, w, h = bbox
            bbox_center_y = y + h  # Bottom of bounding box
            
            # Only calculate speed for vehicles below the detection cutoff (consistent perspective)
            # Vehicles above the cutoff are too close to camera and have perspective distortion
            if bbox_center_y < DETECTION_CUTOFF_Y:
                # Don't calculate speed for vehicles above cutoff - they're too close
                speed = None
            else:
                # Calculate speed after vehicle has been tracked for at least 10 frames
                speed = None
                if len(data['positions']) >= 10:
                    speed = self.calculate_speed(vid)
            
            # Apply smoothing to reduce jitter
            if speed is not None:
                speed = self.smooth_speed(vid, speed)
            
            # Detect if vehicle is stationary FIRST - this takes priority
            is_stationary = self.is_stationary(vid)
            
            # If stationary, never show speed
            if is_stationary:
                self.stationary_vehicles.add(vid)
                if vid in self.speeds:
                    del self.speeds[vid]
                if vid in self.smoothed_speeds:
                    del self.smoothed_speeds[vid]
            # Otherwise, update speeds with real-time values
            # Use smoothed speed for display, with threshold of 1 km/h (more sensitive)
            elif speed is not None and speed > 1:
                self.speeds[vid] = speed
                self.tracked_vehicles.add(vid)
                # Remove from stationary set if it starts moving again
                if vid in self.stationary_vehicles:
                    self.stationary_vehicles.discard(vid)
            else:
                # Speed too low or None - treat as stationary
                self.stationary_vehicles.add(vid)
                if vid in self.speeds:
                    del self.speeds[vid]
                if vid in self.smoothed_speeds:
                    del self.smoothed_speeds[vid]
            
            speed_text = f"{int(self.speeds[vid])} km/h" if vid in self.speeds else "..."
            
            display_data.append({
                'id': vid,
                'bbox': bbox,
                'speed': speed_text,
                'has_speed': vid in self.speeds,
                'is_stationary': is_stationary
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
    
    # Initialize background subtractor (MOG2 for dynamic background removal)
    # Tuned for fast-moving vehicles: lower history = faster adaptation, lower varThreshold = more sensitive
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=40, detectShadows=False)
    
    # Create ROI mask for the road area
    roi_mask = create_roi_mask(height, width)
    
    # Flag to enable/disable MOG2 (disabled for now - pure Haar cascade works better)
    use_mog2 = False
    
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Video writer (optional - to save output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_speed.mp4', fourcc, FRAMES_PER_SECOND, (width, height))
    
    frame_number = 0
    
    print("Processing video... Press 'q' to quit")
    print(f"Video dimensions: {width}x{height}")
    print("Enhancements enabled: MOG2 Background Subtraction + ROI masking + Position Smoothing")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_number += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # MOG2 background subtraction disabled - pure Haar cascade works better for this video
        # If you want to re-enable, set use_mog2 = True above
        if use_mog2:
            # Apply MOG2 background subtraction to isolate moving vehicles (only below detection cutoff)
            fg_mask = bg_subtractor.apply(gray)
            
            # Create a mask that only applies background subtraction below the detection zone
            detection_zone_mask = np.zeros((height, width), dtype=np.uint8)
            detection_zone_mask[DETECTION_CUTOFF_Y:, :] = 255  # Only the area below the yellow line
            
            # Apply the detection zone mask to foreground mask
            fg_mask = cv2.bitwise_and(fg_mask, detection_zone_mask)
            
            # Apply morphological operations to clean up the foreground mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply ROI mask to focus only on road area
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
            
            # Combine foreground mask with grayscale for enhanced detection
            gray_enhanced = cv2.bitwise_and(gray, gray, mask=fg_mask)
            
            # For areas above detection cutoff, use original gray (no background subtraction)
            gray_above_cutoff = gray.copy()
            gray_above_cutoff[:DETECTION_CUTOFF_Y, :] = gray[:DETECTION_CUTOFF_Y, :]
            
            # Blend: use enhanced below cutoff, original above cutoff
            gray = np.where(detection_zone_mask[:, :, None] > 0, gray_enhanced[:, :, None], gray_above_cutoff[:, :, None]).squeeze()
        
        # Apply preprocessing to improve detection
        gray = preprocess_frame(gray)
        
        # Detect vehicles
        cars = car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Standard scale detection
            minNeighbors=7,  # Stricter - filters out fences and false positives
            minSize=(40, 40)  # Reasonable minimum size
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
        
        # Draw detection cutoff line
        cv2.line(frame, (0, DETECTION_CUTOFF_Y), (width, DETECTION_CUTOFF_Y), (0, 255, 255), 2)
        cv2.putText(
            frame, 
            "Detection Zone", 
            (10, DETECTION_CUTOFF_Y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1
        )
        
        # Draw on frame
        for vehicle in display_data:
            x, y, w, h = vehicle['bbox']
            vehicle_id = vehicle['id']
            speed_text = vehicle['speed']
            
            # Draw bounding box in green
            box_color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Only draw speed text if vehicle is below the detection cutoff
            if y + h > DETECTION_CUTOFF_Y:
                # Draw vehicle ID and speed in red
                text_color = (0, 0, 255)  # Red
                label = f"{speed_text}"
                cv2.putText(
                    frame, 
                    label, 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
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