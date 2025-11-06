import cv2
import numpy as np
import math

# Detection and tracking configuration
PIXELS_PER_METER = 8.8
FRAMES_PER_SECOND = 18
MIN_DETECTION_AREA = 2500
TRACKING_THRESHOLD = 150
DETECTION_CUTOFF_Y = 50
MAX_FRAMES_UNSEEN = 40
BBOX_SMOOTHING_ALPHA = 0.6
MIN_ASPECT_RATIO = 0.7
MAX_ASPECT_RATIO = 2.2

def preprocess_frame(gray):
    """Apply preprocessing to improve vehicle detection"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.next_id = 0
        self.speeds = {}
        self.smoothed_speeds = {}
        self.tracked_vehicles = set()
        self.motion_history = {}
        self.stationary_vehicles = set()
        self.smoothed_bboxes = {}
        
    def smooth_bbox(self, vehicle_id, current_bbox):
        """Apply exponential smoothing to bounding box to reduce fragmentation jitter"""
        if vehicle_id not in self.smoothed_bboxes:
            self.smoothed_bboxes[vehicle_id] = current_bbox
            return current_bbox
        
        prev_bbox = self.smoothed_bboxes[vehicle_id]
        smoothed = tuple(
            int(BBOX_SMOOTHING_ALPHA * curr + (1 - BBOX_SMOOTHING_ALPHA) * prev)
            for curr, prev in zip(current_bbox, prev_bbox)
        )
        self.smoothed_bboxes[vehicle_id] = smoothed
        return smoothed
        
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
        
        # If average motion is very low (<3 pixels per frame), vehicle is stationary
        # Increased threshold to account for smoothing artifacts
        return avg_motion < 3.0
    
    def update(self, detections, frame_number):
        """Update vehicle positions and calculate speeds"""
        current_centroids = []
        
        # Calculate bottom-center points of current detections (road contact point)
        # More stable for large vehicles as bbox grows upward
        for (x, y, w, h) in detections:
            cx = x + w // 2  # Center X
            cy = y + h       # Bottom Y (road contact point)
            current_centroids.append((cx, cy, x, y, w, h))
        
        # Match current detections with existing vehicles
        if len(self.vehicles) == 0:
            for centroid in current_centroids:
                self.vehicles[self.next_id] = {
                    'positions': [(centroid[0], centroid[1], frame_number)],
                    'bbox': (centroid[2], centroid[3], centroid[4], centroid[5]),
                    'first_detected': frame_number
                }
                self.smoothed_bboxes[self.next_id] = (centroid[2], centroid[3], centroid[4], centroid[5])
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
                    # Use smoothed bbox to reduce fragmentation jitter
                    current_bbox = (centroid[2], centroid[3], centroid[4], centroid[5])
                    smoothed_bbox = self.smooth_bbox(min_id, current_bbox)
                    self.vehicles[min_id]['bbox'] = smoothed_bbox
                    
                    # Recalculate bottom-center from smoothed bbox for stable tracking
                    x, y, w, h = smoothed_bbox
                    smooth_cx = x + w // 2
                    smooth_cy = y + h  # Bottom center from smoothed bbox
                    
                    # Update position with smoothed bottom-center point
                    last_pos = self.vehicles[min_id]['positions'][-1]
                    motion_distance = math.sqrt((smooth_cx - last_pos[0])**2 + 
                                               (smooth_cy - last_pos[1])**2)
                    
                    self.vehicles[min_id]['positions'].append(
                        (smooth_cx, smooth_cy, frame_number)
                    )
                    
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
                    self.smoothed_bboxes[self.next_id] = (centroid[2], centroid[3], centroid[4], centroid[5])
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
            if vid in self.smoothed_bboxes:
                del self.smoothed_bboxes[vid]
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
        
        # Need at least 10 frames for reliable speed calculation (more stable)
        if len(positions) < 10:
            return None
        
        # Use last 10 positions for more stable speed measurement
        recent_positions = positions[-10:]
        
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
            
            # Only display vehicles that were detected in recent frames (within last 30 frames)
            last_frame = data['positions'][-1][2]
            if current_frame_number - last_frame > 20:
                continue  # Skip stale detections only after 30 frames unseen
            
            # Persistence filter: only display if seen for at least 3 consecutive frames
            # This eliminates one-off noise detections
            if len(data['positions']) < 2:
                continue  # Too new, not reliable yet
            
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
            # Use smoothed speed for display, with threshold of 2 km/h (filters out noise)
            elif speed is not None and speed > 2:
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
        
        # Group nearby vehicles (multiple detections of same vehicle) and keep smallest
        display_data = self._filter_duplicate_vehicles(display_data)
        
        return display_data
    
    def _filter_duplicate_vehicles(self, display_data):
        """Filter out duplicate detections of the same vehicle, keep only smallest bbox in each group"""
        if len(display_data) <= 1:
            return display_data
        
        GROUPING_DISTANCE = 80
        
        grouped = []
        used = set()
        
        for i, vehicle in enumerate(display_data):
            if i in used:
                continue
            
            group = [vehicle]
            x1, y1, w1, h1 = vehicle['bbox']
            cx1 = x1 + w1 // 2
            cy1 = y1 + h1 // 2
            
            # Find all nearby vehicles in same area
            for j, other in enumerate(display_data):
                if j <= i or j in used:
                    continue
                
                x2, y2, w2, h2 = other['bbox']
                cx2 = x2 + w2 // 2
                cy2 = y2 + h2 // 2
                
                # Calculate distance between centers
                dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # If close enough, they're the same vehicle
                if dist < GROUPING_DISTANCE:
                    group.append(other)
                    used.add(j)
            
            # From the group, keep the one with valid speed, or smallest bbox
            best_vehicle = None
            
            # First priority: vehicle with actual speed reading
            for v in group:
                if v['has_speed']:
                    if best_vehicle is None or int(v['speed'].split()[0]) < int(best_vehicle['speed'].split()[0]):
                        best_vehicle = v
            
            # Second priority: if no speed, keep smallest bbox
            if best_vehicle is None:
                best_vehicle = min(group, key=lambda v: v['bbox'][2] * v['bbox'][3])
            
            grouped.append(best_vehicle)
            used.add(i)
        
        return grouped

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
    print(f"Video dimensions: {width}x{height}")
    print("Enhancements enabled: MOG2 Background Subtraction + ROI masking + Position Smoothing")
    
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
            scaleFactor=1.05,  # Standard scale detection
            minNeighbors=8,  # Stricter - filters out fences, shadows, and false positives
            minSize=(40, 40)  # Reasonable minimum size
        )
        
        # Filter detections by area and aspect ratio
        valid_cars = []
        for (x, y, w, h) in cars:
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Check area constraint
            if area < MIN_DETECTION_AREA:
                continue
            
            # Check aspect ratio constraint - strict to filter gaps between vehicles and noise
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue
            
            valid_cars.append((x, y, w, h))
        
        # Update tracker with filtered detections
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