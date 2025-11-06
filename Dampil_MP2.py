import cv2
import numpy as np
import math

# Detection and tracking configuration
PIXELS_PER_METER = 8.8
FRAMES_PER_SECOND = 18
MIN_DETECTION_AREA = 2500
TRACKING_THRESHOLD = 150
DETECTION_CUTOFF_Y = 50
MAX_FRAMES_UNSEEN = 10
BBOX_SMOOTHING_ALPHA = 0.6
MIN_ASPECT_RATIO = 0.7
MAX_ASPECT_RATIO = 2.2

# Pre-create reusable preprocessing components

def preprocess_frame(gray):
    """Gentle CLAHE + Gaussian blur preprocessing"""
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.next_id = 0
        self.speeds = {}
        self.motion_history = {}
        self.smoothed_bboxes = {}
        
    def smooth_bbox(self, vehicle_id, current_bbox):
        """Apply exponential smoothing to bounding box to reduce fragmentation jitter"""
        if vehicle_id not in self.smoothed_bboxes:
            self.smoothed_bboxes[vehicle_id] = current_bbox
            return current_bbox
        
        prev_bbox = self.smoothed_bboxes[vehicle_id]
        alpha = BBOX_SMOOTHING_ALPHA
        beta = 1 - alpha
        smoothed = (
            int(alpha * current_bbox[0] + beta * prev_bbox[0]),
            int(alpha * current_bbox[1] + beta * prev_bbox[1]),
            int(alpha * current_bbox[2] + beta * prev_bbox[2]),
            int(alpha * current_bbox[3] + beta * prev_bbox[3])
        )
        self.smoothed_bboxes[vehicle_id] = smoothed
        return smoothed
    
    def is_stationary(self, vehicle_id):
        """Detect if vehicle is stationary based on motion history"""
        if vehicle_id not in self.motion_history or len(self.motion_history[vehicle_id]) < 8:
            return False
        
        # Check last 8 frames of motion
        recent_motions = self.motion_history[vehicle_id][-8:]
        avg_motion = sum(recent_motions) / len(recent_motions)
        
        return avg_motion < 2.5
    
    def update(self, detections, frame_number):
        """Update vehicle positions and calculate speeds"""
        # Pre-compute centroids
        centroids_data = [
            (x + w // 2, y + h, x, y, w, h)
            for x, y, w, h in detections
        ]
        
        if not self.vehicles:
            # Initialize new vehicles
            for i, (cx, cy, x, y, w, h) in enumerate(centroids_data):
                vid = self.next_id + i
                self.vehicles[vid] = {
                    'positions': [(cx, cy, frame_number)],
                    'bbox': (x, y, w, h),
                    'first_detected': frame_number
                }
                self.smoothed_bboxes[vid] = (x, y, w, h)
                self.motion_history[vid] = []
            self.next_id += len(centroids_data)
        else:
            # Match detections to existing vehicles
            threshold_sq = TRACKING_THRESHOLD ** 2  # Avoid sqrt computation
            matched = set()
            
            for cx, cy, x, y, w, h in centroids_data:
                min_dist_sq = float('inf')
                min_id = -1
                
                for vid, data in self.vehicles.items():
                    if vid in matched:
                        continue
                    
                    lx, ly, _ = data['positions'][-1]
                    dist_sq = (cx - lx) ** 2 + (cy - ly) ** 2
                    
                    if dist_sq < threshold_sq and dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_id = vid
                
                if min_id != -1:
                    # Update existing vehicle
                    smoothed_bbox = self.smooth_bbox(min_id, (x, y, w, h))
                    self.vehicles[min_id]['bbox'] = smoothed_bbox
                    
                    sx, sy, sw, sh = smoothed_bbox
                    smooth_cx = sx + sw // 2
                    smooth_cy = sy + sh
                    
                    last_pos = self.vehicles[min_id]['positions'][-1]
                    motion_dist_sq = (smooth_cx - last_pos[0]) ** 2 + (smooth_cy - last_pos[1]) ** 2
                    
                    self.vehicles[min_id]['positions'].append((smooth_cx, smooth_cy, frame_number))
                    
                    history = self.motion_history[min_id]
                    history.append(motion_dist_sq ** 0.5)
                    if len(history) > 30:
                        history.pop(0)
                    
                    matched.add(min_id)
                else:
                    # New vehicle
                    self.vehicles[self.next_id] = {
                        'positions': [(cx, cy, frame_number)],
                        'bbox': (x, y, w, h),
                        'first_detected': frame_number
                    }
                    self.smoothed_bboxes[self.next_id] = (x, y, w, h)
                    self.motion_history[self.next_id] = []
                    self.next_id += 1
        
        # Remove old vehicles - use list comprehension for efficiency
        cutoff_frame = frame_number - MAX_FRAMES_UNSEEN
        to_remove = [
            vid for vid, data in self.vehicles.items()
            if data['positions'][-1][2] < cutoff_frame
        ]
        
        for vid in to_remove:
            self.speeds.pop(vid, None)
            self.smoothed_bboxes.pop(vid, None)
            self.motion_history.pop(vid, None)
            del self.vehicles[vid]
    
    def calculate_speed(self, vehicle_id):
        """Calculate real-time speed for a vehicle based on recent movement"""
        positions = self.vehicles[vehicle_id]['positions']
        
        if len(positions) < 8:
            return None
        
        # Use last 8 positions
        recent_positions = positions[-8:]
        
        # Vectorized speed calculation
        speeds = []
        inv_fps = 1.0 / FRAMES_PER_SECOND
        inv_ppm = 1.0 / PIXELS_PER_METER
        
        for i in range(len(recent_positions) - 1):
            x0, y0, t0 = recent_positions[i]
            x1, y1, t1 = recent_positions[i + 1]
            
            pixel_dist_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
            if pixel_dist_sq == 0:
                continue
            
            pixel_dist = pixel_dist_sq ** 0.5
            time_s = (t1 - t0) * inv_fps
            
            if time_s > 0:
                speed_kmh = (pixel_dist * inv_ppm / time_s) * 3.6
                speeds.append(speed_kmh)
        
        if not speeds:
            return None
        
        # Return median
        speeds.sort()
        return speeds[len(speeds) // 2]
    
    def get_display_info(self, current_frame_number):
        """Get vehicles and their speeds for display"""
        display_data = []
        
        for vid, data in self.vehicles.items():
            bbox = data['bbox']
            last_frame = data['positions'][-1][2]
            
            # Skip old detections
            if current_frame_number - last_frame > 5:
                continue
            
            # Skip if tracked for less than 2 frames
            if len(data['positions']) < 2:
                continue
            
            x, y, w, h = bbox
            
            # Calculate speed only if below detection cutoff and tracked long enough
            speed = None
            if y + h > DETECTION_CUTOFF_Y and len(data['positions']) >= 8:
                speed = self.calculate_speed(vid)
            
            # Determine if stationary
            is_stationary = self.is_stationary(vid)
            
            # Store speed if valid and moving
            if speed is not None and speed > 2 and not is_stationary:
                self.speeds[vid] = speed
                speed_text = f"{int(speed)} km/h"
            else:
                if vid in self.speeds:
                    del self.speeds[vid]
                speed_text = "..."
            
            display_data.append({
                'id': vid,
                'bbox': bbox,
                'speed': speed_text,
                'has_speed': vid in self.speeds
            })
        
        return self._filter_duplicate_vehicles(display_data)
    
    def _filter_duplicate_vehicles(self, display_data):
        """Filter out duplicate detections of the same vehicle, keep only smallest bbox in each group"""
        if len(display_data) <= 1:
            return display_data
        
        GROUPING_DISTANCE_SQ = 80 ** 2  # Pre-compute squared distance
        grouped = []
        used = set()
        
        for i, vehicle in enumerate(display_data):
            if i in used:
                continue
            
            group = [vehicle]
            x1, y1, w1, h1 = vehicle['bbox']
            cx1 = x1 + w1 // 2
            cy1 = y1 + h1 // 2
            
            # Find all nearby vehicles
            for j in range(i + 1, len(display_data)):
                if j in used:
                    continue
                
                other = display_data[j]
                x2, y2, w2, h2 = other['bbox']
                cx2 = x2 + w2 // 2
                cy2 = y2 + h2 // 2
                
                # Use squared distance to avoid sqrt
                dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                
                if dist_sq < GROUPING_DISTANCE_SQ:
                    group.append(other)
                    used.add(j)
            
            # Keep vehicle with speed, else smallest bbox
            best = next((v for v in group if v['has_speed']), None)
            if best is None:
                best = min(group, key=lambda v: v['bbox'][2] * v['bbox'][3])
            
            grouped.append(best)
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
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_number += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve detection
        gray = preprocess_frame(gray)
        
        # Detect vehicles - more conservative parameters
        cars = car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,      # Increased from 1.05 (less sensitive)
            minNeighbors=10,      # Increased from 8 (stricter grouping)
            minSize=(50, 50),     # Increased from (40, 40) (larger minimum)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter detections by area and aspect ratio (vectorized)
        valid_cars = [
            (x, y, w, h) for x, y, w, h in cars
            if MIN_DETECTION_AREA <= w * h and 
               MIN_ASPECT_RATIO <= (w / h if h > 0 else 0) <= MAX_ASPECT_RATIO
        ]
        
        # Update tracker with filtered detections
        tracker.update(valid_cars, frame_number)
        
        # Get display information
        display_data = tracker.get_display_info(frame_number)
        
        # Draw vehicles
        for vehicle in display_data:
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if y + h > DETECTION_CUTOFF_Y:
                cv2.putText(frame, vehicle['speed'], (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display info
        cv2.putText(frame, f"Frame: {frame_number} | Vehicles: {len(display_data)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
    for vid, speed in sorted(tracker.speeds.items()):
        print(f"  Vehicle {vid}: {speed:.2f} km/h")

if __name__ == "__main__":
    main()