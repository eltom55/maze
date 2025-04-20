#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
import time
import math
import numpy as np
import cv2
from collections import deque
from enum import Enum
from cv_bridge import CvBridge

# ROS 2 message types
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from ros_robot_controller_msgs.msg import MotorsState, MotorState

class RobotState(Enum):
    EXPLORING = 1
    NAVIGATING = 2
    TURNING = 3
    REACHED_DESTINATION = 4
    WALL_FOLLOWING = 5
    TRACKING_COLOR = 6

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class ColorRoom(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    PURPLE = 5
    GRAY = 6
    UNKNOWN = 0

class GridCell:
    def __init__(self):
        self.visited = False
        self.walls = [True, True, True, True]  # North, East, South, West
        self.color = ColorRoom.UNKNOWN
        self.distance = float('inf')  # Distance from start for pathfinding
        self.parent = None  # Parent cell for pathfinding

class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')
        
        # Debug mode
        self.debug = True
        
        # ROS Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.motor_pub = self.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 10)
        
        # ROS Subscribers
        self.create_subscription(Image, 'ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Maze representation
        self.grid_size = 20  # Initialize with a reasonable size
        self.grid = [[GridCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_position = (self.grid_size // 2, self.grid_size // 2)  # Start in the middle of the grid
        self.current_direction = Direction.NORTH
        
        # Room positions
        self.red_room_position = None
        self.blue_room_position = None
        self.room_positions = {}
        
        # Navigation variables
        self.state = RobotState.EXPLORING
        self.target_position = None
        self.path = []
        self.exploring_stack = []
        self.turning_target = None
        
        # Laser scan data
        self.laser_ranges = []
        self.min_front_distance = float('inf')
        self.min_left_distance = float('inf')
        self.min_right_distance = float('inf')
        self.min_back_distance = float('inf')
        
        # Color tracking variables
        self.tracking_blue = False
        self.centroid_x = None
        self.centroid_y = None
        self.centroid_area = None
        
        # Wall following variables
        self.turning_right = False
        self.right_turn_start_time = 0
        self.right_turn_end_time = 0
        self.following_wall = False
        
        # Robot parameters
        self.turn_90_duration = 2.0     # Time to turn 90 degrees (increase if turns are incomplete)
        self.move_cell_duration = 2.5   # Time to move one cell forward (adjust based on cell size)
        self.wall_threshold = 0.4       # Distance threshold to detect walls
        
        # Color detection parameters
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'red2': ([170, 100, 100], [180, 255, 255]),  # Red wraps around in HSV
            'green': ([40, 100, 100], [80, 255, 255]),
            'blue': ([100, 100, 100], [140, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255]),
            'purple': ([140, 100, 100], [170, 255, 255]),
            'gray': ([0, 0, 120], [180, 30, 200])
        }
        
        # LAB color space tracking (from example code)
        self.blue_lower_bound = np.array([42, 118, 80])
        self.blue_upper_bound = np.array([90, 151, 111])
        
        # Initialize log
        self.get_logger().info('Maze Solver initialized')
        
        # Map visualization
        self.map_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # LIDAR debugging variables
        self.last_lidar_time = time.time()
        self.lidar_received_count = 0
        self.lidar_debug_timer = self.create_timer(5.0, self.check_lidar_health)  # Check every 5 seconds
        
    def image_callback(self, msg):
        """Process camera image to detect colors"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_frame = cv_image
            
            # Process for color tracking (blue target from example code)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            lab = cv2.GaussianBlur(lab, (3, 3), 0)
            mask_lab = cv2.inRange(lab, self.blue_lower_bound, self.blue_upper_bound)
            
            self.centroid_x, self.centroid_y, self.centroid_area = self.get_color_centroid(mask_lab, cv_image)
            
            if self.centroid_x is not None:
                self.tracking_blue = True
                self.state = RobotState.TRACKING_COLOR
                
                # Target following logic
                height, width, _ = cv_image.shape
                frame_center_x = width // 2
                error_x = self.centroid_x - frame_center_x
                
                if abs(error_x) < 20:
                    self.send_motor_command([-1.0, -1.0, 1.2, 1.2])  # right wheels faster = turn CC
                elif error_x > 0:
                    self.send_motor_command([-1.2, -1.2, 1.0, 1.0])  # left wheels faster = turn counter CW
                else:
                    self.send_motor_command([-1.0, -1.0, 1.0, 1.0])  # Straight
            else:
                self.tracking_blue = False
                
            # Only process room colors if we're exploring
            if self.state == RobotState.EXPLORING:
                self.detect_room_color()
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def get_color_centroid(self, mask, frame):
        """Return the centroid of the largest contour in the binary image 'mask'"""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Set a minimum area required for contours
        min_contour_area = 100
        
        if not contours:
            return None, None, None
            
        largest_contour = max(contours, key=cv2.contourArea)
        centroid_area = cv2.contourArea(largest_contour)
        
        if centroid_area < min_contour_area:
            return None, None, None
            
        # Compute the centroid location of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0
            
        # Draw bounding box for visualization
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.debug:
            self.get_logger().info(f"Contour found with area: {centroid_area}")
            
        return centroid_x, centroid_y, centroid_area
    
    def lidar_callback(self, msg):
        """Process LiDAR data for wall detection"""
        try:
            # Update LIDAR health tracking
            self.last_lidar_time = time.time()
            self.lidar_received_count += 1
            
            # Print detailed scan info every 50 messages
            if self.lidar_received_count % 50 == 0:
                self.get_logger().info(f"LIDAR message #{self.lidar_received_count}: " +
                                     f"scan size={len(msg.ranges)}, " + 
                                     f"angle_min={msg.angle_min:.2f}, " +
                                     f"angle_max={msg.angle_max:.2f}, " +
                                     f"angle_increment={msg.angle_increment:.4f}")
            
            # Check if scan is empty
            if len(msg.ranges) == 0:
                self.get_logger().error("⚠️ Received empty LIDAR scan! ⚠️")
                return
            
            # Check for all-infinity values (which means no obstacles detected)
            valid_ranges = [r for r in msg.ranges if not math.isinf(r) and not math.isnan(r)]
            if not valid_ranges:
                self.get_logger().warning("⚠️ All LIDAR readings are infinite - no obstacles detected in range ⚠️")
                # Don't return, as this might be valid (e.g., in a very large room)
            
            # Store the full scan
            self.laser_ranges = msg.ranges
            
            # Calculate the front sector (0°±15°)
            front_indices = list(range(0, 31)) + list(range(len(msg.ranges)-33, len(msg.ranges)))
            front_ranges = [msg.ranges[i] for i in front_indices if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
            if front_ranges:
                self.min_front_distance = min(front_ranges)
                if self.debug and self.lidar_received_count % 10 == 0:
                    self.get_logger().info(f"Front distance: {self.min_front_distance:.2f}m (from {len(front_ranges)} points)")
            else:
                self.get_logger().warning("No valid front LIDAR readings")
                self.min_front_distance = float('inf')
            
            # Calculate the left sector (90°±30°)
            left_indices = list(range(93, 156))
            left_ranges = [msg.ranges[i] for i in left_indices if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
            if left_ranges:
                self.min_left_distance = min(left_ranges)
                if self.debug and self.lidar_received_count % 10 == 0:
                    self.get_logger().info(f"Left distance: {self.min_left_distance:.2f}m (from {len(left_ranges)} points)")
            else:
                self.get_logger().warning("No valid left LIDAR readings")
                self.min_left_distance = float('inf')
            
            # Calculate the right sector (270°±30°)
            right_indices = list(range(218, 281))
            right_ranges = [msg.ranges[i] for i in right_indices if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
            if right_ranges:
                self.min_right_distance = min(right_ranges)
                if self.debug and self.lidar_received_count % 10 == 0:
                    self.get_logger().info(f"Right distance: {self.min_right_distance:.2f}m (from {len(right_ranges)} points)")
            else:
                self.get_logger().warning("No valid right LIDAR readings")
                self.min_right_distance = float('inf')
            
            # Calculate the back sector (180°±30°)
            back_indices = list(range(343, 406))
            back_ranges = [msg.ranges[i] for i in back_indices if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
            if back_ranges:
                self.min_back_distance = min(back_ranges)
            else:
                self.min_back_distance = float('inf')
            
            # Log diagnostic summary if we're wall following
            if self.state == RobotState.WALL_FOLLOWING and self.lidar_received_count % 5 == 0:
                self.get_logger().info(f"🔍 WALL FOLLOWING - Front: {self.min_front_distance:.2f}m, " +
                                      f"Left: {self.min_left_distance:.2f}m, " +
                                      f"Right: {self.min_right_distance:.2f}m, " +
                                      f"Back: {self.min_back_distance:.2f}m")
            
            # If in wall following mode, the main loop handles this
            if self.state == RobotState.WALL_FOLLOWING:
                # Wall following logic will be called from the main loop
                pass
            else:
                # Update the grid with wall information
                self.update_walls()
            
        except IndexError as e:
            self.get_logger().error(f"❌ LIDAR INDEX ERROR: {e}. Scan length={len(msg.ranges)}")
            # Print the problematic indices
            self.get_logger().error(f"Front indices: 0-31 and {len(msg.ranges)-33}-{len(msg.ranges)-1}")
            self.get_logger().error(f"Left indices: 93-156, Right indices: 218-281, Back indices: 343-406")
        except Exception as e:
            self.get_logger().error(f"❌ LIDAR PROCESSING ERROR: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def wall_follow_logic(self):
        """Logic for wall following, adapted from example code"""
        # Debug info
        if self.debug:
            self.get_logger().info(f"Front: {self.min_front_distance:.2f} Left: {self.min_left_distance:.2f} "
                                   f"Back: {self.min_back_distance:.2f} Right: {self.min_right_distance:.2f}")
        
        # Check for wall in front or no wall to the left
        if self.wall_in_front_of_robot():
            self.get_logger().info("Wall detected in front, turning right")
            self.turn_right()
            time.sleep(0.5)  # Add a small delay after turning
        elif not self.wall_to_left_of_robot():
            self.get_logger().info("No wall on left, going around corner")
            self.go_around_wall()
        else:
            self.get_logger().info("Following wall, going straight")
            self.go_straight()
            time.sleep(0.5)  # Add a small delay after moving
    
    def wall_to_left_of_robot(self):
        """Check if there's a wall to the left"""
        return self.min_left_distance < 0.3
    
    def wall_in_front_of_robot(self):
        """Check if there's a wall in front"""
        return self.min_front_distance < 0.2
    
    def go_around_wall(self):
        """Navigate around a corner in the wall"""
        left_turn_end_time = time.time() + self.turn_90_duration
        while time.time() < left_turn_end_time:
            self.turn_left()
            
        # Drive forward for a bit
        WALL_BUFFER_TIME = 1.0
        drive_straight_end_time = time.time() + WALL_BUFFER_TIME
        while time.time() < drive_straight_end_time:
            self.go_straight()
            
        # Check if we found the wall to our left
        if self.wall_to_left_of_robot():
            return
        else:
            # Turn left again to follow the wall
            left_turn_end_time = time.time() + self.turn_90_duration
            while time.time() < left_turn_end_time:
                self.turn_left()
                
            # Drive forward again
            drive_straight_end_time = time.time() + WALL_BUFFER_TIME
            while time.time() < drive_straight_end_time:
                self.go_straight()
    
    def detect_room_color(self):
        """Detect the color of the current room"""
        if self.latest_frame is None:
            return
        
        hsv = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2HSV)
        
        # Define minimum area for a valid color region
        min_color_area = 5000
        
        # Detect each color
        detected_colors = []
        
        # Check red with both ranges (red wraps around in HSV)
        red_mask1 = cv2.inRange(hsv, np.array(self.color_ranges['red'][0]), np.array(self.color_ranges['red'][1]))
        red_mask2 = cv2.inRange(hsv, np.array(self.color_ranges['red2'][0]), np.array(self.color_ranges['red2'][1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if red_contours and cv2.contourArea(max(red_contours, key=cv2.contourArea)) > min_color_area:
            detected_colors.append(ColorRoom.RED)
        
        # Check other colors
        color_mapping = {
            'green': ColorRoom.GREEN,
            'blue': ColorRoom.BLUE,
            'yellow': ColorRoom.YELLOW,
            'purple': ColorRoom.PURPLE,
            'gray': ColorRoom.GRAY
        }
        
        for color_name, color_enum in color_mapping.items():
            lower, upper = self.color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > min_color_area:
                detected_colors.append(color_enum)
        
        # If we detected a color, update the grid
        if detected_colors:
            # If multiple colors are detected, choose the one with the largest area
            areas = []
            for color in detected_colors:
                if color == ColorRoom.RED:
                    mask = red_mask
                else:
                    for name, enum_val in color_mapping.items():
                        if enum_val == color:
                            lower, upper = self.color_ranges[name]
                            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                            break
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    area = cv2.contourArea(max(contours, key=cv2.contourArea))
                    areas.append((color, area))
            
            if areas:
                # Choose color with largest area
                color = max(areas, key=lambda x: x[1])[0]
                
                # Update the grid cell color
                x, y = self.current_position
                self.grid[x][y].color = color
                
                # Log the color detection
                color_name = color.name
                self.get_logger().info(f"Detected {color_name} room at position {self.current_position}")
                
                # Store room positions
                self.room_positions[color] = self.current_position
                
                # If we found the red room (start) or blue room (destination)
                if color == ColorRoom.RED:
                    self.red_room_position = self.current_position
                    self.get_logger().info(f"RED ROOM (START) FOUND at {self.current_position}")
                elif color == ColorRoom.BLUE:
                    self.blue_room_position = self.current_position
                    self.get_logger().info(f"BLUE ROOM (DESTINATION) FOUND at {self.current_position}")
                    # If we found the blue room, we can switch to tracking blue for precision
                    self.state = RobotState.TRACKING_COLOR
    
    def update_walls(self):
        """Update the wall information for the current cell based on LiDAR data"""
        # Get current grid cell
        x, y = self.current_position
        
        # Check walls based on current direction
        if self.current_direction == Direction.NORTH:
            if self.min_front_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.NORTH.value] = True
            else:
                self.grid[x][y].walls[Direction.NORTH.value] = False
                
            if self.min_left_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.WEST.value] = True
            else:
                self.grid[x][y].walls[Direction.WEST.value] = False
                
            if self.min_right_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.EAST.value] = True
            else: 
                self.grid[x][y].walls[Direction.EAST.value] = False
                
        elif self.current_direction == Direction.EAST:
            if self.min_front_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.EAST.value] = True
            else:
                self.grid[x][y].walls[Direction.EAST.value] = False
                
            if self.min_left_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.NORTH.value] = True
            else:
                self.grid[x][y].walls[Direction.NORTH.value] = False
                
            if self.min_right_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.SOUTH.value] = True
            else:
                self.grid[x][y].walls[Direction.SOUTH.value] = False
                
        elif self.current_direction == Direction.SOUTH:
            if self.min_front_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.SOUTH.value] = True
            else:
                self.grid[x][y].walls[Direction.SOUTH.value] = False
                
            if self.min_left_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.EAST.value] = True
            else:
                self.grid[x][y].walls[Direction.EAST.value] = False
                
            if self.min_right_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.WEST.value] = True
            else:
                self.grid[x][y].walls[Direction.WEST.value] = False
                
        elif self.current_direction == Direction.WEST:
            if self.min_front_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.WEST.value] = True
            else:
                self.grid[x][y].walls[Direction.WEST.value] = False
                
            if self.min_left_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.SOUTH.value] = True
            else:
                self.grid[x][y].walls[Direction.SOUTH.value] = False
                
            if self.min_right_distance < self.wall_threshold:
                self.grid[x][y].walls[Direction.NORTH.value] = True
            else:
                self.grid[x][y].walls[Direction.NORTH.value] = False
        
        # Mark cell as visited
        self.grid[x][y].visited = True
        
        # Debug logging
        if self.debug:
            self.get_logger().info(f"Current position: {self.current_position}, Direction: {self.current_direction.name}")
            self.get_logger().info(f"Front distance: {self.min_front_distance}, Left distance: {self.min_left_distance}, Right distance: {self.min_right_distance}")
            wall_info = f"Walls: N={self.grid[x][y].walls[Direction.NORTH.value]}, " \
                       f"E={self.grid[x][y].walls[Direction.EAST.value]}, " \
                       f"S={self.grid[x][y].walls[Direction.SOUTH.value]}, " \
                       f"W={self.grid[x][y].walls[Direction.WEST.value]}" 
            self.get_logger().info(wall_info)
    
    def move_robot(self, linear_x, linear_y, angular_z, duration):
        """Move the robot with specified velocities for a given duration using cmd_vel"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)
        
        # Stop the robot
        self.stop_robot()
    
    def send_motor_command(self, motor_speeds):
        """Send motor speed commands to the mecanum wheels."""
        msg = MotorsState()
        msg.data = [
            MotorState(id=1, rps=motor_speeds[0]),  # top left
            MotorState(id=2, rps=motor_speeds[1]),  # bottom left
            MotorState(id=3, rps=motor_speeds[2]),  # top right
            MotorState(id=4, rps=motor_speeds[3]),  # bottom right
        ]
        self.motor_pub.publish(msg)
        if self.debug:
            self.get_logger().info(f"Moving: {motor_speeds}")
    
    def stop_robot(self):
        """Stop the robot's movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)
        # Also stop motors
        self.send_motor_command([0.0, 0.0, 0.0, 0.0])
    
    def turn_left(self):
        """Turn the robot 90 degrees to the left"""
        self.state = RobotState.TURNING
        self.get_logger().info("Starting left turn")
        
        # Using motor commands like in example and ensuring the motors run for sufficient time
        self.send_motor_command([1.0, 1.0, 1.0, 1.0])
        time.sleep(self.turn_90_duration)  # Wait for the full duration of the turn
        self.stop_robot()  # Explicitly stop after turning
        
        # Update direction after turning (for grid tracking)
        self.current_direction = Direction((self.current_direction.value - 1) % 4)
        self.get_logger().info(f"Completed turn left, now facing {self.current_direction.name}")
    
    def turn_right(self):
        """Turn the robot 90 degrees to the right"""
        self.state = RobotState.TURNING
        self.get_logger().info("Starting right turn")
        
        # Using motor commands like in example and ensuring the motors run for sufficient time
        self.send_motor_command([-1.0, -1.0, -1.0, -1.0])
        time.sleep(self.turn_90_duration)  # Wait for the full duration of the turn
        self.stop_robot()  # Explicitly stop after turning
        
        # Update direction after turning (for grid tracking)
        self.current_direction = Direction((self.current_direction.value + 1) % 4)
        self.get_logger().info(f"Completed turn right, now facing {self.current_direction.name}")
    
    def turn_around(self):
        """Turn the robot 180 degrees"""
        self.state = RobotState.TURNING
        # Using motor commands for twice as long
        self.send_motor_command([1.0, 1.0, 1.0, 1.0])
        time.sleep(self.turn_90_duration * 2)
        self.stop_robot()
        
        # Update direction after turning
        self.current_direction = Direction((self.current_direction.value + 2) % 4)
        self.get_logger().info(f"Turned around, now facing {self.current_direction.name}")
    
    def go_straight(self):
        """Move the robot forward using motor commands"""
        self.get_logger().info("Moving forward")
        
        # Send command to move forward (all wheels moving forward)
        # Top-left and bottom-left wheels spin negative, top-right and bottom-right positive
        # This creates forward motion with mecanum wheels
        self.send_motor_command([-0.5, -0.5, 0.5, 0.5])  # Using slower speed for safety
        
        # Wait for movement to complete
        time.sleep(self.move_cell_duration)
        
        # Stop the robot
        self.stop_robot()
        
        # Update position based on current direction
        x, y = self.current_position
        if self.current_direction == Direction.NORTH:
            self.current_position = (x, y - 1)
        elif self.current_direction == Direction.EAST:
            self.current_position = (x + 1, y)
        elif self.current_direction == Direction.SOUTH:
            self.current_position = (x, y + 1)
        elif self.current_direction == Direction.WEST:
            self.current_position = (x - 1, y)
        
        self.get_logger().info(f"Completed forward movement to {self.current_position}")
    
    def move_forward(self):
        """Move the robot forward one cell using timed movement"""
        self.send_motor_command([-1.0, -1.0, 1.0, 1.0])
        time.sleep(self.move_cell_duration)
        self.stop_robot()
        
        # Update position after moving
        x, y = self.current_position
        if self.current_direction == Direction.NORTH:
            self.current_position = (x, y - 1)
        elif self.current_direction == Direction.EAST:
            self.current_position = (x + 1, y)
        elif self.current_direction == Direction.SOUTH:
            self.current_position = (x, y + 1)
        elif self.current_direction == Direction.WEST:
            self.current_position = (x - 1, y)
        
        self.get_logger().info(f"Moved forward to {self.current_position}")
    
    def get_move_direction(self, from_pos, to_pos):
        """Calculate which direction to face to move from from_pos to to_pos"""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        if x2 > x1:
            return Direction.EAST
        elif x2 < x1:
            return Direction.WEST
        elif y2 > y1:
            return Direction.SOUTH
        elif y2 < y1:
            return Direction.NORTH
        else:
            return None
    
    def get_available_moves(self):
        """Get available moves from current position"""
        x, y = self.current_position
        moves = []
        
        # Check each direction
        if not self.grid[x][y].walls[Direction.NORTH.value]:
            moves.append((x, y - 1))
        if not self.grid[x][y].walls[Direction.EAST.value]:
            moves.append((x + 1, y))
        if not self.grid[x][y].walls[Direction.SOUTH.value]:
            moves.append((x, y + 1))
        if not self.grid[x][y].walls[Direction.WEST.value]:
            moves.append((x - 1, y))
        
        return moves
    
    def get_unvisited_moves(self):
        """Get unvisited cells that are accessible from current position"""
        moves = self.get_available_moves()
        return [(x, y) for x, y in moves if not self.grid[x][y].visited]
    
    def explore_maze(self):
        """Use depth-first search to explore the maze"""
        self.get_logger().info("Starting maze exploration")
        
        # Mark start position as visited
        x, y = self.current_position
        self.grid[x][y].visited = True
        
        # Use a stack for DFS
        stack = [self.current_position]
        
        while stack and self.state == RobotState.EXPLORING:
            # Get unvisited neighbors
            unvisited = self.get_unvisited_moves()
            
            if unvisited:
                # Choose a random unvisited neighbor
                next_pos = unvisited[0]
                stack.append(self.current_position)
                
                # Calculate which direction to move
                target_direction = self.get_move_direction(self.current_position, next_pos)
                
                # Turn to face the target direction
                turns_needed = (target_direction.value - self.current_direction.value) % 4
                if turns_needed == 1:
                    self.turn_right()
                elif turns_needed == 2:
                    self.turn_around()
                elif turns_needed == 3:
                    self.turn_left()
                
                # Move to the next cell
                self.move_forward()
                
                # Update walls before continuing
                time.sleep(0.5)  # Wait for LiDAR to update
                
            elif stack:
                # Backtrack
                prev_pos = stack.pop()
                
                # Calculate which direction to move
                target_direction = self.get_move_direction(self.current_position, prev_pos)
                
                # Turn to face the target direction
                turns_needed = (target_direction.value - self.current_direction.value) % 4
                if turns_needed == 1:
                    self.turn_right()
                elif turns_needed == 2:
                    self.turn_around()
                elif turns_needed == 3:
                    self.turn_left()
                
                # Move to the previous cell
                self.move_forward()
                
                # Update walls before continuing
                time.sleep(0.5)  # Wait for LiDAR to update
        
        self.get_logger().info("Maze exploration completed")
    
    def wall_follow_exploration(self):
        """Explore the maze using wall following"""
        self.get_logger().info("Starting wall following exploration")
        self.state = RobotState.WALL_FOLLOWING
        
        # First, make sure we have recent LIDAR data
        time.sleep(1.0)  # Wait for sensors to update
        
        # Check if we need to find a wall first
        if self.min_left_distance > 0.5:  # No wall nearby on the left
            self.get_logger().info("Finding wall on left")
            # Turn left to try to find a wall
            self.turn_left()
            time.sleep(1.0)  # Wait for sensors to update
        
        # Begin exploration
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        cells_explored = 0
        
        # Main exploration loop
        while (self.state == RobotState.WALL_FOLLOWING and 
               time.sleep(0.1) or time.time() - start_time < timeout and
               cells_explored < 50):  # Limit to 50 cells
            
            # Check for walls around current position
            self.update_walls()
            
            # Mark current cell as visited
            x, y = self.current_position
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if not self.grid[x][y].visited:
                    self.grid[x][y].visited = True
                    cells_explored += 1
                    self.get_logger().info(f"Visiting cell {cells_explored} at {self.current_position}")
                    
                    # Check for colored rooms
                    self.detect_room_color()
            
            # Wall following decision logic
            # This is the critical part that needs to be fixed
            if self.min_front_distance < 0.25:  # Wall in front
                self.get_logger().info(f"Wall detected in front ({self.min_front_distance}m), turning right")
                self.turn_right()
            elif self.min_left_distance > 0.35:  # No wall on left
                self.get_logger().info(f"No wall on left ({self.min_left_distance}m), turning left")
                self.turn_left()
                # Move forward to get closer to the wall
                self.go_straight()
            else:  # Wall on left, no wall in front
                self.get_logger().info(f"Following wall on left ({self.min_left_distance}m), moving forward")
                self.go_straight()
            
            # Check if we found what we're looking for
            if self.red_room_position and self.blue_room_position:
                self.get_logger().info("Found both red and blue rooms! Ending exploration.")
                break
            
        self.get_logger().info(f"Wall following exploration completed. Explored {cells_explored} cells.")
        self.state = RobotState.EXPLORING  # Reset state
    
    def find_path(self, start, end):
        """Find a path from start to end using BFS"""
        self.get_logger().info(f"Finding path from {start} to {end}")
        
        # Reset distance and parent information
        for row in self.grid:
            for cell in row:
                cell.distance = float('inf')
                cell.parent = None
        
        # BFS
        queue = deque([start])
        x, y = start
        self.grid[x][y].distance = 0
        
        while queue:
            current = queue.popleft()
            x, y = current
            
            # If we reached the end, break
            if current == end:
                break
            
            # Check each direction
            for direction, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                if not self.grid[x][y].walls[direction]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.grid[nx][ny].distance == float('inf'):
                            self.grid[nx][ny].distance = self.grid[x][y].distance + 1
                            self.grid[nx][ny].parent = current
                            queue.append((nx, ny))
        
        # Build path from end to start
        if end != start and self.grid[end[0]][end[1]].parent is not None:
            path = []
            current = end
            while current != start:
                path.append(current)
                x, y = current
                current = self.grid[x][y].parent
            path.append(start)
            path.reverse()
            return path
        else:
            self.get_logger().warning(f"No path found from {start} to {end}")
            return [start]
    
    def follow_path(self, path):
        """Follow a path through the maze"""
        self.get_logger().info(f"Following path: {path}")
        self.state = RobotState.NAVIGATING
        
        for i in range(1, len(path)):
            current_pos = path[i-1]
            next_pos = path[i]
            
            # Calculate which direction to move
            target_direction = self.get_move_direction(current_pos, next_pos)
            
            # Turn to face the target direction
            turns_needed = (target_direction.value - self.current_direction.value) % 4
            if turns_needed == 1:
                self.turn_right()
                time.sleep(self.turn_90_duration)
            elif turns_needed == 2:
                self.turn_around()
            elif turns_needed == 3:
                self.turn_left()
                time.sleep(self.turn_90_duration)
            
            # Move to the next cell
            self.move_forward()
            
            # Update our current position
            self.current_position = next_pos
            
            time.sleep(0.5)  # Wait before next move
        
        self.get_logger().info("Path following completed")
        self.state = RobotState.REACHED_DESTINATION
    
    def visualize_map(self):
        """Visualize the maze map"""
        # Create a white background
        map_size = 400
        cell_size = map_size // self.grid_size
        map_image = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Calculate pixel coordinates
                px = x * cell_size
                py = y * cell_size
                
                # Draw cell
                if self.grid[x][y].visited:
                    # Color based on room color
                    cell_color = (255, 255, 255)  # Default white
                    if self.grid[x][y].color == ColorRoom.RED:
                        cell_color = (0, 0, 255)  # BGR format: Red
                    elif self.grid[x][y].color == ColorRoom.GREEN:
                        cell_color = (0, 255, 0)  # BGR format: Green
                    elif self.grid[x][y].color == ColorRoom.BLUE:
                        cell_color = (255, 0, 0)  # BGR format: Blue
                    elif self.grid[x][y].color == ColorRoom.YELLOW:
                        cell_color = (0, 255, 255)  # BGR format: Yellow
                    elif self.grid[x][y].color == ColorRoom.PURPLE:
                        cell_color = (255, 0, 255)  # BGR format: Purple
                    elif self.grid[x][y].color == ColorRoom.GRAY:
                        cell_color = (128, 128, 128)  # BGR format: Gray
                    
                    cv2.rectangle(map_image, (px, py), (px + cell_size, py + cell_size), cell_color, -1)
                
                # Draw walls
                if self.grid[x][y].walls[Direction.NORTH.value]:
                    cv2.line(map_image, (px, py), (px + cell_size, py), (0, 0, 0), 2)
                if self.grid[x][y].walls[Direction.EAST.value]:
                    cv2.line(map_image, (px + cell_size, py), (px + cell_size, py + cell_size), (0, 0, 0), 2)
                if self.grid[x][y].walls[Direction.SOUTH.value]:
                    cv2.line(map_image, (px, py + cell_size), (px + cell_size, py + cell_size), (0, 0, 0), 2)
                if self.grid[x][y].walls[Direction.WEST.value]:
                    cv2.line(map_image, (px, py), (px, py + cell_size), (0, 0, 0), 2)
        
        # Mark current position
        cx, cy = self.current_position
        px = cx * cell_size + cell_size // 2
        py = cy * cell_size + cell_size // 2
        cv2.circle(map_image, (px, py), cell_size // 4, (0, 0, 0), -1)
        
        # Draw an arrow to show direction
        arrow_length = cell_size // 3
        if self.current_direction == Direction.NORTH:
            cv2.line(map_image, (px, py), (px, py - arrow_length), (0, 0, 0), 2)
            cv2.line(map_image, (px, py - arrow_length), (px - arrow_length//2, py - arrow_length//2), (0, 0, 0), 2)
            cv2.line(map_image, (px, py - arrow_length), (px + arrow_length//2, py - arrow_length//2), (0, 0, 0), 2)
        elif self.current_direction == Direction.EAST:
            cv2.line(map_image, (px, py), (px + arrow_length, py), (0, 0, 0), 2)
            cv2.line(map_image, (px + arrow_length, py), (px + arrow_length//2, py - arrow_length//2), (0, 0, 0), 2)
            cv2.line(map_image, (px + arrow_length, py), (px + arrow_length//2, py + arrow_length//2), (0, 0, 0), 2)
        elif self.current_direction == Direction.SOUTH:
            cv2.line(map_image, (px, py), (px, py + arrow_length), (0, 0, 0), 2)
            cv2.line(map_image, (px, py + arrow_length), (px - arrow_length//2, py + arrow_length//2), (0, 0, 0), 2)
            cv2.line(map_image, (px, py + arrow_length), (px + arrow_length//2, py + arrow_length//2), (0, 0, 0), 2)
        elif self.current_direction == Direction.WEST:
            cv2.line(map_image, (px, py), (px - arrow_length, py), (0, 0, 0), 2)
            cv2.line(map_image, (px - arrow_length, py), (px - arrow_length//2, py - arrow_length//2), (0, 0, 0), 2)
            cv2.line(map_image, (px - arrow_length, py), (px - arrow_length//2, py + arrow_length//2), (0, 0, 0), 2)
        
        # Save the map image
        cv2.imwrite('/tmp/maze_map.png', map_image)
        self.get_logger().info("Map visualization saved to /tmp/maze_map.png")
        
        # Show the map
        cv2.imshow('Maze Map', map_image)
        cv2.waitKey(1)
        
        return map_image
    
    def complete_maze_run(self):
        """Complete maze run: explore, map, and navigate to destination"""
        self.get_logger().info("Starting maze run")
        
        # Phase 1: Exploration
        self.state = RobotState.EXPLORING
        self.get_logger().info("Starting exploration phase")
        
        # Choose exploration method - wall following is more robust
        self.wall_follow_exploration()
        
        # Visualize the map
        self.visualize_map()
        
        self.get_logger().info("Exploration finished")
        
        # Phase 2: Navigation to blue room if found
        if self.red_room_position and self.blue_room_position:
            self.get_logger().info(f"Found start (red) at {self.red_room_position} and destination (blue) at {self.blue_room_position}")
            
            # Find path from current position to the red room (start)
            self.get_logger().info("Navigating back to start (red room)")
            path_to_start = self.find_path(self.current_position, self.red_room_position)
            if path_to_start:
                self.follow_path(path_to_start)
            
            # Wait a moment at the start
            time.sleep(2.0)
            
            # Find path from red room to blue room
            self.get_logger().info("Navigating from start to destination (blue room)")
            path_to_destination = self.find_path(self.red_room_position, self.blue_room_position)
            if path_to_destination:
                self.follow_path(path_to_destination)
                self.get_logger().info("Destination reached!")
                return True
            else:
                self.get_logger().error("No path found to destination")
                return False
        else:
            self.get_logger().error("Could not find start or destination rooms")
            if self.red_room_position:
                self.get_logger().info(f"Found start (red) at {self.red_room_position}")
            if self.blue_room_position:
                self.get_logger().info(f"Found destination (blue) at {self.blue_room_position}")
            return False
    
    def check_lidar_health(self):
        """Monitor LIDAR health by checking if messages are being received"""
        current_time = time.time()
        time_since_last_scan = current_time - self.last_lidar_time
        
        if time_since_last_scan > 5.0:  # No LIDAR data for 5 seconds
            self.get_logger().error(f"‼️ NO LIDAR DATA RECEIVED in {time_since_last_scan:.1f} seconds! ‼️")
            self.get_logger().error("Please check LIDAR connection, power, and ROS topic")
        else:
            self.get_logger().info(f"LIDAR health: Received {self.lidar_received_count} messages, " + 
                                 f"last message {time_since_last_scan:.2f} seconds ago")
        
        # Print current sensor values
        self.get_logger().info(f"Current sensor values - Front: {self.min_front_distance:.2f}m, " +
                              f"Left: {self.min_left_distance:.2f}m, " +
                              f"Right: {self.min_right_distance:.2f}m")

    def verify_lidar_connection(self, timeout=10.0):
        """Verify LIDAR is connected and publishing before starting navigation"""
        self.get_logger().info("Verifying LIDAR connection...")
        
        # Reset the counter
        self.lidar_received_count = 0
        start_time = time.time()
        
        # Wait for LIDAR data
        while time.time() - start_time < timeout:
            if self.lidar_received_count > 0:
                self.get_logger().info(f"✅ LIDAR connection verified! Received {self.lidar_received_count} messages.")
                return True
            self.get_logger().info("Waiting for LIDAR data...")
            time.sleep(1.0)
        
        # If we get here, we didn't receive any LIDAR data within the timeout period
        self.get_logger().error("❌ LIDAR CONNECTION FAILED - No messages received within timeout!")
        self.get_logger().error("Please check:")
        self.get_logger().error(" - Is the LIDAR connected?")
        self.get_logger().error(" - Is the LIDAR powered?")
        self.get_logger().error(" - Is the ROS LIDAR node running?")
        self.get_logger().error(" - Is the topic name correct? ('/scan_raw')")
        
        # You could also run a ROS topic list command to help diagnose
        import subprocess
        try:
            self.get_logger().error("Available topics:")
            result = subprocess.run(["ros2", "topic", "list"], stdout=subprocess.PIPE, text=True)
            for line in result.stdout.splitlines():
                self.get_logger().error(f" - {line}")
        except Exception as e:
            self.get_logger().error(f"Could not list topics: {e}")
        
        return False

def main(args=None):
    rclpy.init(args=args)
    node = MazeSolver()
    
    # Run the maze solver in a separate thread
    def run_solver():
        # Wait for node to initialize
        time.sleep(2.0)
        
        # First, verify LIDAR connection
        if not node.verify_lidar_connection(timeout=15.0):
            node.get_logger().error("Cannot start maze solving without LIDAR - please fix connection issues")
            return
            
        # Test basic movements
        node.get_logger().info("Testing basic movements before maze solving...")
        
        # Test forward movement
        node.get_logger().info("Moving forward")
        node.go_straight()
        time.sleep(1.0)
        
        # Verify LIDAR readings after movement
        node.get_logger().info(f"After forward move - Front: {node.min_front_distance:.2f}m, " +
                             f"Left: {node.min_left_distance:.2f}m, " +
                             f"Right: {node.min_right_distance:.2f}m")
        
        # Now start the actual maze solving
        node.get_logger().info("Starting maze solving...")
        node.complete_maze_run()
    
    solver_thread = threading.Thread(target=run_solver)
    solver_thread.daemon = True
    solver_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()  # Make sure to stop the robot
        node.get_logger().info("Shutting down")
        rclpy.shutdown()

if __name__ == '__main__':
    main()