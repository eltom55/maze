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

class RobotState(Enum):
    EXPLORING = 1
    NAVIGATING = 2
    TURNING = 3
    REACHED_DESTINATION = 4

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
        
        # ROS Subscribers
        self.create_subscription(Image, 'ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 1)
        
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
        
        # Robot parameters
        self.turn_90_duration = 1.8  # Time to turn 90 degrees
        self.move_cell_duration = 1.5  # Time to move one cell forward
        self.wall_threshold = 0.4  # Distance threshold to detect walls
        
        # Initialize log
        self.get_logger().info('Maze Solver initialized')
        
        # Map visualization
        self.map_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
    def image_callback(self, msg):
        """Process camera image to detect colors"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_frame = cv_image
            
            # Only process colors if we're in the exploring state
            if self.state == RobotState.EXPLORING:
                self.detect_room_color()
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def lidar_callback(self, msg):
        """Process LiDAR data for wall detection"""
        try:
            # Store the full scan
            self.laser_ranges = msg.ranges
            
            # Extract front, left and right distances
            num_readings = len(msg.ranges)
            front_indices = list(range(-10, 11))  # Front view spans -10° to 10°
            left_indices = list(range(80, 101))   # Left view spans 80° to 100°
            right_indices = list(range(-101, -80))  # Right view spans -100° to -80°
            
            # Wrap negative indices for proper array access
            front_indices = [i % num_readings for i in front_indices]
            left_indices = [i % num_readings for i in left_indices]
            right_indices = [i % num_readings for i in right_indices]
            
            # Get minimum valid readings
            front_readings = [r for i, r in enumerate(msg.ranges) if i in front_indices and r != float('inf') and not math.isnan(r)]
            left_readings = [r for i, r in enumerate(msg.ranges) if i in left_indices and r != float('inf') and not math.isnan(r)]
            right_readings = [r for i, r in enumerate(msg.ranges) if i in right_indices and r != float('inf') and not math.isnan(r)]
            
            # Update distances
            self.min_front_distance = min(front_readings) if front_readings else float('inf')
            self.min_left_distance = min(left_readings) if left_readings else float('inf')
            self.min_right_distance = min(right_readings) if right_readings else float('inf')
            
            # Update grid walls
            self.update_walls()
            
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")
    
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
        
        # If we detect a wall where there should be a passage, update the map
        # Detect connections between cells
        if self.debug:
            self.get_logger().info(f"Current position: {self.current_position}, Direction: {self.current_direction.name}")
            self.get_logger().info(f"Front distance: {self.min_front_distance}, Left distance: {self.min_left_distance}, Right distance: {self.min_right_distance}")
            wall_info = f"Walls: N={self.grid[x][y].walls[Direction.NORTH.value]}, " \
                       f"E={self.grid[x][y].walls[Direction.EAST.value]}, " \
                       f"S={self.grid[x][y].walls[Direction.SOUTH.value]}, " \
                       f"W={self.grid[x][y].walls[Direction.WEST.value]}" 
            self.get_logger().info(wall_info)
        
    def move_robot(self, linear_x, linear_y, angular_z, duration):
        """Move the robot with specified velocities for a given duration"""
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
    
    def stop_robot(self):
        """Stop the robot's movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)
    
    def turn_right(self):
        """Turn the robot 90 degrees to the right"""
        self.state = RobotState.TURNING
        self.move_robot(0.0, 0.0, -1.5, self.turn_90_duration)
        
        # Update direction after turning
        self.current_direction = Direction((self.current_direction.value + 1) % 4)
        self.get_logger().info(f"Turned right, now facing {self.current_direction.name}")
    
    def turn_left(self):
        """Turn the robot 90 degrees to the left"""
        self.state = RobotState.TURNING
        self.move_robot(0.0, 0.0, 1.5, self.turn_90_duration)
        
        # Update direction after turning
        self.current_direction = Direction((self.current_direction.value - 1) % 4)
        self.get_logger().info(f"Turned left, now facing {self.current_direction.name}")
    
    def turn_around(self):
        """Turn the robot 180 degrees"""
        self.state = RobotState.TURNING
        self.move_robot(0.0, 0.0, 1.5, self.turn_90_duration * 2)
        
        # Update direction after turning
        self.current_direction = Direction((self.current_direction.value + 2) % 4)
        self.get_logger().info(f"Turned around, now facing {self.current_direction.name}")
    
    def move_forward(self):
        """Move the robot forward one cell"""
        self.move_robot(0.3, 0.0, 0.0, self.move_cell_duration)
        
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
    
    def find_path(self, start, end):
        """Find the shortest path from start to end using BFS"""
        # Initialize all cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i][j].distance = float('inf')
                self.grid[i][j].parent = None
        
        # Set start distance to 0
        x, y = start
        self.grid[x][y].distance = 0
        
        # Use a queue for BFS
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            x, y = current
            
            # Check if we've reached the end
            if current == end:
                break
            
            # Get available moves
            moves = []
            if not self.grid[x][y].walls[Direction.NORTH.value]:
                moves.append((x, y - 1))
            if not self.grid[x][y].walls[Direction.EAST.value]:
                moves.append((x + 1, y))
            if not self.grid[x][y].walls[Direction.SOUTH.value]:
                moves.append((x, y + 1))
            if not self.grid[x][y].walls[Direction.WEST.value]:
                moves.append((x - 1, y))
            
            # Process each available move
            for next_pos in moves:
                nx, ny = next_pos
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx][ny].distance == float('inf'):
                        self.grid[nx][ny].distance = self.grid[x][y].distance + 1
                        self.grid[nx][ny].parent = current
                        queue.append(next_pos)
        
        # Reconstruct path
        path = []
        current = end
        while current != start:
            path.append(current)
            x, y = current
            current = self.grid[x][y].parent
            if current is None:
                self.get_logger().error(f"No path found from {start} to {end}")
                return []
        
        path.append(start)
        path.reverse()
        
        return path
    
    def navigate_to_position(self, target_position):
        """Navigate to a specific position using the internal map"""
        self.get_logger().info(f"Navigating from {self.current_position} to {target_position}")
        
        # Find path
        path = self.find_path(self.current_position, target_position)
        
        if not path:
            self.get_logger().error(f"No path found to {target_position}")
            return False
        
        # Follow path
        for i in range(1, len(path)):
            next_pos = path[i]
            
            # Calculate direction to face
            target_direction = self.get_move_direction(self.current_position, next_pos)
            
            # Turn to face the target direction
            turns_needed = (target_direction.value - self.current_direction.value) % 4
            if turns_needed == 1:
                self.turn_right()
            elif turns_needed == 2:
                self.turn_around()
            elif turns_needed == 3:
                self.turn_left()
            
            # Move to the next position
            self.move_forward()
            
            # Wait for perception to update
            time.sleep(0.5)
        
        self.get_logger().info(f"Navigation complete. Current position: {self.current_position}")
        return True
    
    def visualize_map(self):
        """Create a visual representation of the map"""
        # Create a blank image
        cell_size = 20
        map_width = self.grid_size * cell_size
        map_height = self.grid_size * cell_size
        map_image = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw cells and walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i][j]
                cell_x = i * cell_size
                cell_y = j * cell_size
                
                # Draw cell background based on color
                if cell.visited:
                    cv2.rectangle(map_image, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (240, 240, 240), -1)
                
                # Draw room colors
                if cell.color != ColorRoom.UNKNOWN:
                    color_dict = {
                        ColorRoom.RED: (0, 0, 255),  # BGR format
                        ColorRoom.GREEN: (0, 255, 0),
                        ColorRoom.BLUE: (255, 0, 0),
                        ColorRoom.YELLOW: (0, 255, 255),
                        ColorRoom.PURPLE: (255, 0, 255),
                        ColorRoom.GRAY: (128, 128, 128)
                    }
                    cv2.rectangle(map_image, (cell_x + 2, cell_y + 2), (cell_x + cell_size - 2, cell_y + cell_size - 2), 
                                 color_dict[cell.color], -1)
                
                # Draw walls
                if cell.walls[Direction.NORTH.value]:
                    cv2.line(map_image, (cell_x, cell_y), (cell_x + cell_size, cell_y), (0, 0, 0), 2)
                if cell.walls[Direction.EAST.value]:
                    cv2.line(map_image, (cell_x + cell_size, cell_y), (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), 2)
                if cell.walls[Direction.SOUTH.value]:
                    cv2.line(map_image, (cell_x, cell_y + cell_size), (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), 2)
                if cell.walls[Direction.WEST.value]:
                    cv2.line(map_image, (cell_x, cell_y), (cell_x, cell_y + cell_size), (0, 0, 0), 2)
        
        # Mark current position
        current_x, current_y = self.current_position
        cv2.circle(map_image, (current_x * cell_size + cell_size // 2, current_y * cell_size + cell_size // 2), 
                  cell_size // 3, (255, 0, 0), -1)
        
        # Mark direction with a triangle
        direction_offset = {
            Direction.NORTH: (0, -cell_size // 4),
            Direction.EAST: (cell_size // 4, 0),
            Direction.SOUTH: (0, cell_size // 4),
            Direction.WEST: (-cell_size // 4, 0)
        }
        
        dx, dy = direction_offset[self.current_direction]
        center_x = current_x * cell_size + cell_size // 2
        center_y = current_y * cell_size + cell_size // 2
        tip_x = center_x + dx
        tip_y = center_y + dy
        
        if self.current_direction == Direction.NORTH:
            triangle_pts = np.array([[center_x - cell_size // 6, center_y], 
                                     [tip_x, tip_y], 
                                     [center_x + cell_size // 6, center_y]])
        elif self.current_direction == Direction.EAST:
            triangle_pts = np.array([[center_x, center_y - cell_size // 6], 
                                     [tip_x, tip_y], 
                                     [center_x, center_y + cell_size // 6]])
        elif self.current_direction == Direction.SOUTH:
            triangle_pts = np.array([[center_x - cell_size // 6, center_y], 
                                     [tip_x, tip_y], 
                                     [center_x + cell_size // 6, center_y]])
        else:  # WEST
            triangle_pts = np.array([[center_x, center_y - cell_size // 6], 
                                     [tip_x, tip_y], 
                                     [center_x, center_y + cell_size // 6]])
            
        cv2.fillPoly(map_image, [triangle_pts], (0, 0, 255))
        
        # Display the map
        cv2.imshow("Maze Map", map_image)
        cv2.waitKey(1)
        
        return map_image
    
    def run(self):
        """Main function to run the maze solver"""
        # Wait for the first frame and laser scan
        self.get_logger().info("Waiting for camera and LiDAR data...")
        while self.latest_frame is None or len(self.laser_ranges) == 0:
            time.sleep(0.1)
    
        self.get_logger().info("Sensors initialized, starting maze exploration")
        
        # Start in exploration mode
        self.state = RobotState.EXPLORING
        
        # Main loop
        try:
            # First phase: explore the maze
            self.get_logger().info("Starting maze exploration phase")
            self.explore_maze()
            
            # After exploration, visualize the final map
            map_image = self.visualize_map()
            cv2.imwrite("maze_map.png", map_image)
            self.get_logger().info("Map saved to maze_map.png")
            
            # Check if we found both red and blue rooms
            if self.red_room_position and self.blue_room_position:
                self.get_logger().info(f"Red room found at {self.red_room_position}")
                self.get_logger().info(f"Blue room found at {self.blue_room_position}")
                
                # Navigate to the blue room
                self.get_logger().info("Starting navigation to the blue room")
                self.state = RobotState.NAVIGATING
                success = self.navigate_to_position(self.blue_room_position)
                
                if success:
                    self.state = RobotState.REACHED_DESTINATION
                    self.get_logger().info("Successfully reached the blue room (destination)!")
                else:
                    self.get_logger().error("Failed to reach the blue room")
            else:
                if not self.red_room_position:
                    self.get_logger().warn("Red room (start) was not found during exploration")
                if not self.blue_room_position:
                    self.get_logger().warn("Blue room (destination) was not found during exploration")
        
        except Exception as e:
            self.get_logger().error(f"Error in maze solver: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        finally:
            # Make sure the robot stops
            self.stop_robot()
            self.get_logger().info("Maze solver finished")
    

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    maze_solver = MazeSolver()
    
    # Run in a separate thread to keep rclpy spinning
    thread = threading.Thread(target=maze_solver.run)
    thread.daemon = True
    thread.start()
    
    try:
        rclpy.spin(maze_solver)
    except KeyboardInterrupt:
        pass
    finally:
        maze_solver.stop_robot()
        maze_solver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
