import math
import pygame
import os
import numpy as np
import logging

from players.player import Player
from players.aggressive_player import AggresivePlayer
from players.sticky_player import StickyPlayer
from players.deep_traffic_player import DeepTrafficPlayer
import torch
from config import (
    VISION_B,
    VISION_F,
    VISION_W,
    VISUALENABLED,
    EMERGENCY_BRAKE_MAX_SPEED_DIFF,
    ROAD_VIEW_OFFSET,
    VISUAL_VISION_B,
    VISUAL_VISION_F,
    VISUAL_VISION_W,
)


MAX_SPEED = 110  # km/h

DEFAULT_CAR_POS = 700

IMAGE_PATH = "./images"

if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, "red_car.png"))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, "white_car.png"))
    white_car = pygame.transform.scale(white_car, (34, 70))

direction_weight = {
    "L": 0.01,
    "M": 0.98,
    "R": 0.01,
}

move_weight = {"A": 0.30, "M": 0.50, "D": 0.20}


class Car:
    def __init__(
        self,
        surface,
        lane_map,
        speed=0,
        y=0,
        lane=4,
        is_subject=False,
        subject=None,
        score=None,
        agent=None,
    ):
        self.surface = surface
        self.lane_map = lane_map
        self.sprite = (
            None if not VISUALENABLED else red_car if is_subject else white_car
        )
        self.speed = min(max(speed, 0), MAX_SPEED)
        self.y = y
        self.lane = lane
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET
        self.is_subject = is_subject
        self.subject = subject
        self.max_speed = -1
        self.removed = False
        self.emergency_brake = None

        self.switching_lane = -1
        self.available_directions = ["M"]
        self.available_moves = ["D"]

        self.score = score

        # Subject car uses DeepTrafficPlayer and object car uses random choice of player
        self.player = (
            np.random.choice([Player(self), AggresivePlayer(self), StickyPlayer(self)])
            if not self.is_subject
            else DeepTrafficPlayer(self, agent=agent)
        )

        self.hard_brake_count = 0
        self.alternate_line_switching = 0

    def identify(self):
        """
            Updates the lane_map grid to track the car's current and target lane positions.

            - Calculates grid boxes (10 units tall) the car occupies based on vertical position (y).
            - Removes the car if it goes off screen (y < -200 or y > 1200).
            - Marks the car's presence in the current lane and, if switching, in the target lane.
            - Ensures grid updates are within bounds (0-99) for safe mapping.
            
            Returns:
                bool: False if the car is removed (out of bounds), True otherwise.
            """
        # Calculate the grid boxes that the car occupies
        # Each box is 10 units tall (self.y / 10.0)
        # min_box is the front of the car (subtract 1 to include full car length)
        min_box = int(math.floor(self.y / 10.0)) - 1
        # max_box is the rear of the car
        max_box = int(math.ceil(self.y / 10.0))

        # Remove car if it's too far off screen (either above or below)
        if self.y < -200 or self.y > 1200:
            self.removed = True # no longer in play
            return False

        # Register the car's presence in the front grid box
        if 0 <= min_box < 100:  # Check if within valid grid range (0-99)
            # Mark current lane
            self.lane_map[min_box][self.lane - 1] = self
            # If car is switching lanes, mark presence in target lane too
            if 1 <= self.switching_lane <= 7:
                self.lane_map[min_box][self.switching_lane - 1] = self

        # Register the car's presence in subsequent boxes (car length spans multiple boxes)
        # Iterate through 9 boxes to ensure full car length is covered
        for i in range(-1, 9):
            if 0 <= max_box + i < 100:  # Check if within valid grid range
                # Mark current lane
                self.lane_map[max_box + i][self.lane - 1] = self
                # If car is switching lanes, mark presence in target lane too
                if 1 <= self.switching_lane <= 7:
                    self.lane_map[max_box + i][self.switching_lane - 1] = self
        
        return True

    def accelerate(self):
        # If in front has car then cannot accelerate but follow
        self.speed += 1.0 if self.speed < MAX_SPEED else 0.0

    def decelerate(self):
        if self.max_speed > -1:
            self.speed = self.max_speed
        else:
            self.speed -= 1.0 if self.speed > 0 else 0.0


    def check_switch_lane(self):
        # If not currently switching lanes, return
        if self.switching_lane == -1:
            return
            
        # Update x position based on target lane
        # Move 50 pixels per lane (lane width)
        self.x += (self.switching_lane - self.lane) * 50
        
        # Check if lane switch is complete
        # ROAD_VIEW_OFFSET is the base x-coordinate of the leftmost lane
        if self.x == ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8:
            # Update to new lane and reset switching flag
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        moves = self.available_moves

        if action not in moves:
            action = moves[0]
            if self.subject is None:
                self.score.action_mismatch_penalty()

        if action == "A":
            self.accelerate()
        elif action == "D":
            self.decelerate()

        return action

    def switch_lane(self, direction):
        directions = self.available_directions
        if direction == "R":
            if "R" in directions:
                if self.lane < 7:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return "M"
        if direction == "L":
            if "L" in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return "M"
        return direction

    def identify_available_moves(self):
        # Reset max_speed limit (used for following cars ahead)
        self.max_speed = -1
        
        # Initialize possible moves and directions
        moves = ["M", "A", "D"]  # Maintain speed, Accelerate, Decelerate
        directions = ["M", "L", "R"]  # Maintain lane, Left, Right
        
        # If currently switching lanes, only allow maintaining current direction
        if self.switching_lane >= 0:
            directions = ["M"]
            
        # Remove invalid lane change options based on current position
        if self.lane == 1 and "L" in directions:  # Can't go left from leftmost lane
            directions.remove("L")
        if self.lane == 7 and "R" in directions:  # Can't go right from rightmost lane
            directions.remove("R")

        # Calculate the front box position of the car in the grid
        max_box = int(math.ceil(self.y / 10.0)) - 1

        # Check for cars in front (in current lane)
        for i in range(-1, 7):  # Check 7 boxes ahead
            if 0 <= max_box + i < 100:  # Ensure we're within grid bounds
                # If there's a car ahead (not empty and not self)
                if (self.lane_map[max_box + i][self.lane - 1] != 0 
                    and self.lane_map[max_box + i][self.lane - 1] != self):
                    car_in_front = self.lane_map[max_box + i][self.lane - 1]
                    # Can't accelerate when there's a car ahead
                    if "A" in moves:
                        moves.remove("A")
                    # If car ahead is slower, need to match their speed or slow down
                    if car_in_front.speed < self.speed:
                        if "M" in moves:
                            moves.remove("M")
                        # Calculate speed difference for emergency brake check
                        self.emergency_brake = self.speed - car_in_front.speed
                        # Match speed with car ahead
                        self.max_speed = car_in_front.speed
                    break

        # Check for cars in the target lane if switching lanes
        for i in range(-1, 7):  # Check 7 boxes ahead
            if 0 <= max_box + i < 100:  # Ensure we're within grid bounds
                if self.switching_lane > 0:  # If currently switching lanes
                    # If there's a car in the target lane ahead
                    if (self.lane_map[max_box + i][self.switching_lane - 1] != 0 
                        and self.lane_map[max_box + i][self.switching_lane - 1] != self):
                        # Can't accelerate while switching lanes with car ahead
                        if "A" in moves:
                            moves.remove("A")
                        car_in_front = self.lane_map[max_box + i][self.switching_lane - 1]
                        # If car ahead in target lane is slower
                        if car_in_front.speed < self.speed:
                            if "M" in moves:
                                moves.remove("M")
                            # Update max_speed to match slowest car ahead
                            self.max_speed = (car_in_front.speed 
                                            if self.max_speed == -1 
                                            or self.max_speed > car_in_front.speed 
                                            else self.max_speed)

        # Check if left lane change is possible
        if "L" in directions:
            for i in range(0, 9):  # Check 9 boxes (full car length)
                if 0 <= max_box + i < 100:
                    # If there's a car in the left lane
                    if self.lane_map[max_box + i][self.lane - 2] != 0:
                        directions.remove("L")
                        break

        # Check if right lane change is possible
        if "R" in directions:
            for i in range(0, 9):  # Check 9 boxes (full car length)
                if 0 <= max_box + i < 100:
                    # If there's a car in the right lane
                    if self.lane_map[max_box + i][self.lane] != 0:
                        directions.remove("R")
                        break

        # Update available moves and directions
        self.available_moves = moves
        self.available_directions = directions

        return moves, directions

    
    # Function not being used at all
    # def random(self):
    #     moves, directions = self.identify_available_moves()

    #     ds = np.random.choice(direction_weight.keys(), 3, p=direction_weight.values())
    #     ms = np.random.choice(move_weight.keys(), 3, p=move_weight.values())
    #     for d in ds:
    #         if d in directions:
    #             self.switch_lane(d)
    #             break

    #     for m in ms:
    #         if m in moves:
    #             self.move(m)
    #             break

    def relative_pos_subject(self):
        # If this is the subject car, handle emergency brake penalties
        if self.is_subject:
            if (self.emergency_brake is not None and 
                self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF):
                # Apply brake penalty if emergency brake threshold exceeded
                self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            return

        # Calculate relative position for non-subject cars
        # Convert speed difference from km/h to m/s
        dvdt = self.speed - self.subject.speed  # Speed difference in km/h
        dmds = dvdt / 3.6  # Convert to m/s (3.6 is conversion factor)
        # Position adjustment calculations
        dbdm = 1.0 / 0.25  # = 4 (Distance per meter)
        # This represents 4 pixels per meter in the game
        # Higher value = larger position changes
        # 1/0.25 means each meter is represented by 4 pixels on screen

        dsdf = 1.0 / 50.0  # = 0.02 (Scale factor for distance)
        # This is a scaling factor to convert speed differences to position changes
        # Smaller value = smoother movement
        # 1/50 means speed differences are scaled down by a factor of 50

        # Calculate final position adjustment
        dmdf = dmds * dsdf  # Scale the speed difference
        dbdf = dbdm * dmdf * 10.0  # Final pixels to move

        # Example calculation:
        # If speed difference is 10 km/h:
        # dmds = 10/3.6 ≈ 2.78 m/s
        # dmdf = 2.78 * 0.02 ≈ 0.056
        # dbdf = 4 * 0.056 * 10 ≈ 2.24 pixels per frame        
        # Update y position relative to subject car
        self.y = self.y - dbdf

        # Define safety zones
        LONGITUDINAL_DANGER = 50    # y-axis danger zone (pixels)
        LONGITUDINAL_SAFE = 150     # y-axis safe zone (pixels)
        LATERAL_DANGER = 25         # x-axis danger zone (pixels)
        OPTIMAL_FOLLOWING = 100     # optimal following distance

        # Calculate distances
        y_distance = abs(self.y - DEFAULT_CAR_POS)
        x_distance = abs(self.x - self.subject.x)
        relative_speed = abs(dvdt)  # km/h

        # Determine if the current position is safe
        def is_position_safe(x_dist, y_dist, rel_speed):
            # Check for dangerous conditions
            if y_dist < LONGITUDINAL_DANGER:
                return False  # Too close longitudinally
            
            if x_dist < LATERAL_DANGER and y_dist < LONGITUDINAL_SAFE:
                return False  # Too close laterally and within safety zone
            
            if rel_speed > 20 and y_dist < LONGITUDINAL_SAFE:
                return False  # Approaching too fast
                
            # Check for optimal conditions
            if (OPTIMAL_FOLLOWING - 20 <= y_dist <= OPTIMAL_FOLLOWING + 20 and 
                x_dist >= LATERAL_DANGER and 
                rel_speed < 10):
                return True  # Optimal following distance and safe lateral distance
            
            # Default to safe if no dangerous conditions detected
            return True

        # Apply score based on position safety
        if is_position_safe(x_distance, y_distance, relative_speed):
            self.score.add()      # Safe position
        else:
            self.score.subtract() # Unsafe position


        # Apply constant penalty per frame
        self.score.penalty()
        
    def decide(self, end_episode, cache=False, is_training=True):
        # If the car is a subject car (self.subject is None), it uses the decide_with_vision method to decide its next action.
        # It also checks if the result is a lane switch ('L' or 'R') and penalizes if there was recent lane switching.
        if self.subject is None:
            # Subject car uses DeepTrafficPlayer
            q_values, result = self.player.decide_with_vision(
                self.get_vision(), # vision of subject car
                self.score.score,
                end_episode,
                cache=cache,
                is_training=is_training,
            )
            if result == "L" or result == "R":
                # Check for "ping-pong" behavior - rapidly switching between lanes
                if (
                    # If turning Left (L) and previous actions included Right (4)
                    (result == "L" and 4 in self.player.agent.previous_actions) or
                    # OR if turning Right (R) and previous actions included Left (3)
                    (result == "R" and 3 in self.player.agent.previous_actions)
                ):
                    # Apply penalty for unnecessary lane switching
                    self.score.switching_lane_penalty()  # Defined in config.py as -0.00001
                    
                    # Increment counter tracking oscillating lane changes
                    self.alternate_line_switching += 1

            # Note: Action encoding in self.player.agent.previous_actions:
            # 3 = Left lane change
            # 4 = Right lane change
            return q_values, result
        else:
            # For Object cars, the Player class decide function is taken
            # Either player, AggressivePlayer or StickyPlayer is used randomly to decide
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        # First, update the car's position relative to the subject car
        self.relative_pos_subject()
        # Then, handle any ongoing lane changes
        self.check_switch_lane()
        # Finally, draw the car sprite if visual mode is enabled
        if VISUALENABLED:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    def get_vision(self):
        # Calculate horizontal (lane) boundaries for vision
        # VISION_W=1 means we look 1 lane to left and right of current lane
        min_x = min(max(0, self.lane - 1 - VISION_W), 6)  # Don't go below lane 0
        max_x = min(max(0, self.lane - 1 + VISION_W), 6)  # Don't go above lane 6
        # Store the raw input boundaries for padding later
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        # Calculate vertical (forward/backward) boundaries for vision
        # Convert car's y position to grid coordinates (10 units per grid)
        current_grid_y = int(math.floor(self.y / 10.0))
        # VISION_F=21 boxes ahead, VISION_B=14 boxes behind
        input_min_y = current_grid_y - VISION_F  # Look forward
        input_max_y = current_grid_y + VISION_B  # Look backward
        # Ensure vision stays within grid bounds (0-100)
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        # Create a set of tuples containing positions of all cars in vision range
        # Each tuple contains (lane_index, grid_y_position)
        cars_in_vision = set(
            [
                (
                    self.lane_map[y][x].lane - 1,  # Convert to 0-based lane index
                    int(math.floor(self.lane_map[y][x].y / 10.0)),  # Grid y-position
                )
                for y in range(min_y, max_y + 1)
                for x in range(min_x, max_x + 1)
                if self.lane_map[y][x] != 0  # Only include positions with cars
            ]
        )
        # Example output:
        #  {(5, 66), (4, 70), (5, 81), (3, 84), (4, 79), (3, 54), (3, 67), (5, 44)}

        # Initialize empty vision grid (100 vertical positions × 7 lanes)
        vision = np.zeros((100, 7), dtype=int)
        
        # Mark car positions in vision grid
        # Each car occupies 7 vertical grid cells (car length)
        for car in cars_in_vision:
            for y in range(7):
                vision[car[1] + y][car[0]] = 1  # Mark car presence with 1

        # Crop vision to relevant area only
        vision = vision[min_y : max_y + 1, min_x : max_x + 1]

        # Add padding to maintain consistent vision size
        # This ensures the neural network always gets same-sized input
        vision = np.pad(
            vision,
            (
                # Vertical padding (forward/backward)
                (min_y - input_min_y, input_max_y - max_y),
                # Horizontal padding (left/right lanes)
                (min_x - input_min_xx, input_max_xx - max_x),
            ),
            "constant",
            constant_values=(-1),  # Use -1 for padding (distinguishable from cars and empty space)
        )

        # Reshape vision for neural network input
        # Shape: [Forward+Backward+1, Width*2+1, 1]
        # Example: [36, 3, 1] for VISION_F=21, VISION_B=14, VISION_W=1
        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    # Function not being used at all (used in advanced view)
    def get_subjective_vision(self):
        min_x = min(max(0, self.lane - 1 - VISUAL_VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISUAL_VISION_W), 6)
        input_min_xx = self.lane - 1 - VISUAL_VISION_W
        input_max_xx = self.lane - 1 + VISUAL_VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISUAL_VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISUAL_VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars = [
            (self.lane_map[y][x].lane, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None
        ]

        return cars
