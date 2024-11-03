import math
import pygame
import os
import numpy as np

from players.player import Player
from players.aggressive_player import AggresivePlayer
from players.sticky_player import StickyPlayer
from players.deep_traffic_player import DeepTrafficPlayer

from config import VISION_B, VISION_F, VISION_W, \
    VISUALENABLED, EMERGENCY_BRAKE_MAX_SPEED_DIFF, ROAD_VIEW_OFFSET, \
    VISUAL_VISION_B, VISUAL_VISION_F, VISUAL_VISION_W


MAX_SPEED = 110  # km/h

DEFAULT_CAR_POS = 700

IMAGE_PATH = './images'

if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, 'red_car.png'))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, 'white_car.png'))
    white_car = pygame.transform.scale(white_car, (34, 70))

direction_weight = {
    'L': 0.01,
    'M': 0.98,
    'R': 0.01,
}

move_weight = {
    'A': 0.30,
    'M': 0.50,
    'D': 0.20
}


class Car():
    def __init__(self, surface, lane_map, speed=0, y=0, lane=4, is_subject=False, subject=None, score=None, agent=None):
        self.surface = surface      # Pygame surface where the car will be drawn.
        self.lane_map = lane_map    # Reference to the lane map, which helps check other cars’ positions and handle lane changes.
        self.sprite = None if not VISUALENABLED else red_car if is_subject else white_car       # The visual representation of the car (either a red or white car image).
        self.speed = min(max(speed, 0), MAX_SPEED)      # Sets the car’s speed within a defined range from 0 to MAX_SPEED (110 km/h).
        self.y = y      # Vertical position, initially set by the y parameter.
        self.lane = lane        # Lane index where the car starts.
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET       # Horizontal position calculated based on the lane, with offset adjustments (fixed values like 50, 15, and 8) to position the car correctly on the road.
        self.is_subject = is_subject
        self.subject = subject      # Stores a reference to the subject car. If this car is the subject, self.subject is None.
        self.max_speed = -1         # Tracks the maximum speed the car can reach, dynamically set based on road conditions.
        self.removed = False        # Marks whether the car is removed from the scene (e.g., if it goes out of bounds).
        self.emergency_brake = None # Holds a value for braking in emergency scenarios.s

        self.switching_lane = -1    # Tracks the lane the car intends to switch to. -1 means no switching.
        self.available_directions = ['M']       # Default options for lane directions and speed moves, starting with maintain direction (‘M’) and decelerate (‘D’).
        self.available_moves = ['D']

        self.score = score          # Holds a reference to the score-tracking object, allowing the car to update its score based on actions or penalties.

        self.player = np.random.choice([
                Player(self),
                AggresivePlayer(self),
                StickyPlayer(self)
            ]) if not self.is_subject else DeepTrafficPlayer(self, agent=agent)

        self.hard_brake_count = 0           # Tracks the number of hard braking events (used for penalty or scoring).
        self.alternate_line_switching = 0   # Counts instances of alternate lane switching, useful for scoring or penalizing.

    def identify(self):
        # Determines whether the car is within the game boundaries.
        # Updates the lane map to reflect the car's current position and lane.
        # Handles lane switching if applicable.
        min_box = int(math.floor(self.y / 10.0)) - 1    # represent vertical positions (in units of 10) on the lane map where the car occupies space.
        max_box = int(math.ceil(self.y / 10.0))         # ""

        # Out of bound
        if self.y < -200 or self.y > 1200:
            self.removed = True # no longer in play
            return False

        # updates the lane_map at the position corresponding to the car's current lane (self.lane - 1)
        # with the car instance (self). If the car is switching lanes (self.switching_lane is between 1 and 7),
        # it also updates the lane map at the switching lane position.
        if 0 <= min_box < 100:
            self.lane_map[min_box][self.lane - 1] = self        # The lane map is a 2D array where each row represents a vertical section, and each column represents a lane.
            if 1 <= self.switching_lane <= 7:                   # If the car is in the process of switching lanes (self.switching_lane between 1 and 7), the lane map is updated at both the current and target lanes.
                self.lane_map[min_box][self.switching_lane - 1] = self
        # Extend Lane Map Updates Along the Car’s Length
        for i in range(-1, 9):                                  
            if 0 <= max_box + i < 100:
                self.lane_map[max_box + i][self.lane - 1] = self
                if 1 <= self.switching_lane <= 7:
                    self.lane_map[max_box + i][self.switching_lane - 1] = self
        return True         # If the car is within bounds, True is returned, indicating that the car is active in the game.

    def accelerate(self):
        # If in front has car then cannot accelerate but follow(this is not followed here)
        self.speed += 1.0 if self.speed < MAX_SPEED else 0.0

    # not convinced by logic
    def decelerate(self):
        if self.max_speed > -1:
            self.speed = self.max_speed
        else:
            self.speed -= 1.0 if self.speed > 0 else 0.0

    def check_switch_lane(self):
        # self.switching_lane - this attribute holds the target lane if the car is switching lanes.
        if self.switching_lane == -1:
            return
        self.x += (self.switching_lane - self.lane) * 50        # Moves the car horizontally by adjusting self.x.
        if self.x == ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8:    # calculates the exact horizontal position (self.x) for the car in the target lane.
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        """
        Validates the given action against the allowed moves.
        Defaults to a safe move if the provided action is not allowed, applying a penalty if it’s an agent-controlled car.
        Performs the action by calling the corresponding method (accelerate or decelerate) and returns the action taken.
        """
        moves = self.available_moves

        # If the car is agent-controlled and performs an invalid action, it triggers an "action mismatch penalty," 
        # deducting points or applying a penalty to discourage unauthorized moves.
        if action not in moves:
            action = moves[0]
            if self.subject is None:
                self.score.action_mismatch_penalty()        # 

        if action == 'A':
            self.accelerate()
        elif action == 'D':
            self.decelerate()

        return action

    def switch_lane(self, direction):
        """
        Checks if a lane switch is possible based on the requested direction and lane boundaries.
        Updates self.switching_lane to reflect the intended target lane, calling identify to update the lane map.
        If the switch isn’t possible, it applies a penalty for agent-controlled cars and defaults to maintaining the current lane ('M').
        Returns the final decision, ensuring the car can only perform safe, valid lane switches.
        """
        directions = self.available_directions
        if direction == 'R':
            if 'R' in directions:
                if self.lane < 7:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        if direction == 'L':
            if 'L' in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        return direction        # either L, R or M if the switch is not allowed

    def identify_available_moves(self):
        self.max_speed = -1
        moves = ['M', 'A', 'D']
        directions = ['M', 'L', 'R']
        if self.switching_lane >= 0:
            directions = ['M']
        # Left Lane Check: If the car is in the leftmost lane (self.lane == 1), it cannot move left, 
        # so 'L' is removed from directions.
        # Right Lane Check: If the car is in the rightmost lane (self.lane == 7), it cannot move right, 
        # so 'R' is removed from directions.
        if self.lane == 1 and 'L' in directions:
            directions.remove('L')
        if self.lane == 7 and 'R' in directions:
            directions.remove('R')

        # Determines the position in the lane map corresponding to the car’s position.
        max_box = int(math.ceil(self.y / 10.0)) - 1
        # Front checking(check for car in front in current lane)
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                # If a car (car_in_front) is detected directly ahead in the same lane:
                if self.lane_map[max_box + i][self.lane - 1] != 0 and self.lane_map[max_box + i][self.lane - 1] != self:
                    car_in_front = self.lane_map[max_box + i][self.lane - 1]
                    # 'A' (accelerate) is removed from moves, as accelerating would be unsafe
                    if 'A' in moves:
                        moves.remove('A')
                    # If the car ahead is slower than this car, 'M' (maintain speed) is also removed, 
                    # and self.emergency_brake is set to the difference in speed to indicate the need for braking.
                    # self.max_speed is updated to match the speed of car_in_front, which acts as a speed cap to avoid collisions.
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        self.emergency_brake = self.speed - car_in_front.speed
                        self.max_speed = car_in_front.speed
                    break
        # Consider car in target switching lane
        for i in range(-1, 7):
            # Similar to the front check but applies to the switching_lane, ensuring that the target lane is clear of obstacles.
            if 0 <= max_box + i < 100:
                if self.switching_lane > 0:
                    # If an obstacle is found in the target lane, 'A' and 'M' may be removed based on the relative speed.
                    if self.lane_map[max_box + i][self.switching_lane - 1] != 0 and self.lane_map[max_box + i][
                                self.switching_lane - 1] != self:
                        if 'A' in moves:
                            moves.remove('A')
                        car_in_front = self.lane_map[max_box + i][self.switching_lane - 1]
                        if car_in_front.speed < self.speed:
                            if 'M' in moves:
                                moves.remove('M')
                            # emergency_brake = self.speed - car_in_front.speed
                            #  # Update max_speed to match slowest car ahead---------din't understand
                            self.max_speed = car_in_front.speed \
                                if self.max_speed == -1 or self.max_speed > car_in_front.speed else self.max_speed

        # Left lane checking
        if 'L' in directions:
            # Left Lane: If moving left is allowed, it checks if the left lane (indexed as self.lane - 2) is clear for a certain distance (range(0, 9)).
            # If an obstacle is detected, 'L' is removed from directions, prohibiting the left lane switch.
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    # is the lane width 2 ?
                    if self.lane_map[max_box + i][self.lane - 2] != 0:
                        directions.remove('L')
                        break

        # Right lane checking
        if 'R' in directions:
            # Similarly, it checks if the right lane (indexed as self.lane) is clear.
            # If an obstacle is found, 'R' is removed from directions.
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    if self.lane_map[max_box + i][self.lane] != 0:
                        directions.remove('R')
                        break
        # self.available_moves and self.available_directions are updated with the filtered lists.
        # Returns moves and directions to indicate the actions and directions available based on the current road context.
        self.available_moves = moves
        self.available_directions = directions

        return moves, directions

    def random(self):
        moves, directions = self.identify_available_moves()

        ds = np.random.choice(direction_weight.keys(), 3, p=direction_weight.values())
        ms = np.random.choice(move_weight.keys(), 3, p=move_weight.values())
        # Checks if the chosen direction (d) is allowed (based on the identify_available_moves method).
        for d in ds:
            # If the direction is allowed, self.switch_lane(d) initiates the lane switch.
            # break: Exits the loop after the first valid switch, ensuring only one lane switch occurs.
            if d in directions:
                self.switch_lane(d)
                break
        # similar to direction selection, below is move selection
        # break: Stops the loop after performing the first valid move.
        for m in ms:
            if m in moves:
                self.move(m)
                break

    def relative_pos_subject(self):
        """
        Evaluates and adjusts the car’s position and score based on its relation to a designated “subject” car. 
        This method is used primarily for scoring adjustments, speed control, and emergency braking when the car is 
        following or interacting with the subject car.
        """
        if self.is_subject:
            # If self.emergency_brake has a value (indicating a need to brake) and it exceeds a predefined threshold (EMERGENCY_BRAKE_MAX_SPEED_DIFF), a braking penalty is applied:
            # self.score.brake_penalty(): Deducts points or penalizes the subject car for the emergency brake event.
            # self.hard_brake_count += 1: Increments the count of hard braking instances for tracking purposes.
            # self.emergency_brake = None: Resets the emergency braking status after checking.
            if self.emergency_brake is not None and self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF:
                self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            # Ends the method if self.is_subject is True, as no further calculations are necessary for the subject car itself.
            return
        # Converts dvdt from km/h to meters per second by dividing by 3.6.
        # dmds: Represents the car's relative movement in meters per second.
        dvdt = self.speed - self.subject.speed
        dmds = dvdt / 3.6
        # These conversions factor in adjustments for vertical position (self.y), representing distance changes based on speed.
        dbdm = 1.0 / 0.25
        dsdf = 1.0 / 50.0
        dmdf = dmds * dsdf
        # Final vertical position adjustment, based on the relative speed. The formula effectively translates the speed difference
        # into a distance adjustment for this car’s vertical position (self.y).
        dbdf = dbdm * dmdf * 10.0
        self.y = self.y - dbdf      # Updates self.y, moving the car either closer to or farther from the subject car depending on the value of dbdf
        #  If the car’s vertical position (self.y) falls within a specific range relative to DEFAULT_CAR_POS, a score deduction is applied.
        if DEFAULT_CAR_POS - dbdf <= self.y < DEFAULT_CAR_POS:
            self.score.subtract()
        # If the car’s position is slightly behind the subject car, a score bonus may be applied.
        elif DEFAULT_CAR_POS - dbdf > self.y >= DEFAULT_CAR_POS:
            self.score.add()
        self.score.penalty()            # why is this commented in Suryaansh branch ?

    def decide(self, end_episode, cache=False, is_training=True):
        # If the car is a subject car (self.subject is None), it uses the decide_with_vision method to decide its next action.
        # It also checks if the result is a lane switch ('L' or 'R') and penalizes if there was recent lane switching.
        if self.subject is None:
            q_values, result = self.player.decide_with_vision(self.get_vision(),
                                                  self.score.score,
                                                  end_episode,
                                                  cache=cache,
                                                  is_training=is_training)
            # Check for recent lane switching
            if result == 'L' or result == 'R':
                if (result == 'L' and 4 in self.player.agent.previous_actions) or \
                        (result == 'R' and 3 in self.player.agent.previous_actions):
                    self.score.switching_lane_penalty()
                    self.alternate_line_switching += 1
            return q_values, result
        else:
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        """
        handles the car’s visual update and position adjustments on the screen, ensuring that its position and lane-switching 
        status are correctly managed and displayed.
        """
        self.relative_pos_subject()
        self.check_switch_lane()
        if VISUALENABLED:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    def get_vision(self):
        """
        The get_vision method captures a "vision" or snapshot of the surrounding cars within a specified range relative to 
        the car’s position. This vision is represented as a grid, where each cell indicates the presence of a car, allowing 
        the car to "see" its immediate environment.
        """
        min_x = min(max(0, self.lane - 1 - VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISION_W), 6)
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars_in_vision = set([
            (self.lane_map[y][x].lane - 1, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0])

        vision = np.zeros((100, 7), dtype=int)
        for car in cars_in_vision:
            for y in range(7):
                vision[car[1] + y][car[0]] = 1

        # Crop vision from lane_map
        vision = vision[min_y: max_y + 1, min_x: max_x + 1]

        # Add padding if required
        vision = np.pad(vision,
                        ((min_y - input_min_y, input_max_y - max_y), (min_x - input_min_xx, input_max_xx - max_x)),
                        'constant',
                        constant_values=(-1))

        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    def get_subjective_vision(self):
        """
        Used when there are many subject cars
        """
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
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None]

        return cars


# The __init__ method in car.py sets up a Car instance with:

# Positioning and speed parameters,
# Visual settings based on configuration,
# Behavioral control type based on whether it’s the subject car or not,
# State and score tracking attributes to support the car’s movement, lane-switching, and other actions in the simulation.


# line 91-95
# This loop extends the lane map updates from max_box to cover the vertical space the car occupies. The range -1 to 9 ensures all sections of the car are reflected in the lane map.
# self.lane_map[max_box + i][self.lane - 1] = self marks the car’s presence in its current lane, and if switching lanes, it also updates the target lane.
# This additional mapping accounts for the car's size, making collision checks more accurate.