# Deep Traffic
import os
# Import required packages
import pygame
import sys
from pygame.locals import *
import numpy as np
import torch  # Add this import
import logging

# Import model and GUI related modules
from car import Car, DEFAULT_CAR_POS
from gui_util import draw_basic_road, \
    draw_road_overlay_safety, \
    draw_road_overlay_vision, \
    control_car, \
    identify_free_lane, \
    Score, \
    draw_inputs, \
    draw_actions, \
    draw_gauge, \
    draw_score
from deep_traffic_agent import DeepTrafficAgent

# Advanced view
from advanced_view.road import AdvancedRoad

import config
from logging_config import setup_logger

# Model name
model_name = config.MODEL_NAME

logger = setup_logger('GUI', ['logs/gui.log'])
deep_traffic_agent = DeepTrafficAgent(model_name)

# Define game constant
OPTIMAL_CARS_IN_SCENE = 15
ACTION_MAP = ['A', 'M', 'D', 'L', 'R']
# correspond to the keyboard keys
monitor_keys = [pygame.K_UP, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN]

if config.VISUALENABLED:
    pygame.init()       # Initializes all pygame modules
    pygame.font.init()  # Initializes font module enabling text rendering
    pygame.display.set_caption('DeepTraffic')   # Title of pygame window set to this name
    fpsClock = pygame.time.Clock()  # creates clock to manage frame rate of the game

    # sets up main display surface with resolution of 1600x800 pixels. flags use are:
    # pygame.DOUBLEBUF - Uses double buffering to help with smooth animations
    # pygame.HWSURFACE: Uses hardware acceleration if available. Using pygame.HWSURFACE is a way to leverage the GPU for better performance in rendering graphics. 
    main_surface = pygame.display.set_mode((700, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
    advanced_road = AdvancedRoad(main_surface, 0, 550, 1010, 800, lane=6)
else:
    os.environ["SDL_VIDEODRIVER"] = "dummy"     # Run without opening window(during testing for eg.)
    main_surface = None

lane_map = [[0 for x in range(7)] for y in range(100)]
episode_count = deep_traffic_agent.model.get_count_episodes()   # deep_traffic_agent-->cnn-->get_count_episodes-->self.count_episodes

speed_counter_avg = []
hard_brake_avg = []
alternate_line_switching = []

action_stats = np.zeros(5, np.int32)

PREDEFINED_MAX_CAR = config.MAX_SIMULATION_CAR  # 60


# New episode/game round
while episode_count < config.MAX_EPISODE + config.TESTING_EPISODE * 3:      # 2000 + 200*3
    is_training = config.DL_IS_TRAINING and episode_count < config.MAX_EPISODE and not config.VISUALENABLED     #False and ep_count< 2000 and not True

    # Score object
    score = Score(score=0)
    # All cars start with speed 60
    subject_car = Car(main_surface,
                      lane_map,
                      speed=60,
                      y=DEFAULT_CAR_POS,    # 700
                      lane=4,
                      is_subject=True,
                      score=score, # score initialized to 0
                      agent=deep_traffic_agent)
    object_cars = [Car(main_surface,
                       lane_map,
                       speed=60,
                       y=800,           # means wat?
                       lane=6,          # means ?
                       is_subject=False,
                       score=score, # score initialized to 0
                       subject=subject_car)
                   ]

    frame = 0

    game_ended = False

    delay_count = 0
    speed_counter = []
    subject_car_action = 'M'

    while True: # frame < config.MAX_FRAME_COUNT:
        if config.VISUALENABLED: # Show GUI, playing using DeepTrafficAgent
            pressed_key = pygame.key.get_pressed()
            keydown_key = []

            for event in pygame.event.get():
                if event.type == QUIT or event.type == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and not config.DLAGENTENABLED:
                    keydown_key.append(event.key)

        advanced_road.draw(frame, subject_car) # Moved here to render advanced view first and basic view on top of it
        

            # Setup game background
        draw_basic_road(main_surface, subject_car.speed)

        # Car to identify available moves in the order from top to bottom
        cars = [subject_car]
        cars.extend([o_car for o_car in object_cars if o_car.removed is False])     # filtering out any cars that have been marked for removal.
        # This line sorts the cars list based on the y attribute of each car (t_car.y).
        # The reverse=True argument sorts the cars in descending order, meaning cars with higher y values (presumably further down the screen) come first in the list.
        cars.sort(key=lambda t_car: t_car.y, reverse=True)
        available_lanes_for_new_car = identify_free_lane(cars)
        # Add more cars to the scene
        
        # is used to create a random 50/50 chance of adding a new car to the scene.
        # np.random.standard_normal(1)[0] generates a random number from a standard normal distribution, 
        # and the comparison with 0 determines if this number is greater than 0 (50% chance).
        if len(cars) < PREDEFINED_MAX_CAR and np.random.standard_normal(1)[0] >= 0: 
            # Decide position(Front or back)
            map_position = np.random.choice([0, 1], 1)[0] # 0 for front, 1 for back
            
            position = available_lanes_for_new_car[map_position]
            if len(position) > 0:
                # Back
                if map_position: # 1 for back
                    new_car_speed = np.random.randint(30, 91) # speed of new car between 30 and 90
                    new_car_y = 1010 # position of new car at the back of the road
                    new_car_lane = np.random.choice(position) # choose a random lane from the available lanes
                else: # 0 for front
                    new_car_speed = np.random.randint(30, 61) # speed of new car between 30 and 60
                    new_car_y = -100 # position of new car at the front of the road
                    new_car_lane = np.random.choice(position) # choose a random lane from the available lanes
                # Decide lanes
                new_car = Car(main_surface,
                              lane_map,
                              speed=new_car_speed,
                              y=new_car_y,
                              lane=new_car_lane,
                              is_subject=False,
                              subject=subject_car,
                              score=score) # create a new car object
                object_cars.append(new_car) # append to the end of the list
                if position: # 1 for back
                    cars.append(new_car) # append to the end of the list
                else: # 0 for front
                    cars.insert(0, new_car) # insert at the beginning of the list

        # main game logic
        # Reinitialize lane map
        
        # lane_map is a 2D array that keeps track of the number of cars in each lane and position on the road.
        for y in range(100):
            for x in range(7):
                lane_map[y][x] = 0

        
        # Identify car position
        for car in cars:
            car.identify() 

        # Identify available moves
        for car in cars:
            moves, directions = car.identify_available_moves()
        # output comes in the form of a list of moves and directions:
        #  Car:  1
        # Moves:  ['M', 'D'] Directions:  ['M', 'R']
        # Car:  2
        # Moves:  ['M', 'A', 'D'] Directions:  ['M', 'L']

        # Initialize cache flag for memory buffer management
        cache = False

        # Check if we should cache the current state-action pair
        if delay_count < config.DELAY and not game_ended and is_training:
            # DELAY is used to collect multiple frames before making a decision
            # This helps create a more stable learning environment by reducing
            # rapid fluctuations in state observations
            
            delay_count += 1  # Increment the delay counter
            # cache = True      # Set cache flag to accumulate experiences
        else:
            # Reset delay counter when:
            # 1. We've reached our desired delay count, or
            # 2. Game has ended, or
            # 3. We're not in training mode
            delay_count = 0

        # This caching mechanism serves several purposes:
        # 1. Frame Skip: Not every frame needs to be processed for learning
        # 2. State Stability: Allows the agent to observe multiple frames before deciding
        # 3. Memory Efficiency: Reduces memory usage by not storing every single frame


        q_values = None
        # Car react to road according to order
        i = 0
        for car in cars[::-1]:
            
            # Car.subject is None for subject cars
            if car.subject is not None:
                # Object car
                car.decide(game_ended, cache=cache, is_training=is_training)
                continue

            if config.DLAGENTENABLED: # Using DeepTrafficAgent
                # Get prediction from DeepTrafficAgent
                q_values, temp_action = car.decide(game_ended, cache=cache, is_training=is_training)
                # if q_values is not None and np.any(q_values != 0):
                #     logger.info("Q-values: {}, Result: {}, Score: {}".format(q_values, temp_action, score.score))
                if not cache:
                    subject_car_action = temp_action
                    q_values = q_values.sum().item()  # Convert PyTorch tensor to Python scalar
                    if not is_training:
                        action_stats[deep_traffic_agent.get_action_index(temp_action)] += 1
            elif config.VISUALENABLED:
                # Manual control
                is_controlled = False
                for key in monitor_keys:
                    if pressed_key[key] or key in keydown_key:
                        is_controlled = True
                        control_car(subject_car, key)
                if not is_controlled:
                    car.move('M')

        # Show road overlay (Safety)
        # draw_road_overlay_safety(main_surface, lane_map)
        draw_road_overlay_vision(main_surface, subject_car)

        for car in cars: # Draw all cars
            car.draw()

        # Decide end of game
        if game_ended:
            deep_traffic_agent.remember(score.score,
                                        subject_car.get_vision(),
                                        end_episode=True,
                                        is_training=is_training)
            break
        elif frame >= config.MAX_FRAME_COUNT: # abs(score.score) >= config.GOAL:
            game_ended = True

        # Show statistics
        if config.VISUALENABLED:
            draw_score(main_surface, score.score)

            draw_inputs(main_surface, subject_car.get_vision())
            draw_actions(main_surface, subject_car_action)
            draw_gauge(main_surface, subject_car.speed)
            fpsClock.tick(20000)
            pygame.event.poll()
            pygame.display.flip()

        frame += 1
        speed_counter.append(subject_car.speed)

    # Increment episode counter and calculate basic statistics
    episode_count = deep_traffic_agent.model.increase_count_episodes()
    avg_speed = np.average(speed_counter)
    total_reward = score.score
    # Log episode completion details
    logger.info(f"Episode {episode_count} completed. Average speed: {avg_speed}, Total frames: {frame}, Total reward: {total_reward}, Final speed: {subject_car.speed}")

    # Handle different logging based on training vs testing mode
    if not is_training:
        # Testing mode: collect statistics for later analysis
        speed_counter_avg.append(avg_speed)


    # Post-training analysis (after MAX_EPISODE is reached)
    if episode_count > config.MAX_EPISODE:
        # Collect statistics about driving behavior
        alternate_line_switching.append(subject_car.alternate_line_switching)
        hard_brake_avg.append(subject_car.hard_brake_count)
        
        # Every TESTING_EPISODE episodes, calculate and print comprehensive statistics
        if (episode_count - config.MAX_EPISODE) % config.TESTING_EPISODE == 0:
            # Calculate averages and medians for key metrics
            avg_speed = np.average(speed_counter_avg)
            median_speed = np.median(speed_counter_avg)
            avg_hard_brake = np.average(hard_brake_avg)
            median_hard_brake = np.median(hard_brake_avg)
            avg_alternate_line_switching = np.average(alternate_line_switching)
            median_alternate_line_switching = np.median(alternate_line_switching)
            
            # Print detailed statistics
            print("Car:{},Speed:(Mean: {}, Median: {}),Hard_Brake:(Mean: {}, Median: {}), Line::(Mean: {}, Median: {})"
                .format(PREDEFINED_MAX_CAR, avg_speed, median_speed, avg_hard_brake, median_hard_brake,
                        avg_alternate_line_switching, median_alternate_line_switching))
            
            # Test agent performance with different traffic densities
            # Changes PREDEFINED_MAX_CAR between 20, 40, and 60 cars
            if abs(PREDEFINED_MAX_CAR - 40) < 1:
                # Testing with 40 cars
                PREDEFINED_MAX_CAR = 20  # Switch to testing with 20 cars
            elif abs(PREDEFINED_MAX_CAR - 20) < 1:
                # Testing with 20 cars
                PREDEFINED_MAX_CAR = 60  # Switch to testing with 60 cars
                
            # Reset statistics collectors for next testing batch
            speed_counter_avg = []
            hard_brake_avg = []
            alternate_line_switching = []
logger.info(f"Training completed. Total episodes: {episode_count}, Action frequencies: {action_stats}")