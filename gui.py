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

# Model name
model_name = config.MODEL_NAME

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
    main_surface = pygame.display.set_mode((1600, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
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

logging.basicConfig(filename='logs/training_progress.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepTraffic')

# New episode/game round
while episode_count < config.MAX_EPISODE + config.TESTING_EPISODE * 3:      # 2000 + 200*3
    is_training = config.DL_IS_TRAINING and episode_count < config.MAX_EPISODE and not config.VISUALENABLED     #False and ep_count< 2000 and not True

    # Score object
    score = Score(score=0)

    subject_car = Car(main_surface,     # initialized above
                      lane_map,         #  ""
                      speed=60,
                      y=DEFAULT_CAR_POS,    # 700
                      lane=4,
                      is_subject=True,
                      score=score,
                      agent=deep_traffic_agent)
    object_cars = [Car(main_surface,
                       lane_map,
                       speed=60,
                       y=800,           # means wat?
                       lane=6,          # means ?
                       is_subject=False,
                       score=score,
                       subject=subject_car)
                   for i in range(6, 7)] # TODO Can remove this line.

    frame = 0

    game_ended = False

    delay_count = 0
    speed_counter = []
    subject_car_action = 'M'

    while True: # frame < config.MAX_FRAME_COUNT:
        # brick draw
        # bat and ball draw
        # events

        # Alternate for lines 105 to 123 is below(124-133)
        # if config.VISUALENABLED and not config.DLAGENTENABLED:      # True and not False
        #     pressed_key = pygame.key.get_pressed()  # returns a list representing the state of every key on the keyboard. Each element in the list is 1 if the corresponding key is currently being held down, and 0 otherwise.
        #     keydown_key = []

        #     for event in pygame.event.get():    # iterates through all the events in the Pygame event queue, allowing the program to respond to user inputs and other events.
        #         if event.type == QUIT:
        #             pygame.quit()
        #             sys.exit()
        #         elif event.type == pygame.KEYDOWN:  # If a KEYDOWN event is detected (i.e., a key was just pressed down), the key code of the pressed key is appended to the keydown_key list.
        #             keydown_key.append(event.key)

        # if config.VISUALENABLED:
        #     pressed_key = pygame.key.get_pressed()
        #     keydown_key = []

        #     for event in pygame.event.get():
        #         if event.type == QUIT or event.type == pygame.K_q:  # pygame.K_q is being used to refer to the Q key
        #             pygame.quit()
        #             sys.exit()
        if config.VISUALENABLED:
            pressed_key = pygame.key.get_pressed()
            keydown_key = []

            for event in pygame.event.get():
                if event.type == QUIT or event.type == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and not config.DLAGENTENABLED:
                    keydown_key.append(event.key)


        # Setup game background
        draw_basic_road(main_surface, subject_car.speed)

        # Car to identify available moves in the order from top to bottom
        cars = [subject_car]
        cars.extend([o_car for o_car in object_cars if o_car.removed is False])     # filtering out any cars that have been marked for removal.
        # This line sorts the cars list based on the y attribute of each car (t_car.y).
        # The reverse=True argument sorts the cars in descending order, meaning cars with higher y values (presumably further down the screen) come first in the list.
        cars.sort(key=lambda t_car: t_car.y, reverse=True)

        available_lanes_for_new_car = identify_free_lane(cars)  # to be explored

        # Add more cars to the scene
        if len(cars) < PREDEFINED_MAX_CAR and np.random.standard_normal(1)[0] >= 0:
            # Decide position(Front or back) 0 for back and 1 for front
            map_position = np.random.choice([0, 1], 1)[0]
            position = available_lanes_for_new_car[map_position]
            if len(position) > 0:
                # Back
                if map_position:    # back : 1
                    new_car_speed = np.random.randint(30, 91)  # 91 because randint's upper bound is exclusive
                    new_car_y = 1010    # indicating that it will appear at the back of the screen
                    new_car_lane = np.random.choice(position)
                    new_car_y = 1010
                else:
                    new_car_speed = np.random.randint(30, 61)
                    new_car_y = -100    # will appear front of the screen
                    new_car_lane = np.random.choice(position)
                # Decide lanes
                new_car = Car(main_surface,
                              lane_map,
                              speed=new_car_speed,
                              y=new_car_y,
                              lane=new_car_lane,
                              is_subject=False,
                              subject=subject_car,
                              score=score)
                object_cars.append(new_car)
                if position:    # if available lane, new car is added to end of cars
                    cars.append(new_car)
                else:           # else new car inserted at beginning of cars list, ensuring it gets processed first in subsequent operations.
                    cars.insert(0, new_car)

        # main game logic
        # Reinitialize lane map
        for y in range(100):
            for x in range(7):
                lane_map[y][x] = 0

        # Identify car position
        for car in cars:
            car.identify()  # updates lane positions and validates car positions

        for car in cars:
            car.identify_available_moves()

        cache = False
        if delay_count < config.DELAY and not game_ended and is_training:
            delay_count += 1
            cache = True
        else:
            delay_count = 0

        q_values = None
        # Car react to road according to order
        for car in cars[::-1]:
            # For object car
            if car.subject is not None:
                car.decide(game_ended, cache=cache, is_training=is_training)
                continue

            if config.DLAGENTENABLED:
                # Get prediction from DeepTrafficAgent
                q_values, temp_action = car.decide(game_ended, cache=cache, is_training=is_training)
                print("Q-values: ", q_values, 'Car is subject: ', car.subject)
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

        for car in cars:
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

            # Setup advanced view
            advanced_road.draw(frame, subject_car)

            # collision detection
            fpsClock.tick(20000)
            pygame.event.poll()
            pygame.display.flip()

        frame += 1
        speed_counter.append(subject_car.speed)

        if q_values is not None:
            deep_traffic_agent.model.log_q_values(q_values) # While loop ends here
            logger.info(f"Episode {episode_count}, Frame {frame}: Q-values: {q_values}")


    episode_count = deep_traffic_agent.model.increase_count_episodes()
    avg_speed = np.average(speed_counter)
    total_reward = score.score
    logger.info(f"Episode {episode_count} completed. Average speed: {avg_speed}, Total frames: {frame}, Total reward: {total_reward}, Final speed: {subject_car.speed}")

    if not is_training:
        speed_counter_avg.append(avg_speed)
        deep_traffic_agent.model.log_testing_speed(avg_speed)
    else:
        print("Average speed for episode{}: {}".format(episode_count, avg_speed))
        deep_traffic_agent.model.log_average_speed(avg_speed)
    deep_traffic_agent.model.log_total_frame(frame)
    deep_traffic_agent.model.log_terminated(frame < config.MAX_FRAME_COUNT - 1)
    deep_traffic_agent.model.log_reward(score.score)

    deep_traffic_agent.model.log_hard_brake_count(subject_car.hard_brake_count)

    if episode_count > config.MAX_EPISODE:
        alternate_line_switching.append(subject_car.alternate_line_switching)
        hard_brake_avg.append(subject_car.hard_brake_count)
        if (episode_count - config.MAX_EPISODE) % config.TESTING_EPISODE == 0:
            avg_speed = np.average(speed_counter_avg)
            median_speed = np.median(speed_counter_avg)
            avg_hard_brake = np.average(hard_brake_avg)
            median_hard_brake = np.median(hard_brake_avg)
            avg_alternate_line_switching = np.average(alternate_line_switching)
            median_alternate_line_switching = np.median(alternate_line_switching)
            print("Car:{},Speed:(Mean: {}, Median: {}),Hard_Brake:(Mean: {}, Median: {}), Line::(Mean: {}, Median: {})"
                  .format(PREDEFINED_MAX_CAR, avg_speed, median_speed, avg_hard_brake, median_hard_brake,
                          avg_alternate_line_switching, median_alternate_line_switching))
            if abs(PREDEFINED_MAX_CAR - 40) < 1:
                deep_traffic_agent.model.log_average_test_speed_40(avg_speed)
                PREDEFINED_MAX_CAR = 20
            elif abs(PREDEFINED_MAX_CAR - 20) < 1:
                deep_traffic_agent.model.log_average_test_speed_20(avg_speed)
                PREDEFINED_MAX_CAR = 60
            else:
                deep_traffic_agent.model.log_average_test_speed_60(avg_speed)
            speed_counter_avg = []
            hard_brake_avg = []
            alternate_line_switching = []

deep_traffic_agent.model.log_action_frequency(action_stats)
logger.info(f"Training completed. Total episodes: {episode_count}, Action frequencies: {action_stats}")