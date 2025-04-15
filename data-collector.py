# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import socket
import common.const as const
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
r = s.connect_ex((const.HOST, const.PORT))

s.close()

import glob
import os
import sys
try:
    sys.path.append(glob.glob('carla_v09/dist/carla-0.9.6-py%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
    from carla import ColorConverter as cc
except IndexError:
    raise IndexError



# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================



import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_o
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from PIL import Image
import csv
import shutil

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.env_init_pos = {"OffRoad_1": [
            carla.Transform(carla.Location(8770.0 / 100, 5810.0 / 100, 120.0 / 100), carla.Rotation(yaw=-93.0)),
            carla.Transform(carla.Location(-660.0 / 100, 260.0 / 100, 120.0 / 100), carla.Rotation(yaw=165.0))],
            "OffRoad_2": [carla.Transform(carla.Location(1774.0 / 100, 4825.0 / 100, 583.0 / 100),
                                          carla.Rotation(yaw=92.0)),
                          carla.Transform(carla.Location(20417.0 / 100, 15631.0 / 100, 411.0 / 100),
                                          carla.Rotation(yaw=-55.0))],
            "OffRoad_3": [carla.Transform(carla.Location(26407.0 / 100, -4893.0 / 100, 181.0 / 100),
                                          carla.Rotation(yaw=-90.0)),
                          carla.Transform(carla.Location(-13270.0 / 100, 3264.0 / 100, 124.0 / 100),
                                          carla.Rotation(yaw=-38.0))],
            "OffRoad_4": [carla.Transform(carla.Location(-12860.0 / 100, 22770.0 / 100, 210.0 / 100),
                                          carla.Rotation(yaw=0.0)),
                          carla.Transform(carla.Location(-17110.0 / 100, 11780.0 / 100, 130.0 / 100),
                                          carla.Rotation(yaw=-53.0))],
            "Track1": [carla.Transform(carla.Location(6187.0 / 100, 6686.0 / 100, 138.0 / 100),
                                       carla.Rotation(yaw=-91.0))],
            "Track1_Inner": [carla.Transform(carla.Location(6187.0 / 100, 6686.0 / 100, 138.0 / 100),
                                       carla.Rotation(yaw=-91.0))],
            "Track1_Outer": [carla.Transform(carla.Location(6187.0 / 100, 6686.0 / 100, 138.0 / 100),
                                             carla.Rotation(yaw=-91.0))],
            "Town01": [carla.Transform(carla.Location(8850.0/100, 9470.0/100, 100.0/100), carla.Rotation(yaw=90.0)), 
                                            carla.Transform(carla.Location(33880.0/100, 24380.0/100, 100.0/100), carla.Rotation(yaw=-90.0))],
            "Town02": [carla.Transform(carla.Location(8760.0/100, 18760.0/100, 100.0/100), carla.Rotation(yaw=-0.0)), 
                                                carla.Transform(carla.Location(10850.0/100, 30690.0/100, 100.0/100), carla.Rotation(yaw=0.0))]
        }
        self.env_name = args.env_name
        self.world = carla_world
        self.actor_role_name = args.rolename
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        # self.lane_invasion_sensor = None
        # self.gnss_sensor = None
        # self.rss_sensor = None
        self.camera_manager = None
        self.frame = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.recording_customize_enabled = True
        self._controller = None
        self.replaying = False

    def set_controller(self, controller):
        self._controller = controller

    def restart(self):
        """
        Restart the simulation environment by resetting the player, sensors, and camera configurations.

        This method performs the following steps:
        1. Retains the current camera configuration if a camera manager exists.
        2. Selects a random blueprint for the player actor and sets its attributes such as role name, color, driver ID, 
           and invincibility (if applicable).
        3. Spawns the player actor at a predefined spawn point. If the player already exists, it is destroyed and respawned.
        4. Initializes the collision sensor and camera manager for the player.
        5. Configures the camera manager with the retained or default camera position and recording settings.
        6. Displays a notification with the actor type of the player.

        Attributes:
            cam_index (int): Index of the current camera configuration, defaulting to 0 if no camera manager exists.
            record_cam_index (list): Indices of cameras to be used for recording.
            cam_pos_index (int): Index of the camera position, defaulting to 3 if no camera manager exists.
            blueprint (carla.ActorBlueprint): Randomly selected blueprint for the player actor.
            spawn_point (carla.Transform): Predefined spawn point for the player actor.
            actor_type (str): Display name of the player actor.

        Raises:
            RuntimeError: If the player actor cannot be spawned after multiple attempts.
        """
        #self.world.apply_settings(carla.WorldSettings(synchronous_mode=True,fixed_delta_seconds=60,no_rendering_mode=True))
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        record_cam_index = [0, 3]
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 3
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        print("Spawned vehicle blueprint ID:", blueprint.id)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.env_init_pos[self.env_name][0]
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_point = self.env_init_pos[self.env_name][0]
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        print("Spawn point:", spawn_point)
  
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.set_record_sensor(record_cam_index)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def save(self, clock, save_iter, ofd):
        if self.replaying is False:
            if save_iter % 5 == 0:
                if self.recording_customize_enabled:
                    self.camera_manager.save(clock)
                    self.hud.save(self, clock, ofd)
                    return clock + 1
            return clock
        else:
            self.hud.save(self, clock, ofd)
            return clock + 1
        

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

        for sensor in self.camera_manager.record_sensor:
            sensor.destroy()
        self.camera_manager.record_index = [None, None]

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.player]

        for actor in actors:
            if actor is not None:
                actor.destroy()
        for actor in self.camera_manager.record_sensor:
            if actor is not None:
                actor.destroy()
        

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """
    KeyboardControl class provides keyboard-based control for a player in a CARLA simulation environment.

    This class handles user input via keyboard to control a vehicle or a walker in the simulation. It also provides
    functionality for toggling autopilot, recording, replaying, and other simulation-related features.

    Attributes:
        world (object): The simulation world object.
        recording (bool): Indicates whether recording is currently active.
        replaying (bool): Indicates whether replaying is currently active.
        _autopilot_enabled (bool): Indicates whether autopilot is enabled.
        _control (object): Control object for the player (VehicleControl or WalkerControl).
        _steer_cache (float): Cached steering value for smooth steering.
        _rotation (object): Rotation of the walker (used for WalkerControl).

    Methods:
        __init__(world, start_in_autopilot):
            Initializes the KeyboardControl instance with the given world and autopilot state.

        parse_events(client, world, clock):
            Parses keyboard events and updates the simulation state accordingly.

        _parse_vehicle_keys(keys, milliseconds):
            Handles keyboard input for controlling a vehicle.

        _parse_walker_keys(keys, milliseconds):
            Handles keyboard input for controlling a walker.

        get_action():
            Returns the current control action as a dictionary containing 'steer', 'throttle', and 'brake'.

        _is_quit_shortcut(key):
            Static method to check if the given key is a quit shortcut (ESC or Ctrl+Q).
    """
    def __init__(self, world, start_in_autopilot):
        self.world = world
        self.recording = False
        self.replaying = False
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                    self.recording = not self.recording
                    if world.recording_customize_enabled:
                        world.recording_customize_enabled = False
                    else:
                        world.recording_customize_enabled = True
                # elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                elif event.key == K_o:
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        self.recording = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("./manual_recording.txt")
                        world.recording_enabled = True
                        self.recording = True
                        world.hud.notification("Recorder is ON")
                    self.replaying = False
                elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                    self._autopilot_enabled = not (self._autopilot_enabled)
                    if self._autopilot_enabled:
                        world.hud.notification("Autopilot On")
                    else:
                        world.hud.notification("Autopilot Off")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    self.replaying = not (self.replaying)
                    world.replaying = self.replaying
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    self.recording = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.txt'")
                    # replayer
                    client.replay_file("./manual_recording.txt", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl): 
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 0.8 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 3.333 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    def get_action(self):
        action = {
            'steer': self._control.steer,
            'throttle': self._control.throttle,
            'brake': self._control.brake
        }
        return action

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- PredefinedControl -----------------------------------------------------------
# ==============================================================================


class PredifinedControl(object):
    def __init__(self, world, action_set):
        self.world = world
        self.action_set = action_set
        self.action_count = 1
        self.recording = False
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
        else:
            raise NotImplementedError("Actor type not supported")
    
    def parse_events(self, client, world, clock):
        if self.action_count > len(self.action_set):
            print("Saved Action finished..")
            return True
            
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if self._is_quit_shortcut(event.key):
                        return True
                    elif event.key == K_BACKSPACE:
                        world.restart()
                    elif event.key == K_TAB:
                        world.camera_manager.toggle_camera()
                    elif event.key == K_r:
                        world.camera_manager.toggle_recording()
                        self.recording = not (self.recording)
                        if world.recording_customize_enabled:
                            world.recording_customize_enabled = False
                        else:
                            world.recording_customize_enabled = True
                    # self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())        
            if self.recording:
                self._control.steer = float(self.action_set[self.action_count][0])
                self._control.throttle = float(self.action_set[self.action_count][1])
                self._control.brake = float(self.action_set[self.action_count][2])
            world.player.apply_control(self._control)
            
            if self.recording:
                print("With saved action #{}".format(self.action_count))
                print(" Expert Steer: %s   Throttle: %s   Brake: %s" %(self.action_set[self.action_count][0], self.action_set[self.action_count][1], self.action_set[self.action_count][2]))
                c = world.player.get_control()
                print("Vehicle Steer: %s   Throttle: %s   Brake: %s" %(c.steer, c.throttle, c.brake))
                
                self.action_count += 10
            return False
    
    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle += 1.0 if keys[K_UP] or keys[K_w] else self._control.throttle
        self._control.brake += 1.0 if keys[K_DOWN] or keys[K_s] else self._control.brake

        steer_increment = 5e-3 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """
    Heads-Up Display (HUD) class for rendering and managing on-screen information 
    in a simulation environment.
    Attributes:
        dim (tuple): Dimensions of the display (width, height).
        server_fps (float): Frames per second of the server.
        frame (int): Current frame number.
        simulation_time (float): Elapsed simulation time in seconds.
        _show_info (bool): Flag to toggle the display of information.
        _info_text (list): List of information strings to display.
        _server_clock (pygame.time.Clock): Clock to track server FPS.
        _font_mono (pygame.font.Font): Monospaced font for rendering text.
        _notifications (FadingText): Object for managing fading text notifications.
        help (HelpText): Object for rendering help text.
    Methods:
        __init__(width, height):
            Initializes the HUD with the given dimensions.
        on_world_tick(timestamp):
            Updates server FPS, frame number, and simulation time based on the world tick.
        save(world, clock, ofd):
            Saves vehicle control and state information to a file.
        tick(world, clock):
            Updates HUD information based on the world state and clock.
        toggle_info():
            Toggles the display of HUD information.
        notification(text, seconds=2.0):
            Displays a temporary notification on the HUD.
        error(text):
            Displays an error message on the HUD.
        render(display):
            Renders the HUD information and notifications onto the given display surface.
    """
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def save(self, world, clock, ofd):
        v = world.player.get_velocity()
        c = world.player.get_control()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) # m/s
        
        target = {'step': 'rgb_%08d' % clock, 'steer': c.steer, 'throttle': c.throttle, 'brake': c.brake, 'speed':speed}
        w = csv.DictWriter(ofd, target.keys())
        w.writerow(target)

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            #'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            #'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """
    A sensor class to detect and handle collision events in a simulation.

    Attributes:
        sensor (carla.Actor): The collision sensor actor attached to the parent actor.
        history (list): A list of tuples containing collision frame and intensity.
        _parent (carla.Actor): The parent actor to which the collision sensor is attached.
        hud (HUD): The HUD object used to display notifications.

    Methods:
        __init__(parent_actor, hud):
            Initializes the CollisionSensor, attaches it to the parent actor, and sets up a listener for collision events.

        get_collision_history():
            Retrieves the collision history as a dictionary with frame numbers as keys and intensity as values.

        _on_collision(weak_self, event):
            Handles collision events, updates the collision history, and displays a notification on the HUD.
    """
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- RssSensor --------------------------------------------------------
# ==============================================================================


class RssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.timestamp = None
        self.response_valid = False
        self.lon_response = None
        self.lat_response_right = None
        self.lat_response_left = None
        self.acceleration_restriction = None
        self.ego_velocity = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.rss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=0.0, z=0.0)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.

        def check_rss_class(clazz):
            return inspect.isclass(clazz) and "RssSensor" in clazz.__name__

        if not inspect.getmembers(carla, check_rss_class):
            raise RuntimeError('CARLA PythonAPI not compiled in RSS variant, please "make PythonAPI.rss"')
        weak_self = weakref.ref(self)
        self.sensor.visualize_results = True
        self.sensor.listen(lambda event: RssSensor._on_rss_response(weak_self, event))

    @staticmethod
    def _on_rss_response(weak_self, response):
        self = weak_self()
        if not self or not response:
            return
        self.timestamp = response.timestamp
        self.response_valid = response.response_valid
        self.lon_response = response.longitudinal_response
        self.lat_response_right = response.lateral_response_right
        self.lat_response_left = response.lateral_response_left
        self.acceleration_restriction = response.acceleration_restriction
        self.ego_velocity = response.ego_velocity


# ==============================================================================
# -- LaneInvasionSensor -------------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GNSSSensor -------------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))
    
    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.record_sensor = []
        self.record_image = [None, None]
        self.record_points = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        self.frame = 0
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        #    ↑ Z (Up)
        #    |
        #    |
        #    o------> X (Right)
        #    |
        #    |
        #  Forward (Y)

        self._record_camera_transforms = [
            (carla.Transform(carla.Location(x=1.5, y=0.0, z=2.3), carla.Rotation()), Attachment.Rigid),
            (carla.Transform(carla.Location(x=0.0, y=0.0, z=2.6), carla.Rotation()), Attachment.Rigid)]
        
        self.transform_index = 1
        self.record_transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
            # ['sensor.lidar.ray_cast_semantic', 'Semantic LiDAR']]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov','90')
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                bp.set_attribute('sensor_tick', '0.1')
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '1000.0')  # Typical for OS1-64
                bp.set_attribute('rotation_frequency', '20.0')  # Matches Ouster's supported RPM
                bp.set_attribute('channels', '64')  # OS1-64 has 64 channels
                bp.set_attribute('points_per_second', '1310720')  # 20Hz * 65536 points/sec

                # Ouster OS1-64 has a vertical FoV of about 45°
                bp.set_attribute('upper_fov', '22.5')   # Half of 45°
                bp.set_attribute('lower_fov', '-22.5')  # Half of -45°

                bp.set_attribute('sensor_tick', '0.1')  # 10hz


            item.append(bp)
        self.index = None
        self.record_index = [None, None]

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_record_sensor(self, index_list, force_respawn=False):
        """
        Configures and spawns the recording sensors based on the provided index list.
        This method checks if the recording sensors need to be respawned based on the 
        provided indices or a forced respawn flag. If respawn is required, it destroys 
        the existing recording sensors and spawns new ones at the specified transforms. 
        The new sensors are then set up to listen for image data.
        Args:
            index_list (list): A list of two indices specifying the sensors to be used 
                               for recording.
            force_respawn (bool, optional): If True, forces the respawn of the recording 
                                            sensors regardless of their current state. 
                                            Defaults to False.
        Behavior:
            - If the recording sensors need to be respawned:
                - Destroys any existing recording sensors.
                - Spawns new sensors based on the provided indices and attaches them 
                  to the parent actor.
                - Sets up listeners for the new sensors to process image data.
            - Updates the `record_index` attribute with the provided indices.
        """
        needs_respawn = True if (self.record_index[0] is None and self.record_index[1] is None) else \
            (force_respawn or self.sensors[index_list[0]][0] != self.sensors[self.record_index[0]][0] and self.sensors[index_list[1]][0] != self.sensors[self.record_index[1]][0])
        # print("needs_respawn: ", needs_respawn)
        # print("record_index: ", self.record_index)
        # print("index_list: ", index_list)

        if needs_respawn:
            if len(self.record_sensor) != 0:
                for sensor in self.record_sensor:
                    sensor.destroy()    

            for i in range(2):
                # rsensor = self._parent.get_world().spawn_actor(
                #     self.sensors[index_list[i]][-1],
                #     self._record_camera_transforms[self.record_transform_index][0],
                #     attach_to = self._parent,
                #     attachment_type= self._record_camera_transforms[self.record_transform_index][1])
                # print(self.record_transform_index)
                # print(self._record_camera_transforms[self.record_transform_index])
                # print(self._record_camera_transforms[self.record_transform_index])

                rsensor = self._parent.get_world().spawn_actor(
                    self.sensors[index_list[i]][-1],
                    self._record_camera_transforms[i][0],
                    attach_to=self._parent,
                    attachment_type=self._record_camera_transforms[i][1])

                print(self._record_camera_transforms[i][0])
                print(self._record_camera_transforms[i][1])


                self.record_sensor.append(rsensor)
            weak_self = weakref.ref(self)
            self.record_sensor[0].listen(lambda image: CameraManager._parse_record_image(weak_self, image, 0))
            self.record_sensor[1].listen(lambda image: CameraManager._parse_record_image(weak_self, image, 1))
        self.record_index = index_list
     
    
    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    def save(self, clock):
        # image save to disk
        # if self.recording:
        # print(self.record_image[0], self.record_image[1])
        if self.record_image[0] is not None and self.record_image[1] is not None:
            self.record_image[0].save_to_disk(os.path.join(const.SAVE_PATH, 'RGB','rgb_%08d' % clock))
            self.record_image[1].save(os.path.join(const.SAVE_PATH, 'lidar_pc','lidar_%08d.png' % clock))
        if self.record_points is not None:
            np.save(os.path.join(const.SAVE_PATH, 'points', 'points_%08d.npy' % clock), self.record_points)
            # self.record_image[1].save_to_disk(os.path.join(const.SAVE_PATH, 'depth_%08d' % clock))
    

    @staticmethod
    def _get_transform_matrix(transform: carla.Transform):
        """Convert CARLA transform to 4x4 matrix"""
        rotation = transform.rotation
        location = transform.location

        cy, sy = np.cos(np.radians(rotation.yaw)), np.sin(np.radians(rotation.yaw))
        cp, sp = np.cos(np.radians(rotation.pitch)), np.sin(np.radians(rotation.pitch))
        cr, sr = np.cos(np.radians(rotation.roll)), np.sin(np.radians(rotation.roll))

        matrix = np.identity(4)
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z

        matrix[0, 0] = cp * cy
        matrix[0, 1] = cy * sp * sr - sy * cr
        matrix[0, 2] = -cy * sp * cr - sy * sr

        matrix[1, 0] = cp * sy
        matrix[1, 1] = sy * sp * sr + cy * cr
        matrix[1, 2] = -sy * sp * cr + cy * sr

        matrix[2, 0] = -sp
        matrix[2, 1] = cp * sr
        matrix[2, 2] = cp * cr

        return matrix
    def _camera_to_standard_coords(cam_coords):
        """
        Convert from custom Camera-style XYZ (X right, Y backward, Z down)
        to standard Camera-style XYZ (X right, Y down, Z forward)
        """
        cam2std = np.array([
            [1,  0,  0],   # X stays the same
            [0,  0,  1],   # Y = Z
            [0, -1,  0]    # Z = -Y
        ])
        return cam_coords @ cam2std.T


    def _parse_record_image(weak_self, image, index):
        from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plot
        import matplotlib.pyplot as plt
        self = weak_self()
        if not self:
            return
        lidar_data = None
        
        if self.sensors[self.record_index[index]][0].startswith('sensor.lidar'):
            # print("Sensor name:", self.sensors[self.record_index[index]][0])
            print("Raw data length (bytes):", len(image.raw_data))
            lidar_data = np.frombuffer(image.raw_data, dtype=np.float32).reshape(-1, 3).copy()
            # print(f"LiDAR data shape: {lidar_data.shape}")
            # np.save("/home/yixin/Off-road-Benchmark/data/dataset1/lidar_points.npy", lidar_data)
            
            sensor_l = self.record_sensor[index]
            sensor_c = self.record_sensor[index-1]

            # Reorder columns: from [x, y, z] → [z, x, y]
            # lidar_data = lidar_data[:, [2, 1, 0]]
            lidar_transform= sensor_l.get_transform()
            camera_transform = sensor_c.get_transform()

            # print(f"LiDAR Transform: {lidar_transform}")
            # print(f"Camera Transform: {camera_transform}")
        
            # lidar_data *= 5.0  # scale to [-50, 50] → total spread ~100m
            lidar_matrix = CameraManager._get_transform_matrix(lidar_transform)
            camera_matrix =CameraManager._get_transform_matrix(camera_transform)

            lidar_to_camera = np.linalg.inv(camera_matrix) @ lidar_matrix
            # print(lidar_to_camera)
            lidar_xyz = lidar_data[:, :3]
            lidar_points_hom = np.concatenate([lidar_xyz, np.ones((lidar_xyz.shape[0], 1))], axis=1)

            points_camera = (lidar_to_camera @ lidar_points_hom.T).T[:, :3]
            # np.save("/home/yixin/Off-road-Benchmark/data/dataset1/points_camera.npy", points_camera)

            # X right, Y up, Z forward
            points_camera_fixed = CameraManager._camera_to_standard_coords(points_camera)
            # Filter once
            valid = points_camera_fixed[:, 2] > 0
            points_camera = points_camera_fixed[valid]
            depth = points_camera_fixed[:, 2]
            # np.save("/home/yixin/Off-road-Benchmark/data/dataset1/points_camera1.npy", points_camera)

            print(f"LiDAR points projected into camera frame: {points_camera.shape[0]}")
            print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")

            # return
            

            image_w = int(1200)
            image_h = int(1080)
            fov = float(90)
            camera_image = np.zeros((image_h, image_w, 3), dtype=np.uint8)

            focal = image_w / (2.0 * np.tan(np.radians(fov / 2.0)))
            K = np.array([
                [focal, 0, image_w / 2.0],
                [0, focal, image_h / 2.0],
                [0,     0,           1.0]
            ])

            # print(K)
            # Project
            points_2d = (K @ points_camera_fixed[:, :3].T).T
            points_2d[:, 0] /= points_2d[:, 2]
            points_2d[:, 1] /= points_2d[:, 2]
            x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
            y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
            d_min, d_max = depth.min(), depth.max()

            # print(f"➤ X range: {x_min:.2f} to {x_max:.2f}")
            # print(f"➤ Y range: {y_min:.2f} to {y_max:.2f}")
            # print(f"➤ Depth range: {d_min:.2f} to {d_max:.2f}")

            # print(f"points 2d after projection: {points_2d.shape[0]}")
            # print("➤ Projected points (image space):", points_2d.shape)
            # print("➤ Depth values:", depth.shape)
            # print("➤ depth > 0:", np.sum(depth > 0))
            # print(f"➤ X valid:", np.sum((points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_w)))
            # print("➤ Y valid:", np.sum((points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_h)))

            mask = (
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_w) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_h) &
                (depth > 0)
            )

            pixels = points_2d[mask]
            # print(f"Valid image points after projection: {pixels.shape[0]}")
            depths = depth[mask]

            # import matplotlib.pyplot as plt

            # Normalize depth to 0–1
            depth_min = np.percentile(depths, 1)
            depth_max = np.percentile(depths, 99)
            depths_clipped = np.clip((depths - depth_min) / (depth_max - depth_min), 0.0, 1.0)

            # Use matplotlib colormap
            colors = plt.cm.plasma(1.0 - depths_clipped)[:, :3] * 255  # RGB from colormap
            colors = colors.astype(np.uint8)

            for i in range(pixels.shape[0]):
                x = int(pixels[i, 0])
                y = int(pixels[i, 1])
                color = tuple(colors[i])  # (R, G, B)
                if 0 <= x < image_w and 0 <= y < image_h:
                    camera_image[y, x] = color

            # import cv2
            # cv2.imshow("LiDAR Depth on Camera", camera_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image =  image = Image.fromarray(camera_image)
          

        else:
            image.convert(self.sensors[self.record_index[index]][1])
        self.record_image[index] = image
        self.record_points = lidar_data
       
            # if image.frame % 10 == 0:
            #     if index == 0:
            #         sentence = 'rgb'
            #     else:
            #         sentence = 'lidar'
            #     image.save_to_disk('dataset/'+sentence+'/%08d' % image.frame)
                
            

    @staticmethod
    def _parse_image(weak_self, image):
        """
        Processes and visualizes image or LiDAR data from a sensor.

        Args:
            weak_self (weakref): A weak reference to the instance of the class containing this method.
            image: The image or LiDAR data object to be processed.

        Behavior:
            - If the sensor is a LiDAR sensor (identified by 'sensor.lidar' in the sensor name):
                - Converts raw LiDAR data into a 2D point cloud.
                - Scales and transforms the point cloud to fit the HUD dimensions.
                - Creates a 2D image representation of the LiDAR data.
                - Stores the resulting image as a pygame surface in `self.surface`.
            - If the sensor is a camera:
                - Converts the image to the appropriate format.
                - Extracts the RGB channels and flips the color order from BGR to RGB.
                - Stores the resulting image as a pygame surface in `self.surface`.

        Notes:
            - The method uses weak references to avoid circular references.
            - The `self.recording` block is commented out, which would save images to disk if enabled.
        """
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            self.frame = image.frame
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # if self.recording:
            # image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main game loop for the off-road benchmark data collection.
    This function initializes the CARLA simulator, sets up the environment, 
    and collects data during the simulation. The data is saved to a CSV file 
    and includes information such as vehicle speed, control inputs, and other 
    relevant measurements.
    Args:
        args: An object containing the following attributes:
            - host (str): The IP address of the CARLA server.
            - port (int): The port number of the CARLA server.
            - env_name (str): The name of the environment to load in CARLA.
            - width (int): The width of the display window.
            - height (int): The height of the display window.
            - autopilot (bool): Whether to enable autopilot for the vehicle.
    Behavior:
        - Initializes the CARLA client and connects to the server.
        - Loads the specified environment and sets the weather to clear noon.
        - Creates a CSV file to store simulation data.
        - Runs the simulation loop, where:
            - Vehicle control inputs are captured.
            - Data is filtered and saved if certain conditions are met.
            - The simulation world is updated and rendered.
        - Stops the simulation after a predefined number of iterations or 
          when the user exits.
    Notes:
        - The function ensures proper cleanup of resources, including stopping 
          the CARLA recorder and destroying all actors, before exiting.
        - The simulation runs at a fixed frame rate of 60 FPS.
    """
    # create a dataset to save data -> dataset1
    if os.path.exists(const.SAVE_PATH) is not True:
        os.makedirs(const.SAVE_PATH)
    # Make a CSV file where we can store data like vehicle speed, location, ID,
    ofd = open(os.path.join(const.SAVE_PATH, 'measurements.csv'), 'w', newline='')
    target = {'step': None, 'steer': None, 'throttle': None, 'brake': None, 'speed':None, 'direction': None}
    w = csv.DictWriter(ofd, target.keys())
    w.writeheader()

    pygame.init()
    pygame.font.init()
    world = None
    count = 0
    step = 0

    os.makedirs(os.path.join(const.SAVE_PATH, 'RGB'), exist_ok=True)
    os.makedirs(os.path.join(const.SAVE_PATH, 'lidar_pc'), exist_ok=True)
    os.makedirs(os.path.join(const.SAVE_PATH, 'points'), exist_ok=True)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
    
        client.load_world(args.env_name)
        client.get_world().set_weather(carla.WeatherParameters.ClearNoon)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        

        clock = pygame.time.Clock()
        
        while True:

            if count > 80000 * 5:
                break
            clock.tick_busy_loop(60)
            
            if controller.parse_events(client, world, clock):
                return

            c = world.player.get_control()

            #filter data
            if c.throttle > 0.1:
                step = world.save(step, count, ofd)
            
            #display
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            count += 1
    finally:
        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
            print("All actors destroyed..")

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information'
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot'
    )
    argparser.add_argument(
        '--width',
        metavar='WIDTH',
        type=int,
        default=const.WIDTH,
        help='window width'
    )
    argparser.add_argument(
        '--height',
        metavar='HEIGHT',
        type=int,
        default=const.HEIGHT,
        help='window height'
    )
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.nissan.patrol',
        help='actor filter (default: "vehicle.*")'
    )
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")'
    )
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)'
    )
    argparser.add_argument(
        '--env_name',
        default='OffRoad_2',
        help='map name'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
  
    main()
