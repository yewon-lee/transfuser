import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from contrastiveFC.models import ContrastiveLearningModel,  ControlsModel_FC
from contrastiveFC.config import GlobalConfig
from contrastiveFC.data import scale_and_crop_image, transform_2d_points, lidar_to_histogram_features
from team_code.planner import RoutePlanner
from team_code.auto_pilot import AutoPilot
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'ContrastiveFC_Agent'

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1


def check_episode_has_noise(lat_noise_percent, long_noise_percent):
    lat_noise = False
    long_noise = False
    if random.randint(0, 101) < lat_noise_percent:
        lat_noise = True

    if random.randint(0, 101) < long_noise_percent:
        long_noise = True

    return lat_noise, long_noise

class ContrastiveFC_Agent(autonomous_agent.AutonomousAgent):
    # AUTOPILOT
    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        # Import autopilot class (for throttle/brake later on)
        print(path_to_conf_file)
        #AutoPilot = AutoPilot(autonomous_agent.AutonomousAgent)
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque, 'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

        # TODO: path to trained models
        cvm_path = '/home/julia/transfuser/models/bestContrastive.pt'
        clm_path = '/home/julia/transfuser/models/bestControl.pt'


        self.config = GlobalConfig() #TODO: may not need this config file

        # Load trained model
        self.net_contrastive = ContrastiveLearningModel().cuda()
        self.net_contrastive.load_state_dict(torch.load(cvm_path))
        self.net_contrastive.eval()
        self.net_controls = ControlsModel_FC(self.net_contrastive).cuda()
        self.net_controls.load_state_dict(torch.load(clm_path))
        self.net_controls.eval()

        self.save_path = None
        if SAVE_PATH is not None:
                now = datetime.datetime.now()
                string = pathlib.Path(os.environ['ROUTES']).stem + '_'
                string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

                print (string)
                self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
                self.save_path.mkdir(parents=True, exist_ok=False)

                (self.save_path / 'rgb').mkdir()
                (self.save_path / 'meta').mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

        #AUTOPILOT
        #super()._init()
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True

        self._traffic_lights = list()
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign


    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_rear'
                    },
                {   
                'type': 'sensor.lidar.ray_cast',
                'x': 1.3, 'y': 0.0, 'z': 2.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'id': 'lidar'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        self.step += 1
        #TODO: check dimensions
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        lidar = input_data['lidar'][1][:, :3]
        result = {'rgb': rgb, 'rgb_left': rgb_left, 'rgb_right': rgb_right, 'rgb_rear': rgb_rear, 'lidar': lidar, 'gps': gps, 'speed': speed, 'compass': compass}
        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        gps = self._get_position(tick_data)

        if self.step < self.config.seq_len:
            rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
        
                
            """if not self.config.ignore_sides: # ignore = True
                rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
                
                rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

            if not self.config.ignore_rear: # ignore = True
                rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

            """
    
            self.input_buffer['lidar'].append(tick_data['lidar'])
            self.input_buffer['gps'].append(tick_data['gps'])
            self.input_buffer['thetas'].append(tick_data['compass']) # TODO: look into thetas. I think I removed a couple theta lines

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            
            return control

        # AUTOPILOT 
        print("Getting AUTOPILOT")
        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data)

        control_AP = carla.VehicleControl()
        control_AP.steer = steer + 1e-2 * np.random.randn()
        control_AP.throttle = throttle
        control_AP.brake = float(brake)

        print("Getting ContrastiveFC")
        # ContrastiveFC
        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                            torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        #encoding = []
        rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
        self.input_buffer['rgb'].popleft()
        self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
        #encoding.append(self.net.image_encoder(list(self.input_buffer['rgb'])))
    
        """
        if not self.config.ignore_sides: # ignore = True
            rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_left'].popleft()
            self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_left'])))
            
            rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_right'].popleft()
            self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_right'])))

        if not self.config.ignore_rear: # ignore = True
            rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_rear'].popleft()
            self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_rear'])))
        """

        self.input_buffer['lidar'].popleft()
        self.input_buffer['lidar'].append(tick_data['lidar'])
        self.input_buffer['gps'].popleft()
        self.input_buffer['gps'].append(tick_data['gps'])
        self.input_buffer['thetas'].popleft()
        self.input_buffer['thetas'].append(tick_data['compass'])

        lidar_processed = list()
        # transform the lidar point clouds to local coordinate frame
        ego_theta = self.input_buffer['thetas'][-1]
        ego_x, ego_y = self.input_buffer['gps'][-1]
        for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
            curr_theta = self.input_buffer['thetas'][i]
            curr_x, curr_y = self.input_buffer['gps'][i]
            lidar_point_cloud[:,1] *= -1 # inverts x, y
            lidar_transformed = transform_2d_points(lidar_point_cloud,
                    np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
            lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
            lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))
        #encoding.append(self.net.lidar_encoder(lidar_processed))

        #pred_wp = self.net(encoding, target_point)
        #steer, throttle, brake, metadata = self.net.control_pid(pred_wp, gt_velocity)
        #self.pid_metadata = metadata

        # Get steering from network & throttle/brake from autopilot
        #print("here")
        steer = self.net_controls(rgb.to('cuda',dtype=torch.float32), lidar_transformed.to('cuda',dtype=torch.float32))
        #near_node, near_command = Autopilot._waypoint_planner.run_step(gps) #TODO: haven't figured out where these are from
        throttle=1 #far_node, far_command = Autopilot._command_planner.run_step(gps) # TODO
        brake=0#_, throttle, brake, _ = AutoPilot._get_control_autopilot(near_node, far_node, tick_data)
        #throttle = 1
        #brake = 0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = throttle
        control.brake = float(brake)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)

        return control_AP

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        
        #outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        #json.dump(self.pid_metadata, outfile, indent=4)
        #outfile.close()

    # AUTOPILOT
    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle 

        return angle

    # AUTOPILOT
    def _get_control_autopilot(self, target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4.0 if should_slow else 7.0
        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed

    # AUTOPILOT
    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

        return any(x is not None for x in [vehicle, light, walker, stop_sign])

    # AUTOPILOT
    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    # AUTOPILOT
    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    # AUTOPILOT
    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    # AUTOPILOT
    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    # AUTOPILOT
    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    light.set_state(carla.libcarla.TrafficLightState.Green)

        return None

    # AUTOPILOT
    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    # AUTOPILOT
    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None

    def destroy(self):
        del self.net_contrastive, self.net_controls
