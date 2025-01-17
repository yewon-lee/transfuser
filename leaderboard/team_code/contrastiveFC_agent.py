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


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'ContrastiveFC_Agent'


class ContrastiveFC_Agent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		# Import autopilot class (for throttle/brake later on)
		#AutoPilot = AutoPilot(autonomous_agent.AutonomousAgent)
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque, 'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		# TODO: path to trained models
		cvm_path = '/home/yewon/semi-hard-negative-mining/models/13-Hard-NegSampling-32Batch/bestContrastive.pt'
		clm_path = '/home/yewon/semi-hard-negative-mining/models/13-Hard-NegSampling-32Batch/bestControl.pt'


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

		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
		
		#outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		#json.dump(self.pid_metadata, outfile, indent=4)
		#outfile.close()

	def destroy(self):
		del self.net_contrastive, self.net_controls
