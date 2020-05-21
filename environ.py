#Environment
import numpy as np
from enum import Enum

class Team(Enum):
	CT = 0
	T = 1
	
class Direction(Enum):
	UP = 1
	DOWN = 2
	LEFT = 3
	RIGHT = 4

class Board:
	def __init__(self):
		self.size = 10
		self.players = [] #Define a list of players on the map
		self.boxes = [] #Define a list for the boxes on the map
		self.observation = np.zeros((2, 2*self.size, 2*self.size))
		self.timer = 10000
		self.bomb = None
		self.defused = False
	
	def init_observation(self):
		for i in range(int(self.size*2)):
			t_x = i - self.size
			for j in range(int(self.size*2)):
				t_y = j - self.size

				if self.in_bombsite(t_x, t_y):
					self.observation[1][j][i] = -0.5


				for player in self.players:
					if t_x == player.x and t_y == player.y:
						if player.team == Team.T:
							self.observation[0][j][i] = -0.1
						else:
							self.observation[0][j][i] = -0.5

						break
					else:
						self.observation[0][j][i] = 1

				for box in self.boxes: #Box plane
					if box.in_box(t_x, t_y):
						val = 1
						if box.passable:
							val = 0.5

						if box.bomb:
							val = -1

						self.observation[1][j][i] = val
						self.observation[0][j][i] = 0
						break

				

	def in_bombsite(self, objectx, objecty):
		if objectx <= -5 and objectx >= -8:
			if objecty <= -2 and objecty >= -6:
				return True
		elif objectx <= 10 and objectx >= 5:
			if objecty <= 5 and objecty >= -5:
				return True
		return False

	def dist_to_site(self, objectx, objecty):
		distA = (objectx - 7.5)**2 + (objecty)**2
		d1 = (7.5)**2 + (8.0)**2
		distA = 1 - (distA/d1)

		distB = (objectx + 6.5)**2 + (objecty + 4)**2
		d2 = (6.5)**2 + (12)**2
		distB = 1 - (distB/d2)
		reward = np.maximum(distA, distB)
		return reward
	
	def add_player(self, player):
		self.players.append(player)
	
	def add_box(self, box):
		self.boxes.append(box)
	
	def in_board(self, objectx, objecty):
		if objectx <= self.size-1 and objectx >= -self.size:
			if objecty <= self.size-1 and objecty >= -self.size:
				return True
		return False
	
class Box:
	def __init__(self, width, height,  centerX, centerY, board, passable=False, bomb=False):
		self.w = width
		self.h = height
		self.x = centerX
		self.y = centerY
		self.board = board
		self.bomb = bomb
		self.passable = passable
		self.board.add_box(self)
		
	
	def in_box(self, objectx, objecty):
		if objectx <= self.x + self.w/2 and objectx >= self.x - self.w/2:
			if objecty <= self.y + self.h/2 and objecty >= self.y - self.h/2:
				return True
		return False
		
		

class Weapon:
	def __init__(self, damage, rate, clip, speed, range):
		self.damage = damage 
		self.rate = rate
		self.clip = clip
		self.speed = speed
		self.range = range
	
class Player:
	def __init__(self, centerX, centerY, team, weapon, view, board):
		self.size = 1
		self.x = centerX
		self.y = centerY 
		self.team = team #what team are they on
		self.weapon = weapon #What weapon will be equipped
		self.view = view #0 - 2pi.. where they are looking
		self.hp = 100
		self.board = board #define what map the player is on 
		self.board.add_player(self)
		self.fov = np.deg2rad(110) #The player's field of view
		self.bomb = False
		self.smoke = True
		if team == Team.T:
			self.bomb = True
	
	def view_mask(self): #return a mask of what pixels the player can see(1 where you can view and 0 where you cannot)
		view_width = 11 #how many units wide their view is
		view_length = self.board.size #how far they can see
		mask = np.zeros((self.board.size*2, self.board.size*2))
		
		perp_view = self.view + np.deg2rad(90)
		for k in range(view_width):
			offsetx = (k - (view_width  - 1)/2)*np.cos(perp_view)
			offsety = (k - (view_width  - 1)/2)*np.sin(perp_view)
			
			for t in range(view_length):
				rise = t*np.sin(self.view)
				run = t*np.cos(self.view)
				
				viewx = int(offsetx + run + self.x) + self.board.size
				viewy = int(offsety + rise + self.y) + self.board.size
				
				if viewx >= 0 and viewx <= 2*self.board.size-1:
					if viewy >= 0 and viewy <= 2*self.board.size-1:
						if self.board.observation[1][viewy][viewx] > 0: #There is a box on this pixel
							break #Don't look any further!
						mask[viewy][viewx] = 1
		
		return mask  
	
	def move(self, direction): #Move players position
		if self.hp > 0:
			tempx = self.x
			tempy = self.y
			
			if direction == Direction.UP:
				self.y += self.weapon.speed
			elif direction == Direction.DOWN:
				self.y -= self.weapon.speed
			elif direction == Direction.LEFT:
				self.x -= self.weapon.speed
			elif direction == Direction.RIGHT:
				self.x += self.weapon.speed
			
			for box in self.board.boxes: #Check movement collisions with boxes
				if box.in_box(self.x, self.y):
					if not box.passable:
						self.x = tempx
						self.y = tempy
			
			if not self.board.in_board(self.x, self.y):
				self.x = tempx
				self.y = tempy
					
					
	
	def set_view(self, new_view):
		self.view = new_view
	
	def plant(self): #Plant the bomb
		if self.bomb: #If you have the bomb
			if self.board.in_bombsite(self.x, self.y): #If you are in the bombsite
				box = Box(1.0, 1.0, self.x, self.y, self.board, passable=True, bomb=True) #Create a box of 1.0 at ur location
				self.board.bomb = box
				self.bomb = False
				return True
		return False
	
	def defuse(self):
		if not self.bomb and self.team == Team.CT:
			if self.board.bomb is not None:
				if self.board.bomb.in_box(self.x, self.y):
					self.board.defused = True
					return True
		return False
	
	def fire_smoke(self):
		if self.smoke: #If a smoke is in your inventory
			#smoke has range of 15
			smokeX = 7.5 * np.cos(self.view) + self.x
			smokeY = 7.5 * np.sin(self.view) + self.y
			smoke = Box(3, 3, smokeX, smokeY, self.board, passable=True)
			self.smoke = False
					
	
	def fire(self): #Fire your weapon
		if self.hp > 0 and self.weapon.clip > 0:
			self.weapon.clip = self.weapon.clip - 1
			vectX = np.cos(self.view)
			vectY = np.sin(self.view)

			for i in range(len(self.board.players)):
				player = self.board.players[i]

				if player != self:
					Y = player.y - self.y
					X = player.x - self.x
					scalar = (X * vectX + Y * vectY)/(np.square(vectX) + np.square(vectY))
					if scalar >= 0:
						projX = scalar * vectX
						projY = scalar * vectY
						dist = np.square(Y - projY) + np.square(X - projX)
						if dist <= np.square(2.0): #If our weapon has hit
							cancel = False
							for j in range(len(self.board.boxes)):
								box = self.board.boxes[j]
								if not box.passable:
									for k in range(100):
										bx = (k/100)*scalar * vectX + self.x
										by = (k/100)*scalar * vectY + self.y
										if box.in_box(bx, by):
											cancel = True
											break
								if cancel:
									break
							if not cancel:
								player.hp -= self.weapon.damage
								self.board.players[i] = player