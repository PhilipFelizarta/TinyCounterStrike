def to_Direction(action_policy, temp=1.0):
	import numpy as np
	from environ import Direction

	size = 63
	onehot = np.zeros((size,))
	action_policy = np.clip(action_policy, 1e-7, 1)
	action_policy = np.power(action_policy, 1/temp)
	probs = np.squeeze(action_policy/np.sum(action_policy))
	choice = np.random.choice(size, p=probs)
	onehot[choice] = 1
	action = -1
	view_angle = 0 
	
	if choice <= 8:
		action = -1
	elif choice <= 17:
		action = Direction.UP
	elif choice <= 26:
		action = Direction.DOWN
	elif choice <= 35:
		action = Direction.LEFT
	elif choice <= 44:
		action = Direction.RIGHT
	elif choice <= 53:
		action = -2
	elif choice <= 62:
		action = -3
	
	if choice % 9 == 0:
		view_angle = np.deg2rad(0)
	elif choice % 9 == 1:
		view_angle = np.deg2rad(40)
	elif choice % 9 == 2:
		view_angle = np.deg2rad(80)
	elif choice % 9 == 3:
		view_angle = np.deg2rad(120)
	elif choice % 9 == 4:
		view_angle = np.deg2rad(160)
	elif choice % 9 == 5:
		view_angle = np.deg2rad(200)
	elif choice % 9 == 6:
		view_angle = np.deg2rad(240)
	elif choice % 9 == 7:
		view_angle = np.deg2rad(280)
	elif choice % 9 == 8:
		view_angle = np.deg2rad(320)
	
	return action, view_angle, onehot      

def data_to_planes(data, frames_array):
	import numpy as np
	from environ import Team
	
	hp = data[0]
	wep = data[1]
	time = data[2]
	team = data[3]
	ammo = data[4]/30
	smoke = data[5]
	plant = data[6]
	view = data[7] / (2*np.pi)

	if team == Team.CT:
		team = 1
	else:
		team = 0
	if smoke:
		smoke = 1
	else:
		smoke = 0
	if plant:
		plant = 1
	else:
		plant = 0
	
	hp_plane = np.full((1, 20, 20, 1), hp)
	ammo_plane = np.full((1, 20, 20, 1), ammo)
	smoke_plane = np.full((1, 20, 20, 1), smoke)
	wep_plane = np.full((1, 20, 20, 1), wep)
	time_plane = np.full((1, 20, 20, 1), time)
	team_plane = np.full((1, 20, 20, 1), team)
	plant_plane = np.full((1, 20, 20, 1), plant)
	view_plane = np.full((1, 20, 20, 1), view)
	
	
	frames_array = np.reshape(np.array(frames_array), [-1, 20, 20, 20])
	frames_array = np.swapaxes(frames_array,3,1)
	
	frames_array = np.concatenate((frames_array, hp_plane), axis=-1)
	frames_array = np.concatenate((frames_array, ammo_plane), axis=-1)
	frames_array = np.concatenate((frames_array, smoke_plane), axis=-1)
	frames_array = np.concatenate((frames_array, wep_plane), axis=-1)
	frames_array = np.concatenate((frames_array, time_plane), axis=-1)
	frames_array = np.concatenate((frames_array, team_plane), axis=-1)
	frames_array = np.concatenate((frames_array, plant_plane), axis=-1)
	frames_array = np.concatenate((frames_array, view_plane), axis=-1)
	
	return frames_array

#Save model method
def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
	import keras
	from keras.models import model_from_json


	model_json = model.to_json()
	with open(modelFile, "w") as json_file:
		json_file.write(model_json)
	model.save_weights(weightFile)
	print("Model saved!")

def load_model(modelFile, weightFile, update=False): #load model from json and HDF5
	import keras
	from keras.models import model_from_json


	json_file = open(modelFile, 'r')
	load_model_json = json_file.read()
	json_file.close()
	if not update:
		load_model = model_from_json(load_model_json)
	else:
		load_model = create_model(64)
	load_model.load_weights(weightFile)
	print("Model Loaded!")
	return load_model