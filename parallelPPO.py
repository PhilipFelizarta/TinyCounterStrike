import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def pool_job(f_model_weights, g_model_weights, h_model_weights, PPO_data, lock):
	import keras
	import numpy as np

	from keras.models import Model
	from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout
	from keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape
	from keras.optimizers import SGD
	from keras.initializers import glorot_uniform
	from keras.regularizers import l1, l2
	from keras import backend as K
	import tensorflow as tf
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)


	lambd = 0.001 #L2 regularization
	resblocks = 2
	filters = 32

	def stem(X, filters, stage="stem", size=3):
		stem = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					  name='Conv_' + stage, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		stem = Activation('relu')(stem)
		return stem

	def res_block(X, filters, block, size=3):
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block1_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		res = Activation('relu')(res)
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block2_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(res)
		
		X = Add()([X, res])
		X = Activation('relu')(X)
		return X

	image = Input(shape=(40, 40, 27)) #Masked images of last 10 board frames + 6 planes of info
	
	#Stem for our resnet
	X = stem(image, filters)
	
	#Resnet
	for i in range(resblocks):
		X = res_block(X, filters, str(i + 1))
	
	#Create latent representation
	latent = stem(X, 1, stage="latent", size=1)
	latent = Flatten()(latent)
	latent = Dense(50, activation="sigmoid", name="latentspace", kernel_initializer=glorot_uniform(),
				kernel_regularizer = l2(lambd))(latent)
	
	#Create state -> hidden state model
	f_model = Model(inputs=image, outputs=latent)
	
	#Create policy for player 1 [Fire, Left, Right, Up, Down, Plant/Defuse] x [-90, -45, -27.5, -13.75, 0 .... 90] = 63
	p = Dense(63, activation="softmax", name="fcPolicy1", kernel_initializer=glorot_uniform(),
				kernel_regularizer = l2(lambd), input_shape=(50,))
	
	#Create value function for actor critic
	v = Dense(1, activation="tanh", name="fcValue", kernel_initializer=glorot_uniform(),
				kernel_regularizer = l2(lambd), input_shape=(50,))
	
	#Create hidden state -> policy/value model
	prev_latent = Input(shape=(50,))
	pol = p(prev_latent)
	val = v(prev_latent)
	g_model = Model(inputs=prev_latent, outputs=[pol, val])
	
	#Create hidden state -> hidden state model (MCTS in LATENT SPACE)
	p_latent = Input(shape=(50,))
	new_policy = Input(shape=(63,))
	concat_input = Concatenate(axis=-1)([new_policy, p_latent])
	new_latent = Dense(50, activation="sigmoid", name="new_latent", kernel_initializer=glorot_uniform(),
				kernel_regularizer = l2(lambd))(concat_input)
	h_model = Model(inputs=[p_latent, new_policy], outputs=new_latent)

	f_model.set_weights(f_model_weights)
	g_model.set_weights(g_model_weights)
	h_model.set_weights(h_model_weights)

	#Import the game environment from environ.py
	from environ import Weapon, Board, Player, Box, Team, Direction
	from helper import to_Direction, data_to_planes
	from MCTS import muMCTS
	for _ in range(8):
		frames = 155

		ak = Weapon(50, 1, 30, 2, 100)
		awp = Weapon(50, 1, 30, 2, 100)
		Map = Board()
		t_spawn = np.random.rand()*40 - 20
		ct_spawn = np.random.rand()*40 - 20
		t = Player(t_spawn, 16, Team.T, ak, 0, Map)
		ct = Player(ct_spawn, -16, Team.CT, awp, 0, Map)

		box1 = Box(10, -7.5, -10, Map)
		box2 = Box(10, 13.5, -10, Map)
		box3 = Box(10, -10, 10, Map)
		box4 = Box(10, 10, 10, Map)

		Map.init_observation()
		T_frames = [] #Will hold the last 10 frames
		CT_frames = [] #Will hold the last 10 frames

		mask_T = t.view_mask()
		obs_T = Map.observation
		obs_T[0] = obs_T[1] * mask_T
		obs_T[1] = obs_T[1] * mask_T

		mask_CT = ct.view_mask()
		obs_CT = Map.observation
		obs_CT[0] = obs_CT[1] * mask_CT
		obs_CT[1] = obs_CT[1] * mask_CT

		#[model.input, target, expert, action_t, old_latent, new_latent]
		model_image = []
		target = []
		advantages = []
		action = []
		expert = []

		action_t = []
		old_latent = []
		new_latent = []

		powers = [] #This will hold information for the advantage calc
		turn = []

		for _ in range(10): #Initialize list to have 10 frames
			T_frames.append(obs_T)
			CT_frames.append(obs_CT)

		time_done = True
		ct_won = True
		final_time = 0

		#Set up current data to be sent to our NN
		plant = False
		T_data = [t.hp/100, 0.0, 1.0, 0, t.weapon.clip, t.smoke, plant]
		T_image = data_to_planes(T_data, T_frames)
		T_image = np.array(T_image)
		T_image = np.reshape(T_image, [1, 40, 40, 27])

		CT_data = [ct.hp/100, 0.0, 1.0, 1, ct.weapon.clip, ct.smoke, plant]
		CT_image = data_to_planes(CT_data, CT_frames)
		CT_image = np.array(CT_image)
		CT_image = np.reshape(CT_image, [1, 40, 40, 27])

		t_old_latent = f_model.predict(T_image)
		ct_old_latent = f_model.predict(CT_image)

		#Game loop
		for time in range(frames):
			final_time = time + 1
			time_left = (frames - time)/(frames) #Time left scaled from 0 to 1
			hiddenT = f_model.predict(T_image)
			t_policy, t_value = g_model.predict(hiddenT)
			hiddenCT = f_model.predict(CT_image)
			ct_policy, ct_value = g_model.predict(hiddenCT)
			
			#T should have slight advantage in game loop processing since CT wins
			#by time
			direction, t_angle, t_a1 = to_Direction(t_policy, temp=1.0)
			t.set_view(t.view + t_angle) #Implement our view policy
			
			#Add T data to PPO lists
			r = (ct.hp - t.hp)/100
			model_image.append(T_image)
			target.append(r)
			advantages.append(t_value)
			expert.append(t_policy)
			action.append(t_a1)
			
			powers.append(time)
			turn.append(-1)
			
			if direction == -1:
				t.fire()
			elif direction == -2:
				planted = t.plant()
				if planted:
					plant = True
					Map.timer = 15
			elif direction == -3:
				t.fire_smoke()
			else:
				t.move(direction)
			
			if ct.hp <= 0:
				time_done= False
				ct_won = False
				break
				
			#Process CT
			direction, ct_angle, ct_a1 = to_Direction(ct_policy, temp=1.0)
			ct.set_view(ct.view + ct_angle) #Implement our view policy
			
			#Add CT data to PPO lists
			r = (ct.hp - t.hp)/100
			model_image.append(CT_image)
			target.append(r)
			advantages.append(ct_value)
			expert.append(ct_policy)
			action.append(ct_a1)
			
			
			powers.append(time)
			turn.append(1)
			
			if direction == -1:
				ct.fire()
			elif direction == -2:
				defused = ct.defuse()
				if defused:
					time_done = False
					break
			elif direction == -3:
				ct.fire_smoke()
			else:
				ct.move(direction)
			
			if t.hp <= 0 and Map.bomb is None:
				time_done = False
				break
			
			#Update our observations
			Map.update_observation()
			mask_T = t.view_mask()
			obs_T = Map.observation
			obs_T[0] = obs_T[1] * mask_T
			obs_T[1] = obs_T[1] * mask_T

			mask_CT = ct.view_mask()
			obs_CT = Map.observation
			obs_CT[0] = obs_CT[1] * mask_CT
			obs_CT[1] = obs_CT[1] * mask_CT
			
			T_frames.append(obs_T)
			T_frames.pop(0) #Remove the first frame
			CT_frames.append(obs_CT)
			CT_frames.pop(0) #Remove the first frame
			
			#Set up current data to be sent to our NN
			T_data = [t.hp/100, 0.0, time_left, 0, t.weapon.clip, t.smoke, plant]
			T_image = data_to_planes(T_data, T_frames)
			T_image = np.array(T_image)
			T_image = np.reshape(T_image, [1, 40, 40, 27])
			
			CT_data = [ct.hp/100, 1.0, time_left, 1, ct.weapon.clip, ct.smoke, plant]
			CT_image = data_to_planes(CT_data, CT_frames)
			CT_image = np.array(CT_image)
			CT_image = np.reshape(CT_image, [1, 40, 40, 27])
			
			
			
			t_new_latent = f_model.predict(T_image)
			ct_new_latent = f_model.predict(CT_image)
			
			#No counterfactual regret min
			action_t.append(t_a1)
			old_latent.append(t_old_latent)
			new_latent.append(t_new_latent)
			
			
			action_t.append(ct_a1)
			old_latent.append(ct_old_latent)
			new_latent.append(ct_new_latent)
			
			
			t_old_latent = t_new_latent
			ct_old_latent = ct_new_latent
			
			if Map.defused:
				time_done = False
				break
				
			if Map.timer == 0: #If bomb explodes
				time_done = False
				ct_won = False

				break
			Map.timer = Map.timer - 1

		reward = 1
		if not ct_won:
			reward = -1
		for j in range(len(target)):
			power = powers[-1] - powers[j]
			coeff = np.power(0.99, power)
			target[j] = (reward * coeff) + target[j]*(1-coeff)
			advantages[j] = (target[j] - advantages[j]) * turn[j]

		lock.acquire()
		PPO_data.append([model_image, target, expert, action_t, old_latent, new_latent, advantages, action])
		lock.release()

if __name__ == "__main__":
	import numpy as np
	import os
	
	#Do not import keras here
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
	from concurrent.futures import ProcessPoolExecutor
	from model import create_model
	import numpy as np
	from tqdm import tqdm
	from multiprocessing import Process, Lock, Manager, Queue
	from helper import save_model

	queue = Queue()
	manager = Manager()
	lock = Lock()


	
	model, fmodel, gmodel, hmodel, trainfn, trainh = create_model(32, resblocks=2)
	workers = 9
	iterations = 100
	epochs = 10

	for it in range(iterations):
		PPO_data = manager.list()
		f_weights = fmodel.get_weights()
		g_weights = gmodel.get_weights()
		h_weights = hmodel.get_weights()

		#Gather data
		print("Gathering data...")
		processes = []
		for _ in range(workers):
			
			p = Process(target=pool_job, args=(f_weights, g_weights, h_weights, PPO_data, lock))
			p.start()
			processes.append(p)
			
			#pool_job(f_weights, g_weights, h_weights, PPO_data, lock)

		for i in tqdm(range(workers)):
			processes[i].join()

		model_image = []
		targ = []
		expert = []
		action_t = []
		old_latent = []
		new_latent = []
		adv = []
		act = []
		print("Loading data...")
		for i in tqdm(range(len(PPO_data))):
			data = PPO_data[i]
			if i == 0:
				model_image = np.reshape(np.array(data[0]), [-1, 40, 40, 27])
				targ = np.reshape(np.array(data[1]), [-1, 1])
				expert = np.reshape(np.array(data[2]), [-1, 63])
				action_t = np.reshape(np.array(data[3]), [-1, 63])
				old_latent = np.reshape(np.array(data[4]), [-1, 50])
				new_latent = np.reshape(np.array(data[5]), [-1, 50])
				adv = np.reshape(np.array(data[6]), [-1, 1])
				act = np.reshape(np.array(data[7]), [-1, 63])
			else:
				model_image = np.vstack((model_image, np.reshape(np.array(data[0]), [-1, 40, 40, 27])))
				targ = np.vstack((targ, np.reshape(np.array(data[1]), [-1, 1])))
				expert = np.vstack((expert, np.reshape(np.array(data[2]), [-1, 63])))
				action_t = np.vstack((action_t, np.reshape(np.array(data[3]), [-1, 63])))
				old_latent = np.vstack((old_latent, np.reshape(np.array(data[4]), [-1, 50])))
				new_latent = np.vstack((new_latent, np.reshape(np.array(data[5]), [-1, 50])))
				adv = np.vstack((adv, np.reshape(np.array(data[6]), [-1, 1])))
				act = np.vstack((act, np.reshape(np.array(data[7]), [-1, 63])))

		splits = int(len(targ)/64) #We want batch size of experience to be ~64
		exp_mi = np.array_split(model_image, splits)
		exp_targ = np.array_split(targ, splits)
		exp_expert = np.array_split(expert, splits)
		exp_adv = np.array_split(adv, splits)
		exp_act = np.array_split(act, splits)
		
		splits2 = int(len(action_t)/64)
		exp_at = np.array_split(action_t, splits2)
		exp_ol = np.array_split(old_latent, splits2)
		exp_nl = np.array_split(new_latent, splits2)

		
		pe = 0
		trans = 0
		mse = 0
		ent = 0
		print("Training...")
		for epoch in tqdm(range(epochs)):
			for k in range(splits):
				loss = trainfn([exp_mi[k], exp_targ[k], exp_expert[k], exp_adv[k], exp_act[k]]) 
				pe += loss[0]
				mse += loss[1]
				ent += loss[2]
			
			for k in range(splits2):
				lossT = trainh([exp_ol[k], exp_at[k], exp_nl[k]])
				trans += lossT[0]
		
		print("Iteration ", it, ": action_policy loss: ", pe/(splits*epochs),
			  " value loss: ", mse/(splits*epochs), "entropy: ", ent/(splits*epochs),
			  " transition: ", trans/splits2, " Game Length: ", len(targ)/(8*workers*2),
			 " Data: ", len(targ))

		if it % 10 == 1:
			save_model(model, "mu_CSGO.json", "mu_weights.h5")
			save_model(fmodel, "mu_CSGOf.json", "mu_weightsf.h5")
			save_model(gmodel, "mu_CSGOg.json", "mu_weightsg.h5")
			save_model(hmodel, "mu_CSGOh.json", "mu_weightsh.h5")

