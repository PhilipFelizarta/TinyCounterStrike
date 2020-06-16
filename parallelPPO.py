import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def pool_job(f_model_weights, g_model_weights, h_model_weights, PPO_data, g):
	import keras
	import numpy as np

	from keras.models import Model
	from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout, Lambda
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
	resblocks = 10
	filters = 32

	class Invert(keras.layers.Layer):
		def __init__(self):
			super(Invert, self).__init__()

		def call(self, inputs):
			one = K.ones(1)
			return one - inputs

	def stem(X, filters, stage="stem", size=3):
		stem = Conv2D(filters=filters, kernel_size=(size,size), strides=(2,2), padding='valid', data_format='channels_last', 
					  name='Conv_' + stage, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		stem = BatchNormalization(axis=-1)(stem)
		stem = Activation('relu')(stem)
		return stem

	def res_block(X, filters, block, size=3):
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block1_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		res = BatchNormalization(axis=-1)(res)
		res = Activation('relu')(res)
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block2_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(res)
		
		X = Add()([X, res])
		X = BatchNormalization(axis=-1)(X)
		X = Activation('relu')(X)
		return X

	image = Input(shape=(20, 20, 28)) #Masked images of last 10 board frames + 6 planes of info
	
	#Stem for our resnet
	X = stem(image, filters) #Resolution 10x10x
	
	#Resnet
	for i in range(resblocks):
		X = res_block(X, filters, str(i + 1))
	
	#Create latent representation
	latent = stem(X, 32, stage="latent", size=1)
	latent = Flatten()(latent)
	latent = Dense(50, activation="tanh", name="latentspace", kernel_initializer=glorot_uniform(),
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
	
	concat_1 = Concatenate(axis=-1)([p_latent, new_policy])

	#Update Gate
	z = Dense(50, activation="sigmoid")(concat_1)

	#Reset Gate
	r = Dense(50, activation="sigmoid")(concat_1)
	r = Multiply()([r, p_latent])

	#Output gate
	sub = Invert()(z)
	new_latent = Multiply()([sub, p_latent])

	concat_2 = Concatenate(axis=-1)([r, new_policy])
	h = Dense(50, activation="tanh")(concat_2)
	h = Multiply()([h, z])
	
	new_latent = Add()([new_latent, h])

	h_model = Model(inputs=[p_latent, new_policy], outputs=new_latent)

	f_model.set_weights(f_model_weights)
	g_model.set_weights(g_model_weights)
	h_model.set_weights(h_model_weights)

	#Import the game environment from environ.py
	from environ import Weapon, Board, Player, Box, Team, Direction
	from helper import to_Direction, data_to_planes
	from MCTS import muMCTS
	for _ in range(g):
		frames = 100

		ak = Weapon(50, 1, 30, 1.0, 100)
		awp = Weapon(50, 1, 30, 1.0, 100)
		Map = Board()
		t_spawn = 0
		ct_spawn = 0
		t = Player(t_spawn, 9, Team.T, ak, 0, Map)
		ct = Player(ct_spawn, -9, Team.CT, awp, 0, Map)

		#Mid
		Mid = Box(5.0, 16.0, 0.0, 0.0, Map)
		
		#A site
		Asite = Box(4.0, 3.0, 6.0, 6.5, Map)
		Asite = Box(2.5, 3.0, 7.0, -3.5, Map)
		
		#Bsite
		Bsite = Box(5.0, 5.0, -5.0, -1.0, Map)
		

		Map.init_observation()
		T_frames = [] #Will hold the last 10 frames
		CT_frames = [] #Will hold the last 10 frames

		mask_T = t.view_mask()
		obs_T = np.copy(Map.observation)
		obs_T[0] = obs_T[0] * mask_T

		mask_CT = ct.view_mask()
		obs_CT = np.copy(Map.observation)
		obs_CT[0] = obs_CT[0] * mask_CT

		#[model.input, target, expert, action_t, old_latent, new_latent]
		model_image = []
		target = []
		advantages = []
		action = []
		expert = []
		p_opp = []

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
		#hp, wep, time, team, clip, smoke, plant
		plant = False
		T_data = [t.hp/100, 0.0, 1.0, Team.T, t.weapon.clip, t.smoke, plant, t.view]
		T_image = data_to_planes(T_data, T_frames)
		T_image = np.array(T_image)
		T_image = np.reshape(T_image, [1, 20, 20, 28])

		CT_data = [ct.hp/100, 0.0, 1.0, Team.CT, ct.weapon.clip, ct.smoke, plant, ct.view]
		CT_image = data_to_planes(CT_data, CT_frames)
		CT_image = np.array(CT_image)
		CT_image = np.reshape(CT_image, [1, 20, 20, 28])

		t_old_latent = f_model.predict(T_image)
		ct_old_latent = f_model.predict(CT_image)

		#Game loop
		for time in range(frames):
			final_time = time + 1
			time_left = (frames - time)/(frames) #Time left scaled from 0 to 1
			if plant:
				time_left = np.minimum(time_left, Map.timer/10)
				
			t_expert, t_value, t_policy = muMCTS(T_image, False, f_model, g_model, h_model, simulations=0)
			ct_expert, ct_value, ct_policy = muMCTS(CT_image, True, f_model, g_model, h_model, simulations=0)
			
			#T should have slight advantage in game loop processing since CT wins
			#by time
			direction, t_angle, t_a1 = to_Direction(t_expert, temp=1.0)
			t.set_view(t_angle) #Implement our view policy
			
			#Add T data to PPO lists
			r = np.clip((ct.hp - t.hp)/100 - Map.dist_to_site(t.x, t.y), -1, 1)
			model_image.append(T_image)
			target.append(r)
			advantages.append(np.sum(t_value*t_a1))
			expert.append(t_expert)
			action.append(t_a1)
			
			powers.append(time)
			turn.append(-1)
			

			if direction == -1:
				t.fire()
			elif direction == -2:
				planted = t.plant()
				if planted:
					plant = True
					Map.timer = 10
			elif direction == -3:
				t.fire_smoke()
			else:
				t.move(direction)
			
				
			#Process CT
			direction, ct_angle, ct_a1 = to_Direction(ct_expert, temp=1.0)
			ct.set_view(ct_angle) #Implement our view policy
			
			#Add CT data to PPO lists
			r = np.clip((ct.hp - t.hp)/100, -1, 1)
			model_image.append(CT_image)
			target.append(r)
			advantages.append(np.sum(ct_value*ct_a1))
			expert.append(ct_expert)
			action.append(ct_a1)

			#Probability of information set/advantage
			p_opp.append(np.sum(ct_a1*ct_expert))
			p_opp.append(np.sum(t_a1*t_expert))
			
			powers.append(time)
			turn.append(1)

			if ct.hp <= 0:
				time_done= False
				ct_won = False
				break
			
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
			Map.init_observation()
			mask_T = t.view_mask()
			obs_T = np.copy(Map.observation)
			obs_T[0] = obs_T[0] * mask_T

			mask_CT = ct.view_mask()
			obs_CT = np.copy(Map.observation)
			obs_CT[0] = obs_CT[0] * mask_CT
			
			T_frames.append(obs_T)
			T_frames.pop(0) #Remove the first frame
			CT_frames.append(obs_CT)
			CT_frames.pop(0) #Remove the first frame
			
			#Set up current data to be sent to our NN
			#hp, wep, time, team, clip, smoke, plant
			T_data = [t.hp/100, 0.0, time_left, Team.T, t.weapon.clip, t.smoke, plant, t.view]
			T_image = data_to_planes(T_data, T_frames)
			T_image = np.array(T_image)
			T_image = np.reshape(T_image, [1, 20, 20, 28])
			
			CT_data = [ct.hp/100, 0.0, time_left, Team.CT, ct.weapon.clip, ct.smoke, plant, ct.view]
			CT_image = data_to_planes(CT_data, CT_frames)
			CT_image = np.array(CT_image)
			CT_image = np.reshape(CT_image, [1, 20, 20, 28])
			
			
			
			t_new_latent = f_model.predict(T_image)
			ct_new_latent = f_model.predict(CT_image)
			
			#counterfactual regret min
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
			power = len(target) - 1 - j
			coeff = np.power(0.95, power)
			target[j] = np.clip((reward * coeff) + (1 - coeff)*target[j], -1, 1)

			advantages[j] = (target[j] - advantages[j]) * turn[j]

		PPO_data.append([model_image, target, expert, action, old_latent, new_latent, advantages, action_t])

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

	def shuffle_data(a, b, c, d, e, f, z, y):
		import numpy as np
		a_s = np.copy(a)
		b_s = np.copy(b)
		c_s = np.copy(c)
		d_s = np.copy(d)
		e_s = np.copy(e)
		z_s = np.copy(z)
		f_s = np.copy(f)
		y_s = np.copy(y)

		rand_state = np.random.get_state()
		np.random.shuffle(a_s)
		np.random.set_state(rand_state)
		np.random.shuffle(b_s)
		np.random.set_state(rand_state)
		np.random.shuffle(c_s)
		np.random.set_state(rand_state)
		np.random.shuffle(d_s)
		np.random.set_state(rand_state)
		np.random.shuffle(e_s)
		np.random.set_state(rand_state)
		np.random.shuffle(z_s)
		np.random.set_state(rand_state)
		np.random.shuffle(y_s)
		np.random.set_state(rand_state)
		np.random.shuffle(f_s)


		return a_s, b_s, c_s, d_s, e_s, f_s, z_s, y_s
	
	def load_model(modelFile, weightFile, update=True): #load model from json and HDF5
		import keras
		from keras.models import model_from_json


		json_file = open(modelFile, 'r')
		load_model_json = json_file.read()
		json_file.close()
		if not update:
			load_model = model_from_json(load_model_json)
		else:
			_, _, _, load_model, _, _ = create_model(32, resblocks=10)
			load_model.summary()
		load_model.load_weights(weightFile)
		print("Model Loaded!")
		return load_model
	model, fmodel, gmodel, hmodel, trainfn, trainh = create_model(32, resblocks=10)

	#mod = load_model("PPO_CSGO.json", "PPO_weights.h5", update=False)
	#fmod = load_model("PPO_CSGOf.json", "PPO_weightsf.h5", update=False)
	#gmod = load_model("PPO_CSGOg.json", "PPO_weightsg.h5", update=False)
	#hmod = load_model("PPO_CSGOh.json", "PPO_weightsh.h5", update=True)

	#model.set_weights(mod.get_weights())
	#fmodel.set_weights(fmod.get_weights())
	#gmodel.set_weights(gmod.get_weights())
	#hmodel.set_weights(hmod.get_weights())

	workers = 5
	games = 10
	iterations = 1002
	epochs = 10
	model_image = []
	targ = []
	expert = []
	action_t = []
	old_latent = []
	new_latent = []
	adv = []
	act = []

	n = 0
	l = 0
	for it in range(iterations):
		PPO_data = manager.list()
		f_weights = fmodel.get_weights()
		g_weights = gmodel.get_weights()
		h_weights = hmodel.get_weights()

		#Gather data
		print("Gathering data...")
		processes = []
		for _ in range(workers):
			p = Process(target=pool_job, args=(f_weights, g_weights, h_weights, PPO_data, games))
			p.start()
			processes.append(p)
			
			#pool_job(f_weights, g_weights, h_weights, PPO_data, lock)

		for i in tqdm(range(workers)):
			processes[i].join()

		
		print("Loading data...")
		PPO_data = list(PPO_data)
		for i in tqdm(range(len(PPO_data))):
			data = PPO_data.pop(0)
			if i == 0:
				model_image = np.reshape(np.array(data.pop(0)), [-1, 20, 20, 28])
				targ = np.reshape(np.array(data.pop(0)), [-1, 1])
				expert = np.reshape(np.array(data.pop(0)), [-1, 63])
				action_t = np.reshape(np.array(data.pop(0)), [-1, 63])
				old_latent = np.reshape(np.array(data.pop(0)), [-1, 50])
				new_latent = np.reshape(np.array(data.pop(0)), [-1, 50])
				adv = np.reshape(np.array(data.pop(0)), [-1, 1])
				act = np.reshape(np.array(data.pop(0)), [-1, 63])
			else:
				model_image = np.append(model_image, np.reshape(np.array(data.pop(0)), [-1, 20, 20, 28]), axis=0)
				targ = np.append(targ, np.reshape(np.array(data.pop(0)), [-1, 1]), axis=0)
				expert = np.append(expert, np.reshape(np.array(data.pop(0)), [-1, 63]), axis=0)
				action_t = np.append(action_t, np.reshape(np.array(data.pop(0)), [-1, 63]), axis=0)
				old_latent = np.append(old_latent, np.reshape(np.array(data.pop(0)), [-1, 50]), axis=0)
				new_latent = np.append(new_latent, np.reshape(np.array(data.pop(0)), [-1, 50]), axis=0)
				adv = np.append(adv, np.reshape(np.array(data.pop(0)), [-1, 1]), axis=0)
				act = np.append(act, np.reshape(np.array(data.pop(0)), [-1, 63]), axis=0)

		n = len(targ) - l
		if len(targ) > 250000:
			m = len(targ) - 250000
			model_image = model_image[m:]
			targ = targ[m:]
			expert = expert[m:]
			action_t = action_t[m:]
			old_latent = old_latent[m:]
			new_latent = new_latent[m:]
			adv = adv[m:]
			act = act[m:]
		#l = len(targ)

		mi, ta, e, ad, at_t, at, ol, nl = shuffle_data(model_image, targ, expert, adv, action_t, act, old_latent, new_latent)
		print("Mixup...") #Double the size of the dataset using mixup
		alpha = 4.0
		lam = np.transpose(np.random.beta(alpha, alpha, size=len(targ)))
		mix_mi = np.transpose(lam * np.transpose(mi) + (1 - lam) * np.transpose(model_image))
		mix_ta = np.transpose(lam * np.transpose(ta) + (1 - lam) * np.transpose(targ))
		mix_e = np.transpose(lam * np.transpose(e) + (1 - lam) * np.transpose(expert))
		mix_adv = np.transpose(lam * np.transpose(ad) + (1 - lam) * np.transpose(adv))
		mix_act = np.transpose(lam * np.transpose(at_t) + (1 - lam) * np.transpose(action_t))

		lam = np.transpose(np.random.beta(alpha, alpha, size=len(act)))
		mix_at = np.transpose(lam * np.transpose(at) + (1 - lam) * np.transpose(act))
		mix_ol = np.transpose(lam * np.transpose(ol) + (1 - lam) * np.transpose(old_latent))
		mix_nl = np.transpose(lam * np.transpose(nl) + (1 - lam) * np.transpose(new_latent))

		mi = np.append(mi, mix_mi, axis=0)
		ta = np.append(ta, mix_ta, axis=0)
		e = np.append(e, mix_e, axis=0)
		ad = np.append(ad, mix_adv, axis=0)
		at_t = np.append(at_t, mix_act, axis=0)
		at = np.append(at, mix_at, axis=0)
		ol = np.append(ol, mix_ol, axis=0)
		nl = np.append(nl, mix_nl, axis=0)

		splits = int(len(ta)/128) #We want batch size of experience to be ~64
		exp_mi = np.array_split(mi, splits)
		exp_targ = np.array_split(ta, splits)
		exp_expert = np.array_split(e, splits)
		exp_adv = np.array_split(ad, splits)
		exp_act = np.array_split(at_t, splits)
		
		splits2 = int(len(at)/128)
		exp_at = np.array_split(at, splits2)
		exp_ol = np.array_split(ol, splits2)
		exp_nl = np.array_split(nl, splits2)

		

		
		pe = 0
		trans = 0
		mse = 0
		print("Training...")
		for epoch in tqdm(range(epochs)):
			for k in range(splits):
				loss = trainfn([exp_mi[k], exp_targ[k], exp_expert[k], exp_adv[k], exp_act[k]]) 
				pe += loss[0]
				mse += loss[1]
			
			for k in range(splits2):
				lossT = trainh([exp_ol[k], exp_at[k], exp_nl[k]])
				trans += lossT[0]
		
		print("Iteration ", it, ": action_policy loss: ", pe/(splits*epochs),
			  " value loss: ", mse/(splits*epochs)," transition: ", 
			  trans/(splits2*epochs), " Game Length: ", n/(games*workers*2),
			 " Data: ", len(ta))


		if it % 10 == 1:
			save_model(model, "PPO_CSGO.json", "PPO_weights.h5")
			save_model(fmodel, "PPO_CSGOf.json", "PPO_weightsf.h5")
			save_model(gmodel, "PPO_CSGOg.json", "PPO_weightsg.h5")
			save_model(hmodel, "PPO_CSGOh.json", "PPO_weightsh.h5")

