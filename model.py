def create_model(filters, resblocks=5):
	#Create 1v1 NN
	import keras
	import numpy as np

	from keras.models import Model
	from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout
	from keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape
	from keras.optimizers import SGD
	from keras.initializers import glorot_uniform
	from keras.regularizers import l1, l2
	from keras import backend as K

	lambd = 0.001 #L2 regularization

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
		
	image = Input(shape=(20, 20, 28)) #Masked images of last 10 board frames + 7 planes of info
	
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
	fmodel = Model(inputs=image, outputs=latent)
	
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
	gmodel = Model(inputs=prev_latent, outputs=[pol, val]) 
	
	#Create full model
	policy1 = p(latent)
	value = v(latent)
	model = Model(inputs=image, outputs=[policy1, value])
	
	#Create hidden state -> hidden state model (MCTS in LATENT SPACE)
	p_latent = Input(shape=(50,))
	new_policy = Input(shape=(63,))
	concat_input = Concatenate(axis=-1)([new_policy, p_latent])
	new_latent = Dense(50, activation="sigmoid", name="new_latent", kernel_initializer=glorot_uniform(),
				kernel_regularizer = l2(lambd))(concat_input)
	hmodel = Model(inputs=[p_latent, new_policy], outputs=new_latent)
	
	
	#Create updates for h model
	true_latent = K.placeholder(shape=(None, 50), name="truelatent")
	latent_MSE = K.sum(K.square(true_latent - new_latent), axis=1)
	L_h = K.mean(latent_MSE)
	
	opt_h = keras.optimizers.Adam(3e-3)
	updates_h = opt_h.get_updates(params=hmodel.trainable_weights, loss=L_h)
	train_fn_h = K.function(inputs=[hmodel.input[0], hmodel.input[1], true_latent],
						   outputs=[L_h], updates=updates_h)
	
	
	#Online Learning training function... updates the the full, f, g models
	p1 = K.clip(policy1, K.epsilon(), 1)
	
	#placeholder variables for PPO algorithm
	target = K.placeholder(shape=(None,1), name="target_value")
	expert = K.placeholder(shape=(None,63), name="expert_policy")
	
	
	adv = K.placeholder(shape=(None,1), name="adv")
	action =  K.placeholder(shape=(None,63), name="action")
	
	
	MSE = K.sum(K.square(target-value), axis=1)
	
	y_true = K.clip(expert, K.epsilon(), 1)
	r = K.sum(action*p1/y_true, axis=1)
	Lclip = K.minimum(r*adv, K.clip(r, 0.8, 1.2)*adv)
	
	#Lclip = K.sum(y_true * K.log(p1), axis=-1)
	#entropy = 0.0001*K.sum(p1 * K.log(p1), axis=1)
	
	#Lclip = K.sum(action * K.log(p1), axis=-1) * adv #Just reinforce our results
	loss = -K.mean(Lclip) + K.mean(MSE) #+ K.mean(entropy)
	
	c1_print = -K.mean(Lclip)
	v_print = K.mean(MSE)
	#e_print = -0.01*K.mean(entropy)
	
	#optimizer
	opt = keras.optimizers.Adam(1e-4)
	updates = opt.get_updates(params=model.trainable_weights, loss=loss)
	train_fn = K.function(inputs=[model.input, target, expert, adv, action], 
						  outputs=[c1_print, v_print], updates=updates)
	
	return model, fmodel, gmodel, hmodel, train_fn, train_fn_h