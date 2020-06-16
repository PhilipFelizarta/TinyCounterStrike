import numpy as np

def muMCTS(observation, turn, f_model, g_model, h_model, simulations=10000, dirichlet=False, alpha=10.0/(63.0)):
	if simulations == 0:
		hidden_state = f_model.predict(observation)
		policy, value = g_model.predict(hidden_state)
		return policy, value, policy
	hidden_state = f_model.predict(observation)
	policy, init_value = g_model.predict(hidden_state)
	init_policy = policy
	new_priors = np.power(policy, 1/2.2)
	policy = new_priors/np.sum(new_priors)
	if dirichlet:
		dirich = np.random.dirichlet([alpha] * 63)
		policy = (policy * 0.75) + (0.25 * dirich)
	root = init_root(hidden_state, policy, turn, init_value)

	for i in range(simulations):
		#print("Simulation: ", i)
		leaf_parent, state, action = root.leaf()
		#Create action one_hot
		action_onehot = np.zeros((1,63))
		action_onehot[0][action] = 1
		new_state = h_model.predict([state, action_onehot]) #Estimate hidden state
		policy, value = g_model.predict(new_state) #Get prediction from hidden state
		new_priors = np.power(policy, 1/2.2)
		policy = new_priors/np.sum(new_priors)
		leaf_parent.expand_backup(action, new_state, policy, np.squeeze(value))

	#val = root.child_Q()[np.argmax(root.child_plays)] #Return the q value of our most visited node

	return root.child_plays/np.sum(root.child_plays), root.child_Q(), init_policy

def init_root(hidden_state, policy, turn, fpu): #Handle root case
	root = Node(None, 0, hidden_state, policy, turn, fpu)
	return root

class Node:
	def __init__(self, parent, index, state, policy, turn, fpu): #On init we need to define the parent and the index the node is in the parent child array
		self.parent = parent
		self.index = index
		self.turn = turn #Boolean to switch our pUCT conditions
		self.policy = policy
		self.state = state

		fpu_red = 0.0
		if not turn:
			fpu_red = 0.0


		self.child_plays = np.zeros([63], dtype=np.int32) #Keep track of how many times our children have played
		self.child_values = np.full([63], np.clip(fpu - fpu_red, -1, 1), dtype=np.float32) #Keep track of the sum of q values

		self.children = [None]*63 #A list of children, there will 1924 of them.. no python chess to tell us less children

	def child_Q(self): #return average Q-values
		return self.child_values / (1 + self.child_plays)

	def child_U(self): #return puct bound
		#DEFINE HYPERPARAMETERS HERE
		c1 = 2.5
		c2 = 19652

		#Define sum of plays among the children
		total_plays = np.sum(self.child_plays)

		u = (c1 + np.log((total_plays + c2 + 1)/c2)) * np.sqrt(total_plays + 1) / (1 + self.child_plays)
		return self.policy * u

	def pUCT_child(self): #Returns state action pair (s', a)
		#print("calc puct")
		if self.turn: #CT
			child_index = np.argmax(self.child_Q() + self.child_U())
			return self.children[child_index], child_index
		else: #T
			child_index = np.argmin(self.child_Q() - self.child_U())
			return self.children[child_index], child_index

	def leaf(self, depth_max=10):
		#print("finding leaf")
		current = self
		parent = self
		depth = 0
		while current is not None:
			parent = current
			current, action = current.pUCT_child()
			depth += 1
		return parent, parent.state, action #Action must be converted to one-hot or other formatting in search function

	def expand_backup(self, index, new_state, policy, value): #Create a child at index with state and policy
		#print("expanding")
		child = Node(self, index, new_state, policy, self.turn, value) #Counterfactual regret
		self.children[index] = child
		self.child_values[index] += value
		self.child_plays[index] += 1
		self.backup(value)

	def backup(self, value):
		#print("backing")
		current = self
		while current.parent != None:
			current.parent.child_values[current.index] += value
			current.parent.child_plays[current.index] += 1
			current = current.parent
