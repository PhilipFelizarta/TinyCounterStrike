import numpy as np

def muMCTS(observation, turn, f_model, g_model, h_model, simulations=10000):
	if simulations == 0:
		hidden_state = f_model.predict(observation)
		policy, value = g_model.predict(hidden_state)
		return policy, value
	hidden_state = f_model.predict(observation)
	policy, _ = g_model.predict(hidden_state)
	root = init_root(hidden_state, policy, turn)

	for i in range(simulations):
		#print("Simulation: ", i)
		leaf_parent, state, action = root.leaf()
		#Create action one_hot
		action_onehot = np.zeros((1,63))
		action_onehot[0][action] = 1
		new_state = h_model.predict([state, action_onehot]) #Estimate hidden state
		policy, value = g_model.predict(new_state) #Get prediction from hidden state
		leaf_parent.expand_backup(action, new_state, policy, np.squeeze(value))

	return root.child_plays/np.sum(root.child_plays), np.mean(root.child_Q())

def init_root(hidden_state, policy, turn): #Handle root case
	root = Node(None, 0, hidden_state, policy, turn)
	return root

class Node:
	def __init__(self, parent, index, state, policy, turn): #On init we need to define the parent and the index the node is in the parent child array
		self.parent = parent
		self.index = index
		self.turn = turn #Boolean to switch our pUCT conditions
		self.policy = policy
		self.state = state

		self.child_plays = np.zeros([63], dtype=np.int32) #Keep track of how many times our children have played
		self.child_values = np.zeros([63], dtype=np.float32) #Keep track of the sum of q values

		self.children = [None]*63 #A list of children, there will 1924 of them.. no python chess to tell us less children

	def child_Q(self): #return average Q-values
		return self.child_values / (1 + self.child_plays)

	def child_U(self): #return puct bound
		#DEFINE HYPERPARAMETERS HERE
		c1 = 1.2
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
		while current is not None and depth <= depth_max:
			parent = current
			current, action = self.pUCT_child()
			depth += 1
		return parent, parent.state, action #Action must be converted to one-hot or other formatting in search function

	def expand_backup(self, index, new_state, policy, value): #Create a child at index with state and policy
		#print("expanding")
		child = Node(self, index, new_state, policy, self.turn) #Counterfactual regret
		self.children[index] = child
		self.child_values[index] = value
		self.child_plays[index] += 1
		self.backup(value)

	def backup(self, value):
		#print("backing")
		current = self
		while current.parent != None:
			current.parent.child_values[self.index] += value
			current.parent.child_plays[self.index] += 1
			current = current.parent
