from collections import defaultdict
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import get_symbols
from train_encoder import Lang
import random
from collections import Counter
from coq_serapy.contexts import ProofContext
import argparse
import wandb
import coq2vec


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_log', action= 'store_true')
parser.add_argument('--run_name', default= 'Training from Memory')
args = parser.parse_args()
if args.wandb_log :
	wandb.init(project="Proverbot", entity="avarghese", name= "Training from memory")

termvectorizer = coq2vec.CoqTermRNNVectorizer()
termvectorizer.load_weights("data/term2vec-weights-59.dat")


			

class Agent_model(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, device):
		super(Agent_model, self).__init__()
		self.device = device
		

		self.lin1 = nn.Linear(input_size,hidden_size)
		self.lin2 = nn.Linear(hidden_size,hidden_size)
		self.linfinal = nn.Linear(hidden_size,output_size)
		self.softmax = nn.Softmax()
		self.relu = nn.LeakyReLU()
		self.sigmoid  = nn.Sigmoid()

	def network_pass(self,x) :
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.linfinal(x)
		return x
		
	def forward(self, states) : #debug the return shape
		# print("len states",len(states))
		# print(states)
		goals, hyps = list(zip(*states))
		num_hyp_per_obligation = []
		for hyp_list in hyps :
			num_hyp_per_obligation.append(len(hyp_list))

		# print(" === num hyp per Obligation")
		# print(num_hyp_per_obligation)
		goals = torch.tensor(np.array(goals), dtype=torch.float32).to(self.device)
		hyps = np.concatenate(hyps, axis=0)
		hyps = torch.tensor(hyps, dtype=torch.float32).to(self.device)
		# print(" === Goals")
		# print(goals)
		# print("=== Hyps")
		# print(hyps)

		encoded_sum_hyp = torch.zeros_like(goals).to(self.device)
		for i in range(len(num_hyp_per_obligation)) :
			# print("In loop summing from", sum(num_hyp_per_obligation[:i]), "index to", sum(num_hyp_per_obligation[:i]) + num_hyp_per_obligation[i], "index at dim 0")
			encoded_sum_hyp[i,:] = (torch.sum(hyps[ sum(num_hyp_per_obligation[:i]) : sum(num_hyp_per_obligation[:i]) + num_hyp_per_obligation[i] ], dim=0 ))
			# print("Updated encoded sum hyp is", encoded_sum_hyp)
		
		# print(" == Encoded sum hyp ==")
		# print(encoded_sum_hyp)
		# print("encoded sum hyp", encoded_sum_hyp.shape, encoded_sum_hyp.dtype)
		concatenated_tensor= torch.cat( (goals, encoded_sum_hyp) , dim = 1 )
		# print("== Concatenated Tensor ==")
		# print(concatenated_tensor)
		# print("concat shape",concatenated_tensor.shape, concatenated_tensor.dtype)
		# print(concatenated_tensor)
		regressor_output = self.network_pass(concatenated_tensor)
		
		# print(regressor_output)
		softmaxed = self.sigmoid(regressor_output)
		# print(softmaxed)
		# print("final state shape",softmaxed.shape)
		return softmaxed

# a = Agent_model(4,1,1,'cpu')
# point = [ [[1,1], [[1,1],[1,1],[1,1]]], [[2,2], [[2,2],[2,2],[2,2],[2,2]]],  [[3,3], [[3,3], [3,3],[3,3]]] ]
# print(a(point))

# quit()
def context_to_state_vec(proof_context) :
	# print(type(proof_context))
	obligation = proof_context.fg_goals[0]
	goal_state_sentence = obligation.goal.strip()
	goal_vector = np.array(termvectorizer.term_to_vector(goal_state_sentence)).flatten()

	all_hyp_vectors = []
	for hyp in obligation.hypotheses :
		hyp_sentence =  hyp.strip()
		hyp_vector =  np.array(termvectorizer.term_to_vector(hyp_sentence)).flatten()
		all_hyp_vectors.append(hyp_vector)
	
	all_hyp_vectors.append(  np.array(termvectorizer.term_to_vector(":")).flatten() )
	return [goal_vector, all_hyp_vectors]





def context_to_name(proof_context) :
	list_of_obligation_names = []
	for obl  in proof_context.fg_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)
	
	for obl  in proof_context.bg_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)

	for obl  in proof_context.shelved_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)
	
	for obl  in proof_context.given_up_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)

	list_of_obligation_names.sort()
	
	return " ".join(list_of_obligation_names)


class Memory_graph :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = {}
		self.add_called = 0
		self.reward_map = { -1 : [], 0 : [], 1 : []}
		self.forward_index_dict = {}
		self.backward_index_dict = {}
				
	def add(self, s_contexts, rewards, next_state_contexts) :
		self.add_called += 1
		if args.wandb_log :
			wandb.log({"Num times add called" : self.add_called})
		# print("Who shaoll say ....", s_text, G)
		# print("debug",(s_text in self.index_dict, s_text, self.index_dict))
		
		indexes = []
		next_state_vectors_indexes = []
		end_reward = rewards[-1]
		for i in range(len(s_contexts)) :
			s_vector = context_to_state_vec(s_contexts[i])
			s_text = context_to_name(s_contexts[i])
			print("index", i, "State text ~=", s_text)
			if s_text in self.index_dict :
				j = self.index_dict[s_text]
				indexes.append(j)
				print("Found", i, "in memory with index", j, "and vval", self.mem[j][1])
			else :
				self.index_dict[s_text] = self.num_items
				self.forward_index_dict[self.num_items] = set()
				self.backward_index_dict[self.num_items] = set()
				self.mem.append([s_vector,0])
				self.reward_map[0].append(self.num_items)
				indexes.append(self.num_items)
				print("Added", i, "in memory with index", self.num_items, "and vval", 0)
				self.num_items += 1

			curr_next_state_vectors_indexes = []
			print("Looping over next states")
			for j in range(len(next_state_contexts[i])) :
				curr_next_state_vector =  context_to_state_vec(next_state_contexts[i][j])
				curr_next_state_vector_text = context_to_name(next_state_contexts[i][j])
				print("Next state index", j, "State text ~=", curr_next_state_vector_text)
				if curr_next_state_vector_text in self.index_dict :
					k = self.index_dict[curr_next_state_vector_text]
					curr_next_state_vectors_indexes.append(k)
					print("Found", j, "in memory with index", k, "and vval", self.mem[k][1])
				else :
					self.index_dict[curr_next_state_vector_text] = self.num_items
					self.forward_index_dict[self.num_items] = set()
					self.backward_index_dict[self.num_items] = set()
					self.mem.append([curr_next_state_vector,0])
					self.reward_map[0].append(self.num_items)
					curr_next_state_vectors_indexes.append(self.num_items)
					print("Added", i, "in memory with index", self.num_items, "and vval", 0)
					self.num_items += 1
			next_state_vectors_indexes.append(curr_next_state_vectors_indexes)
			print("Finished looping over next states")

		assert len(indexes) == len(s_contexts) == len(next_state_vectors_indexes)
		for i in range(1,len(indexes)) :
			prev_ind = indexes[i-1]
			now_ind = indexes[i]
			self.forward_index_dict[prev_ind].add(now_ind)
			self.backward_index_dict[now_ind].add(prev_ind)

			for j in range(len(next_state_vectors_indexes[i-1])) :
				curr_next_state_vector_ind = next_state_vectors_indexes[i-1][j]
				self.forward_index_dict[prev_ind].add(curr_next_state_vector_ind)
				self.backward_index_dict[curr_next_state_vector_ind].add(prev_ind)
		
		for ind in indexes :
			print(ind, self.forward_index_dict[ind],self.backward_index_dict[ind])
		print("^^^^^")
		last_ind = indexes[-1]
		curr_last_ind_vval = self.mem[last_ind][1]
		self.mem[last_ind][1] = end_reward
		self.reward_map[curr_last_ind_vval].remove(last_ind)
		self.reward_map[end_reward].append(last_ind)

		
		if curr_last_ind_vval != 0 and curr_last_ind_vval != end_reward :
			print(curr_last_ind_vval,end_reward)
			raise ValueError("Last Vval Changed when not 0")

		self.curr_visited_while_fixing = []
		print("Starting fix graph")
		self.fix_graph(last_ind)
		print("Finished fix graph")


	def fix_graph(self, ind) :
		print(ind, self.forward_index_dict[ind],self.backward_index_dict[ind])
		# if ind in self.curr_visited_while_fixing :
		# 	return
		# 	raise ValueError("Loops in Proof graph")
		self.curr_visited_while_fixing.append(ind)

		forward_vvals = []
		for forward_ind in self.forward_index_dict[ind] :
			forward_vvals.append( self.mem[forward_ind][1] )
		
		if len(forward_vvals) > 0 :
			currvval = self.mem[ind][1] 
			maxforwardvval = max(forward_vvals)
			self.reward_map[currvval].remove(ind)
			self.reward_map[maxforwardvval].append(ind)
			self.mem[ind][1] = maxforwardvval

		for backward_ind in self.backward_index_dict[ind] :
			self.fix_graph(backward_ind)

	def change_value(self, index, new_value) :
		print("Updating index", index, "with Old Value -", self.mem[index][1], "to New Value -", new_value)
		assert self.mem[index][1] != -new_value
		if new_value == 0 :
			assert self.mem[index][1] == 0

		self.mem[index][1] = new_value

	def clear(self) :
		self.mem = []
		self.num_items = 0
	
	def sample_random_minibatch(self,n = None) :
		if n :
			mem_batch = random.sample(self.mem,n)
		else :
			mem_batch = list(self.mem)
			random.shuffle(mem_batch)
		return mem_batch
	
	def sample_random_nonzero_minibatch(self, n = None) :
		numpymem = np.array(self.mem)
		positive_sample = list(numpymem[self.reward_map[1] ])
		negative_sample = list(numpymem[self.reward_map[-1] ])
		assert len(self.reward_map[1]) + len(self.reward_map[-1]) == len(set( list(self.reward_map[1]) + list(self.reward_map[-1])))
		mem_batch = positive_sample + negative_sample
		if n :
			mem_batch = random.sample(mem_batch,n)
		else :
			random.shuffle(mem_batch)
		return mem_batch

	def sample_balanced(self) :
		# print(len(self.reward_map[-1]), len(self.reward_map[1]))
		sample_size = min(len(self.reward_map[-1]), len(self.reward_map[1]))
		numpymem = np.array(self.mem)
		positive_sample = list(numpymem[ random.sample(self.reward_map[1], sample_size) ])
		negative_sample = list(numpymem[  random.sample(self.reward_map[-1], sample_size)  ])
		return_list = positive_sample + negative_sample
		random.shuffle(return_list)
		return return_list

	@property
	def num_nonzero_items(self):
		return  len(self.reward_map[-1]) +  len(self.reward_map[1])









def indexesFromSentence(lang, sentence, ignore_missing = False):
    if ignore_missing :
        to_ret = []
        for word in list(sentence) :
            if word in lang.char2index :
                to_ret.append(lang.char2index[word])
        return to_ret
    else :
        return [lang.char2index[word] for word in list(sentence)]









# with open("data/agent_state_vval.pkl", 'rb') as f:
#     agent_model_mem = pickle.load(f)
with open("data/memory_graph_obligation_c2v.pkl","rb") as f:
	memory = pickle.load(f)

# X = []
# Y = []
# for x,y in agent_model_mem.items() :
# 	X.append(preprocess_state(x))
# 	Y.append(y)

batch = memory.sample_random_nonzero_minibatch() #sample_balanced()
X, Y = zip(*batch)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

newX = []
newY = []
for i in range(len(Y)) :
	if Y[i] in (-1,1) :
		newX.append(X[i])
		newY.append(Y[i])

X = np.array(newX)
Y= np.array(newY)
Y[Y==-1] = 0
classes = defaultdict(lambda : 0)
for i in Y :
	classes[i] +=1 
print(classes)




X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.1)

# print(set(Y))
# print(sum(Y))
# print(len(X))
# print(X[0], Y[0])
# print(list(Y))
# quit()


def train_step( agent_model ,optimizer, loss_object, X_train,y_train, batch_size = 50, device="cpu") :

	random_index0 = np.random.choice(np.flatnonzero(y_train == 0), size = batch_size//2 )
	random_index1 = np.random.choice(np.flatnonzero(y_train == 1), size = batch_size//2)
	random_index = np.concatenate((random_index0,random_index1), axis = None)
	np.random.shuffle(random_index)
	states = X_train[random_index]
	rewards = y_train[random_index]
	
	# print(states)
	# state_tensor = torch.tensor(states).to(device)
	reward_tensor = torch.tensor(rewards,dtype = torch.float).to(device)


	optimizer.zero_grad()
	predicted_rewards = agent_model(states).squeeze()

	# print("Shapes - predicted, reward_tensor", predicted_rewards.shape, reward_tensor.shape)
	# quit()

	loss = loss_object(predicted_rewards,reward_tensor)
	loss.backward()
	optimizer.step()
	return loss.item()




device = "cuda"

agent_model = Agent_model( 1565 * 2, 2000,1, device).to(device)
loss_object = nn.BCELoss()
optimizer = optim.SGD(agent_model.parameters(),lr = 0.01)
main_batch_size = 200
batch_size = min(main_batch_size,len(X_train))
for _ in range(300) :
	loss = train_step( agent_model ,optimizer, loss_object, X_train,y_train,batch_size = batch_size, device = device)
	if args.wandb_log :
		wandb.log({"Loss":loss})
	if _%100 == 0 :
		print(loss)


print("Train Classification Report")
print(X_train.shape)
y_train_pred = []
for X_sample in np.array_split(X_train, len(X_train)//batch_size) :
	curr_pred = agent_model(X_sample)
	curr_pred =  curr_pred.cpu().detach().numpy().flatten() > 0.5
	y_train_pred += list(curr_pred)

y_train_pred = np.array(y_train_pred)
print(y_train_pred.shape)
print(classification_report(y_train,y_train_pred))

print("Test Classification Report")
print(X_test.shape)
y_test_pred = []
batch_size = min(main_batch_size,len(X_test))
for X_sample in np.array_split(X_test, len(X_test)//batch_size) :
	curr_pred = agent_model(X_sample)
	curr_pred =  curr_pred.cpu().detach().numpy().flatten() > 0.5
	y_test_pred += list(curr_pred)

y_test_pred = np.array(y_test_pred)
print(y_test_pred.shape)
print(classification_report(y_test,y_test_pred))


torch.save(agent_model, "data/agent_model_obligation_c2v.torch")

del agent_model
agent_model = torch.load("data/agent_model_obligation_c2v.torch")
agent_model.eval()

print("Post reload Test Classification Report")
print(X_test.shape)
y_test_pred_raw = []
batch_size = min(main_batch_size,len(X_test))
for X_sample in np.array_split(X_test, len(X_test)//batch_size) :
	curr_pred = agent_model(X_sample)
	curr_pred =  curr_pred.cpu().detach().numpy().flatten()
	y_test_pred_raw += list(curr_pred)

y_test_pred = (np.array(y_test_pred_raw) > 0.5 ).astype(int)
print(y_test_pred.shape)
print(y_test_pred_raw)
print(y_test_pred)
print(y_test)
print(classification_report(y_test,y_test_pred))



print("Manual test on everything in memory")
target_y = []
raw_y = []
pred_y = []
indices = []
X_temp = []
Y_temp = []
for name in memory.index_dict :
	i = memory.index_dict[name]
	if memory.mem[i][1] == 0 :
		continue
	else :
		assert abs(memory.mem[i][1]) == 1
		vec = memory.mem[i][0]
		X_temp.append(vec)
		Y_temp.append(memory.mem[i][1])
		prediction = agent_model( np.array([vec]) ).cpu().detach().numpy().flatten()[0]
		# print("For index",i," = ", prediction, '(',? memory.mem[i][1],')')
		# print(prediction)
		raw_y.append(prediction)
		pred_y.append(int(prediction > 0.5))
		target_y.append( memory.mem[i][1])
		indices.append(i)
		# quit()

target_y = np.array(target_y)
target_y[ target_y == -1] = 0
target_y = list(target_y)
# print(raw_y)
# print(pred_y)
# print(target_y)
print(classification_report(target_y,pred_y))

compare = np.array(sorted(memory.reward_map[-1] + memory.reward_map[1]))
indices = np.array(indices)

print(len(compare) == len(indices) and np.all(compare == indices))


print("Full training set Classification Report")
y_pred = []
for X_sample in np.array_split(X, len(X)//batch_size) :
	curr_pred = agent_model(X_sample)
	curr_pred =  curr_pred.cpu().detach().numpy().flatten() > 0.5
	y_pred += list(curr_pred)
y_pred = np.array(y_pred)
print(classification_report(Y,y_pred))


print("New memory generated samples Classification Report")
X_temp = np.array(X_temp)
Y_temp = np.array(Y_temp)
Y_temp[ Y_temp == -1] = 0
y_pred = []
y_raw = []
for X_sample in np.array_split(X_temp, len(X_temp)//batch_size) :
	curr_pred = agent_model(X_sample)
	curr_pred =  curr_pred.cpu().detach().numpy().flatten() 
	y_raw += list(curr_pred)
	# print(curr_pred)
	curr_pred = curr_pred > 0.5
	y_pred += list(curr_pred)
y_pred = np.array(y_pred)
print(classification_report(Y_temp,y_pred))

print(len(X) == len(X_temp))
# print(y_raw)


print("New memory generated samples Classification Report but one by one")
X_temp = np.array(X_temp)
Y_temp = np.array(Y_temp)
Y_temp[ Y_temp == -1] = 0
y_pred = []
y_raw = []
for X_sample in X_temp :
	curr_pred = agent_model([X_sample])
	curr_pred =  curr_pred.cpu().detach().numpy().flatten() 
	y_raw += list(curr_pred)
	curr_pred = curr_pred > 0.5
	y_pred += list(curr_pred)
y_pred = np.array(y_pred)
print(classification_report(Y_temp,y_pred))