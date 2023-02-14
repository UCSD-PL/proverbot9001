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


MAX_STATE_SYMBOLS = 40
with open("data/encoder_language_symbols.pkl","rb") as f:
	language_model = pickle.load(f)


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True)

	def forward(self, input, hidden, cell):
		embedded = self.embedding(input)
		output, (hidden, cell) = self.lstm(embedded, (hidden,cell))
		return output, hidden, cell

	def initHidden(self,device,batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, device=device)
	
	def initCell(self,device,batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, device=device)


class RegressorNN(nn.Module) :
	def __init__(self, input_size,output_size,hidden_size=200) :
		super(RegressorNN,self).__init__()
		self.lin1 = nn.Linear(input_size,hidden_size)
		self.lin2 = nn.Linear(hidden_size,hidden_size)

		self.linfinal = nn.Linear(hidden_size,output_size)
		self.softmax = nn.Softmax()
		self.relu = nn.LeakyReLU()
		# self.apply(self.init_weights)
		
	def forward(self,x) :
		x = self.relu(self.lin1(x))
		# x = self.relu(self.lin2(x))
		x = self.linfinal(x)
		return x		

class Agent_model(nn.Module):
	def __init__(self, dictionary_size, hidden_size, output_size, device):
		super(Agent_model, self).__init__()
		self.encoder = EncoderRNN(dictionary_size,hidden_size)
		self.regressor = RegressorNN(hidden_size,output_size)
		self.device = device

	
	def forward(self,input_tensor) :
		encoder_output, encoder_hidden,encoder_cell = self.encoder(input_tensor, self.encoder.initHidden(self.device,input_tensor.shape[0]), self.encoder.initCell(self.device, input_tensor.shape[0]))
		# print("EH",encoder_hidden.shape)
		encoder_hidden = torch.squeeze(encoder_hidden)
		regressor_output = self.regressor(encoder_hidden)
		return regressor_output


class Memory_max :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = {}
		self.add_called = 0
				
	def add(self,s_vector ,s_text, G) :
		# print("Who shaoll say ....", s_text, G)
		# print("debug",(s_text in self.index_dict, s_text, self.index_dict))
		self.add_called += 1
		if s_text in self.index_dict :
			i = self.index_dict[s_text]
			self.mem[i][1] = max(G, self.mem[i][1]) #Index 1 is the reward, 0 is the vector
		
		else :
			self.index_dict[s_text] = self.num_items
			self.mem.append([s_vector,G])
			self.num_items += 1


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





def indexesFromSentence(lang, sentence, ignore_missing = False):
    if ignore_missing :
        to_ret = []
        for word in list(sentence) :
            if word in lang.char2index :
                to_ret.append(lang.char2index[word])
        return to_ret
    else :
        return [lang.char2index[word] for word in list(sentence)]


def preprocess_state(state_text) :
	#make this cleaner in association with select action and get all states, especially checking if state is string part
	state_sentence = get_symbols(state_text)
	# print(state_sentence)
	state_text_vec = indexesFromSentence(language_model, state_sentence, ignore_missing = True)
	if len(state_text_vec) < MAX_STATE_SYMBOLS :
		state_text_vec = np.pad(state_text_vec, pad_width= ((0,MAX_STATE_SYMBOLS - len(state_text_vec)),) )
	else :
		state_text_vec = np.array(state_text_vec)
		state_text_vec = state_text_vec[:MAX_STATE_SYMBOLS]
	return state_text_vec







# with open("data/agent_state_vval.pkl", 'rb') as f:
#     agent_model_mem = pickle.load(f)
with open("data/memory_2_mix_maxval.pkl","rb") as f:
	memory = pickle.load(f)

# X = []
# Y = []
# for x,y in agent_model_mem.items() :
# 	X.append(preprocess_state(x))
# 	Y.append(y)

batch = memory.sample_random_minibatch()
X, Y = zip(*batch)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)
# print(set(Y))
# print(sum(Y))
# print(len(X))
# print(X[0], Y[0])
# print(list(Y))
# quit()

def train_step( agent_model ,optimizer, loss_object, X_train,y_train, batch_size = 200, device="cpu") :
	random_index = np.random.choice(range(len(X_train)), size = batch_size )

	state = X_train[random_index]
	reward = y_train[random_index]
	state_tensor = torch.tensor(state).to(device)
	reward_tensor = torch.tensor(reward,dtype = torch.float).to(device)
	
	# print(state_tensor.shape)
	# print(reward_tensor.shape)

	optimizer.zero_grad()
	predicted_rewards = agent_model(state_tensor)

	loss = loss_object(predicted_rewards,reward_tensor)
	loss.backward()
	optimizer.step()
	return loss.item()


device = "cuda"

agent_model = Agent_model( language_model.n_chars, 500,1, device).to(device)
loss_object = nn.MSELoss()
optimizer = optim.SGD(agent_model.parameters(),lr = 0.001)

for _ in range(1000) :
	loss = train_step( agent_model ,optimizer, loss_object, X_train,y_train,batch_size = len(X_train), device = device)
	if _%100 == 0 :
		print(loss)


print("Train Classification Report")
print(X_train.shape)
y_train_pred = agent_model(torch.tensor(X_train,dtype=torch.int).to(device))
print(y_train_pred.shape)
y_train_pred = y_train_pred.cpu().detach().numpy().flatten() > 0.5
print(y_train_pred.shape)
print(classification_report(y_train,y_train_pred))

print("Test Classification Report")
print(X_test.shape)
y_test_pred = agent_model(torch.tensor(X_test,dtype=torch.int).to(device))
print(y_test_pred.shape)
y_test_pred = y_test_pred.cpu().detach().numpy().flatten() > 0.5
print(y_test_pred.shape)
print(classification_report(y_test,y_test_pred))
